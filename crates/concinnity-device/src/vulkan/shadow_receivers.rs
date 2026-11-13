//! Shadow receiver bounds on Vulkan: the two dispatches that reduce the Hi-Z
//! pyramid's first level into the light-space bounds of what the camera saw in
//! each cascade's band, copied into a per-frame `HOST_VISIBLE` readback buffer.
//! Frame slot `i` writes buffer `i`, and `begin` reads it back once the fence
//! has retired the frame that wrote it, so a reading is never waited on.
//! Mirrors `metal/shadow_receivers.rs` and `directx/shadow_receivers.rs`.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::render_types::{
    RECEIVER_MIN_LANES, ReceiverBands, ReceiverBounds, ReceiverBoundsParams, receiver_tiles,
};
use concinnity_engine::render::backend::error::RenderResult;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::context::{CullState, ShadowState, VkContext, VkHardware};
use super::descriptor_layout::{Binding, PoolSizes};
use super::pipeline_desc::compute_pipeline;
use super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::set_writes::SetWrites;
use crate::vulkan::builtin_shaders::CompileProgram;
use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedPipeline, OwnedPipelineLayout, OwnedSetLayout, VkDevice,
};
use crate::vulkan::record::Recorder;

const BOUNDS_BYTES: vk::DeviceSize = size_of::<ReceiverBounds>() as vk::DeviceSize;

// The reduction's pipelines, its sets, the per-tile scratch and the readback
// ring.
pub(in crate::vulkan) struct ShadowReceiversGpu {
    reduce_pipeline: OwnedPipeline,
    reduce_layout: OwnedPipelineLayout,
    _reduce_set_layout: OwnedSetLayout,
    // The pyramid's sampled view and the partials.
    reduce_set: vk::DescriptorSet,
    merge_pipeline: OwnedPipeline,
    merge_layout: OwnedPipelineLayout,
    _merge_set_layout: OwnedSetLayout,
    // The partials and the device-local bounds.
    merge_set: vk::DescriptorSet,
    _descriptor_pool: OwnedDescriptorPool,
    // One `ReceiverBands` per tile of the depth, device-local.
    partials: PooledBuffer,
    // The tiles `partials` holds room for.
    tiles: u32,
    // The merge's output, device-local, copied into this frame's readback.
    bounds: PooledBuffer,
    // One `ReceiverBounds` per frame in flight, persistently mapped and zeroed
    // at creation, so a slot no frame wrote reads back as serial 0.
    readbacks: Vec<PooledBuffer>,
}

// The device, its allocator and the pyramid a reduction reads.
pub(in crate::vulkan) struct ReceiverTarget<'a> {
    pub alloc: &'a DeviceAllocator,
    pub device: &'a VkDevice,
    // The Hi-Z pyramid's all-mips sampled view; the kernel loads level 0.
    pub pyramid_view: vk::ImageView,
    // The depth's size in texels.
    pub size: [u32; 2],
}

impl ShadowReceiversGpu {
    // The reduction beside `cull`'s Hi-Z pyramid of the `extent` depth and
    // `shadow`'s cascades, on a device whose waves can run it, with a readback
    // slot per frame in flight; `None` without all three.
    pub(in crate::vulkan) fn beside(
        hw: &VkHardware,
        cull: &CullState,
        shadow: &ShadowState,
        extent: vk::Extent2D,
        frames: usize,
        hot_reload: bool,
    ) -> RenderResult<Option<Self>> {
        match cull.hiz.as_ref() {
            Some(hiz)
                if shadow.map_sizes().cascades > 0
                    && waves_fold(&hw.instance, hw.physical_device) =>
            {
                let target = ReceiverTarget {
                    alloc: &hw.alloc,
                    device: &hw.device,
                    pyramid_view: hiz.read_set_view(),
                    size: [extent.width, extent.height],
                };
                Self::new(&target, frames, hot_reload).map(Some)
            }
            _ => Ok(None),
        }
    }

    // The reduction over `target`'s pyramid, with a readback slot per frame in
    // flight.
    pub(in crate::vulkan) fn new(
        target: &ReceiverTarget<'_>,
        frames: usize,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let device = target.device;
        let reduce_set_layout = create_descriptor_set_layout(device, &reduce_bindings())?;
        let merge_set_layout = create_descriptor_set_layout(device, &merge_bindings())?;
        let reduce_layout = pipeline_layout(device, &reduce_set_layout, "receiver reduce")?;
        let merge_layout = pipeline_layout(device, &merge_set_layout, "receiver merge")?;
        let (reduce_pipeline, merge_pipeline) = create_pipelines(
            device,
            reduce_layout.handle(),
            merge_layout.handle(),
            hot_reload,
        )?;

        let pool_sizes = PoolSizes::default()
            .sets(&reduce_bindings(), 1)
            .sets(&merge_bindings(), 1)
            .build();
        let descriptor_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(2)
                    .pool_sizes(&pool_sizes),
            )
            .map_err(|e| super::error::map_vk_result(e, "shadow receiver descriptor pool"))?;
        let sets = alloc_descriptor_sets(
            device,
            descriptor_pool.handle(),
            &[reduce_set_layout.handle(), merge_set_layout.handle()],
        )?;

        let tiles = tile_count(target.size);
        let partials = create_partials(target.alloc, tiles)?;
        let bounds = target.alloc.create_buffer(
            BOUNDS_BYTES,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let readbacks = (0..frames.max(1))
            .map(|_| {
                let buffer = target.alloc.create_buffer(
                    BOUNDS_BYTES,
                    vk::BufferUsageFlags::TRANSFER_DST,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )?;
                // SAFETY: the mapping is a live HOST_COHERENT range of at least `BOUNDS_BYTES`
                // bytes, and no GPU work references the buffer yet.
                unsafe {
                    std::ptr::write_bytes(buffer.mapped_ptr(), 0, size_of::<ReceiverBounds>());
                }
                Ok(buffer)
            })
            .collect::<RenderResult<Vec<_>>>()?;

        let gpu = Self {
            reduce_pipeline,
            reduce_layout,
            _reduce_set_layout: reduce_set_layout,
            reduce_set: sets[0],
            merge_pipeline,
            merge_layout,
            _merge_set_layout: merge_set_layout,
            merge_set: sets[1],
            _descriptor_pool: descriptor_pool,
            partials,
            tiles,
            bounds,
            readbacks,
        };
        gpu.write_sets(device, target.pyramid_view);
        Ok(gpu)
    }

    // Re-point the reduction at a resized pyramid, and regrow the partials if
    // they hold too few tiles. Called with the device idle.
    pub(in crate::vulkan) fn resize(&mut self, target: &ReceiverTarget<'_>) -> RenderResult<()> {
        let tiles = tile_count(target.size);
        if tiles > self.tiles {
            self.partials = create_partials(target.alloc, tiles)?;
            self.tiles = tiles;
        }
        self.write_sets(target.device, target.pyramid_view);
        Ok(())
    }

    // Recompile both kernels against the existing layouts, for hot reload.
    pub(in crate::vulkan) fn rebuild_pipelines(
        &self,
        device: &VkDevice,
        hot_reload: bool,
    ) -> RenderResult<(OwnedPipeline, OwnedPipeline)> {
        create_pipelines(
            device,
            self.reduce_layout.handle(),
            self.merge_layout.handle(),
            hot_reload,
        )
    }

    // Swap rebuilt pipelines in. The caller has idled the device.
    pub(in crate::vulkan) fn swap_pipelines(
        &mut self,
        (reduce, merge): (OwnedPipeline, OwnedPipeline),
    ) {
        self.reduce_pipeline = reduce;
        self.merge_pipeline = merge;
    }

    // What frame slot `frame_idx`'s readback holds, once the fence has retired
    // its writer.
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "mapped memory is aligned to at least `minMemoryMapAlignment`"
    )]
    pub(in crate::vulkan) fn read(&self, frame_idx: usize) -> Option<ReceiverBounds> {
        let readback = self.readbacks.get(frame_idx)?;
        // SAFETY: the mapping is this slot's HOST_COHERENT buffer of one `ReceiverBounds`; the
        // fence wait gated the copy that wrote it, its host barrier made the copy visible, and
        // every bit pattern is a valid `ReceiverBounds`.
        Some(unsafe { (readback.mapped_ptr() as *const ReceiverBounds).read() })
    }

    // Free the buffers. Called from `Drop for VkContext` after the device idles.
    pub(in crate::vulkan) fn destroy(&mut self) {
        self.partials = PooledBuffer::null();
        self.bounds = PooledBuffer::null();
        self.readbacks.clear();
    }

    fn write_sets(&self, device: &VkDevice, pyramid_view: vk::ImageView) {
        SetWrites::new(self.reduce_set)
            .sampled_image(0, pyramid_view)
            .storage_buffer(1, self.partials.buffer(), vk::WHOLE_SIZE)
            .apply(device);
        SetWrites::new(self.merge_set)
            .storage_buffer(0, self.partials.buffer(), vk::WHOLE_SIZE)
            .storage_buffer(1, self.bounds.buffer(), vk::WHOLE_SIZE)
            .apply(device);
    }

    // Record both dispatches under `params` and copy the bounds into frame
    // slot `frame_idx`'s readback, made visible to the host.
    fn record(&self, rec: &Recorder<'_>, params: &ReceiverBoundsParams, frame_idx: usize) {
        let Some(readback) = self.readbacks.get(frame_idx) else {
            return;
        };
        let [groups_x, groups_y] = receiver_tiles(params.width, params.height);
        if groups_x * groups_y > self.tiles {
            return;
        }
        let compute = vk::ShaderStageFlags::COMPUTE;
        rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &self.reduce_pipeline);
        rec.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            &self.reduce_layout,
            0,
            std::slice::from_ref(&self.reduce_set),
            &[],
        );
        rec.push_constants(&self.reduce_layout, compute, 0, params);
        rec.dispatch(groups_x, groups_y, 1);

        memory_barrier(
            rec,
            (
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_WRITE,
            ),
            (
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_READ,
            ),
        );
        rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &self.merge_pipeline);
        rec.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            &self.merge_layout,
            0,
            std::slice::from_ref(&self.merge_set),
            &[],
        );
        rec.push_constants(&self.merge_layout, compute, 0, params);
        rec.dispatch(1, 1, 1);

        memory_barrier(
            rec,
            (
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_WRITE,
            ),
            (
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_READ,
            ),
        );
        let copy = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: BOUNDS_BYTES,
        };
        rec.copy_buffer(
            self.bounds.buffer(),
            readback.buffer(),
            std::slice::from_ref(&copy),
        );
        // The fence orders the host's read after this frame; this makes the
        // copy visible to it. The compute stage takes no access: it orders
        // this frame's merge read of the partials and copy of the bounds ahead
        // of the next frame's dispatches writing them.
        memory_barrier(
            rec,
            (
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_WRITE,
            ),
            (
                vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::HOST_READ,
            ),
        );
    }
}

fn memory_barrier(
    rec: &Recorder<'_>,
    (src_stage, src_access): (vk::PipelineStageFlags, vk::AccessFlags),
    (dst_stage, dst_access): (vk::PipelineStageFlags, vk::AccessFlags),
) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);
    rec.pipeline_barrier(
        src_stage,
        dst_stage,
        std::slice::from_ref(&barrier),
        &[],
        &[],
    );
}

fn tile_count([width, height]: [u32; 2]) -> u32 {
    let [x, y] = receiver_tiles(width.max(1), height.max(1));
    x * y
}

fn create_partials(alloc: &DeviceAllocator, tiles: u32) -> RenderResult<PooledBuffer> {
    alloc.create_buffer(
        vk::DeviceSize::from(tiles.max(1)) * size_of::<ReceiverBands>() as vk::DeviceSize,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
}

// Reduce set: the pyramid, read by texel, and the partials.
fn reduce_bindings() -> [Binding; 2] {
    let compute = vk::ShaderStageFlags::COMPUTE;
    [
        (0, vk::DescriptorType::SAMPLED_IMAGE, compute),
        (1, vk::DescriptorType::STORAGE_BUFFER, compute),
    ]
}

// Merge set: the partials and the bounds.
fn merge_bindings() -> [Binding; 2] {
    let compute = vk::ShaderStageFlags::COMPUTE;
    [
        (0, vk::DescriptorType::STORAGE_BUFFER, compute),
        (1, vk::DescriptorType::STORAGE_BUFFER, compute),
    ]
}

fn pipeline_layout(
    device: &VkDevice,
    set_layout: &OwnedSetLayout,
    label: &str,
) -> RenderResult<OwnedPipelineLayout> {
    let push = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(u32::saturating_from(size_of::<ReceiverBoundsParams>()));
    let layouts = [set_layout.handle()];
    device
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(std::slice::from_ref(&push)),
        )
        .map_err(|e| super::error::map_vk_result(e, label))
}

fn create_pipelines(
    device: &VkDevice,
    reduce_layout: vk::PipelineLayout,
    merge_layout: vk::PipelineLayout,
    hot_reload: bool,
) -> RenderResult<(OwnedPipeline, OwnedPipeline)> {
    let reduce = super::builtin_shaders::SHADOW_RECEIVERS_REDUCE.compile(hot_reload)?;
    let merge = super::builtin_shaders::SHADOW_RECEIVERS_MERGE.compile(hot_reload)?;
    Ok((
        compute_pipeline(device, reduce_layout, &reduce, "shadow receiver reduce")?,
        compute_pipeline(device, merge_layout, &merge, "shadow receiver merge")?,
    ))
}

/// Whether the device's waves can run the reduction: arithmetic wave
/// operations in compute shaders, on waves at least `RECEIVER_MIN_LANES` wide.
fn waves_fold(instance: &ash::Instance, pd: vk::PhysicalDevice) -> bool {
    let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
    // SAFETY: a property query on a live handle; it only reads.
    unsafe { instance.get_physical_device_properties2(pd, &mut props2) };
    subgroups_fold(&subgroup)
}

// Whether `subgroup` offers what the reduction's wave fold needs.
fn subgroups_fold(subgroup: &vk::PhysicalDeviceSubgroupProperties<'_>) -> bool {
    let ops = vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC;
    subgroup
        .supported_stages
        .contains(vk::ShaderStageFlags::COMPUTE)
        && subgroup.supported_operations.contains(ops)
        && subgroup.subgroup_size >= RECEIVER_MIN_LANES
}

impl VkContext {
    // Reduce the Hi-Z pyramid's first level by this frame's receiver
    // parameters into frame slot `frame_idx`'s readback. A no-op without the
    // reduction or a measurement planned this frame.
    pub(in crate::vulkan) fn encode_shadow_receivers(&self, rec: &Recorder<'_>, frame_idx: usize) {
        let (Some(gpu), Some(params)) = (
            self.shadow_receivers.as_ref(),
            self.state.planner.shadow.receivers.as_ref(),
        ) else {
            return;
        };
        let extent = self.targets.render_extent;
        // A frame planned for another size than the pyramid holds measures
        // nothing.
        if [extent.width, extent.height] != [params.width, params.height] {
            return;
        }
        gpu.record(rec, params, frame_idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn props(
        stages: vk::ShaderStageFlags,
        ops: vk::SubgroupFeatureFlags,
        size: u32,
    ) -> vk::PhysicalDeviceSubgroupProperties<'static> {
        vk::PhysicalDeviceSubgroupProperties::default()
            .supported_stages(stages)
            .supported_operations(ops)
            .subgroup_size(size)
    }

    // The reduction runs only where compute waves fold arithmetically and
    // are wide enough for its group memory.
    #[test]
    fn the_reduction_needs_wide_arithmetic_compute_waves() {
        let compute = vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::FRAGMENT;
        let ops = vk::SubgroupFeatureFlags::BASIC
            | vk::SubgroupFeatureFlags::ARITHMETIC
            | vk::SubgroupFeatureFlags::BALLOT;
        assert!(subgroups_fold(&props(compute, ops, 32)));
        assert!(subgroups_fold(&props(compute, ops, RECEIVER_MIN_LANES)));
        assert!(!subgroups_fold(&props(
            compute,
            ops,
            RECEIVER_MIN_LANES / 2
        )));
        assert!(!subgroups_fold(&props(
            vk::ShaderStageFlags::FRAGMENT,
            ops,
            32
        )));
        assert!(!subgroups_fold(&props(
            compute,
            vk::SubgroupFeatureFlags::BASIC,
            32
        )));
    }
}
