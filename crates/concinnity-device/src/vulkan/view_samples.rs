//! View samples on Vulkan: the dispatch that samples the Hi-Z pyramid's first
//! level over a coarse grid of cells, copied into a per-frame `HOST_VISIBLE`
//! readback buffer. Frame slot `i` writes buffer `i`, and `begin` reads it back
//! once the fence has retired the frame that wrote it, so a reading is never
//! waited on. Mirrors `metal/view_samples.rs` and `directx/view_samples.rs`.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::render_types::{VIEW_SAMPLE_CELLS, ViewSampleCell, ViewSampleParams};
use concinnity_engine::render::backend::error::RenderResult;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::context::VkContext;
use super::descriptor_layout::{Binding, PoolSizes};
use super::pipeline_desc::compute_pipeline;
use super::readback_ring::ReadbackRing;
use super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::set_writes::SetWrites;
use crate::vulkan::builtin_shaders::CompileProgram;
use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedPipeline, OwnedPipelineLayout, OwnedSetLayout, VkDevice,
};
use crate::vulkan::record::Recorder;

const CELLS_BYTES: vk::DeviceSize =
    (VIEW_SAMPLE_CELLS * size_of::<ViewSampleCell>()) as vk::DeviceSize;

// The sampling's pipeline, its set, the device-local cells and the readback
// ring.
pub(in crate::vulkan) struct ViewSamplesGpu {
    pipeline: OwnedPipeline,
    layout: OwnedPipelineLayout,
    _set_layout: OwnedSetLayout,
    // The pyramid's sampled view and the cells.
    set: vk::DescriptorSet,
    _descriptor_pool: OwnedDescriptorPool,
    // The pass's output, device-local, copied into this frame's readback.
    cells: PooledBuffer,
    // One grid of cells per frame in flight; a slot no frame wrote reads back
    // as serial 0.
    readbacks: ReadbackRing<ViewSampleCell>,
}

impl ViewSamplesGpu {
    // The sampling over the pyramid whose all-mips sampled view is
    // `pyramid_view`, with a readback slot per frame in flight.
    pub(in crate::vulkan) fn new(
        alloc: &DeviceAllocator,
        device: &VkDevice,
        pyramid_view: vk::ImageView,
        frames: usize,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let set_layout = create_descriptor_set_layout(device, &bindings())?;
        let push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(u32::saturating_from(size_of::<ViewSampleParams>()));
        let layouts = [set_layout.handle()];
        let layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(std::slice::from_ref(&push)),
            )
            .map_err(|e| super::error::map_vk_result(e, "view samples layout"))?;
        let pipeline = create_pipeline(device, layout.handle(), hot_reload)?;

        let pool_sizes = PoolSizes::default().sets(&bindings(), 1).build();
        let descriptor_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
            )
            .map_err(|e| super::error::map_vk_result(e, "view samples descriptor pool"))?;
        let sets = alloc_descriptor_sets(device, descriptor_pool.handle(), &layouts)?;

        let cells = alloc.create_buffer(
            CELLS_BYTES,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let readbacks = ReadbackRing::new(alloc, frames, VIEW_SAMPLE_CELLS)?;

        let gpu = Self {
            pipeline,
            layout,
            _set_layout: set_layout,
            set: sets[0],
            _descriptor_pool: descriptor_pool,
            cells,
            readbacks,
        };
        gpu.write_set(device, pyramid_view);
        Ok(gpu)
    }

    // Re-point the sampling at a rebuilt pyramid. Called with the device idle.
    pub(in crate::vulkan) fn resize(&self, device: &VkDevice, pyramid_view: vk::ImageView) {
        self.write_set(device, pyramid_view);
    }

    // Recompile the kernel against the existing layout, for hot reload.
    pub(in crate::vulkan) fn rebuild_pipeline(
        &self,
        device: &VkDevice,
        hot_reload: bool,
    ) -> RenderResult<OwnedPipeline> {
        create_pipeline(device, self.layout.handle(), hot_reload)
    }

    // Swap a rebuilt pipeline in. The caller has idled the device.
    pub(in crate::vulkan) fn swap_pipeline(&mut self, pipeline: OwnedPipeline) {
        self.pipeline = pipeline;
    }

    // The cells frame slot `frame_idx`'s readback holds, once the fence has
    // retired its writer.
    pub(in crate::vulkan) fn read(&self, frame_idx: usize) -> Option<&[ViewSampleCell]> {
        self.readbacks.read(frame_idx)
    }

    // Free the buffers. Called from `Drop for VkContext` after the device idles.
    pub(in crate::vulkan) fn destroy(&mut self) {
        self.cells = PooledBuffer::null();
        self.readbacks.destroy();
    }

    fn write_set(&self, device: &VkDevice, pyramid_view: vk::ImageView) {
        SetWrites::new(self.set)
            .sampled_image(0, pyramid_view)
            .storage_buffer(1, self.cells.buffer(), vk::WHOLE_SIZE)
            .apply(device);
    }

    // Record the dispatch under `params` and copy the cells into frame slot
    // `frame_idx`'s readback, made visible to the host.
    fn record(&self, rec: &Recorder<'_>, params: &ViewSampleParams, frame_idx: usize) {
        let compute = vk::ShaderStageFlags::COMPUTE;
        rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &self.pipeline);
        rec.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            &self.layout,
            0,
            std::slice::from_ref(&self.set),
            &[],
        );
        rec.push_constants(&self.layout, compute, 0, params);
        rec.dispatch(params.columns, params.rows, 1);

        let cells = params.columns as usize * params.rows as usize;
        self.readbacks
            .copy_from(rec, self.cells.buffer(), cells, frame_idx);
    }
}

// The set: the pyramid, read by texel, and the cells.
fn bindings() -> [Binding; 2] {
    let compute = vk::ShaderStageFlags::COMPUTE;
    [
        (0, vk::DescriptorType::SAMPLED_IMAGE, compute),
        (1, vk::DescriptorType::STORAGE_BUFFER, compute),
    ]
}

fn create_pipeline(
    device: &VkDevice,
    layout: vk::PipelineLayout,
    hot_reload: bool,
) -> RenderResult<OwnedPipeline> {
    let program = super::builtin_shaders::VIEW_SAMPLES.compile(hot_reload)?;
    compute_pipeline(device, layout, &program, "view samples")
}

impl VkContext {
    // Sample the Hi-Z pyramid's first level by this frame's view parameters
    // into frame slot `frame_idx`'s readback. A no-op without the sampling or
    // samples planned this frame.
    pub(in crate::vulkan) fn encode_view_samples(&self, rec: &Recorder<'_>, frame_idx: usize) {
        let (Some(gpu), Some(params)) = (
            self.view_samples.as_ref(),
            self.state.planner.view.samples.as_ref(),
        ) else {
            return;
        };
        let extent = self.targets.render_extent;
        // A frame planned for another size than the pyramid holds samples
        // nothing.
        if [extent.width, extent.height] != [params.width, params.height] {
            return;
        }
        gpu.record(rec, params, frame_idx);
    }
}
