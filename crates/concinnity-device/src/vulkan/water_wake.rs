//! The water's wake on Vulkan (see
//! `concinnity_engine::render::environment::water::wake`): the state the
//! shared `water_wake.hlsl` kernels run forward each frame, the field a planet
//! sea's draw samples, and their dispatch inside the `WaterSim` node, after
//! the ocean's cascades, which already orders the node against the last
//! frame's draw at its head and the next draw at its tail.
//!
//! The field rests in `GENERAL` for good, like the cascade arrays: the
//! kernels write it as storage and the draw samples it in that layout, so
//! every step is a memory barrier that names no image. The block is a
//! host-visible uniform per frame in flight, since it outgrows a push.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::environment::water::wake::{
    WAKE_COLUMN_GROUPS, WAKE_MIP_LEVELS, WAKE_RESOLUTION, WAKE_ROW_GROUPS, wake_groups,
};
use concinnity_engine::render::shader::programs::ShaderProgram;
use concinnity_engine::render::uniforms::wake::WaterWakeParams;

use super::allocator::{PooledBuffer, PooledImage};
use super::builtin_shaders::{self, CompileProgram};
use super::descriptor_layout::{Binding, PoolSizes};
use super::owned::{
    OwnedDescriptorPool, OwnedPipeline, OwnedPipelineLayout, OwnedSetLayout, VkDevice,
};
use super::pipeline_desc::compute_pipeline;
use super::record::Recorder;
use super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::set_writes::SetWrites;
use super::texture::{
    GpuUploadContext, LayoutTransition, SubresourceRange, one_shot_submit,
    transition_image_layout_range,
};

const FIELD_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

// The kernels' set: the block, the state, the transform, the destination mip
// of the field and the source mip (read-only).
fn kernel_bindings() -> [Binding; 5] {
    use vk::DescriptorType as T;
    let c = vk::ShaderStageFlags::COMPUTE;
    [
        (0, T::UNIFORM_BUFFER, c),
        (1, T::STORAGE_BUFFER, c),
        (2, T::STORAGE_BUFFER, c),
        (3, T::STORAGE_IMAGE, c),
        (4, T::SAMPLED_IMAGE, c),
    ]
}

// A field and its views: the sampled one over every mip, and one per mip,
// which the kernels write as storage and read as sampled.
pub(in crate::vulkan) struct WakeField {
    _image: PooledImage,
    pub(in crate::vulkan) sampled: vk::ImageView,
    mips: Vec<vk::ImageView>,
}

impl WakeField {
    // A `size`-square field with `mips` levels, moved into `GENERAL`.
    pub(in crate::vulkan) fn new(
        gpu: &GpuUploadContext<'_>,
        size: u32,
        mips: u32,
    ) -> RenderResult<Self> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: size,
                height: size,
                depth: 1,
            })
            .mip_levels(mips)
            .array_layers(1)
            .format(FIELD_FORMAT)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);
        let image = gpu
            .alloc
            .create_image(&info, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .map_err(|e| e.context(format_args!("wake field")))?;
        let sampled = field_view(gpu.device, image.image(), 0, mips)?;
        image.attach_view(sampled);
        let mut views = Vec::with_capacity(mips as usize);
        for level in 0..mips {
            let view = field_view(gpu.device, image.image(), level, 1)?;
            image.attach_view(view);
            views.push(view);
        }
        one_shot_submit(gpu.device, gpu.command_pool, gpu.queue, |cmd| {
            transition_image_layout_range(
                gpu.device,
                cmd,
                image.image(),
                LayoutTransition {
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::GENERAL,
                    aspect: vk::ImageAspectFlags::COLOR,
                },
                SubresourceRange {
                    base_layer: 0,
                    layer_count: 1,
                    base_mip: 0,
                    mip_count: mips,
                },
            );
        })?;
        Ok(Self {
            _image: image,
            sampled,
            mips: views,
        })
    }
}

// The wake's kernels, state and field.
pub(in crate::vulkan) struct WakeSim {
    rows: OwnedPipeline,
    columns: OwnedPipeline,
    inverse: OwnedPipeline,
    shade: OwnedPipeline,
    mip: OwnedPipeline,
    layout: OwnedPipelineLayout,
    _set_layout: OwnedSetLayout,
    _pool: OwnedDescriptorPool,
    // Per frame in flight, one set per mip level; a frame's set 0 also
    // serves the transforms and the shading.
    sets: Vec<Vec<vk::DescriptorSet>>,
    // Per frame in flight, the block.
    params: Vec<PooledBuffer>,
    _state: PooledBuffer,
    _work: PooledBuffer,
    pub(in crate::vulkan) field: WakeField,
}

impl WakeSim {
    pub(in crate::vulkan) fn new(
        gpu: &GpuUploadContext<'_>,
        frames: usize,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let device = gpu.device;
        let cells = u64::from(WAKE_RESOLUTION * WAKE_RESOLUTION);
        let storage = |bytes: u64| {
            gpu.alloc.create_buffer(
                bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
        };
        let state_bytes = 2 * cells * size_of::<[f32; 4]>() as u64;
        let work_bytes = cells * size_of::<[f32; 2]>() as u64;
        let state = storage(state_bytes)?;
        let work = storage(work_bytes)?;
        let params_bytes = size_of::<WaterWakeParams>() as u64;
        let params = (0..frames)
            .map(|_| {
                gpu.alloc.create_buffer(
                    params_bytes,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
            })
            .collect::<RenderResult<Vec<_>>>()?;
        let field = WakeField::new(gpu, WAKE_RESOLUTION, WAKE_MIP_LEVELS)?;

        let set_layout = create_descriptor_set_layout(device, &kernel_bindings())?;
        let set_layouts = [set_layout.handle()];
        let layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
            )
            .map_err(|e| super::error::map_vk_result(e, "wake pipeline layout"))?;
        let kernel = |program: &ShaderProgram, label: &str| {
            compute_pipeline(
                device,
                layout.handle(),
                &program.compile(hot_reload)?,
                label,
            )
        };
        let levels = field.mips.len();
        let count = u32::saturating_from(frames * levels);
        let sizes = PoolSizes::default().sets(&kernel_bindings(), count).build();
        let pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(count)
                    .pool_sizes(&sizes),
            )
            .map_err(|e| super::error::map_vk_result(e, "wake descriptor pool"))?;
        let mut sets = Vec::with_capacity(frames);
        for block in &params {
            let frame_sets =
                alloc_descriptor_sets(device, pool.handle(), &vec![set_layout.handle(); levels])?;
            for (level, &set) in frame_sets.iter().enumerate() {
                SetWrites::new(set)
                    .uniform_buffer(0, block.buffer(), params_bytes)
                    .storage_buffer(1, state.buffer(), state_bytes)
                    .storage_buffer(2, work.buffer(), work_bytes)
                    .storage_image(3, field.mips[level])
                    .sampled_image_in(
                        4,
                        field.mips[level.saturating_sub(1)],
                        vk::ImageLayout::GENERAL,
                    )
                    .apply(device);
            }
            sets.push(frame_sets);
        }
        Ok(Self {
            rows: kernel(&builtin_shaders::WATER_WAKE_ROWS, "wake rows")?,
            columns: kernel(&builtin_shaders::WATER_WAKE_COLUMNS, "wake columns")?,
            inverse: kernel(&builtin_shaders::WATER_WAKE_INVERSE, "wake inverse")?,
            shade: kernel(&builtin_shaders::WATER_WAKE_SHADE, "wake shading")?,
            mip: kernel(&builtin_shaders::WATER_WAKE_MIP, "wake mips")?,
            layout,
            _set_layout: set_layout,
            _pool: pool,
            sets,
            params,
            _state: state,
            _work: work,
            field,
        })
    }

    // Run the wake one frame forward under `params`, frame slot `frame_idx`,
    // and reduce its field. `step` orders each kernel after the last. Returns
    // whether it encoded the run.
    pub(in crate::vulkan) fn encode(
        &self,
        rec: &Recorder<'_>,
        frame_idx: usize,
        params: &WaterWakeParams,
        step: impl Fn(&Recorder<'_>),
    ) -> bool {
        let (Some(sets), Some(block)) = (self.sets.get(frame_idx), self.params.get(frame_idx))
        else {
            return false;
        };
        let Some(first) = sets.first() else {
            return false;
        };
        block.write_val(0, params);
        let bind = |set: &vk::DescriptorSet| {
            rec.bind_descriptor_sets(
                vk::PipelineBindPoint::COMPUTE,
                &self.layout,
                0,
                core::slice::from_ref(set),
                &[],
            );
        };
        bind(first);
        for (kernel, groups) in [
            (&self.rows, WAKE_ROW_GROUPS),
            (&self.columns, WAKE_COLUMN_GROUPS),
            (&self.inverse, WAKE_ROW_GROUPS),
        ] {
            rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, kernel);
            rec.dispatch(groups, 1, 1);
            step(rec);
        }
        rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &self.shade);
        rec.dispatch(wake_groups(0), wake_groups(0), 1);
        rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &self.mip);
        for (level, set) in sets.iter().enumerate().skip(1) {
            step(rec);
            bind(set);
            let g = wake_groups(u32::saturating_from(level));
            rec.dispatch(g, g, 1);
        }
        step(rec);
        true
    }
}

fn field_view(
    device: &VkDevice,
    image: vk::Image,
    base_mip: u32,
    mips: u32,
) -> RenderResult<vk::ImageView> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(FIELD_FORMAT)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: base_mip,
            level_count: mips,
            base_array_layer: 0,
            layer_count: 1,
        });
    // SAFETY: the create-info is live for the call and names an image of this device.
    unsafe { device.create_image_view(&info, None) }
        .map_err(|e| super::error::map_vk_result(e, "wake field view"))
}
