//! The GPU-driven shadow passes: the shadow cull kernel with its per-(frame,
//! cascade) and per-(frame, spot slice) views, each packing its kept draws into
//! its own lists, and the depth-only bindless pipeline both draw with.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::render_types;
use concinnity_engine::render::backend::error::RenderResult;

use super::CullPlan;
use super::bindless::BindlessPass;
use super::compute::{ComputeCull, VARIANT_BUFFER_BYTES, shadow_cull_set};
use crate::vulkan::allocator::PooledBuffer;
use crate::vulkan::compact::{CompactBuffers, DrawLists, INDIRECT_USAGE};
use crate::vulkan::context::{VkDescriptors, VkShadow};
use crate::vulkan::cull::VkShadowView;
use crate::vulkan::init::InitGpu;
use crate::vulkan::owned::{OwnedPipeline, OwnedPipelineLayout, OwnedSetLayout};
use crate::vulkan::pipeline::{
    CULL_PUSH_CONSTANT_BYTES, compile_shadow_bindless_vs, compile_shadow_cull_shader,
    create_shadow_pipeline,
};
use crate::vulkan::pipeline_desc::compute_pipeline;
use crate::vulkan::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use crate::vulkan::set_writes::SetWrites;

// The GPU-driven shadow pass; `None`/empty unless the bindless cull path is
// active and shadows are enabled.
pub(super) struct ShadowCull {
    pub(super) cull_pipeline: Option<OwnedPipeline>,
    pub(super) cull_pipeline_layout: Option<OwnedPipelineLayout>,
    pub(super) set_layout: Option<OwnedSetLayout>,
    pub(super) views: Vec<Vec<VkShadowView>>,
    pub(super) bindless_pipeline: Option<OwnedPipeline>,
    pub(super) bindless_pipeline_layout: Option<OwnedPipelineLayout>,
    pub(super) spot_views: Vec<Vec<VkShadowView>>,
}

// Build the GPU-driven shadow pass: a frustum + distance cull per cascade and
// the depth-only bindless pipeline that draws its survivors.
pub(super) fn build_shadow_cull(
    gpu: &InitGpu<'_>,
    bindless: &BindlessPass,
    compute: &ComputeCull,
    shadow: &VkShadow,
    descriptors: &VkDescriptors,
    plan: &CullPlan,
    spot_slices: usize,
) -> RenderResult<ShadowCull> {
    let InitGpu { hw, hot_reload, .. } = *gpu;
    let device = &hw.device;
    let mut spot_shadow_views = Vec::new();
    let (object_buffers, draw_args_buffers) =
        (&bindless.object_buffers, &compute.draw_args_buffers);
    // GPU-driven shadow pass resources. Built when the bindless cull path and
    // its packing are active AND shadows are enabled: a frustum-only cull
    // pipeline (`SHADOW_CULL`, lean 4-SSBO set: objects + draw-args + group
    // counts + status), a view per (frame, cascade) whose packing lays its kept
    // draws into its own lists, and a depth-only bindless graphics pipeline
    // (shadow-global set 0 + the bindless GpuObjectData set 1 + a cascade-index
    // push constant). Each re-rendered cascade then runs one cull dispatch, its
    // packing, and one indirect draw by count for the static + instance stream
    // and one for the skinned tail, replacing the CPU per-object shadow loop.
    type ShadowCullResources = (
        Option<OwnedPipeline>,
        Option<OwnedPipelineLayout>,
        Option<OwnedSetLayout>,
        Vec<Vec<VkShadowView>>,
        Option<OwnedPipeline>,
        Option<OwnedPipelineLayout>,
    );
    let (
        shadow_cull_pipeline,
        shadow_cull_pipeline_layout,
        shadow_cull_set_layout,
        shadow_views,
        shadow_bindless_pipeline,
        shadow_bindless_pipeline_layout,
    ): ShadowCullResources = if plan.bindless_active
        && shadow.enabled()
        && let Some(bl_set_layout) = bindless.set_layout.as_ref()
        && let Some(compact) = compute.compact.as_ref()
    {
        let cascades = render_types::NUM_SHADOW_CASCADES;
        // Lean shadow cull set layout: objects(0) + draw-args(1) + group
        // counts(2) + status(3).
        let sc_set_layout = create_descriptor_set_layout(device, &shadow_cull_set())?;

        let sc_push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(CULL_PUSH_CONSTANT_BYTES);
        let sc_layouts = [sc_set_layout.handle()];
        let sc_pl = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&sc_layouts)
                    .push_constant_ranges(std::slice::from_ref(&sc_push)),
            )
            .map_err(|e| crate::vulkan::error::map_vk_result(e, "shadow cull pipeline layout"))?;
        let sc_spv = compile_shadow_cull_shader(hot_reload)?;
        let sc_pipeline = compute_pipeline(device, sc_pl.handle(), &sc_spv, "shadow cull")?;

        // Depth-only bindless shadow graphics pipeline: shadow-global set 0 +
        // the bindless GpuObjectData set 1 + the per-view `ShadowPassPush`.
        let sb_push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(u32::saturating_from(
                size_of::<render_types::ShadowPassPush>(),
            ));
        let shadow_global_set_layout = &shadow.global_set_layout;
        let sb_layouts = [shadow_global_set_layout.handle(), bl_set_layout.handle()];
        let sb_pl = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&sb_layouts)
                    .push_constant_ranges(std::slice::from_ref(&sb_push)),
            )
            .map_err(|e| {
                crate::vulkan::error::map_vk_result(e, "shadow bindless pipeline layout")
            })?;
        let sb_spv = compile_shadow_bindless_vs(hot_reload)?;
        let sb_pipeline =
            create_shadow_pipeline(device, shadow.render_pass.handle(), sb_pl.handle(), &sb_spv)?;

        // Per-(frame, view) views, for the cascades and then the spot slices.
        // The cull dispatch for a view binds its cull set, its packing its
        // packing set, and the view's draws read its lists.
        let views = ViewCullInputs {
            gpu,
            set_layout: sc_set_layout.handle(),
            compact_set_layout: compact.set_layout.handle(),
            pool: descriptors.descriptor_pool.handle(),
            object_buffers,
            draw_args_buffers,
            variant_buffers: &compute.variant_buffers,
            n_cull: plan.n_cull,
        };
        let cascade_views = views.build(cascades)?;
        spot_shadow_views = views.build(spot_slices)?;

        (
            Some(sc_pipeline),
            Some(sc_pl),
            Some(sc_set_layout),
            cascade_views,
            Some(sb_pipeline),
            Some(sb_pl),
        )
    } else {
        (None, None, None, Vec::new(), None, None)
    };
    Ok(ShadowCull {
        cull_pipeline: shadow_cull_pipeline,
        cull_pipeline_layout: shadow_cull_pipeline_layout,
        set_layout: shadow_cull_set_layout,
        views: shadow_views,
        bindless_pipeline: shadow_bindless_pipeline,
        bindless_pipeline_layout: shadow_bindless_pipeline_layout,
        spot_views: spot_shadow_views,
    })
}

// What every (frame, view) binds: this frame's object, draw-args and variant
// SSBOs, beside the view's own group counts, status and lists.
struct ViewCullInputs<'a> {
    gpu: &'a InitGpu<'a>,
    set_layout: vk::DescriptorSetLayout,
    compact_set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    object_buffers: &'a [PooledBuffer],
    draw_args_buffers: &'a [PooledBuffer],
    variant_buffers: &'a [PooledBuffer],
    n_cull: usize,
}

impl ViewCullInputs<'_> {
    // `views` views per frame, indexed [frame][view].
    fn build(&self, views: usize) -> RenderResult<Vec<Vec<VkShadowView>>> {
        let InitGpu {
            hw,
            command_pool,
            frames,
            ..
        } = *self.gpu;
        let (device, alloc) = (&hw.device, &hw.alloc);
        if views == 0 {
            return Ok(Vec::new());
        }
        let n = self.n_cull;
        let object_buffer_size = (n * size_of::<render_types::GpuObjectData>()) as u64;
        let draw_args_size = (n * size_of::<render_types::GpuDrawArgs>()) as u64;
        let status_size = (n.max(1) * size_of::<u32>()) as u64;
        let group_count_size = crate::vulkan::compact::group_count_bytes(n);
        let device_local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let mut all = Vec::with_capacity(frames);
        for f in 0..frames {
            let layouts: Vec<_> = (0..views)
                .flat_map(|_| [self.set_layout, self.compact_set_layout])
                .collect();
            let sets = alloc_descriptor_sets(device, self.pool, &layouts)?;
            let mut frame_views = Vec::with_capacity(views);
            for pair in sets.chunks_exact(2) {
                let lists = alloc.create_buffer(
                    crate::vulkan::compact::indirect_buffer_bytes(n),
                    INDIRECT_USAGE,
                    device_local,
                )?;
                let status = alloc.create_buffer(status_size, storage, device_local)?;
                let group_counts = alloc.create_buffer(group_count_size, storage, device_local)?;
                SetWrites::new(pair[0])
                    .storage_buffer(0, self.object_buffers[f].buffer(), object_buffer_size)
                    .storage_buffer(1, self.draw_args_buffers[f].buffer(), draw_args_size)
                    .storage_buffer(2, group_counts.buffer(), group_count_size)
                    .storage_buffer(3, status.buffer(), status_size)
                    .apply(device);
                crate::vulkan::compact::write_compact_set(
                    device,
                    pair[1],
                    &CompactBuffers {
                        variants: (self.variant_buffers[f].buffer(), VARIANT_BUFFER_BYTES),
                        draw_args: (self.draw_args_buffers[f].buffer(), draw_args_size),
                        status: (status.buffer(), status_size),
                        group_counts: (group_counts.buffer(), group_count_size),
                        lists: DrawLists {
                            buffer: lists.buffer(),
                            commands: n,
                        },
                    },
                );
                frame_views.push(VkShadowView {
                    cull_set: pair[0],
                    compact_set: pair[1],
                    lists,
                    _status: status,
                    _group_counts: group_counts,
                    commands: n,
                });
            }
            all.push(frame_views);
        }
        crate::vulkan::compact::zero_indirect_buffers(
            device,
            command_pool,
            hw.graphics_queue,
            all.iter().flatten().map(|v| &v.lists),
        )?;
        Ok(all)
    }
}
