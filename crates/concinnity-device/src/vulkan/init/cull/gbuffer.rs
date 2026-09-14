//! The GPU-driven G-buffer pre-pass: the bindless MRT pipeline, its per-frame
//! sets, and the model-history ring with the snapshot kernel that fills it.

use ash::vk;
use concinnity_core::render::error::RenderResult;

use super::CullPlan;
use super::bindless::BindlessPass;
use super::compute::ComputeCull;
use crate::vulkan::context::VkDescriptors;
use crate::vulkan::init::InitGpu;
use crate::vulkan::owned::{OwnedPipeline, OwnedPipelineLayout, OwnedSetLayout};
use crate::vulkan::post::gbuffer::GbufferResources;

// The GPU-driven G-buffer pre-pass; `None`/empty unless the bindless cull path
// is active and the G-buffer is enabled.
pub(super) struct GbufferPass {
    pub(super) pipeline: Option<OwnedPipeline>,
    pub(super) pipeline_layout: Option<OwnedPipelineLayout>,
    pub(super) set_layout: Option<OwnedSetLayout>,
    pub(super) sets: Vec<vk::DescriptorSet>,
    pub(super) prev_model_buffers: Vec<crate::vulkan::allocator::PooledBuffer>,
    pub(super) model_history: Option<crate::vulkan::post::gbuffer::ModelHistoryPipeline>,
}

// Build the GPU-driven G-buffer pre-pass: the 3-MRT bindless pipeline and the
// per-frame model-history ring with the snapshot kernel that fills it.
pub(super) fn build_gbuffer_pass(
    gpu: &InitGpu<'_>,
    bindless: &BindlessPass,
    compute: &ComputeCull,
    gbuffer: Option<&GbufferResources>,
    descriptors: &VkDescriptors,
    plan: &CullPlan,
) -> RenderResult<GbufferPass> {
    let InitGpu {
        hw,
        frames,
        hot_reload,
        ..
    } = *gpu;
    let (device, alloc) = (&hw.device, &hw.alloc);
    let gbuffer_active = plan.bindless_active && gbuffer.is_some();
    // GPU-driven G-buffer pre-pass resources. Built when the bindless cull
    // path is active AND the G-buffer is enabled: a 3-MRT bindless pipeline +
    // the per-frame model-history ring and the snapshot kernel that fills it,
    // drawn by reusing the main pass's per-frame indirect buffer (camera
    // frustum, NO extra cull dispatch).
    type GbufferBindlessResources = (
        Option<OwnedPipeline>,
        Option<OwnedPipelineLayout>,
        Option<OwnedSetLayout>,
        Vec<vk::DescriptorSet>,
        Vec<crate::vulkan::allocator::PooledBuffer>,
        Option<crate::vulkan::post::gbuffer::ModelHistoryPipeline>,
    );
    let (
        gbuffer_bindless_pipeline,
        gbuffer_bindless_pipeline_layout,
        gbuffer_set_layout,
        gbuffer_sets,
        prev_model_buffers,
        model_history,
    ): GbufferBindlessResources = if let (true, Some(gb), Some(bl_set_layout)) =
        (gbuffer_active, gbuffer, bindless.set_layout.as_ref())
    {
        let gbb = crate::vulkan::post::gbuffer::build_gbuffer_bindless(
            crate::vulkan::post::gbuffer::GbufferDeviceCtx { alloc, device },
            crate::vulkan::post::gbuffer::GbufferBindlessDescriptors {
                descriptor_pool: descriptors.descriptor_pool.handle(),
                bindless_set_layout: bl_set_layout.handle(),
            },
            crate::vulkan::post::gbuffer::GbufferBindlessRecords {
                object_buffers: &bindless.object_buffers,
                draw_args_buffers: &compute.draw_args_buffers,
            },
            gb,
            crate::vulkan::post::gbuffer::GbufferBindlessScene {
                n_cull: plan.n_cull,
                frames,
            },
            hot_reload,
        )?;
        (
            Some(gbb.pipeline),
            Some(gbb.pipeline_layout),
            Some(gbb.set_layout),
            gbb.sets,
            gbb.prev_model_buffers,
            Some(gbb.history),
        )
    } else {
        (None, None, None, Vec::new(), Vec::new(), None)
    };
    Ok(GbufferPass {
        pipeline: gbuffer_bindless_pipeline,
        pipeline_layout: gbuffer_bindless_pipeline_layout,
        set_layout: gbuffer_set_layout,
        sets: gbuffer_sets,
        prev_model_buffers,
        model_history,
    })
}
