//! The GPU-driven G-buffer pre-pass's per-frame sets and the model-history ring
//! with the snapshot kernel that fills it. Its pipelines are the shader
//! buckets' own, built with the main pass's.

use ash::vk;
use concinnity_core::render::error::RenderResult;

use super::CullPlan;
use super::bindless::BindlessPass;
use super::compute::ComputeCull;
use crate::vulkan::context::VkDescriptors;
use crate::vulkan::init::InitGpu;
use crate::vulkan::post::gbuffer::GbufferResources;

// The GPU-driven G-buffer pre-pass; empty unless the bindless cull path is
// active and the G-buffer is enabled.
pub(super) struct GbufferPass {
    pub(super) sets: Vec<vk::DescriptorSet>,
    pub(super) prev_model_buffers: Vec<crate::vulkan::allocator::PooledBuffer>,
    pub(super) model_history: Option<crate::vulkan::post::gbuffer::ModelHistoryPipeline>,
}

// Build the GPU-driven G-buffer pre-pass's per-frame sets and model-history
// ring with the snapshot kernel that fills it. The pre-pass draws by reusing
// the main pass's per-frame indirect buffer (camera frustum, NO extra cull
// dispatch).
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
    let (true, Some(gb), Some(prepass)) = (
        plan.bindless_active,
        gbuffer,
        bindless.prepass_layout.as_ref(),
    ) else {
        return Ok(GbufferPass {
            sets: Vec::new(),
            prev_model_buffers: Vec::new(),
            model_history: None,
        });
    };
    let gbb = crate::vulkan::post::gbuffer::build_gbuffer_bindless(
        crate::vulkan::post::gbuffer::GbufferDeviceCtx { alloc, device },
        crate::vulkan::post::gbuffer::GbufferBindlessDescriptors {
            descriptor_pool: descriptors.descriptor_pool.handle(),
            prepass_set_layout: prepass.set_layout.handle(),
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
    Ok(GbufferPass {
        sets: gbb.sets,
        prev_model_buffers: gbb.prev_model_buffers,
        model_history: Some(gbb.history),
    })
}
