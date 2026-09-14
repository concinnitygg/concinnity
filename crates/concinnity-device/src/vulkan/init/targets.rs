//! Render targets: the main render pass, the off-screen HDR attachments and
//! their framebuffers, and the transient image pool.

use ash::vk;
use concinnity_core::render::error::RenderResult;

use super::{Features, InitGpu};
use crate::vulkan::context::{HDR_FORMAT, SwapchainState, VkTargets};
use crate::vulkan::render_pass::*;
use crate::vulkan::swapchain::*;

pub(super) struct TargetInputs<'a> {
    pub(super) swapchain: &'a SwapchainState,
    pub(super) features: &'a Features,
    pub(super) render_extent: vk::Extent2D,
    pub(super) ssao_enabled: bool,
}

// Build the render-resolution scene targets.
pub(super) fn build_render_targets(
    gpu: &InitGpu<'_>,
    inputs: TargetInputs<'_>,
) -> RenderResult<VkTargets> {
    let InitGpu {
        hw,
        command_pool,
        frames,
        ..
    } = *gpu;
    let TargetInputs {
        swapchain,
        features,
        render_extent,
        ssao_enabled,
    } = inputs;
    let (device, msaa_samples) = (&hw.device, features.msaa_samples);
    let main_render_pass = create_main_render_pass(device, HDR_FORMAT, msaa_samples)?;

    // Off-screen HDR attachments, one set per frame-in-flight slot.
    let (color_images, depth_images, hdr_resolve_images) = create_attachments(
        &AttachmentDeviceCtx {
            alloc: &hw.alloc,
            device,
            command_pool,
            queue: hw.graphics_queue,
        },
        render_extent.width,
        render_extent.height,
        msaa_samples,
        frames,
    )?;

    let framebuffers = create_main_framebuffers(
        device,
        main_render_pass.handle(),
        &color_images,
        &depth_images,
        &hdr_resolve_images,
        render_extent,
        msaa_samples,
    )?;

    // Transient image pool: the graph-owned transients (`ao_output`,
    // `bloom_top`, and the three G-buffer color channels). Built before the
    // bloom chain so bloom mip 0 binds the pooled `bloom_top` image, before
    // SSAO so its blur framebuffers + the main pass binding 6 bind the pooled
    // `ao_output`, and before the G-buffer pre-pass so its framebuffers bind
    // the pooled MRT channels.
    let transient_pool = crate::vulkan::transient_pool::TransientImagePool::build(
        &crate::vulkan::transient_pool::TransientPoolGpu {
            instance: &hw.instance,
            device,
            physical_device: hw.physical_device,
            command_pool,
            queue: hw.graphics_queue,
        },
        frames,
        &crate::vulkan::transient_pool::transient_slots(
            ssao_enabled,
            features.bloom_on,
            features.gbuffer_enabled,
            render_extent,
            swapchain.extent,
        )?,
    )?;
    Ok(VkTargets {
        render_extent,
        main_render_pass,
        msaa_samples,
        color_images,
        depth_images,
        hdr_resolve_images,
        framebuffers,
        transient_pool,
    })
}
