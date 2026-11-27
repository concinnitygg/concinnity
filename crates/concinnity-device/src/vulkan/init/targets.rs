//! Render targets: the off-screen HDR attachments and the transient image
//! pool.

use ash::vk;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::render_graph::{PoolGates, plan_pool_slots};

use super::{Features, InitGpu};
use crate::vulkan::context::{SwapchainState, Targets};
use crate::vulkan::swapchain::{AttachmentDeviceCtx, create_attachments};
use crate::vulkan::texture::GpuUploadContext;

#[derive(Clone, Copy)]
pub(super) struct TargetInputs<'a> {
    pub(super) swapchain: &'a SwapchainState,
    pub(super) features: &'a Features,
    pub(super) render_extent: vk::Extent2D,
}

// Build the render-resolution scene targets.
pub(super) fn build_render_targets(
    gpu: &InitGpu<'_>,
    inputs: TargetInputs<'_>,
) -> RenderResult<Targets> {
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
    } = inputs;
    let (device, msaa_samples) = (&hw.device, features.msaa_samples);

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

    // The transient image pool (`ao_output`, `bloom_top`, the G-buffer channels),
    // planned for every feature a setting can enable, and built before its
    // consumers bind it.
    let transient_pool = crate::vulkan::transient_pool::TransientImagePool::build(
        &crate::vulkan::transient_pool::TransientPoolGpu {
            instance: &hw.instance,
            device,
            physical_device: hw.physical_device,
            command_pool,
            queue: hw.graphics_queue,
        },
        frames,
        &plan_pool_slots(
            PoolGates::REACHABLE,
            (render_extent.width, render_extent.height),
            (swapchain.extent.width, swapchain.extent.height),
        )?,
    )?;
    let reactive_mask_images = crate::vulkan::reactive_mask::create_reactive_masks(
        GpuUploadContext {
            alloc: &hw.alloc,
            device,
            command_pool,
            queue: hw.graphics_queue,
        },
        render_extent,
        frames,
    )?;
    Ok(Targets {
        render_extent,
        msaa_samples,
        color_images,
        depth_images,
        hdr_resolve_images,
        reactive_mask_images,
        transient_pool,
    })
}
