//! Render targets: the cascade and spot shadow map arrays, the render passes,
//! the off-screen HDR attachments and their framebuffers, the transient image
//! pool, and the bloom mip chain.

use ash::vk;
use concinnity_core::gfx::render_types::{self, NUM_SHADOW_CASCADES, SpotShadowData};
use concinnity_core::render::error::RenderResult;

use super::InitGpu;
use crate::vulkan::context::HDR_FORMAT;
use crate::vulkan::owned::{OwnedFramebuffer, OwnedRenderPass};
use crate::vulkan::post::bloom::{
    BloomDeviceContext, create_bloom_chain, create_bloom_framebuffers,
};
use crate::vulkan::post::gbuffer::GbufferPooled;
use crate::vulkan::render_pass::*;
use crate::vulkan::swapchain::*;
use crate::vulkan::texture::*;
use crate::vulkan::transient_pool::TransientImagePool;

// Shadow map: a 4-layer D32_SFLOAT array image, one slice per cascade.
// CSM is gated on `shadow_map_size` (from GraphicsConfig; 0 disables
// shadows). The shadow vertex shader is engine-internal. Mirrors the
// Metal internal-shadow path.
pub(super) fn build_shadow_map(
    gpu: &InitGpu<'_>,
    effective_shadow_size: u32,
) -> RenderResult<GpuImage> {
    let InitGpu {
        device,
        alloc,
        command_pool,
        queue: graphics_queue,
        ..
    } = *gpu;
    let shadow_map = create_shadow_map_array(
        &GpuUploadContext {
            alloc,
            device,
            command_pool,
            queue: graphics_queue,
        },
        effective_shadow_size,
        NUM_SHADOW_CASCADES as u32,
    )?;
    Ok(shadow_map)
}

pub(super) struct SpotShadowMap {
    pub(super) slice_size: u32,
    pub(super) map: GpuImage,
}

// Spot shadow map array: one layer per shadow-casting spot, at a quarter
// the cascade resolution (a spot slice covers a single cone, not a
// view-frustum slab). Passing size 0 yields the 1x1 fallback, which is
// what a world with no shadowed spot binds.
pub(super) fn build_spot_shadow_map(
    gpu: &InitGpu<'_>,
    effective_shadow_size: u32,
    spot_shadows: &[SpotShadowData],
) -> RenderResult<SpotShadowMap> {
    let InitGpu {
        device,
        alloc,
        command_pool,
        queue: graphics_queue,
        ..
    } = *gpu;
    let spot_shadow_slice_size = render_types::spot_shadow_slice_size(effective_shadow_size);
    let spot_shadow_map = create_shadow_map_array(
        &GpuUploadContext {
            alloc,
            device,
            command_pool,
            queue: graphics_queue,
        },
        if spot_shadows.is_empty() {
            0
        } else {
            spot_shadow_slice_size
        },
        spot_shadows.len().max(1) as u32,
    )?;
    Ok(SpotShadowMap {
        slice_size: spot_shadow_slice_size,
        map: spot_shadow_map,
    })
}

pub(super) struct RenderTargetConfig<'a> {
    pub(super) msaa_samples: vk::SampleCountFlags,
    pub(super) swapchain_format: vk::Format,
    pub(super) swapchain_extent: vk::Extent2D,
    pub(super) swapchain_image_views: &'a [vk::ImageView],
    pub(super) render_extent: vk::Extent2D,
    pub(super) ssao_enabled: bool,
    pub(super) bloom_on: bool,
    pub(super) gbuffer_enabled: bool,
}

pub(super) struct RenderTargets {
    pub(super) main_render_pass: OwnedRenderPass,
    pub(super) shadow_render_pass: OwnedRenderPass,
    pub(super) composite_render_pass: OwnedRenderPass,
    pub(super) bloom_write_pass: OwnedRenderPass,
    pub(super) bloom_blend_pass: OwnedRenderPass,
    pub(super) color_images: Vec<GpuImage>,
    pub(super) depth_images: Vec<GpuImage>,
    pub(super) hdr_resolve_images: Vec<GpuImage>,
    pub(super) framebuffers: Vec<OwnedFramebuffer>,
    pub(super) composite_framebuffers: Vec<OwnedFramebuffer>,
    pub(super) transient_pool: TransientImagePool,
    pub(super) gbuffer_pooled: GbufferPooled,
    pub(super) bloom_mips: Vec<Vec<GpuImage>>,
    pub(super) bloom_mip_extents: Vec<vk::Extent2D>,
    pub(super) bloom_write_framebuffers: Vec<Vec<OwnedFramebuffer>>,
    pub(super) bloom_blend_framebuffers: Vec<Vec<OwnedFramebuffer>>,
}

pub(super) fn build_render_targets(
    gpu: &InitGpu<'_>,
    cfg: RenderTargetConfig<'_>,
) -> RenderResult<RenderTargets> {
    let InitGpu {
        instance,
        device,
        physical_device,
        alloc,
        command_pool,
        queue: graphics_queue,
        frames,
        ..
    } = *gpu;
    let RenderTargetConfig {
        msaa_samples,
        swapchain_format,
        swapchain_extent,
        swapchain_image_views,
        render_extent,
        ssao_enabled,
        bloom_on,
        gbuffer_enabled,
    } = cfg;
    let main_render_pass = create_main_render_pass(device, HDR_FORMAT, msaa_samples)?;
    let shadow_render_pass = create_shadow_render_pass(device)?;
    let composite_render_pass = create_composite_render_pass(device, swapchain_format)?;
    let bloom_write_pass = create_bloom_render_pass(device, HDR_FORMAT, false)?;
    let bloom_blend_pass = create_bloom_render_pass(device, HDR_FORMAT, true)?;

    // Off-screen HDR attachments, one set per frame-in-flight slot.
    let (color_images, depth_images, hdr_resolve_images) = create_attachments(
        &AttachmentDeviceCtx {
            alloc,
            device,
            command_pool,
            queue: graphics_queue,
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
    let composite_framebuffers = create_composite_framebuffers(
        device,
        composite_render_pass.handle(),
        swapchain_image_views,
        swapchain_extent,
    )?;

    // Transient image pool: the graph-owned transients (`ao_output`,
    // `bloom_top`, and the three G-buffer color channels). Built before the
    // bloom chain so bloom mip 0 binds the pooled `bloom_top` image, before
    // SSAO (below) so its blur framebuffers + the main pass binding 6 bind
    // the pooled `ao_output`, and before the G-buffer pre-pass so its
    // framebuffers bind the pooled MRT channels.
    let transient_pool = crate::vulkan::transient_pool::TransientImagePool::build(
        &crate::vulkan::transient_pool::TransientPoolGpu {
            instance,
            device,
            physical_device,
            command_pool,
            queue: graphics_queue,
        },
        frames,
        &crate::vulkan::transient_pool::transient_slots(
            ssao_enabled,
            bloom_on,
            gbuffer_enabled,
            render_extent,
            swapchain_extent,
        )?,
    )?;
    let bloom_top_pairs = transient_pool.pairs_for_frames("bloom_top", frames);
    let gbuffer_pooled = transient_pool.gbuffer_pooled(frames);

    let (bloom_mips, bloom_mip_extents) = create_bloom_chain(
        &BloomDeviceContext {
            alloc,
            device,
            command_pool,
            queue: graphics_queue,
        },
        swapchain_extent,
        frames,
        &bloom_top_pairs,
    )?;
    let (bloom_write_framebuffers, bloom_blend_framebuffers) = create_bloom_framebuffers(
        device,
        bloom_write_pass.handle(),
        bloom_blend_pass.handle(),
        &bloom_mips,
        &bloom_mip_extents,
    )?;
    Ok(RenderTargets {
        main_render_pass,
        shadow_render_pass,
        composite_render_pass,
        bloom_write_pass,
        bloom_blend_pass,
        color_images,
        depth_images,
        hdr_resolve_images,
        framebuffers,
        composite_framebuffers,
        transient_pool,
        gbuffer_pooled,
        bloom_mips,
        bloom_mip_extents,
        bloom_write_framebuffers,
        bloom_blend_framebuffers,
    })
}
