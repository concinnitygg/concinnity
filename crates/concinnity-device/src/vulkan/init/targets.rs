//! Render targets: the cascade and spot shadow map arrays, the render passes,
//! the off-screen HDR attachments and their framebuffers, the transient image
//! pool, and the bloom mip chain.

use ash::vk;
use concinnity_core::gfx::render_types::{self, NUM_SHADOW_CASCADES, SpotShadowData};
use concinnity_core::render::error::RenderResult;

use super::{Features, InitGpu, PassStates};
use crate::vulkan::context::{HDR_FORMAT, SwapchainState, VkShadow, VkTargets};
use crate::vulkan::post::bloom::{
    BloomDeviceContext, create_bloom_chain, create_bloom_framebuffers,
};
use crate::vulkan::render_pass::*;
use crate::vulkan::swapchain::*;
use crate::vulkan::texture::*;

// Shadow map: a 4-layer D32_SFLOAT array image, one slice per cascade.
// CSM is gated on `shadow_map_size` (from GraphicsConfig; 0 disables
// shadows). The shadow vertex shader is engine-internal. Mirrors the
// Metal internal-shadow path.
pub(super) fn build_shadow_map(gpu: &InitGpu<'_>, shadow: &mut VkShadow) -> RenderResult<()> {
    shadow.map =
        create_shadow_map_array(&gpu.upload(), shadow.map_size, NUM_SHADOW_CASCADES as u32)?;
    Ok(())
}

// Spot shadow map array: one layer per shadow-casting spot, at a quarter
// the cascade resolution (a spot slice covers a single cone, not a
// view-frustum slab). Passing size 0 yields the 1x1 fallback, which is
// what a world with no shadowed spot binds.
pub(super) fn build_spot_shadow_map(
    gpu: &InitGpu<'_>,
    shadow_map_size: u32,
    spot_shadows: &[SpotShadowData],
) -> RenderResult<GpuImage> {
    let spot_shadow_slice_size = render_types::spot_shadow_slice_size(shadow_map_size);
    create_shadow_map_array(
        &gpu.upload(),
        if spot_shadows.is_empty() {
            0
        } else {
            spot_shadow_slice_size
        },
        spot_shadows.len().max(1) as u32,
    )
}

pub(super) struct TargetInputs<'a> {
    pub(super) swapchain: &'a SwapchainState,
    pub(super) features: &'a Features,
    pub(super) render_extent: vk::Extent2D,
    pub(super) ssao_enabled: bool,
}

// Build the scene targets, and the render passes, composite framebuffers and
// bloom chain of the pass states.
pub(super) fn build_render_targets(
    gpu: &InitGpu<'_>,
    inputs: TargetInputs<'_>,
    passes: &mut PassStates,
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
    passes.shadow.render_pass = create_shadow_render_pass(device)?;
    passes.composite.render_pass = create_composite_render_pass(device, swapchain.format)?;
    passes.bloom.write_pass = create_bloom_render_pass(device, HDR_FORMAT, false)?;
    passes.bloom.blend_pass = create_bloom_render_pass(device, HDR_FORMAT, true)?;

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
    passes.composite.framebuffers = create_composite_framebuffers(
        device,
        passes.composite.render_pass.handle(),
        &swapchain.image_views,
        swapchain.extent,
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
    let bloom_top_pairs = transient_pool.pairs_for_frames("bloom_top", frames);

    let bloom = &mut passes.bloom;
    (bloom.mips, bloom.mip_extents) = create_bloom_chain(
        &BloomDeviceContext {
            alloc: &hw.alloc,
            device,
            command_pool,
            queue: hw.graphics_queue,
        },
        swapchain.extent,
        frames,
        &bloom_top_pairs,
    )?;
    (bloom.write_framebuffers, bloom.blend_framebuffers) = create_bloom_framebuffers(
        device,
        bloom.write_pass.handle(),
        bloom.blend_pass.handle(),
        &bloom.mips,
        &bloom.mip_extents,
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
