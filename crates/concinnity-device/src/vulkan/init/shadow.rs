//! Shadows: the cascade shadow map with its render pass, pipeline and uniform
//! ring, and the spot shadow array that reuses them.

use ash::vk;
use concinnity_core::gfx::render_types::{
    self, LightUniforms, NUM_SHADOW_CASCADES, ShadowUniforms, SpotShadowData,
};
use concinnity_core::render::backend_init::ShadowParams;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::lights;

use super::InitGpu;
use crate::vulkan::context::{VkShadow, VkSpotShadow};
use crate::vulkan::draw::upload_shadow_uniforms;
use crate::vulkan::pipeline::{create_shadow_pipeline, resolve_shadow_shader};
use crate::vulkan::render_pass::create_shadow_render_pass;
use crate::vulkan::resources::create_descriptor_set_layout;
use crate::vulkan::swapchain::create_shadow_framebuffers;
use crate::vulkan::texture::{create_sampler_shadow, create_shadow_map_array};

// Build the cascade shadow state. CSM is gated on `shadow_map_size` (from
// GraphicsConfig; 0 disables shadows), and the shadow vertex shader is
// engine-internal. Mirrors the Metal internal-shadow path.
pub(super) fn build_shadow(
    gpu: &InitGpu<'_>,
    shadows: &ShadowParams,
    light_uniforms: &LightUniforms,
) -> RenderResult<VkShadow> {
    let InitGpu {
        hw,
        frames,
        hot_reload,
        ..
    } = *gpu;
    let device = &hw.device;
    // Shadow map: a 4-layer D32_SFLOAT array image, one slice per cascade.
    let map = create_shadow_map_array(&gpu.upload(), shadows.map_size, NUM_SHADOW_CASCADES as u32)?;
    let render_pass = create_shadow_render_pass(device)?;
    let sampler = create_sampler_shadow(device)?;
    // Shadow global set (set 0 for shadow pass): ShadowUniforms UBO.
    let global_set_layout = create_descriptor_set_layout(
        device,
        &crate::vulkan::descriptor_layout::shadow_global_set(),
    )?;
    let (pipeline, pipeline_layout, framebuffers) = if shadows.map_size > 0
        && let Ok(Some(shadow_spv)) = resolve_shadow_shader(hot_reload)
    {
        let shadow_pc_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            // 64 bytes for model + 16 bytes for cascade_idx + padding.
            .size(80);
        let shadow_set_layouts = [global_set_layout.handle()];
        let layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&shadow_set_layouts)
                    .push_constant_ranges(std::slice::from_ref(&shadow_pc_range)),
            )
            .map_err(|e| crate::vulkan::error::map_vk_result(e, "shadow pipeline layout"))?;
        let pl =
            create_shadow_pipeline(device, render_pass.handle(), layout.handle(), &shadow_spv)?;
        let fbs = create_shadow_framebuffers(device, render_pass.handle(), &map, shadows.map_size)?;
        (Some(pl), Some(layout), fbs)
    } else {
        // No shadow pipeline: no transition needed. create_shadow_map_array
        // already rests the (1x1 fallback) shadow_map in SHADER_READ_ONLY,
        // the layout the main-pass descriptor expects, and with no shadow
        // loop nothing ever moves it out of that layout.
        (None, None, Vec::new())
    };

    // Per-frame-in-flight `ShadowUniforms` UBO ring, persistently mapped.
    // One slot per frame so writing this frame's cascade VPs cannot land in
    // memory an in-flight frame is still sampling: under `Hybrid` a far
    // cascade's VP is frozen for several frames and then jumps a whole
    // texel-snap quantum, so an aliased read samples that cascade with the
    // jumped VP against depth rasterized with the old one.
    let shadow_ubo_size = std::mem::size_of::<ShadowUniforms>() as u64;
    let mut ubos = Vec::with_capacity(frames);
    for _ in 0..frames {
        ubos.push(hw.alloc.create_buffer(
            shadow_ubo_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?);
    }
    // Per-frame cascade computation lives in `gfx::csm::compute_shadow_uniforms`
    // and runs from `draw.rs` each frame; the shadow state starts from
    // `empty_shadow_uniforms()` so the descriptor write at startup has a valid
    // (fully-lit) buffer.
    let uniforms = concinnity_core::render::csm::empty_shadow_uniforms();
    for ubo in &ubos {
        upload_shadow_uniforms(ubo, &uniforms);
    }
    Ok(VkShadow {
        render_pass,
        map,
        map_size: shadows.map_size,
        framebuffers,
        pipeline,
        pipeline_layout,
        global_set_layout,
        sampler,
        skinned_pipeline: None,
        skinned_pipeline_layout: None,
        ubos,
        uniforms,
        // Per-frame CSM updates use the first directional light's direction;
        // cached at init so subsequent frames don't have to look it up.
        // Matches the Metal/DirectX pattern.
        light_dir: lights::sun_direction(light_uniforms),
        update: shadows.update,
        distance: shadows.distance,
        cascades: shadows.cascades,
        scheduler: Default::default(),
        render_mask: 0,
    })
}

// Spot shadows reuse the cascade pass's depth-only render pass, pipeline
// and one-UBO set layout; only the framebuffers, the per-slice
// projections, and the per-slice uniform slots are their own. Built even
// with no shadowed spot (the 1x1 fallback array + a one-element buffer)
// so the main pass's bindings 12/13 are always valid.
pub(super) fn build_spot_shadow(
    gpu: &InitGpu<'_>,
    shadow: &VkShadow,
    spot_shadows: &[SpotShadowData],
) -> RenderResult<VkSpotShadow> {
    let hw = gpu.hw;
    // Spot shadow map array: one layer per shadow-casting spot, at a quarter
    // the cascade resolution (a spot slice covers a single cone, not a
    // view-frustum slab). Passing size 0 yields the 1x1 fallback, which is
    // what a world with no shadowed spot binds.
    let slice_size = render_types::spot_shadow_slice_size(shadow.map_size);
    let map = create_shadow_map_array(
        &gpu.upload(),
        if spot_shadows.is_empty() {
            0
        } else {
            slice_size
        },
        spot_shadows.len().max(1) as u32,
    )?;
    crate::vulkan::draw::spot_shadow::build_spot_shadow(
        crate::vulkan::draw::spot_shadow::SpotShadowBuild {
            alloc: &hw.alloc,
            instance: &hw.instance,
            device: &hw.device,
            physical_device: hw.physical_device,
            map,
            render_pass: shadow.render_pass.handle(),
            set_layout: shadow.global_set_layout.handle(),
            slice_size,
            spot_shadows,
        },
    )
}
