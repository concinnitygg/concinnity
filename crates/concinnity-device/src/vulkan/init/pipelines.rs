//! Graphics pipelines: the cascade shadow, spot shadow, text, composite and
//! bloom pipelines, and the bindless main-pass pipeline per material shader.

use ash::vk;
use concinnity_core::gfx::render_types::{self, SpotShadowData};
use concinnity_core::render::backend_init::WorldShader;
use concinnity_core::render::error::RenderResult;

use super::{InitGpu, PassStates};
use crate::vulkan::context::{VkCull, VkSpotShadow, VkTargets};
use crate::vulkan::pipeline::*;
use crate::vulkan::post::bloom::{compile_bloom_shaders, create_bloom_pipeline};
use crate::vulkan::swapchain::create_shadow_framebuffers;
use crate::vulkan::texture::GpuImage;

// Build the shadow, text, composite and bloom pipelines into the pass states,
// and the spot shadow state that reuses the cascade pass.
pub(super) fn build_main_pipelines(
    gpu: &InitGpu<'_>,
    passes: &mut PassStates,
    spot_shadow_map: GpuImage,
    spot_shadows: &[SpotShadowData],
) -> RenderResult<VkSpotShadow> {
    let InitGpu { hw, hot_reload, .. } = *gpu;
    let device = &hw.device;
    let PassStates {
        shadow,
        text,
        composite,
        bloom,
    } = passes;
    let shadow_set_layout = shadow
        .global_set_layout
        .as_ref()
        .expect("the layout stage builds the shadow global set layout")
        .handle();
    let (shadow_pipeline, shadow_framebuffers) = if shadow.map_size > 0
        && let Ok(Some(shadow_spv)) = resolve_shadow_shader(hot_reload)
    {
        let pl = create_shadow_pipeline(
            device,
            shadow.render_pass.handle(),
            shadow
                .pipeline_layout
                .as_ref()
                .expect("the layout stage builds the shadow pipeline layout")
                .handle(),
            &shadow_spv,
        )?;
        let fbs = create_shadow_framebuffers(
            device,
            shadow.render_pass.handle(),
            &shadow.map,
            shadow.map_size,
        )?;
        (Some(pl), fbs)
    } else {
        // No shadow pipeline: no transition needed. create_shadow_map_array
        // already rests the (1x1 fallback) shadow_map in SHADER_READ_ONLY,
        // the layout the main-pass descriptor expects, and with no shadow
        // loop nothing ever moves it out of that layout. The pipeline layout
        // is not kept either.
        shadow.pipeline_layout = None;
        (None, Vec::new())
    };
    shadow.pipeline = shadow_pipeline;
    shadow.framebuffers = shadow_framebuffers;

    // Spot shadows reuse the cascade pass's depth-only render pass, pipeline
    // and one-UBO set layout; only the framebuffers, the per-slice
    // projections, and the per-slice uniform slots are their own. Built even
    // with no shadowed spot (the 1x1 fallback array + a one-element buffer)
    // so the main pass's bindings 12/13 are always valid.
    let spot_shadow = crate::vulkan::draw::spot_shadow::build_spot_shadow(
        crate::vulkan::draw::spot_shadow::SpotShadowBuild {
            alloc: &hw.alloc,
            instance: &hw.instance,
            device,
            physical_device: hw.physical_device,
            map: spot_shadow_map,
            render_pass: shadow.render_pass.handle(),
            set_layout: shadow_set_layout,
            slice_size: render_types::spot_shadow_slice_size(shadow.map_size),
            spot_shadows,
        },
    )?;

    // Text renders in the composite pass (post-tonemap, single-sample), so
    // its pipeline targets the composite render pass.
    text.pipeline = if !text.atlas_textures.is_empty() {
        let (tv, tf) = compile_text_shaders(hot_reload)?;
        let tp = create_text_pipeline(
            device,
            composite.render_pass.handle(),
            text.pipeline_layout.handle(),
            &tv,
            &tf,
            vk::SampleCountFlags::TYPE_1,
        )?;
        Some(tp)
    } else {
        None
    };

    composite.pipeline = {
        let (cv, cf) = compile_composite_shaders(hot_reload)?;
        create_composite_pipeline(
            device,
            composite.render_pass.handle(),
            composite.pipeline_layout.handle(),
            &cv,
            &cf,
        )?
    };

    // Bloom pipelines: prefilter, downsample and upsample.
    let bs = compile_bloom_shaders(hot_reload)?;
    bloom.pipeline_prefilter = create_bloom_pipeline(
        device,
        bloom.write_pass.handle(),
        bloom.pipeline_layout.handle(),
        &bs.vert,
        &bs.prefilter,
        false,
    )?;
    bloom.pipeline_downsample = create_bloom_pipeline(
        device,
        bloom.write_pass.handle(),
        bloom.pipeline_layout.handle(),
        &bs.vert,
        &bs.downsample,
        false,
    )?;
    // The upsample pipeline targets the LOAD blend pass and blends
    // additively onto the mip already there.
    bloom.pipeline_upsample = create_bloom_pipeline(
        device,
        bloom.blend_pass.handle(),
        bloom.pipeline_layout.handle(),
        &bs.vert,
        &bs.upsample,
        true,
    )?;
    Ok(spot_shadow)
}

// Material-referenced shaders (ShaderHandle 1..) each get their own
// bindless main-pass pipeline, so their draws route into their own region
// of the GPU-culled command buffer.
pub(super) fn build_world_pipelines(
    gpu: &InitGpu<'_>,
    cull: &mut VkCull,
    world_shaders: &[WorldShader<'_>],
    targets: &VkTargets,
    swapchain_format: vk::Format,
    probe_cube_count: u32,
) -> RenderResult<()> {
    let InitGpu { hw, hot_reload, .. } = *gpu;
    let bucket_shaders = world_shaders.get(1..).unwrap_or(&[]);
    cull.world_pipelines = match (
        cull.bindless_pipeline_layout.as_ref(),
        bucket_shaders.is_empty(),
    ) {
        (Some(layout), false) => {
            let max = render_types::MAX_SHADER_BUCKETS;
            if bucket_shaders.len() + 1 > max {
                return Err(format!(
                    "world declares {} Shaders but at most {max} can be routed",
                    bucket_shaders.len() + 1
                )
                .into());
            }
            build_world_pipeline_table(
                &hw.device,
                BucketPipelineTargets {
                    render_pass: targets.main_render_pass.handle(),
                    layout: layout.handle(),
                    msaa_samples: targets.msaa_samples,
                    swapchain_format,
                    hot_reload,
                    probe_count: probe_cube_count as usize,
                },
                bucket_shaders,
                &cull.bindless_main_spv,
            )?
        }
        _ => Vec::new(),
    };
    Ok(())
}
