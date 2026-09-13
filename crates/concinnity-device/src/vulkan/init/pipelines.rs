//! Graphics pipelines: the cascade shadow, spot shadow, text, composite and
//! bloom pipelines, and the bindless main-pass pipeline per material shader.

use ash::vk;
use concinnity_core::gfx::render_types::{self, SpotShadowData};
use concinnity_core::render::backend_init::WorldShader;
use concinnity_core::render::error::RenderResult;

use super::InitGpu;
use crate::vulkan::context::VkSpotShadow;
use crate::vulkan::owned::{
    OwnedFramebuffer, OwnedPipeline, OwnedPipelineLayout, OwnedRenderPass, OwnedSetLayout,
};
use crate::vulkan::pipeline::*;
use crate::vulkan::post::bloom::{compile_bloom_shaders, create_bloom_pipeline};
use crate::vulkan::swapchain::create_shadow_framebuffers;
use crate::vulkan::texture::GpuImage;

pub(super) struct MainPipelineInputs<'a> {
    pub(super) effective_shadow_size: u32,
    pub(super) shadow_render_pass: &'a OwnedRenderPass,
    pub(super) shadow_pipeline_layout: &'a OwnedPipelineLayout,
    pub(super) shadow_global_set_layout: &'a OwnedSetLayout,
    pub(super) shadow_map: &'a GpuImage,
    pub(super) spot_shadow_map: GpuImage,
    pub(super) spot_shadow_slice_size: u32,
    pub(super) spot_shadows: &'a [SpotShadowData],
    pub(super) gpu_text_atlases: &'a [GpuImage],
    pub(super) text_pipeline_layout: &'a OwnedPipelineLayout,
    pub(super) composite_render_pass: &'a OwnedRenderPass,
    pub(super) composite_pipeline_layout: &'a OwnedPipelineLayout,
    pub(super) bloom_write_pass: &'a OwnedRenderPass,
    pub(super) bloom_blend_pass: &'a OwnedRenderPass,
    pub(super) bloom_pipeline_layout: &'a OwnedPipelineLayout,
}

pub(super) struct MainPipelines {
    pub(super) shadow_pipeline_opt: Option<OwnedPipeline>,
    pub(super) shadow_framebuffers_vec: Vec<OwnedFramebuffer>,
    pub(super) spot_shadow: VkSpotShadow,
    pub(super) text_pipeline_opt: Option<OwnedPipeline>,
    pub(super) composite_pipeline: OwnedPipeline,
    pub(super) bloom_pipeline_prefilter: OwnedPipeline,
    pub(super) bloom_pipeline_downsample: OwnedPipeline,
    pub(super) bloom_pipeline_upsample: OwnedPipeline,
}

pub(super) fn build_main_pipelines(
    gpu: &InitGpu<'_>,
    inputs: MainPipelineInputs<'_>,
) -> RenderResult<MainPipelines> {
    let InitGpu {
        instance,
        device,
        physical_device,
        alloc,
        hot_reload,
        ..
    } = *gpu;
    let MainPipelineInputs {
        effective_shadow_size,
        shadow_render_pass,
        shadow_pipeline_layout,
        shadow_global_set_layout,
        shadow_map,
        spot_shadow_map,
        spot_shadow_slice_size,
        spot_shadows,
        gpu_text_atlases,
        text_pipeline_layout,
        composite_render_pass,
        composite_pipeline_layout,
        bloom_write_pass,
        bloom_blend_pass,
        bloom_pipeline_layout,
    } = inputs;
    let (shadow_pipeline_opt, shadow_framebuffers_vec) = if effective_shadow_size > 0
        && let Ok(Some(shadow_spv)) = resolve_shadow_shader(hot_reload)
    {
        let pl = create_shadow_pipeline(
            device,
            shadow_render_pass.handle(),
            shadow_pipeline_layout.handle(),
            &shadow_spv,
        )?;
        let fbs = create_shadow_framebuffers(
            device,
            shadow_render_pass.handle(),
            shadow_map,
            effective_shadow_size,
        )?;
        (Some(pl), fbs)
    } else {
        // No shadow pipeline: no transition needed. create_shadow_map_array
        // already rests the (1x1 fallback) shadow_map in SHADER_READ_ONLY,
        // the layout the main-pass descriptor expects, and with no shadow
        // loop nothing ever moves it out of that layout.
        (None, Vec::new())
    };

    // Spot shadows reuse the cascade pass's depth-only render pass, pipeline
    // and one-UBO set layout; only the framebuffers, the per-slice
    // projections, and the per-slice uniform slots are their own. Built even
    // with no shadowed spot (the 1x1 fallback array + a one-element buffer)
    // so the main pass's bindings 12/13 are always valid.
    let spot_shadow = crate::vulkan::draw::spot_shadow::build_spot_shadow(
        crate::vulkan::draw::spot_shadow::SpotShadowBuild {
            alloc,
            instance,
            device,
            physical_device,
            map: spot_shadow_map,
            render_pass: shadow_render_pass.handle(),
            set_layout: shadow_global_set_layout.handle(),
            slice_size: spot_shadow_slice_size,
            spot_shadows,
        },
    )?;

    // Text renders in the composite pass (post-tonemap, single-sample), so
    // its pipeline targets the composite render pass.
    let text_pipeline_opt = if !gpu_text_atlases.is_empty() {
        let (tv, tf) = compile_text_shaders(hot_reload)?;
        let tp = create_text_pipeline(
            device,
            composite_render_pass.handle(),
            text_pipeline_layout.handle(),
            &tv,
            &tf,
            vk::SampleCountFlags::TYPE_1,
        )?;
        Some(tp)
    } else {
        None
    };

    let composite_pipeline = {
        let (cv, cf) = compile_composite_shaders(hot_reload)?;
        create_composite_pipeline(
            device,
            composite_render_pass.handle(),
            composite_pipeline_layout.handle(),
            &cv,
            &cf,
        )?
    };

    // Bloom pipelines: prefilter, downsample and upsample.
    let (bloom_pipeline_prefilter, bloom_pipeline_downsample, bloom_pipeline_upsample) = {
        let bs = compile_bloom_shaders(hot_reload)?;
        let prefilter = create_bloom_pipeline(
            device,
            bloom_write_pass.handle(),
            bloom_pipeline_layout.handle(),
            &bs.vert,
            &bs.prefilter,
            false,
        )?;
        let downsample = create_bloom_pipeline(
            device,
            bloom_write_pass.handle(),
            bloom_pipeline_layout.handle(),
            &bs.vert,
            &bs.downsample,
            false,
        )?;
        // The upsample pipeline targets the LOAD blend pass and blends
        // additively onto the mip already there.
        let upsample = create_bloom_pipeline(
            device,
            bloom_blend_pass.handle(),
            bloom_pipeline_layout.handle(),
            &bs.vert,
            &bs.upsample,
            true,
        )?;
        (prefilter, downsample, upsample)
    };
    Ok(MainPipelines {
        shadow_pipeline_opt,
        shadow_framebuffers_vec,
        spot_shadow,
        text_pipeline_opt,
        composite_pipeline,
        bloom_pipeline_prefilter,
        bloom_pipeline_downsample,
        bloom_pipeline_upsample,
    })
}

pub(super) struct WorldPipelineInputs<'a> {
    pub(super) world_shaders: &'a [WorldShader<'a>],
    pub(super) bindless_pipeline_layout: Option<&'a OwnedPipelineLayout>,
    pub(super) bindless_main_spv: &'a (Vec<u8>, Vec<u8>),
    pub(super) main_render_pass: &'a OwnedRenderPass,
    pub(super) msaa_samples: vk::SampleCountFlags,
    pub(super) swapchain_format: vk::Format,
    pub(super) probe_cube_count: u32,
}

pub(super) struct WorldPipelines {
    pub(super) world_pipelines: Vec<Option<OwnedPipeline>>,
    pub(super) shader_bucket_count: usize,
}

// Material-referenced shaders (ShaderHandle 1..) each get their own
// bindless main-pass pipeline, so their draws route into their own region
// of the GPU-culled command buffer.
pub(super) fn build_world_pipelines(
    gpu: &InitGpu<'_>,
    inputs: WorldPipelineInputs<'_>,
) -> RenderResult<WorldPipelines> {
    let InitGpu {
        device, hot_reload, ..
    } = *gpu;
    let WorldPipelineInputs {
        world_shaders,
        bindless_pipeline_layout,
        bindless_main_spv,
        main_render_pass,
        msaa_samples,
        swapchain_format,
        probe_cube_count,
    } = inputs;
    let bucket_shaders = world_shaders.get(1..).unwrap_or(&[]);
    let world_pipelines = match (bindless_pipeline_layout, bucket_shaders.is_empty()) {
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
                device,
                BucketPipelineTargets {
                    render_pass: main_render_pass.handle(),
                    layout: layout.handle(),
                    msaa_samples,
                    swapchain_format,
                    hot_reload,
                    probe_count: probe_cube_count as usize,
                },
                bucket_shaders,
                bindless_main_spv,
            )?
        }
        _ => Vec::new(),
    };
    let shader_bucket_count = 1 + world_pipelines.len();
    Ok(WorldPipelines {
        world_pipelines,
        shader_bucket_count,
    })
}
