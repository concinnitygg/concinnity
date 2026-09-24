//! The GPU-driven cull state: record sizing, the bindless static pass, the
//! compute cull with its Hi-Z pyramid, and the GPU-driven shadow, G-buffer and
//! two-pass occlusion passes built over them.

use ash::vk;
use concinnity_core::bake::texture::TextureImage;
use concinnity_core::gfx::render_types::clone_reserve;
use concinnity_core::render::backend_init::{SceneData, WorldShader};
use concinnity_core::render::error::RenderResult;
use concinnity_core::transform::IDENTITY;

use super::InitGpu;
use crate::vulkan::context::{VkCull, VkDescriptors, VkSceneAssets, VkShadow, VkTargets};
use crate::vulkan::post::gbuffer::GbufferResources;
use crate::vulkan::probe_prefilter::ProbePrefilterPipelines;

mod bindless;
mod compute;
mod gbuffer;
mod shadow;
mod two_pass;

// How many records the GPU-driven cull buffers hold and how the bindless texture
// pool is laid out, decided from the world before any of them is built.
pub(super) struct CullPlan {
    pub(super) n_instances: usize,
    pub(super) n_cull: usize,
    pub(super) bindless_active: bool,
    pub(super) bindless_pool_size: usize,
    pub(super) bindless_uab: bool,
}

pub(super) fn plan_cull(
    gpu: &InitGpu<'_>,
    world: &SceneData<'_>,
    textures: &[TextureImage],
) -> CullPlan {
    let hw = gpu.hw;
    let draw_objects = &world.draw_objects;
    // Instanced props fold into the GPU-driven bindless cull buffers: each
    // instance becomes a `GpuObjectData` record appended after the `n_objects`
    // static records (written once at init), so the object / draw-args /
    // indirect / cull-status buffers size for the combined `n_cull` count and
    // the cull kernel tests every instance independently. Skinned objects fold
    // in after the instances (a per-frame-rebuilt tail of `n_skinned` records),
    // so `n_cull` reserves their slots too. Mirrors `directx/init`.
    let n_instances: usize = world
        .instanced_clusters
        .iter()
        .map(|c| c.instances.len())
        .sum();
    // Runtime record reserve (`[n_objects + n_instances, +n_runtime)`): the
    // worst-case resident streamed-chunk window plus the runtime-clone cap,
    // between the instances and the skinned tail; resident chunks fold in per
    // frame. 0 for a non-voxel world.
    let n_cull = draw_objects.len()
        + n_instances
        + world.n_chunk_max
        + clone_reserve(draw_objects.len())
        + world.n_skinned;

    // GPU-driven static pass: active when there is anything to drive --
    // build-time static geometry, instances, streamed chunks, or skinned
    // meshes (`n_cull > 0`). A pure-voxel world has no build-time geometry but
    // folds its chunks here. A world default Shader drives it through bucket
    // 0, built from the world's own bindless pair. Its texture pool is the
    // deduplicated [albedo..] ++ [normal-map..] image set; the helper derives
    // the same value from the texture table so the export-time precompile
    // matches.
    let bindless_active = n_cull > 0;
    // The pool is sized to the world's own texture table, which keeps every
    // index in range by construction. The shaders declare the array unsized,
    // so this length reaches the descriptor layout and nothing else: it is not
    // part of any source text and cannot make a program miss its precompiled
    // artifact.
    let bindless_pool_size = if bindless_active {
        crate::vulkan::descriptor_layout::world_pool_size(textures.len())
    } else {
        0
    };
    // The texture pool's length is the world's texture table, so it cannot be
    // clamped to the device's per-stage headroom. Where it does not fit, its set
    // layout is declared update-after-bind, which moves it off
    // `maxPerStageDescriptorSampledImages` (256 on MoltenVK) and onto the
    // update-after-bind limit (a million there). This reshapes the layout, its
    // binding flags, and the descriptor pool it is allocated from, so it is
    // resolved once here. Desktop drivers report six figures and always stay on
    // the plain path.
    let limits = super::descriptors::stage_limits(hw);
    let pool_overflows = bindless_active
        && crate::vulkan::descriptor_layout::bindless_pool_needs_update_after_bind(
            limits,
            bindless_pool_size as u32,
        );
    if pool_overflows && !hw.update_after_bind {
        tracing::warn!(
            "bindless texture pool: {bindless_pool_size} images exceed the device's \
             per-stage budget ({limits:?}) and update-after-bind is unavailable"
        );
    }
    let bindless_uab = pool_overflows && hw.update_after_bind;
    CullPlan {
        n_instances,
        n_cull,
        bindless_active,
        bindless_pool_size,
        bindless_uab,
    }
}

pub(super) struct CullInputs<'a> {
    pub(super) world: &'a SceneData<'a>,
    pub(super) world_shaders: &'a [WorldShader<'a>],
    pub(super) plan: &'a CullPlan,
    pub(super) occlusion_two_pass: bool,
    pub(super) descriptors: &'a VkDescriptors,
    pub(super) targets: &'a VkTargets,
    pub(super) scene: &'a VkSceneAssets,
    pub(super) shadow: &'a VkShadow,
    pub(super) gbuffer: Option<&'a GbufferResources>,
    pub(super) swapchain_format: vk::Format,
}

// Build the cull state, and the probe convolution kernels the bake needs, which
// it returns.
pub(super) fn build_cull(
    gpu: &InitGpu<'_>,
    inputs: CullInputs<'_>,
) -> RenderResult<(VkCull, Option<ProbePrefilterPipelines>)> {
    let CullInputs {
        world,
        world_shaders,
        plan,
        occlusion_two_pass,
        descriptors,
        targets,
        scene,
        shadow,
        gbuffer,
        swapchain_format,
    } = inputs;
    let bindless = bindless::build_bindless_pass(
        gpu,
        bindless::BindlessInputs {
            world_shaders,
            plan,
            descriptors,
            targets,
            scene,
            swapchain_format,
        },
    )?;
    let world_pipelines =
        bindless::build_world_pipelines(gpu, &bindless, world_shaders, targets, swapchain_format)?;
    let shader_bucket_count = 1 + world_pipelines.len();
    let compute = compute::build_compute_cull(
        gpu,
        compute::ComputeInputs {
            world,
            scene,
            targets,
            descriptors,
            plan,
            bindless: &bindless,
            shader_bucket_count,
            occlusion_two_pass,
        },
    )?;
    let shadow_cull =
        shadow::build_shadow_cull(gpu, &bindless, &compute, shadow, descriptors, plan)?;
    let gbuffer_pass =
        gbuffer::build_gbuffer_pass(gpu, &bindless, &compute, gbuffer, descriptors, plan)?;
    // The reflection-probe convolution kernels, under the same gate the bake
    // itself needs: a probe capture renders through the bindless GPU cull, so a
    // world without the cull pipeline never bakes one and never needs them.
    let probe_prefilter = match compute.pipeline.is_some() {
        true => Some(ProbePrefilterPipelines::new(
            &gpu.hw.device,
            gpu.hot_reload,
        )?),
        false => None,
    };
    let two_pass = two_pass::build_two_pass_cull(
        gpu,
        &bindless,
        &compute,
        plan,
        shader_bucket_count,
        targets.msaa_samples,
        occlusion_two_pass,
    )?;
    let cull = VkCull {
        bindless_pipeline: bindless.pipeline,
        bindless_pipeline_layout: bindless.pipeline_layout,
        bindless_set_layout: bindless.set_layout,
        bindless_pool_size: plan.bindless_pool_size,
        bindless_update_after_bind: plan.bindless_uab,
        world_pipelines,
        bucket_stride: plan.n_cull,
        bindless_main_spv: bindless.main_spv,
        bindless_sets: bindless.sets,
        object_buffers: bindless.object_buffers,
        cull_pipeline: compute.pipeline,
        cull_pipeline_layout: compute.pipeline_layout,
        cull_set_layout: compute.set_layout,
        cull_sets: compute.sets,
        draw_args_buffers: compute.draw_args_buffers,
        indirect_buffers: compute.indirect_buffers,
        cull_status_buffers: compute.status_buffers,
        occlusion_two_pass,
        cull_pipeline_phase2: two_pass.pipeline,
        cull_sets2: two_pass.sets,
        _two_pass_pool: two_pass.pool,
        indirect_buffers2: two_pass.indirect_buffers,
        main_render_pass_phase1: two_pass.main_render_pass_phase1,
        main_render_pass_phase2: two_pass.main_render_pass_phase2,
        hiz: compute.hiz,
        hiz_valid: false,
        hiz_prev_view_proj: IDENTITY,
        shadow_cull_pipeline: shadow_cull.cull_pipeline,
        shadow_cull_pipeline_layout: shadow_cull.cull_pipeline_layout,
        _shadow_cull_set_layout: shadow_cull.set_layout,
        shadow_cull_sets: shadow_cull.sets,
        shadow_bindless_pipeline: shadow_cull.bindless_pipeline,
        shadow_bindless_pipeline_layout: shadow_cull.bindless_pipeline_layout,
        shadow_indirect_buffers: shadow_cull.indirect_buffers,
        gbuffer_bindless_pipeline: gbuffer_pass.pipeline,
        gbuffer_bindless_pipeline_layout: gbuffer_pass.pipeline_layout,
        _gbuffer_set_layout: gbuffer_pass.set_layout,
        gbuffer_sets: gbuffer_pass.sets,
        prev_model_buffers: gbuffer_pass.prev_model_buffers,
        model_history: gbuffer_pass.model_history,
    };
    Ok((cull, probe_prefilter))
}
