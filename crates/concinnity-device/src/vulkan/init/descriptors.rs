//! Descriptor plumbing: the set and pipeline layouts, the shared descriptor
//! pool with the per-frame global and shadow sets, and the text atlas,
//! composite and bloom input sets.

use ash::vk;
use concinnity_core::gfx::render_types::{self, InstancedCluster};
use concinnity_core::render::error::RenderResult;

use super::{GlobalBindings, InitGpu};
use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedPipelineLayout, OwnedSampler, OwnedSetLayout,
};
use crate::vulkan::post::bloom::{MAX_BLOOM_MIPS, alloc_bloom_input_sets, rebind_bloom_input0};
use crate::vulkan::post::reflection_composite::ReflectionCompositeResources;
use crate::vulkan::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use crate::vulkan::swapchain::write_composite_set;
use crate::vulkan::texture::GpuImage;

pub(super) struct DescriptorLayouts {
    pub(super) max_per_stage_samplers: u32,
    pub(super) global_update_after_bind: bool,
    pub(super) probe_cube_count: u32,
    pub(super) global_set_layout: OwnedSetLayout,
    pub(super) text_set_layout: OwnedSetLayout,
    pub(super) shadow_global_set_layout: OwnedSetLayout,
    pub(super) composite_set_layout: OwnedSetLayout,
    pub(super) bloom_set_layout: OwnedSetLayout,
    pub(super) shadow_pipeline_layout: OwnedPipelineLayout,
    pub(super) text_pipeline_layout: OwnedPipelineLayout,
    pub(super) composite_pipeline_layout: OwnedPipelineLayout,
    pub(super) bloom_pipeline_layout: OwnedPipelineLayout,
}

pub(super) fn build_layouts(
    gpu: &InitGpu<'_>,
    update_after_bind: bool,
) -> RenderResult<DescriptorLayouts> {
    let InitGpu {
        instance,
        device,
        physical_device,
        ..
    } = *gpu;
    // Global set 0 is bound by the geometry path, glass, and the SSR resolve
    // alike, so its sampler cost is paid by all three pipeline layouts.
    // `maxPerStageDescriptorSamplers` is 16 on MoltenVK (Metal's per-stage
    // sampler argument table, reported the same under every argument-buffer
    // mode) against six figures on desktop drivers, which is not enough for
    // the set plus the widest of those passes. Such a device declares the set
    // update-after-bind so it budgets against
    // `maxPerStageDescriptorUpdateAfterBindSamplers` (1024 on MoltenVK) and
    // drops out of the plain per-layout count entirely. Desktop stays on the
    // plain path untouched.
    let max_per_stage_samplers =
        // SAFETY: a property query on a live handle; it only reads.
        unsafe { instance.get_physical_device_properties(physical_device) }
            .limits
            .max_per_stage_descriptor_samplers;
    let global_constrained =
        crate::vulkan::descriptor_layout::sampler_budget_is_constrained(max_per_stage_samplers);
    if global_constrained && !update_after_bind {
        tracing::warn!(
            "global descriptor set: per-stage sampler budget ({max_per_stage_samplers}) is \
             too tight for the widest pass and update-after-bind is unavailable; the \
             reflection-probe cube array will be clamped"
        );
    }
    let global_update_after_bind = global_constrained && update_after_bind;
    // Reflection-probe cube-array length this device affords. Sizes the
    // binding below, the descriptor pool, every probe cube write, the GLSL
    // arrays, and the placement list, so they can never disagree.
    let probe_cube_count = crate::vulkan::descriptor_layout::probe_cube_array_count(
        max_per_stage_samplers,
        global_update_after_bind,
    );
    if (probe_cube_count as usize) < concinnity_core::render::uniforms::MAX_PROBES {
        tracing::info!(
            "reflection probes: device sampler headroom binds {probe_cube_count} of {}",
            concinnity_core::render::uniforms::MAX_PROBES
        );
    }
    // Global set (set 0): view UBO, light UBO, shadow UBO, shadow array
    // sampler (binding 3), IBL irradiance cube (4), IBL prefilter cube (5),
    // SSAO occlusion (binding 6, bound to the live `ssao.ao` image when
    // SSAO is enabled, otherwise to the 1×1 `ssao_white` fallback so the
    // main pass's `ambient *= ao` multiplier collapses to a pass-through).
    // Global set (set 0): the geometry path's view / light / shadow UBOs +
    // shadow-map + IBL cubes + SSAO sampler + ProbeSet UBO (binding 7) + the
    // reflection-probe cube array (binding 8). Binding table + lock-down test
    // live in `descriptor_layout.rs`. Built inline (not via the count-1
    // `create_descriptor_set_layout` helper) because binding 8 is a
    // `probe_cube_count` cube array; the count-1 bindings come from the locked
    // `global_set()` table, then the array binding is appended (the same shape
    // as the bindless texture pool's array binding below).
    let global_set_layout = {
        let mut bindings: Vec<vk::DescriptorSetLayoutBinding> =
            crate::vulkan::descriptor_layout::global_set()
                .iter()
                .map(|&(b, ty, stage)| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b)
                        .descriptor_type(ty)
                        .descriptor_count(1)
                        .stage_flags(stage)
                })
                .collect();
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(crate::vulkan::descriptor_layout::PROBE_CUBE_ARRAY_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(probe_cube_count)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        );
        // Binding 9: per-scene local-light SSBO (count-1 STORAGE_BUFFER, FS).
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(crate::vulkan::descriptor_layout::LOCAL_LIGHT_SSBO_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        );
        // Binding 10: ClusterParams UBO + binding 11: the per-cluster
        // light-index lists the LightCull compute pass writes.
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(crate::vulkan::descriptor_layout::CLUSTER_PARAMS_UBO_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        );
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(crate::vulkan::descriptor_layout::CLUSTER_LIGHT_LIST_SSBO_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        );
        // Spot shadows: the depth array the forward pass compares against
        // and the per-slice projections it projects through.
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(crate::vulkan::descriptor_layout::SPOT_SHADOW_MAP_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        );
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(crate::vulkan::descriptor_layout::SPOT_SHADOW_DATA_SSBO_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        );
        // Area lights: the per-scene table and the two LTC lookups.
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(crate::vulkan::descriptor_layout::AREA_LIGHT_SSBO_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        );
        for b in [
            crate::vulkan::descriptor_layout::LTC_MATRIX_BINDING,
            crate::vulkan::descriptor_layout::LTC_MAGNITUDE_BINDING,
        ] {
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            );
        }
        // On a sampler-constrained device the whole set is declared
        // update-after-bind, which is purely how it is budgeted: no binding
        // takes `VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT`, so the update
        // timing rules are unchanged and no extra descriptor-indexing feature
        // is required. Every pool that allocates the set must declare the
        // matching flag in turn.
        let mut info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        if global_update_after_bind {
            info = info.flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL);
        }
        device
            .create_descriptor_set_layout(&info)
            .map_err(|e| format!("global set layout: {e}"))?
    };
    // Text set (set 0 for text pass): atlas sampler.
    let text_set_layout = create_descriptor_set_layout(
        device,
        &[(
            0,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            vk::ShaderStageFlags::FRAGMENT,
        )],
    )?;
    // Shadow global set (set 0 for shadow pass): ShadowUniforms UBO.
    let shadow_global_set_layout = create_descriptor_set_layout(
        device,
        &crate::vulkan::descriptor_layout::shadow_global_set(),
    )?;
    // Composite set (set 0 for composite pass): HDR resolve image at
    // binding 0, bloom mip 0 at binding 1, the 3D color LUT at binding 2,
    // then the G-buffer channels the debug view modes visualize (3 =
    // normal+depth, 4 = roughness, 5 = SSAO occlusion).
    let composite_set_layout = create_descriptor_set_layout(
        device,
        &(0..6)
            .map(|b| {
                (
                    b,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    vk::ShaderStageFlags::FRAGMENT,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    // Bloom set (set 0 for every bloom pass): the single input image.
    let bloom_set_layout = create_descriptor_set_layout(
        device,
        &[(
            0,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            vk::ShaderStageFlags::FRAGMENT,
        )],
    )?;

    let shadow_pc_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        // 64 bytes for model + 16 bytes for cascade_idx + padding.
        .size(80);
    let shadow_set_layouts = [shadow_global_set_layout.handle()];
    let shadow_pipeline_layout = device
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&shadow_set_layouts)
                .push_constant_ranges(std::slice::from_ref(&shadow_pc_range)),
        )
        .map_err(|e| format!("shadow pipeline layout: {e}"))?;

    let text_pc_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(16);
    let text_set_layouts = [text_set_layout.handle()];
    let text_pipeline_layout = device
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&text_set_layouts)
                .push_constant_ranges(std::slice::from_ref(&text_pc_range)),
        )
        .map_err(|e| format!("text pipeline layout: {e}"))?;

    // Post-process push constant: the full `PostProcessParams` struct,
    // fragment-stage. Read by the bloom-prefilter shader.
    let post_pc_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(std::mem::size_of::<render_types::PostProcessParams>() as u32);

    // The composite shader reads the same tunables plus the scene fade, so
    // its range covers the wider `CompositeParams`.
    let composite_pc_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(std::mem::size_of::<render_types::CompositeParams>() as u32);

    // Composite layout: one descriptor set (HDR resolve + bloom mip 0).
    let composite_set_layouts = [composite_set_layout.handle()];
    let composite_pipeline_layout = device
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&composite_set_layouts)
                .push_constant_ranges(std::slice::from_ref(&composite_pc_range)),
        )
        .map_err(|e| format!("composite pipeline layout: {e}"))?;

    // Bloom layout: one descriptor set (the input image) + the shared
    // post-process push constant (read only by the prefilter).
    let bloom_set_layouts = [bloom_set_layout.handle()];
    let bloom_pipeline_layout = device
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&bloom_set_layouts)
                .push_constant_ranges(std::slice::from_ref(&post_pc_range)),
        )
        .map_err(|e| format!("bloom pipeline layout: {e}"))?;
    Ok(DescriptorLayouts {
        max_per_stage_samplers,
        global_update_after_bind,
        probe_cube_count,
        global_set_layout,
        text_set_layout,
        shadow_global_set_layout,
        composite_set_layout,
        bloom_set_layout,
        shadow_pipeline_layout,
        text_pipeline_layout,
        composite_pipeline_layout,
        bloom_pipeline_layout,
    })
}

pub(super) struct SetPoolInputs<'a> {
    pub(super) instanced_clusters: &'a [InstancedCluster],
    pub(super) gpu_text_atlases: &'a [GpuImage],
    pub(super) bindless_active: bool,
    pub(super) has_gbuffer: bool,
    pub(super) has_shadow_pipeline: bool,
    pub(super) bindless_pool_size: usize,
    pub(super) bindless_uab: bool,
    pub(super) probe_cube_count: u32,
    pub(super) global_update_after_bind: bool,
    pub(super) global_set_layout: &'a OwnedSetLayout,
    pub(super) shadow_global_set_layout: &'a OwnedSetLayout,
}

pub(super) struct DescriptorSets {
    pub(super) descriptor_pool: OwnedDescriptorPool,
    pub(super) gbuffer_active: bool,
    pub(super) global_sets: Vec<vk::DescriptorSet>,
    pub(super) shadow_global_sets: Vec<vk::DescriptorSet>,
}

pub(super) fn build_sets(
    gpu: &InitGpu<'_>,
    pool: SetPoolInputs<'_>,
    bindings: &GlobalBindings<'_>,
) -> RenderResult<DescriptorSets> {
    let InitGpu { device, frames, .. } = *gpu;
    let SetPoolInputs {
        instanced_clusters,
        gpu_text_atlases,
        bindless_active,
        has_gbuffer,
        has_shadow_pipeline,
        bindless_pool_size,
        bindless_uab,
        probe_cube_count,
        global_update_after_bind,
        global_set_layout,
        shadow_global_set_layout,
    } = pool;
    let GlobalBindings {
        view_ubo_buffers,
        view_ubo_size,
        probe_set_ubo_buffers,
        probe_set_ubo_size,
        light_ubo_buffers,
        light_ubo_size,
        shadow_ubos,
        shadow_ubo_size,
        local_light_buffer,
        local_light_buffer_size,
        light_cull,
        shadow_map,
        shadow_sampler,
        spot_shadow,
        area_light_buffer,
        ltc_matrix_image,
        ltc_magnitude_image,
        ltc_sampler,
        env_map,
        cube_sampler,
        transient_pool,
        ssao_white,
        linear_sampler,
    } = *bindings;
    let n_cluster = instanced_clusters.len() as u32;
    let n_atlas = gpu_text_atlases.len().max(1) as u32;
    let n_frames = frames as u32;
    let bindless_sets_count = if bindless_active { n_frames } else { 0 };
    // GPU-driven G-buffer pre-pass: one set 0 per frame (1 UBO + 2 SSBOs: the
    // previous frame's model-history slot and this frame's draw args),
    // allocated only when the bindless cull path is active AND the G-buffer is
    // enabled. The depth/MRT draw reuses the bindless GpuObjectData set (set 1),
    // so it adds no further sets here. The snapshot kernel that fills the ring
    // takes a square (frame, slot) table of its own, each set 1 UBO + 2 SSBOs.
    let gbuffer_active = bindless_active && has_gbuffer;
    let gbuffer_sets_count = if gbuffer_active { n_frames } else { 0 };
    let history_sets_count = gbuffer_sets_count * n_frames;

    // A pool size with descriptorCount 0 is invalid, so the storage-buffer
    // entry is only added when there are instanced clusters / bindless sets
    // to size it.
    let mut pool_sizes = vec![
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            // global (5 per frame: view + light + shadow + ProbeSet +
            // ClusterParams) + shadow global (1 per frame) + gbuffer bindless
            // GbView UBO (1 per frame).
            .descriptor_count(n_frames * 5 + n_frames + gbuffer_sets_count + history_sets_count),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            // per-frame {shadow + spot shadow + IBL irradiance + IBL
            // prefilter + SSAO occlusion + the 2 area-light LTC tables} +
            // per-frame probe cube array + text atlas + per-frame
            // composite(6: HDR resolve + bloom mip 0 + 3D color LUT + the 3
            // view-mode G-buffer channels) + per-frame bindless texture pool.
            .descriptor_count(
                n_frames * 7
                    + n_frames * probe_cube_count
                    + n_atlas
                    + n_frames * 6
                    + bindless_pool_size as u32 * bindless_sets_count,
            ),
    ];
    // GPU-driven shadow: one cull set per (frame, cascade), each with 3
    // STORAGE_BUFFER descriptors (objects + draw-args + that cascade's
    // indirect-command buffer). Allocated only when the bindless cull path is
    // active AND shadows are enabled. The depth-only shadow draw reuses the
    // shadow-global + bindless sets, so it adds no sets here.
    let shadow_cull_set_count = if bindless_active && has_shadow_pipeline {
        n_frames * render_types::NUM_SHADOW_CASCADES as u32
    } else {
        0
    };
    // Storage buffers: one per cluster per frame (instance matrices) + one
    // per frame for the bindless GpuObjectData buffer + four per frame for
    // the GPU-cull set (object + draw-args + indirect-command + cull-status
    // SSBOs) + three per (frame, cascade) for the shadow cull sets. The
    // phase-2 cull sets (two-pass occlusion) draw from their own dedicated
    // pool, so they don't enter this count.
    let storage_count = n_cluster * n_frames
        + bindless_sets_count
        + 4 * bindless_sets_count
        + 3 * shadow_cull_set_count
        // GPU-driven G-buffer: the model-history slot + the draw args per
        // frame, and the object buffer + history slot per snapshot set.
        + 2 * gbuffer_sets_count
        + 2 * history_sets_count
        // Per-scene local-light SSBO, the per-cluster light-list SSBO, the
        // spot shadow projections SSBO, and the area-light table: one of each
        // per global set (per frame).
        + n_frames
        + n_frames
        + n_frames
        + n_frames;
    if storage_count > 0 {
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(storage_count),
        );
    }
    // total sets: global (n_frames) + shadow global (n_frames) + atlas +
    // per-frame×cluster instance sets + per-frame composite sets + per-frame
    // bindless sets + per-frame GPU-cull sets.
    let total_sets = n_frames
        + n_frames
        + n_atlas
        + n_frames * n_cluster
        + n_frames
        + bindless_sets_count
        + bindless_sets_count
        + shadow_cull_set_count
        + gbuffer_sets_count
        + history_sets_count;
    // An update-after-bind set layout can only be allocated from a pool that
    // declares the same. This pool allocates both the global sets and the
    // bindless set, so either opting in forces the flag.
    let mut pool_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(total_sets);
    if bindless_uab || global_update_after_bind {
        pool_info = pool_info.flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);
    }
    let descriptor_pool = device
        .create_descriptor_pool(&pool_info)
        .map_err(|e| format!("descriptor pool: {e}"))?;

    // Global sets (one per frame).
    let global_layouts: Vec<_> = (0..frames).map(|_| global_set_layout.handle()).collect();
    let global_sets = alloc_descriptor_sets(device, descriptor_pool.handle(), &global_layouts)?;
    // Update global sets.
    for (i, &set) in global_sets.iter().enumerate() {
        let view_info = vk::DescriptorBufferInfo::default()
            .buffer(view_ubo_buffers[i].buffer())
            .offset(0)
            .range(view_ubo_size);
        let light_info = vk::DescriptorBufferInfo::default()
            .buffer(light_ubo_buffers[i].buffer())
            .offset(0)
            .range(light_ubo_size);
        let shadow_info = vk::DescriptorBufferInfo::default()
            .buffer(shadow_ubos[i].buffer())
            .offset(0)
            .range(shadow_ubo_size);
        // Layout must match the post-cascade transition in draw.rs, which
        // flips the shadow array to SHADER_READ_ONLY_OPTIMAL.
        let shadow_img_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(shadow_map.view)
            .sampler(shadow_sampler.handle());
        // Same resting layout as the cascade array: the SpotShadow producer
        // barrier opens it for the depth loop and Main returns it here.
        let spot_shadow_img_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(spot_shadow.map.view)
            .sampler(shadow_sampler.handle());
        let spot_shadow_data_info = vk::DescriptorBufferInfo::default()
            .buffer(spot_shadow.data_buffer.buffer())
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let area_light_info = vk::DescriptorBufferInfo::default()
            .buffer(area_light_buffer.buffer())
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let ltc_matrix_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(ltc_matrix_image.view)
            .sampler(ltc_sampler.handle());
        let ltc_magnitude_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(ltc_magnitude_image.view)
            .sampler(ltc_sampler.handle());
        let irr_img_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(env_map.irradiance.view)
            .sampler(cube_sampler.handle());
        let pre_img_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(env_map.prefilter.view)
            .sampler(cube_sampler.handle());
        // SSAO occlusion: this frame's blurred occlusion when SSAO is on
        // (per frame in flight, pooled), or the 1×1 white fallback when it
        // is off. Either way the descriptor is bound so the main pass's
        // `ambient *= ao` always samples a valid texture.
        let ssao_view = transient_pool
            .view_for("ao_output", i)
            .unwrap_or(ssao_white.view);
        let ssao_img_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(ssao_view)
            .sampler(linear_sampler.handle());
        // ProbeSet UBO (binding 7): this frame's reflection-probe set.
        let probe_set_info = vk::DescriptorBufferInfo::default()
            .buffer(probe_set_ubo_buffers[i].buffer())
            .offset(0)
            .range(probe_set_ubo_size);
        // Local-light SSBO (binding 9): the single static buffer, bound into
        // every frame's global set.
        let local_light_info = vk::DescriptorBufferInfo::default()
            .buffer(local_light_buffer.buffer())
            .offset(0)
            .range(local_light_buffer_size);
        // Clustered lighting: this frame's live ClusterParams (binding 10)
        // and the shared per-cluster light lists (binding 11).
        let cluster_params_info = vk::DescriptorBufferInfo::default()
            .buffer(light_cull.params_buffers[i].buffer())
            .offset(0)
            .range(std::mem::size_of::<render_types::ClusterParams>() as u64);
        let cluster_list_info = vk::DescriptorBufferInfo::default()
            .buffer(light_cull.cluster_buffer.buffer())
            .offset(0)
            .range(crate::vulkan::light_cull::cluster_list_size());
        // Probe cube array (binding 8): every slot points at the IBL prefilter
        // cube until a probe bakes. No descriptor-indexing extension is
        // enabled, so every one of the `probe_cube_count` descriptors must hold
        // a valid cube (an unwritten slot is UB); the EMPTY ProbeSet (count 0)
        // keeps the shader on the sky path, so these are never actually sampled
        // yet.
        let probe_cube_infos: Vec<vk::DescriptorImageInfo> = (0..probe_cube_count)
            .map(|_| {
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(env_map.prefilter.view)
                    .sampler(cube_sampler.handle())
            })
            .collect();
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&view_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&light_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&shadow_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&shadow_img_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&irr_img_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&pre_img_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(6)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&ssao_img_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(7)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&probe_set_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(crate::vulkan::descriptor_layout::PROBE_CUBE_ARRAY_BINDING)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&probe_cube_infos),
            // Binding 9: per-scene local-light SSBO.
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(crate::vulkan::descriptor_layout::LOCAL_LIGHT_SSBO_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&local_light_info)),
            // Binding 10: ClusterParams UBO (this frame's live params).
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(crate::vulkan::descriptor_layout::CLUSTER_PARAMS_UBO_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&cluster_params_info)),
            // Binding 11: per-cluster light-index lists.
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(crate::vulkan::descriptor_layout::CLUSTER_LIGHT_LIST_SSBO_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&cluster_list_info)),
            // Binding 12: spot shadow depth array.
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(crate::vulkan::descriptor_layout::SPOT_SHADOW_MAP_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&spot_shadow_img_info)),
            // Binding 13: per-slice spot shadow projections.
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(crate::vulkan::descriptor_layout::SPOT_SHADOW_DATA_SSBO_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&spot_shadow_data_info)),
            // Binding 14: the per-scene area-light table.
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(crate::vulkan::descriptor_layout::AREA_LIGHT_SSBO_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&area_light_info)),
            // Bindings 15 + 16: the two area-light LTC lookup tables.
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(crate::vulkan::descriptor_layout::LTC_MATRIX_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&ltc_matrix_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(crate::vulkan::descriptor_layout::LTC_MAGNITUDE_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&ltc_magnitude_info)),
        ];
        // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
        // every set and resource it names belongs to this device.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    // Shadow global sets (one per frame).
    let shadow_global_layouts: Vec<_> = (0..frames)
        .map(|_| shadow_global_set_layout.handle())
        .collect();
    let shadow_global_sets =
        alloc_descriptor_sets(device, descriptor_pool.handle(), &shadow_global_layouts)?;
    for (i, &set) in shadow_global_sets.iter().enumerate() {
        let su_info = vk::DescriptorBufferInfo::default()
            .buffer(shadow_ubos[i].buffer())
            .offset(0)
            .range(shadow_ubo_size);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&su_info));
        // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
        // every set and resource it names belongs to this device.
        unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
    }
    Ok(DescriptorSets {
        descriptor_pool,
        gbuffer_active,
        global_sets,
        shadow_global_sets,
    })
}

pub(super) struct PostSetInputs<'a> {
    pub(super) gpu_text_atlases: &'a [GpuImage],
    pub(super) text_set_layout: &'a OwnedSetLayout,
    pub(super) text_sampler: &'a OwnedSampler,
    pub(super) descriptor_pool: &'a OwnedDescriptorPool,
    pub(super) composite_set_layout: &'a OwnedSetLayout,
    pub(super) composite_sampler: &'a OwnedSampler,
    pub(super) composite_opt: Option<&'a ReflectionCompositeResources>,
    pub(super) hdr_resolve_images: &'a [GpuImage],
    pub(super) bloom_mips: &'a [Vec<GpuImage>],
    pub(super) bloom_set_layout: &'a OwnedSetLayout,
    pub(super) color_lut: &'a GpuImage,
}

pub(super) struct PostSets {
    pub(super) text_atlas_sets: Vec<vk::DescriptorSet>,
    pub(super) composite_sets: Vec<vk::DescriptorSet>,
    pub(super) bloom_descriptor_pool: OwnedDescriptorPool,
    pub(super) bloom_input_sets: Vec<Vec<vk::DescriptorSet>>,
}

pub(super) fn build_post_sets(
    gpu: &InitGpu<'_>,
    inputs: PostSetInputs<'_>,
) -> RenderResult<PostSets> {
    let InitGpu { device, frames, .. } = *gpu;
    let n_frames = frames as u32;
    let PostSetInputs {
        gpu_text_atlases,
        text_set_layout,
        text_sampler,
        descriptor_pool,
        composite_set_layout,
        composite_sampler,
        composite_opt,
        hdr_resolve_images,
        bloom_mips,
        bloom_set_layout,
        color_lut,
    } = inputs;
    // Text atlas sets.
    let text_atlas_layouts: Vec<_> = gpu_text_atlases
        .iter()
        .map(|_| text_set_layout.handle())
        .collect();
    let text_atlas_sets = if text_atlas_layouts.is_empty() {
        vec![]
    } else {
        let sets = alloc_descriptor_sets(device, descriptor_pool.handle(), &text_atlas_layouts)?;
        for (&set, atlas) in sets.iter().zip(gpu_text_atlases.iter()) {
            let img_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(atlas.view)
                .sampler(text_sampler.handle());
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&img_info));
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        }
        sets
    };

    // Composite sets (one per frame-in-flight slot): binding 0 = the
    // scene image (SSR output when SSR is on, else this slot's HDR
    // resolve), binding 1 = that slot's bloom mip 0, binding 2 = the
    // shared 3D color LUT. The TAA wiring later overrides binding 0 to
    // the TAA output when TAA is on.
    let composite_layouts: Vec<_> = (0..frames).map(|_| composite_set_layout.handle()).collect();
    let composite_sets =
        alloc_descriptor_sets(device, descriptor_pool.handle(), &composite_layouts)?;
    for (i, &set) in composite_sets.iter().enumerate() {
        // Scene image: the reflection composite output (the SSR / RT reflection
        // blended over the scene) when a reflection path is active, else the raw
        // HDR resolve (a SSGI-only build composited its bounce into the latter
        // upstream). The TAA / upscale wiring overrides this later.
        let scene_view = composite_opt
            .as_ref()
            .map(|c| c.output.view)
            .unwrap_or(hdr_resolve_images[i].view);
        write_composite_set(
            device,
            set,
            scene_view,
            bloom_mips[i][0].view,
            color_lut.view,
            composite_sampler.handle(),
        );
    }

    // A dedicated, resettable pool isolates bloom's variable set count
    // (the octave count can shift on resize) from the main pool. Sized for
    // the worst case (`MAX_BLOOM_MIPS + 1` sets per frame).
    let bloom_pool_capacity = n_frames * (MAX_BLOOM_MIPS + 1);
    let bloom_pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(bloom_pool_capacity);
    let bloom_descriptor_pool = device
        .create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(std::slice::from_ref(&bloom_pool_size))
                .max_sets(bloom_pool_capacity),
        )
        .map_err(|e| format!("bloom descriptor pool: {e}"))?;
    let bloom_input_sets = alloc_bloom_input_sets(
        device,
        bloom_descriptor_pool.handle(),
        bloom_set_layout.handle(),
        composite_sampler.handle(),
        hdr_resolve_images,
        bloom_mips,
    )?;
    // The reflection composite replaces the bloom prefilter's scene input
    // (input 0) with its output, the same scene image the composite pass
    // samples when a reflection path is active and TAA is off (a SSGI-only
    // build leaves the prefilter on the raw HDR resolve). One shared image, so
    // every frame's prefilter input 0 points at it.
    if let Some(view) = composite_opt.map(|c| c.output.view) {
        for frame_sets in &bloom_input_sets {
            rebind_bloom_input0(device, frame_sets[0], view, composite_sampler.handle());
        }
    }
    Ok(PostSets {
        text_atlas_sets,
        composite_sets,
        bloom_descriptor_pool,
        bloom_input_sets,
    })
}
