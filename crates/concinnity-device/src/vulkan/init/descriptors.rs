//! Descriptor plumbing: the global set layout and its sampler budget, the shared
//! descriptor pool, and the per-frame global and shadow global sets.

use ash::vk;
use concinnity_core::gfx::render_types::{self, InstancedCluster, LightUniforms, ShadowUniforms};
use concinnity_core::render::error::RenderResult;

use super::cull::CullPlan;
use super::{GlobalBindings, InitGpu};
use crate::vulkan::context::{VkDescriptors, VkHardware, VkShadow};
use crate::vulkan::owned::{OwnedDescriptorPool, OwnedSetLayout, VkDevice};
use crate::vulkan::resources::alloc_descriptor_sets;

// The device's per-stage sampler budget, which both the global set's probe cube
// array and the bindless texture pool are sized against.
pub(super) fn max_per_stage_samplers(hw: &VkHardware) -> u32 {
    // SAFETY: a property query on a live handle; it only reads.
    unsafe {
        hw.instance
            .get_physical_device_properties(hw.physical_device)
    }
    .limits
    .max_per_stage_descriptor_samplers
}

// How the global set is budgeted on this device, decided once before the layout
// and anything sized against it are built.
pub(super) struct GlobalSetBudget {
    pub(super) update_after_bind: bool,
    pub(super) probe_cube_count: u32,
}

pub(super) fn global_set_budget(hw: &VkHardware) -> GlobalSetBudget {
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
    let max_per_stage_samplers = max_per_stage_samplers(hw);
    let global_constrained =
        crate::vulkan::descriptor_layout::sampler_budget_is_constrained(max_per_stage_samplers);
    if global_constrained && !hw.update_after_bind {
        tracing::warn!(
            "global descriptor set: per-stage sampler budget ({max_per_stage_samplers}) is \
             too tight for the widest pass and update-after-bind is unavailable; the \
             reflection-probe cube array will be clamped"
        );
    }
    let update_after_bind = global_constrained && hw.update_after_bind;
    // Reflection-probe cube-array length this device affords. Sizes the
    // binding below, the descriptor pool, every probe cube write, the GLSL
    // arrays, and the placement list, so they can never disagree.
    let probe_cube_count = crate::vulkan::descriptor_layout::probe_cube_array_count(
        max_per_stage_samplers,
        update_after_bind,
    );
    if (probe_cube_count as usize) < concinnity_core::render::uniforms::MAX_PROBES {
        tracing::info!(
            "reflection probes: device sampler headroom binds {probe_cube_count} of {}",
            concinnity_core::render::uniforms::MAX_PROBES
        );
    }
    GlobalSetBudget {
        update_after_bind,
        probe_cube_count,
    }
}

pub(super) struct SetPoolInputs<'a> {
    pub(super) instanced_clusters: &'a [InstancedCluster],
    pub(super) text_atlas_count: usize,
    pub(super) plan: &'a CullPlan,
    pub(super) has_gbuffer: bool,
}

// Build the global set layout, the shared descriptor pool, and the per-frame
// global and shadow global sets allocated from it.
pub(super) fn build_descriptors(
    gpu: &InitGpu<'_>,
    budget: GlobalSetBudget,
    pool: SetPoolInputs<'_>,
    bindings: &GlobalBindings<'_>,
) -> RenderResult<VkDescriptors> {
    let global_set_layout = create_global_set_layout(&gpu.hw.device, &budget)?;
    let descriptor_pool = create_descriptor_pool(gpu, &budget, pool, bindings.shadow)?;
    let global_sets = write_global_sets(
        gpu,
        &global_set_layout,
        &descriptor_pool,
        budget.probe_cube_count,
        bindings,
    )?;
    let shadow_global_sets = write_shadow_global_sets(gpu, &descriptor_pool, bindings.shadow)?;
    Ok(VkDescriptors {
        global_set_layout,
        global_update_after_bind: budget.update_after_bind,
        probe_cube_count: budget.probe_cube_count,
        descriptor_pool,
        global_sets,
        shadow_global_sets,
    })
}

// Global set (set 0): the geometry path's view / light / shadow UBOs +
// shadow-map + IBL cubes + SSAO sampler (binding 6, bound to the pooled
// `ao_output` when SSAO is enabled, otherwise to the 1x1 `ssao_white`
// fallback so the main pass's `ambient *= ao` multiplier collapses to a
// pass-through) + ProbeSet UBO (binding 7) + the reflection-probe cube
// array (binding 8). Binding table + lock-down test live in
// `descriptor_layout.rs`. Built inline (not via the count-1
// `create_descriptor_set_layout` helper) because binding 8 is a
// `probe_cube_count` cube array; the count-1 bindings come from the locked
// `global_set()` table, then the array binding is appended (the same shape
// as the bindless texture pool's array binding).
fn create_global_set_layout(
    device: &VkDevice,
    budget: &GlobalSetBudget,
) -> RenderResult<OwnedSetLayout> {
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
            .descriptor_count(budget.probe_cube_count)
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
    if budget.update_after_bind {
        info = info.flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL);
    }
    device
        .create_descriptor_set_layout(&info)
        .map_err(|e| crate::vulkan::error::map_vk_result(e, "global set layout"))
}

// Create the shared descriptor pool, sized for every set allocated from it: the
// global and shadow global sets, the text atlas and composite sets, and the
// bindless, cull, shadow cull and G-buffer sets.
fn create_descriptor_pool(
    gpu: &InitGpu<'_>,
    budget: &GlobalSetBudget,
    pool: SetPoolInputs<'_>,
    shadow: &VkShadow,
) -> RenderResult<OwnedDescriptorPool> {
    let InitGpu { hw, frames, .. } = *gpu;
    let SetPoolInputs {
        instanced_clusters,
        text_atlas_count,
        plan,
        has_gbuffer,
    } = pool;
    let CullPlan {
        bindless_active,
        bindless_pool_size,
        bindless_uab,
        ..
    } = *plan;
    let probe_cube_count = budget.probe_cube_count;
    let n_cluster = instanced_clusters.len() as u32;
    let n_atlas = text_atlas_count.max(1) as u32;
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
    let shadow_cull_set_count = if bindless_active && shadow.pipeline.is_some() {
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
    if bindless_uab || budget.update_after_bind {
        pool_info = pool_info.flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);
    }
    hw.device
        .create_descriptor_pool(&pool_info)
        .map_err(|e| crate::vulkan::error::map_vk_result(e, "descriptor pool"))
}

// Allocate and write the per-frame global sets.
fn write_global_sets(
    gpu: &InitGpu<'_>,
    layout: &OwnedSetLayout,
    pool: &OwnedDescriptorPool,
    probe_cube_count: u32,
    bindings: &GlobalBindings<'_>,
) -> RenderResult<Vec<vk::DescriptorSet>> {
    let InitGpu { hw, frames, .. } = *gpu;
    let device = &hw.device;
    let GlobalBindings {
        uniforms,
        light_cull,
        shadow,
        spot_shadow,
        area_light,
        scene,
        targets,
    } = *bindings;
    let global_layouts: Vec<_> = (0..frames).map(|_| layout.handle()).collect();
    let global_sets = alloc_descriptor_sets(device, pool.handle(), &global_layouts)?;
    let view_ubo_size = std::mem::size_of::<crate::vulkan::draw::ViewUniforms>() as u64;
    let light_ubo_size = std::mem::size_of::<LightUniforms>() as u64;
    let shadow_ubo_size = std::mem::size_of::<ShadowUniforms>() as u64;
    let probe_set_ubo_size =
        std::mem::size_of::<concinnity_core::render::uniforms::ProbeSet>() as u64;
    // Update global sets.
    for (i, &set) in global_sets.iter().enumerate() {
        let view_info = vk::DescriptorBufferInfo::default()
            .buffer(uniforms.view_ubo_buffers[i].buffer())
            .offset(0)
            .range(view_ubo_size);
        let light_info = vk::DescriptorBufferInfo::default()
            .buffer(uniforms.light_ubo_buffers[i].buffer())
            .offset(0)
            .range(light_ubo_size);
        let shadow_info = vk::DescriptorBufferInfo::default()
            .buffer(shadow.ubos[i].buffer())
            .offset(0)
            .range(shadow_ubo_size);
        // Layout must match the post-cascade transition in draw.rs, which
        // flips the shadow array to SHADER_READ_ONLY_OPTIMAL.
        let shadow_img_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(shadow.map.view)
            .sampler(shadow.sampler.handle());
        // Same resting layout as the cascade array: the SpotShadow producer
        // barrier opens it for the depth loop and Main returns it here.
        let spot_shadow_img_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(spot_shadow.map.view)
            .sampler(shadow.sampler.handle());
        let spot_shadow_data_info = vk::DescriptorBufferInfo::default()
            .buffer(spot_shadow.data_buffer.buffer())
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let area_light_info = vk::DescriptorBufferInfo::default()
            .buffer(area_light.buffer.buffer())
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let ltc_matrix_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(area_light.ltc_matrix.view)
            .sampler(area_light.sampler.handle());
        let ltc_magnitude_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(area_light.ltc_magnitude.view)
            .sampler(area_light.sampler.handle());
        let irr_img_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(scene.env_map.irradiance.view)
            .sampler(scene.cube_sampler.handle());
        let pre_img_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(scene.env_map.prefilter.view)
            .sampler(scene.cube_sampler.handle());
        // SSAO occlusion: this frame's blurred occlusion when SSAO is on
        // (per frame in flight, pooled), or the 1×1 white fallback when it
        // is off. Either way the descriptor is bound so the main pass's
        // `ambient *= ao` always samples a valid texture.
        let ssao_view = targets
            .transient_pool
            .view_for("ao_output", i)
            .unwrap_or(scene.ssao_white.view);
        let ssao_img_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(ssao_view)
            .sampler(scene.linear_sampler.handle());
        // ProbeSet UBO (binding 7): this frame's reflection-probe set.
        let probe_set_info = vk::DescriptorBufferInfo::default()
            .buffer(uniforms.probe_set_ubo_buffers[i].buffer())
            .offset(0)
            .range(probe_set_ubo_size);
        // Local-light SSBO (binding 9): the single static buffer, bound into
        // every frame's global set.
        let local_light_info = vk::DescriptorBufferInfo::default()
            .buffer(uniforms.local_light_buffer.buffer())
            .offset(0)
            .range(uniforms.local_light_size);
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
                    .image_view(scene.env_map.prefilter.view)
                    .sampler(scene.cube_sampler.handle())
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
    Ok(global_sets)
}

// Allocate and write the per-frame shadow global sets over the shadow state's
// set layout and uniform ring.
fn write_shadow_global_sets(
    gpu: &InitGpu<'_>,
    pool: &OwnedDescriptorPool,
    shadow: &VkShadow,
) -> RenderResult<Vec<vk::DescriptorSet>> {
    let InitGpu { hw, frames, .. } = *gpu;
    let device = &hw.device;
    let layout = shadow.global_set_layout.handle();
    let shadow_global_layouts: Vec<_> = (0..frames).map(|_| layout).collect();
    let shadow_global_sets = alloc_descriptor_sets(device, pool.handle(), &shadow_global_layouts)?;
    let shadow_ubo_size = std::mem::size_of::<ShadowUniforms>() as u64;
    for (i, &set) in shadow_global_sets.iter().enumerate() {
        let su_info = vk::DescriptorBufferInfo::default()
            .buffer(shadow.ubos[i].buffer())
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
    Ok(shadow_global_sets)
}
