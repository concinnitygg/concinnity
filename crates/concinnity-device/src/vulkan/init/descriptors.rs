//! Descriptor plumbing: the global set layout, the shared descriptor pool, and
//! the per-frame global and shadow global sets.

use ash::vk;
use concinnity_core::gfx::render_types::{self, InstancedCluster, ShadowUniforms};
use concinnity_core::render::error::RenderResult;

use super::InitGpu;
use super::cull::CullPlan;
use crate::vulkan::context::{VkDescriptors, VkHardware, VkShadow};
use crate::vulkan::descriptor_layout::{
    PoolSizes, StageDescriptors, global_set, shadow_global_set,
};
use crate::vulkan::global_set::GlobalBindings;
use crate::vulkan::owned::{OwnedDescriptorPool, OwnedSetLayout};
use crate::vulkan::resources::{alloc_descriptor_sets, create_descriptor_set_layout};

// The device's plain per-stage budget for samplers and sampled images, which
// the bindless texture pool is sized against.
pub(super) fn stage_limits(hw: &VkHardware) -> StageDescriptors {
    // SAFETY: a property query on a live handle; it only reads.
    let properties = unsafe {
        hw.instance
            .get_physical_device_properties(hw.physical_device)
    };
    crate::vulkan::descriptor_layout::stage_limits(&properties.limits)
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
    pool: SetPoolInputs<'_>,
    bindings: &GlobalBindings<'_>,
) -> RenderResult<VkDescriptors> {
    let global_set_layout = create_descriptor_set_layout(&gpu.hw.device, &global_set())?;
    let descriptor_pool = create_descriptor_pool(gpu, pool, bindings.shadow)?;
    let global_sets = write_global_sets(gpu, &global_set_layout, &descriptor_pool, bindings)?;
    let shadow_global_sets = write_shadow_global_sets(gpu, &descriptor_pool, bindings.shadow)?;
    Ok(VkDescriptors {
        global_set_layout,
        descriptor_pool,
        global_sets,
        shadow_global_sets,
    })
}

// Create the shared descriptor pool, sized for every set allocated from it: the
// global and shadow global sets, the text atlas and composite sets, and the
// bindless, cull, shadow cull and G-buffer sets.
fn create_descriptor_pool(
    gpu: &InitGpu<'_>,
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
    let pool_sizes = PoolSizes::default()
        .sets(&global_set(), n_frames)
        .sets(&shadow_global_set(), n_frames)
        // The GPU-driven G-buffer's GbView UBO per frame and per snapshot set.
        .add(
            vk::DescriptorType::UNIFORM_BUFFER,
            gbuffer_sets_count + history_sets_count,
        )
        // Text atlas + per-frame composite (6: HDR resolve + bloom mip 0 + 3D
        // color LUT + the 3 view-mode G-buffer channels), each with a sampler,
        // and the per-frame bindless texture pool, which has none.
        .add(
            vk::DescriptorType::SAMPLED_IMAGE,
            n_atlas + n_frames * 6 + bindless_pool_size as u32 * bindless_sets_count,
        )
        .add(vk::DescriptorType::SAMPLER, n_atlas + n_frames * 6)
        // One per cluster per frame (instance matrices) + one per frame for the
        // bindless GpuObjectData buffer + four per frame for the GPU-cull set
        // (object + draw-args + indirect-command + cull-status SSBOs) + three
        // per (frame, cascade) for the shadow cull sets + the G-buffer's
        // model-history slot and draw args per frame and object buffer and
        // history slot per snapshot set. The phase-2 cull sets (two-pass
        // occlusion) draw from their own dedicated pool.
        .add(
            vk::DescriptorType::STORAGE_BUFFER,
            n_cluster * n_frames
                + bindless_sets_count
                + 4 * bindless_sets_count
                + 3 * shadow_cull_set_count
                + 2 * gbuffer_sets_count
                + 2 * history_sets_count,
        )
        .build();
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
    // declares the same, and this pool allocates the bindless set.
    let mut pool_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(total_sets);
    if bindless_uab {
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
    bindings: &GlobalBindings<'_>,
) -> RenderResult<Vec<vk::DescriptorSet>> {
    let InitGpu { hw, frames, .. } = *gpu;
    let layouts = vec![layout.handle(); frames];
    let global_sets = alloc_descriptor_sets(&hw.device, pool.handle(), &layouts)?;
    for (i, &set) in global_sets.iter().enumerate() {
        bindings.frame(i).write(&hw.device, set);
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
