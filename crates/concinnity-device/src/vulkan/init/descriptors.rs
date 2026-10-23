//! Descriptor plumbing: the global set layout, the shared descriptor pool, and
//! the per-frame global and shadow global sets.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::render_types::{self, InstancedCluster, ShadowUniforms};
use concinnity_engine::render::backend::error::RenderResult;

use super::InitGpu;
use super::cull::CullPlan;
use crate::vulkan::context::{VkDescriptors, VkHardware, VkShadow};
use crate::vulkan::descriptor_layout::{
    PoolSizes, StageDescriptors, global_set, shadow_global_set,
};
use crate::vulkan::global_set::GlobalBindings;
use crate::vulkan::owned::{OwnedDescriptorPool, OwnedSetLayout};
use crate::vulkan::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use crate::vulkan::set_writes::SetWrites;

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

#[derive(Clone, Copy)]
pub(super) struct SetPoolInputs<'a> {
    pub(super) instanced_clusters: &'a [InstancedCluster],
    pub(super) text_atlas_count: usize,
    pub(super) plan: &'a CullPlan,
}

// Build the global set layout, the shared descriptor pool, and the per-frame
// global and shadow global sets allocated from it.
pub(super) fn build_descriptors(
    gpu: &InitGpu<'_>,
    pool: SetPoolInputs<'_>,
    bindings: &GlobalBindings<'_>,
) -> RenderResult<VkDescriptors> {
    let global_set_layout = create_descriptor_set_layout(&gpu.hw.device, &global_set())?;
    let shadow_views =
        u32::saturating_from(render_types::NUM_SHADOW_CASCADES) + bindings.spot_shadow.count();
    let descriptor_pool = create_descriptor_pool(gpu, pool, bindings.shadow, shadow_views)?;
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
    // Cascades plus spot slices: each gets a cull set per frame.
    shadow_views: u32,
) -> RenderResult<OwnedDescriptorPool> {
    let InitGpu { hw, frames, .. } = *gpu;
    let SetPoolInputs {
        instanced_clusters,
        text_atlas_count,
        plan,
    } = pool;
    let CullPlan {
        bindless_active,
        bindless_pool_size,
        bindless_uab,
        ..
    } = *plan;
    let n_cluster = u32::saturating_from(instanced_clusters.len());
    let n_atlas = u32::saturating_from(text_atlas_count.max(1));
    let n_frames = u32::saturating_from(frames);
    let bindless_sets_count = if bindless_active { n_frames } else { 0 };
    // GPU-driven G-buffer pre-pass: one set per frame (1 UBO + 2 SSBOs: the
    // previous frame's model-history slot and this frame's draw args),
    // reserved whenever the bindless cull path is active, since a quality
    // change that adds a G-buffer consumer allocates them mid-session. The
    // depth/MRT draw reuses the global and bindless sets, so it adds no further
    // sets here. The snapshot kernel that fills the ring takes a square
    // (frame, slot) table of its own, each set 1 UBO + 2 SSBOs.
    let gbuffer_sets_count = if bindless_active { n_frames } else { 0 };
    let history_sets_count = gbuffer_sets_count * n_frames;

    // GPU-driven shadow: one cull set per (frame, view) over the cascades and
    // the spot slices, each with 3 STORAGE_BUFFER descriptors (objects +
    // draw-args + that view's indirect-command buffer). Allocated only when the
    // bindless cull path is active AND shadows are enabled. The depth-only
    // shadow draw reuses the shadow-global, spot-slice and bindless sets, so it
    // adds no sets here.
    let shadow_cull_set_count = if bindless_active && shadow.enabled() {
        n_frames * shadow_views
    } else {
        0
    };
    let pool_sizes = pool_sizes(&PoolCounts {
        frames: n_frames,
        atlases: n_atlas,
        clusters: n_cluster,
        bindless_sets: bindless_sets_count,
        bindless_pool_size: u32::saturating_from(bindless_pool_size),
        gbuffer_sets: gbuffer_sets_count,
        history_sets: history_sets_count,
        shadow_cull_sets: shadow_cull_set_count,
    })
    .build();
    // total sets: global (n_frames) + shadow global (n_frames) + atlas +
    // per-frame×cluster instance sets + per-frame composite sets + per-frame
    // bindless sets + per-frame GPU-cull and packing sets.
    let total_sets = n_frames
        + n_frames
        + n_atlas
        + n_frames * n_cluster
        + n_frames
        + bindless_sets_count
        + 2 * bindless_sets_count
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

// How many of each set the shared pool allocates.
struct PoolCounts {
    frames: u32,
    atlases: u32,
    clusters: u32,
    bindless_sets: u32,
    bindless_pool_size: u32,
    gbuffer_sets: u32,
    history_sets: u32,
    shadow_cull_sets: u32,
}

// The shared pool's descriptors for `c`.
fn pool_sizes(c: &PoolCounts) -> PoolSizes {
    PoolSizes::default()
        .sets(&global_set(), c.frames)
        .sets(&shadow_global_set(), c.frames)
        // The GPU-driven G-buffer's GbView UBO per frame and per snapshot set.
        .add(
            vk::DescriptorType::UNIFORM_BUFFER,
            c.gbuffer_sets + c.history_sets,
        )
        // Text atlas + per-frame composite (HDR resolve + bloom top octave + 3D
        // color LUT + the view-mode channels), each with a sampler, and the
        // per-frame bindless texture pool, which has none.
        .add(
            vk::DescriptorType::SAMPLED_IMAGE,
            c.atlases
                + c.frames * super::composite::COMPOSITE_SOURCES
                + c.bindless_pool_size * c.bindless_sets,
        )
        .add(
            vk::DescriptorType::SAMPLER,
            c.atlases + c.frames * super::composite::COMPOSITE_SOURCES,
        )
        // One per cluster per frame (instance matrices) + three per frame for
        // the bindless set's GpuObjectData buffer, material parameter table and
        // vertex stream + three per (frame, cascade) for the shadow cull sets +
        // the G-buffer's model-history slot and draw args per frame and object
        // buffer and history slot per snapshot set. The phase-2 cull sets
        // (two-pass occlusion) draw from their own dedicated pool.
        .add(
            vk::DescriptorType::STORAGE_BUFFER,
            c.clusters * c.frames
                + 3 * c.bindless_sets
                + 3 * c.shadow_cull_sets
                + 2 * c.gbuffer_sets
                + 2 * c.history_sets,
        )
        // The GPU-cull and packing sets per frame.
        .sets(&super::main_cull_set(), c.bindless_sets)
        .sets(&crate::vulkan::compact::compact_set(), c.bindless_sets)
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
    let shadow_ubo_size = size_of::<ShadowUniforms>() as u64;
    for (&set, ubo) in shadow_global_sets.iter().zip(&shadow.ubos) {
        SetWrites::new(set)
            .uniform_buffer(0, ubo.buffer(), shadow_ubo_size)
            .apply(device);
    }
    Ok(shadow_global_sets)
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    // The shared pool holds every binding of the sets it allocates by layout,
    // the GPU-cull set's included, beside its hand-counted ones.
    #[test]
    fn the_shared_pool_covers_its_cull_sets() {
        let c = PoolCounts {
            frames: 3,
            atlases: 1,
            clusters: 0,
            bindless_sets: 3,
            bindless_pool_size: 0,
            gbuffer_sets: 0,
            history_sets: 0,
            shadow_cull_sets: 0,
        };
        let pool = pool_sizes(&c);
        assert!(pool.covers(&[
            (&global_set(), 3),
            (&shadow_global_set(), 3),
            (&super::super::main_cull_set(), 3),
            (&crate::vulkan::compact::compact_set(), 3),
        ]));
        let bindless_ssbos = 3 * c.bindless_sets;
        let need = PoolSizes::default()
            .sets(&global_set(), 3)
            .sets(&super::super::main_cull_set(), 3)
            .sets(&crate::vulkan::compact::compact_set(), 3)
            .count(vk::DescriptorType::STORAGE_BUFFER);
        assert!(pool.count(vk::DescriptorType::STORAGE_BUFFER) >= need + bindless_ssbos);
    }
}
