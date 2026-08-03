// src/vulkan/descriptor_layout.rs
//
// Canonical descriptor-set binding tables for the geometry render path, kept in
// one place so the `layout(set = N, binding = M)` indices the GLSL shaders use
// stay greppable and locked. `init.rs` builds the real `vk::DescriptorSetLayout`s
// from these via `create_descriptor_set_layout`, and the per-frame descriptor
// writes target the same binding numbers. The unit tests below assert each table
// is gap-free + unique and pin the binding -> (type, stage) contract, so a
// reordering, retype, or stage-flag change that would silently desync from the
// shaders fails `cargo test` instead of reading garbage on the GPU. Vulkan
// analogue of `directx/init/heap_layout.rs`'s slot tests.
//
// Only the geometry-path sets (global / per-object / shadow) are centralized
// here; the post-process sets (composite, bloom, text) are simpler 1-3 binding
// layouts still declared inline in `init.rs`.

use ash::vk;

use super::probe_uniforms::MAX_PROBES;

// One descriptor binding: (binding index, descriptor type, shader stages).
pub(in crate::vulkan) type Binding = (u32, vk::DescriptorType, vk::ShaderStageFlags);

// Global set (set 0), shared by the main / instanced / skinned pipelines:
//   0  ViewUniforms UBO          (VS + FS)
//   1  LightUniforms UBO         (FS)
//   2  ShadowUniforms UBO        (VS + FS)
//   3  shadow-map cascade array  (FS)
//   4  irradiance cube           (FS)
//   5  prefiltered env cube      (FS)
//   6  SSAO occlusion / fallback (FS)
//   7  ProbeSet UBO              (FS)
// Binding 8 is the reflection-probe cube array; it carries the
// `probe_cube_array_count` descriptors rather than 1, so it does NOT go through
// `create_descriptor_set_layout` (which fixes every binding at count 1). It is
// appended inline where the global layout is built in `init.rs` (the same way the
// bindless texture pool's array binding is built); `PROBE_CUBE_ARRAY_BINDING` is
// its locked binding number.
pub(in crate::vulkan) fn global_set() -> [Binding; 8] {
    use vk::DescriptorType as T;
    use vk::ShaderStageFlags as S;
    [
        (0, T::UNIFORM_BUFFER, S::VERTEX | S::FRAGMENT),
        (1, T::UNIFORM_BUFFER, S::FRAGMENT),
        (2, T::UNIFORM_BUFFER, S::VERTEX | S::FRAGMENT),
        (3, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
        (4, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
        (5, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
        (6, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
        (7, T::UNIFORM_BUFFER, S::FRAGMENT),
    ]
}

// Binding number of the reflection-probe cube array in global set 0. A
// `samplerCube probe_cubes[N]` array sized by `probe_cube_array_count`, appended
// to the layout inline in `init.rs` since the count-1 layout helper cannot
// express it. Sits exactly one past the count-1 `global_set()` table.
pub(in crate::vulkan) const PROBE_CUBE_ARRAY_BINDING: u32 = 8;

// Binding number of the per-scene local-light storage buffer (SSBO) in global
// set 0: a `readonly buffer LocalLightBlock { GpuLight local_lights[]; }`,
// count-1 STORAGE_BUFFER in the fragment stage, created once at init and bound
// static. Appended to the global layout inline in `init.rs` alongside the probe
// cube array (which caps the count-1 `global_set()` table at binding 7), one
// past `PROBE_CUBE_ARRAY_BINDING`.
pub(in crate::vulkan) const LOCAL_LIGHT_SSBO_BINDING: u32 = 9;

// Binding number of the clustered-lighting `ClusterParams` UBO in global set 0:
// a count-1 UNIFORM_BUFFER in the fragment stage. The main camera binds this
// frame's live params; the planar / probe re-renders bind the static
// `use_clusters = 0` copy. Appended inline in `init.rs` alongside the probe cube
// array + local-light SSBO, one past `LOCAL_LIGHT_SSBO_BINDING`.
pub(in crate::vulkan) const CLUSTER_PARAMS_UBO_BINDING: u32 = 10;

// Binding number of the per-cluster light-index list SSBO in global set 0: a
// count-1 STORAGE_BUFFER in the fragment stage, written by the `LightCull`
// compute pass. Appended inline in `init.rs`, one past
// `CLUSTER_PARAMS_UBO_BINDING`.
pub(in crate::vulkan) const CLUSTER_LIGHT_LIST_SSBO_BINDING: u32 = 11;

// Binding number of the spot shadow depth array in global set 0: a count-1
// `sampler2DArrayShadow` in the fragment stage, one layer per shadow-casting
// spot. A world with no shadowed spot binds a 1x1 fallback array so the
// descriptor is never left unwritten. Appended inline in `init.rs`, one past
// `CLUSTER_LIGHT_LIST_SSBO_BINDING`.
pub(in crate::vulkan) const SPOT_SHADOW_MAP_BINDING: u32 = 12;

// Binding number of the per-slice `SpotShadowData` storage buffer in global set
// 0: a count-1 STORAGE_BUFFER in the fragment stage holding each slice's
// light-space projection, indexed by `GpuLight.shadow_index`. Written once at
// init (local lights are static). Appended inline in `init.rs`, one past
// `SPOT_SHADOW_MAP_BINDING`.
pub(in crate::vulkan) const SPOT_SHADOW_DATA_SSBO_BINDING: u32 = 13;

// Binding number of the per-scene `AreaLightData` storage buffer in global set
// 0: a count-1 STORAGE_BUFFER in the fragment stage holding each rectangular
// area light's in-plane edge vectors, indexed by `GpuLight.data_index`. Written
// once at init (local lights are static). Appended inline in `init.rs`, one past
// `SPOT_SHADOW_DATA_SSBO_BINDING`.
pub(in crate::vulkan) const AREA_LIGHT_SSBO_BINDING: u32 = 14;

// Binding numbers of the two area-light LTC lookup tables in global set 0, both
// count-1 COMBINED_IMAGE_SAMPLER in the fragment stage: 15 is the inverse
// transform (RGBA32F), 16 the magnitude / Fresnel pair (RG32F). Scene
// independent (fitted at build time), so they are uploaded and bound even with
// no area light declared.
pub(in crate::vulkan) const LTC_MATRIX_BINDING: u32 = 15;
pub(in crate::vulkan) const LTC_MAGNITUDE_BINDING: u32 = 16;

// The count-1 sampler bindings appended to global set 0 inline in `init.rs`,
// past the `global_set()` table. Listed here so `probe_cube_array_count` can
// budget against every sampler the geometry pipeline layout declares.
const INLINE_GLOBAL_SAMPLERS: [u32; 3] = [
    SPOT_SHADOW_MAP_BINDING,
    LTC_MATRIX_BINDING,
    LTC_MAGNITUDE_BINDING,
];

fn count_fragment_samplers(bindings: &[Binding]) -> u32 {
    bindings
        .iter()
        .filter(|&&(_, ty, stage)| {
            ty == vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                && stage.contains(vk::ShaderStageFlags::FRAGMENT)
        })
        .count() as u32
}

// Count-1 fragment samplers global set 0 declares, outside the reflection-probe
// cube array: the `global_set()` table's shadow / IBL / SSAO taps plus the
// inline-appended spot shadow array and LTC tables. Every pipeline layout that
// binds set 0 pays this.
fn global_fragment_samplers() -> u32 {
    count_fragment_samplers(&global_set()) + INLINE_GLOBAL_SAMPLERS.len() as u32
}

// Count-1 fragment samplers per-object set 1 declares (albedo + normal map).
// Only the geometry path pays these; the bindless path replaces the set with its
// texture pool.
fn object_fragment_samplers() -> u32 {
    count_fragment_samplers(&object_set())
}

// Fragment-stage samplers the geometry pipeline layout (global set 0 + per-object
// set 1) declares outside the reflection-probe cube array. All are
// descriptorCount 1, so this is the fixed cost the probe array is budgeted
// against.
fn fixed_fragment_samplers() -> u32 {
    global_fragment_samplers() + object_fragment_samplers()
}

// How many descriptors the reflection-probe cube array (global set 0, binding 8)
// may declare on a device reporting `max_per_stage_samplers` for
// `VkPhysicalDeviceLimits::maxPerStageDescriptorSamplers`. `MAX_PROBES` is the
// CPU-side ceiling; a driver with less headroom than that binds fewer probes
// rather than building a pipeline layout it rejects. MoltenVK reports 16, which
// leaves room for `MAX_PROBES - 1`; desktop Vulkan drivers report six figures or
// more, where this is exactly `MAX_PROBES`. Never returns 0: the GLSL array
// declaration needs at least one element, and a device with no headroom at all
// cannot run the geometry path either way.
pub(in crate::vulkan) fn probe_cube_array_count(max_per_stage_samplers: u32) -> u32 {
    let headroom = max_per_stage_samplers.saturating_sub(fixed_fragment_samplers());
    headroom.clamp(1, MAX_PROBES as u32)
}

// Whether the bindless main pipeline layout must declare its texture pool
// `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT`. The layout is
// global set 0 + the bindless set, so its per-stage sampler cost is the count-1
// global samplers, the probe cube array, and the pool itself; unlike the probe
// array the pool cannot be clamped, since its length is the world's texture
// table. Descriptors in an update-after-bind set layout are budgeted against
// `maxPerStageDescriptorUpdateAfterBindSamplers` instead, which MoltenVK reports
// as 1024 against a plain limit of 16. Desktop drivers report six figures for
// both and never take the update-after-bind path.
pub(in crate::vulkan) fn bindless_pool_needs_update_after_bind(
    max_per_stage_samplers: u32,
    probe_cube_count: u32,
    pool_size: u32,
) -> bool {
    let declared = global_fragment_samplers() + probe_cube_count + pool_size;
    declared > max_per_stage_samplers
}

// Whether a device's per-stage sampler budget is too tight to seat the geometry
// path at its `MAX_PROBES` ceiling. Such a device cannot seat a bindless texture
// pool of any useful length either, so device creation enables the
// descriptor-indexing update-after-bind features the pool's set layout opts into
// (see `bindless_pool_needs_update_after_bind`). True on MoltenVK, false on every
// desktop driver, which leaves their device-creation feature chain untouched.
pub(in crate::vulkan) fn sampler_budget_is_constrained(max_per_stage_samplers: u32) -> bool {
    max_per_stage_samplers < fixed_fragment_samplers() + MAX_PROBES as u32
}

// Per-object set (set 1): albedo at 0, normal map at 1.
pub(in crate::vulkan) fn object_set() -> [Binding; 2] {
    use vk::DescriptorType as T;
    use vk::ShaderStageFlags as S;
    [
        (0, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
        (1, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
    ]
}

// Shadow global set (set 0 for the shadow pass): ShadowUniforms UBO, vertex-only
// (the shadow fragment stage is a depth-only no-op).
pub(in crate::vulkan) fn shadow_global_set() -> [Binding; 1] {
    [(
        0,
        vk::DescriptorType::UNIFORM_BUFFER,
        vk::ShaderStageFlags::VERTEX,
    )]
}

#[cfg(test)]
mod tests {
    use super::*;

    // Sorted binding indices must be exactly 0..n: any duplicate or gap (a
    // fat-fingered binding number) breaks this.
    fn assert_gap_free_and_unique(bindings: &[Binding]) {
        let mut idx: Vec<u32> = bindings.iter().map(|b| b.0).collect();
        idx.sort_unstable();
        for (expected, &got) in idx.iter().enumerate() {
            assert_eq!(
                got, expected as u32,
                "descriptor bindings must be 0..n gap-free and unique, got {idx:?}"
            );
        }
    }

    #[test]
    fn geometry_path_sets_are_gap_free() {
        assert_gap_free_and_unique(&global_set());
        assert_gap_free_and_unique(&object_set());
        assert_gap_free_and_unique(&shadow_global_set());
    }

    // Golden lock: an independent copy of the binding -> (type, stage) contract.
    // Editing `global_set()` without updating this (and the matching shader
    // `layout(...)` qualifiers) is a deliberate review gate, not a silent change.
    #[test]
    fn global_set_contract_is_locked() {
        use vk::DescriptorType as T;
        use vk::ShaderStageFlags as S;
        assert_eq!(
            global_set(),
            [
                (0, T::UNIFORM_BUFFER, S::VERTEX | S::FRAGMENT),
                (1, T::UNIFORM_BUFFER, S::FRAGMENT),
                (2, T::UNIFORM_BUFFER, S::VERTEX | S::FRAGMENT),
                (3, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
                (4, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
                (5, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
                (6, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
                (7, T::UNIFORM_BUFFER, S::FRAGMENT),
            ]
        );
    }

    // The probe cube array binding sits exactly one past the count-1 global-set
    // table, so the layout (count-1 bindings 0..n + the count-MAX_PROBES array
    // binding) stays gap-free. A reorder that collides it with the table fails here.
    #[test]
    fn probe_cube_array_binding_follows_global_set() {
        assert_eq!(PROBE_CUBE_ARRAY_BINDING, global_set().len() as u32);
    }

    // The local-light SSBO binding sits exactly one past the probe cube array, so
    // the two inline-appended bindings (probe cubes at 8, local lights at 9) stay
    // gap-free above the count-1 global-set table. A reorder that collides them
    // fails here.
    #[test]
    fn local_light_ssbo_binding_follows_probe_cube_array() {
        assert_eq!(LOCAL_LIGHT_SSBO_BINDING, PROBE_CUBE_ARRAY_BINDING + 1);
    }

    // The clustered-lighting bindings continue the same gap-free run above the
    // count-1 global-set table: probe cubes 8, local lights 9, ClusterParams 10,
    // cluster lists 11. A reorder that collides them fails here.
    #[test]
    fn cluster_bindings_follow_local_light_ssbo() {
        assert_eq!(CLUSTER_PARAMS_UBO_BINDING, LOCAL_LIGHT_SSBO_BINDING + 1);
        assert_eq!(
            CLUSTER_LIGHT_LIST_SSBO_BINDING,
            CLUSTER_PARAMS_UBO_BINDING + 1
        );
    }

    // The spot shadow bindings continue the same gap-free run: depth array 12,
    // per-slice projections 13.
    #[test]
    fn spot_shadow_bindings_follow_cluster_light_list() {
        assert_eq!(SPOT_SHADOW_MAP_BINDING, CLUSTER_LIGHT_LIST_SSBO_BINDING + 1);
        assert_eq!(SPOT_SHADOW_DATA_SSBO_BINDING, SPOT_SHADOW_MAP_BINDING + 1);
    }

    // The area-light bindings close the run: table 14, LTC matrix 15, LTC
    // magnitude 16.
    #[test]
    fn area_light_bindings_follow_spot_shadow() {
        assert_eq!(AREA_LIGHT_SSBO_BINDING, SPOT_SHADOW_DATA_SSBO_BINDING + 1);
        assert_eq!(LTC_MATRIX_BINDING, AREA_LIGHT_SSBO_BINDING + 1);
        assert_eq!(LTC_MAGNITUDE_BINDING, LTC_MATRIX_BINDING + 1);
    }

    // The nine count-1 fragment samplers the probe array is budgeted against:
    // global set 0's shadow cascades / irradiance / prefilter / SSAO taps (3-6),
    // the spot shadow array (12), the two LTC tables (15/16), and per-object set
    // 1's albedo + normal maps.
    #[test]
    fn fixed_fragment_sampler_count_is_nine() {
        assert_eq!(fixed_fragment_samplers(), 9);
        assert_eq!(global_fragment_samplers(), 7);
        assert_eq!(object_fragment_samplers(), 2);
    }

    // A driver with room to spare gets the full CPU-side ceiling, so every
    // desktop Vulkan device keeps binding `MAX_PROBES` probes exactly as before.
    #[test]
    fn probe_cube_array_count_is_max_probes_on_desktop_drivers() {
        for limit in [1_048_576, 500_000, 1_048_575, MAX_PROBES as u32 + 9] {
            assert_eq!(probe_cube_array_count(limit), MAX_PROBES as u32);
        }
    }

    // MoltenVK reports 16, which leaves 7 after the nine fixed samplers: one
    // short of `MAX_PROBES`, and the reason the count is a runtime value at all.
    #[test]
    fn probe_cube_array_count_fits_moltenvk_limit() {
        let count = probe_cube_array_count(16);
        assert_eq!(count, 7);
        assert_eq!(fixed_fragment_samplers() + count, 16);
    }

    // Any reported limit yields a declarable array (a zero-length GLSL sampler
    // array will not compile) that never exceeds the ceiling, and stays inside
    // the reported budget wherever the fixed samplers leave room at all.
    #[test]
    fn probe_cube_array_count_never_zero_or_over_ceiling() {
        let fixed = fixed_fragment_samplers();
        for limit in 0..=64u32 {
            let count = probe_cube_array_count(limit);
            assert!(count >= 1, "limit {limit} produced a zero-length array");
            assert!(count <= MAX_PROBES as u32, "limit {limit} exceeded ceiling");
            if limit > fixed {
                assert!(
                    fixed + count <= limit,
                    "limit {limit} produced {count}, overrunning the budget"
                );
            }
        }
    }

    // The bindless layout drops per-object set 1 for the texture pool, so it
    // budgets against the global samplers only. On MoltenVK's 16 that leaves
    // 16 - 7 - 7 = 2 pool entries, the one-texture world; anything larger needs
    // update-after-bind. This is the overflow the runtime decision exists for.
    #[test]
    fn bindless_pool_needs_update_after_bind_past_moltenvk_headroom() {
        let probes = probe_cube_array_count(16);
        assert!(!bindless_pool_needs_update_after_bind(16, probes, 2));
        assert!(bindless_pool_needs_update_after_bind(16, probes, 3));
    }

    // A driver with room to spare keeps the plain layout for any pool a world
    // can realistically declare, so desktop never changes descriptor path.
    #[test]
    fn bindless_pool_stays_plain_on_desktop_drivers() {
        let probes = probe_cube_array_count(1_048_576);
        for pool_size in [2, 64, 4096, 65_536] {
            assert!(!bindless_pool_needs_update_after_bind(
                1_048_576, probes, pool_size
            ));
        }
    }

    // The device-creation gate and the per-layout decision must agree about who
    // is starved: every device that can overflow on a plausible pool has to have
    // had the update-after-bind features enabled for it.
    #[test]
    fn constrained_budget_covers_every_device_that_can_overflow() {
        for limit in 0..=64u32 {
            let probes = probe_cube_array_count(limit);
            if bindless_pool_needs_update_after_bind(limit, probes, 2) {
                assert!(
                    sampler_budget_is_constrained(limit),
                    "limit {limit} overflows on a one-texture world but reads unconstrained"
                );
            }
        }
        assert!(sampler_budget_is_constrained(16));
        assert!(!sampler_budget_is_constrained(1_048_576));
    }

    #[test]
    fn object_and_shadow_sets_contract_is_locked() {
        use vk::DescriptorType as T;
        use vk::ShaderStageFlags as S;
        assert_eq!(
            object_set(),
            [
                (0, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
                (1, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
            ]
        );
        assert_eq!(shadow_global_set(), [(0, T::UNIFORM_BUFFER, S::VERTEX)]);
    }
}
