//! Canonical descriptor-set binding tables for the geometry render path, kept
//! in one place so the `[[vk::binding(M, N)]]` indices the HLSL shaders declare
//! stay greppable and locked. `init/descriptors.rs` builds the real
//! `OwnedSetLayout`s from these via `create_descriptor_set_layout`, every pool
//! that allocates the global set sizes itself from the table, and
//! `global_set.rs` writes the same binding numbers. The unit tests
//! below assert each table is gap-free + unique and pin the binding -> (type,
//! stage) contract, so a reordering, retype, or stage-flag change that would
//! silently desync from the shaders fails `cargo test` instead of reading
//! garbage on the GPU. Vulkan analogue of `directx/init/heap_layout.rs`'s slot
//! tests.
//!
//! Only the geometry-path sets (global / shadow) are centralized here; the
//! post-process sets (composite, bloom, text) are simpler layouts declared
//! where they are built.

use ash::vk;
use concinnity_core::gfx::render_types::FALLBACK_TEXTURE_COUNT;
use concinnity_core::render::uniforms::BINDLESS_POOL_SIZE;

// One descriptor binding: (binding index, descriptor type, shader stages).
pub(in crate::vulkan) type Binding = (u32, vk::DescriptorType, vk::ShaderStageFlags);

// Global set 0 bindings: the geometry path's set, which the transparent pass and
// the reflection resolves bind as well. Every texture is a sampled image read
// through one of the set's three engine samplers, as the DirectX root
// signature's static samplers are read.
pub(in crate::vulkan) const VIEW_UBO_BINDING: u32 = 0;
pub(in crate::vulkan) const LIGHT_UBO_BINDING: u32 = 1;
pub(in crate::vulkan) const SHADOW_UBO_BINDING: u32 = 2;
// The directional shadow cascade array.
pub(in crate::vulkan) const SHADOW_MAP_BINDING: u32 = 3;
pub(in crate::vulkan) const IRRADIANCE_CUBE_BINDING: u32 = 4;
pub(in crate::vulkan) const PREFILTER_CUBE_BINDING: u32 = 5;
// This frame's SSAO occlusion, or a 1x1 white image when nothing computes one.
pub(in crate::vulkan) const SSAO_BINDING: u32 = 6;
// The probe count, the cube array holding a cube per probe, and one parallax
// record per probe.
pub(in crate::vulkan) const PROBE_SET_UBO_BINDING: u32 = 7;
pub(in crate::vulkan) const PROBE_CUBES_BINDING: u32 = 8;
pub(in crate::vulkan) const LOCAL_LIGHT_SSBO_BINDING: u32 = 9;
// The clustered-lighting grid params and the per-cluster light and probe lists
// the `LightCull` pass writes.
pub(in crate::vulkan) const CLUSTER_PARAMS_UBO_BINDING: u32 = 10;
pub(in crate::vulkan) const CLUSTER_LIST_SSBO_BINDING: u32 = 11;
// The spot shadow depth array, one layer per shadowed spot, and each layer's
// light-space projection.
pub(in crate::vulkan) const SPOT_SHADOW_MAP_BINDING: u32 = 12;
pub(in crate::vulkan) const SPOT_SHADOW_DATA_SSBO_BINDING: u32 = 13;
// The area-light table and its two LTC lookups: the inverse transform and the
// magnitude / Fresnel pair.
pub(in crate::vulkan) const AREA_LIGHT_SSBO_BINDING: u32 = 14;
pub(in crate::vulkan) const LTC_MATRIX_BINDING: u32 = 15;
pub(in crate::vulkan) const LTC_MAGNITUDE_BINDING: u32 = 16;
pub(in crate::vulkan) const PROBE_RECORDS_SSBO_BINDING: u32 = 17;
// The depth-compare state both shadow arrays read, the trilinear clamp state the
// cubes, the SSAO occlusion and the LTC tables read, and the linear repeat state
// the bindless texture pool reads.
pub(in crate::vulkan) const SHADOW_SAMPLER_BINDING: u32 = 18;
pub(in crate::vulkan) const CUBE_SAMPLER_BINDING: u32 = 19;
pub(in crate::vulkan) const LINEAR_SAMPLER_BINDING: u32 = 20;

// Every binding of global set 0, each count 1. The layout, the per-stage budget
// and every pool that allocates the set are built from this table.
pub(in crate::vulkan) fn global_set() -> [Binding; 21] {
    use vk::DescriptorType as T;
    let fs = vk::ShaderStageFlags::FRAGMENT;
    let vs_fs = vk::ShaderStageFlags::VERTEX | fs;
    [
        (VIEW_UBO_BINDING, T::UNIFORM_BUFFER, vs_fs),
        (LIGHT_UBO_BINDING, T::UNIFORM_BUFFER, fs),
        (SHADOW_UBO_BINDING, T::UNIFORM_BUFFER, vs_fs),
        (SHADOW_MAP_BINDING, T::SAMPLED_IMAGE, fs),
        (IRRADIANCE_CUBE_BINDING, T::SAMPLED_IMAGE, fs),
        (PREFILTER_CUBE_BINDING, T::SAMPLED_IMAGE, fs),
        (SSAO_BINDING, T::SAMPLED_IMAGE, fs),
        (PROBE_SET_UBO_BINDING, T::UNIFORM_BUFFER, fs),
        (PROBE_CUBES_BINDING, T::SAMPLED_IMAGE, fs),
        (LOCAL_LIGHT_SSBO_BINDING, T::STORAGE_BUFFER, fs),
        (CLUSTER_PARAMS_UBO_BINDING, T::UNIFORM_BUFFER, fs),
        (CLUSTER_LIST_SSBO_BINDING, T::STORAGE_BUFFER, fs),
        (SPOT_SHADOW_MAP_BINDING, T::SAMPLED_IMAGE, fs),
        (SPOT_SHADOW_DATA_SSBO_BINDING, T::STORAGE_BUFFER, fs),
        (AREA_LIGHT_SSBO_BINDING, T::STORAGE_BUFFER, fs),
        (LTC_MATRIX_BINDING, T::SAMPLED_IMAGE, fs),
        (LTC_MAGNITUDE_BINDING, T::SAMPLED_IMAGE, fs),
        (PROBE_RECORDS_SSBO_BINDING, T::STORAGE_BUFFER, fs),
        (SHADOW_SAMPLER_BINDING, T::SAMPLER, fs),
        (CUBE_SAMPLER_BINDING, T::SAMPLER, fs),
        (LINEAR_SAMPLER_BINDING, T::SAMPLER, fs),
    ]
}

// Descriptor counts per type for a pool, accumulated from the set layouts it
// allocates and any extra descriptors its callers add by hand.
#[derive(Default)]
pub(in crate::vulkan) struct PoolSizes(Vec<vk::DescriptorPoolSize>);

impl PoolSizes {
    // Room for `sets` sets of `table`.
    pub(in crate::vulkan) fn sets(mut self, table: &[Binding], sets: u32) -> Self {
        for &(_, ty, _) in table {
            self = self.add(ty, sets);
        }
        self
    }

    // Room for `count` more descriptors of type `ty`.
    pub(in crate::vulkan) fn add(mut self, ty: vk::DescriptorType, count: u32) -> Self {
        match self.0.iter_mut().find(|s| s.ty == ty) {
            Some(size) => size.descriptor_count += count,
            None => self.0.push(vk::DescriptorPoolSize {
                ty,
                descriptor_count: count,
            }),
        }
        self
    }

    // The pool sizes, leaving out every type with no descriptor: a zero
    // `descriptorCount` is invalid.
    pub(in crate::vulkan) fn build(self) -> Vec<vk::DescriptorPoolSize> {
        self.0
            .into_iter()
            .filter(|s| s.descriptor_count > 0)
            .collect()
    }
}

// The descriptors one shader stage of a pipeline layout declares, in the two
// classes a per-stage limit counts separately: `maxPerStageDescriptorSamplers`
// and `maxPerStageDescriptorSampledImages`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::vulkan) struct StageDescriptors {
    pub(in crate::vulkan) samplers: u32,
    pub(in crate::vulkan) sampled_images: u32,
}

impl StageDescriptors {
    fn plus(self, other: Self) -> Self {
        Self {
            samplers: self.samplers + other.samplers,
            sampled_images: self.sampled_images + other.sampled_images,
        }
    }

    // Whether a plain pipeline layout declaring this much exceeds `limits` in
    // either class.
    fn exceeds(self, limits: StageDescriptors) -> bool {
        self.samplers > limits.samplers || self.sampled_images > limits.sampled_images
    }
}

// The device's plain per-stage budget in both classes.
pub(in crate::vulkan) fn stage_limits(limits: &vk::PhysicalDeviceLimits) -> StageDescriptors {
    StageDescriptors {
        samplers: limits.max_per_stage_descriptor_samplers,
        sampled_images: limits.max_per_stage_descriptor_sampled_images,
    }
}

// Fragment descriptors global set 0 declares, which every pipeline layout that
// binds the set pays.
fn global_fragment_descriptors() -> StageDescriptors {
    let count = |ty| {
        global_set()
            .iter()
            .filter(|&&(_, t, stage)| t == ty && stage.contains(vk::ShaderStageFlags::FRAGMENT))
            .count() as u32
    };
    StageDescriptors {
        samplers: count(vk::DescriptorType::SAMPLER),
        sampled_images: count(vk::DescriptorType::SAMPLED_IMAGE),
    }
}

// Slots a world with `texture_count` table entries needs: one image per slot (a
// single fallback when the table is empty) plus the reserved fallbacks,
// flat-normal and white. The pool's descriptor count, and the only length it
// has: the shaders declare the array unsized and read whatever the set layout
// was built with, so this never reaches the source text.
pub(in crate::vulkan) fn world_pool_size(texture_count: usize) -> usize {
    texture_count.max(1) + FALLBACK_TEXTURE_COUNT
}

// The texture pool's per-stage cost: one sampled image per slot. It reads
// through the global set's linear sampler, so it declares no sampler of its own.
fn pool_descriptors(pool_size: u32) -> StageDescriptors {
    StageDescriptors {
        samplers: 0,
        sampled_images: pool_size,
    }
}

// Whether the bindless main pipeline layout must declare its texture pool
// `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT`. The layout is
// global set 0 + the bindless set, so its plain per-stage cost is the global
// set's plus the pool itself; the pool cannot be clamped, since its length is
// the world's texture table. Descriptors in an update-after-bind set layout are
// budgeted against `maxPerStageDescriptorUpdateAfterBindSampledImages` instead,
// which MoltenVK reports as a million against a plain limit of 256. Desktop
// drivers report six figures for both and never take the update-after-bind
// path.
pub(in crate::vulkan) fn bindless_pool_needs_update_after_bind(
    limits: StageDescriptors,
    pool_size: u32,
) -> bool {
    global_fragment_descriptors()
        .plus(pool_descriptors(pool_size))
        .exceeds(limits)
}

// Whether a full texture pool can overflow a device's plain per-stage budget,
// which is what device creation enables the descriptor-indexing
// update-after-bind features for. True on MoltenVK, whose 256 sampled images
// cannot seat a full pool, and false on every desktop driver, which leaves
// their device-creation feature chain and descriptor path untouched.
pub(in crate::vulkan) fn update_after_bind_is_wanted(limits: StageDescriptors) -> bool {
    bindless_pool_needs_update_after_bind(limits, BINDLESS_POOL_SIZE as u32)
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

    // MoltenVK's plain per-stage budget on Apple silicon.
    const MOLTENVK: StageDescriptors = StageDescriptors {
        samplers: 16,
        sampled_images: 256,
    };

    // A desktop driver's, six figures in both classes.
    const DESKTOP: StageDescriptors = StageDescriptors {
        samplers: 1_048_576,
        sampled_images: 1_048_576,
    };

    #[test]
    fn the_shadow_global_set_is_gap_free() {
        assert_gap_free_and_unique(&shadow_global_set());
    }

    // Golden lock: an independent copy of the binding -> (type, stage) contract
    // the shaders' `[[vk::binding(n, set)]]`s declare, gap-free from 0. Editing
    // the table without updating this (and the shaders) is a deliberate review
    // gate, not a silent change.
    #[test]
    fn the_global_set_contract_is_locked_and_gap_free() {
        use vk::DescriptorType as T;
        use vk::ShaderStageFlags as S;
        let fs = S::FRAGMENT;
        let vs_fs = S::VERTEX | S::FRAGMENT;
        let table = global_set();
        assert_gap_free_and_unique(&table);
        assert_eq!(
            table,
            [
                (0, T::UNIFORM_BUFFER, vs_fs),
                (1, T::UNIFORM_BUFFER, fs),
                (2, T::UNIFORM_BUFFER, vs_fs),
                (3, T::SAMPLED_IMAGE, fs),
                (4, T::SAMPLED_IMAGE, fs),
                (5, T::SAMPLED_IMAGE, fs),
                (6, T::SAMPLED_IMAGE, fs),
                (7, T::UNIFORM_BUFFER, fs),
                (8, T::SAMPLED_IMAGE, fs),
                (9, T::STORAGE_BUFFER, fs),
                (10, T::UNIFORM_BUFFER, fs),
                (11, T::STORAGE_BUFFER, fs),
                (12, T::SAMPLED_IMAGE, fs),
                (13, T::STORAGE_BUFFER, fs),
                (14, T::STORAGE_BUFFER, fs),
                (15, T::SAMPLED_IMAGE, fs),
                (16, T::SAMPLED_IMAGE, fs),
                (17, T::STORAGE_BUFFER, fs),
                (18, T::SAMPLER, fs),
                (19, T::SAMPLER, fs),
                (20, T::SAMPLER, fs),
            ]
        );
    }

    fn count_of(sizes: &[vk::DescriptorPoolSize], ty: vk::DescriptorType) -> u32 {
        sizes
            .iter()
            .filter(|s| s.ty == ty)
            .map(|s| s.descriptor_count)
            .sum()
    }

    // One global set needs five uniform buffers, five storage buffers, eight
    // images and three samplers; `n` sets need `n` times that.
    #[test]
    fn pool_sizes_scale_the_table_by_the_set_count() {
        use vk::DescriptorType as T;
        let sizes = PoolSizes::default().sets(&global_set(), 3).build();
        assert_eq!(count_of(&sizes, T::UNIFORM_BUFFER), 15);
        assert_eq!(count_of(&sizes, T::STORAGE_BUFFER), 15);
        assert_eq!(count_of(&sizes, T::SAMPLED_IMAGE), 24);
        assert_eq!(count_of(&sizes, T::SAMPLER), 9);
        assert_eq!(sizes.len(), 4, "one entry per type");
    }

    // Hand-added counts merge into the table's entries, and a type with no
    // descriptor is left out rather than declared with a zero count.
    #[test]
    fn pool_sizes_merge_extras_and_drop_empty_types() {
        use vk::DescriptorType as T;
        let sizes = PoolSizes::default()
            .sets(&shadow_global_set(), 2)
            .add(T::UNIFORM_BUFFER, 1)
            .add(T::STORAGE_BUFFER, 0)
            .build();
        assert_eq!(
            sizes
                .iter()
                .map(|s| (s.ty, s.descriptor_count))
                .collect::<Vec<_>>(),
            [(T::UNIFORM_BUFFER, 3)]
        );
        assert!(
            PoolSizes::default()
                .sets(&global_set(), 0)
                .build()
                .is_empty()
        );
    }

    // Eight images -- shadow cascades / irradiance / prefilter / SSAO / probe
    // cubes (3-6, 8), the spot shadow array (12) and the two LTC tables (15/16)
    // -- read through three samplers. No probe count reaches either: the cube
    // array is one descriptor however many probes it holds.
    #[test]
    fn the_global_set_declares_eight_images_and_three_samplers() {
        assert_eq!(
            global_fragment_descriptors(),
            StageDescriptors {
                samplers: 3,
                sampled_images: 8,
            }
        );
    }

    // The spec's minimum `maxPerStageDescriptorSamplers` and
    // `maxPerStageDescriptorSampledImages`.
    const SPEC_MINIMUM: StageDescriptors = StageDescriptors {
        samplers: 16,
        sampled_images: 16,
    };

    // Fragment descriptors the transparent pipeline layouts declare outside the
    // global set: the scene snapshot, the main depth and the two reduced
    // reflection layers on the view set with the snapshot's sampler, and one
    // record's planar reflection with its sampler on the params set; the sky
    // cube is the global set's. Mirrors `transparent.rs`'s
    // `create_view_set_layout` + `create_params_set_layout`, which every flat
    // and RT layout shares.
    const TRANSPARENT_PASS: StageDescriptors = StageDescriptors {
        samplers: 2,
        sampled_images: 5,
    };

    // Fragment descriptors the SSR resolve pipeline layout declares outside the
    // global set: scene, G-buffer, roughness and the prefilter cube it falls
    // back to, each with a sampler of its own. Mirrors the resolve set in
    // `post/ssr.rs`; the RT resolve declares the same screen inputs but reads
    // the cube from the global set, so it is narrower.
    const REFLECTION_RESOLVE: StageDescriptors = StageDescriptors {
        samplers: 4,
        sampled_images: 4,
    };

    // Global set 0 is bound by the geometry path, the transparent pass and the
    // reflection resolves alike, so every one of those layouts pays for it.
    // Beside the widest of them, the texture pool aside, it fits the budget
    // every conforming device reports, so the set never needs
    // update-after-bind.
    #[test]
    fn the_global_set_and_its_widest_pass_fit_the_spec_minimum() {
        let widest = StageDescriptors {
            samplers: TRANSPARENT_PASS.samplers.max(REFLECTION_RESOLVE.samplers),
            sampled_images: TRANSPARENT_PASS
                .sampled_images
                .max(REFLECTION_RESOLVE.sampled_images),
        };
        let seat = global_fragment_descriptors().plus(widest);
        assert_eq!(
            seat,
            StageDescriptors {
                samplers: 7,
                sampled_images: 13,
            }
        );
        assert!(!seat.exceeds(SPEC_MINIMUM));
    }

    // Beside a plain global set on MoltenVK the pool has 256 - 8 images; one
    // more needs update-after-bind. The pool declares no sampler, so the
    // sampler budget never decides it.
    #[test]
    fn the_pool_overflows_on_sampled_images_alone() {
        assert!(!bindless_pool_needs_update_after_bind(MOLTENVK, 248));
        assert!(bindless_pool_needs_update_after_bind(MOLTENVK, 249));
        let few_samplers = StageDescriptors {
            samplers: 3,
            sampled_images: 1_048_576,
        };
        assert!(!bindless_pool_needs_update_after_bind(few_samplers, 65_536));
    }

    // A driver with room to spare keeps the plain layout for any pool a world
    // can realistically declare, so desktop never changes descriptor path.
    #[test]
    fn the_pool_stays_plain_on_desktop_drivers() {
        for pool_size in [2, 64, 4096, 65_536] {
            assert!(!bindless_pool_needs_update_after_bind(DESKTOP, pool_size));
        }
        assert!(!update_after_bind_is_wanted(DESKTOP));
    }

    // MoltenVK still wants the features, for a pool larger than its 256 images.
    #[test]
    fn moltenvk_wants_update_after_bind_for_a_full_pool() {
        assert!(update_after_bind_is_wanted(MOLTENVK));
    }

    #[test]
    fn the_stage_limits_are_the_two_per_stage_counts() {
        let limits = vk::PhysicalDeviceLimits {
            max_per_stage_descriptor_samplers: 16,
            max_per_stage_descriptor_sampled_images: 256,
            ..Default::default()
        };
        assert_eq!(stage_limits(&limits), MOLTENVK);
    }

    #[test]
    fn shadow_set_contract_is_locked() {
        use vk::DescriptorType as T;
        use vk::ShaderStageFlags as S;
        assert_eq!(shadow_global_set(), [(0, T::UNIFORM_BUFFER, S::VERTEX)]);
    }

    #[test]
    fn pool_size_counts_fallbacks() {
        // One slot per table entry (an empty table still pads to one) plus the
        // two reserved fallbacks, flat-normal then white.
        assert_eq!(world_pool_size(0), 3);
        assert_eq!(world_pool_size(1), 3);
        assert_eq!(world_pool_size(7), 9);
    }

    // The uploaded image vectors reproduce the world-sized pool exactly: init
    // pads an empty texture table to one image and always uploads the reserved
    // fallbacks alongside it. A raw texture count is never a valid pool length,
    // so a compile handed one silently drops the last slots.
    #[test]
    fn world_pool_size_matches_the_uploaded_image_counts() {
        for texture_count in [0usize, 1, 7, 64] {
            let gpu_textures = texture_count.max(1);
            assert_eq!(
                world_pool_size(texture_count),
                gpu_textures + FALLBACK_TEXTURE_COUNT
            );
            assert!(world_pool_size(texture_count) > texture_count);
        }
    }
}
