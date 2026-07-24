// src/rt_geom.rs
//
// GPU-free builders for the ray-tracing geometry table plus the dynamic-update
// mode ladder, shared by the backends that hardware-ray-trace reflections. Each
// backend fills its own `RtGeomEntry` table from the participating draw set;
// the per-entry packing (index slice, resolved shared-pool texture indices,
// material, model matrix) is identical across backends and lives here. The
// per-backend TLAS instance transform (`MTLPackedFloat4x3` / `VkTransformMatrixKHR`
// / DXR `[f32; 12]`) is a real hardware type and stays in each backend.

use crate::render_types::{
    DrawObject, InstancedCluster, RtGeomEntry, SkinnedDrawObject, albedo_pool_index,
    normal_pool_index,
};

// Marks a `RtGeomEntry.normal_index` as belonging to a skinned object: the
// reflection trace then fetches the hit triangle from the deformed-vertex / u16
// skinned index buffers instead of the static u32 ones. Bit 31 is free (bindless
// pool indices never approach 2^31); matches the flag in each backend's RT-hit
// shader.
pub const RT_SKINNED_FLAG: u32 = 0x8000_0000;

// How the scene acceleration structure is kept current when props move. Selected
// once at init from `CN_RT_DYNAMIC`; unset gives `Auto`, the shipping behaviour.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RtDynamicMode {
    // Build once, never update. Forces a static BVH even if props move: the
    // pre-dynamic behaviour, kept as a fast path / diagnostic (`off`).
    Off,
    // Default. Rebuild the TLAS + table (fresh allocations, static BLAS) only on
    // the frames a participating transform actually changed. Static scenes never
    // rebuild, so they pay only a cheap per-frame matrix compare.
    Auto,
    // Force a full BVH rebuild every frame, dirty or not. Diagnostic (`rebuild`);
    // the most expensive path.
    Rebuild,
    // Force a fresh TLAS + table rebuild every frame, dirty or not. Diagnostic
    // (`tlas`); the same GPU work `Auto` does, minus the dirty gate.
    Tlas,
}

impl RtDynamicMode {
    // Parse the mode from `CN_RT_DYNAMIC`. Unset / unrecognised -> `Auto`.
    pub fn from_env() -> Self {
        match std::env::var("CN_RT_DYNAMIC").as_deref() {
            Ok("off") => Self::Off,
            Ok("rebuild") => Self::Rebuild,
            Ok("tlas") => Self::Tlas,
            _ => Self::Auto,
        }
    }

    // Whether this mode updates the BVH after the initial build at all.
    pub fn is_dynamic(self) -> bool {
        self != Self::Off
    }
}

// Shared-pool (albedo, normal) indices for a draw whose authored albedo /
// normal-map slots are `texture_slot` / `normal_map_slot`. Albedo and normal
// maps share one handle-indexed pool, so albedo = `texture_slot` and normal =
// the normal map's own handle (or the flat-normal fallback slot when the draw
// has none), resolved through the shared `render_types` helpers and matching the
// bindless main pass. `texture_count` is the real-texture count (the flat-normal
// fallback sits at `texture_count`).
pub fn pool_indices(texture_slot: usize, normal_map_slot: usize, texture_count: u32) -> (u32, u32) {
    (
        albedo_pool_index(texture_slot, texture_count),
        normal_pool_index(normal_map_slot, texture_count),
    )
}

// Build the geometry-table entry for one static draw object.
pub fn geom_entry(obj: &DrawObject, texture_count: u32) -> RtGeomEntry {
    let (albedo_index, normal_index) =
        pool_indices(obj.texture_slot, obj.normal_map_slot, texture_count);
    RtGeomEntry {
        index_offset: obj.index_offset as u32,
        base_vertex: obj.base_vertex as u32,
        albedo_index,
        normal_index,
        tint: obj.material.tint,
        roughness: obj.material.roughness,
        metallic: obj.material.metallic,
        emissive: obj.material.emissive,
        model: obj.model,
        emissive_map_index: obj.material.emissive_map_index,
        _pad: [0; 3],
    }
}

// Build the geometry-table entry for one instance of an instanced cluster: the
// cluster's shared mesh slice + material, with this instance's transform. Cluster
// geometry uses base_vertex 0 (its indices are already absolute).
pub fn cluster_geom_entry(
    cluster: &InstancedCluster,
    model: [[f32; 4]; 4],
    texture_count: u32,
) -> RtGeomEntry {
    let (albedo_index, normal_index) =
        pool_indices(cluster.texture_slot, cluster.normal_map_slot, texture_count);
    RtGeomEntry {
        index_offset: cluster.index_offset as u32,
        base_vertex: 0,
        albedo_index,
        normal_index,
        tint: cluster.material.tint,
        roughness: cluster.material.roughness,
        metallic: cluster.material.metallic,
        emissive: cluster.material.emissive,
        model,
        emissive_map_index: cluster.material.emissive_map_index,
        _pad: [0; 3],
    }
}

// Build the geometry-table entry for one skinned object. The skinned BLAS is
// baked from the posed (model-space) deformed buffer with absolute u16 indices,
// so `base_vertex` is 0 and the model matrix brings the hit to world space. The
// skinned flag is OR'd into `normal_index` so the trace fetches from the
// deformed / u16 buffers. Albedo / normal resolve through the shared pool by the
// object's `texture_slot` / `normal_map_slot`, so skinned hits shade textured
// like static ones (the flag bit lives above any valid pool index).
pub fn skinned_geom_entry(obj: &SkinnedDrawObject, texture_count: u32) -> RtGeomEntry {
    let (albedo_index, normal_index) =
        pool_indices(obj.texture_slot, obj.normal_map_slot, texture_count);
    RtGeomEntry {
        index_offset: obj.index_offset as u32,
        base_vertex: 0,
        albedo_index,
        normal_index: normal_index | RT_SKINNED_FLAG,
        tint: obj.material.tint,
        roughness: obj.material.roughness,
        metallic: obj.material.metallic,
        emissive: obj.material.emissive,
        model: obj.model,
        emissive_map_index: obj.material.emissive_map_index,
        _pad: [0; 3],
    }
}

// True when any participating object's current model matrix differs from the one
// baked into the live TLAS. Pure (no GPU) so the dirty gate is unit-testable.
pub fn models_dirty(cached: &[[[f32; 4]; 4]], current: &[[[f32; 4]; 4]]) -> bool {
    cached.len() != current.len() || cached.iter().zip(current).any(|(a, b)| a != b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_mode_from_env_default_is_auto() {
        // The env var isn't set in the test process, so it resolves to Auto.
        assert_eq!(RtDynamicMode::from_env(), RtDynamicMode::Auto);
        assert!(RtDynamicMode::Auto.is_dynamic());
        assert!(RtDynamicMode::Rebuild.is_dynamic());
        assert!(RtDynamicMode::Tlas.is_dynamic());
        assert!(!RtDynamicMode::Off.is_dynamic());
    }

    #[test]
    fn pool_indices_share_one_handle_indexed_pool() {
        use crate::render_types::NO_NORMAL_MAP_SLOT;
        // Albedo and a real normal map both index the shared pool by their own
        // handle. 5 real textures; the flat-normal fallback sits at slot 5.
        assert_eq!(pool_indices(2, 1, 5), (2, 1));
        // Out-of-range real slots clamp to the last real texture (4).
        assert_eq!(pool_indices(9, 9, 5), (4, 4));
        // A draw with no normal map addresses the flat-normal fallback slot.
        assert_eq!(pool_indices(2, NO_NORMAL_MAP_SLOT, 5), (2, 5));
    }

    #[test]
    fn models_dirty_detects_a_changed_transform() {
        let a = [[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]];
        let mut b = a;
        assert!(!models_dirty(&a, &b));
        b[0][3][0] = 5.0;
        assert!(models_dirty(&a, &b));
        // A length change is dirty.
        assert!(models_dirty(&a, &[]));
    }

    #[test]
    fn skinned_flag_is_bit_31_and_masks_back_to_the_pool_index() {
        // The flag occupies the top bit; the shader recovers the real bindless
        // normal index with `normal_index & ~RT_SKINNED_FLAG`. Mirror both here.
        assert_eq!(RT_SKINNED_FLAG, 1u32 << 31);
        for normal_index in [0u32, 1, 5, 96, 1000] {
            let flagged = normal_index | RT_SKINNED_FLAG;
            assert_ne!(flagged & RT_SKINNED_FLAG, 0, "flag set");
            assert_eq!(flagged & !RT_SKINNED_FLAG, normal_index, "masks back");
        }
        // Realistic bindless pool indices never reach the flag bit, so a static
        // entry's normal index is never misread as skinned.
        assert_eq!(96u32 & RT_SKINNED_FLAG, 0);
    }

    #[test]
    fn skinned_geom_entry_flags_and_zeroes_base_vertex() {
        use crate::render_types::{MaterialUniforms, SkinnedDrawObject};
        let material = MaterialUniforms {
            tint: [0.2, 0.4, 0.6],
            roughness: 0.3,
            metallic: 0.5,
            emissive: [0.1, 0.0, 0.0],
            ..MaterialUniforms::DEFAULT
        };
        let obj = SkinnedDrawObject {
            vertex_base: 7,
            vertex_count: 100,
            index_offset: 42,
            index_count: 300,
            model: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [3.0, 4.0, 5.0, 1.0],
            ],
            texture_slot: 9,
            normal_map_slot: 3,
            material,
            visible: true,
            joint_count: 12,
            local_bb_min: [-1.0, -1.0, -1.0],
            local_bb_max: [1.0, 1.0, 1.0],
            lod_alternates: Vec::new(),
        };
        let texture_count = 12u32;
        let e = skinned_geom_entry(&obj, texture_count);
        // The skinned BLAS bakes absolute indices, so base_vertex is folded to 0.
        assert_eq!(e.base_vertex, 0);
        // The skinned flag is set; masking it off recovers the real shared-pool
        // index, computed the same way as a static draw (so skinned hits texture).
        assert_ne!(e.normal_index & RT_SKINNED_FLAG, 0);
        let (exp_albedo, exp_normal) =
            pool_indices(obj.texture_slot, obj.normal_map_slot, texture_count);
        assert_eq!(e.albedo_index, exp_albedo);
        assert_eq!(e.normal_index & !RT_SKINNED_FLAG, exp_normal);
        // Material + index offset carry through; the model lifts the hit to world.
        assert_eq!(e.index_offset, 42);
        assert_eq!(e.tint, [0.2, 0.4, 0.6]);
        assert_eq!(e.model[3], [3.0, 4.0, 5.0, 1.0]);
    }
}
