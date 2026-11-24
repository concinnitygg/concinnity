//! Backend-agnostic helpers for the transparent (translucent) pass. The pass
//! itself is encoded per backend; this module owns the whole CPU-side ordering
//! policy so it can be unit-tested without a GPU and so the three backends
//! cannot disagree about draw order.
//!
//! Transparent fragments use `SRC_ALPHA` / `ONE_MINUS_SRC_ALPHA` blending, which is
//! order-dependent: a draw must be composited after everything behind it.
//! [`back_to_front`] orders the draw indices farthest-first by camera
//! distance so the blend resolves correctly. This is a single fixed
//! sorted draw list, not order-independent transparency.
//!
//! Vulkan and DirectX keep their records per producer ([`producer`]) and order
//! them each frame with [`TransparentFramePlan::order_draws`], which does the
//! filtering, the interleave and the sort in one step over reused buffers. Metal flattens all three producers into one draw list upstream, so
//! it computes each record's [`sort_distance`] as it builds that list and calls
//! [`back_to_front`] on the result. Different entry points, one policy.

use alloc::vec::Vec;

use crate::gfx::lod::camera_distance;
use crate::gfx::render_types::DrawObject;
use crate::render::uniforms::GlassMeshParams;

pub(crate) mod glass_panel;
mod plan;
pub mod producer;
pub use plan::TransparentFramePlan;

/// Which producer a transparent record belongs to, and so which pipeline draws
/// it. The records are identical in shape, so this is the only thing a combined
/// draw loop needs to tell them apart.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Producer {
    /// A `GlassPanel`: one static world-space quad.
    Glass,
    /// A `WaterSurface`: one origin-centered grid.
    Water,
    /// A `GlassMesh`: an arbitrary mesh drawn with the glass shading.
    GlassMesh,
}

/// World-space distance from the camera to a record center. Larger = farther =
/// drawn first.
pub fn sort_distance(center: [f32; 3], cam: [f32; 3]) -> f32 {
    let dx = center[0] - cam[0];
    let dy = center[1] - cam[1];
    let dz = center[2] - cam[2];
    concinnity_core::math::sqrt(dx * dx + dy * dy + dz * dz)
}

/// Screen-space refraction offset a see-through glass mesh draws with. A
/// `Material` carries no glass tunables, so this and
/// [`GLASS_MESH_FRESNEL_POWER`] match the `GlassPanel` defaults.
const GLASS_MESH_REFRACTION: f32 = 0.02;
/// Schlick-Fresnel exponent a see-through glass mesh draws with: a subtle
/// reflection head-on, a full mirror at grazing angles.
const GLASS_MESH_FRESNEL_POWER: f32 = 1.0;

/// One see-through glass mesh as this frame's transparent pass draws it: the
/// LOD slice the opaque passes would have picked at the same camera distance,
/// and the per-draw block its shading reads.
#[derive(Clone, Copy)]
pub struct SeeThroughMesh {
    /// The per-draw block the glass mesh shader reads.
    pub params: GlassMeshParams,
    /// First index of the chosen LOD slice in the shared index buffer.
    pub index_offset: usize,
    /// Index count of the chosen LOD slice.
    pub index_count: usize,
    /// The draw's base vertex.
    pub base_vertex: i32,
    /// World-space AABB center.
    pub center: [f32; 3],
    /// Camera distance the LOD was chosen at, which is also the sort key.
    pub distance: f32,
}

impl SeeThroughMesh {
    /// The draw of `obj` seen from `cam`, with `prefilter_mip_count` the bound
    /// IBL prefilter cube's mip count for the ray-miss fallback.
    pub fn new(obj: &DrawObject, cam: [f32; 3], prefilter_mip_count: f32) -> Self {
        let distance = camera_distance(obj, cam);
        let (index_offset, index_count) = obj.active_lod(distance);
        let t = obj.material.tint;
        Self {
            params: GlassMeshParams {
                model: obj.model,
                tint: [t[0], t[1], t[2], 0.0],
                opacity: obj.material.opacity,
                refraction_strength: GLASS_MESH_REFRACTION,
                fresnel_power: GLASS_MESH_FRESNEL_POWER,
                prefilter_mip_count,
            },
            index_offset,
            index_count,
            base_vertex: obj.base_vertex,
            center: [
                0.5 * (obj.bb_min[0] + obj.bb_max[0]),
                0.5 * (obj.bb_min[1] + obj.bb_max[1]),
                0.5 * (obj.bb_min[2] + obj.bb_max[2]),
            ],
            distance,
        }
    }
}

/// Back-to-front order between draw `a` and draw `b`, each a camera distance
/// and its position in the unsorted list: farther first, equal distances in
/// list order, and a non-finite distance nearest so a degenerate value never
/// pushes a draw behind valid geometry.
pub fn farther_first(a: (f32, usize), b: (f32, usize)) -> core::cmp::Ordering {
    let finite = |d: f32| if d.is_finite() { d } else { f32::NEG_INFINITY };
    finite(b.0).total_cmp(&finite(a.0)).then(a.1.cmp(&b.1))
}

/// Fill `order` with `0..count` ordered [`farther_first`] by camera
/// `distance`, reusing its storage.
pub fn back_to_front(order: &mut Vec<usize>, count: usize, distance: impl Fn(usize) -> f32) {
    order.clear();
    order.extend(0..count);
    order.sort_unstable_by(|&a, &b| farther_first((distance(a), a), (distance(b), b)));
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::gfx::render_types::LodSlice;
    use alloc::vec;

    #[test]
    fn a_see_through_mesh_draws_its_lod_slice_with_glass_shading() {
        let mut obj = crate::test_support::draw_object();
        obj.bb_min = [-1.0, -1.0, -1.0];
        obj.bb_max = [1.0, 1.0, 1.0];
        obj.material.tint = [0.2, 0.4, 0.6];
        obj.material.opacity = 0.3;
        obj.lod_alternates = vec![LodSlice {
            index_offset: 100,
            index_count: 6,
            switch_distance: 5.0,
        }];
        let near = SeeThroughMesh::new(&obj, [0.0, 0.0, 3.0], 9.0);
        assert_eq!((near.index_offset, near.index_count), (12, 36));
        assert_eq!(near.distance, 3.0);
        assert_eq!(near.center, [0.0; 3]);
        assert_eq!(near.base_vertex, obj.base_vertex);
        assert_eq!(near.params.model, obj.model);
        assert_eq!(near.params.tint, [0.2, 0.4, 0.6, 0.0]);
        assert_eq!(near.params.opacity, 0.3);
        assert_eq!(near.params.refraction_strength, GLASS_MESH_REFRACTION);
        assert_eq!(near.params.fresnel_power, GLASS_MESH_FRESNEL_POWER);
        assert_eq!(near.params.prefilter_mip_count, 9.0);
        let far = SeeThroughMesh::new(&obj, [0.0, 0.0, 8.0], 9.0);
        assert_eq!((far.index_offset, far.index_count), (100, 6));
    }

    fn ordered(d: &[f32]) -> Vec<usize> {
        let mut order = vec![9];
        back_to_front(&mut order, d.len(), |i| d[i]);
        order
    }

    #[test]
    fn orders_farthest_first() {
        assert_eq!(ordered(&[1.0, 5.0, 3.0]), vec![1, 2, 0]);
    }

    #[test]
    fn empty_is_empty() {
        assert!(ordered(&[]).is_empty());
    }

    #[test]
    fn equal_distances_keep_declaration_order() {
        let d: Vec<f32> = (0..40).map(|i| (i / 10) as f32).collect();
        let expected: Vec<usize> = (0..4).rev().flat_map(|b| b * 10..b * 10 + 10).collect();
        assert_eq!(ordered(&d), expected);
    }

    #[test]
    fn sort_distance_is_euclidean_and_monotone() {
        let cam = [0.0, 0.0, 0.0];
        let near = sort_distance([0.0, 0.0, 1.0], cam);
        let far = sort_distance([0.0, 0.0, 5.0], cam);
        assert!((near - 1.0).abs() < 1e-5);
        assert!((far - 5.0).abs() < 1e-5);
        assert!(far > near);
    }

    #[test]
    fn nan_sorts_last() {
        let d = [4.0, f32::NAN, 2.0];
        // 4.0 (farthest) → index 0, then 2.0 → index 2, then NaN → index 1.
        assert_eq!(ordered(&d), vec![0, 2, 1]);
    }
}
