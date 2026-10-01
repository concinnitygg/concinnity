//! Backend-agnostic helpers for the transparent (translucent) pass. The pass
//! itself is encoded per backend; this module owns the whole CPU-side ordering
//! policy so it can be unit-tested without a GPU and so the three backends
//! cannot disagree about draw order.
//!
//! Transparent fragments use SRC_ALPHA / ONE_MINUS_SRC_ALPHA blending, which is
//! order-dependent: a draw must be composited after everything behind it.
//! [`back_to_front_order`] returns the draw indices sorted farthest-first by
//! camera distance so the blend resolves correctly. This is a single fixed
//! sorted draw list, not order-independent transparency.
//!
//! Vulkan and DirectX keep their records per producer and call
//! [`ordered_visible`], which does the filtering, the interleave and the sort in
//! one step. Metal flattens all three producers into one draw list upstream, so
//! it computes each record's [`sort_distance`] as it builds that list and calls
//! [`back_to_front_order`] on the result. Different entry points, one policy.

use alloc::vec::Vec;

use crate::gfx::lod::camera_distance;
use crate::gfx::render_types::DrawObject;
use crate::render::uniforms::GlassMeshParams;

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
    crate::math::sqrt(dx * dx + dy * dy + dz * dz)
}

/// Screen-space refraction offset a see-through glass mesh draws with. A
/// `Material` carries no glass tunables, so this and
/// [`GLASS_MESH_FRESNEL_POWER`] match the `GlassPanel` defaults.
pub const GLASS_MESH_REFRACTION: f32 = 0.02;
/// Schlick-Fresnel exponent a see-through glass mesh draws with: a subtle
/// reflection head-on, a full mirror at grazing angles.
pub const GLASS_MESH_FRESNEL_POWER: f32 = 1.0;

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

/// Every visible record of every producer, ordered farthest-camera-distance
/// first. Invisible records are excluded, and the producers interleave so a
/// pane standing in a pool composites in the right order.
///
/// `glass` and `water` are `(center, visible)` per record. `meshes` is already
/// filtered to this frame's visible meshes by the caller, so every entry it
/// carries is live.
pub fn ordered_visible(
    glass: &[([f32; 3], bool)],
    water: &[([f32; 3], bool)],
    meshes: &[[f32; 3]],
    cam: [f32; 3],
) -> Vec<(Producer, usize)> {
    let live_of = |records: &[([f32; 3], bool)], kind: Producer| -> Vec<(Producer, usize)> {
        records
            .iter()
            .enumerate()
            .filter(|(_, (_, vis))| *vis)
            .map(|(i, _)| (kind, i))
            .collect()
    };
    let live: Vec<(Producer, usize)> = live_of(glass, Producer::Glass)
        .into_iter()
        .chain(live_of(water, Producer::Water))
        .chain((0..meshes.len()).map(|i| (Producer::GlassMesh, i)))
        .collect();
    let dists: Vec<f32> = live
        .iter()
        .map(|&(kind, i)| {
            let center = match kind {
                Producer::Glass => glass[i].0,
                Producer::Water => water[i].0,
                Producer::GlassMesh => meshes[i],
            };
            sort_distance(center, cam)
        })
        .collect();
    back_to_front_order(&dists)
        .into_iter()
        .map(|oi| live[oi])
        .collect()
}

/// Return the indices `0..distances.len()` ordered farthest camera distance
/// first (back-to-front). The sort is stable, so draws at equal distance keep
/// their original (declaration) order. Non-finite distances (NaN) are treated
/// as nearest so a degenerate value never pushes a draw behind valid geometry.
pub fn back_to_front_order(distances: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..distances.len()).collect();
    order.sort_by(|&a, &b| {
        // Farther (larger distance) sorts first. Map NaN to -inf so it lands
        // last (nearest), keeping a total order for `sort_by`.
        let da = if distances[a].is_finite() {
            distances[a]
        } else {
            f32::NEG_INFINITY
        };
        let db = if distances[b].is_finite() {
            distances[b]
        } else {
            f32::NEG_INFINITY
        };
        db.partial_cmp(&da).unwrap_or(core::cmp::Ordering::Equal)
    });
    order
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

    #[test]
    fn orders_farthest_first() {
        let d = [1.0, 5.0, 3.0];
        assert_eq!(back_to_front_order(&d), vec![1, 2, 0]);
    }

    #[test]
    fn empty_is_empty() {
        assert!(back_to_front_order(&[]).is_empty());
    }

    #[test]
    fn equal_distances_keep_declaration_order() {
        let d = [2.0, 2.0, 2.0];
        assert_eq!(back_to_front_order(&d), vec![0, 1, 2]);
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
    fn ordered_visible_excludes_hidden_and_sorts_back_to_front() {
        // Pane 1 is hidden; 0 (dist 5) and 2 (dist 3) are visible. Farthest
        // first => [0, 2]; the hidden pane never appears.
        let glass = [
            ([0.0, 0.0, 5.0], true),
            ([0.0, 0.0, 9.0], false),
            ([0.0, 0.0, 3.0], true),
        ];
        let order = ordered_visible(&glass, &[], &[], [0.0, 0.0, 0.0]);
        assert_eq!(order, vec![(Producer::Glass, 0), (Producer::Glass, 2)]);
    }

    #[test]
    fn ordered_visible_interleaves_the_two_producers() {
        // A pane standing in a pool has to composite in distance order, not in
        // producer order: the far pane draws first, then the water, then the near
        // pane.
        let glass = [([0.0, 0.0, 9.0], true), ([0.0, 0.0, 1.0], true)];
        let water = [([0.0, 0.0, 5.0], true), ([0.0, 0.0, 7.0], false)];
        let order = ordered_visible(&glass, &water, &[], [0.0, 0.0, 0.0]);
        assert_eq!(
            order,
            vec![
                (Producer::Glass, 0),
                (Producer::Water, 0),
                (Producer::Glass, 1),
            ]
        );
    }

    #[test]
    fn ordered_visible_interleaves_mesh_draws_with_the_static_producers() {
        // A see-through mesh sorts against panes and water by the same camera
        // distance, so it is not simply appended after them. Every mesh entry the
        // encoder passes is already visible, which is why the slice carries
        // centers alone.
        let glass = [([0.0, 0.0, 9.0], true)];
        let water = [([0.0, 0.0, 3.0], true)];
        let meshes = [[0.0, 0.0, 6.0], [0.0, 0.0, 1.0]];
        let order = ordered_visible(&glass, &water, &meshes, [0.0, 0.0, 0.0]);
        assert_eq!(
            order,
            vec![
                (Producer::Glass, 0),
                (Producer::GlassMesh, 0),
                (Producer::Water, 0),
                (Producer::GlassMesh, 1),
            ]
        );
    }

    #[test]
    fn ordered_visible_orders_meshes_alone_back_to_front() {
        // A world whose only transparent content is see-through meshes: the pass
        // still runs, and they still sort farthest first.
        let meshes = [[0.0, 0.0, 2.0], [0.0, 0.0, 8.0]];
        let order = ordered_visible(&[], &[], &meshes, [0.0, 0.0, 0.0]);
        assert_eq!(
            order,
            vec![(Producer::GlassMesh, 1), (Producer::GlassMesh, 0)]
        );
    }

    #[test]
    fn ordered_visible_is_empty_with_no_visible_records() {
        let glass = [([0.0, 0.0, 5.0], false)];
        let water = [([0.0, 0.0, 3.0], false)];
        assert!(ordered_visible(&glass, &water, &[], [0.0, 0.0, 0.0]).is_empty());
    }

    #[test]
    fn nan_sorts_last() {
        let d = [4.0, f32::NAN, 2.0];
        // 4.0 (farthest) → index 0, then 2.0 → index 2, then NaN → index 1.
        assert_eq!(back_to_front_order(&d), vec![0, 2, 1]);
    }
}
