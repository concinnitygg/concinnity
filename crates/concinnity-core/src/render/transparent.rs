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

    use alloc::vec;
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
