// src/gfx/lod_select.rs
//
// Runtime LOD selection: which baked index slice a draw uses at a given camera
// distance, and the camera-to-object distance the pick is keyed on. Every
// backend runs these per draw, per frame. The build-time decimator that bakes
// the alternates these choose between is compute and lives above this crate.

use crate::gfx::render_types::{DrawObject, LodSlice, SkinnedDrawObject};
use crate::math::sqrt;

/// Whether an AABB encodes valid finite bounds suitable for frustum / distance
/// culling. A degenerate box (non-finite corner) disables culling.
pub fn bounds_finite(bb_min: [f32; 3], bb_max: [f32; 3]) -> bool {
    bb_min.iter().chain(bb_max.iter()).all(|c| c.is_finite())
}

/// The LOD level active at `distance`: 0 for LOD0, or `i + 1` for the
/// highest-indexed alternate whose `switch_distance` is at or below it.
/// `alternates` is in ascending threshold order, so the scan stops early.
pub fn pick_lod_level(alternates: &[LodSlice], distance: f32) -> usize {
    let mut level = 0usize;
    for (i, slice) in alternates.iter().enumerate() {
        if distance >= slice.switch_distance {
            level = i + 1;
        } else {
            break;
        }
    }
    level
}

/// The `(index_offset, index_count)` slice active at `distance`, given the LOD0
/// pair and the alternates. Returns `lod0` when no alternate applies.
pub fn pick_lod_slice(
    lod0: (usize, usize),
    alternates: &[LodSlice],
    distance: f32,
) -> (usize, usize) {
    match pick_lod_level(alternates, distance) {
        0 => lod0,
        level => {
            let slice = &alternates[level - 1];
            (slice.index_offset, slice.index_count)
        }
    }
}

/// Distance from `cam_pos` to a skinned object's authored placement (the
/// column-3 translation of its model matrix). Skinned objects deform every
/// frame, so they have no static AABB: this is the cheap stand-in the
/// per-frame LOD picks use.
pub fn skinned_camera_distance(obj: &SkinnedDrawObject, cam_pos: [f32; 3]) -> f32 {
    distance_to(obj.translation(), cam_pos)
}

/// Distance from `cam_pos` to the centre of `obj`'s world AABB, used to pick
/// the active LOD slice each frame. Dynamic props (sentinel non-finite AABB)
/// fall back to the model-matrix translation so they still LOD by their
/// authored placement.
pub fn camera_distance(obj: &DrawObject, cam_pos: [f32; 3]) -> f32 {
    let centre = if obj.cullable() {
        [
            0.5 * (obj.bb_min[0] + obj.bb_max[0]),
            0.5 * (obj.bb_min[1] + obj.bb_max[1]),
            0.5 * (obj.bb_min[2] + obj.bb_max[2]),
        ]
    } else {
        [obj.model[3][0], obj.model[3][1], obj.model[3][2]]
    };
    distance_to(centre, cam_pos)
}

fn distance_to(point: [f32; 3], cam_pos: [f32; 3]) -> f32 {
    let dx = point[0] - cam_pos[0];
    let dy = point[1] - cam_pos[1];
    let dz = point[2] - cam_pos[2];
    sqrt(dx * dx + dy * dy + dz * dz)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{draw_object, skinned_draw_object};
    use alloc::vec;

    fn slice(switch_distance: f32, index_offset: usize) -> LodSlice {
        LodSlice {
            index_offset,
            index_count: 3,
            switch_distance,
        }
    }

    #[test]
    fn bounds_finite_rejects_a_sentinel_box() {
        assert!(bounds_finite([0.0; 3], [1.0; 3]));
        assert!(!bounds_finite([f32::NAN, 0.0, 0.0], [1.0; 3]));
        assert!(!bounds_finite([0.0; 3], [f32::INFINITY, 1.0, 1.0]));
    }

    #[test]
    fn lod_level_walks_ascending_thresholds() {
        let alts = vec![slice(10.0, 30), slice(20.0, 60)];
        assert_eq!(pick_lod_level(&alts, 0.0), 0);
        assert_eq!(pick_lod_level(&alts, 9.9), 0);
        // The threshold itself selects the alternate.
        assert_eq!(pick_lod_level(&alts, 10.0), 1);
        assert_eq!(pick_lod_level(&alts, 19.9), 1);
        assert_eq!(pick_lod_level(&alts, 100.0), 2);
        assert_eq!(pick_lod_level(&[], 100.0), 0);
    }

    #[test]
    fn lod_slice_returns_lod0_until_an_alternate_applies() {
        let alts = vec![slice(10.0, 30), slice(20.0, 60)];
        assert_eq!(pick_lod_slice((0, 9), &alts, 1.0), (0, 9));
        assert_eq!(pick_lod_slice((0, 9), &alts, 15.0), (30, 3));
        assert_eq!(pick_lod_slice((0, 9), &alts, 25.0), (60, 3));
        assert_eq!(pick_lod_slice((0, 9), &[], 25.0), (0, 9));
    }

    #[test]
    fn camera_distance_uses_the_aabb_centre_and_falls_back_to_the_translation() {
        let mut obj = draw_object();
        obj.bb_min = [-1.0, -1.0, -1.0];
        obj.bb_max = [1.0, 1.0, 1.0];
        obj.model[3] = [10.0, 0.0, 0.0, 1.0];
        // Cullable: measured from the AABB centre (the origin), not the model.
        assert!((camera_distance(&obj, [5.0, 0.0, 0.0]) - 5.0).abs() < 1e-4);
        // A sentinel box falls back to the model-matrix translation.
        obj.bb_min = [f32::NAN; 3];
        assert!((camera_distance(&obj, [5.0, 0.0, 0.0]) - 5.0).abs() < 1e-4);
        obj.model[3] = [20.0, 0.0, 0.0, 1.0];
        assert!((camera_distance(&obj, [5.0, 0.0, 0.0]) - 15.0).abs() < 1e-4);
    }

    #[test]
    fn skinned_distance_measures_from_the_model_translation() {
        let mut obj = skinned_draw_object();
        obj.model[3] = [0.0, 3.0, 4.0, 1.0];
        assert!((skinned_camera_distance(&obj, [0.0; 3]) - 5.0).abs() < 1e-4);
    }
}
