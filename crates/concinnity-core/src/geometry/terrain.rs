// Subdivided terrain grid with deterministic height displacement.
//
// Heights are three octaves of bilinear value noise over the grid lattice, so
// the output is identical across builds. The physics heightfield collider
// samples the same function, so the collided surface matches the rendered one
// vertex for vertex.

use alloc::string::String;
use alloc::vec::Vec;

use super::Vert;
use super::grid::Grid;
use crate::math::floor;
use crate::math::noise::lattice_value;

/// Build a displaced terrain grid. `subdivisions` is the grid resolution per
/// axis (clamped to 4..=255); `amplitude` is the peak height above the base
/// plane in meters.
pub fn build_terrain(
    half_width: f32,
    half_depth: f32,
    subdivisions: u32,
    amplitude: f32,
) -> Result<(Vec<Vert>, Vec<u16>), String> {
    let subdivisions = terrain_subdivisions(subdivisions);
    let grid = Grid::new("terrain", half_width, half_depth, subdivisions as usize)?;
    let verts = grid.displaced([0.55, 0.62, 0.42], |p| {
        terrain_height(p.col as f32, p.row as f32, subdivisions, amplitude)
    });
    Ok((verts, grid.indices()))
}

// The grid resolution a terrain authored with `subdivisions` is built at.
pub(crate) fn terrain_subdivisions(subdivisions: u32) -> u32 {
    subdivisions.clamp(4, 255)
}

// The Y displacement at fractional lattice position (col, row), each in
// [0, subdivisions]. Three octaves of bilinear value noise give coarse hills,
// medium bumps, and fine surface variation.
pub(crate) fn terrain_height(col: f32, row: f32, subdivisions: u32, amplitude: f32) -> f32 {
    const OCTAVES: [(u32, f32); 3] = [(1, 1.00), (3, 0.40), (9, 0.15)];

    let mut sum = 0.0f32;
    let mut weight_sum = 0.0f32;

    for (divisor, weight) in OCTAVES {
        let scale = (subdivisions / divisor).max(1) as f32;
        let gs = col / scale;
        let gt = row / scale;
        let gx = floor(gs) as u32;
        let gy = floor(gt) as u32;
        let fx = gs - gx as f32;
        let fy = gt - gy as f32;

        let h00 = lattice_value(gx, gy);
        let h10 = lattice_value(gx + 1, gy);
        let h01 = lattice_value(gx, gy + 1);
        let h11 = lattice_value(gx + 1, gy + 1);
        let top = h00 + (h10 - h00) * fx;
        let bot = h01 + (h11 - h01) * fx;
        sum += (top + (bot - top) * fy) * weight;
        weight_sum += weight;
    }

    let normalized = sum / weight_sum;
    (normalized - 0.05).max(0.0) * amplitude
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_counts_and_extents_follow_the_subdivision_count() {
        let (verts, idxs) = build_terrain(10.0, 5.0, 8, 3.0).unwrap();
        assert_eq!(verts.len(), 9 * 9);
        assert_eq!(idxs.len(), 8 * 8 * 6);
        assert!(idxs.iter().all(|&i| (i as usize) < verts.len()));

        let mut mn = [f32::INFINITY; 3];
        let mut mx = [f32::NEG_INFINITY; 3];
        for (pos, ..) in &verts {
            for k in 0..3 {
                mn[k] = mn[k].min(pos[k]);
                mx[k] = mx[k].max(pos[k]);
            }
        }
        assert_eq!((mn[0], mx[0]), (-10.0, 10.0));
        assert_eq!((mn[2], mx[2]), (-5.0, 5.0));
        // Heights stay inside [0, amplitude] and vary across the grid.
        assert!(mn[1] >= 0.0 && mx[1] <= 3.0);
        assert!(mx[1] > mn[1], "expected height variation, got flat {mx:?}");
    }

    #[test]
    fn zero_amplitude_produces_a_flat_grid() {
        let (verts, _) = build_terrain(64.0, 64.0, 4, 0.0).unwrap();
        assert!(verts.iter().all(|(pos, ..)| pos[1] == 0.0));
        assert!(verts.iter().all(|(_, n, ..)| *n == [0.0, 1.0, 0.0]));
    }

    #[test]
    fn subdivisions_clamp_to_the_supported_range() {
        // Below the floor clamps to 4 (5x5 lattice)...
        let (small, _) = build_terrain(64.0, 64.0, 0, 4.0).unwrap();
        assert_eq!(small.len(), 5 * 5);
        // ...and above the ceiling clamps to 255, the largest grid that still
        // indexes with u16.
        let (large, idxs) = build_terrain(64.0, 64.0, 4096, 4.0).unwrap();
        assert_eq!(large.len(), 256 * 256);
        assert_eq!(idxs.len(), 255 * 255 * 6);
    }

    #[test]
    fn terrain_height_is_deterministic_and_scales_with_amplitude() {
        let a = terrain_height(3.0, 7.0, 32, 4.0);
        assert_eq!(a, terrain_height(3.0, 7.0, 32, 4.0));
        assert!((terrain_height(3.0, 7.0, 32, 8.0) - a * 2.0).abs() < 1e-5);
        // The noise is floored at the base plane, never negative.
        for col in 0..32u32 {
            for row in 0..32u32 {
                assert!(terrain_height(col as f32, row as f32, 32, 4.0) >= 0.0);
            }
        }
    }

    #[test]
    fn flat_terrain_is_zero_height() {
        assert_eq!(terrain_height(16.0, 16.0, 32, 0.0), 0.0);
        assert_eq!(terrain_height(21.0, 13.5, 32, 0.0), 0.0);
    }

    #[test]
    fn terrain_height_is_continuous_and_bounded() {
        // Height never exceeds the amplitude and neighboring samples are close.
        let mut prev = terrain_height(0.0, 16.0, 32, 4.0);
        let mut col = 0.0;
        while col <= 32.0 {
            let h = terrain_height(col, 16.0, 32, 4.0);
            assert!(
                (0.0..=4.0).contains(&h),
                "height {h} out of range at col={col}"
            );
            assert!((h - prev).abs() < 1.0, "terrain jumped at col={col}");
            prev = h;
            col += 0.25;
        }
    }

    // Between two lattice points the height is the bilinear blend of theirs,
    // so a fractional sample lands between its neighbors.
    #[test]
    fn a_fractional_sample_lies_between_its_lattice_neighbors() {
        let lo = terrain_height(5.0, 9.0, 32, 4.0);
        let hi = terrain_height(6.0, 9.0, 32, 4.0);
        let mid = terrain_height(5.5, 9.0, 32, 4.0);
        assert!(mid >= lo.min(hi) - 1e-6 && mid <= lo.max(hi) + 1e-6);
    }
}
