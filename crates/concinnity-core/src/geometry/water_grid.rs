//! Flat tessellated quad for a WaterSurface.
//!
//! The mesh sits in the XZ plane at Y = 0. All vertical motion comes from the
//! per-frame Gerstner displacement applied by the water vertex shader; the
//! build-time geometry is just the rest pose. Per-vertex normals are flat
//! (+Y); the shader rebuilds them analytically from the wave derivatives.
//!
//! Parameters:
//!   half_width    -- half extent along X    (default 10.0)
//!   half_depth    -- half extent along Z    (default 10.0)
//!   subdivisions  -- grid resolution per axis (default 64, clamped 8..=255)

use alloc::string::String;
use alloc::vec::Vec;

use super::grid::Grid;

type Verts = Vec<([f32; 3], [f32; 3], [f32; 3], [f32; 2])>;

/// Build a flat water grid of `subdivisions` quads per axis.
pub fn build_water_grid(
    half_width: f32,
    half_depth: f32,
    subdivisions: u32,
) -> Result<(Verts, Vec<u16>), String> {
    let subdivisions = subdivisions.clamp(8, 255) as usize;

    let grid = Grid::new("water_grid", half_width, half_depth, subdivisions)?;
    let normal = [0.0f32, 1.0, 0.0];
    let color = [1.0f32, 1.0, 1.0];
    let verts: Verts = grid
        .points()
        .map(|p| ([p.x, 0.0, p.z], normal, color, [p.s, p.t]))
        .collect();
    Ok((verts, grid.indices()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_and_index_counts_match_grid() {
        let (verts, idxs) = build_water_grid(5.0, 5.0, 8).expect("builds");
        assert_eq!(verts.len(), 9 * 9);
        assert_eq!(idxs.len(), 8 * 8 * 6);
        // All vertices on Y = 0.
        for v in &verts {
            assert!(v.0[1].abs() < 1e-6);
        }
        // Corner positions span the half-widths.
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        for v in &verts {
            min_x = min_x.min(v.0[0]);
            max_x = max_x.max(v.0[0]);
        }
        assert!((min_x - -5.0).abs() < 1e-5);
        assert!((max_x - 5.0).abs() < 1e-5);
    }

    #[test]
    fn subdivisions_clamps_to_minimum() {
        let (verts, _) = build_water_grid(10.0, 10.0, 2).expect("builds");
        // Clamped to 8 → 9x9 grid.
        assert_eq!(verts.len(), 9 * 9);
    }
}
