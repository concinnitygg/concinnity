// The square height grid a terrain is cooked into, and the surface its two
// triangles per cell span.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::math::{floor, sqrt};

/// `resolution` cells along each side of a rectangle `2 * extent` across, with
/// one height per corner.
///
/// Heights are row-major, `(resolution + 1)^2` of them: rows run along `+Z`
/// and columns along `+X`, starting at the `-X, -Z` corner. Each is meters
/// above the terrain's base. Every cell is split into two triangles along the
/// diagonal from its `+X, -Z` corner to its `-X, +Z` corner, which is the
/// surface the render mesh draws, the collider holds and the grass roots in.
#[derive(Debug, Clone, PartialEq)]
pub struct TerrainGrid {
    resolution: u32,
    extent: [f32; 2],
    heights: Vec<f32>,
}

impl TerrainGrid {
    /// A grid of `heights` over `extent`, or an error naming what does not fit:
    /// a resolution under one cell, a non-positive extent, the wrong number of
    /// heights, or a height that is not a finite number.
    pub fn new(resolution: u32, extent: [f32; 2], heights: Vec<f32>) -> Result<Self, String> {
        if resolution == 0 {
            return Err("a terrain grid needs at least one cell".into());
        }
        if !extent.iter().all(|e| e.is_finite() && *e > 0.0) {
            return Err(format!("terrain extent {extent:?} is not positive"));
        }
        let side = resolution as usize + 1;
        if heights.len() != side * side {
            return Err(format!(
                "a {resolution}-cell terrain grid needs {} heights, not {}",
                side * side,
                heights.len()
            ));
        }
        if heights.iter().any(|h| !h.is_finite()) {
            return Err("a terrain height is not a finite number".into());
        }
        Ok(Self {
            resolution,
            extent,
            heights,
        })
    }

    /// Cells along each side.
    pub fn resolution(&self) -> u32 {
        self.resolution
    }

    /// Corners along each side.
    pub fn side(&self) -> usize {
        self.resolution as usize + 1
    }

    /// Half-width and half-depth, in meters.
    pub fn extent(&self) -> [f32; 2] {
        self.extent
    }

    /// The heights, row-major.
    pub fn heights(&self) -> &[f32] {
        &self.heights
    }

    /// Width and depth of one cell, in meters.
    pub fn cell_size(&self) -> [f32; 2] {
        let n = self.resolution as f32;
        [2.0 * self.extent[0] / n, 2.0 * self.extent[1] / n]
    }

    /// The height at corner (`col`, `row`).
    pub fn height(&self, col: usize, row: usize) -> f32 {
        self.heights[row * self.side() + col]
    }

    /// Lowest and highest heights.
    pub fn height_range(&self) -> [f32; 2] {
        self.heights
            .iter()
            .fold([f32::INFINITY, f32::NEG_INFINITY], |[lo, hi], &h| {
                [lo.min(h), hi.max(h)]
            })
    }

    /// The position of corner (`col`, `row`) relative to the terrain's center.
    pub fn corner(&self, col: usize, row: usize) -> [f32; 3] {
        let [cx, cz] = self.cell_size();
        [
            -self.extent[0] + col as f32 * cx,
            self.height(col, row),
            -self.extent[1] + row as f32 * cz,
        ]
    }

    // The cell under `local` (relative to the center) and the fraction across
    // it on each axis, or `None` off the grid.
    fn cell_at(&self, local: [f32; 2]) -> Option<([usize; 2], [f32; 2])> {
        let [cx, cz] = self.cell_size();
        let gx = (local[0] + self.extent[0]) / cx;
        let gz = (local[1] + self.extent[1]) / cz;
        let n = self.resolution as f32;
        if !(0.0..=n).contains(&gx) || !(0.0..=n).contains(&gz) {
            return None;
        }
        let col = floor(gx).min(n - 1.0);
        let row = floor(gz).min(n - 1.0);
        Some(([col as usize, row as usize], [gx - col, gz - row]))
    }

    /// The surface height at `local` (relative to the center, `[x, z]`), on
    /// the triangle the point stands over. `None` off the grid.
    pub fn height_at(&self, local: [f32; 2]) -> Option<f32> {
        let ([col, row], [fx, fz]) = self.cell_at(local)?;
        let h00 = self.height(col, row);
        let h10 = self.height(col + 1, row);
        let h01 = self.height(col, row + 1);
        let h11 = self.height(col + 1, row + 1);
        Some(if fx + fz <= 1.0 {
            h00 + (h10 - h00) * fx + (h01 - h00) * fz
        } else {
            h11 + (h01 - h11) * (1.0 - fx) + (h10 - h11) * (1.0 - fz)
        })
    }

    /// The surface's slope `[dh/dx, dh/dz]` at `local`, blended across the
    /// cell so it turns smoothly rather than jumping at each triangle edge.
    /// `None` off the grid.
    pub fn slope_at(&self, local: [f32; 2]) -> Option<[f32; 2]> {
        let ([col, row], [fx, fz]) = self.cell_at(local)?;
        let h00 = self.height(col, row);
        let h10 = self.height(col + 1, row);
        let h01 = self.height(col, row + 1);
        let h11 = self.height(col + 1, row + 1);
        let [cx, cz] = self.cell_size();
        let dx = ((h10 - h00) * (1.0 - fz) + (h11 - h01) * fz) / cx;
        let dz = ((h01 - h00) * (1.0 - fx) + (h11 - h10) * fx) / cz;
        Some([dx, dz])
    }

    /// The smooth surface normal at corner (`col`, `row`), from the heights on
    /// either side of it.
    pub fn corner_normal(&self, col: usize, row: usize) -> [f32; 3] {
        let last = self.resolution as usize;
        let [cx, cz] = self.cell_size();
        let (c0, c1) = (col.saturating_sub(1), (col + 1).min(last));
        let (r0, r1) = (row.saturating_sub(1), (row + 1).min(last));
        let dx = (self.height(c1, row) - self.height(c0, row)) / ((c1 - c0) as f32 * cx);
        let dz = (self.height(col, r1) - self.height(col, r0)) / ((r1 - r0) as f32 * cz);
        let len = sqrt(dx * dx + 1.0 + dz * dz);
        [-dx / len, 1.0 / len, -dz / len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // A 2-cell grid over [-2, 2] x [-1, 1] whose height is the column index
    // plus ten times the row index.
    fn ramp() -> TerrainGrid {
        let heights = (0..3)
            .flat_map(|row| (0..3).map(move |col| col as f32 + 10.0 * row as f32))
            .collect();
        TerrainGrid::new(2, [2.0, 1.0], heights).unwrap()
    }

    #[test]
    fn a_grid_refuses_shapes_that_name_no_surface() {
        assert!(TerrainGrid::new(0, [1.0, 1.0], vec![0.0]).is_err());
        assert!(TerrainGrid::new(1, [0.0, 1.0], vec![0.0; 4]).is_err());
        let err = TerrainGrid::new(2, [1.0, 1.0], vec![0.0; 4]).unwrap_err();
        assert!(err.contains("needs 9 heights"), "{err}");
        assert!(TerrainGrid::new(1, [1.0, 1.0], vec![0.0, f32::NAN, 0.0, 0.0]).is_err());
    }

    #[test]
    fn corners_run_row_major_from_the_low_corner() {
        let g = ramp();
        assert_eq!(g.side(), 3);
        assert_eq!(g.cell_size(), [2.0, 1.0]);
        assert_eq!(g.corner(0, 0), [-2.0, 0.0, -1.0]);
        assert_eq!(g.corner(2, 1), [2.0, 12.0, 0.0]);
        assert_eq!(g.height_range(), [0.0, 22.0]);
    }

    #[test]
    fn the_surface_passes_through_every_corner() {
        let g = ramp();
        for row in 0..3 {
            for col in 0..3 {
                let [x, h, z] = g.corner(col, row);
                assert_eq!(g.height_at([x, z]), Some(h), "corner {col},{row}");
            }
        }
    }

    // A plane is exact on both triangles of every cell.
    #[test]
    fn a_planar_grid_interpolates_exactly() {
        let g = ramp();
        for &(x, z) in &[(-1.5, -0.75), (0.5, 0.9), (1.9, -0.1), (-0.2, 0.3)] {
            let expected = (x + 2.0) / 2.0 + 10.0 * (z + 1.0);
            let h = g.height_at([x, z]).unwrap();
            assert!((h - expected).abs() < 1e-4, "({x}, {z}): {h} vs {expected}");
            let [dx, dz] = g.slope_at([x, z]).unwrap();
            assert!((dx - 0.5).abs() < 1e-5 && (dz - 10.0).abs() < 1e-5);
        }
    }

    // One raised corner: the cell's two triangles meet along the +X,-Z to
    // -X,+Z diagonal, so the raised -X,-Z corner lifts only the first one.
    #[test]
    fn each_cell_splits_along_its_diagonal() {
        let g = TerrainGrid::new(1, [1.0, 1.0], vec![4.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(g.height_at([0.0, 0.0]), Some(0.0), "on the diagonal");
        assert_eq!(g.height_at([-0.5, -0.5]), Some(2.0), "first triangle");
        assert_eq!(g.height_at([0.5, 0.5]), Some(0.0), "second triangle");
    }

    #[test]
    fn points_off_the_grid_have_no_height() {
        let g = ramp();
        assert_eq!(g.height_at([2.01, 0.0]), None);
        assert_eq!(g.height_at([0.0, -1.01]), None);
        assert_eq!(g.slope_at([-3.0, 0.0]), None);
    }

    #[test]
    fn a_flat_grid_points_every_normal_up() {
        let g = TerrainGrid::new(3, [5.0, 5.0], vec![1.5; 16]).unwrap();
        for row in 0..4 {
            for col in 0..4 {
                assert_eq!(g.corner_normal(col, row), [0.0, 1.0, 0.0]);
            }
        }
    }

    #[test]
    fn a_normal_leans_away_from_the_rise() {
        let g = ramp();
        let n = g.corner_normal(1, 1);
        assert!(n[0] < 0.0 && n[2] < 0.0 && n[1] > 0.0, "{n:?}");
        let len = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        assert!((len - 1.0).abs() < 1e-5);
    }
}
