// A regular XZ vertex grid: (subdivisions + 1)^2 vertices spanning
// [-half_width, half_width] x [-half_depth, half_depth], row-major with rows
// running along Z, two triangles per cell, which the water generator fills.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

pub(super) struct Grid {
    half_width: f32,
    half_depth: f32,
    subdivisions: usize,
}

// One grid vertex: its normalized position in [0, 1] on each axis, and its
// world XZ.
pub(super) struct GridPoint {
    pub(super) s: f32,
    pub(super) t: f32,
    pub(super) x: f32,
    pub(super) z: f32,
}

impl Grid {
    // `name` labels the refusal when the grid has more vertices than a u16
    // index can address.
    pub(super) fn new(
        name: &str,
        half_width: f32,
        half_depth: f32,
        subdivisions: usize,
    ) -> Result<Self, String> {
        let count = (subdivisions + 1) * (subdivisions + 1);
        if count > 65536 {
            return Err(format!(
                "{name} subdivisions {subdivisions} produces {count} vertices, exceeding the u16 limit; use subdivisions <= 255"
            ));
        }
        Ok(Self {
            half_width,
            half_depth,
            subdivisions,
        })
    }

    fn cols(&self) -> usize {
        self.subdivisions + 1
    }

    pub(super) fn points(&self) -> impl Iterator<Item = GridPoint> + '_ {
        let n = self.subdivisions as f32;
        (0..self.cols()).flat_map(move |row| {
            (0..self.cols()).map(move |col| {
                let s = col as f32 / n;
                let t = row as f32 / n;
                GridPoint {
                    s,
                    t,
                    x: -self.half_width + s * self.half_width * 2.0,
                    z: -self.half_depth + t * self.half_depth * 2.0,
                }
            })
        })
    }

    pub(super) fn indices(&self) -> Vec<u16> {
        let cols = self.cols();
        let mut idxs = Vec::with_capacity(self.subdivisions * self.subdivisions * 6);
        for row in 0..self.subdivisions {
            for col in 0..self.subdivisions {
                let tl = (row * cols + col) as u16;
                let tr = tl + 1;
                let bl = tl + cols as u16;
                let br = bl + 1;
                idxs.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
            }
        }
        idxs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn points_run_row_major_across_the_extent() {
        let grid = Grid::new("test", 2.0, 1.0, 2).unwrap();
        let points: Vec<GridPoint> = grid.points().collect();
        assert_eq!(points.len(), 9);
        let first = &points[0];
        assert_eq!((first.s, first.t, first.x, first.z), (0.0, 0.0, -2.0, -1.0));
        let second = &points[1];
        assert_eq!((second.s, second.x), (0.5, 0.0));
        let last = &points[8];
        assert_eq!((last.s, last.t, last.x, last.z), (1.0, 1.0, 2.0, 1.0));
    }

    #[test]
    fn each_cell_is_two_triangles_over_its_four_corners() {
        let grid = Grid::new("test", 1.0, 1.0, 1).unwrap();
        assert_eq!(grid.indices(), [0, 2, 1, 1, 2, 3]);
        let grid = Grid::new("test", 1.0, 1.0, 3).unwrap();
        assert_eq!(grid.indices().len(), 3 * 3 * 6);
    }

    #[test]
    fn a_grid_past_the_u16_index_range_is_refused() {
        assert!(Grid::new("test", 1.0, 1.0, 255).is_ok());
        let err = Grid::new("water_grid", 1.0, 1.0, 256).err().unwrap();
        assert!(err.starts_with("water_grid subdivisions 256"), "{err}");
    }
}
