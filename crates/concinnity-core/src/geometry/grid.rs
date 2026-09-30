// A regular XZ vertex grid: (subdivisions + 1)^2 vertices spanning
// [-half_width, half_width] x [-half_depth, half_depth], row-major with rows
// running along Z, two triangles per cell. The terrain, heightfield and water
// generators differ only in how they fill each vertex.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use super::Vert;
use crate::math::vec3::{vec3_add, vec3_face_normal, vec3_normalize};

pub(super) struct Grid {
    half_width: f32,
    half_depth: f32,
    subdivisions: usize,
}

// One grid vertex: its lattice cell, its normalized position in [0, 1] on each
// axis, and its world XZ.
pub(super) struct GridPoint {
    pub(super) col: usize,
    pub(super) row: usize,
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
                    col,
                    row,
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

    // Vertices displaced to `height` at each grid point, with smooth normals
    // accumulated from every triangle sharing a vertex and a world-XZ uv.
    pub(super) fn displaced(
        &self,
        color: [f32; 3],
        mut height: impl FnMut(&GridPoint) -> f32,
    ) -> Vec<Vert> {
        let positions: Vec<[f32; 3]> = self.points().map(|p| [p.x, height(&p), p.z]).collect();
        let normals = self.smooth_normals(&positions);
        positions
            .iter()
            .zip(normals)
            .map(|(&[x, y, z], n)| ([x, y, z], vec3_normalize(n), color, [x, z]))
            .collect()
    }

    fn smooth_normals(&self, positions: &[[f32; 3]]) -> Vec<[f32; 3]> {
        let mut normals = vec![[0.0; 3]; positions.len()];
        for tri in self.indices().chunks_exact(3) {
            let [a, b, c] = [tri[0], tri[1], tri[2]].map(usize::from);
            let n = vec3_face_normal(positions[a], positions[b], positions[c]);
            for v in [a, b, c] {
                vec3_add(&mut normals[v], n);
            }
        }
        normals
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
        assert_eq!((first.col, first.row, first.x, first.z), (0, 0, -2.0, -1.0));
        let second = &points[1];
        assert_eq!(
            (second.col, second.row, second.s, second.x),
            (1, 0, 0.5, 0.0)
        );
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

    #[test]
    fn a_flat_displacement_faces_up_with_world_xz_uvs() {
        let grid = Grid::new("test", 3.0, 3.0, 4).unwrap();
        let verts = grid.displaced([0.1, 0.2, 0.3], |_| 0.0);
        assert_eq!(verts.len(), 25);
        for (pos, n, color, uv) in &verts {
            assert_eq!(*n, [0.0, 1.0, 0.0]);
            assert_eq!(*color, [0.1, 0.2, 0.3]);
            assert_eq!(*uv, [pos[0], pos[2]]);
        }
    }

    #[test]
    fn a_slope_tilts_every_normal_against_it() {
        let grid = Grid::new("test", 1.0, 1.0, 2).unwrap();
        let verts = grid.displaced([1.0; 3], |p| p.x);
        let expected = vec3_normalize([-1.0, 1.0, 0.0]);
        for (_, n, ..) in &verts {
            for k in 0..3 {
                assert!((n[k] - expected[k]).abs() < 1e-5, "{n:?}");
            }
        }
    }
}
