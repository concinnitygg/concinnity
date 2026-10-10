// A terrain grid as render geometry: square chunks of cells, each small enough
// for 16-bit indices and culled on its own.

use alloc::vec::Vec;

use super::TerrainGrid;
use crate::gfx::mesh_payload::Vertex;
use crate::math::vec3::{dot, normalize_or, scale, sub};

/// Cells along each side of one render chunk.
pub const TERRAIN_CHUNK_CELLS: usize = 64;

/// One chunk of a terrain's render mesh, positioned relative to the terrain's
/// center.
#[derive(Debug, Clone)]
pub struct TerrainChunk {
    /// The chunk's corners, row-major.
    pub vertices: Vec<Vertex>,
    /// Two triangles per cell, wound to face up.
    pub indices: Vec<u16>,
    /// The chunk's bounds, `(min, max)`.
    pub bounds: ([f32; 3], [f32; 3]),
}

/// The render mesh of `grid`: every cell, two triangles each split along the
/// grid's diagonal, in chunks of at most [`TERRAIN_CHUNK_CELLS`] cells a side.
/// Normals are the grid's smooth corner normals, so neighboring chunks shade
/// seamlessly; texture coordinates are the corner's position in meters.
pub fn terrain_chunks(grid: &TerrainGrid) -> Vec<TerrainChunk> {
    let cells = grid.resolution() as usize;
    let starts = || (0..cells).step_by(TERRAIN_CHUNK_CELLS);
    let mut chunks = Vec::new();
    for row0 in starts() {
        for col0 in starts() {
            let col1 = (col0 + TERRAIN_CHUNK_CELLS).min(cells);
            let row1 = (row0 + TERRAIN_CHUNK_CELLS).min(cells);
            chunks.push(chunk(grid, [col0, col1], [row0, row1]));
        }
    }
    chunks
}

// The chunk spanning corners `cols[0]..=cols[1]` by `rows[0]..=rows[1]`.
fn chunk(grid: &TerrainGrid, cols: [usize; 2], rows: [usize; 2]) -> TerrainChunk {
    let across = cols[1] - cols[0] + 1;
    let down = rows[1] - rows[0] + 1;
    let mut vertices = Vec::with_capacity(across * down);
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for row in rows[0]..=rows[1] {
        for col in cols[0]..=cols[1] {
            let pos = grid.corner(col, row);
            let normal = grid.corner_normal(col, row);
            for k in 0..3 {
                lo[k] = lo[k].min(pos[k]);
                hi[k] = hi[k].max(pos[k]);
            }
            // The texture's u runs along +X, so the tangent is +X laid onto
            // the surface.
            let x = [1.0, 0.0, 0.0];
            let tangent = normalize_or(sub(x, scale(normal, dot(normal, x))), 1e-6, x);
            vertices.push(Vertex {
                pos,
                normal,
                tangent,
                color: [1.0; 3],
                uv: [pos[0], pos[2]],
            });
        }
    }
    let mut indices = Vec::with_capacity((across - 1) * (down - 1) * 6);
    for r in 0..down - 1 {
        for c in 0..across - 1 {
            let tl = (r * across + c) as u16;
            let tr = tl + 1;
            let bl = tl + across as u16;
            let br = bl + 1;
            indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }
    TerrainChunk {
        vertices,
        indices,
        bounds: (lo, hi),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::vec3::cross;
    use alloc::vec;

    fn grid(resolution: u32) -> TerrainGrid {
        let side = resolution as usize + 1;
        let heights = (0..side * side).map(|i| (i % 7) as f32 * 0.25).collect();
        TerrainGrid::new(resolution, [10.0, 6.0], heights).unwrap()
    }

    #[test]
    fn a_small_grid_is_one_chunk_of_every_corner() {
        let g = grid(4);
        let chunks = terrain_chunks(&g);
        assert_eq!(chunks.len(), 1);
        let c = &chunks[0];
        assert_eq!(c.vertices.len(), 25);
        assert_eq!(c.indices.len(), 4 * 4 * 6);
        for (i, v) in c.vertices.iter().enumerate() {
            assert_eq!(v.pos, g.corner(i % 5, i / 5));
        }
        assert_eq!(c.bounds.0[0], -10.0);
        assert_eq!(c.bounds.1[2], 6.0);
    }

    // A grid wider than one chunk splits into chunks that share their edge
    // corners, and together hold every cell exactly once.
    #[test]
    fn a_large_grid_splits_into_chunks_that_cover_every_cell() {
        let g = grid(TERRAIN_CHUNK_CELLS as u32 + 6);
        let chunks = terrain_chunks(&g);
        assert_eq!(chunks.len(), 4);
        let triangles: usize = chunks.iter().map(|c| c.indices.len() / 3).sum();
        let cells = g.resolution() as usize;
        assert_eq!(triangles, cells * cells * 2);
        let last = chunks.last().unwrap();
        assert_eq!(last.vertices.len(), 7 * 7);
        assert_eq!(
            last.vertices.last().unwrap().pos,
            g.corner(cells, cells),
            "the far corner closes the last chunk"
        );
        for c in &chunks {
            assert!(c.vertices.len() <= u16::MAX as usize + 1);
        }
    }

    // Every triangle is one the grid's own surface holds: its centroid sits at
    // the height the grid reports there, and it faces up.
    #[test]
    fn every_triangle_lies_on_the_grid_surface() {
        let g = grid(6);
        for c in terrain_chunks(&g) {
            for tri in c.indices.chunks_exact(3) {
                let [a, b, d] = [tri[0], tri[1], tri[2]].map(|i| c.vertices[i as usize].pos);
                let centroid = [
                    (a[0] + b[0] + d[0]) / 3.0,
                    (a[1] + b[1] + d[1]) / 3.0,
                    (a[2] + b[2] + d[2]) / 3.0,
                ];
                let h = g.height_at([centroid[0], centroid[2]]).unwrap();
                assert!((h - centroid[1]).abs() < 1e-4, "{h} vs {}", centroid[1]);
                assert!(cross(sub(b, a), sub(d, a))[1] > 0.0, "faces up");
            }
        }
    }

    #[test]
    fn tangents_follow_the_surface_along_x() {
        let flat = TerrainGrid::new(2, [1.0, 1.0], vec![0.0; 9]).unwrap();
        for v in &terrain_chunks(&flat)[0].vertices {
            assert_eq!(v.tangent, [1.0, 0.0, 0.0]);
            assert_eq!(v.normal, [0.0, 1.0, 0.0]);
            assert_eq!(v.uv, [v.pos[0], v.pos[2]]);
        }
    }
}
