//! The grid a simulated ocean is drawn on: centered on the camera rather than
//! on the surface, square cells near it and cells widening geometrically
//! toward the edge, so a sea kilometers across gets fine geometry where it is
//! seen up close. Each frame the grid's anchor snaps to the camera by whole
//! inner cells, so the near vertices sit still on the water as it moves; the
//! shader clamps every vertex to the surface's rectangle.

use crate::gfx::mesh_payload::Vertex;
use alloc::vec::Vec;
use concinnity_core::math::SaturatingFrom;
use concinnity_core::math::{hypot, powi};

/// Grid lines on each side of the anchor.
const OCEAN_GRID_HALF_LINES: u32 = 127;

/// Square cells on each side of the anchor before they start to widen.
const OCEAN_GRID_INNER_CELLS: u32 = 16;

/// The inner cell, in peak wavelengths, and the range it is held to, meters.
const INNER_CELL_PER_PEAK: f32 = 1.0 / 48.0;
const INNER_CELL_RANGE: [f32; 2] = [0.25, 8.0];

/// The layout of an ocean's grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OceanGrid {
    /// The side of an inner cell, in meters.
    pub inner_cell: f32,
    /// How much wider each outer cell is than the one before it.
    pub growth: f32,
}

impl OceanGrid {
    /// The grid for a sea peaking at `peak_wavelength` meters on a surface of
    /// half-widths `extent`: wide enough to reach every corner from anywhere
    /// on the surface.
    pub fn new(peak_wavelength: f32, extent: [f32; 2]) -> Self {
        Self::reaching(peak_wavelength, 2.0 * hypot(extent[0], extent[1]))
    }

    /// The grid for a sea peaking at `peak_wavelength` meters, reaching
    /// `reach` meters from its anchor.
    pub fn reaching(peak_wavelength: f32, reach: f32) -> Self {
        let inner_cell =
            (peak_wavelength * INNER_CELL_PER_PEAK).clamp(INNER_CELL_RANGE[0], INNER_CELL_RANGE[1]);
        Self {
            inner_cell,
            growth: growth_to_reach(inner_cell, reach),
        }
    }

    /// Distance from the anchor to grid line `i` (signed).
    pub fn line(&self, i: i32) -> f32 {
        let n0 = i32::saturating_from(OCEAN_GRID_INNER_CELLS);
        let m = i32::saturating_from(i.unsigned_abs()) - n0;
        let at = if m <= 0 {
            i.unsigned_abs() as f32 * self.inner_cell
        } else {
            n0 as f32 * self.inner_cell
                + outer_span(self.inner_cell, self.growth, u32::saturating_from(m))
        };
        if i < 0 { -at } else { at }
    }

    /// How far the grid reaches from its anchor.
    pub fn reach(&self) -> f32 {
        self.line(i32::saturating_from(OCEAN_GRID_HALF_LINES))
    }

    /// The anchor for a camera at `xz`: snapped to whole inner cells.
    pub fn snap(&self, xz: [f64; 2]) -> [f64; 2] {
        let cell = f64::from(self.inner_cell);
        xz.map(|v| libm::round(v / cell) * cell)
    }

    /// `[inner cell, inner half-width, widening per meter, _]`: what the
    /// shader estimates the local cell size from, `inner + max(|x| - half,
    /// 0) * widening` along each axis.
    pub fn gpu_lane(&self) -> [f32; 4] {
        let half = OCEAN_GRID_INNER_CELLS as f32 * self.inner_cell;
        [
            self.inner_cell,
            half,
            (self.growth - 1.0) / self.growth,
            0.0,
        ]
    }

    /// The cell size the shader estimates at `x` meters from the anchor.
    #[cfg(test)]
    fn estimated_cell(self, x: f32) -> f32 {
        let [inner, half, widening, _] = self.gpu_lane();
        inner + (x.abs() - half).max(0.0) * widening
    }

    /// The square rings of cells around the anchor: ring `r` is the cells
    /// that lie `r` whole cells out from the four around it.
    pub fn rings(&self) -> u32 {
        OCEAN_GRID_HALF_LINES
    }

    /// Indices the first `k` rings take in [`Self::ring_mesh`]: the square
    /// block of cells out to grid line `k` on each side.
    pub fn ring_indices(&self, k: u32) -> u32 {
        let k = k.min(OCEAN_GRID_HALF_LINES);
        24 * k * k
    }

    /// [`Self::mesh`] with its triangles in rings out from the anchor, the
    /// innermost first, so the block within any ring is a prefix.
    pub fn ring_mesh(&self) -> (Vec<Vertex>, Vec<u16>) {
        let (vertices, rows) = self.mesh();
        let cells = 2 * OCEAN_GRID_HALF_LINES as usize;
        let ring = |c: usize| {
            let u = i32::saturating_from(c) - i32::saturating_from(OCEAN_GRID_HALF_LINES);
            u32::saturating_from(if u >= 0 { u } else { -u - 1 })
        };
        let mut order: Vec<usize> = (0..cells * cells).collect();
        order.sort_by_key(|&cell| ring(cell / cells).max(ring(cell % cells)));
        let indices = order
            .iter()
            .flat_map(|&cell| rows[cell * 6..cell * 6 + 6].iter().copied())
            .collect();
        (vertices, indices)
    }

    /// The grid's vertices, relative to the anchor, and its triangles.
    pub fn mesh(&self) -> (Vec<Vertex>, Vec<u16>) {
        let n = i32::saturating_from(OCEAN_GRID_HALF_LINES);
        let lines: Vec<f32> = (-n..=n).map(|i| self.line(i)).collect();
        let cols = lines.len();
        let mut vertices = Vec::with_capacity(cols * cols);
        for &z in &lines {
            for &x in &lines {
                vertices.push(Vertex {
                    pos: [x, 0.0, z],
                    normal: [0.0, 1.0, 0.0],
                    tangent: [1.0, 0.0, 0.0],
                    color: [1.0, 1.0, 1.0],
                    uv: [0.0, 0.0],
                });
            }
        }
        let mut indices = Vec::with_capacity((cols - 1) * (cols - 1) * 6);
        for row in 0..cols - 1 {
            for col in 0..cols - 1 {
                let tl = u16::saturating_from(row * cols + col);
                let tr = tl + 1;
                let bl = tl + u16::saturating_from(cols);
                let br = bl + 1;
                indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
            }
        }
        (vertices, indices)
    }
}

// The span of the first `m` outer cells, each `growth` times the last,
// starting at `inner * growth`.
fn outer_span(inner: f32, growth: f32, m: u32) -> f32 {
    if growth <= 1.0 {
        return inner * m as f32;
    }
    inner * growth * (powi(growth, i32::saturating_from(m)) - 1.0) / (growth - 1.0)
}

// The least growth at which the grid's outer cells reach `reach` meters.
fn growth_to_reach(inner: f32, reach: f32) -> f32 {
    let outer = OCEAN_GRID_HALF_LINES - OCEAN_GRID_INNER_CELLS;
    let need = reach - OCEAN_GRID_INNER_CELLS as f32 * inner;
    if need <= inner * outer as f32 {
        return 1.0;
    }
    let (mut lo, mut hi) = (1.0_f32, 2.0_f32);
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        if outer_span(inner, mid, outer) < need {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ocean() -> OceanGrid {
        OceanGrid::new(56.0, [5000.0, 5000.0])
    }

    // The grid reaches every corner of the surface from its far corner, its
    // lines rise monotonically, and its cells widen only past the inner ring.
    #[test]
    fn the_grid_reaches_across_the_surface() {
        let g = ocean();
        assert!(
            g.reach() >= 2.0 * 5000.0_f32.hypot(5000.0) * 0.999,
            "{}",
            g.reach()
        );
        assert!((g.inner_cell - 56.0 / 48.0).abs() < 1e-5);
        let n = i32::saturating_from(OCEAN_GRID_HALF_LINES);
        for i in -n..n {
            assert!(g.line(i + 1) > g.line(i), "{i}");
            assert_eq!(g.line(-i), -g.line(i));
        }
        let inner = i32::saturating_from(OCEAN_GRID_INNER_CELLS);
        for i in 0..inner {
            assert!((g.line(i + 1) - g.line(i) - g.inner_cell).abs() < 1e-4);
        }
        let last = g.line(n) - g.line(n - 1);
        let before = g.line(n - 1) - g.line(n - 2);
        assert!((last / before - g.growth).abs() < 1e-3);
    }

    // The shader's cell estimate stays within one step of growth of the cell
    // a vertex actually sits in.
    #[test]
    fn the_cell_estimate_tracks_the_grid() {
        let g = ocean();
        let n = i32::saturating_from(OCEAN_GRID_HALF_LINES);
        for i in 0..n {
            let cell = g.line(i + 1) - g.line(i);
            let mid = 0.5 * (g.line(i + 1) + g.line(i));
            let est = g.estimated_cell(mid);
            assert!(
                est <= cell * g.growth * 1.01 && est >= cell / g.growth * 0.99,
                "{i}: {est} vs {cell}"
            );
        }
    }

    // A surface the inner cells already span needs no widening, and a calm
    // sea's cells are held to their floor.
    #[test]
    fn small_seas_take_uniform_grids() {
        let g = OceanGrid::new(500.0, [100.0, 100.0]);
        assert_eq!(g.growth, 1.0);
        assert_eq!(g.inner_cell, 8.0);
        assert!(g.reach() >= 2.0 * 100.0_f32.hypot(100.0));
        assert_eq!(OceanGrid::new(4.0, [100.0, 100.0]).inner_cell, 0.25);
    }

    #[test]
    fn the_anchor_snaps_to_whole_inner_cells() {
        let g = ocean();
        let cell = f64::from(g.inner_cell);
        let a = g.snap([1.0e6 + 0.3, -2.0e5 - 0.6]);
        for v in a {
            let cells = v / cell;
            assert!((cells - cells.round()).abs() < 1e-6);
        }
        assert!((a[0] - (1.0e6 + 0.3)).abs() <= 0.5 * cell + 1e-9);
    }

    // The ring mesh holds every triangle of the row mesh once, and its first
    // `k` rings are exactly the cells within grid line `k` of the anchor.
    #[test]
    fn the_ring_mesh_reorders_the_triangles_into_nested_blocks() {
        let g = ocean();
        let (rows_v, rows) = g.mesh();
        let (rings_v, rings) = g.ring_mesh();
        assert_eq!(rows_v.len(), rings_v.len());
        let triangles = |idx: &[u16]| {
            let mut t: Vec<[u16; 3]> = idx.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
            t.sort_unstable();
            t
        };
        assert_eq!(triangles(&rows), triangles(&rings));
        assert_eq!(g.ring_indices(g.rings()) as usize, rings.len());
        for k in [0, 1, 2, 16, 40, g.rings()] {
            let prefix = &rings[..g.ring_indices(k) as usize];
            let half = g.line(i32::saturating_from(k));
            for &i in prefix {
                let p = rings_v[i as usize].pos;
                assert!(p[0].abs() <= half && p[2].abs() <= half, "ring {k}: {p:?}");
            }
            let outside = &rings[g.ring_indices(k) as usize..];
            for tri in outside.chunks_exact(3) {
                let inside = tri.iter().all(|&i| {
                    let p = rings_v[i as usize].pos;
                    p[0].abs() <= half && p[2].abs() <= half
                });
                assert!(
                    !inside,
                    "ring {k}: a triangle within the block is past its prefix"
                );
            }
        }
    }

    #[test]
    fn the_mesh_fits_a_u16_index() {
        let (verts, idx) = ocean().mesh();
        let cols = 2 * OCEAN_GRID_HALF_LINES as usize + 1;
        assert_eq!(verts.len(), cols * cols);
        assert!(verts.len() <= 65536);
        assert_eq!(idx.len(), (cols - 1) * (cols - 1) * 6);
        assert!(idx.iter().all(|&i| (i as usize) < verts.len()));
        assert_eq!(
            verts[0].pos[0],
            ocean().line(-i32::saturating_from(OCEAN_GRID_HALF_LINES))
        );
    }
}
