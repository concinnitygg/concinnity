// How a planet tile's vertices morph toward its parent's mesh near the end of
// their level's range: the row a tile's draw carries to the GPU, and the
// morph itself, which `tile_morph.hlsl` mirrors.
//
// A child tile's even grid corners are its parent's corners. Each odd corner
// moves toward the middle of the two neighbors that bound it on the parent's
// grid, across the parent cell's diagonal where both of its coordinates are
// odd, so a fully morphed tile is its parent's mesh.

use super::seams::TileSeams;
use super::tile::{TILE_CELLS, TileSide};
use crate::gfx::render_types::{GpuTileMorph, PlanetTileDraw};
use crate::math::narrow_f32;

/// The camera a planet's tiles morph by, this frame and the last, in
/// the simulated frame, and how far level 0 reaches from it: what every
/// tile's morph is measured from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileMorphFrame {
    /// The camera this frame.
    pub eye: [f32; 3],
    /// The camera last frame, in this frame's
    /// coordinates.
    pub prev_eye: [f32; 3],
    /// How far level 0 reaches, in meters; each level reaches half as far as
    /// the one above it.
    pub level0_range: f32,
    /// The coarsest level tiles are drawn at: no parent replaces a tile
    /// there, so it reaches without end and only its seams morph it.
    pub coarsest_level: u8,
}

impl TileMorphFrame {
    /// The morph row of a draw that is `tile`, its first vertex at
    /// `vertex_base` in the shared stream, or the still row of one that is
    /// not.
    pub fn row(&self, tile: Option<PlanetTileDraw>, vertex_base: u32) -> GpuTileMorph {
        let Some(tile) = tile else {
            return GpuTileMorph::default();
        };
        let range = if tile.level <= self.coarsest_level {
            UNBOUNDED_RANGE
        } else {
            libm::ldexpf(self.level0_range, -i32::from(tile.level))
        };
        GpuTileMorph {
            eye: self.eye,
            range,
            prev_eye: self.prev_eye,
            seams: packed_seams(tile.seams),
            vertex_base,
            _pad: [0; 3],
        }
    }
}

// A range no eye comes near the morph of, finite so the shader's arithmetic
// on it stays exact.
const UNBOUNDED_RANGE: f32 = 1e30;

// How many cells in from a seam side the side's pin fades out over, so the
// rows between a pinned edge and an interior morphed by distance do not fold.
const SEAM_RAMP_CELLS: f32 = 4.0;

/// The seams as [`GpuTileMorph::seams`] carries them.
pub fn packed_seams(seams: TileSeams) -> u32 {
    u32::from(seams.coarser & 0xf) | (u32::from(seams.finer & 0xf) << 4)
}

/// How far vertex `(i, j)` of a tile's grid has morphed, `distance` meters
/// from the eye of `row`: by distance across the last stretch of the range,
/// all the way on a side beside a coarser tile, and not at all on a side
/// beside finer ones, so the tile's edge matches its neighbor's there.
///
/// Inside the tile each pin fades out over a few cells from its
/// side. A corner on any edge takes no fade, so neighbors of one level, which
/// share it, morph it alike whatever their other sides meet.
pub fn morph_factor(row: &GpuTileMorph, i: usize, j: usize, distance: f32) -> f32 {
    if row.range <= 0.0 {
        return 0.0;
    }
    let start = narrow_f32(super::MORPH_START);
    let k = ((distance - start * row.range) / ((1.0 - start) * row.range)).clamp(0.0, 1.0);
    match side_of(i, j) {
        Some(side) if row.seams & u32::from(side.bit()) != 0 => 1.0,
        Some(side) if (row.seams >> 4) & u32::from(side.bit()) != 0 => 0.0,
        Some(_) => k,
        None if on_edge(i, j) => k,
        None => seam_ramp(row.seams, i, j, k),
    }
}

// Whether grid corner `(i, j)` lies on any of the tile's edges.
fn on_edge(i: usize, j: usize) -> bool {
    let n = TILE_CELLS as usize;
    i == 0 || j == 0 || i == n || j == n
}

// The morph `k` of interior corner `(i, j)` drawn toward each seam side's pin
// by how near the side it lies: up toward 1 beside a coarser tile, then down
// toward 0 beside finer ones.
fn seam_ramp(seams: u32, i: usize, j: usize, k: f32) -> f32 {
    let n = TILE_CELLS as usize;
    let away = |side: TileSide| match side {
        TileSide::MinS => i,
        TileSide::MaxS => n - i,
        TileSide::MinT => j,
        TileSide::MaxT => n - j,
    } as f32
        / SEAM_RAMP_CELLS;
    let mut k = k;
    for side in TileSide::ALL {
        if seams & u32::from(side.bit()) != 0 {
            k = k.max(1.0 - away(side));
        }
    }
    for side in TileSide::ALL {
        if (seams >> 4) & u32::from(side.bit()) != 0 {
            k = k.min(away(side));
        }
    }
    k.clamp(0.0, 1.0)
}

// The side an odd corner on a tile's edge lies on; `None` inside the tile and
// for the even corners, which never move.
fn side_of(i: usize, j: usize) -> Option<TileSide> {
    let n = TILE_CELLS as usize;
    match (i, j) {
        (0, j) if j % 2 == 1 => Some(TileSide::MinS),
        (i, j) if i == n && j % 2 == 1 => Some(TileSide::MaxS),
        (i, 0) if i % 2 == 1 => Some(TileSide::MinT),
        (i, j) if j == n && i % 2 == 1 => Some(TileSide::MaxT),
        _ => None,
    }
}

/// The two grid corners whose middle corner `(i, j)` morphs toward, or
/// `None` for an even corner, which is its parent's own.
pub fn parent_span(i: usize, j: usize) -> Option<[(usize, usize); 2]> {
    match (i % 2, j % 2) {
        (0, 0) => None,
        (1, 0) => Some([(i - 1, j), (i + 1, j)]),
        (0, 1) => Some([(i, j - 1), (i, j + 1)]),
        _ => Some([(i + 1, j - 1), (i - 1, j + 1)]),
    }
}

/// Where corner `(i, j)` of `grid`, a tile's `(TILE_CELLS + 1)^2` surface
/// points row-major along `t`, sits once morphed `k` of the way toward its
/// parent's mesh.
pub fn morph_point(grid: &[[f32; 3]], i: usize, j: usize, k: f32) -> [f32; 3] {
    let side = TILE_CELLS as usize + 1;
    let p = grid[j * side + i];
    let Some([a, b]) = parent_span(i, j) else {
        return p;
    };
    let (a, b) = (grid[a.1 * side + a.0], grid[b.1 * side + b.0]);
    core::array::from_fn(|c| p[c] + ((a[c] + b[c]) * 0.5 - p[c]) * k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::SaturatingFrom;
    use crate::math::dvec3;
    use crate::math::dvec3::DVec3;
    use crate::planet::shape::{CubeFace, PlanetShape};
    use crate::planet::tile::{TileId, tile_mesh};
    use alloc::vec::Vec;

    fn shape() -> PlanetShape {
        PlanetShape {
            center: [0.0, -50_000.0, 0.0],
            radius: 50_000.0,
            amplitude: 60.0,
            feature_size: 400.0,
            octaves: 8,
            seed: 9,
        }
    }

    // A tile's surface points in authored coordinates.
    fn surface(s: &PlanetShape, t: TileId) -> (DVec3, Vec<[f32; 3]>) {
        let mesh = tile_mesh(s, t);
        let side = TILE_CELLS as usize + 1;
        let grid = mesh.vertices[..side * side].iter().map(|v| v.pos).collect();
        (mesh.origin, grid)
    }

    // The point at `(fs, ft)` across a tile on its mesh: the triangle under
    // it, split along the same diagonal the mesh's indices split each cell.
    fn on_mesh(origin: DVec3, grid: &[[f32; 3]], fs: f64, ft: f64) -> DVec3 {
        let n = f64::from(TILE_CELLS);
        let side = TILE_CELLS as usize + 1;
        let (gs, gt) = (fs * n, ft * n);
        let (ci, cj) = (
            usize::saturating_from(gs.floor()).min(side - 2),
            usize::saturating_from(gt.floor()).min(side - 2),
        );
        let (rs, rt) = (gs - ci as f64, gt - cj as f64);
        let at = |i: usize, j: usize| dvec3::add(origin, dvec3::from_f32(grid[j * side + i]));
        let (p00, p10, p01, p11) = (
            at(ci, cj),
            at(ci + 1, cj),
            at(ci, cj + 1),
            at(ci + 1, cj + 1),
        );
        if rs + rt <= 1.0 {
            dvec3::add(
                p00,
                dvec3::add(
                    dvec3::scale(dvec3::sub(p10, p00), rs),
                    dvec3::scale(dvec3::sub(p01, p00), rt),
                ),
            )
        } else {
            dvec3::add(
                p11,
                dvec3::add(
                    dvec3::scale(dvec3::sub(p01, p11), 1.0 - rs),
                    dvec3::scale(dvec3::sub(p10, p11), 1.0 - rt),
                ),
            )
        }
    }

    // Fully morphed, each child's every corner lies on its parent's mesh;
    // unmorphed, every corner stays put.
    #[test]
    fn a_fully_morphed_child_is_its_parent_s_mesh() {
        let s = shape();
        let parent = TileId {
            face: CubeFace(2),
            level: 9,
            x: 200,
            y: 301,
        };
        let (p_origin, p_grid) = surface(&s, parent);
        let n = TILE_CELLS as usize;
        for (q, child) in parent.children().into_iter().enumerate() {
            let (c_origin, c_grid) = surface(&s, child);
            let (ox, oy) = ((q % 2) as f64 * 0.5, (q / 2) as f64 * 0.5);
            let mut moved: f64 = 0.0;
            for j in 0..=n {
                for i in 0..=n {
                    let still = morph_point(&c_grid, i, j, 0.0);
                    assert_eq!(still, c_grid[j * (n + 1) + i]);
                    let p = dvec3::add(c_origin, dvec3::from_f32(morph_point(&c_grid, i, j, 1.0)));
                    let at = (
                        ox + i as f64 / n as f64 * 0.5,
                        oy + j as f64 / n as f64 * 0.5,
                    );
                    let want = on_mesh(p_origin, &p_grid, at.0, at.1);
                    assert!(
                        dvec3::length(dvec3::sub(p, want)) < 2e-3,
                        "{child:?} ({i}, {j})"
                    );
                    let raw = dvec3::add(c_origin, dvec3::from_f32(still));
                    moved = moved.max(dvec3::length(dvec3::sub(p, raw)));
                }
            }
            assert!(moved > 0.01, "the morph moves the odd corners: {moved}");
        }
    }

    // The shaders walk a tile's mesh with the cell count it is built with and
    // start the morph where the CPU does.
    #[test]
    fn the_shader_morph_shares_the_cell_count_and_start() {
        let src = crate::render::shader::embedded::TILE_MORPH;
        let uint = |name| crate::render::shader::consts::uint(src, name);
        assert_eq!(uint("PLANET_TILE_CELLS"), TILE_CELLS as usize);
        let start = crate::render::shader::consts::float(src, "TILE_MORPH_START");
        assert_eq!(start, narrow_f32(crate::planet::MORPH_START));
        let ramp = crate::render::shader::consts::float(src, "TILE_SEAM_RAMP_CELLS");
        assert_eq!(ramp, SEAM_RAMP_CELLS);
    }

    #[test]
    fn even_corners_are_their_parent_s_own() {
        assert_eq!(parent_span(4, 6), None);
        assert_eq!(parent_span(3, 6), Some([(2, 6), (4, 6)]));
        assert_eq!(parent_span(4, 7), Some([(4, 6), (4, 8)]));
        assert_eq!(parent_span(3, 7), Some([(4, 6), (2, 8)]));
    }

    fn row(range: f32, seams: TileSeams) -> GpuTileMorph {
        GpuTileMorph {
            eye: [0.0; 3],
            range,
            prev_eye: [0.0; 3],
            seams: packed_seams(seams),
            vertex_base: 0,
            _pad: [0; 3],
        }
    }

    #[test]
    fn the_factor_runs_over_the_end_of_the_range() {
        let r = row(100.0, TileSeams::default());
        let start = narrow_f32(crate::planet::MORPH_START) * 100.0;
        assert_eq!(morph_factor(&r, 3, 5, 0.0), 0.0);
        assert_eq!(morph_factor(&r, 3, 5, start), 0.0);
        assert!((morph_factor(&r, 3, 5, (start + 100.0) * 0.5) - 0.5).abs() < 1e-5);
        assert_eq!(morph_factor(&r, 3, 5, 100.0), 1.0);
        assert_eq!(
            morph_factor(&row(0.0, TileSeams::default()), 3, 5, 1e6),
            0.0
        );
    }

    // On a side beside a coarser tile the edge is all the way morphed, beside
    // finer tiles not at all, wherever the eye is; inside, distance decides.
    #[test]
    fn seams_pin_the_edge_corners() {
        let n = TILE_CELLS as usize;
        let seams = TileSeams {
            coarser: TileSide::MinS.bit() | TileSide::MaxT.bit(),
            finer: TileSide::MaxS.bit(),
        };
        let r = row(100.0, seams);
        let mid = (narrow_f32(crate::planet::MORPH_START) * 100.0 + 100.0) * 0.5;
        assert_eq!(morph_factor(&r, 0, 3, 0.0), 1.0);
        assert_eq!(morph_factor(&r, 5, n, 0.0), 1.0);
        assert_eq!(morph_factor(&r, n, 3, 1e6), 0.0);
        let k = morph_factor(&r, 3, 0, mid);
        assert!(k > 0.0 && k < 1.0, "an unpinned side follows distance");
        let k = morph_factor(&r, 9, 9, mid);
        assert!(k > 0.0 && k < 1.0, "the inside follows distance");
    }

    // Inside, a pin fades out row by row from its side, so no row folds
    // against the next; corners on an edge never take the fade.
    #[test]
    fn a_pin_fades_inward_from_its_side() {
        let n = TILE_CELLS as usize;
        let ramp = usize::saturating_from(SEAM_RAMP_CELLS);
        let near = row(
            100.0,
            TileSeams {
                coarser: TileSide::MinS.bit(),
                finer: TileSide::MaxS.bit(),
            },
        );
        // The eye at the tile: distance alone would leave every corner still.
        let fades: Vec<f32> = (0..=n).map(|i| morph_factor(&near, i, 7, 0.0)).collect();
        assert_eq!(fades[0], 1.0);
        for i in 1..ramp {
            assert!(fades[i] < fades[i - 1] && fades[i] > 0.0, "{fades:?}");
        }
        assert!(
            fades[ramp..=n - ramp].iter().all(|&k| k == 0.0),
            "{fades:?}"
        );
        // Far out, the morph by distance reaches 1, and the finer side pulls
        // its rows back down to its unmorphed edge.
        let far: Vec<f32> = (0..=n).map(|i| morph_factor(&near, i, 7, 1e6)).collect();
        assert_eq!(far[n], 0.0);
        for i in (n - ramp + 1)..n {
            assert!(far[i] < far[i - 1] && far[i] < 1.0, "{far:?}");
        }
        // An odd corner on an unpinned edge follows distance alone.
        assert_eq!(morph_factor(&near, 3, 0, 0.0), 0.0);
        assert_eq!(morph_factor(&near, 3, n, 1e6), 1.0);
    }

    #[test]
    fn a_row_carries_its_level_s_range_and_seams() {
        let frame = TileMorphFrame {
            eye: [1.0, 2.0, 3.0],
            prev_eye: [4.0, 5.0, 6.0],
            level0_range: 1024.0,
            coarsest_level: 2,
        };
        let seams = TileSeams {
            coarser: 0b0101,
            finer: 0b0010,
        };
        let row = frame.row(Some(PlanetTileDraw { level: 3, seams }), 77);
        assert_eq!(row.eye, [1.0, 2.0, 3.0]);
        assert_eq!(row.prev_eye, [4.0, 5.0, 6.0]);
        assert_eq!(row.range, 128.0);
        assert_eq!(row.seams, 0b0010_0101);
        assert_eq!(row.vertex_base, 77);
        assert_eq!(frame.row(None, 77), GpuTileMorph::default());
    }

    // A tile at the coarsest drawn level never morphs by distance, however
    // far, but still pins its edge beside a coarser tile drawn while its
    // own children load.
    #[test]
    fn the_coarsest_level_morphs_only_at_its_seams() {
        let frame = TileMorphFrame {
            eye: [0.0; 3],
            prev_eye: [0.0; 3],
            level0_range: 1024.0,
            coarsest_level: 2,
        };
        let seams = TileSeams {
            coarser: TileSide::MinS.bit(),
            finer: 0,
        };
        let row = frame.row(Some(PlanetTileDraw { level: 2, seams }), 0);
        assert!(row.range > 0.0);
        assert_eq!(morph_factor(&row, 5, 3, 1e9), 0.0);
        assert_eq!(morph_factor(&row, 0, 3, 1e9), 1.0);
        let finer = frame.row(Some(PlanetTileDraw { level: 3, seams }), 0);
        assert_eq!(morph_factor(&finer, 5, 3, 1e9), 1.0);
    }
}
