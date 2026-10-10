//! Which of a planet's drawn tiles a frame grows blades on: those within
//! reach of the camera, each with the block of its grass tiles the reach
//! covers, as the records the kernel reads. The kernel roots every blade on
//! its tile's own mesh, read from the shared vertex buffer through the tile's
//! model matrix, so the blades stand exactly on the ground the tile draws, in
//! whatever frame the world is simulated in.

use alloc::vec::Vec;

use super::cells::PlanetGrassGrid;
use crate::planet::dvec;
use crate::planet::{PlanetShape, TileId};
use crate::render::uniforms::grass::{MAX_PLANET_GRASS_TILES, PlanetGrassTileGpu};
use crate::transform::Mat4;

/// One planet tile as it is drawn this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawnPlanetTile {
    /// Which square of the planet it covers.
    pub id: TileId,
    /// Its model matrix, column-major: the one its draw renders with.
    pub model: Mat4,
    /// Index of its first vertex in the shared vertex buffer.
    pub vertex_base: u32,
    /// The lowest and highest its surface lies along the up at its origin, in
    /// meters from the origin.
    pub radial: [f32; 2],
}

/// One run's records and the kernel group rows they span.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlanetGrassTiles {
    /// The tiles, nearest the camera first.
    pub records: Vec<PlanetGrassTileGpu>,
    /// Kernel group rows (dispatch `y`) the records span together.
    pub groups: u32,
}

// The rotation part of `model` applied to `v`.
fn rotate(model: &Mat4, v: [f64; 3]) -> [f64; 3] {
    core::array::from_fn(|r| (0..3).map(|c| f64::from(model[c][r]) * v[c]).sum())
}

fn translation(model: &Mat4) -> [f64; 3] {
    core::array::from_fn(|r| f64::from(model[3][r]))
}

/// The planet's center in the frame `tile` is drawn in.
pub fn planet_center(shape: &PlanetShape, tile: &DrawnPlanetTile) -> [f32; 3] {
    let dir = tile.id.direction(0.5, 0.5);
    let to_origin = dvec::scale(dir, shape.radius + shape.height(dir));
    dvec::to_f32(dvec::sub(
        translation(&tile.model),
        rotate(&tile.model, to_origin),
    ))
}

// How far from its origin any point of a tile at `level` can lie, in meters.
fn tile_reach(shape: &PlanetShape, grid: &PlanetGrassGrid, level: u8) -> f64 {
    let width = shape.radius * core::f64::consts::FRAC_PI_2 / f64::from(1u32 << level);
    width * libm::sqrt(grid.max_stretch) * 0.8 + shape.amplitude
}

/// The drawn tiles within `distance` of `camera` (both in the simulated
/// frame, around the planet's `center` there), nearest first and at most
/// [`MAX_PLANET_GRASS_TILES`], each with the block of grass tiles the reach
/// covers. A block is a square around the camera in the tile's own face
/// coordinates, a grass tile wider than the reach on every side; the kernel
/// culls what lies outside the reach itself.
pub fn tiles_in_reach(
    shape: &PlanetShape,
    grid: &PlanetGrassGrid,
    drawn: &[DrawnPlanetTile],
    center: [f32; 3],
    camera: [f32; 3],
    distance: f32,
) -> PlanetGrassTiles {
    let near = records_in_reach(shape, grid, drawn, center, camera, distance);
    let mut groups = 0u32;
    let records = near
        .into_iter()
        .take(MAX_PLANET_GRASS_TILES)
        .map(|(_, mut r)| {
            r.first_group = groups;
            groups += (r.block_count & 0xffff) * (r.block_count >> 16);
            r
        })
        .collect();
    PlanetGrassTiles { records, groups }
}

// Every record within reach with its distance, nearest first.
fn records_in_reach(
    shape: &PlanetShape,
    grid: &PlanetGrassGrid,
    drawn: &[DrawnPlanetTile],
    center: [f32; 3],
    camera: [f32; 3],
    distance: f32,
) -> Vec<(f64, PlanetGrassTileGpu)> {
    let cam = dvec::from_f32(camera);
    let center = dvec::from_f32(center);
    let mut near: Vec<(f64, PlanetGrassTileGpu)> = drawn
        .iter()
        .filter_map(|tile| {
            let origin = translation(&tile.model);
            let gap = dvec::length(dvec::sub(origin, cam));
            if gap > f64::from(distance) + tile_reach(shape, grid, tile.id.level) {
                return None;
            }
            let record = block(grid, tile, center, cam, f64::from(distance))?;
            Some((gap, record))
        })
        .collect();
    near.sort_by(|a, b| a.0.total_cmp(&b.0));
    near
}

// `tile`'s record with the block of grass tiles within `distance` of `cam`,
// `None` when the block is empty. The tile is taken as the parallelogram its
// corners span on the sphere through its origin, which is close enough to
// find the block: the block is grown a grass tile on every side.
fn block(
    grid: &PlanetGrassGrid,
    tile: &DrawnPlanetTile,
    center: [f64; 3],
    cam: [f64; 3],
    distance: f64,
) -> Option<PlanetGrassTileGpu> {
    let origin = translation(&tile.model);
    let rho = dvec::length(dvec::sub(origin, center));
    let corner = |fs: f64, ft: f64| {
        dvec::add(
            center,
            rotate(&tile.model, dvec::scale(tile.id.direction(fs, ft), rho)),
        )
    };
    let p00 = corner(0.0, 0.0);
    let es = dvec::sub(corner(1.0, 0.0), p00);
    let et = dvec::sub(corner(0.0, 1.0), p00);
    let q = dvec::sub(cam, p00);
    let (ss, st, tt) = (dvec::dot(es, es), dvec::dot(es, et), dvec::dot(et, et));
    let det = ss * tt - st * st;
    if det <= 0.0 {
        return None;
    }
    let (qs, qt) = (dvec::dot(q, es), dvec::dot(q, et));
    let u = (qs * tt - qt * st) / det;
    let v = (qt * ss - qs * st) / det;
    let n = grid.tiles_per_side(tile.id.level);
    let span = |at: f64, len: f64| -> Option<(u32, u32)> {
        let reach = distance / len.max(1e-6);
        let lo = (libm::floor((at - reach) * f64::from(n)) - 1.0).max(0.0);
        let hi = (libm::ceil((at + reach) * f64::from(n)) + 1.0).min(f64::from(n));
        (lo < hi).then_some((lo as u32, (hi - lo) as u32))
    };
    let (x0, xn) = span(u, libm::sqrt(ss))?;
    let (y0, yn) = span(v, libm::sqrt(tt))?;
    let m = &tile.model;
    Some(PlanetGrassTileGpu {
        model: core::array::from_fn(|r| [m[0][r], m[1][r], m[2][r], m[3][r]]),
        vertex_base: tile.vertex_base,
        face_level: u32::from(tile.id.face.0) | (u32::from(tile.id.level) << 8),
        tile: [tile.id.x, tile.id.y],
        first_group: 0,
        block_origin: x0 | (y0 << 16),
        block_count: xn | (yn << 16),
        grass_tiles: n,
        radial: tile.radial,
        _pad: [0.0; 2],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::{CubeFace, LocalFrame, TileLod, tile_mesh};
    use crate::render::grass::tiles::GRASS_DRAW_DISTANCE;

    fn shape() -> PlanetShape {
        PlanetShape {
            center: [0.0, -50_000.0, 0.0],
            radius: 50_000.0,
            amplitude: 40.0,
            feature_size: 2_000.0,
            octaves: 8,
            seed: 4,
        }
    }

    fn drawn(shape: &PlanetShape, frame: &LocalFrame, id: TileId) -> DrawnPlanetTile {
        let mesh = tile_mesh(shape, id);
        DrawnPlanetTile {
            id,
            model: frame.model_at(mesh.origin),
            vertex_base: 0,
            radial: mesh.radial,
        }
    }

    // A frame moved a kilometer over and tilted onto the radial up there.
    fn moved() -> LocalFrame {
        let up = |p: [f32; 3]| {
            let w = LocalFrame::AUTHORED.to_world(p);
            dvec::normalize_or(dvec::sub(w, shape().center), [0.0, 1.0, 0.0])
        };
        let at = [700.0, 30.0, -714.0];
        LocalFrame::AUTHORED.recentered(at, up(at)).0
    }

    #[test]
    fn the_center_is_where_the_frame_puts_it() {
        let s = shape();
        for frame in [LocalFrame::AUTHORED, moved()] {
            let id = TileId {
                face: CubeFace(2),
                level: 10,
                x: 511,
                y: 512,
            };
            let c = planet_center(&s, &drawn(&s, &frame, id));
            let expected = frame.to_local(s.center);
            for i in 0..3 {
                assert!((c[i] - expected[i]).abs() < 0.02, "{c:?} {expected:?}");
            }
        }
    }

    // With every tile under a standing camera drawn, the run covers the
    // tiles within reach, nearest first, every block inside its tile, and
    // the group rows count the blocks' grass tiles.
    #[test]
    fn the_run_covers_the_tiles_within_reach() {
        let s = shape();
        let grid = PlanetGrassGrid::new(&s);
        let frame = LocalFrame::AUTHORED;
        let lod = TileLod::new(s, crate::planet::FINEST_CELL_SIZE);
        let cam_world = dvec::add(
            s.center,
            [0.0, s.radius + s.height([0.0, 1.0, 0.0]) + 1.7, 0.0],
        );
        let sel = lod.select(cam_world, &|_| true);
        let tiles: Vec<DrawnPlanetTile> =
            sel.draw.iter().map(|&id| drawn(&s, &frame, id)).collect();
        let center = frame.to_local(s.center);
        let cam = frame.to_local(cam_world);
        let run = tiles_in_reach(&s, &grid, &tiles, center, cam, 90.0);
        assert!(!run.records.is_empty());
        assert!(
            run.records.len() < MAX_PLANET_GRASS_TILES,
            "{}",
            run.records.len()
        );
        let mut groups = 0;
        for r in &run.records {
            assert_eq!(r.first_group, groups);
            let (x0, y0) = (r.block_origin & 0xffff, r.block_origin >> 16);
            let (xn, yn) = (r.block_count & 0xffff, r.block_count >> 16);
            assert!(xn > 0 && yn > 0);
            assert!(x0 + xn <= r.grass_tiles && y0 + yn <= r.grass_tiles);
            groups += xn * yn;
        }
        assert_eq!(run.groups, groups);
        // About the reach's square of four-meter grass tiles, not the tiles'
        // whole area.
        let square = (2.0 * 90.0 / grid.tile_size()) as u32;
        assert!(run.groups > square * square / 2, "{}", run.groups);
        assert!(run.groups < square * square * 3, "{}", run.groups);
        // A tile far beyond the reach grows nothing.
        let far = drawn(
            &s,
            &frame,
            TileId {
                face: CubeFace(3),
                level: 4,
                x: 3,
                y: 3,
            },
        );
        assert!(
            tiles_in_reach(&s, &grid, &[far], center, cam, 90.0)
                .records
                .is_empty()
        );
        // High above the ground nothing is in reach.
        let aloft = [cam[0], cam[1] + 400.0, cam[2]];
        assert!(
            tiles_in_reach(&s, &grid, &tiles, center, aloft, 90.0)
                .records
                .is_empty()
        );
    }

    // The same tiles in a moved frame cover the same grass tiles: what the
    // run covers is the planet's, not the frame's.
    #[test]
    fn a_moved_frame_covers_the_same_grass_tiles() {
        let s = shape();
        let grid = PlanetGrassGrid::new(&s);
        let a = LocalFrame::AUTHORED;
        let b = moved();
        let parent = TileId {
            face: CubeFace(2),
            level: 10,
            x: 521,
            y: 521,
        };
        let dir = parent.direction(0.5, 0.5);
        let cam_world = dvec::add(s.surface(dir), dvec::scale(dir, 1.7));
        let run = |frame: &LocalFrame| {
            let tiles: Vec<DrawnPlanetTile> = parent
                .children()
                .iter()
                .map(|&c| drawn(&s, frame, c))
                .collect();
            let center = frame.to_local(s.center);
            tiles_in_reach(&s, &grid, &tiles, center, frame.to_local(cam_world), 40.0)
        };
        let (ra, rb) = (run(&a), run(&b));
        let blocks = |r: &PlanetGrassTiles| {
            let mut v: Vec<_> = r
                .records
                .iter()
                .map(|t| (t.tile, t.block_origin, t.block_count))
                .collect();
            v.sort();
            v
        };
        assert_eq!(blocks(&ra), blocks(&rb));
    }

    // Every tile at the finest level whose bounds come within `reach` of
    // `camera`, across faces.
    fn finest_around(s: &PlanetShape, max_level: u8, camera: [f64; 3], reach: f64) -> Vec<TileId> {
        let mut out = Vec::new();
        let mut stack: Vec<TileId> = (0..6).map(|f| TileId::root(CubeFace(f))).collect();
        while let Some(t) = stack.pop() {
            let b = crate::planet::tile_bounds(s, t);
            if dvec::length(dvec::sub(camera, b.center)) - b.radius > reach {
                continue;
            }
            if t.level >= max_level {
                out.push(t);
            } else {
                stack.extend(t.children());
            }
        }
        out
    }

    // The cap holds every tile a run can reach, whatever the planet. A tile
    // at the deepest level is over 16 m wide (its cells are over half the
    // finest cell size), and a run covers a square about 196 m across in each
    // tile's face coordinates, so at most 14 tiles lie across it each way.
    // The worst planet's deepest tiles are just over 16 m wide; tall hills
    // split every tile near the camera to the deepest level, and every tile
    // drawn there is the bound for any mix of levels.
    #[test]
    fn the_cap_holds_every_tile_a_run_can_reach() {
        let mut worst = 0;
        for (level, amplitude) in [(8u8, 40.0), (12, 40.0), (12, 2000.0), (16, 40.0)] {
            let width = 16.001 * crate::planet::FINEST_CELL_SIZE;
            let radius = width * f64::from(1u32 << level) / core::f64::consts::FRAC_PI_2;
            let s = PlanetShape {
                center: [0.0; 3],
                radius,
                amplitude,
                feature_size: 2_000.0,
                octaves: 8,
                seed: 4,
            };
            let grid = PlanetGrassGrid::new(&s);
            let lod = TileLod::new(s, crate::planet::FINEST_CELL_SIZE);
            assert_eq!(lod.max_level, level);
            let frame = LocalFrame::AUTHORED;
            let light = |id: TileId| DrawnPlanetTile {
                id,
                model: frame.model_at(s.surface(id.direction(0.5, 0.5))),
                vertex_base: 0,
                radial: [0.0; 2],
            };
            let step = 1.0 / f64::from(1u32 << level);
            // The face's middle, part way across a tile, beside an edge, at a
            // cube corner and beside one.
            for (fs, ft) in [
                (0.0, 0.0),
                (0.25 * step, 0.0),
                (0.5 * step, 0.75 * step),
                (0.5, 0.5),
                (0.999, 0.0),
                (1.0, 1.0),
                (-0.999, 0.999),
            ] {
                let dir = CubeFace(2).direction(fs, ft);
                let cam_world = dvec::add(s.surface(dir), dvec::scale(dir, 1.7));
                let center = frame.to_local(s.center);
                let cam = frame.to_local(cam_world);
                let finest: Vec<_> = finest_around(&s, level, cam_world, 200.0)
                    .into_iter()
                    .map(light)
                    .collect();
                let selected: Vec<_> = lod
                    .select(cam_world, &|_| true)
                    .draw
                    .iter()
                    .map(|&id| light(id))
                    .collect();
                for drawn in [&finest, &selected] {
                    let n =
                        records_in_reach(&s, &grid, drawn, center, cam, GRASS_DRAW_DISTANCE).len();
                    worst = worst.max(n);
                }
            }
        }
        assert!(worst <= 14 * 14, "{worst}");
        assert!(MAX_PLANET_GRASS_TILES >= worst + worst / 4, "{worst}");
    }
}
