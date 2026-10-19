//! Which of a planet's drawn tiles a frame grows blades on: those within
//! reach of the camera, each with the block of its grass tiles the reach
//! covers, as the records the kernel reads. The kernel roots every blade on
//! its tile's own mesh, read from the shared vertex buffer through the tile's
//! model matrix, so the blades stand exactly on the ground the tile draws, in
//! whatever frame the world is simulated in.

use crate::math::{SaturatingFrom, dvec3, narrow_f32};
use alloc::vec::Vec;

use super::cells::PlanetGrassGrid;
use crate::gfx::render_types::GpuTileMorph;
use crate::planet::{PlanetShape, TileId};
use crate::render::grass::nest::GRASS_NEST_LEVELS;
use crate::render::grass::tiles::{GrassBox, NestView, NestedTiles, nested_tiles};
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
    /// How its vertices morph toward its parent's mesh.
    pub morph: GpuTileMorph,
    /// The farthest any of its corners moves when it morphs all the way, in
    /// meters.
    pub morph_extent: f32,
}

/// The farthest a planet's blades reach, in meters: [`MAX_PLANET_GRASS_TILES`]
/// holds every drawn tile a run this far can reach.
pub const MAX_PLANET_GRASS_REACH: f32 = 707.0;

/// The drawn tiles one run grows grass on, each with the square of its grass
/// tiles of level 0 the reach covers, and the working space finding them
/// reuses from one run to the next.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlanetGrassTiles {
    /// The tiles, nearest the camera first.
    pub records: Vec<PlanetGrassTileGpu>,
    /// For each record, the drawn tile it was made from and its block: the
    /// first grass tile of level 0 and how many across and down.
    pub blocks: Vec<(usize, [u32; 2], [u32; 2])>,
    /// Whether more tiles were in reach than [`MAX_PLANET_GRASS_TILES`], the
    /// farthest of them dropped.
    pub clipped: bool,
    near: Vec<InReach>,
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
    let to_origin = dvec3::scale(dir, shape.radius + shape.height(dir));
    dvec3::to_f32(dvec3::sub(
        translation(&tile.model),
        rotate(&tile.model, to_origin),
    ))
}

// How far from its origin any point of a tile at `level` can lie, in meters.
fn tile_reach(shape: &PlanetShape, grid: &PlanetGrassGrid, level: u8) -> f64 {
    let width = shape.radius * core::f64::consts::FRAC_PI_2 / f64::from(1u32 << level);
    width * libm::sqrt(grid.max_stretch) * 0.8 + shape.amplitude
}

// How far a blade's root may sit from the ground its tile draws, in meters:
// a tile whose morph moves no corner farther roots its grass on its raw mesh.
const GRASS_MORPH_TOLERANCE: f32 = 0.01;

// How far from its origin any point of `tile` can lie, in meters: its width
// as `tile_reach` takes it, and its own surface's height range.
fn morph_reach(shape: &PlanetShape, grid: &PlanetGrassGrid, tile: &DrawnPlanetTile) -> f64 {
    let width = shape.radius * core::f64::consts::FRAC_PI_2 / f64::from(1u32 << tile.id.level);
    let height = tile.radial[0].abs().max(tile.radial[1].abs());
    width * libm::sqrt(grid.max_stretch) * 0.8 + f64::from(height)
}

// Whether no vertex of a tile with morph row `morph`, every point of it
// within `reach` of `origin`, has begun to morph: it meets no tile of another
// level, and all of it lies short of where its level's morph starts.
fn unmorphed(morph: &GpuTileMorph, origin: [f64; 3], reach: f64) -> bool {
    if morph.range <= 0.0 {
        return true;
    }
    let start = crate::planet::MORPH_START * f64::from(morph.range);
    let gap = dvec3::length(dvec3::sub(origin, dvec3::from_f32(morph.eye)));
    morph.seams == 0 && gap + reach < start
}

impl PlanetGrassTiles {
    /// Become the drawn tiles within `distance` of `camera` (both in the
    /// simulated frame, around the planet's `center` there), nearest first
    /// and at most [`MAX_PLANET_GRASS_TILES`], each with the block of grass
    /// tiles the reach covers. A block is a square around the camera in the
    /// tile's own face coordinates, a grass tile wider than the reach on every
    /// side; the nested tiles within it are culled against the reach itself
    /// (see [`nested_planet_tiles`]).
    pub fn gather(
        &mut self,
        (shape, grid): (&PlanetShape, &PlanetGrassGrid),
        drawn: &[DrawnPlanetTile],
        center: [f32; 3],
        camera: [f32; 3],
        distance: f32,
    ) {
        records_in_reach(
            (shape, grid),
            drawn,
            [center, camera],
            distance,
            &mut self.near,
        );
        self.records.clear();
        self.blocks.clear();
        self.clipped = self.near.len() > MAX_PLANET_GRASS_TILES;
        for &(_, index, record, block) in self.near.iter().take(MAX_PLANET_GRASS_TILES) {
            self.records.push(record);
            self.blocks.push((index, block[0], block[1]));
        }
    }
}

/// The nested grass tiles of planet tile `tile`, record `record` of its run,
/// that a layer of blades `width` meters wide reaching `reach` meters from
/// their cells visits, within `block` (its first grass tile of level 0 and
/// how many across and down), appended to `out.tiles`. A grass tile never crosses
/// its drawn tile, so the drawn tile caps the levels at the ones that divide
/// its grass tiles evenly.
pub fn nested_planet_tiles(
    view: &NestView<'_>,
    (shape, grid, center): (&PlanetShape, &PlanetGrassGrid, [f32; 3]),
    tile: &DrawnPlanetTile,
    (origin, count): ([u32; 2], [u32; 2]),
    [reach, width]: [f32; 2],
    out: &mut NestedTiles,
) {
    let n = grid.tiles_per_side(tile.id.level);
    let top = n.trailing_zeros().min(GRASS_NEST_LEVELS);
    let first = origin.map(|o| o >> top);
    let last: [u32; 2] = core::array::from_fn(|i| (origin[i] + count[i] - 1) >> top);
    let mut roots = (first[1]..=last[1]).flat_map(|y| {
        (first[0]..=last[0]).map(move |x| [i32::saturating_from(x), i32::saturating_from(y)])
    });
    let model = &tile.model;
    let o = translation(model);
    let rho = dvec3::length(dvec3::sub(o, dvec3::from_f32(center)));
    let bounds = |level: u32, t: [i32; 2]| -> Option<GrassBox> {
        let per = n >> level;
        if t[0] < 0
            || t[1] < 0
            || u32::saturating_from(t[0]) >= per
            || u32::saturating_from(t[1]) >= per
        {
            return None;
        }
        let f = |c: i32| f64::from(c) / f64::from(per);
        let edge = shape.radius * core::f64::consts::FRAC_PI_2
            / f64::from(1u32 << tile.id.level)
            / f64::from(per);
        // The patch bulges past the box of its corners by the sagitta of its
        // diagonal; the drawn mesh's own heights lie within its radial range.
        let slack = edge * edge * grid.max_stretch / (4.0 * shape.radius) + 0.5;
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for (fs, ft) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let dir = rotate(model, tile.id.direction(f(t[0] + fs), f(t[1] + ft)));
            for r in [
                rho + f64::from(tile.radial[0]) - slack,
                rho + f64::from(tile.radial[1]) + slack,
            ] {
                let p = dvec3::add(dvec3::from_f32(center), dvec3::scale(dir, r));
                for i in 0..3 {
                    lo[i] = lo[i].min(p[i]);
                    hi[i] = hi[i].max(p[i]);
                }
            }
        }
        let grow = f64::from(reach) + slack;
        Some((
            core::array::from_fn(|i| narrow_f32(lo[i] - grow)),
            core::array::from_fn(|i| narrow_f32(hi[i] + grow)),
        ))
    };
    nested_tiles(view, width, (top, &mut roots), &bounds, out);
}

// A record within reach: its distance, its drawn tile, the record and its
// block of grass tiles.
type InReach = (f64, usize, PlanetGrassTileGpu, [[u32; 2]; 2]);

// Every record within reach, nearest first, into `near`.
fn records_in_reach(
    (shape, grid): (&PlanetShape, &PlanetGrassGrid),
    drawn: &[DrawnPlanetTile],
    [center, camera]: [[f32; 3]; 2],
    distance: f32,
    near: &mut Vec<InReach>,
) {
    let cam = dvec3::from_f32(camera);
    let center = dvec3::from_f32(center);
    near.clear();
    near.extend(drawn.iter().enumerate().filter_map(|(index, tile)| {
        let origin = translation(&tile.model);
        let gap = dvec3::length(dvec3::sub(origin, cam));
        if gap > f64::from(distance) + tile_reach(shape, grid, tile.id.level) {
            return None;
        }
        let (mut record, block) = block(grid, tile, center, cam, f64::from(distance))?;
        if tile.morph_extent < GRASS_MORPH_TOLERANCE
            || unmorphed(&tile.morph, origin, morph_reach(shape, grid, tile))
        {
            record.tile_morph = GpuTileMorph::default();
        }
        Some((gap, index, record, block))
    }));
    near.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
}

// `tile`'s record and the block of its grass tiles within `distance` of
// `cam` (first and count), `None` when the block is empty. The tile is taken as the parallelogram its
// corners span on the sphere through its origin, which is close enough to
// find the block: the block is grown a grass tile on every side.
fn block(
    grid: &PlanetGrassGrid,
    tile: &DrawnPlanetTile,
    center: [f64; 3],
    cam: [f64; 3],
    distance: f64,
) -> Option<(PlanetGrassTileGpu, [[u32; 2]; 2])> {
    let origin = translation(&tile.model);
    let rho = dvec3::length(dvec3::sub(origin, center));
    let corner = |fs: f64, ft: f64| {
        dvec3::add(
            center,
            rotate(&tile.model, dvec3::scale(tile.id.direction(fs, ft), rho)),
        )
    };
    let p00 = corner(0.0, 0.0);
    let es = dvec3::sub(corner(1.0, 0.0), p00);
    let et = dvec3::sub(corner(0.0, 1.0), p00);
    let q = dvec3::sub(cam, p00);
    let (ss, st, tt) = (dvec3::dot(es, es), dvec3::dot(es, et), dvec3::dot(et, et));
    let det = ss * tt - st * st;
    if det <= 0.0 {
        return None;
    }
    let (qs, qt) = (dvec3::dot(q, es), dvec3::dot(q, et));
    let u = (qs * tt - qt * st) / det;
    let v = (qt * ss - qs * st) / det;
    let n = grid.tiles_per_side(tile.id.level);
    let span = |at: f64, len: f64| -> Option<(u32, u32)> {
        let reach = distance / len.max(1e-6);
        let lo = (libm::floor((at - reach) * f64::from(n)) - 1.0).max(0.0);
        let hi = (libm::ceil((at + reach) * f64::from(n)) + 1.0).min(f64::from(n));
        (lo < hi).then_some((u32::saturating_from(lo), u32::saturating_from(hi - lo)))
    };
    let (x0, xn) = span(u, libm::sqrt(ss))?;
    let (y0, yn) = span(v, libm::sqrt(tt))?;
    let m = &tile.model;
    let record = PlanetGrassTileGpu {
        model: core::array::from_fn(|r| [m[0][r], m[1][r], m[2][r], m[3][r]]),
        vertex_base: tile.vertex_base,
        face_level: u32::from(tile.id.face.0) | (u32::from(tile.id.level) << 8),
        tile: [tile.id.x, tile.id.y],
        grass_tiles: n,
        _pad: 0,
        radial: tile.radial,
        tile_morph: tile.morph,
    };
    Some((record, [[x0, y0], [xn, yn]]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::{CubeFace, LocalFrame, TileLod, tile_mesh};

    fn tiles_in_reach(
        shape: &PlanetShape,
        grid: &PlanetGrassGrid,
        drawn: &[DrawnPlanetTile],
        center: [f32; 3],
        camera: [f32; 3],
        distance: f32,
    ) -> PlanetGrassTiles {
        let mut run = PlanetGrassTiles::default();
        run.gather((shape, grid), drawn, center, camera, distance);
        run
    }

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
            morph: GpuTileMorph::default(),
            morph_extent: 0.0,
        }
    }

    // A frame moved a kilometer over and tilted onto the radial up there.
    fn moved() -> LocalFrame {
        let up = |p: [f32; 3]| {
            let w = LocalFrame::AUTHORED.to_world(p);
            dvec3::normalize_or(dvec3::sub(w, shape().center), [0.0, 1.0, 0.0])
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
        let cam_world = dvec3::add(
            s.center,
            [0.0, s.radius + s.height([0.0, 1.0, 0.0]) + 1.7, 0.0],
        );
        let sel = lod.select(
            &crate::planet::TileView::nearest(cam_world),
            &|_| true,
            &mut crate::planet::TileBoundsCache::new(s),
        );
        let tiles: Vec<DrawnPlanetTile> =
            sel.draw.iter().map(|d| drawn(&s, &frame, d.id)).collect();
        let center = frame.to_local(s.center);
        let cam = frame.to_local(cam_world);
        let run = tiles_in_reach(&s, &grid, &tiles, center, cam, 90.0);
        assert!(!run.records.is_empty());
        assert!(
            run.records.len() < MAX_PLANET_GRASS_TILES,
            "{}",
            run.records.len()
        );
        for (r, &(_, origin, count)) in run.records.iter().zip(&run.blocks) {
            assert!(count[0] > 0 && count[1] > 0);
            assert!(origin[0] + count[0] <= r.grass_tiles && origin[1] + count[1] <= r.grass_tiles);
        }
        // The nested tiles cover about the reach's disc of four-meter grass
        // tiles, each once.
        let spanned = spanned_tiles(&s, &grid, &tiles, &run, cam, 90.0);
        let disc = core::f64::consts::PI * 90.0 * 90.0 / (grid.tile_size() * grid.tile_size());
        let n = spanned as f64;
        assert!(n > 0.8 * disc && n < 2.0 * disc, "{n} vs {disc}");
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

    // Planes every point is inside.
    const OPEN: [[f32; 4]; 6] = [[0.0, 0.0, 0.0, 1.0]; 6];

    // The level-0 grass tiles `run`'s nested tiles span for the meadow seen
    // from a 1800 pixel tall view, asserting none is spanned twice.
    fn spanned_tiles(
        s: &PlanetShape,
        grid: &PlanetGrassGrid,
        drawn: &[DrawnPlanetTile],
        run: &PlanetGrassTiles,
        cam: [f32; 3],
        distance: f32,
    ) -> usize {
        let center = planet_center(s, &drawn[run.blocks[0].0]);
        let view = NestView {
            camera: cam,
            distance,
            planes: &OPEN,
            keep_scale: 1285.0 / 4.0,
        };
        let mut seen = hashbrown::HashSet::new();
        for &(index, origin, count) in &run.blocks {
            let tile = &drawn[index];
            let mut out = NestedTiles::default();
            nested_planet_tiles(
                &view,
                (s, grid, center),
                tile,
                (origin, count),
                [0.7, 0.03],
                &mut out,
            );
            for t in out.tiles {
                let k = 1i32 << t.level;
                for y in 0..k {
                    for x in 0..k {
                        let fine = [t.tile[0] * k + x, t.tile[1] * k + y];
                        assert!(seen.insert((tile.id, fine)), "{:?} {fine:?}", tile.id);
                    }
                }
            }
        }
        seen.len()
    }

    // Seen from a view whose blades reach hundreds of meters, the far grass
    // tiles grow coarser, so the visited tiles stay a small share of the
    // grass tiles they span.
    #[test]
    fn far_grass_tiles_nest_coarser() {
        let s = shape();
        let grid = PlanetGrassGrid::new(&s);
        let frame = LocalFrame::AUTHORED;
        let lod = TileLod::new(s, crate::planet::FINEST_CELL_SIZE);
        let cam_world = dvec3::add(
            s.center,
            [0.0, s.radius + s.height([0.0, 1.0, 0.0]) + 2.5, 0.0],
        );
        let sel = lod.select(
            &crate::planet::TileView::nearest(cam_world),
            &|_| true,
            &mut crate::planet::TileBoundsCache::new(s),
        );
        let tiles: Vec<DrawnPlanetTile> =
            sel.draw.iter().map(|d| drawn(&s, &frame, d.id)).collect();
        let center = frame.to_local(s.center);
        let cam = frame.to_local(cam_world);
        let run = tiles_in_reach(&s, &grid, &tiles, center, cam, 471.0);
        let view = NestView {
            camera: cam,
            distance: 471.0,
            planes: &OPEN,
            keep_scale: 1285.0 / 4.0,
        };
        let mut visited = NestedTiles::default();
        for &(index, origin, count) in &run.blocks {
            nested_planet_tiles(
                &view,
                (&s, &grid, center),
                &tiles[index],
                (origin, count),
                [0.7, 0.03],
                &mut visited,
            );
        }
        let visited = visited.tiles;
        let spanned = spanned_tiles(&s, &grid, &tiles, &run, cam, 471.0);
        assert!(visited.iter().any(|t| t.level >= 2));
        assert!(
            visited.len() * 8 < spanned,
            "{} of {spanned}",
            visited.len()
        );
        assert!(visited.len() < 8192, "{}", visited.len());
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
        let cam_world = dvec3::add(s.surface(dir), dvec3::scale(dir, 1.7));
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
                .zip(&r.blocks)
                .map(|(t, b)| (t.tile, b.1, b.2))
                .collect();
            v.sort_unstable();
            v
        };
        assert_eq!(blocks(&ra), blocks(&rb));
    }

    // The cap holds every tile a run reaching the farthest a planet's blades
    // do can reach, whatever the planet: the drawn tiles grow with distance,
    // so each level adds a ring of about as many tiles.
    #[test]
    fn the_cap_holds_every_tile_a_run_can_reach() {
        const REACH: f32 = MAX_PLANET_GRASS_REACH;
        let mut worst = 0;
        for (level, amplitude) in [(8u8, 40.0), (12, 40.0), (12, 2000.0), (16, 40.0)] {
            let width =
                0.5001 * f64::from(crate::planet::TILE_CELLS) * crate::planet::FINEST_CELL_SIZE;
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
                morph: GpuTileMorph::default(),
                morph_extent: 0.0,
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
                let cam_world = dvec3::add(s.surface(dir), dvec3::scale(dir, 1.7));
                let center = frame.to_local(s.center);
                let cam = frame.to_local(cam_world);
                let selected: Vec<_> = lod
                    .select(
                        &crate::planet::TileView {
                            eye: cam_world,
                            range_widths: crate::planet::MAX_RANGE_WIDTHS,
                        },
                        &|_| true,
                        &mut crate::planet::TileBoundsCache::new(s),
                    )
                    .draw
                    .iter()
                    .map(|d| light(d.id))
                    .collect();
                let mut near = Vec::new();
                records_in_reach((&s, &grid), &selected, [center, cam], REACH, &mut near);
                worst = worst.max(near.len());
            }
        }
        assert!(MAX_PLANET_GRASS_TILES >= worst + worst / 4, "{worst}");
    }

    // A tile wholly short of where its morph starts, beside no tile of
    // another level, roots its grass on its raw mesh; one reaching into the
    // morph, or on a seam, keeps its row.
    #[test]
    fn only_tiles_reaching_their_morph_keep_it() {
        let row = |eye: [f32; 3], seams: u32| GpuTileMorph {
            eye,
            range: 100.0,
            seams,
            ..GpuTileMorph::default()
        };
        let origin = [0.0; 3];
        assert!(unmorphed(&row([10.0, 0.0, 0.0], 0), origin, 20.0));
        assert!(!unmorphed(&row([70.0, 0.0, 0.0], 0), origin, 20.0));
        assert!(!unmorphed(&row([10.0, 0.0, 0.0], 1), origin, 20.0));
        assert!(unmorphed(&GpuTileMorph::default(), origin, 1e9));
    }

    // More drawn tiles in reach than a run holds keeps the nearest and says
    // so; a run within the cap says nothing.
    #[test]
    fn a_run_past_the_cap_keeps_the_nearest_and_says_so() {
        let s = shape();
        let grid = PlanetGrassGrid::new(&s);
        let frame = LocalFrame::AUTHORED;
        let lod = TileLod::new(s, crate::planet::FINEST_CELL_SIZE);
        let at = |x: u32| TileId {
            face: CubeFace(2),
            level: lod.max_level,
            x,
            y: 1 << (lod.max_level - 1),
        };
        let mid = 1u32 << (lod.max_level - 1);
        let cam = frame.to_local(dvec3::add(
            s.center,
            [0.0, s.radius + s.height([0.0, 1.0, 0.0]) + 1.7, 0.0],
        ));
        let center = frame.to_local(s.center);
        let near = drawn(&s, &frame, at(mid));
        let far = drawn(&s, &frame, at(mid + 1));
        let mut tiles = alloc::vec![far; MAX_PLANET_GRASS_TILES];
        tiles.push(near);
        let mut run = PlanetGrassTiles::default();
        run.gather((&s, &grid), &tiles, center, cam, 90.0);
        assert!(run.clipped);
        assert_eq!(run.records.len(), MAX_PLANET_GRASS_TILES);
        assert_eq!(run.blocks[0].0, MAX_PLANET_GRASS_TILES);
        run.gather((&s, &grid), &tiles[1..], center, cam, 90.0);
        assert!(!run.clipped);
        assert_eq!(run.records.len(), MAX_PLANET_GRASS_TILES);
    }

    // A tile whose morph would move its corners less than a blade's root can
    // show roots its grass on its raw mesh; one moving them more keeps its
    // morph row.
    #[test]
    fn a_tile_whose_morph_cannot_show_drops_it() {
        let s = shape();
        let grid = PlanetGrassGrid::new(&s);
        let frame = LocalFrame::AUTHORED;
        let lod = TileLod::new(s, crate::planet::FINEST_CELL_SIZE);
        let id = TileId {
            face: CubeFace(2),
            level: lod.max_level,
            x: 1 << (lod.max_level - 1),
            y: 1 << (lod.max_level - 1),
        };
        let cam = frame.to_local(dvec3::add(
            s.center,
            [0.0, s.radius + s.height([0.0, 1.0, 0.0]) + 1.7, 0.0],
        ));
        let center = frame.to_local(s.center);
        let morphing = |extent: f32| {
            let mut tile = drawn(&s, &frame, id);
            tile.morph = GpuTileMorph {
                eye: cam,
                range: 1.0,
                ..GpuTileMorph::default()
            };
            tile.morph_extent = extent;
            let run = tiles_in_reach(&s, &grid, &[tile], center, cam, 90.0);
            run.records[0].tile_morph.range > 0.0
        };
        assert!(!morphing(GRASS_MORPH_TOLERANCE * 0.5));
        assert!(morphing(GRASS_MORPH_TOLERANCE * 2.0));
    }
}
