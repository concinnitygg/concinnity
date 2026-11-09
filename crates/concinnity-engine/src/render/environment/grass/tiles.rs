//! Where the grass kernel looks for blades: the world-aligned tiles around the
//! camera, larger the farther they lie (see [`nest`](super::nest)), and the
//! jittered cell grid inside each tile that holds one blade per cell.
//!
//! Cells are counted from the world origin, so a blade's place depends only on
//! its cell and not on which tiles the camera currently covers.

use alloc::vec::Vec;
use concinnity_core::math::{SaturatingFrom, ceil, floor, round, sqrt};

use super::lod::GRASS_SHRINK_BAND;
use super::nest::{GRASS_NEST_LEVELS, nest_level};

/// Edge of one grass tile, in meters: the unit the kernel frustum- and
/// occlusion-culls.
pub const GRASS_TILE_SIZE: f32 = 4.0;

/// Highest density the grid places, in blades per square meter.
pub const MAX_GRASS_DENSITY: f32 = 1000.0;

/// Most blades the visible-blade buffer holds, whatever the field asks for.
pub const MAX_GRASS_BLADES: u32 = 1 << 21;

/// Threads per kernel group: each thread places one candidate blade.
pub const GRASS_GROUP_SIZE: u32 = 64;

/// The cell grid inside every tile: `cells_per_side` squared cells, one blade
/// each.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassGrid {
    /// Cells along each tile edge.
    pub cells_per_side: u32,
}

impl GrassGrid {
    /// The grid closest to `density` blades per square meter, or `None` when
    /// the density places no blade at all.
    pub fn for_density(density: f32) -> Option<Self> {
        if density.is_nan() || density <= 0.0 {
            return None;
        }
        let density = density.min(MAX_GRASS_DENSITY);
        let cells = round(GRASS_TILE_SIZE * sqrt(density)).max(1.0);
        Some(Self {
            cells_per_side: u32::saturating_from(cells),
        })
    }

    /// Edge of one cell, in meters.
    pub fn cell_size(&self) -> f32 {
        GRASS_TILE_SIZE / self.cells_per_side as f32
    }

    /// Candidate blades per tile.
    pub fn blades_per_tile(&self) -> u32 {
        self.cells_per_side * self.cells_per_side
    }

    /// The density the grid actually places, in blades per square meter.
    pub fn density(&self) -> f32 {
        self.blades_per_tile() as f32 / (GRASS_TILE_SIZE * GRASS_TILE_SIZE)
    }

    /// Kernel groups across one tile's candidates.
    pub fn groups_per_tile(&self) -> u32 {
        self.blades_per_tile().div_ceil(GRASS_GROUP_SIZE)
    }
}

/// An axis-aligned rectangle on the ground plane, `[x, z]` corners.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundRect {
    /// Minimum corner.
    pub min: [f32; 2],
    /// Maximum corner.
    pub max: [f32; 2],
}

impl GroundRect {
    /// The rectangle centered on `center` with half-extents `extent`.
    pub fn centered(center: [f32; 2], extent: [f32; 2]) -> Self {
        Self {
            min: [center[0] - extent[0], center[1] - extent[1]],
            max: [center[0] + extent[0], center[1] + extent[1]],
        }
    }

    /// Area in square meters; 0 for an empty rectangle.
    pub fn area(&self) -> f32 {
        (self.max[0] - self.min[0]).max(0.0) * (self.max[1] - self.min[1]).max(0.0)
    }

    /// The overlap of the two rectangles, empty when they do not meet.
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Self {
        Self {
            min: [self.min[0].max(other.min[0]), self.min[1].max(other.min[1])],
            max: [self.max[0].min(other.max[0]), self.max[1].min(other.max[1])],
        }
    }
}

/// What one run of the kernel chooses its tiles for: the camera, how far
/// blades reach, the frustum, and how distance thins them.
#[derive(Debug, Clone, Copy)]
pub struct NestView<'a> {
    /// The camera, in the frame the ground is laid out in.
    pub camera: [f32; 3],
    /// Distance past which no blade is placed.
    pub distance: f32,
    /// The frustum's `(normal, d)` planes, `dot(normal, p) + d >= 0` inside.
    pub planes: &'a [[f32; 4]; 6],
    /// Thinning's scale: a layer keeps `width * keep_scale / distance` of its
    /// blades (see [`GrassLodView::keep_scale`](super::lod::GrassLodView::keep_scale)).
    pub keep_scale: f32,
}

/// One tile a run visits: its coordinates at its level, its level, and how
/// near the camera any blade rooted in it can be.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NestedTile {
    /// The tile's column and row among the tiles of its level.
    pub tile: [i32; 2],
    /// Its level: it spans `2^level` tiles of level 0 along each side, and the
    /// kernel visits its cells of that level.
    pub level: u32,
    /// The least distance from the camera to the box its blades stay in.
    pub near: f32,
}

/// The tiles a run visits, and the working space finding them reuses from
/// one run to the next.
#[derive(Debug, Clone, Default)]
pub struct NestedTiles {
    /// The tiles, in the order the quadtree finds them.
    pub tiles: Vec<NestedTile>,
    stack: Vec<(u32, [i32; 2])>,
}

impl NestedTiles {
    /// Forget the tiles, keeping the space they took.
    pub fn clear(&mut self) {
        self.tiles.clear();
    }
}

/// An axis-aligned box, min and max corners.
pub type GrassBox = ([f32; 3], [f32; 3]);

/// Whether box `b` lies wholly outside one of `planes`.
fn box_outside(planes: &[[f32; 4]; 6], (lo, hi): &GrassBox) -> bool {
    planes.iter().any(|p| {
        let v: [f32; 3] = core::array::from_fn(|i| if p[i] >= 0.0 { hi[i] } else { lo[i] });
        p[0] * v[0] + p[1] * v[1] + p[2] * v[2] + p[3] < 0.0
    })
}

/// The distance from `p` to box `b`, 0 inside it.
fn box_distance(p: [f32; 3], (lo, hi): &GrassBox) -> f32 {
    let d: [f32; 3] = core::array::from_fn(|i| (lo[i] - p[i]).max(p[i] - hi[i]).max(0.0));
    sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
}

/// The tiles one layer's run visits, appended to `out.tiles`: a quadtree from tiles
/// of level `top` at `roots`, each split into its four children while a blade
/// of `width` meters in it could survive thinning without representing its
/// cell of the tile's level. A tile lying wholly past the reach or outside the
/// frustum is dropped, as is one `bounds` finds no ground for; `bounds` gives the
/// box every blade rooted in a tile stays inside. The tiles kept cover every
/// blade the run can keep exactly once.
pub fn nested_tiles(
    view: &NestView<'_>,
    width: f32,
    (top, roots): (u32, &mut dyn Iterator<Item = [i32; 2]>),
    bounds: &dyn Fn(u32, [i32; 2]) -> Option<GrassBox>,
    out: &mut NestedTiles,
) {
    let NestedTiles { tiles, stack } = out;
    stack.clear();
    stack.extend(roots.map(|t| (top, t)));
    while let Some((level, tile)) = stack.pop() {
        let Some(b) = bounds(level, tile) else {
            continue;
        };
        let near = box_distance(view.camera, &b);
        if near > view.distance || box_outside(view.planes, &b) {
            continue;
        }
        let keep = (width * view.keep_scale / near.max(1e-6)).min(1.0);
        if level == 0 || nest_level(keep, 1.0 - GRASS_SHRINK_BAND, level) == level {
            tiles.push(NestedTile { tile, level, near });
            continue;
        }
        let [x, z] = [tile[0] * 2, tile[1] * 2];
        stack.extend([[x, z], [x + 1, z], [x, z + 1], [x + 1, z + 1]].map(|t| (level - 1, t)));
    }
}

/// The tiles one terrain layer's run visits: the layer's ground in reach of
/// the camera, its heights `y_range` and its blades reaching `reach` meters
/// from their cells, its blades `width` meters wide.
pub fn terrain_tiles(
    view: &NestView<'_>,
    ground: &GroundRect,
    [y_range, [reach, width]]: [[f32; 2]; 2],
    out: &mut NestedTiles,
) {
    let top = GRASS_NEST_LEVELS;
    let size = GRASS_TILE_SIZE * (1u32 << top) as f32;
    let cam = [view.camera[0], view.camera[2]];
    let span = view.distance + reach;
    let area = ground.intersect(&GroundRect::centered(cam, [span, span]));
    if area.area() <= 0.0 {
        return;
    }
    let lo = |v: f32| i32::saturating_from(floor(v / size));
    let hi = |v: f32| i32::saturating_from(ceil(v / size)) - 1;
    let (x0, x1, z0, z1) = (
        lo(area.min[0]),
        hi(area.max[0]),
        lo(area.min[1]),
        hi(area.max[1]),
    );
    let mut roots = (z0..=z1).flat_map(|z| (x0..=x1).map(move |x| [x, z]));
    let bounds = |level: u32, t: [i32; 2]| {
        let s = GRASS_TILE_SIZE * (1u32 << level) as f32;
        let rect = GroundRect {
            min: [t[0] as f32 * s, t[1] as f32 * s],
            max: [(t[0] + 1) as f32 * s, (t[1] + 1) as f32 * s],
        };
        (ground.intersect(&rect).area() > 0.0).then(|| {
            (
                [rect.min[0] - reach, y_range[0] - reach, rect.min[1] - reach],
                [rect.max[0] + reach, y_range[1] + reach, rect.max[1] + reach],
            )
        })
    };
    nested_tiles(view, width, (top, &mut roots), &bounds, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grid_rounds_density_to_whole_cells_per_tile() {
        let grid = GrassGrid::for_density(100.0).unwrap();
        assert_eq!(grid.cells_per_side, 40);
        assert_eq!(grid.blades_per_tile(), 1600);
        assert_eq!(grid.density(), 100.0);
        assert_eq!(grid.cell_size(), 0.1);
        assert_eq!(grid.groups_per_tile(), 25);
    }

    #[test]
    fn a_sparse_field_keeps_one_blade_per_tile() {
        let grid = GrassGrid::for_density(0.001).unwrap();
        assert_eq!(grid.cells_per_side, 1);
        assert_eq!(grid.groups_per_tile(), 1);
    }

    #[test]
    fn no_density_places_nothing_and_density_is_capped() {
        assert_eq!(GrassGrid::for_density(0.0), None);
        assert_eq!(GrassGrid::for_density(-3.0), None);
        assert_eq!(GrassGrid::for_density(f32::NAN), None);
        let capped = GrassGrid::for_density(1.0e9).unwrap();
        assert_eq!(capped, GrassGrid::for_density(MAX_GRASS_DENSITY).unwrap());
    }

    // Planes every point is inside.
    const OPEN: [[f32; 4]; 6] = [[0.0, 0.0, 0.0, 1.0]; 6];

    // The meadow's thinning on a 1800 pixel tall view at 70 degrees.
    const KEEP_SCALE: f32 = 1285.0 / 4.0;
    const WIDTH: f32 = 0.03;

    fn view(camera: [f32; 3], distance: f32, planes: &[[f32; 4]; 6]) -> NestView<'_> {
        NestView {
            camera,
            distance,
            planes,
            keep_scale: KEEP_SCALE,
        }
    }

    fn tiles(view: &NestView<'_>, ground: &GroundRect) -> Vec<NestedTile> {
        let mut out = NestedTiles::default();
        terrain_tiles(view, ground, [[-1.0, 2.0], [0.7, WIDTH]], &mut out);
        out.tiles
    }

    // Every level-0 tile the visited tiles span, with the level it is
    // visited at.
    fn spanned(tiles: &[NestedTile]) -> hashbrown::HashMap<[i32; 2], u32> {
        let mut map = hashbrown::HashMap::new();
        for t in tiles {
            let n = 1i32 << t.level;
            for z in 0..n {
                for x in 0..n {
                    let fine = [t.tile[0] * n + x, t.tile[1] * n + z];
                    assert!(
                        map.insert(fine, t.level).is_none(),
                        "{fine:?} visited twice"
                    );
                }
            }
        }
        map
    }

    // The visited tiles never overlap and cover every level-0 tile with
    // ground in reach.
    #[test]
    fn the_tiles_partition_the_ground_in_reach() {
        let ground = GroundRect::centered([10.0, -20.0], [300.0, 250.0]);
        let v = view([3.0, 2.5, 7.0], 200.0, &OPEN);
        let visited = spanned(&tiles(&v, &ground));
        let n = i32::saturating_from(210.0 / GRASS_TILE_SIZE) + 2;
        let mut in_reach = 0;
        for z in -n..n {
            for x in -n..n {
                let rect = GroundRect {
                    min: [x as f32 * 4.0, z as f32 * 4.0],
                    max: [(x + 1) as f32 * 4.0, (z + 1) as f32 * 4.0],
                };
                let b = (
                    [rect.min[0] - 0.7, -1.7, rect.min[1] - 0.7],
                    [rect.max[0] + 0.7, 2.7, rect.max[1] + 0.7],
                );
                if ground.intersect(&rect).area() > 0.0 && box_distance(v.camera, &b) <= v.distance
                {
                    in_reach += 1;
                    assert!(visited.contains_key(&[x, z]), "{x} {z} missed");
                }
            }
        }
        assert!(in_reach > 1000);
    }

    // Each tile is as coarse as thinning allows at its nearest point: its
    // level's cells hold every blade that survives in it, and its parent's
    // would not.
    #[test]
    fn tiles_coarsen_as_thinning_allows() {
        let ground = GroundRect::centered([0.0, 0.0], [1000.0, 1000.0]);
        let v = view([0.0, 2.5, 0.0], 471.0, &OPEN);
        let span = 1.0 - GRASS_SHRINK_BAND;
        let mut levels = [0u32; GRASS_NEST_LEVELS as usize + 1];
        for t in tiles(&v, &ground) {
            let keep = (WIDTH * KEEP_SCALE / t.near).min(1.0);
            assert!(t.level == 0 || keep <= span / (1u32 << (2 * t.level)) as f32);
            levels[t.level as usize] += 1;
        }
        // Full density out to 13.8 m, thinning to a quarter by 55 m, where
        // level 1 starts, and level 2 from 220 m; level 3 would start at 883.
        assert!(levels[..3].iter().all(|&n| n > 0), "{levels:?}");
        assert_eq!(levels[3], 0);
    }

    // Candidates follow the blades thinning keeps: a few times the blades
    // alive in reach, not the reach's square of full-density cells.
    #[test]
    fn candidates_follow_the_kept_blades() {
        let ground = GroundRect::centered([0.0, 0.0], [1000.0, 1000.0]);
        let grid = GrassGrid::for_density(100.0).unwrap();
        let v = view([0.0, 2.5, 0.0], 471.0, &OPEN);
        let visited = tiles(&v, &ground);
        let candidates = visited.len() as f32 * grid.blades_per_tile() as f32;
        // Blades alive over the disc: full near, then width * scale / d of
        // them over the shrink span.
        let span = 1.0 - GRASS_SHRINK_BAND;
        let mut alive = 0.0f32;
        let mut r = 0.0f32;
        while r < 471.0 {
            let d = sqrt(r * r + 2.5 * 2.5);
            let keep = ((WIDTH * KEEP_SCALE / d).min(1.0) / span).min(1.0);
            alive += core::f32::consts::TAU * r * 0.5 * keep * grid.density();
            r += 0.5;
        }
        assert!(candidates < 3.0 * alive, "{candidates} vs {alive}");
        let flat = core::f32::consts::PI * 471.0 * 471.0 * grid.density();
        assert!(candidates < flat / 6.0, "{candidates} vs {flat}");
    }

    #[test]
    fn tiles_outside_the_frustum_are_dropped() {
        let ground = GroundRect::centered([0.0, 0.0], [500.0, 500.0]);
        // Only z >= 0 is inside.
        let mut planes = OPEN;
        planes[0] = [0.0, 0.0, 1.0, 0.0];
        let v = view([0.0, 2.5, 0.0], 300.0, &planes);
        let visited = tiles(&v, &ground);
        assert!(!visited.is_empty());
        // A box reaches 0.7 m past its tile, into the row before z = 0.
        assert!(visited.iter().all(|t| t.tile[1] >= -1), "{visited:?}");
        let all = tiles(&view([0.0, 2.5, 0.0], 300.0, &OPEN), &ground);
        let (a, b) = (spanned(&visited).len(), spanned(&all).len());
        assert!(a < b * 6 / 10, "{a} vs {b}");
    }

    #[test]
    fn only_the_ground_is_visited() {
        let ground = GroundRect::centered([2.0, 2.0], [1.0, 1.0]);
        let visited = tiles(&view([0.0, 1.0, 0.0], 60.0, &OPEN), &ground);
        assert_eq!(
            visited,
            [NestedTile {
                tile: [0, 0],
                level: 0,
                near: 0.0,
            }]
        );
        let far = tiles(&view([500.0, 1.0, 0.0], 60.0, &OPEN), &ground);
        assert!(far.is_empty());
    }
}
