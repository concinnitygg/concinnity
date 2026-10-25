// Which tiles of a planet's quadtree to keep and which to draw, from where the
// camera is, how large the ground looks from there, and which tiles have their
// meshes.
//
// Each level hands over to its parent at a range proportional to its tiles'
// width. Its vertices morph toward the parent's mesh over the last stretch of
// that range, so a tile and its children look alike wherever the quadtree
// swaps them.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use concinnity_core::math::dvec3;

use super::edits::EditArea;
use super::seams::{DrawnTile, balance, seams};
use super::shape::{CubeFace, PlanetShape};
use super::tile::{MAX_TILE_LEVEL, TILE_CELLS, TileBounds, TileId, tile_bounds};
use concinnity_core::math::dvec3::DVec3;

/// Fewest nominal widths of its tiles a level's range spans. Nearer, a tile's
/// own morph could still be under way where it gives way to its children,
/// and the swap would show.
pub const MIN_RANGE_WIDTHS: f64 = 5.0;

/// Most nominal widths of its tiles a level's range spans: the bound the
/// resident tile count is sized from.
pub const MAX_RANGE_WIDTHS: f64 = 6.0;

/// Where along its range a level's vertices start to morph toward its
/// parent's mesh, as a fraction of the range.
pub const MORPH_START: f64 = 0.85;

/// How wide a level's cells aim to look at the end of its range, in logical
/// points. A level's range is bounded by [`MIN_RANGE_WIDTHS`] and
/// [`MAX_RANGE_WIDTHS`], so this only sets where in that span a view lands.
pub const RANGE_CELL_POINTS: f64 = 8.0;

/// How far the camera may stray from where its tiles were chosen before they
/// are chosen again, as a fraction of its height over the highest ground, or
/// of the finest level's range when that is more.
pub const REPLAN_DRIFT: f64 = 0.05;

/// How far a tile's flat cells may sag under the sphere at most, as a
/// fraction of the planet's radius: the coarsest level any tile is drawn at
/// keeps its cells this close to the curve wherever the camera is, so ground
/// seen from far off holds its height as well as its outline.
pub const MAX_CELL_SAG: f64 = 2e-5;

/// A split tile's children stay resident until the camera is this many times
/// their level's range away, so hovering at the split does not rebuild them.
pub const RETAIN_RANGE: f64 = 1.3;

/// Most tiles a [`TileBoundsCache`] holds before it starts over: several times
/// what any camera visits, so a steady camera never recomputes one.
const MAX_CACHED_BOUNDS: usize = 8 * super::MAX_RESIDENT_TILES;

/// Each tile's bounds on one planet, computed the first time it is asked for.
///
/// Bounds depend only on the planet and the tile, so a camera that revisits
/// the quadtree every frame pays for each tile once. Room for every tile it
/// can hold is reserved up front, so filling it never reallocates.
#[derive(Debug, Clone)]
pub struct TileBoundsCache {
    shape: PlanetShape,
    bounds: hashbrown::HashMap<TileId, TileBounds>,
}

impl TileBoundsCache {
    /// An empty cache for the tiles of `shape`.
    pub fn new(shape: PlanetShape) -> Self {
        Self {
            shape,
            bounds: hashbrown::HashMap::with_capacity(MAX_CACHED_BOUNDS),
        }
    }

    /// The bounds of `tile`.
    pub fn get(&mut self, tile: TileId) -> TileBounds {
        if let Some(b) = self.bounds.get(&tile) {
            return *b;
        }
        if self.bounds.len() >= MAX_CACHED_BOUNDS {
            self.bounds.clear();
        }
        let b = tile_bounds(&self.shape, tile);
        self.bounds.insert(tile, b);
        b
    }

    /// Take `shape`, whose ground differs from the last one's only within
    /// `area`: the bounds of the tiles reaching into it are forgotten.
    pub fn reshape_within(&mut self, shape: PlanetShape, area: &EditArea) {
        let center = shape.center;
        self.bounds.retain(|_, b| {
            let to = dvec3::sub(b.center, center);
            let distance = dvec3::length(to);
            let angle = if distance > b.radius {
                libm::asin(b.radius / distance)
            } else {
                core::f64::consts::PI
            };
            !area.overlaps(&EditArea {
                center: dvec3::normalize_or(to, [0.0, 1.0, 0.0]),
                angle,
            })
        });
        self.shape = shape;
    }

    /// How many tiles' bounds are held.
    pub fn len(&self) -> usize {
        self.bounds.len()
    }

    /// Whether no tile's bounds are held.
    pub fn is_empty(&self) -> bool {
        self.bounds.is_empty()
    }
}

/// Where a camera sees a planet's tiles from, and how far each level of them
/// reaches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileView {
    /// The camera, in authored coordinates.
    pub eye: DVec3,
    /// How many nominal widths of its tiles each level's range spans.
    pub range_widths: f64,
}

impl TileView {
    /// A camera at `eye` whose view spans `points_per_radian` logical points
    /// per radian: each level reaches as far as its cells look
    /// [`RANGE_CELL_POINTS`] wide, within [`MIN_RANGE_WIDTHS`] and
    /// [`MAX_RANGE_WIDTHS`].
    pub fn new(eye: DVec3, points_per_radian: f64) -> Self {
        let widths = points_per_radian / (f64::from(TILE_CELLS) * RANGE_CELL_POINTS);
        Self {
            eye,
            range_widths: if widths.is_finite() {
                widths.clamp(MIN_RANGE_WIDTHS, MAX_RANGE_WIDTHS)
            } else {
                MIN_RANGE_WIDTHS
            },
        }
    }

    /// A camera at `eye` whose levels reach [`MIN_RANGE_WIDTHS`], the least
    /// any view does.
    pub fn nearest(eye: DVec3) -> Self {
        Self {
            eye,
            range_widths: MIN_RANGE_WIDTHS,
        }
    }
}

// The coarsest level whose nominal cells sag at most `MAX_CELL_SAG` of the
// radius: a cell spanning `a` radians sags `a^2 / 8` of it.
fn sag_level() -> u8 {
    let face = core::f64::consts::FRAC_PI_2 / f64::from(TILE_CELLS);
    let widest = libm::sqrt(8.0 * MAX_CELL_SAG);
    let mut level = 0u8;
    while level < MAX_TILE_LEVEL && face / f64::from(1u32 << level) > widest {
        level += 1;
    }
    level
}

/// How many logical points a radian of view spans, near the middle of a
/// view `viewport_height` points tall with a vertical field of view of
/// `fov_y_degrees`; 0 for a view with no height.
pub fn points_per_radian(fov_y_degrees: f32, viewport_height: f32) -> f64 {
    let half = (f64::from(fov_y_degrees) * 0.5).to_radians();
    let tan = libm::tan(half);
    if viewport_height > 0.0 && tan > 0.0 && tan.is_finite() {
        0.5 * f64::from(viewport_height) / tan
    } else {
        0.0
    }
}

/// How fine a planet's tiles get: the camera's distance decides each tile's
/// level, down to `max_level`.
#[derive(Debug, Clone, PartialEq)]
pub struct TileLod {
    /// The surface the tiles cover.
    pub shape: PlanetShape,
    /// The deepest level any tile reaches.
    pub max_level: u8,
    /// The coarsest level any tile is drawn at, once resident: its cells sag
    /// at most [`MAX_CELL_SAG`] of the radius under the sphere.
    pub min_level: u8,
}

impl TileLod {
    /// Levels deep enough that the finest cells are at most `cell_size`
    /// meters across.
    pub fn new(shape: PlanetShape, cell_size: f64) -> Self {
        let face_width = shape.radius * core::f64::consts::FRAC_PI_2;
        let mut level = 0u8;
        while level < MAX_TILE_LEVEL
            && face_width / f64::from(1u32 << level) / f64::from(TILE_CELLS) > cell_size
        {
            level += 1;
        }
        Self {
            shape,
            max_level: level,
            min_level: sag_level().min(level),
        }
    }

    /// The width of a tile at `level` across the middle of an undistorted
    /// face, in meters: the measure every level's range is set in.
    fn nominal_width(&self, level: u8) -> f64 {
        self.shape.radius * core::f64::consts::FRAC_PI_2 / f64::from(1u32 << level)
    }

    /// How far from `view`'s camera tiles at `level` are drawn before their
    /// parent's mesh takes over, in meters.
    pub fn range(&self, level: u8, view: &TileView) -> f64 {
        view.range_widths * self.nominal_width(level)
    }

    /// How far a vertex of a tile at `level`, `distance` meters from `view`'s
    /// camera, has morphed toward the parent's mesh: 0 until
    /// [`MORPH_START`] of the level's range, 1 from the range on.
    ///
    /// A tile at [`Self::min_level`] or coarser never gives way to its
    /// parent, so never morphs.
    pub fn morph(&self, level: u8, distance: f64, view: &TileView) -> f64 {
        if level <= self.min_level {
            return 0.0;
        }
        let range = self.range(level, view);
        ((distance - MORPH_START * range) / ((1.0 - MORPH_START) * range)).clamp(0.0, 1.0)
    }

    /// How far `view`'s camera may stray before its tiles are chosen again,
    /// in meters (see [`REPLAN_DRIFT`]). [`Self::select`] splits a tile this
    /// much early, so wherever within it the camera is, no drawn tile reaches
    /// nearer the camera than its children's range, where they would be
    /// morphed short of its mesh.
    pub fn drift(&self, view: &TileView) -> f64 {
        let height = dvec3::length(dvec3::sub(view.eye, self.shape.center))
            - self.shape.radius
            - self.shape.highest_height();
        REPLAN_DRIFT * height.max(self.range(self.max_level, view))
    }

    // How near a camera within `slack` of `view`'s can come to `tile`, over
    // its children's range: below 1 the tile splits into them. 0 for a tile
    // coarser than the coarsest drawn level, infinite for one at the deepest.
    fn nearness(
        &self,
        tile: TileId,
        view: &TileView,
        slack: f64,
        bounds: &mut TileBoundsCache,
    ) -> f64 {
        if tile.level >= self.max_level {
            return f64::INFINITY;
        }
        if tile.level < self.min_level {
            return 0.0;
        }
        let b = bounds.get(tile);
        let gap = dvec3::length(dvec3::sub(view.eye, b.center)) - b.radius - slack;
        gap / self.range(tile.level + 1, view)
    }

    /// The tiles `view`'s camera wants, given which are `resident`, reading
    /// each tile's bounds from `bounds`.
    ///
    /// A tile within its children's range of any camera within
    /// [`Self::drift`] of `view`'s splits into them once all four are
    /// resident; until then it draws itself. The six face roots are never
    /// split past while not resident, so nothing is drawn for a face whose
    /// root is missing, and a drawn set never overlaps or leaves a hole. A
    /// tile beside one two or more levels coarser gives way to its parent, so
    /// drawn neighbors are at most a level apart.
    pub fn select(
        &self,
        view: &TileView,
        resident: &dyn Fn(TileId) -> bool,
        bounds: &mut TileBoundsCache,
    ) -> TileSelection {
        let mut drawn = BTreeSet::new();
        let mut wanted = Vec::new();
        let mut retained = Vec::new();
        let mut visit = Visit {
            lod: self,
            view,
            slack: self.drift(view),
            resident,
            bounds,
            drawn: &mut drawn,
            wanted: &mut wanted,
            retained: &mut retained,
        };
        for face in CubeFace::ALL {
            let root = TileId::root(face);
            visit.tile(root, true, resident(root));
        }
        balance(&mut drawn);
        TileSelection {
            draw: seams(&drawn),
            wanted,
            retained,
        }
    }
}

// One walk of the quadtree for `TileLod::select`.
struct Visit<'a> {
    lod: &'a TileLod,
    view: &'a TileView,
    slack: f64,
    resident: &'a dyn Fn(TileId) -> bool,
    bounds: &'a mut TileBoundsCache,
    drawn: &'a mut BTreeSet<TileId>,
    wanted: &'a mut Vec<TileId>,
    retained: &'a mut Vec<TileId>,
}

impl Visit<'_> {
    // `wanted`: every ancestor split; `drawable`: every ancestor split with
    // its children resident.
    fn tile(&mut self, tile: TileId, wanted: bool, drawable: bool) {
        self.retained.push(tile);
        let nearness = self.lod.nearness(tile, self.view, self.slack, self.bounds);
        let children = tile.children();
        let split = wanted && nearness < 1.0;
        if wanted {
            self.wanted.push(tile);
            let ready = split && children.iter().all(|&c| (self.resident)(c));
            if drawable && !ready {
                self.drawn.insert(tile);
            }
            if split {
                for child in children {
                    self.tile(child, true, drawable && ready);
                }
                return;
            }
        }
        if nearness < RETAIN_RANGE {
            for child in children {
                self.tile(child, false, false);
            }
        }
    }
}

/// One frame's choice of tiles.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TileSelection {
    /// The tiles to draw, in ascending order, each with the sides it meets a
    /// coarser or a finer tile on: together they cover every face whose root
    /// is resident, once.
    pub draw: Vec<DrawnTile>,
    /// Every tile the camera wants: the ones not yet resident are the ones to
    /// build.
    pub wanted: Vec<TileId>,
    /// The tiles worth keeping resident: every tile wanted, plus the children
    /// of tiles just past their split.
    pub retained: Vec<TileId>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::tile::TileSide;

    fn lod() -> TileLod {
        TileLod::new(
            PlanetShape {
                center: [0.0, -50_000.0, 0.0],
                radius: 50_000.0,
                amplitude: 60.0,
                feature_size: 2_000.0,
                octaves: 8,
                seed: 1,
                continents: crate::planet::Continents::NONE,
                edits: None,
            },
            1.0,
        )
    }

    // A camera standing on the planet's top.
    const STANDING: DVec3 = [0.0, 62.0, 0.0];

    fn view(eye: DVec3, range_widths: f64) -> TileView {
        TileView { eye, range_widths }
    }

    fn standing() -> TileView {
        view(STANDING, MIN_RANGE_WIDTHS)
    }

    fn cache(l: &TileLod) -> TileBoundsCache {
        TileBoundsCache::new(l.shape.clone())
    }

    fn ids(sel: &TileSelection) -> Vec<TileId> {
        sel.draw.iter().map(|d| d.id).collect()
    }

    // The cache hands back exactly what the bounds function computes, and a
    // walk asks for each tile once.
    #[test]
    fn the_cache_returns_the_computed_bounds_once_per_tile() {
        let l = lod();
        let mut c = cache(&l);
        let sel = l.select(&standing(), &|_| true, &mut c);
        let held = c.len();
        assert!(held > 0 && held <= sel.retained.len());
        let again = l.select(&standing(), &|_| true, &mut c);
        assert_eq!(again, sel);
        assert_eq!(c.len(), held, "a second walk finds every tile cached");
        for &t in &sel.wanted {
            assert_eq!(c.get(t), tile_bounds(&l.shape, t));
        }
    }

    #[test]
    fn a_full_cache_starts_over_in_the_room_it_reserved() {
        let l = lod();
        let mut c = cache(&l);
        let reserved = c.bounds.capacity();
        assert!(reserved >= MAX_CACHED_BOUNDS);
        let side = 1u32 << 8;
        let tiles = (0..side * side).map(|i| TileId {
            face: CubeFace::ALL[0],
            level: 8,
            x: i % side,
            y: i / side,
        });
        for tile in tiles.take(MAX_CACHED_BOUNDS + 1) {
            c.get(tile);
        }
        assert_eq!(c.len(), 1);
        assert_eq!(c.bounds.capacity(), reserved);
    }

    #[test]
    fn the_finest_cells_meet_the_requested_size() {
        let l = lod();
        let finest = l.nominal_width(l.max_level) / f64::from(TILE_CELLS);
        assert!(finest <= 1.0 && finest * 2.0 > 1.0, "{finest}");
    }

    // A view's ranges follow how many points a radian spans, within their
    // bounds whatever the screen.
    #[test]
    fn ranges_follow_the_view_s_points_per_radian() {
        let cells = f64::from(TILE_CELLS) * RANGE_CELL_POINTS;
        let mid = (MIN_RANGE_WIDTHS + MAX_RANGE_WIDTHS) * 0.5;
        let widths = |ppr: f64| TileView::new(STANDING, ppr).range_widths;
        assert_eq!(widths(mid * cells), mid);
        assert!(widths(5.2 * cells) < widths(5.8 * cells));
        assert_eq!(widths(1.0), MIN_RANGE_WIDTHS);
        assert_eq!(widths(0.0), MIN_RANGE_WIDTHS);
        assert_eq!(widths(f64::NAN), MIN_RANGE_WIDTHS);
        assert_eq!(widths(1e9), MAX_RANGE_WIDTHS);
        assert_eq!(TileView::nearest(STANDING).range_widths, MIN_RANGE_WIDTHS);
        let l = lod();
        let v = view(STANDING, 6.0);
        assert_eq!(l.range(3, &v), 6.0 * l.nominal_width(3));
        assert_eq!(l.range(4, &v) * 2.0, l.range(3, &v));
    }

    // Coarser finest cells stop the quadtree a level sooner per doubling, and
    // the camera's drawn tiles stop with it.
    #[test]
    fn coarser_finest_cells_stop_the_quadtree_sooner() {
        let fine = lod();
        let shape = fine.shape.clone();
        for (size, shallower) in [(2.0, 1), (4.0, 2)] {
            let coarse = TileLod::new(shape.clone(), size);
            assert_eq!(coarse.max_level + shallower, fine.max_level, "{size} m");
            let finest = coarse.nominal_width(coarse.max_level) / f64::from(TILE_CELLS);
            assert!(finest <= size && finest * 2.0 > size, "{finest}");
            let sel = coarse.select(&standing(), &|_| true, &mut cache(&coarse));
            let deepest = sel.draw.iter().map(|d| d.id.level).max().unwrap();
            assert_eq!(deepest, coarse.max_level);
            let fewer = sel.retained.len();
            let all = fine
                .select(&standing(), &|_| true, &mut cache(&fine))
                .retained
                .len();
            assert!(fewer < all, "{fewer} vs {all}");
        }
    }

    #[test]
    fn points_per_radian_follow_the_view() {
        // A 90 degree view 1000 points tall spans 500 points per radian at
        // its middle.
        assert!((points_per_radian(90.0, 1000.0) - 500.0).abs() < 1e-9);
        assert!(points_per_radian(30.0, 1000.0) > points_per_radian(60.0, 1000.0));
        assert_eq!(points_per_radian(60.0, 0.0), 0.0);
        assert_eq!(points_per_radian(0.0, 1000.0), 0.0);
        assert!(points_per_radian(180.0, 1000.0) < 1e-6);
    }

    #[test]
    fn the_morph_runs_over_the_end_of_a_level_s_range() {
        let l = lod();
        let v = standing();
        let r = l.range(7, &v);
        assert_eq!(l.morph(7, 0.0, &v), 0.0);
        assert_eq!(l.morph(7, MORPH_START * r, &v), 0.0);
        assert!((l.morph(7, (1.0 + MORPH_START) * 0.5 * r, &v) - 0.5).abs() < 1e-12);
        assert!(l.morph(7, r, &v) > 1.0 - 1e-12);
        assert_eq!(l.morph(7, 2.0 * r, &v), 1.0);
    }

    // Wherever the camera strays within the drift, every drawn tile stays at
    // least its children's range away, so its children would have morphed all
    // the way into it there: the swap cannot show.
    #[test]
    fn drawn_tiles_stay_past_their_children_s_range_within_the_drift() {
        let l = lod();
        let mut c = cache(&l);
        for height in [2.0, 400.0] {
            let eye = [0.0, 40.0 + height, 0.0];
            let v = view(eye, MIN_RANGE_WIDTHS);
            let drift = l.drift(&v);
            assert!(drift > 0.0);
            let sel = l.select(&v, &|_| true, &mut c);
            for step in [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]] {
                let moved = dvec3::add(eye, dvec3::scale(step, drift * 0.99));
                for d in sel.draw.iter().filter(|d| d.id.level < l.max_level) {
                    let b = c.get(d.id);
                    let gap = dvec3::length(dvec3::sub(moved, b.center)) - b.radius;
                    assert!(
                        gap >= l.range(d.id.level + 1, &v) - 1e-6,
                        "{:?} at {height} m",
                        d.id
                    );
                }
            }
        }
    }

    // With every tile resident, the drawn tiles reach the finest level under
    // the camera and cover each face exactly once.
    #[test]
    fn a_fully_resident_selection_covers_every_face_once() {
        let l = lod();
        let sel = l.select(&standing(), &|_| true, &mut cache(&l));
        let drawn = ids(&sel);
        let deepest = drawn.iter().map(|t| t.level).max().unwrap();
        assert_eq!(deepest, l.max_level);
        assert!(drawn.windows(2).all(|w| w[0] < w[1]), "drawn in order");
        for face in CubeFace::ALL {
            let area: f64 = drawn
                .iter()
                .filter(|t| t.face == face)
                .map(|t| 1.0 / f64::from(1u32 << (2 * u32::from(t.level))))
                .sum();
            assert!((area - 1.0).abs() < 1e-12, "face {face:?} covered {area}");
        }
        // No drawn tile is an ancestor of another.
        let set: BTreeSet<TileId> = drawn.iter().copied().collect();
        for t in &drawn {
            let mut up = t.parent();
            while let Some(p) = up {
                assert!(!set.contains(&p));
                up = p.parent();
            }
        }
    }

    // The coarsest drawn level's cells sag under the sphere within the bound,
    // and a level coarser would not, on a planet of any size.
    #[test]
    fn the_coarsest_level_keeps_its_cells_near_the_sphere() {
        for radius in [5_000.0, 50_000.0, 6_371_000.0] {
            let shape = PlanetShape {
                radius,
                ..lod().shape
            };
            let l = TileLod::new(shape, 1.0);
            assert!(l.min_level > 0 && l.min_level < l.max_level);
            let sag = |level: u8| {
                let a = l.nominal_width(level) / f64::from(TILE_CELLS) / radius;
                a * a / 8.0
            };
            assert!(sag(l.min_level) <= MAX_CELL_SAG, "{radius}");
            assert!(sag(l.min_level - 1) > MAX_CELL_SAG, "{radius}");
        }
        // A planet too small for the bound stops at its finest level.
        let tiny = TileLod::new(lod().shape, 1e9);
        assert_eq!(tiny.min_level, tiny.max_level);
    }

    // However far the camera, every resident face draws no tile coarser
    // than the coarsest level, and those never morph.
    #[test]
    fn a_far_camera_draws_no_tile_coarser_than_the_coarsest_level() {
        let l = lod();
        let far = [0.0, 150_000.0, 0.0];
        let v = view(far, MAX_RANGE_WIDTHS);
        let sel = l.select(&v, &|_| true, &mut cache(&l));
        assert!(sel.draw.iter().all(|d| d.id.level == l.min_level));
        assert_eq!(sel.draw.len(), 6 << (2 * l.min_level));
        assert_eq!(l.morph(l.min_level, 1e12, &v), 0.0);
        assert_eq!(l.morph(l.min_level + 1, 1e12, &v), 1.0);
        // Until its children arrive, a coarser tile stands in for them.
        let sel = l.select(&v, &|t: TileId| t.level == 0, &mut cache(&l));
        assert_eq!(sel.draw.len(), 6);
    }

    // A wider view reaches each level farther, so more tiles are drawn.
    #[test]
    fn a_wider_range_draws_more_tiles() {
        let l = lod();
        let near = l.select(&standing(), &|_| true, &mut cache(&l));
        let far = l.select(&view(STANDING, MAX_RANGE_WIDTHS), &|_| true, &mut cache(&l));
        assert!(far.draw.len() > near.draw.len());
    }

    // A tile whose children are not all resident draws itself instead, and
    // nothing beneath it is drawn.
    #[test]
    fn a_split_waits_for_all_four_children() {
        let l = lod();
        let full = l.select(&standing(), &|_| true, &mut cache(&l));
        let level = 3;
        let resident = |t: TileId| t.level <= level;
        let sel = l.select(&standing(), &resident, &mut cache(&l));
        assert!(sel.draw.iter().all(|d| d.id.level <= level));
        assert!(sel.draw.iter().any(|d| d.id.level == level));
        assert_eq!(sel.wanted, full.wanted, "what is wanted ignores residency");
        // Without the root, a face draws nothing.
        let sel = l.select(
            &standing(),
            &|t: TileId| t.face != CubeFace(2),
            &mut cache(&l),
        );
        assert!(sel.draw.iter().all(|d| d.id.face != CubeFace(2)));
    }

    // A tile whose neighbor is stuck two levels coarser (its children are
    // not resident) gives way to its parent.
    #[test]
    fn a_selection_waiting_on_residency_stays_balanced() {
        let l = lod();
        let full = l.select(&standing(), &|_| true, &mut cache(&l));
        let under = full
            .draw
            .iter()
            .find(|d| d.id.level == l.max_level)
            .unwrap()
            .id;
        let stuck = under
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .neighbor(TileSide::MaxS);
        let resident = |t: TileId| t.parent() != Some(stuck);
        let sel = l.select(&standing(), &resident, &mut cache(&l));
        let drawn: BTreeSet<TileId> = ids(&sel).into_iter().collect();
        for d in &sel.draw {
            for side in TileSide::ALL {
                let mut across = Some(d.id.neighbor(side));
                while let Some(a) = across {
                    if drawn.contains(&a) {
                        assert!(d.id.level - a.level <= 1, "{:?} beside {a:?}", d.id);
                        break;
                    }
                    across = a.parent();
                }
            }
        }
    }

    // Wherever the camera stands, a level's tiles meet the next coarser
    // level only where they have morphed into their parents' mesh, and the
    // coarser tiles have not yet begun to morph into theirs: the seam is
    // closed, and so is every seam the quadtree swaps over.
    #[test]
    fn level_seams_meet_fully_morphed_and_unmorphed() {
        let l = lod();
        let mut bounds = cache(&l);
        let n = TILE_CELLS as usize;
        let side_points = |t: TileId, side: TileSide| -> Vec<DVec3> {
            (0..=n)
                .map(|k| {
                    let f = k as f64 / n as f64;
                    let (fs, ft) = match side {
                        TileSide::MinS => (0.0, f),
                        TileSide::MaxS => (1.0, f),
                        TileSide::MinT => (f, 0.0),
                        TileSide::MaxT => (f, 1.0),
                    };
                    l.shape.surface(t.direction(fs, ft))
                })
                .collect()
        };
        let mut seams_seen = 0;
        for widths in [MIN_RANGE_WIDTHS, MAX_RANGE_WIDTHS] {
            for height in [2.0, 62.0, 900.0, 20_000.0] {
                for i in 0..6 {
                    let a = 0.2 + f64::from(i) * 0.31;
                    let up = dvec3::normalize_or([libm::sin(a), 2.0, libm::cos(a * 1.7)], [0.0; 3]);
                    let eye = dvec3::add(l.shape.center, dvec3::scale(up, 50_000.0 + height));
                    let v = view(eye, widths);
                    let sel = l.select(&v, &|_| true, &mut bounds);
                    for d in &sel.draw {
                        for side in TileSide::ALL {
                            if d.seams.coarser & side.bit() == 0 {
                                continue;
                            }
                            seams_seen += 1;
                            for p in side_points(d.id, side) {
                                let dist = dvec3::length(dvec3::sub(p, eye));
                                assert_eq!(l.morph(d.id.level, dist, &v), 1.0, "{:?}", d.id);
                                assert_eq!(l.morph(d.id.level - 1, dist, &v), 0.0, "{:?}", d.id);
                            }
                        }
                    }
                }
            }
        }
        assert!(seams_seen > 100, "{seams_seen}");
    }

    // The bound the seams rest on: a tile's points span less, in distance
    // from any camera, than its children's range leaves before the tile's
    // own morph starts. Checked at the narrowest range, across the face.
    #[test]
    fn the_narrowest_range_outspans_every_tile() {
        let l = lod();
        let v = view(STANDING, MIN_RANGE_WIDTHS);
        for level in 0..l.max_level {
            let n = 1u32 << level;
            for (x, y) in [(0, 0), (n / 2, n / 2), (n / 3, 0), (n - 1, n / 5)] {
                let t = TileId {
                    face: CubeFace(4),
                    level,
                    x,
                    y,
                };
                let span = 2.0 * tile_bounds(&l.shape, t).radius;
                let children = l.range(level + 1, &v);
                assert!(
                    children + span <= MORPH_START * l.range(level, &v),
                    "level {level} ({x}, {y}): span {span}, range {children}"
                );
            }
        }
    }

    // The retained set holds everything wanted, plus the band past it.
    #[test]
    fn retention_holds_the_wanted_tiles_and_a_band_past_them() {
        let l = lod();
        let wanted: BTreeSet<TileId> = l
            .select(&standing(), &|_| true, &mut cache(&l))
            .wanted
            .into_iter()
            .collect();
        let retained: BTreeSet<TileId> = l
            .select(&standing(), &|_| true, &mut cache(&l))
            .retained
            .into_iter()
            .collect();
        assert!(wanted.is_subset(&retained));
        assert!(retained.len() > wanted.len());
    }

    // Wherever the camera stands or flies, and however wide its view, the
    // tile count stays bounded: the pool the meshes live in is sized from
    // this.
    #[test]
    fn the_retained_count_stays_bounded() {
        let l = lod();
        let mut most = 0;
        for i in 0..24 {
            let a = f64::from(i) * 0.27;
            let up =
                dvec3::normalize_or([libm::sin(a), libm::cos(a), libm::sin(a * 3.1)], [0.0; 3]);
            for height in [2.0, 62.0, 500.0, 5_000.0] {
                let cam = dvec3::add(l.shape.center, dvec3::scale(up, 50_000.0 + height));
                let v = view(cam, MAX_RANGE_WIDTHS);
                most = most.max(l.select(&v, &|_| true, &mut cache(&l)).retained.len());
            }
        }
        assert!(most <= MAX_RESIDENT_HINT, "{most}");
        assert!(
            most * 5 > MAX_RESIDENT_HINT * 4,
            "the pool is sized near the most: {most}"
        );
    }

    const MAX_RESIDENT_HINT: usize = super::super::MAX_RESIDENT_TILES;
}
