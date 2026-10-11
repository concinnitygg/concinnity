// Which tiles of a planet's quadtree to keep and which to draw, from where the
// camera is and which tiles have their meshes.

use alloc::vec::Vec;

use super::dvec::{self, DVec3};
use super::shape::{CubeFace, PlanetShape};
use super::tile::{MAX_TILE_LEVEL, TILE_CELLS, TileBounds, TileId, tile_bounds};

/// A tile splits into its children while the camera is nearer than this many
/// tile widths.
pub const SPLIT_WIDTHS: f64 = 1.0;

/// A split tile's children stay resident until the camera is this many tile
/// widths away, so hovering at the split distance does not rebuild them.
pub const RETAIN_WIDTHS: f64 = 1.3;

/// Most tiles a [`TileBoundsCache`] holds before it starts over: several times
/// what any camera visits, so a steady camera never recomputes one.
const MAX_CACHED_BOUNDS: usize = 8 * super::MAX_RESIDENT_TILES;

/// Each tile's bounds on one planet, computed the first time it is asked for.
///
/// Bounds depend only on the planet and the tile, so a camera that revisits
/// the quadtree every frame pays for each tile once.
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
            bounds: hashbrown::HashMap::new(),
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

    /// How many tiles' bounds are held.
    pub fn len(&self) -> usize {
        self.bounds.len()
    }

    /// Whether no tile's bounds are held.
    pub fn is_empty(&self) -> bool {
        self.bounds.is_empty()
    }
}

/// How fine a planet's tiles get: the camera's distance decides each tile's
/// level, down to `max_level`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileLod {
    /// The surface the tiles cover.
    pub shape: PlanetShape,
    /// The deepest level any tile reaches.
    pub max_level: u8,
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
        }
    }

    // Whether `tile` is nearer `camera` than `widths` of its own width.
    fn near(&self, tile: TileId, camera: DVec3, widths: f64, bounds: &mut TileBoundsCache) -> bool {
        if tile.level >= self.max_level {
            return false;
        }
        let b = bounds.get(tile);
        let gap = dvec::length(dvec::sub(camera, b.center)) - b.radius;
        gap < widths * b.width
    }

    /// The tiles the camera at authored point `camera` wants, given which are
    /// `resident`, reading each tile's bounds from `bounds`.
    ///
    /// A tile near the camera splits into its children once all four are
    /// resident; until then it draws itself. The six face roots are never
    /// split past while not resident, so nothing is drawn for a face whose
    /// root is missing, and a drawn set never overlaps or leaves a hole.
    pub fn select(
        &self,
        camera: DVec3,
        resident: &dyn Fn(TileId) -> bool,
        bounds: &mut TileBoundsCache,
    ) -> TileSelection {
        let mut out = TileSelection::default();
        let mut visit = Visit {
            lod: self,
            camera,
            resident,
            bounds,
            out: &mut out,
        };
        for face in CubeFace::ALL {
            let root = TileId::root(face);
            visit.tile(root, resident(root));
        }
        out
    }

    /// The tiles worth keeping resident for a camera at `camera`: every tile
    /// [`Self::select`] wants, plus the children of tiles just past their
    /// split distance.
    pub fn retained(&self, camera: DVec3, bounds: &mut TileBoundsCache) -> Vec<TileId> {
        let mut out = Vec::new();
        for face in CubeFace::ALL {
            self.retain(TileId::root(face), camera, bounds, &mut out);
        }
        out
    }

    fn retain(
        &self,
        tile: TileId,
        camera: DVec3,
        bounds: &mut TileBoundsCache,
        out: &mut Vec<TileId>,
    ) {
        out.push(tile);
        if self.near(tile, camera, RETAIN_WIDTHS, bounds) {
            for child in tile.children() {
                self.retain(child, camera, bounds, out);
            }
        }
    }
}

// One walk of the quadtree for `TileLod::select`.
struct Visit<'a> {
    lod: &'a TileLod,
    camera: DVec3,
    resident: &'a dyn Fn(TileId) -> bool,
    bounds: &'a mut TileBoundsCache,
    out: &'a mut TileSelection,
}

impl Visit<'_> {
    fn tile(&mut self, tile: TileId, drawable: bool) {
        self.out.wanted.push(tile);
        let split = self.lod.near(tile, self.camera, SPLIT_WIDTHS, self.bounds);
        let children = tile.children();
        let ready = split && children.iter().all(|&c| (self.resident)(c));
        if drawable && !ready {
            self.out.draw.push(tile);
        }
        if split {
            for child in children {
                self.tile(child, drawable && ready);
            }
        }
    }
}

/// One frame's choice of tiles.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TileSelection {
    /// The tiles to draw: together they cover every face whose root is
    /// resident, once.
    pub draw: Vec<TileId>,
    /// Every tile the camera wants: the ones not yet resident are the ones to
    /// build.
    pub wanted: Vec<TileId>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeSet;

    fn lod() -> TileLod {
        TileLod::new(
            PlanetShape {
                center: [0.0, -50_000.0, 0.0],
                radius: 50_000.0,
                amplitude: 60.0,
                feature_size: 2_000.0,
                octaves: 8,
                seed: 1,
            },
            1.0,
        )
    }

    // A camera standing on the planet's top.
    const STANDING: DVec3 = [0.0, 62.0, 0.0];

    fn cache(l: &TileLod) -> TileBoundsCache {
        TileBoundsCache::new(l.shape)
    }

    // The cache hands back exactly what the bounds function computes, and a
    // walk asks for each tile once.
    #[test]
    fn the_cache_returns_the_computed_bounds_once_per_tile() {
        let l = lod();
        let mut c = cache(&l);
        let sel = l.select(STANDING, &|_| true, &mut c);
        let held = c.len();
        assert!(held > 0 && held <= sel.wanted.len());
        let again = l.select(STANDING, &|_| true, &mut c);
        assert_eq!(again, sel);
        assert_eq!(c.len(), held, "a second walk finds every tile cached");
        for &t in &sel.wanted {
            assert_eq!(c.get(t), tile_bounds(&l.shape, t));
        }
    }

    #[test]
    fn the_finest_cells_meet_the_requested_size() {
        let l = lod();
        let face = 50_000.0 * core::f64::consts::FRAC_PI_2;
        let finest = face / f64::from(1u32 << l.max_level) / f64::from(TILE_CELLS);
        assert!(finest <= 1.0 && finest * 2.0 > 1.0, "{finest}");
    }

    // With every tile resident, the drawn tiles reach the finest level under
    // the camera and cover each face exactly once.
    #[test]
    fn a_fully_resident_selection_covers_every_face_once() {
        let l = lod();
        let sel = l.select(STANDING, &|_| true, &mut cache(&l));
        let deepest = sel.draw.iter().map(|t| t.level).max().unwrap();
        assert_eq!(deepest, l.max_level);
        for face in CubeFace::ALL {
            let area: f64 = sel
                .draw
                .iter()
                .filter(|t| t.face == face)
                .map(|t| 1.0 / f64::from(1u32 << (2 * u32::from(t.level))))
                .sum();
            assert!((area - 1.0).abs() < 1e-12, "face {face:?} covered {area}");
        }
        // No drawn tile is an ancestor of another.
        let drawn: BTreeSet<TileId> = sel.draw.iter().copied().collect();
        for t in &sel.draw {
            let mut up = t.parent();
            while let Some(p) = up {
                assert!(!drawn.contains(&p));
                up = p.parent();
            }
        }
    }

    // A tile whose children are not all resident draws itself instead, and
    // nothing beneath it is drawn.
    #[test]
    fn a_split_waits_for_all_four_children() {
        let l = lod();
        let full = l.select(STANDING, &|_| true, &mut cache(&l));
        let level = 3;
        let resident = |t: TileId| t.level <= level;
        let sel = l.select(STANDING, &resident, &mut cache(&l));
        assert!(sel.draw.iter().all(|t| t.level <= level));
        assert!(sel.draw.iter().any(|t| t.level == level));
        assert_eq!(sel.wanted, full.wanted, "what is wanted ignores residency");
        // Without the root, a face draws nothing.
        let sel = l.select(STANDING, &|t: TileId| t.face != CubeFace(2), &mut cache(&l));
        assert!(sel.draw.iter().all(|t| t.face != CubeFace(2)));
    }

    // The retained set holds everything wanted, plus the band past it.
    #[test]
    fn retention_holds_the_wanted_tiles_and_a_band_past_them() {
        let l = lod();
        let wanted: BTreeSet<TileId> = l
            .select(STANDING, &|_| true, &mut cache(&l))
            .wanted
            .into_iter()
            .collect();
        let retained: BTreeSet<TileId> = l.retained(STANDING, &mut cache(&l)).into_iter().collect();
        assert!(wanted.is_subset(&retained));
        assert!(retained.len() > wanted.len());
    }

    // Wherever the camera stands or flies, the tile count stays bounded: the
    // pool the meshes live in is sized from this.
    #[test]
    fn the_retained_count_stays_bounded() {
        let l = lod();
        let mut most = 0;
        for i in 0..24 {
            let a = i as f64 * 0.27;
            let up = dvec::normalize_or([libm::sin(a), libm::cos(a), libm::sin(a * 3.1)], [0.0; 3]);
            for height in [2.0, 62.0, 500.0, 5_000.0] {
                let cam = dvec::add(l.shape.center, dvec::scale(up, 50_000.0 + height));
                most = most.max(l.retained(cam, &mut cache(&l)).len());
            }
        }
        assert!(most <= MAX_RESIDENT_HINT, "{most}");
    }

    const MAX_RESIDENT_HINT: usize = super::super::MAX_RESIDENT_TILES;
}
