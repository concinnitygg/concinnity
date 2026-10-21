//! Where a planet's blades grow: every cube face divided into one fixed grid
//! of cells, a candidate blade in each. A blade's cell is counted across its
//! whole face, so its place, its hash and everything drawn from the hash
//! depend only on the planet, never on the frame the world is simulated in or
//! on which tiles happen to be drawn.
//!
//! The cells nest in the quadtree the planet is drawn with: every tile, at any
//! level, covers a whole square of grass tiles, and every grass tile a whole
//! square of cells. A blade belongs to exactly one drawn tile, whose mesh it
//! roots on.

#[cfg(test)]
use crate::planet::TileId;
use crate::planet::{CubeFace, FINEST_CELL_SIZE, PlanetShape, TileLod};
use crate::render::environment::grass::tiles::{GRASS_TILE_SIZE, MAX_GRASS_DENSITY};
use concinnity_core::math::{SaturatingFrom, dvec3, narrow_f32};

/// How a planet's faces divide into grass tiles, the unit the kernel culls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanetGrassGrid {
    /// The planet's radius, in meters.
    pub radius: f64,
    /// The deepest level of the quadtree the planet is drawn with.
    pub max_level: u8,
    /// Grass tiles along each side of a tile at `max_level`.
    pub finest_tiles: u32,
    /// The most a patch of the face is stretched on the sphere: the largest
    /// area one unit of face coordinates squared covers, over the area it
    /// covers at the face's middle.
    pub max_stretch: f64,
}

impl PlanetGrassGrid {
    /// The grid for `shape`'s faces, drawn to the quadtree depth the planet's
    /// tiles reach.
    pub fn new(shape: &PlanetShape) -> Self {
        let max_level = TileLod::new(*shape, FINEST_CELL_SIZE).max_level;
        let finest = shape.radius * core::f64::consts::FRAC_PI_2 / f64::from(1u32 << max_level);
        let finest_tiles =
            u32::saturating_from(libm::round(finest / f64::from(GRASS_TILE_SIZE)).max(1.0));
        Self {
            radius: shape.radius,
            max_level,
            finest_tiles,
            max_stretch: max_face_stretch(),
        }
    }

    /// Grass tiles along each side of a tile at `level`.
    pub fn tiles_per_side(&self, level: u8) -> u32 {
        self.finest_tiles << self.max_level.saturating_sub(level)
    }

    /// Grass tiles along each side of a whole face.
    fn face_tiles(&self) -> u32 {
        self.tiles_per_side(0)
    }

    /// How wide one grass tile is across the face's middle, in meters.
    pub fn tile_size(&self) -> f64 {
        self.radius * core::f64::consts::FRAC_PI_2 / f64::from(self.face_tiles())
    }

    /// Cells along each side of a grass tile for `density` blades per square
    /// meter, so that no cell anywhere on the face covers more than
    /// `1 / density` of ground. `None` when the density places nothing.
    pub fn cells_per_tile(&self, density: f32) -> Option<u32> {
        if density.is_nan() || density <= 0.0 {
            return None;
        }
        let density = f64::from(density.min(MAX_GRASS_DENSITY));
        let widest = self.tile_size() * libm::sqrt(self.max_stretch);
        Some(u32::saturating_from(
            libm::ceil(widest * libm::sqrt(density)).max(1.0),
        ))
    }

    /// How wide one of `cells` cells per grass tile is across the face's
    /// middle, in meters.
    pub fn cell_size(&self, cells: u32) -> f32 {
        narrow_f32(self.tile_size() / f64::from(cells.max(1)))
    }

    /// The cell of the face at grass tile `grass_tile` of `tile`, cell `cell`
    /// within it, for `cells` cells per grass tile, counted from the face's
    /// `-s`, `-t` corner. Mirrors the kernel's `grass_planet_cell`.
    #[cfg(test)]
    fn face_cell(
        &self,
        tile: TileId,
        grass_tile: [u32; 2],
        cell: [u32; 2],
        cells: u32,
    ) -> [i64; 2] {
        let per = i64::from(self.tiles_per_side(tile.level));
        let at = |t: u32, g: u32, c: u32| {
            (i64::from(t) * per + i64::from(g)) * i64::from(cells) + i64::from(c)
        };
        [
            at(tile.x, grass_tile[0], cell[0]),
            at(tile.y, grass_tile[1], cell[1]),
        ]
    }

    /// Cells along each side of a whole face, for `cells` cells per grass
    /// tile.
    #[cfg(test)]
    fn face_cells(&self, cells: u32) -> u64 {
        u64::from(self.face_tiles()) * u64::from(cells)
    }
}

// The area the patch around face coordinates (`s`, `t`) covers on the unit
// sphere per unit of face coordinates squared.
fn area_scale(face: CubeFace, s: f64, t: f64) -> f64 {
    let h = 1e-4;
    let ds = dvec3::sub(face.direction(s + h, t), face.direction(s - h, t));
    let dt = dvec3::sub(face.direction(s, t + h), face.direction(s, t - h));
    dvec3::length(dvec3::cross(ds, dt)) / (4.0 * h * h)
}

// The largest area scale anywhere on a face over the one at its middle, from
// a dense sampling plus a margin for what lies between the samples. Every face
// is the same, so the first stands for all.
fn max_face_stretch() -> f64 {
    const SAMPLES: u32 = 32;
    let face = CubeFace(0);
    let middle = area_scale(face, 0.0, 0.0);
    let mut most: f64 = 0.0;
    for j in 0..=SAMPLES {
        for i in 0..=SAMPLES {
            let s = -1.0 + 2.0 * f64::from(i) / f64::from(SAMPLES);
            let t = -1.0 + 2.0 * f64::from(j) / f64::from(SAMPLES);
            most = most.max(area_scale(face, s, t));
        }
    }
    most / middle * 1.02
}

/// The unit direction through face coordinates (`s`, `t`) of `face`, in
/// single precision: the kernel's `grass_face_direction`, line for line.
/// Agrees with [`CubeFace::direction`] to single precision.
pub fn face_direction(face: CubeFace, s: f32, t: f32) -> [f32; 3] {
    let [n, u, v] = face.axes();
    let c: [f32; 3] =
        core::array::from_fn(|i| narrow_f32(n[i]) + narrow_f32(u[i]) * s + narrow_f32(v[i]) * t);
    let [x, y, z] = c;
    let (x2, y2, z2) = (x * x, y * y, z * z);
    let p = [
        x * concinnity_core::math::sqrt((1.0 - y2 * 0.5 - z2 * 0.5 + y2 * z2 / 3.0).max(0.0)),
        y * concinnity_core::math::sqrt((1.0 - z2 * 0.5 - x2 * 0.5 + z2 * x2 / 3.0).max(0.0)),
        z * concinnity_core::math::sqrt((1.0 - x2 * 0.5 - y2 * 0.5 + x2 * y2 / 3.0).max(0.0)),
    ];
    concinnity_core::math::vec3::normalize_or(p, 1e-12, [0.0, 1.0, 0.0])
}

/// The face and face coordinates (`s`, `t`) the unit direction `dir` passes
/// through: [`face_direction`] inverted, in single precision, the shader's
/// `grass_face_coords` line for line. The face is the cube axis `dir` leans
/// along most.
#[cfg(test)]
fn face_coords(dir: [f32; 3]) -> (CubeFace, [f32; 2]) {
    let a = dir.map(f32::abs);
    let face = if a[0] >= a[1] && a[0] >= a[2] {
        u8::from(dir[0] < 0.0)
    } else if a[1] >= a[2] {
        if dir[1] >= 0.0 { 2 } else { 3 }
    } else if dir[2] >= 0.0 {
        4
    } else {
        5
    };
    let face = CubeFace(face);
    let [_, u, v] = face.axes();
    let x: f32 = (0..3).map(|i| dir[i] * narrow_f32(u[i])).sum();
    let y: f32 = (0..3).map(|i| dir[i] * narrow_f32(v[i])).sum();
    // The spherify solved for the cube point, written so that no term
    // cancels near the face's middle.
    let (x2, y2) = (2.0 * x * x, 2.0 * y * y);
    let d = (3.0 + x2 - y2) * (3.0 + x2 - y2) - 12.0 * x2;
    let root = concinnity_core::math::sqrt(d.max(0.0));
    let s = x * concinnity_core::math::sqrt(12.0 / (3.0 + x2 - y2 + root));
    let t = y * concinnity_core::math::sqrt(12.0 / (3.0 - x2 + y2 + root));
    (face, [s.clamp(-1.0, 1.0), t.clamp(-1.0, 1.0)])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> PlanetShape {
        PlanetShape {
            center: [0.0, -50_000.0, 0.0],
            radius: 50_000.0,
            amplitude: 40.0,
            feature_size: 2_000.0,
            octaves: 8,
            seed: 4,
            continents: crate::planet::Continents::NONE,
        }
    }

    #[test]
    fn grass_tiles_are_about_four_meters_and_nest_in_every_level() {
        let g = PlanetGrassGrid::new(&shape());
        assert_eq!(g.max_level, 13);
        assert_eq!(g.finest_tiles, 2);
        let size = g.tile_size();
        assert!(size > 3.0 && size < 5.0, "{size}");
        for level in 0..=g.max_level {
            assert_eq!(
                u64::from(g.tiles_per_side(level)) << level,
                u64::from(g.face_tiles())
            );
        }
    }

    // The spherified cube spreads its cells evenly, but not perfectly: the
    // most stretched patch is a little larger than the middle's.
    #[test]
    fn the_stretch_is_modest_and_bounds_every_patch() {
        let stretch = max_face_stretch();
        assert!(stretch > 1.0 && stretch < 1.6, "{stretch}");
        let middle = area_scale(CubeFace(3), 0.0, 0.0);
        for (s, t) in [(1.0, 1.0), (-0.97, 0.33), (0.5, -0.5), (0.0, 0.999)] {
            assert!(area_scale(CubeFace(3), s, t) <= middle * stretch);
        }
    }

    // No cell anywhere holds more ground than one blade's share, so thinning
    // each cell by its own area gives exactly the asked density.
    #[test]
    fn no_cell_covers_more_than_one_blades_share() {
        let shape = shape();
        let g = PlanetGrassGrid::new(&shape);
        for density in [1.0, 40.0, 100.0, 400.0] {
            let cells = g.cells_per_tile(density).unwrap();
            let n = g.face_cells(cells) as f64;
            let middle = area_scale(CubeFace(0), 0.0, 0.0);
            let widest = middle * g.max_stretch / 1.02 * (2.0 / n) * (2.0 / n);
            let area = widest * shape.radius * shape.radius;
            assert!(area <= 1.0 / f64::from(density) + 1e-9, "{density}: {area}");
            assert!(area > 0.5 / f64::from(density), "{density}: {area}");
        }
        assert_eq!(g.cells_per_tile(0.0), None);
        assert_eq!(g.cells_per_tile(f32::NAN), None);
        assert_eq!(g.cells_per_tile(1e9), g.cells_per_tile(MAX_GRASS_DENSITY));
    }

    // A cell is the same cell whichever level's tile is drawn over it.
    #[test]
    fn a_cell_keeps_its_place_across_levels() {
        let g = PlanetGrassGrid::new(&shape());
        let cells = 30;
        let fine = TileId {
            face: CubeFace(2),
            level: 12,
            x: 2049,
            y: 2046,
        };
        let parent = fine.parent().unwrap();
        let a = g.face_cell(fine, [3, 4], [7, 29], cells);
        let per = g.tiles_per_side(12);
        let b = g.face_cell(parent, [per + 3, 4], [7, 29], cells);
        assert_eq!(a, b);
        assert!(u64::saturating_from(a[0]) < g.face_cells(cells));
        // Every cell of a face fits the kernel's signed 32-bit coordinates.
        assert!(g.face_cells(g.cells_per_tile(MAX_GRASS_DENSITY).unwrap()) < 1 << 31);
    }

    // Inverting the spherify lands back on the cell the kernel placed a
    // blade in: within a centimeter on a 50 km planet, everywhere on every
    // face, including its middle and its edges.
    #[test]
    fn face_coords_invert_the_direction() {
        let extent = 50_000.0 * core::f64::consts::FRAC_PI_2;
        for face in CubeFace::ALL {
            for i in 0..=40 {
                for j in 0..=40 {
                    let s = -1.0 + f64::from(i) / 20.0;
                    let t = -1.0 + f64::from(j) / 20.0;
                    let (s, t) = (s + 1e-5 * f64::from(j), t - 1e-6 * f64::from(i));
                    let dir = face.direction(s.clamp(-1.0, 1.0), t.clamp(-1.0, 1.0));
                    let (got, st) = face_coords(dvec3::to_f32(dir));
                    if i == 0 || i == 40 || j == 0 || j == 40 {
                        // Edges belong to either face; the point must agree.
                        let back = face_direction(got, st[0], st[1]);
                        let miss = (0..3).map(|k| (f64::from(back[k]) - dir[k]).abs());
                        assert!(miss.fold(0.0, f64::max) < 1e-6, "{face:?} {s} {t}");
                        continue;
                    }
                    assert_eq!(got, face, "{s} {t}");
                    let ds = (f64::from(st[0]) - s.clamp(-1.0, 1.0)) * extent * 0.5;
                    let dt = (f64::from(st[1]) - t.clamp(-1.0, 1.0)) * extent * 0.5;
                    assert!(
                        ds.abs() < 1e-2 && dt.abs() < 1e-2,
                        "{face:?} {s} {t}: {ds} {dt}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_single_precision_direction_matches_the_face() {
        for face in CubeFace::ALL {
            for (s, t) in [(0.0, 0.0), (0.731, -0.2), (-1.0, 1.0), (0.99, 0.01)] {
                let a = face_direction(face, narrow_f32(s), narrow_f32(t));
                let b = face.direction(s, t);
                for i in 0..3 {
                    assert!(
                        (f64::from(a[i]) - b[i]).abs() < 1e-6,
                        "{face:?} {a:?} {b:?}"
                    );
                }
            }
        }
    }
}
