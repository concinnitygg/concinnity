// One square of a planet's surface: a quadtree node on one cube face, and the
// mesh a worker builds for it from the planet's shape.

use alloc::vec::Vec;
use concinnity_core::math::{SaturatingFrom, dvec3, narrow_f32};

use super::shape::{CubeFace, Ground, PlanetShape};
use crate::gfx::mesh_payload::Vertex;
use concinnity_core::math::dvec3::DVec3;

/// Cells along each side of a tile's mesh, at every level.
pub const TILE_CELLS: u32 = 16;

/// Deepest quadtree level a tile can sit at.
pub const MAX_TILE_LEVEL: u8 = 20;

// Texture coordinates are meters across the face, wrapped to this period so
// they stay precise far across a face; a texture repeating at a whole fraction
// of it tiles seamlessly over every tile.
const UV_PERIOD_M: f64 = 1024.0;

// Directions along each side of a tile its bounds sample the ground at.
const BOUND_SAMPLES: usize = 5;

/// One side of a tile, by the face coordinate it lies across.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TileSide {
    /// The side at the tile's lowest `s`.
    MinS,
    /// The side at the tile's highest `s`.
    MaxS,
    /// The side at the tile's lowest `t`.
    MinT,
    /// The side at the tile's highest `t`.
    MaxT,
}

impl TileSide {
    /// All four sides.
    pub const ALL: [Self; 4] = [Self::MinS, Self::MaxS, Self::MinT, Self::MaxT];

    /// The side's bit in a mask of sides: `1 << index` in [`Self::ALL`]
    /// order.
    pub fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// A tile: face `face`, quadtree `level` (0 is the whole face), and its
/// column `x` and row `y` among the `2^level` across that level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TileId {
    /// The cube face the tile lies on.
    pub face: CubeFace,
    /// The quadtree level: 0 covers the face, each level halves the side.
    pub level: u8,
    /// Column along the face's `s` coordinate.
    pub x: u32,
    /// Row along the face's `t` coordinate.
    pub y: u32,
}

impl TileId {
    /// The whole of `face`.
    pub fn root(face: CubeFace) -> Self {
        Self {
            face,
            level: 0,
            x: 0,
            y: 0,
        }
    }

    /// The four tiles one level down that cover this one.
    pub fn children(self) -> [Self; 4] {
        let level = self.level + 1;
        let (x, y) = (self.x * 2, self.y * 2);
        [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(dx, dy)| Self {
            face: self.face,
            level,
            x: x + dx,
            y: y + dy,
        })
    }

    /// The tile one level up that covers this one, or `None` for a root.
    pub fn parent(self) -> Option<Self> {
        (self.level > 0).then(|| Self {
            face: self.face,
            level: self.level - 1,
            x: self.x / 2,
            y: self.y / 2,
        })
    }

    /// The tile at the same level across `side`, on this face or, past its
    /// edge, the next.
    #[must_use]
    pub fn neighbor(self, side: TileSide) -> Self {
        let last = (1u32 << self.level) - 1;
        let within = match side {
            TileSide::MinS => self.x.checked_sub(1).map(|x| (x, self.y)),
            TileSide::MaxS => (self.x < last).then(|| (self.x + 1, self.y)),
            TileSide::MinT => self.y.checked_sub(1).map(|y| (self.x, y)),
            TileSide::MaxT => (self.y < last).then(|| (self.x, self.y + 1)),
        };
        if let Some((x, y)) = within {
            return Self { x, y, ..self };
        }
        let (s0, t0, span) = self.span();
        // A point just across the side's middle, so it lands in the
        // neighbor's own row or column on whichever face it lies.
        let past = span * (0.5 + 1e-3);
        let (mut s, mut t) = (s0 + span * 0.5, t0 + span * 0.5);
        match side {
            TileSide::MinS => s -= past,
            TileSide::MaxS => s += past,
            TileSide::MinT => t -= past,
            TileSide::MaxT => t += past,
        }
        let (face, s, t) = CubeFace::locate(self.face.cube_point(s, t));
        let index = |c: f64| u32::saturating_from(libm::floor((c + 1.0) / span).max(0.0)).min(last);
        Self {
            face,
            level: self.level,
            x: index(s),
            y: index(t),
        }
    }

    /// The tile at `level` that covers this one: itself at its own level,
    /// `None` below it.
    pub fn ancestor_at(self, level: u8) -> Option<Self> {
        let up = self.level.checked_sub(level)?;
        Some(Self {
            face: self.face,
            level,
            x: self.x >> up,
            y: self.y >> up,
        })
    }

    /// Whether the tile names a square of its face: its level at most
    /// [`MAX_TILE_LEVEL`], its column and row within it.
    pub fn is_valid(self) -> bool {
        self.face.0 < 6
            && self.level <= MAX_TILE_LEVEL
            && self.x < 1u32 << self.level
            && self.y < 1u32 << self.level
    }

    // The face coordinates the tile spans: (s0, t0, side).
    pub(crate) fn span(self) -> (f64, f64, f64) {
        let side = 2.0 / f64::from(1u32 << self.level);
        (
            -1.0 + f64::from(self.x) * side,
            -1.0 + f64::from(self.y) * side,
            side,
        )
    }

    /// The direction through the point `(fs, ft)` of the tile, each in
    /// `[0, 1]` across it.
    pub fn direction(self, fs: f64, ft: f64) -> DVec3 {
        let (s0, t0, side) = self.span();
        self.face.direction(s0 + fs * side, t0 + ft * side)
    }
}

/// Where a tile sits: the sphere around which every point of it lies, and how
/// wide it is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileBounds {
    /// The center of the bounding sphere, in authored coordinates.
    pub center: DVec3,
    /// The bounding sphere's radius, in meters.
    pub radius: f64,
    /// The length of the tile's side along the ground, in meters.
    pub width: f64,
}

/// The bounds of `tile` on `shape`: the ground's height sampled across the
/// tile, widened by how far it can rise between samples.
pub fn tile_bounds(shape: &PlanetShape, tile: TileId) -> TileBounds {
    let n = BOUND_SAMPLES - 1;
    let over = shape.edits_over(tile);
    let samples: [(DVec3, f64); BOUND_SAMPLES * BOUND_SAMPLES] = core::array::from_fn(|k| {
        let (i, j) = (k % BOUND_SAMPLES, k / BOUND_SAMPLES);
        let (dir, ground) = shape.ground_on(
            tile,
            over.as_ref(),
            i as f64 / n as f64,
            j as f64 / n as f64,
        );
        (dir, ground.height)
    });
    let dirs = samples.map(|(d, _)| d);
    let at = |i: usize, j: usize| dirs[j * BOUND_SAMPLES + i];
    // The widest step between neighboring samples, along the ground.
    let arc = |a: DVec3, b: DVec3| shape.radius * libm::acos(dvec3::dot(a, b).clamp(-1.0, 1.0));
    let mut spacing: f64 = 0.0;
    for j in 0..=n {
        for i in 0..n {
            spacing = spacing
                .max(arc(at(i, j), at(i + 1, j)))
                .max(arc(at(j, i), at(j, i + 1)));
        }
    }
    // Any point lies within half a sample cell's diagonal of a sample; the
    // sphere bulges past the chord between samples by the sagitta.
    let reach = spacing * core::f64::consts::FRAC_1_SQRT_2;
    let slack = shape.max_slope() * reach;
    let sagitta = spacing * spacing / (8.0 * shape.radius);
    let (lo, hi) = samples
        .iter()
        .map(|&(_, h)| h)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), h| {
            (lo.min(h), hi.max(h))
        });
    let lo = (lo - slack).max(shape.lowest_height());
    let hi = (hi + slack).min(shape.highest_height());

    let center = dvec3::add(
        shape.center,
        dvec3::scale(tile.direction(0.5, 0.5), shape.radius + (lo + hi) * 0.5),
    );
    let mut radius: f64 = 0.0;
    for &d in &dirs {
        for r in [shape.radius + lo, shape.radius + hi] {
            let p = dvec3::add(shape.center, dvec3::scale(d, r));
            radius = radius.max(dvec3::length(dvec3::sub(p, center)));
        }
    }
    let a = tile.direction(0.0, 0.5);
    let b = tile.direction(1.0, 0.5);
    TileBounds {
        center,
        radius: radius + sagitta,
        width: arc(a, b),
    }
}

/// A tile's render mesh, positioned relative to its `origin`.
#[derive(Debug, Clone)]
pub struct TileMesh {
    /// The point the vertices are relative to, in authored coordinates.
    pub origin: DVec3,
    /// The surface's `(TILE_CELLS + 1)^2` corners, row-major along `t`.
    pub vertices: Vec<Vertex>,
    /// Two triangles per cell wound to face outward.
    pub indices: Vec<u16>,
    /// The farthest any corner moves when the tile morphs all the way into
    /// its parent's mesh, in meters.
    pub morph_extent: f32,
    /// The lowest and highest the surface's corners lie along the up at
    /// `origin`, in meters from it: every point of the surface lies between.
    pub radial: [f32; 2],
    /// How high `origin` lies above the planet's radius, in meters.
    pub origin_altitude: f32,
    /// The box around every vertex, relative to `origin` with authored axes.
    pub bounds: ([f32; 3], [f32; 3]),
}

/// The mesh of `tile` on `shape`.
///
/// Normals are the surface's exact ones, from the height's own gradient, so
/// an edge corner shades the same in every tile that shares it, at any level.
/// Each corner's color carries what a shader rebuilding the normal from the
/// generated hills needs of the edits there: the detail weight, then the
/// gradient the edits add beyond the weighted hills, along the corner's
/// tangent and the direction crossed with it.
/// Each cell splits along the same diagonal at every level, so a tile morphed
/// all the way toward its parent is its parent's mesh, which is how it meets a
/// coarser neighbor without a gap.
pub fn tile_mesh(shape: &PlanetShape, tile: TileId) -> TileMesh {
    let n = TILE_CELLS as usize;
    let side = n + 1;
    let over = shape.edits_over(tile);
    let origin = {
        let (dir, ground) = shape.ground_on(tile, over.as_ref(), 0.5, 0.5);
        shape.point_and_normal(dir, &ground).0
    };
    let (s0, t0, span) = tile.span();
    // Meters across the face from its -s/-t corner, less a whole number of
    // periods so the tile's own stay small.
    let face_m = |c: f64| (c + 1.0) * 0.5 * shape.radius * core::f64::consts::FRAC_PI_2;
    let base = [s0, t0].map(|c| libm::floor(face_m(c) / UV_PERIOD_M) * UV_PERIOD_M);
    let uv = |i: usize, j: usize| {
        [
            narrow_f32(face_m(s0 + i as f64 / n as f64 * span) - base[0]),
            narrow_f32(face_m(t0 + j as f64 / n as f64 * span) - base[1]),
        ]
    };
    let u_axis = tile.face.u_axis();

    let mut points = Vec::with_capacity(side * side);
    let mut vertices = Vec::with_capacity(side * side);
    for j in 0..side {
        for i in 0..side {
            let (dir, ground) = shape.ground_on(
                tile,
                over.as_ref(),
                i as f64 / n as f64,
                j as f64 / n as f64,
            );
            let (p, normal) = shape.point_and_normal(dir, &ground);
            let tangent = dvec3::normalize_or(
                dvec3::sub(u_axis, dvec3::scale(normal, dvec3::dot(normal, u_axis))),
                dvec3::cross(normal, [0.0, 0.0, 1.0]),
            );
            points.push(p);
            vertices.push(Vertex {
                pos: dvec3::to_f32(dvec3::sub(p, origin)),
                normal: dvec3::to_f32(normal),
                tangent: dvec3::to_f32(tangent),
                color: edit_color(dir, tangent, &ground),
                uv: uv(i, j),
            });
        }
    }
    let up = shape.up_at(origin);
    let radial = vertices
        .iter()
        .fold([f32::INFINITY, f32::NEG_INFINITY], |[lo, hi], v| {
            let h = narrow_f32(dvec3::dot(dvec3::from_f32(v.pos), up));
            [lo.min(h), hi.max(h)]
        });

    let indices = tile_indices(1);

    let morph_extent = morph_extent(&points, n);

    let bounds = vertices.iter().fold(
        ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]),
        |(lo, hi), v| {
            (
                core::array::from_fn(|i| lo[i].min(v.pos[i])),
                core::array::from_fn(|i| hi[i].max(v.pos[i])),
            )
        },
    );
    TileMesh {
        origin,
        vertices,
        indices,
        morph_extent,
        radial,
        origin_altitude: narrow_f32(dvec3::length(dvec3::sub(origin, shape.center)) - shape.radius),
        bounds,
    }
}

/// Corners a capture's coarse tile steps across per cell: a divisor of
/// [`TILE_CELLS`] and a power of two, so every corner it keeps is one no level's
/// geomorph moves.
pub const CAPTURE_TILE_STRIDE: usize = 4;

/// A tile's triangles over its `(TILE_CELLS + 1)^2` corners, row-major along
/// `t`, with cells `stride` corners across: two triangles per cell, wound to
/// face outward and split along the diagonal every level uses. `stride` 1 is
/// the tile's own mesh; a larger one reaches only the corners on its grid.
pub fn tile_indices(stride: usize) -> Vec<u16> {
    let n = TILE_CELLS as usize;
    let side = n + 1;
    let cells = n / stride.max(1);
    let mut indices = Vec::with_capacity(cells * cells * 6);
    let corner = |i: usize, j: usize| u16::saturating_from(j * stride * side + i * stride);
    for j in 0..cells {
        for i in 0..cells {
            let (a, b, c, d) = (
                corner(i, j),
                corner(i + 1, j),
                corner(i, j + 1),
                corner(i + 1, j + 1),
            );
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    indices
}

/// The color of a tile corner in unit direction `dir` with tangent
/// `tangent` on the ground `ground`: the detail weight, then the gradient the
/// edits add beyond the weighted hills along the frame a shader rebuilds from
/// the direction and the tangent: the tangent with the direction taken out of
/// it, and the direction crossed with that.
pub(crate) fn edit_color(dir: DVec3, tangent: DVec3, ground: &Ground) -> [f32; 3] {
    let [gx, gy] = edit_frame(dir, tangent).map(|axis| dvec3::dot(ground.correction, axis));
    [narrow_f32(ground.weight), narrow_f32(gx), narrow_f32(gy)]
}

/// The frame a tile corner's edit gradient is held in: `tangent` with unit
/// direction `dir` taken out of it, and `dir` crossed with that.
pub(crate) fn edit_frame(dir: DVec3, tangent: DVec3) -> [DVec3; 2] {
    let along = dvec3::normalize_or(
        dvec3::sub(tangent, dvec3::scale(dir, dvec3::dot(dir, tangent))),
        [0.0; 3],
    );
    [along, dvec3::cross(dir, along)]
}

// The farthest any corner of an `n`-cell grid of `points` lies from the middle
// of the two parent corners it morphs toward.
fn morph_extent(points: &[DVec3], n: usize) -> f32 {
    let side = n + 1;
    let mut most: f64 = 0.0;
    for j in 0..side {
        for i in 0..side {
            if let Some([a, b]) = super::morph::parent_span(i, j) {
                let mid = dvec3::scale(
                    dvec3::add(points[a.1 * side + a.0], points[b.1 * side + b.0]),
                    0.5,
                );
                most = most.max(dvec3::length(dvec3::sub(mid, points[j * side + i])));
            }
        }
    }
    narrow_f32(most)
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::math::vec3::{cross, dot, sub};

    fn shape() -> PlanetShape {
        PlanetShape {
            center: [0.0, -50_000.0, 0.0],
            radius: 50_000.0,
            amplitude: 40.0,
            feature_size: 2_000.0,
            octaves: 6,
            seed: 3,
            continents: crate::planet::Continents::NONE,
            edits: None,
        }
    }

    fn deep_tile() -> TileId {
        TileId {
            face: CubeFace(2),
            level: 9,
            x: 256,
            y: 255,
        }
    }

    #[test]
    fn children_cover_their_parent() {
        let t = deep_tile();
        let kids = t.children();
        for k in kids {
            assert_eq!(k.parent(), Some(t));
        }
        assert_eq!(TileId::root(CubeFace(1)).parent(), None);
        // The children's outer corners are the parent's corners.
        let a = t.direction(0.0, 0.0);
        let b = kids[0].direction(0.0, 0.0);
        assert!(dvec3::length(dvec3::sub(a, b)) < 1e-15);
        let c = t.direction(1.0, 1.0);
        let d = kids[3].direction(1.0, 1.0);
        assert!(dvec3::length(dvec3::sub(c, d)) < 1e-15);
    }

    // Every surface vertex lies inside its tile's bounds, at every level,
    // across the face, though the bounds pad only by the height's slope.
    #[test]
    fn the_bounds_hold_every_vertex() {
        let s = shape();
        let side = TILE_CELLS as usize + 1;
        for level in [0u8, 3, 9, 14] {
            let n = 1u32 << level;
            for (x, y) in [(0, 0), (n / 2, (n / 2).saturating_sub(1)), (n - 1, n / 3)] {
                let t = TileId {
                    face: CubeFace(2),
                    level,
                    x,
                    y,
                };
                let b = tile_bounds(&s, t);
                let mesh = tile_mesh(&s, t);
                for v in &mesh.vertices[..side * side] {
                    let p = dvec3::add(mesh.origin, dvec3::from_f32(v.pos));
                    let gap = dvec3::length(dvec3::sub(p, b.center)) - b.radius;
                    assert!(gap <= 1e-3 * (1.0 + b.radius * 1e-4), "{t:?} {gap}");
                }
            }
        }
        // A face is 50 km * (pi / 2) across the middle, its 512 tiles a level
        // narrower near its center, where the spherified cube crowds them.
        let b = tile_bounds(&s, deep_tile());
        assert!(b.width > 100.0 && b.width < 153.4, "{}", b.width);
        // A deep tile's bounds hug its own ground, not the planet's whole
        // relief.
        assert!(b.radius < b.width, "{} vs {}", b.radius, b.width);
    }

    // The normal blades read, a tile's corner normals interpolated across a
    // cell as the grass kernel does, tilts within a tenth of a degree of the ground's
    // exact normal on the tiles blades grow on, coasts included, so a slope rule
    // places blades and the ground under them alike.
    #[test]
    fn interpolated_corner_normals_follow_the_exact_normal() {
        let s = PlanetShape {
            octaves: 8,
            continents: crate::planet::Continents::new(50_000.0, 95, 20_000.0, 0.4, 60.0, 60.0),
            ..shape()
        };
        let n = TILE_CELLS as usize;
        let side = n + 1;
        let mut worst = 0.0f64;
        for level in [12u8, 13, 14] {
            let count = 1u32 << level;
            for k in 0..24u32 {
                let tile = TileId {
                    face: CubeFace((k % 6) as u8),
                    level,
                    x: (count / 2 + k * 37) % count,
                    y: (count / 3 + k * 53) % count,
                };
                let mesh = tile_mesh(&s, tile);
                for q in 0..16usize {
                    let (i, j) = ((q * 7) % n, (q * 11) % n);
                    let (rx, ry) = (0.3 + 0.03 * q as f64, 0.6 - 0.025 * q as f64);
                    let at =
                        |a: usize, b: usize| dvec3::from_f32(mesh.vertices[b * side + a].normal);
                    let lerp = |a: DVec3, b: DVec3, t: f64| {
                        dvec3::add(a, dvec3::scale(dvec3::sub(b, a), t))
                    };
                    let normal = dvec3::normalize_or(
                        lerp(
                            lerp(at(i, j), at(i + 1, j), rx),
                            lerp(at(i, j + 1), at(i + 1, j + 1), rx),
                            ry,
                        ),
                        [0.0, 1.0, 0.0],
                    );
                    let dir =
                        tile.direction((i as f64 + rx) / n as f64, (j as f64 + ry) / n as f64);
                    let exact = s.normal(dir);
                    let angle = dvec3::dot(normal, exact)
                        .clamp(-1.0, 1.0)
                        .acos()
                        .to_degrees();
                    worst = worst.max(angle);
                }
            }
        }
        assert!(worst < 0.1, "{worst} degrees");
    }

    // Tiles across a coast, from a whole face down to a few meters, hold
    // their ground between the sea floor and the hills.
    #[test]
    fn the_bounds_hold_every_vertex_across_coasts() {
        let s = PlanetShape {
            continents: crate::planet::Continents::new(50_000.0, 3, 20_000.0, 0.4, 60.0, 60.0),
            ..shape()
        };
        let side = TILE_CELLS as usize + 1;
        let face = CubeFace(2);
        // Walk the face's middle row for the first change from sea to land.
        let coast = (0..4_000)
            .map(|i| -0.9 + f64::from(i) * 4.5e-4)
            .find(|&fs| {
                (s.height(face.direction(fs, 0.0)) > 0.0)
                    != (s.height(face.direction(fs + 4.5e-4, 0.0)) > 0.0)
            })
            .expect("a coast crosses the face");
        let (mut seen_sea, mut seen_land) = (false, false);
        for level in [0u8, 4, 8, 11, 14] {
            let n = 1u32 << level;
            let cell = |c: f64| {
                u32::saturating_from(libm::floor((c + 1.0) * 0.5 * f64::from(n))).min(n - 1)
            };
            let t = TileId {
                face,
                level,
                x: cell(coast),
                y: cell(0.0),
            };
            let b = tile_bounds(&s, t);
            let mesh = tile_mesh(&s, t);
            for v in &mesh.vertices[..side * side] {
                let p = dvec3::add(mesh.origin, dvec3::from_f32(v.pos));
                let gap = dvec3::length(dvec3::sub(p, b.center)) - b.radius;
                assert!(gap <= 1e-3 * (1.0 + b.radius * 1e-4), "{t:?} {gap}");
                let h = dvec3::length(dvec3::sub(p, s.center)) - s.radius;
                seen_sea |= h < 0.0;
                seen_land |= h > 0.0;
            }
        }
        assert!(seen_sea && seen_land);
    }

    // Across a side, the neighbor at the same level shares the side's ends,
    // on this face or the next, and finds this tile across one of its own.
    #[test]
    fn neighbors_share_their_side_on_every_face() {
        let level = 3;
        let n = 1u32 << level;
        for face in CubeFace::ALL {
            for k in 0..n {
                for (x, y) in [(0, k), (n - 1, k), (k, 0), (k, n - 1)] {
                    let t = TileId { face, level, x, y };
                    for side in TileSide::ALL {
                        let across = t.neighbor(side);
                        assert_eq!(across.level, level);
                        assert!(
                            TileSide::ALL.iter().any(|&back| across.neighbor(back) == t),
                            "{t:?} {side:?} -> {across:?}"
                        );
                        let ends = side_ends(t, side);
                        let theirs: Vec<DVec3> = TileSide::ALL
                            .iter()
                            .flat_map(|&s| side_ends(across, s))
                            .collect();
                        for e in ends {
                            assert!(
                                theirs
                                    .iter()
                                    .any(|&o| dvec3::length(dvec3::sub(e, o)) < 1e-12),
                                "{t:?} {side:?} -> {across:?}"
                            );
                        }
                    }
                }
            }
        }
        let inner = TileId {
            face: CubeFace(4),
            level,
            x: 3,
            y: 4,
        };
        assert_eq!(inner.neighbor(TileSide::MinS), TileId { x: 2, ..inner });
        assert_eq!(inner.neighbor(TileSide::MaxT), TileId { y: 5, ..inner });
    }

    fn side_ends(t: TileId, side: TileSide) -> [DVec3; 2] {
        let [a, b] = match side {
            TileSide::MinS => [(0.0, 0.0), (0.0, 1.0)],
            TileSide::MaxS => [(1.0, 0.0), (1.0, 1.0)],
            TileSide::MinT => [(0.0, 0.0), (1.0, 0.0)],
            TileSide::MaxT => [(0.0, 1.0), (1.0, 1.0)],
        };
        [t.direction(a.0, a.1), t.direction(b.0, b.1)]
    }

    #[test]
    fn side_bits_are_distinct() {
        let all = TileSide::ALL.iter().fold(0u8, |m, s| {
            assert_eq!(m & s.bit(), 0);
            m | s.bit()
        });
        assert_eq!(all, 0b1111);
    }

    #[test]
    fn the_mesh_box_holds_every_vertex_tightly() {
        let mesh = tile_mesh(&shape(), deep_tile());
        let (lo, hi) = mesh.bounds;
        for v in &mesh.vertices {
            assert!((0..3).all(|i| v.pos[i] >= lo[i] && v.pos[i] <= hi[i]));
        }
        for i in 0..3 {
            assert!(mesh.vertices.iter().any(|v| v.pos[i] == lo[i]));
            assert!(mesh.vertices.iter().any(|v| v.pos[i] == hi[i]));
        }
    }

    #[test]
    fn the_radial_range_holds_the_surface() {
        let s = shape();
        let mesh = tile_mesh(&s, deep_tile());
        let up = s.up_at(mesh.origin);
        let side = TILE_CELLS as usize + 1;
        let [lo, hi] = mesh.radial;
        assert!(lo < hi);
        for v in &mesh.vertices[..side * side] {
            let h = narrow_f32(dvec3::dot(dvec3::from_f32(v.pos), up));
            assert!(h >= lo && h <= hi);
        }
    }

    // Every surface triangle faces away from the center, and so does its
    // smooth normal.
    #[test]
    fn the_surface_faces_outward() {
        let s = shape();
        let mesh = tile_mesh(&s, deep_tile());
        let side = TILE_CELLS as usize + 1;
        let surface_tris = (side - 1) * (side - 1) * 2;
        let up = dvec3::to_f32(s.up_at(mesh.origin));
        for tri in mesh.indices[..surface_tris * 3].chunks_exact(3) {
            let [a, b, c] = [tri[0], tri[1], tri[2]].map(|i| mesh.vertices[i as usize].pos);
            assert!(dot(cross(sub(b, a), sub(c, a)), up) > 0.0);
        }
        for v in &mesh.vertices {
            assert!(dot(v.normal, up) > 0.5);
            assert!(dot(v.tangent, v.normal).abs() < 1e-4);
        }
    }

    // The morph extent bounds how far a fully morphed corner moves.
    #[test]
    fn the_morph_extent_bounds_every_corner_s_move() {
        let s = shape();
        let mesh = tile_mesh(&s, deep_tile());
        let n = TILE_CELLS as usize;
        let side = n + 1;
        let grid: Vec<[f32; 3]> = mesh.vertices[..side * side].iter().map(|v| v.pos).collect();
        let mut most: f32 = 0.0;
        for j in 0..side {
            for i in 0..side {
                let p = grid[j * side + i];
                let q = super::super::morph::morph_point(&grid, i, j, 1.0);
                most = most.max(concinnity_core::math::vec3::length(sub(q, p)));
            }
        }
        assert!(most > 0.0);
        assert!(
            (most - mesh.morph_extent).abs() <= 1e-3 * (1.0 + most),
            "{most} {}",
            mesh.morph_extent
        );
    }

    // A tile is its grid alone: one vertex a corner, two triangles a cell.
    // The capture's coarse tile keeps only corners geomorph never moves
    // (both coordinates even), covers the whole tile, and turns the same way
    // as the full mesh.
    #[test]
    fn the_capture_tile_is_a_coarser_grid_of_fixed_corners() {
        let side = TILE_CELLS as usize + 1;
        let coarse = tile_indices(CAPTURE_TILE_STRIDE);
        let cells = TILE_CELLS as usize / CAPTURE_TILE_STRIDE;
        assert_eq!(coarse.len(), cells * cells * 6);
        for &k in &coarse {
            let (i, j) = (usize::from(k) % side, usize::from(k) / side);
            assert_eq!((i % CAPTURE_TILE_STRIDE, j % CAPTURE_TILE_STRIDE), (0, 0));
            assert!(
                crate::planet::morph::parent_span(i, j).is_none(),
                "corner ({i}, {j}) morphs"
            );
        }
        let used: alloc::collections::BTreeSet<u16> = coarse.iter().copied().collect();
        let last = u16::try_from(side * side - 1).unwrap();
        assert!(
            used.contains(&0) && used.contains(&last),
            "corner to corner"
        );
        // The first cell's first triangle runs the way the full mesh's does.
        let fine = tile_indices(1);
        let (stride, row) = (
            u16::try_from(CAPTURE_TILE_STRIDE).unwrap(),
            u16::try_from(side).unwrap(),
        );
        assert_eq!(&fine[..3], &[0, 1, row]);
        assert_eq!(&coarse[..3], &[0, stride, stride * row]);
    }

    #[test]
    fn the_mesh_is_the_grid() {
        let mesh = tile_mesh(&shape(), deep_tile());
        let n = TILE_CELLS as usize;
        assert_eq!(mesh.vertices.len(), (n + 1) * (n + 1));
        assert_eq!(mesh.indices.len(), n * n * 6);
    }

    // Neighbors at one level share their edge corners exactly: same
    // positions, same normals.
    #[test]
    fn neighbors_share_their_edge() {
        let s = shape();
        let a = deep_tile();
        let b = TileId { x: a.x + 1, ..a };
        let (ma, mb) = (tile_mesh(&s, a), tile_mesh(&s, b));
        let side = TILE_CELLS as usize + 1;
        for j in 0..side {
            let va = ma.vertices[j * side + side - 1];
            let vb = mb.vertices[j * side];
            let pa = dvec3::add(ma.origin, dvec3::from_f32(va.pos));
            let pb = dvec3::add(mb.origin, dvec3::from_f32(vb.pos));
            assert!(dvec3::length(dvec3::sub(pa, pb)) < 0.01, "{pa:?} {pb:?}");
            assert!(dot(va.normal, vb.normal) > 1.0 - 1e-6);
        }
    }

    // A tile beside a coarser one shades every edge corner they share the
    // same: the coarse tile's edge corners are every other corner of the
    // fine one's.
    #[test]
    fn edge_normals_match_across_levels() {
        let s = shape();
        let coarse = TileId {
            face: CubeFace(2),
            level: 8,
            x: 128,
            y: 127,
        };
        // The fine tile's +s edge runs along the lower half of the coarse
        // tile's -s edge.
        let fine = TileId {
            face: CubeFace(2),
            level: 9,
            x: 255,
            y: 254,
        };
        let (mf, mc) = (tile_mesh(&s, fine), tile_mesh(&s, coarse));
        let n = TILE_CELLS as usize;
        let side = n + 1;
        for j in (0..side).step_by(2) {
            let vf = mf.vertices[j * side + n];
            let vc = mc.vertices[(j / 2) * side];
            let pf = dvec3::add(mf.origin, dvec3::from_f32(vf.pos));
            let pc = dvec3::add(mc.origin, dvec3::from_f32(vc.pos));
            assert!(dvec3::length(dvec3::sub(pf, pc)) < 0.01, "{pf:?} {pc:?}");
            assert!(dot(vf.normal, vc.normal) > 1.0 - 1e-6, "corner {j}");
        }
    }

    #[test]
    fn texture_coordinates_continue_across_tiles() {
        let s = shape();
        let a = deep_tile();
        let b = TileId { x: a.x + 1, ..a };
        let (ma, mb) = (tile_mesh(&s, a), tile_mesh(&s, b));
        let side = TILE_CELLS as usize + 1;
        let ua = ma.vertices[side - 1].uv[0];
        let ub = mb.vertices[0].uv[0];
        let wrapped = (ua - ub).abs() % narrow_f32(UV_PERIOD_M);
        assert!(
            wrapped < 0.05 || (narrow_f32(UV_PERIOD_M) - wrapped) < 0.05,
            "{ua} {ub}"
        );
    }
}
