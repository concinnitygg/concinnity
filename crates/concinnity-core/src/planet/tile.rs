// One square of a planet's surface: a quadtree node on one cube face, and the
// mesh a worker builds for it from the planet's shape.

use alloc::vec::Vec;

use super::dvec::{self, DVec3};
use super::shape::{CubeFace, PlanetShape};
use crate::gfx::mesh_payload::Vertex;

/// Cells along each side of a tile's mesh, at every level.
pub const TILE_CELLS: u32 = 32;

/// Deepest quadtree level a tile can sit at.
pub const MAX_TILE_LEVEL: u8 = 20;

// Texture coordinates are meters across the face, wrapped to this period so
// they stay precise far across a face; a texture repeating at a whole fraction
// of it tiles seamlessly over every tile.
const UV_PERIOD_M: f64 = 1024.0;

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

    // The face coordinates the tile spans: (s0, t0, side).
    fn span(self) -> (f64, f64, f64) {
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

/// The bounds of `tile` on `shape`, whatever the ground's height inside it.
pub fn tile_bounds(shape: &PlanetShape, tile: TileId) -> TileBounds {
    let mid = shape.radius + shape.amplitude * 0.5;
    let center = dvec::add(shape.center, dvec::scale(tile.direction(0.5, 0.5), mid));
    let mut radius: f64 = 0.0;
    for fs in [0.0, 0.5, 1.0] {
        for ft in [0.0, 0.5, 1.0] {
            let d = tile.direction(fs, ft);
            for r in [shape.radius, shape.radius + shape.amplitude] {
                let p = dvec::add(shape.center, dvec::scale(d, r));
                radius = radius.max(dvec::length(dvec::sub(p, center)));
            }
        }
    }
    let a = tile.direction(0.0, 0.5);
    let b = tile.direction(1.0, 0.5);
    let width = shape.radius * libm::acos(dvec::dot(a, b).clamp(-1.0, 1.0));
    TileBounds {
        center,
        radius,
        width,
    }
}

/// A tile's render mesh, positioned relative to its `origin`.
#[derive(Debug, Clone)]
pub struct TileMesh {
    /// The point the vertices are relative to, in authored coordinates.
    pub origin: DVec3,
    /// The surface's `(TILE_CELLS + 1)^2` corners, row-major along `t`, then
    /// a skirt hanging below every edge corner.
    pub vertices: Vec<Vertex>,
    /// Two triangles per cell wound to face outward, then the skirt's.
    pub indices: Vec<u16>,
}

/// The mesh of `tile` on `shape`.
///
/// Normals come from the surface's own slopes, sampled one cell past the
/// tile's edges, so a tile shades seamlessly into its neighbors at the same
/// level. A tile next to a coarser one leaves gaps between their edges; the
/// skirt, a strip hanging toward the center from every edge, fills them.
pub fn tile_mesh(shape: &PlanetShape, tile: TileId) -> TileMesh {
    let n = TILE_CELLS as usize;
    let side = n + 1;
    // Corners from -1 to n + 1: the tile's own plus a ring around it.
    let ring = side + 2;
    let mut points = Vec::with_capacity(ring * ring);
    for j in 0..ring {
        for i in 0..ring {
            let fs = (i as f64 - 1.0) / n as f64;
            let ft = (j as f64 - 1.0) / n as f64;
            points.push(shape.surface(tile.direction(fs, ft)));
        }
    }
    let at = |i: usize, j: usize| points[(j + 1) * ring + (i + 1)];
    let origin = shape.surface(tile.direction(0.5, 0.5));
    let bounds = tile_bounds(shape, tile);
    let (s0, t0, span) = tile.span();
    // Meters across the face from its -s/-t corner, less a whole number of
    // periods so the tile's own stay small.
    let face_m = |c: f64| (c + 1.0) * 0.5 * shape.radius * core::f64::consts::FRAC_PI_2;
    let base = [s0, t0].map(|c| libm::floor(face_m(c) / UV_PERIOD_M) * UV_PERIOD_M);
    let uv = |i: usize, j: usize| {
        [
            (face_m(s0 + i as f64 / n as f64 * span) - base[0]) as f32,
            (face_m(t0 + j as f64 / n as f64 * span) - base[1]) as f32,
        ]
    };
    let u_axis = tile.face.u_axis();

    let mut vertices = Vec::with_capacity(side * side + 4 * n);
    for j in 0..side {
        for i in 0..side {
            let p = at(i, j);
            // Central differences across the ring: s then t, so s x t is out.
            let ds = dvec::sub(
                at_ring(&points, ring, i + 2, j + 1),
                at_ring(&points, ring, i, j + 1),
            );
            let dt = dvec::sub(
                at_ring(&points, ring, i + 1, j + 2),
                at_ring(&points, ring, i + 1, j),
            );
            let normal = dvec::normalize_or(dvec::cross(ds, dt), shape.up_at(p));
            let tangent = dvec::normalize_or(
                dvec::sub(u_axis, dvec::scale(normal, dvec::dot(normal, u_axis))),
                ds,
            );
            vertices.push(Vertex {
                pos: dvec::to_f32(dvec::sub(p, origin)),
                normal: dvec::to_f32(normal),
                tangent: dvec::to_f32(tangent),
                color: [1.0; 3],
                uv: uv(i, j),
            });
        }
    }

    let mut indices = Vec::with_capacity(n * n * 6 + 4 * n * 6);
    let corner = |i: usize, j: usize| (j * side + i) as u16;
    for j in 0..n {
        for i in 0..n {
            let (a, b, c, d) = (
                corner(i, j),
                corner(i + 1, j),
                corner(i, j + 1),
                corner(i + 1, j + 1),
            );
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }

    // The edge corners once around, counter-clockwise seen from outside.
    let mut edge: Vec<(usize, usize)> = Vec::with_capacity(4 * n + 1);
    edge.extend((0..n).map(|i| (i, 0)));
    edge.extend((0..n).map(|j| (n, j)));
    edge.extend((0..n).map(|i| (n - i, n)));
    edge.extend((0..n).map(|j| (0, n - j)));
    let depth = skirt_depth(bounds.width);
    let first_skirt = vertices.len() as u16;
    for &(i, j) in &edge {
        let top = vertices[j * side + i];
        let down = shape.up_at(at(i, j));
        let p = dvec::sub(at(i, j), dvec::scale(down, depth));
        vertices.push(Vertex {
            pos: dvec::to_f32(dvec::sub(p, origin)),
            ..top
        });
    }
    let loop_len = edge.len();
    for k in 0..loop_len {
        let next = (k + 1) % loop_len;
        let (e0, e1) = (
            corner(edge[k].0, edge[k].1),
            corner(edge[next].0, edge[next].1),
        );
        let (s0, s1) = (first_skirt + k as u16, first_skirt + next as u16);
        indices.extend_from_slice(&[e0, s0, e1, e1, s0, s1]);
    }

    TileMesh {
        origin,
        vertices,
        indices,
    }
}

fn at_ring(points: &[DVec3], ring: usize, i: usize, j: usize) -> DVec3 {
    points[j * ring + i]
}

// How far a tile's skirt hangs: past the largest step between its edge and a
// coarser neighbor's, which grows with the tile's width.
fn skirt_depth(width: f64) -> f64 {
    0.5 + width * 0.02
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::vec3::{cross, dot, sub};

    fn shape() -> PlanetShape {
        PlanetShape {
            center: [0.0, -50_000.0, 0.0],
            radius: 50_000.0,
            amplitude: 40.0,
            feature_size: 2_000.0,
            octaves: 6,
            seed: 3,
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
        assert!(dvec::length(dvec::sub(a, b)) < 1e-15);
        let c = t.direction(1.0, 1.0);
        let d = kids[3].direction(1.0, 1.0);
        assert!(dvec::length(dvec::sub(c, d)) < 1e-15);
    }

    #[test]
    fn the_bounds_hold_every_vertex() {
        let s = shape();
        let t = deep_tile();
        let b = tile_bounds(&s, t);
        let mesh = tile_mesh(&s, t);
        let side = TILE_CELLS as usize + 1;
        for v in &mesh.vertices[..side * side] {
            let p = dvec::add(mesh.origin, dvec::from_f32(v.pos));
            assert!(dvec::length(dvec::sub(p, b.center)) <= b.radius + 1e-3);
        }
        // A face is 50 km * (pi / 2) across the middle, its 512 tiles a level
        // narrower near its center, where the spherified cube crowds them.
        assert!(b.width > 100.0 && b.width < 153.4, "{}", b.width);
    }

    // Every surface triangle faces away from the center, and so does its
    // smooth normal.
    #[test]
    fn the_surface_faces_outward() {
        let s = shape();
        let mesh = tile_mesh(&s, deep_tile());
        let side = TILE_CELLS as usize + 1;
        let surface_tris = (side - 1) * (side - 1) * 2;
        let up = dvec::to_f32(s.up_at(mesh.origin));
        for tri in mesh.indices[..surface_tris * 3].chunks_exact(3) {
            let [a, b, c] = [tri[0], tri[1], tri[2]].map(|i| mesh.vertices[i as usize].pos);
            assert!(dot(cross(sub(b, a), sub(c, a)), up) > 0.0);
        }
        for v in &mesh.vertices {
            assert!(dot(v.normal, up) > 0.5);
            assert!(dot(v.tangent, v.normal).abs() < 1e-4);
        }
    }

    // The skirt hangs below each edge corner, facing away from the tile.
    #[test]
    fn the_skirt_hangs_below_the_edges_facing_out() {
        let s = shape();
        let mesh = tile_mesh(&s, deep_tile());
        let n = TILE_CELLS as usize;
        let side = n + 1;
        assert_eq!(mesh.vertices.len(), side * side + 4 * n);
        assert_eq!(mesh.indices.len(), n * n * 6 + 4 * n * 6);
        let up = dvec::to_f32(s.up_at(mesh.origin));
        let center: [f32; 3] = [0.0; 3];
        for tri in mesh.indices[n * n * 6..].chunks_exact(3) {
            let [a, b, c] = [tri[0], tri[1], tri[2]].map(|i| mesh.vertices[i as usize].pos);
            let normal = cross(sub(b, a), sub(c, a));
            let mid = [
                (a[0] + b[0] + c[0]) / 3.0,
                (a[1] + b[1] + c[1]) / 3.0,
                (a[2] + b[2] + c[2]) / 3.0,
            ];
            let outward = sub(mid, center);
            let flat = sub(outward, crate::math::vec3::scale(up, dot(outward, up)));
            assert!(dot(normal, flat) > 0.0, "skirt faces out");
        }
        for v in &mesh.vertices[side * side..] {
            assert!(dot(v.pos, up) < 0.0, "below the tile center");
        }
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
            let pa = dvec::add(ma.origin, dvec::from_f32(va.pos));
            let pb = dvec::add(mb.origin, dvec::from_f32(vb.pos));
            assert!(dvec::length(dvec::sub(pa, pb)) < 0.01, "{pa:?} {pb:?}");
            assert!(dot(va.normal, vb.normal) > 0.9999);
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
        let wrapped = (ua - ub).abs() % UV_PERIOD_M as f32;
        assert!(
            wrapped < 0.05 || (UV_PERIOD_M as f32 - wrapped) < 0.05,
            "{ua} {ub}"
        );
    }
}
