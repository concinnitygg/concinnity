// A planet's surface: a sphere of `radius` around `center`, raised by a seeded
// noise field sampled on the sphere itself, so the surface is one function of
// direction with no seams. Every consumer (render tiles, colliders) reads it
// from here, so they cannot disagree about where the ground is.

use super::dvec::{self, DVec3};
use crate::math::noise::lcg_hash;

/// Fewest octaves a planet's height noise sums.
pub const MIN_OCTAVES: u32 = 1;

/// Most octaves a planet's height noise sums.
pub const MAX_OCTAVES: u32 = 16;

/// The surface every part of a planet is built from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanetShape {
    /// The center, in authored coordinates.
    pub center: DVec3,
    /// The radius of the lowest ground, in meters.
    pub radius: f64,
    /// The highest ground above `radius`, in meters.
    pub amplitude: f64,
    /// Width of the broadest hills, in meters.
    pub feature_size: f64,
    /// Noise octaves, each half the width and half the height of the last.
    pub octaves: u32,
    /// Picks the landscape.
    pub seed: u32,
}

impl PlanetShape {
    /// The ground's height above `radius` in direction `dir` (unit length),
    /// in `[0, amplitude]`.
    pub fn height(&self, dir: DVec3) -> f64 {
        if self.amplitude <= 0.0 {
            return 0.0;
        }
        let base = self.radius / self.feature_size.max(1e-3);
        let mut sum = 0.0;
        let mut weight_sum = 0.0;
        let mut frequency = base;
        let mut weight = 1.0;
        for octave in 0..self.octaves.clamp(MIN_OCTAVES, MAX_OCTAVES) {
            let salt = lcg_hash(self.seed ^ octave.wrapping_mul(0x9e37_79b9));
            sum += value_noise(dvec::scale(dir, frequency), salt) * weight;
            weight_sum += weight;
            frequency *= 2.0;
            weight *= 0.5;
        }
        sum / weight_sum * self.amplitude
    }

    /// The ground point in direction `dir` (unit length), in authored
    /// coordinates.
    pub fn surface(&self, dir: DVec3) -> DVec3 {
        dvec::add(
            self.center,
            dvec::scale(dir, self.radius + self.height(dir)),
        )
    }

    /// The direction from the center to authored point `p`.
    pub fn up_at(&self, p: DVec3) -> DVec3 {
        dvec::normalize_or(dvec::sub(p, self.center), [0.0, 1.0, 0.0])
    }
}

// Trilinear value noise in [0, 1] at lattice position `p`, salted per octave.
fn value_noise(p: DVec3, salt: u32) -> f64 {
    let cell = p.map(libm::floor);
    let f = [p[0] - cell[0], p[1] - cell[1], p[2] - cell[2]].map(|t| t * t * (3.0 - 2.0 * t));
    let [x, y, z] = cell.map(|c| c as i64 as u32);
    let v = |dx: u32, dy: u32, dz: u32| {
        lattice_value(
            x.wrapping_add(dx),
            y.wrapping_add(dy),
            z.wrapping_add(dz),
            salt,
        )
    };
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let x00 = lerp(v(0, 0, 0), v(1, 0, 0), f[0]);
    let x10 = lerp(v(0, 1, 0), v(1, 1, 0), f[0]);
    let x01 = lerp(v(0, 0, 1), v(1, 0, 1), f[0]);
    let x11 = lerp(v(0, 1, 1), v(1, 1, 1), f[0]);
    lerp(lerp(x00, x10, f[1]), lerp(x01, x11, f[1]), f[2])
}

// The value at a 3D lattice point, in [0, 1] with 16 bits of resolution.
fn lattice_value(x: u32, y: u32, z: u32, salt: u32) -> f64 {
    let h = lcg_hash(
        lcg_hash(lcg_hash(x ^ salt).wrapping_add(y.wrapping_mul(0x85eb_ca6b)))
            .wrapping_add(z.wrapping_mul(0xc2b2_ae35)),
    );
    f64::from(h & 0xFFFF) / 65535.0
}

/// One of the six faces a planet's surface is divided into, each the sphere
/// seen through one face of a cube around it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CubeFace(pub u8);

impl CubeFace {
    /// All six faces: `+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`.
    pub const ALL: [Self; 6] = [Self(0), Self(1), Self(2), Self(3), Self(4), Self(5)];

    // (outward normal, u axis, v axis), with u x v = normal.
    fn axes(self) -> [DVec3; 3] {
        match self.0 {
            0 => [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            1 => [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            2 => [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            3 => [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            4 => [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            _ => [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }
    }

    /// The direction through this face at face coordinates `(s, t)`, each in
    /// `[-1, 1]` across the face (and beyond, past its edges).
    ///
    /// The cube is spherified rather than normalized, which keeps cells from
    /// crowding toward the face corners; the mapping is the same seen from
    /// either face sharing an edge.
    pub fn direction(self, s: f64, t: f64) -> DVec3 {
        let [n, u, v] = self.axes();
        let c = dvec::add(n, dvec::add(dvec::scale(u, s), dvec::scale(v, t)));
        spherify(c)
    }

    /// The face's `u` axis: the direction `s` grows along.
    pub fn u_axis(self) -> DVec3 {
        self.axes()[1]
    }
}

// A point on (or near) the unit cube mapped onto the unit sphere.
fn spherify(c: DVec3) -> DVec3 {
    let [x, y, z] = c;
    let (x2, y2, z2) = (x * x, y * y, z * z);
    let p = [
        x * libm::sqrt((1.0 - y2 / 2.0 - z2 / 2.0 + y2 * z2 / 3.0).max(0.0)),
        y * libm::sqrt((1.0 - z2 / 2.0 - x2 / 2.0 + z2 * x2 / 3.0).max(0.0)),
        z * libm::sqrt((1.0 - x2 / 2.0 - y2 / 2.0 + x2 * y2 / 3.0).max(0.0)),
    ];
    // Exact on the cube; outside it (a tile's border ring) only close, so
    // land on the sphere either way.
    dvec::normalize_or(p, [0.0, 1.0, 0.0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> PlanetShape {
        PlanetShape {
            center: [0.0, -50_000.0, 0.0],
            radius: 50_000.0,
            amplitude: 60.0,
            feature_size: 2_000.0,
            octaves: 8,
            seed: 7,
        }
    }

    #[test]
    fn heights_are_bounded_deterministic_and_seeded() {
        let s = shape();
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for i in 0..400 {
            let a = i as f64 * 0.0137;
            let dir = dvec::normalize_or([libm::sin(a), libm::cos(a * 0.7), 0.3], [0.0; 3]);
            let h = s.height(dir);
            assert!((0.0..=60.0).contains(&h), "{h}");
            assert_eq!(h, s.height(dir));
            lo = lo.min(h);
            hi = hi.max(h);
        }
        assert!(hi - lo > 10.0, "expected relief, got {lo}..{hi}");
        let other = PlanetShape { seed: 8, ..shape() };
        let dir = [0.0, 1.0, 0.0];
        assert_ne!(s.height(dir), other.height(dir));
        let flat = PlanetShape {
            amplitude: 0.0,
            ..shape()
        };
        assert_eq!(flat.height(dir), 0.0);
    }

    // The field is continuous at walking scale: neighbors a meter apart differ
    // by centimeters, not meters.
    #[test]
    fn heights_are_smooth_at_walking_scale() {
        let s = shape();
        let step = 1.0 / 50_000.0;
        for i in 0..200 {
            let a = 0.3 + i as f64 * step;
            let d0 = dvec::normalize_or([libm::sin(a), 1.0, 0.2], [0.0; 3]);
            let d1 = dvec::normalize_or([libm::sin(a + step), 1.0, 0.2], [0.0; 3]);
            assert!((s.height(d0) - s.height(d1)).abs() < 0.5);
        }
    }

    #[test]
    fn every_face_direction_is_unit_and_points_through_its_face() {
        for face in CubeFace::ALL {
            let [n, _, _] = face.axes();
            for (s, t) in [(0.0, 0.0), (1.0, 1.0), (-1.0, 0.5), (0.25, -1.0)] {
                let d = face.direction(s, t);
                assert!((dvec::length(d) - 1.0).abs() < 1e-12);
                assert!(dvec::dot(d, n) > 0.5);
            }
            assert!((dvec::dot(face.direction(0.0, 0.0), n) - 1.0).abs() < 1e-12);
        }
    }

    // Two faces sharing an edge map that edge to the same directions, so the
    // tiles meeting there share their edge vertices.
    #[test]
    fn faces_agree_along_shared_edges() {
        // +X at s = -1 is the edge with +Z; +Z at s = 1 is the same edge.
        for i in 0..=8 {
            let t = -1.0 + i as f64 * 0.25;
            let a = CubeFace(0).direction(-1.0, t);
            let b = CubeFace(4).direction(1.0, t);
            assert!(dvec::length(dvec::sub(a, b)) < 1e-14, "{a:?} {b:?}");
        }
    }

    #[test]
    fn the_surface_sits_on_the_sphere_plus_the_height() {
        let s = shape();
        let dir = CubeFace(2).direction(0.1, -0.2);
        let p = s.surface(dir);
        let r = dvec::length(dvec::sub(p, s.center));
        assert!((r - 50_000.0 - s.height(dir)).abs() < 1e-6);
        let up = s.up_at(p);
        assert!(dvec::length(dvec::sub(up, dir)) < 1e-12);
    }
}
