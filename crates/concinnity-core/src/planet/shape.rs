// A planet's surface: a sphere of `radius` around `center`, raised by a seeded
// noise field sampled on the sphere itself, so the surface is one function of
// direction with no seams. Every consumer (render tiles, colliders) reads it
// from here, so they cannot disagree about where the ground is.

use super::dvec::{self, DMat3, DVec3};
use crate::math::SaturatingFrom;
use crate::math::dvec3;
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

/// The turn from one octave's noise lattice to the next, column-major: about a
/// skew axis by an angle no fraction of a right angle, so no two octaves'
/// lattice planes line up with each other or with the planet's axes. Aligned
/// lattices leave every octave flat across the same planes, which shades as a
/// crease along each.
pub(crate) const OCTAVE_TURN: DMat3 = [
    [0.811_838_377_983, 0.534_144_405_765, -0.235_813_913_551],
    [-0.459_433_173_493, 0.833_629_154_062, 0.306_567_435_635],
    [0.360_332_634_004, -0.140_542_475_031, 0.922_175_799_717],
];

impl PlanetShape {
    /// The ground's height above `radius` in direction `dir` (unit length),
    /// in `[0, amplitude]`.
    pub fn height(&self, dir: DVec3) -> f64 {
        self.height_gradient(dir).0
    }

    /// The ground's height in direction `dir` (unit length) and its gradient
    /// over directions: how fast the height grows, in meters per unit of
    /// direction, along each authored axis.
    pub fn height_gradient(&self, dir: DVec3) -> (f64, DVec3) {
        if self.amplitude <= 0.0 {
            return (0.0, [0.0; 3]);
        }
        let mut sum = 0.0;
        let mut gradient = [0.0; 3];
        for octave in self.octave_lattices() {
            let p = dvec3::add(dvec::mul_vec(&octave.to_lattice, dir), octave.offset);
            let (value, grad) = value_noise(p, octave.salt);
            sum += value * octave.weight;
            let along = dvec::mul_transpose_vec(&octave.to_lattice, grad);
            gradient = dvec3::add(gradient, dvec3::scale(along, octave.weight));
        }
        let k = self.noise_scale();
        (sum * k, dvec3::scale(gradient, k))
    }

    /// The ground point in direction `dir` (unit length), in authored
    /// coordinates.
    pub fn surface(&self, dir: DVec3) -> DVec3 {
        dvec3::add(
            self.center,
            dvec3::scale(dir, self.radius + self.height(dir)),
        )
    }

    /// The ground's outward unit normal in direction `dir` (unit length),
    /// exact from the height's gradient: the same wherever and however finely
    /// the ground around it is sampled.
    pub fn normal(&self, dir: DVec3) -> DVec3 {
        self.surface_and_normal(dir).1
    }

    /// [`Self::surface`] and [`Self::normal`] in direction `dir` (unit
    /// length), from one evaluation of the height.
    pub fn surface_and_normal(&self, dir: DVec3) -> (DVec3, DVec3) {
        let (height, gradient) = self.height_gradient(dir);
        let r = self.radius + height;
        let along = dvec3::sub(gradient, dvec3::scale(dir, dvec3::dot(dir, gradient)));
        let tilt = dvec3::scale(along, 1.0 / r.max(1e-3));
        let point = dvec3::add(self.center, dvec3::scale(dir, r));
        (point, dvec3::normalize_or(dvec3::sub(dir, tilt), dir))
    }

    /// The steepest the ground can rise along it anywhere, in meters per
    /// meter: a bound from each octave's lattice spacing, not a measurement.
    pub fn max_slope(&self) -> f64 {
        if self.amplitude <= 0.0 {
            return 0.0;
        }
        // Each octave's weight halves as its frequency doubles, so every
        // octave bounds the same, and a value noise's gradient is at most
        // 1.5 along each lattice axis.
        let per_octave = self.noise_scale() * self.base_frequency() * 1.5 * libm::sqrt(3.0);
        per_octave * f64::from(self.octave_count()) / self.radius
    }

    /// The direction from the center to authored point `p`.
    pub fn up_at(&self, p: DVec3) -> DVec3 {
        dvec3::normalize_or(dvec3::sub(p, self.center), [0.0, 1.0, 0.0])
    }

    // Lattice cells of the broadest octave per unit of direction.
    pub(crate) fn base_frequency(&self) -> f64 {
        self.radius / self.feature_size.max(1e-3)
    }

    // The octaves the height sums, broadest first.
    pub(crate) fn octave_count(&self) -> u32 {
        self.octaves.clamp(MIN_OCTAVES, MAX_OCTAVES)
    }

    // Meters of height per unit of the octaves' weighted sum: the amplitude
    // over the weights, each octave half the last.
    pub(crate) fn noise_scale(&self) -> f64 {
        let weight_sum = 2.0 * (1.0 - libm::pow(0.5, f64::from(self.octave_count())));
        self.amplitude / weight_sum
    }

    // Octave `octave`'s hash salt.
    pub(crate) fn octave_salt(&self, octave: u32) -> u32 {
        lcg_hash(self.seed ^ octave.wrapping_mul(0x9e37_79b9))
    }

    fn octave_lattices(&self) -> impl Iterator<Item = OctaveLattice> {
        let base = self.base_frequency();
        let mut to_lattice = dvec::scale_mat(&OCTAVE_TURN, base);
        let mut weight = 1.0;
        (0..self.octave_count()).map(move |octave| {
            let salt = self.octave_salt(octave);
            let lattice = OctaveLattice {
                to_lattice,
                offset: lattice_offset(salt),
                salt,
                weight,
            };
            to_lattice = dvec::scale_mat(&dvec::mul(&OCTAVE_TURN, &to_lattice), 2.0);
            weight *= 0.5;
            lattice
        })
    }
}

// One octave of the height noise: the map from a direction into its lattice,
// the lattice's shift, its salt and its weight in the sum.
struct OctaveLattice {
    to_lattice: DMat3,
    offset: DVec3,
    salt: u32,
    weight: f64,
}

// How far an octave's lattice coordinates shift for its `salt`, in cells: three
// 10-bit fractions of the salt, so no lattice plane passes through the center.
fn lattice_offset(salt: u32) -> DVec3 {
    [0, 10, 20].map(|shift| f64::from((salt >> shift) & 0x3ff) / 1024.0)
}

// Trilinear value noise in [0, 1] at lattice position `p`, salted per octave,
// with its gradient along the lattice axes.
fn value_noise(p: DVec3, salt: u32) -> (f64, DVec3) {
    let cell = p.map(libm::floor);
    let t = [p[0] - cell[0], p[1] - cell[1], p[2] - cell[2]];
    let [u, v, w] = t.map(|t| t * t * (3.0 - 2.0 * t));
    let [du, dv, dw] = t.map(|t| 6.0 * t * (1.0 - t));
    let [x, y, z] = cell.map(|c| i32::saturating_from(c).cast_unsigned());
    let at = |dx: u32, dy: u32, dz: u32| {
        lattice_value(
            x.wrapping_add(dx),
            y.wrapping_add(dy),
            z.wrapping_add(dz),
            salt,
        )
    };
    let (a, b, c, d) = (at(0, 0, 0), at(1, 0, 0), at(0, 1, 0), at(1, 1, 0));
    let (e, f, g, h) = (at(0, 0, 1), at(1, 0, 1), at(0, 1, 1), at(1, 1, 1));
    let k1 = b - a;
    let k2 = c - a;
    let k3 = e - a;
    let k4 = a - b - c + d;
    let k5 = a - c - e + g;
    let k6 = a - b - e + f;
    let k7 = -a + b + c - d + e - f - g + h;
    let value =
        a + k1 * u + k2 * v + k3 * w + k4 * u * v + k5 * v * w + k6 * w * u + k7 * u * v * w;
    let gradient = [
        du * (k1 + k4 * v + k6 * w + k7 * v * w),
        dv * (k2 + k5 * w + k4 * u + k7 * w * u),
        dw * (k3 + k6 * u + k5 * v + k7 * u * v),
    ];
    (value, gradient)
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
    pub(crate) fn axes(self) -> [DVec3; 3] {
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
        spherify(self.cube_point(s, t))
    }

    /// The face's `u` axis: the direction `s` grows along.
    pub fn u_axis(self) -> DVec3 {
        self.axes()[1]
    }

    /// The face that point `c` of the cube's space lies in front of, and its
    /// face coordinates `(s, t)` there: `c` projected onto that face's plane.
    pub(crate) fn locate(c: DVec3) -> (Self, f64, f64) {
        let face = Self::ALL
            .into_iter()
            .max_by(|a, b| dvec3::dot(a.axes()[0], c).total_cmp(&dvec3::dot(b.axes()[0], c)))
            .unwrap_or(Self(0));
        let [n, u, v] = face.axes();
        let on_plane = dvec3::scale(c, 1.0 / dvec3::dot(n, c).max(1e-12));
        (face, dvec3::dot(on_plane, u), dvec3::dot(on_plane, v))
    }

    // The point of the cube's space at face coordinates `(s, t)`.
    pub(crate) fn cube_point(self, s: f64, t: f64) -> DVec3 {
        let [n, u, v] = self.axes();
        dvec3::add(n, dvec3::add(dvec3::scale(u, s), dvec3::scale(v, t)))
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
    dvec3::normalize_or(p, [0.0, 1.0, 0.0])
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
            let a = f64::from(i) * 0.0137;
            let dir = dvec3::normalize_or([libm::sin(a), libm::cos(a * 0.7), 0.3], [0.0; 3]);
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
            let a = 0.3 + f64::from(i) * step;
            let d0 = dvec3::normalize_or([libm::sin(a), 1.0, 0.2], [0.0; 3]);
            let d1 = dvec3::normalize_or([libm::sin(a + step), 1.0, 0.2], [0.0; 3]);
            assert!((s.height(d0) - s.height(d1)).abs() < 0.5);
        }
    }

    #[test]
    fn every_face_direction_is_unit_and_points_through_its_face() {
        for face in CubeFace::ALL {
            let [n, _, _] = face.axes();
            for (s, t) in [(0.0, 0.0), (1.0, 1.0), (-1.0, 0.5), (0.25, -1.0)] {
                let d = face.direction(s, t);
                assert!((dvec3::length(d) - 1.0).abs() < 1e-12);
                assert!(dvec3::dot(d, n) > 0.5);
            }
            assert!((dvec3::dot(face.direction(0.0, 0.0), n) - 1.0).abs() < 1e-12);
        }
    }

    // Two faces sharing an edge map that edge to the same directions, so the
    // tiles meeting there share their edge vertices.
    #[test]
    fn faces_agree_along_shared_edges() {
        // +X at s = -1 is the edge with +Z; +Z at s = 1 is the same edge.
        for i in 0..=8 {
            let t = -1.0 + f64::from(i) * 0.25;
            let a = CubeFace(0).direction(-1.0, t);
            let b = CubeFace(4).direction(1.0, t);
            assert!(dvec3::length(dvec3::sub(a, b)) < 1e-14, "{a:?} {b:?}");
        }
    }

    #[test]
    fn the_surface_sits_on_the_sphere_plus_the_height() {
        let s = shape();
        let dir = CubeFace(2).direction(0.1, -0.2);
        let p = s.surface(dir);
        let r = dvec3::length(dvec3::sub(p, s.center));
        assert!((r - 50_000.0 - s.height(dir)).abs() < 1e-6);
        let up = s.up_at(p);
        assert!(dvec3::length(dvec3::sub(up, dir)) < 1e-12);
    }

    // The analytic gradient is the height's own slope.
    #[test]
    fn the_gradient_matches_the_height_s_slope() {
        let s = shape();
        let e = 1e-7;
        for i in 0..50 {
            let a = 0.3 + f64::from(i) * 0.0011;
            let dir = dvec3::normalize_or([libm::sin(a), 1.0, libm::cos(a * 3.0) * 0.4], [0.0; 3]);
            let (_, g) = s.height_gradient(dir);
            for axis in 0..3 {
                let mut hi = dir;
                let mut lo = dir;
                hi[axis] += e;
                lo[axis] -= e;
                let slope = (s.height(hi) - s.height(lo)) / (2.0 * e);
                assert!(
                    (slope - g[axis]).abs() < 1e-3 * (1.0 + slope.abs()),
                    "{slope} vs {}",
                    g[axis]
                );
            }
        }
    }

    // No two points of the ground differ in height by more than the slope
    // bound times the distance between them.
    #[test]
    fn the_slope_bound_holds() {
        let s = shape();
        let bound = s.max_slope();
        assert!(bound > 0.0 && bound < 1.0, "{bound}");
        let face = CubeFace(2);
        let mut steepest: f64 = 0.0;
        for i in 0..2_000 {
            let (fs, ft) = (-0.4 + f64::from(i) * 3.7e-4, 0.1 + f64::from(i) * 1.3e-4);
            let a = face.direction(fs, ft);
            let b = face.direction(fs + 2e-5, ft - 1e-5);
            let run = 50_000.0 * libm::acos(dvec3::dot(a, b).min(1.0));
            steepest = steepest.max((s.height(a) - s.height(b)).abs() / run);
        }
        assert!(steepest <= bound, "{steepest} > {bound}");
        assert!(steepest > 0.0);
        let flat = PlanetShape {
            amplitude: 0.0,
            ..shape()
        };
        assert_eq!(flat.max_slope(), 0.0);
    }

    // A cube point found in front of a face lies at its own face
    // coordinates there.
    #[test]
    fn located_points_round_trip() {
        for face in CubeFace::ALL {
            for (s, t) in [(0.0, 0.0), (0.9, -0.3), (-0.99, 0.99)] {
                let (f, ls, lt) = CubeFace::locate(face.cube_point(s, t));
                assert_eq!(f, face);
                assert!((ls - s).abs() < 1e-12 && (lt - t).abs() < 1e-12);
            }
        }
        // Just past +X's s = -1 edge lies +Z, at its s = 1 edge.
        let (f, s, t) = CubeFace::locate(CubeFace(0).cube_point(-1.001, 0.5));
        assert_eq!(f, CubeFace(4));
        assert!(s < 1.0 && s > 0.99, "{s}");
        assert!((t - 0.5 / 1.001).abs() < 1e-12, "{t}");
    }

    // The normal is perpendicular to the surface around its point.
    #[test]
    fn the_normal_is_perpendicular_to_the_surface() {
        let s = shape();
        let face = CubeFace(2);
        let e = 1e-6;
        for i in 0..40 {
            let (fs, ft) = (-0.3 + f64::from(i) * 0.013, 0.2 - f64::from(i) * 0.007);
            let n = s.normal(face.direction(fs, ft));
            let ds = dvec3::sub(
                s.surface(face.direction(fs + e, ft)),
                s.surface(face.direction(fs - e, ft)),
            );
            let dt = dvec3::sub(
                s.surface(face.direction(fs, ft + e)),
                s.surface(face.direction(fs, ft - e)),
            );
            let exact = dvec3::normalize_or(dvec3::cross(ds, dt), [0.0; 3]);
            assert!(dvec3::dot(n, exact) > 1.0 - 1e-8, "{n:?} {exact:?}");
        }
    }

    #[test]
    fn surface_and_normal_match_their_separate_forms() {
        let s = shape();
        for i in 0..20 {
            let dir = CubeFace(2).direction(-0.4 + f64::from(i) * 0.03, 0.1);
            let (p, n) = s.surface_and_normal(dir);
            assert_eq!(p, s.surface(dir));
            assert_eq!(n, s.normal(dir));
        }
    }

    // No octave's lattice runs along the planet's axes: the ground is not
    // flat across the meridian through a face's center, which every octave
    // of an axis-aligned lattice would leave it.
    #[test]
    fn no_crease_runs_along_the_axes() {
        let s = shape();
        let across = |x: f64, z: f64| {
            let dir = dvec3::normalize_or([x / 50_000.0, 1.0, z / 50_000.0], [0.0; 3]);
            s.height_gradient(dir).1[0].abs()
        };
        let mean = |x: f64| {
            (0..200)
                .map(|i| across(x, -20.0 * f64::from(i)))
                .sum::<f64>()
                / 200.0
        };
        let on = mean(0.0);
        let off = (mean(37.0) + mean(-61.0)) * 0.5;
        assert!(
            on > off * 0.5,
            "slope across the meridian {on}, beside it {off}"
        );
    }
}
