// A planet's surface: a sphere of `radius` around `center`, shaped by seeded
// noise fields sampled on the sphere itself, so the surface is one function of
// direction with no seams, and by the edits stored over it. Every consumer
// (render tiles, colliders) reads it from here, so they cannot disagree about
// where the ground is.

use alloc::sync::Arc;
use concinnity_core::math::SaturatingFrom;
use concinnity_core::math::dmat3::{self, DMat3};
use concinnity_core::math::dvec3;
use concinnity_core::math::dvec3::DVec3;
use concinnity_core::math::noise::lcg_hash;

use super::coast::{Continents, HILL_RISE};
use super::edits::{EditHit, EditPages, PAGE_CELLS, TileEdits};
use super::tile::{MAX_TILE_LEVEL, TileId};

/// Fewest octaves a planet's height noise sums.
pub const MIN_OCTAVES: u32 = 1;

/// Most octaves a planet's height noise sums.
pub const MAX_OCTAVES: u32 = 16;

/// The surface every part of a planet is built from: the generated ground,
/// changed where `edits` lie over it.
#[derive(Debug, Clone)]
pub struct PlanetShape {
    /// The center, in authored coordinates.
    pub center: DVec3,
    /// The radius of the datum heights are measured from, in meters: mean sea
    /// level when the planet has seas, the lowest ground when it has none.
    pub radius: f64,
    /// The highest generated ground above `radius`, in meters.
    pub amplitude: f64,
    /// Width of the broadest hills, in meters.
    pub feature_size: f64,
    /// Noise octaves, each half the width and half the height of the last.
    pub octaves: u32,
    /// Picks the landscape.
    pub seed: u32,
    /// Where the ground lies above the datum and where below it.
    pub continents: Continents,
    /// The edits stored over the generated ground. Two shapes hold the same
    /// edits only when they share the one set.
    pub edits: Option<Arc<EditPages>>,
}

impl PartialEq for PlanetShape {
    fn eq(&self, other: &Self) -> bool {
        self.same_generated(other)
            && match (&self.edits, &other.edits) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
    }
}

/// The ground at one point: its height above the datum, the gradient of the
/// height over directions, and the part of that gradient the edits add that
/// the generated hills scaled by the detail weight do not give.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Ground {
    pub height: f64,
    pub gradient: DVec3,
    pub correction: DVec3,
    pub weight: f64,
}

// The edits at one point: the height change and the detail weight, each with
// its gradient over directions.
#[derive(Debug, Clone, Copy, PartialEq)]
struct EditAt {
    e: f64,
    e_gradient: DVec3,
    w: f64,
    w_gradient: DVec3,
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
    /// This shape with `edits` stored over its generated ground.
    #[must_use]
    pub fn with_edits(&self, edits: Option<Arc<EditPages>>) -> Self {
        Self {
            edits,
            ..self.clone()
        }
    }

    /// Whether `other` generates the same ground, whatever edits either
    /// holds.
    pub fn same_generated(&self, other: &Self) -> bool {
        self.center == other.center
            && self.radius == other.radius
            && self.amplitude == other.amplitude
            && self.feature_size == other.feature_size
            && self.octaves == other.octaves
            && self.seed == other.seed
            && self.continents == other.continents
    }

    /// The ground's height above `radius` in direction `dir` (unit length),
    /// in `[lowest_height, highest_height]`.
    pub fn height(&self, dir: DVec3) -> f64 {
        self.height_gradient(dir).0
    }

    /// The lowest the ground gets, in meters above `radius`: the ocean depth
    /// below it on a planet with seas, 0 on one without, lowered by as much
    /// as the edits lower any of it.
    pub fn lowest_height(&self) -> f64 {
        let generated = if self.continents.divides() {
            -self.continents.ocean_depth()
        } else {
            0.0
        };
        generated
            + self
                .edits
                .as_ref()
                .map_or(0.0, |e| e.lowest_change().min(0.0))
    }

    /// The radius of the sphere of the lowest ground, in meters: the one
    /// nothing on the planet lies under.
    pub fn lowest_radius(&self) -> f64 {
        self.radius + self.lowest_height()
    }

    /// The highest the ground gets, in meters above `radius`: the hilltops,
    /// or the top of the beach on a planet with seas whose hills are lower,
    /// raised by as much as the edits raise any of it.
    pub fn highest_height(&self) -> f64 {
        let generated = self.amplitude.max(self.continents.beach_rise()).max(0.0);
        generated
            + self
                .edits
                .as_ref()
                .map_or(0.0, |e| e.highest_change().max(0.0))
    }

    /// The ground's height in direction `dir` (unit length) and its gradient
    /// over directions: how fast the height grows, in meters per unit of
    /// direction, along each authored axis.
    fn height_gradient(&self, dir: DVec3) -> (f64, DVec3) {
        let ground = self.ground(dir, self.edit_at(dir).as_ref());
        (ground.height, ground.gradient)
    }

    // The ground in direction `dir` (unit length), the edits there `edit`.
    fn ground(&self, dir: DVec3, edit: Option<&EditAt>) -> Ground {
        let (hill, hill_gradient) = self.hills(dir);
        let Some(edit) = edit else {
            let (height, gradient) = self.shaped(dir, hill, hill_gradient);
            return Ground {
                height,
                gradient,
                correction: [0.0; 3],
                weight: 1.0,
            };
        };
        let weighted = edit.w * hill;
        let weighted_gradient = dvec3::add(
            dvec3::scale(hill_gradient, edit.w),
            dvec3::scale(edit.w_gradient, hill),
        );
        let (height, gradient) = self.shaped(dir, weighted, weighted_gradient);
        let per_hill = if self.continents.divides() {
            let (inland, _) = self.continents.inland(self.radius, dir);
            self.continents
                .shore(inland, weighted, self.amplitude, self.hill_rise())
                .per_hill_meter
        } else {
            1.0
        };
        Ground {
            height: height + edit.e,
            gradient: dvec3::add(gradient, edit.e_gradient),
            correction: dvec3::add(
                dvec3::scale(edit.w_gradient, hill * per_hill),
                edit.e_gradient,
            ),
            weight: edit.w,
        }
    }

    // The ground shaped by the coast from hills `hill` high, rising along
    // `hill_gradient`, in direction `dir`, and its gradient.
    fn shaped(&self, dir: DVec3, hill: f64, hill_gradient: DVec3) -> (f64, DVec3) {
        if !self.continents.divides() {
            return (hill, hill_gradient);
        }
        let (inland, inland_gradient) = self.continents.inland(self.radius, dir);
        let shore = self
            .continents
            .shore(inland, hill, self.amplitude, self.hill_rise());
        let gradient = dvec3::add(
            dvec3::scale(inland_gradient, shore.per_meter_inland),
            dvec3::scale(hill_gradient, shore.per_hill_meter),
        );
        (shore.height, gradient)
    }

    // The edits in direction `dir` (unit length), if any page lies there.
    fn edit_at(&self, dir: DVec3) -> Option<EditAt> {
        let edits = self.edits.as_ref()?;
        let (face, s, t) = CubeFace::of_direction(dir);
        edits
            .at(face, s, t)
            .map(|hit| edit_from_hit(face, s, t, &hit))
    }

    /// The ground at `(fs, ft)` across `tile`, each in `[0, 1]`, its edits
    /// read from `over`, the edits over the tile: its direction and the
    /// ground there. A point on the tile's edge reads the edits as every
    /// tile sharing it does, so tiles meet whatever pages lie either side.
    pub(crate) fn ground_on(
        &self,
        tile: TileId,
        over: Option<&TileEdits<'_>>,
        fs: f64,
        ft: f64,
    ) -> (DVec3, Ground) {
        let dir = tile.direction(fs, ft);
        let edge = fs <= 0.0 || fs >= 1.0 || ft <= 0.0 || ft >= 1.0;
        let edit = match over {
            None => None,
            Some(_) if edge => self.edit_at(dir),
            Some(o) if o.is_empty() => None,
            Some(o) => {
                let (s0, t0, span) = tile.span();
                let (s, t) = (s0 + fs * span, t0 + ft * span);
                o.at(s, t).map(|hit| edit_from_hit(tile.face, s, t, &hit))
            }
        };
        (dir, self.ground(dir, edit.as_ref()))
    }

    /// The edits over `tile`, resolved once for the points read on it.
    pub(crate) fn edits_over(&self, tile: TileId) -> Option<TileEdits<'_>> {
        self.edits.as_ref().map(|e| e.over(tile))
    }

    /// How far inland of the coast direction `dir` (unit length) lies, in
    /// meters (negative out to sea, exact at the coast and an estimate away
    /// from it), and how fast that grows per meter along the ground, along
    /// each authored axis. `None` on a planet without seas.
    pub fn inland(&self, dir: DVec3) -> Option<(f64, DVec3)> {
        if !self.continents.divides() {
            return None;
        }
        let (inland, gradient) = self.continents.inland(self.radius, dir);
        let along = dvec3::sub(gradient, dvec3::scale(dir, dvec3::dot(dir, gradient)));
        Some((inland, dvec3::scale(along, 1.0 / self.radius)))
    }

    /// The ground's height `inland` meters from the coast where no hills
    /// stand: the beach and the sea floor the coast's profile lays down.
    pub fn coast_height(&self, inland: f64) -> f64 {
        self.continents
            .shore(inland, 0.0, self.amplitude, self.hill_rise())
            .height
    }

    // The hills' height in direction `dir` and its gradient over directions.
    fn hills(&self, dir: DVec3) -> (f64, DVec3) {
        if self.hill_height() <= 0.0 {
            return (0.0, [0.0; 3]);
        }
        let (sum, gradient) = fbm(dir, self.base_frequency(), self.octave_count(), |o| {
            self.octave_salt(o)
        });
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
        let ground = self.ground(dir, self.edit_at(dir).as_ref());
        self.point_and_normal(dir, &ground)
    }

    /// The ground point and outward unit normal in direction `dir` (unit
    /// length) of the ground `ground` there.
    pub(crate) fn point_and_normal(&self, dir: DVec3, ground: &Ground) -> (DVec3, DVec3) {
        let r = self.radius + ground.height;
        let gradient = ground.gradient;
        let along = dvec3::sub(gradient, dvec3::scale(dir, dvec3::dot(dir, gradient)));
        let tilt = dvec3::scale(along, 1.0 / r.max(1e-3));
        let point = dvec3::add(self.center, dvec3::scale(dir, r));
        (point, dvec3::normalize_or(dvec3::sub(dir, tilt), dir))
    }

    /// The steepest the ground can rise along it anywhere, in meters per
    /// meter: a bound from each octave's lattice spacing and from the
    /// steepest steps between the edits' samples, not a measurement.
    pub fn max_slope(&self) -> f64 {
        // Each octave's weight halves as its frequency doubles, so every
        // octave bounds the same, and a value noise's gradient is at most
        // 1.5 along each lattice axis.
        let per_octave = self.noise_scale() * self.base_frequency() * 1.5 * libm::sqrt(3.0);
        let hills = per_octave * f64::from(self.octave_count());
        let (generated, per_hill) = if self.continents.divides() {
            let (per_inland, per_hill) = self
                .continents
                .shore_bounds(self.amplitude, self.hill_rise());
            let inland = self.continents.inland_slope_bound(self.radius);
            (
                (per_inland * inland + per_hill * hills) / self.radius,
                per_hill,
            )
        } else {
            (hills / self.radius, 1.0)
        };
        generated + self.edits_slope(per_hill)
    }

    // The steepest the edits make the ground rise beyond the generated
    // ground, in meters per meter, where the height changes `per_hill`
    // meters at most per meter of the hills. A point reads one page, so
    // the steepest page bounds it.
    fn edits_slope(&self, per_hill: f64) -> f64 {
        let Some(edits) = &self.edits else {
            return 0.0;
        };
        (0..=MAX_TILE_LEVEL)
            .map(|level| {
                let [e_step, w_step] = edits.level_steps(level);
                // A sample cell spans this much face coordinate, and a
                // bilinear cell's gradient is at most its steps along both
                // sides over the narrowest the face is mapped.
                let cell = 2.0 / f64::from(1u32 << level) / PAGE_CELLS as f64;
                let step = e_step + per_hill * self.hill_height() * w_step;
                core::f64::consts::SQRT_2 * step / (cell * min_face_rate() * self.radius)
            })
            .fold(0.0, f64::max)
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

    // The tallest the hills get, in meters: the amplitude, less the beach
    // they stand on when the planet has seas.
    pub(crate) fn hill_height(&self) -> f64 {
        self.continents.hill_height(self.amplitude)
    }

    // How far past the beach the hills take to rise to their full height.
    pub(crate) fn hill_rise(&self) -> f64 {
        self.feature_size.max(1e-3) * HILL_RISE
    }

    // Meters of height per unit of the octaves' weighted sum: the hills'
    // height over the weights, each octave half the last.
    pub(crate) fn noise_scale(&self) -> f64 {
        self.hill_height() / octave_weight_sum(self.octave_count())
    }

    // Octave `octave`'s hash salt.
    pub(crate) fn octave_salt(&self, octave: u32) -> u32 {
        lcg_hash(self.seed ^ octave.wrapping_mul(0x9e37_79b9))
    }
}

// The edits at face coordinates `(s, t)` of `face` read from `hit`, their
// slopes along the face turned into gradients over directions.
fn edit_from_hit(face: CubeFace, s: f64, t: f64, hit: &EditHit) -> EditAt {
    let tangents = face.tangents(s, t);
    let p = &hit.sample;
    EditAt {
        e: p.e,
        e_gradient: gradient_along(tangents, p.e_slope),
        w: p.w,
        w_gradient: gradient_along(tangents, p.w_slope),
    }
}

// The gradient over directions, lying along `tangents`, that rises `slope`
// per unit along each.
fn gradient_along([a, b]: [DVec3; 2], slope: [f64; 2]) -> DVec3 {
    if slope == [0.0; 2] {
        return [0.0; 3];
    }
    let (aa, ab, bb) = (dvec3::dot(a, a), dvec3::dot(a, b), dvec3::dot(b, b));
    let det = aa * bb - ab * ab;
    if det.abs() < 1e-24 {
        return [0.0; 3];
    }
    let x = (slope[0] * bb - slope[1] * ab) / det;
    let y = (slope[1] * aa - slope[0] * ab) / det;
    dvec3::add(dvec3::scale(a, x), dvec3::scale(b, y))
}

// The least a direction moves across a face per unit of face coordinate,
// any way: the smallest singular value of the face's tangents anywhere, from
// a dense sampling with a margin for what lies between the samples. Every
// face is the same, so the first stands for all.
fn min_face_rate() -> f64 {
    static RATE: spin::Once<f64> = spin::Once::new();
    *RATE.call_once(|| {
        const SAMPLES: u32 = 64;
        let mut least = f64::INFINITY;
        for j in 0..=SAMPLES {
            for i in 0..=SAMPLES {
                let s = -1.0 + 2.0 * f64::from(i) / f64::from(SAMPLES);
                let t = -1.0 + 2.0 * f64::from(j) / f64::from(SAMPLES);
                least = least.min(smallest_singular(CubeFace(0).tangents(s, t)));
            }
        }
        least * 0.98
    })
}

// The smaller singular value of the 3x2 matrix of columns `a` and `b`.
fn smallest_singular([a, b]: [DVec3; 2]) -> f64 {
    let (aa, ab, bb) = (dvec3::dot(a, a), dvec3::dot(a, b), dvec3::dot(b, b));
    let (half, apart) = ((aa + bb) * 0.5, (aa - bb) * 0.5);
    let gap = libm::sqrt(apart * apart + ab * ab);
    libm::sqrt((half - gap).max(0.0))
}

// The sum of `octaves` octave weights, each half the last, from 1.
pub(crate) fn octave_weight_sum(octaves: u32) -> f64 {
    2.0 * (1.0 - libm::pow(0.5, f64::from(octaves)))
}

// The weighted sum of `octaves` octaves of value noise in direction `dir`, the
// broadest `base` lattice cells per unit of direction, octave `o` salted by
// `salt(o)`, and its gradient over directions.
pub(crate) fn fbm(dir: DVec3, base: f64, octaves: u32, salt: impl Fn(u32) -> u32) -> (f64, DVec3) {
    let mut sum = 0.0;
    let mut gradient = [0.0; 3];
    for octave in octave_lattices(base, octaves, salt) {
        let p = dvec3::add(dmat3::mul_vec(&octave.to_lattice, dir), octave.offset);
        let (value, grad) = value_noise(p, octave.salt);
        sum += value * octave.weight;
        let along = dmat3::mul_transpose_vec(&octave.to_lattice, grad);
        gradient = dvec3::add(gradient, dvec3::scale(along, octave.weight));
    }
    (sum, gradient)
}

/// Where unit direction `dir` lies in each of `octaves` octaves of the noise
/// [`fbm`] sums with the same `base` and `salt`: the lattice cell and the
/// place within it, so a sampler near `dir` can work from small offsets.
pub(crate) fn octave_cells(
    dir: DVec3,
    base: f64,
    octaves: u32,
    salt: impl Fn(u32) -> u32,
) -> impl Iterator<Item = ([i32; 3], DVec3)> {
    octave_lattices(base, octaves, salt).map(move |octave| {
        let p = dvec3::add(dmat3::mul_vec(&octave.to_lattice, dir), octave.offset);
        let cell = p.map(libm::floor);
        (
            cell.map(i32::saturating_from),
            [p[0] - cell[0], p[1] - cell[1], p[2] - cell[2]],
        )
    })
}

fn octave_lattices(
    base: f64,
    count: u32,
    salt: impl Fn(u32) -> u32,
) -> impl Iterator<Item = OctaveLattice> {
    let mut to_lattice = dmat3::scale_mat(&OCTAVE_TURN, base);
    let mut weight = 1.0;
    (0..count).map(move |octave| {
        let salt = salt(octave);
        let lattice = OctaveLattice {
            to_lattice,
            offset: lattice_offset(salt),
            salt,
            weight,
        };
        to_lattice = dmat3::scale_mat(&dmat3::mul(&OCTAVE_TURN, &to_lattice), 2.0);
        weight *= 0.5;
        lattice
    })
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

    /// The face unit direction `dir` passes through, and its face coordinates
    /// `(s, t)` there: [`Self::direction`] inverted. The face is the axis
    /// `dir` leans along most.
    pub(crate) fn of_direction(dir: DVec3) -> (Self, f64, f64) {
        let a = dir.map(f64::abs);
        let face = if a[0] >= a[1] && a[0] >= a[2] {
            u8::from(dir[0] < 0.0)
        } else if a[1] >= a[2] {
            if dir[1] >= 0.0 { 2 } else { 3 }
        } else if dir[2] >= 0.0 {
            4
        } else {
            5
        };
        let face = Self(face);
        let [_, u, v] = face.axes();
        let (x, y) = (dvec3::dot(dir, u), dvec3::dot(dir, v));
        // The spherify solved for the cube point, written so that no term
        // cancels near the face's middle.
        let (x2, y2) = (2.0 * x * x, 2.0 * y * y);
        let d = (3.0 + x2 - y2) * (3.0 + x2 - y2) - 12.0 * x2;
        let root = libm::sqrt(d.max(0.0));
        let s = x * libm::sqrt(12.0 / (3.0 + x2 - y2 + root));
        let t = y * libm::sqrt(12.0 / (3.0 - x2 + y2 + root));
        (face, s.clamp(-1.0, 1.0), t.clamp(-1.0, 1.0))
    }

    /// How the direction through face coordinates `(s, t)` moves per unit
    /// of `s` and of `t`: the two tangents of [`Self::direction`].
    pub(crate) fn tangents(self, s: f64, t: f64) -> [DVec3; 2] {
        let [_, u, v] = self.axes();
        let c = self.cube_point(s, t);
        let (p, jacobian) = spherify_jacobian(c);
        let len = dvec3::length(p).max(1e-12);
        let dir = dvec3::scale(p, 1.0 / len);
        [u, v].map(|axis| {
            let dp = dmat3::mul_vec(&jacobian, axis);
            dvec3::scale(
                dvec3::sub(dp, dvec3::scale(dir, dvec3::dot(dir, dp))),
                1.0 / len,
            )
        })
    }
}

// The unnormalized spherify of cube point `c` and its Jacobian, column-major.
fn spherify_jacobian(c: DVec3) -> (DVec3, DMat3) {
    let [x, y, z] = c;
    let (x2, y2, z2) = (x * x, y * y, z * z);
    let roots = [
        libm::sqrt((1.0 - y2 / 2.0 - z2 / 2.0 + y2 * z2 / 3.0).max(1e-24)),
        libm::sqrt((1.0 - z2 / 2.0 - x2 / 2.0 + z2 * x2 / 3.0).max(1e-24)),
        libm::sqrt((1.0 - x2 / 2.0 - y2 / 2.0 + x2 * y2 / 3.0).max(1e-24)),
    ];
    let p = [x * roots[0], y * roots[1], z * roots[2]];
    // d(root_i)/d(c_k) for k != i: (-c_k + 2 c_k c_m^2 / 3) / (2 root_i).
    let d = |i: usize, k: usize, m: usize| {
        (c[i] * (-c[k] + 2.0 * c[k] * c[m] * c[m] / 3.0)) / (2.0 * roots[i])
    };
    let rows = [
        [roots[0], d(0, 1, 2), d(0, 2, 1)],
        [d(1, 0, 2), roots[1], d(1, 2, 0)],
        [d(2, 0, 1), d(2, 1, 0), roots[2]],
    ];
    let columns = [0, 1, 2].map(|k| [rows[0][k], rows[1][k], rows[2][k]]);
    (p, columns)
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
mod edit_tests;

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
            continents: Continents::NONE,
            edits: None,
        }
    }

    // A planet of seas and continents, 40% land.
    fn coast_shape() -> PlanetShape {
        PlanetShape {
            continents: Continents::new(50_000.0, 7, 20_000.0, 0.4, 60.0, 60.0),
            ..shape()
        }
    }

    // Directions spread over the whole sphere.
    fn everywhere(n: usize) -> impl Iterator<Item = DVec3> {
        (0..n).map(move |i| super::super::coast::fibonacci_direction(i, n))
    }

    // Each octave's cell and place within it are where `fbm` samples: the
    // noise read from them sums to the same value.
    #[test]
    fn the_octave_cells_hold_the_lattice_points() {
        let salt = |o: u32| 17 + 3 * o;
        for dir in everywhere(40) {
            let (want, _) = fbm(dir, 3_000.0, 5, salt);
            let mut sum = 0.0;
            let mut weight = 1.0;
            for (o, (cell, within)) in octave_cells(dir, 3_000.0, 5, salt).enumerate() {
                assert!(within.iter().all(|t| (0.0..1.0).contains(t)));
                let p = core::array::from_fn(|i| f64::from(cell[i]) + within[i]);
                sum += weight * value_noise(p, salt(u32::try_from(o).unwrap())).0;
                weight *= 0.5;
            }
            assert!((sum - want).abs() < 1e-12, "{sum} {want}");
        }
    }

    // The distance inland grows along the ground as its gradient says, the
    // gradient lies along the ground, and the coast's profile is the height
    // wherever the hills are flat. A planet without seas has no coast.
    #[test]
    fn the_distance_inland_and_its_gradient_agree() {
        let s = coast_shape();
        assert!(shape().inland([0.0, 1.0, 0.0]).is_none());
        for dir in everywhere(300) {
            let (_, gradient) = s.inland(dir).unwrap();
            assert!(dvec3::dot(gradient, dir).abs() < 1e-12);
            let step = 0.5;
            for axis in [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]] {
                let along = dvec3::normalize_or(
                    dvec3::sub(axis, dvec3::scale(dir, dvec3::dot(axis, dir))),
                    axis,
                );
                let at = |m: f64| {
                    let d = dvec3::normalize_or(
                        dvec3::add(dir, dvec3::scale(along, m / s.radius)),
                        dir,
                    );
                    s.inland(d).unwrap().0
                };
                let fd = (at(step) - at(-step)) / (2.0 * step);
                let want = dvec3::dot(gradient, along);
                assert!((fd - want).abs() < 1e-3 + 1e-3 * want.abs(), "{fd} {want}");
            }
        }
        let flat = PlanetShape {
            amplitude: 0.0,
            ..coast_shape()
        };
        for dir in everywhere(200) {
            let (inland, _) = flat.inland(dir).unwrap();
            assert!((flat.height(dir) - flat.coast_height(inland)).abs() < 1e-9);
        }
    }

    #[test]
    fn coast_heights_are_bounded_deterministic_and_seeded() {
        let s = coast_shape();
        assert_eq!((s.lowest_height(), s.highest_height()), (-60.0, 60.0));
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for dir in everywhere(3_000) {
            let h = s.height(dir);
            assert!((-60.0..=60.0).contains(&h), "{h}");
            assert_eq!(h, s.height(dir));
            lo = lo.min(h);
            hi = hi.max(h);
        }
        assert!(lo < -50.0 && hi > 30.0, "{lo}..{hi}");
        let other = PlanetShape {
            seed: 8,
            continents: Continents::new(50_000.0, 8, 20_000.0, 0.4, 60.0, 60.0),
            ..shape()
        };
        assert!(everywhere(64).any(|d| (s.height(d) > 0.0) != (other.height(d) > 0.0)));
        let flat = PlanetShape {
            amplitude: 0.0,
            ..coast_shape()
        };
        assert!((flat.highest_height() - 2.4).abs() < 1e-12);
        for dir in everywhere(500) {
            assert!((-60.0..=2.4 + 1e-9).contains(&flat.height(dir)));
        }
    }

    // The sea covers the share of the planet the continents leave it, for
    // every share asked for.
    #[test]
    fn the_sea_covers_the_share_asked_for() {
        for share in [0.2, 0.4, 0.7] {
            let s = PlanetShape {
                continents: Continents::new(50_000.0, 3, 15_000.0, share, 60.0, 60.0),
                ..shape()
            };
            let n = 20_000;
            let land = everywhere(n).filter(|&d| s.height(d) > 0.0).count();
            let got = land as f64 / n as f64;
            assert!((got - share).abs() < 0.03, "{share}: {got}");
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
        let e = 1e-7;
        let spots = (0..50).map(|i| {
            let a = 0.3 + f64::from(i) * 0.0011;
            dvec3::normalize_or([libm::sin(a), 1.0, libm::cos(a * 3.0) * 0.4], [0.0; 3])
        });
        let coast = coast_shape();
        let near_coast: alloc::vec::Vec<DVec3> = everywhere(4_000)
            .filter(|&d| coast.height(d).abs() < 3.0)
            .collect();
        assert!(near_coast.len() > 20, "{}", near_coast.len());
        let cases = spots
            .clone()
            .map(|d| (shape(), d))
            .chain(spots.map(|d| (coast.clone(), d)))
            .chain(near_coast.into_iter().map(|d| (coast.clone(), d)));
        for (s, dir) in cases {
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

    // The bound holds across coasts too: two meters apart, anywhere on a
    // planet of seas.
    #[test]
    fn the_slope_bound_holds_across_coasts() {
        let s = coast_shape();
        let bound = s.max_slope();
        let mut steepest: f64 = 0.0;
        for a in everywhere(20_000) {
            let side = dvec3::normalize_or(dvec3::cross(a, [0.3, 0.5, 0.8]), [1.0, 0.0, 0.0]);
            let b = dvec3::normalize_or(dvec3::add(a, dvec3::scale(side, 2.0 / 50_000.0)), a);
            let run = 50_000.0 * libm::acos(dvec3::dot(a, b).min(1.0));
            steepest = steepest.max((s.height(a) - s.height(b)).abs() / run);
        }
        assert!(steepest > 0.0 && steepest <= bound, "{steepest} > {bound}");
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
