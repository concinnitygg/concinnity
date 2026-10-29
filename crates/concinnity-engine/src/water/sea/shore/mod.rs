//! Waves at a planet sea's shores. Over the shelf the open sea's longest
//! waves give way to a few long-crested trains running in toward the land
//! across the depth contours: each slows and shortens as the water shallows
//! (its phase is the integral of the wavenumber the depth allows), grows as
//! its energy bunches up, and breaks where the wave stands taller than about
//! four fifths of the water under it, its crest pitching forward into a bore
//! that trails foam. Together the trains beat into sets and lulls, so the
//! taller waves break farther out.
//!
//! Everything here is a function of a point's place on the planet and the
//! time: its distance inland of the coast (`PlanetShape::inland`), the
//! coast's profile under it, and a slowly varying phase that turns the crests
//! a few degrees off the contours so breakers peel along a beach. Nothing is
//! stored per frame, so the waves stay where they are when the simulated
//! frame moves, and the CPU reads the same sea the shader draws.

mod dispersion;
mod swash;
mod table;
#[cfg(test)]
pub(crate) mod tests;

pub use dispersion::{group_speed, shoaling, shoaling_at, wavenumber};
pub use swash::{
    DRY_BACK, FILM_DRAIN, RELIEF_SPREAD, RUNUP_LIGHTER, SATURATED, SEABED_WET, SWASH_EDGE,
    SWASH_LOBES, SWASH_MAX_CYCLES, SWASH_MEMORY, SWASH_RELIEF, SWASH_SOFT, Swash, sand_relief,
    stockdon_runup,
};
pub use table::{PhaseTable, node};

use crate::planet::{
    CONTINENT_OCTAVES, Continents, DVec3, PlanetShape, fbm, octave_cells, octave_weight_sum,
};
use crate::water::ocean::MAX_OCEAN_CASCADES;
use crate::water::ocean::cascades::Cascades;
use crate::water::spectrum::{GRAVITY, SeaShape, unit_hash};
use concinnity_core::math::noise::lcg_hash;
use concinnity_core::math::{SaturatingFrom, dvec3};
use core::f64::consts::TAU;

/// Long-crested trains the shore waves are summed from.
pub const SHORE_TRAINS: usize = 3;

/// Nodes of each train's phase table.
pub const SHORE_NODES: usize = 32;

/// The height a wave breaks at, as a share of the water's depth under it.
pub const BREAKER_INDEX: f64 = 0.78;

/// The shallowest water a train's wavenumber is read at, in meters: at the
/// waterline it would grow without bound.
pub const SHALLOWEST: f64 = 0.1;

/// Where the distance inland changes slower than this many meters per meter
/// (a saddle of the coast, the middle of a channel), the direction to the
/// land is unclear and the trains fade out across the band.
pub const GRADIENT_FADE: [f64; 2] = [0.15, 0.3];

/// Width of the phase noise's broadest cells, in meters, and how many
/// radians either way it shifts the crests.
pub const PHASE_NOISE_CELL: f64 = 300.0;
/// See [`PHASE_NOISE_CELL`].
pub const PHASE_NOISE_RADIANS: f64 = 3.0;
/// Octaves the phase noise sums.
pub const PHASE_NOISE_OCTAVES: u32 = 2;

/// Width of the cells the swell's height varies over along a shore, in
/// meters, and how far either way it varies, as a share of the height.
pub const SWELL_CELL: f64 = 400.0;
/// See [`SWELL_CELL`].
pub const SWELL_VARIATION: f64 = 0.4;

/// The period, in meters, the surf's foam pattern repeats over along each of
/// the planet's axes: a whole number of its cells, so the host can wrap where
/// it lays the pattern out from without the pattern moving.
pub const SURF_LACE_PERIOD: f64 = 1024.0;

// The phase noise's octaves, then the swell's one.
const NOISE_SALTS: usize = PHASE_NOISE_OCTAVES as usize + 1;

/// The noise octaves the shore reads: the continents', the phase noise's,
/// then the swell's.
pub const SHORE_OCTAVES: usize = CONTINENT_OCTAVES as usize + NOISE_SALTS;

/// Where a point lies in one noise octave's lattice: the cell and the place
/// within it, each axis in `[0, 1)`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LatticePlace {
    /// The lattice cell.
    pub cell: [i32; 3],
    /// The place within the cell.
    pub within: DVec3,
}

/// How far a breaking crest's profile leans forward at most: the phase warp
/// `p + lean (1 - cos p)` compresses the front face by `1 + lean`.
pub const MAX_LEAN: f64 = 0.75;

/// The most a crest pinches, as a share of the fold.
pub const MAX_PINCH: f64 = 0.9;

/// How long the foam a breaking crest leaves takes to fade by e, in periods.
pub const FOAM_TRAIL: f64 = 0.45;

/// How far ahead of a breaking crest its foam starts, in radians of phase.
pub const FOAM_LEAD: f64 = 0.5;

/// Foam left between the crests of a surf zone most of whose waves break.
pub const RESIDUAL_FOAM: f64 = 0.2;

/// The share of a bore's speed its fresh foam rides shoreward at.
pub const FOAM_RIDE: f64 = 0.8;

// The trains' frequencies, as multiples of the sea's peak: the beat between
// them is twelve or so periods long.
const TRAIN_SPREAD: [f64; SHORE_TRAINS] = [1.0, 0.92, 1.08];

// The band of the spectrum, in multiples of its peak frequency, whose energy
// the trains carry.
const TRAIN_BAND: [f64; 2] = [0.6, 1.6];

// The quiet inland distance of a sea too shallow ever to be quiet.
const QUIET_NOWHERE: f64 = -1.0e30;

// Separates the shore's salts from the ground's.
const SHORE_SALT: u32 = 0x27d4_eb2f;

/// One long-crested train of the shore waves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShoreTrain {
    /// Angular frequency, in radians per second.
    pub omega: f64,
    /// Height above rest in deep water, in meters.
    pub amplitude: f64,
    /// Phase at the waterline at time zero, in radians.
    pub phase: f64,
}

/// The coast's profile where no hills stand, seen from the sea: how deep the
/// water is at a distance inland. Mirrors `PlanetShape::coast_height`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoastProfile {
    /// The sea's height above the planet's datum, in meters.
    pub sea_level: f64,
    /// Where the sea meets the beach, in meters inland of the datum's coast.
    pub waterline: f64,
    /// Deepest the sea floor lies below the datum, in meters.
    pub ocean_depth: f64,
    /// How far out the floor takes to fall most of the way to it, in meters.
    pub shelf: f64,
    /// Width of the beach, in meters.
    pub beach_width: f64,
    /// How high the beach rises, in meters.
    pub beach_rise: f64,
}

impl CoastProfile {
    /// The water's depth `inland` meters inland of the datum's coast; 0 on
    /// dry ground.
    pub fn depth(&self, inland: f64) -> f64 {
        let ground = if inland >= 0.0 {
            let t = (inland / self.beach_width).min(1.0);
            self.beach_rise * t * (2.0 - t)
        } else {
            -self.ocean_depth * (1.0 - libm::exp(inland / self.shelf))
        };
        (self.sea_level - ground).max(0.0)
    }
}

/// Where a point of the sea lies against the shore.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShorePoint {
    /// The water's depth, in meters.
    pub depth: f64,
    /// How strongly the trains run here, in `[0, 1]`.
    pub weight: f64,
    /// Each train's phase from the waterline, in radians.
    pub phases: [f64; SHORE_TRAINS],
    /// Each train's wavenumber here, in radians per meter.
    pub wavenumbers: [f64; SHORE_TRAINS],
    /// How much each train has grown since deep water.
    pub shoaling: [f64; SHORE_TRAINS],
    /// The unit direction along the sea toward the land; zero where it is
    /// unclear.
    pub landward: DVec3,
}

/// The shore's slow, planet-fixed variation at one point: the phase that
/// turns the crests a few degrees off the depth contours, and how tall the
/// swell runs there, so stretches of a beach break earlier or later.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShoreNoise {
    /// The phase shift, in radians.
    pub phase: f64,
    /// Its gradient, in radians per meter along the sea.
    pub gradient: DVec3,
    /// The trains' height there, as a share of their mean.
    pub swell: f64,
}

/// The shore waves at one point and instant.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ShoreSurface {
    /// The rise above the sea's rest, in meters.
    pub height: f64,
    /// How far the water at rest here moves along the sea, in meters.
    pub shift: DVec3,
    /// The surface's rise per meter along the sea, as a vector along it.
    pub slope: DVec3,
    /// How much of the surface breaking crests have whitened, in `[0, 1]`.
    pub foam: f64,
    /// The crests' velocity along the sea, in meters per second.
    pub celerity: DVec3,
}

// The shore's trains summed at one point: the group's height above rest
// and its phase, and the energy the trains carry.
struct Group {
    envelope: f64,
    carrier: f64,
    energy: f64,
}

// How a group breaks in the water under it: the height its crest is held to,
// how far it leans forward, and how close to breaking it stands.
struct Breaker {
    cap: f64,
    ratio: f64,
    held: f64,
    lean: f64,
}

impl Breaker {
    fn of(point: &ShorePoint, g: &Group) -> Self {
        let cap = BREAKER_INDEX * point.depth;
        let ratio = 2.0 * g.envelope / cap.max(1e-6);
        Self {
            cap,
            ratio,
            held: g.envelope.min(0.5 * cap),
            lean: MAX_LEAN * smoothstep(0.5, 1.0, ratio),
        }
    }
}

/// A planet sea's shore waves.
#[derive(Debug, Clone, PartialEq)]
pub struct ShoreWaves {
    trains: [ShoreTrain; SHORE_TRAINS],
    table: PhaseTable,
    coast: CoastProfile,
    peak_wavelength: f64,
    cascade_wavelengths: [f64; MAX_OCEAN_CASCADES],
    radius: f64,
    noise_salts: [u32; NOISE_SALTS],
    continents: Continents,
    planet_radius: f64,
    quiet_inland: f64,
}

impl ShoreWaves {
    /// The shore waves of a sea `sea_level` meters above `planet`'s datum
    /// whose wind sea has `shape`, its simulated cascades `cascades`. `None`
    /// on a planet without coasts, or where the sea stands over its beaches.
    pub fn new(
        planet: &PlanetShape,
        sea_level: f64,
        shape: &SeaShape,
        cascades: Option<&Cascades>,
    ) -> Option<Self> {
        let c = &planet.continents;
        if !c.divides() || sea_level >= c.beach_rise() || sea_level < 0.0 {
            return None;
        }
        // Where the beach's rise `t (2 - t)` reaches the sea.
        let waterline = c.beach_width() * (1.0 - libm::sqrt(1.0 - sea_level / c.beach_rise()));
        let coast = CoastProfile {
            sea_level,
            waterline,
            ocean_depth: c.ocean_depth(),
            shelf: c.shelf_length(),
            beach_width: c.beach_width(),
            beach_rise: c.beach_rise(),
        };
        let peak_omega = f64::from(shape.peak_omega);
        let peak_wavelength = f64::from(shape.peak_wavelength());
        let omegas = TRAIN_SPREAD.map(|r| r * peak_omega);
        let variance = f64::from(shape.variance_between(
            narrow(TRAIN_BAND[0] * peak_omega),
            narrow(TRAIN_BAND[1] * peak_omega),
        ));
        let density = omegas.map(|w| f64::from(shape.density(narrow(w))));
        let total: f64 = density.iter().sum();
        if !(variance > 0.0 && total > 0.0) {
            return None;
        }
        let trains = core::array::from_fn(|i| ShoreTrain {
            omega: omegas[i],
            amplitude: libm::sqrt(2.0 * variance * density[i] / total),
            phase: TAU * f64::from(unit_hash(u32::saturating_from(i), planet.seed ^ SHORE_SALT)),
        });
        let deepest = 0.5 * peak_wavelength;
        let far = deepest.min(0.95 * (coast.ocean_depth + sea_level));
        let reach = offshore_where(&coast, far);
        let table = PhaseTable::new(reach, omegas, SHALLOWEST, |u| coast.depth(waterline - u));
        let quiet_inland = if deepest < coast.ocean_depth + sea_level {
            waterline - reach.max(offshore_where(&coast, deepest))
        } else {
            QUIET_NOWHERE
        };
        let mut cascade_wavelengths = [0.0; MAX_OCEAN_CASCADES];
        if let Some(cascades) = cascades {
            for (c, slot) in cascade_wavelengths
                .iter_mut()
                .enumerate()
                .take(cascades.count())
            {
                let lowest = f64::from(cascades.band(c)[0]);
                *slot = if c == 0 || lowest <= 0.0 {
                    peak_wavelength
                } else {
                    TAU / lowest
                };
            }
        }
        Some(Self {
            trains,
            table,
            coast,
            peak_wavelength,
            cascade_wavelengths,
            radius: planet.radius + sea_level,
            continents: *c,
            planet_radius: planet.radius,
            quiet_inland,
            noise_salts: core::array::from_fn(|o| {
                lcg_hash(
                    planet.seed ^ SHORE_SALT ^ u32::saturating_from(o).wrapping_mul(0x9e37_79b9),
                )
            }),
        })
    }

    /// The trains.
    pub fn trains(&self) -> &[ShoreTrain; SHORE_TRAINS] {
        &self.trains
    }

    /// The trains' phase table.
    pub fn table(&self) -> &PhaseTable {
        &self.table
    }

    /// The coast's profile.
    pub fn coast(&self) -> &CoastProfile {
        &self.coast
    }

    /// The deep-water wavelength the sea's energy peaks at, in meters.
    pub fn peak_wavelength(&self) -> f64 {
        self.peak_wavelength
    }

    /// The wavelength each simulated cascade fades out at, as the water grows
    /// shallower than half of it; 0 past the cascades.
    pub fn cascade_wavelengths(&self) -> [f64; MAX_OCEAN_CASCADES] {
        self.cascade_wavelengths
    }

    /// The phase noise's octave salts, then the swell's.
    pub fn noise_salts(&self) -> [u32; NOISE_SALTS] {
        self.noise_salts
    }

    /// Lattice cells of the phase noise's broadest octave per unit of
    /// direction.
    pub fn noise_frequency(&self) -> f64 {
        self.radius / PHASE_NOISE_CELL
    }

    /// How far inland (negative: offshore) of the datum's coast the water
    /// first lies deep and far enough out that nothing of the shore reaches
    /// it: no train runs and no cascade fades. Very far offshore where the
    /// sea never grows that deep.
    pub fn quiet_inland(&self) -> f64 {
        self.quiet_inland
    }

    /// The planet's continents, which the distance inland is read from.
    pub fn continents(&self) -> &Continents {
        &self.continents
    }

    /// The planet's radius, in meters: the datum the continents lie on.
    pub fn planet_radius(&self) -> f64 {
        self.planet_radius
    }

    /// Lattice cells of the swell's variation per unit of direction.
    pub fn swell_frequency(&self) -> f64 {
        self.radius / SWELL_CELL
    }

    /// The radius of the sea, in meters.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Each train's phase at the waterline `time` seconds into the world,
    /// in `[0, tau)`.
    pub fn phases_at(&self, time: f64) -> [f64; SHORE_TRAINS] {
        self.trains.map(|t| wrap_phase(t.phase + t.omega * time))
    }

    /// How much of each cascade's waves survive in water `depth` meters deep.
    pub fn cascade_damping(&self, depth: f64) -> [f64; MAX_OCEAN_CASCADES] {
        self.cascade_wavelengths.map(|l| {
            if l > 0.0 {
                smoothstep(0.25 * l, 0.5 * l, depth)
            } else {
                1.0
            }
        })
    }

    /// The point `inland` meters inland of the datum's coast, the distance
    /// inland growing at `gradient` meters per meter along the sea.
    pub fn point(&self, inland: f64, gradient: DVec3) -> ShorePoint {
        let mut point = self.located(inland, gradient);
        let shallow = point.depth.max(SHALLOWEST);
        point.shoaling = core::array::from_fn(|i| {
            shoaling_at(self.trains[i].omega, point.wavenumbers[i], shallow)
        });
        point
    }

    /// The point `inland` meters inland, a few meters from `near`, taking its
    /// shoaling from there: it changes over the shelf's tens of meters, and
    /// is the dear part of a point.
    pub fn point_beside(&self, near: &ShorePoint, inland: f64, gradient: DVec3) -> ShorePoint {
        ShorePoint {
            shoaling: near.shoaling,
            ..self.located(inland, gradient)
        }
    }

    // Everything of the point at `inland` but its shoaling.
    fn located(&self, inland: f64, gradient: DVec3) -> ShorePoint {
        let depth = self.coast.depth(inland);
        let offshore = self.coast.waterline - inland;
        let g = dvec3::length(gradient);
        let (phases, wavenumbers) = self.table.at(offshore);
        let per_meter = 1.0 / g.max(GRADIENT_FADE[0]);
        let reach = self.table.reach();
        let peak = self.peak_wavelength;
        let weight = if depth > 0.0 {
            (1.0 - smoothstep(0.25 * peak, 0.5 * peak, depth))
                * (1.0 - smoothstep(0.8 * reach, reach, offshore))
                * smoothstep(GRADIENT_FADE[0], GRADIENT_FADE[1], g)
        } else {
            0.0
        };
        ShorePoint {
            depth,
            weight,
            phases: phases.map(|p| p * per_meter),
            wavenumbers,
            shoaling: [1.0; SHORE_TRAINS],
            landward: if g > 1e-9 {
                dvec3::scale(gradient, 1.0 / g)
            } else {
                [0.0; 3]
            },
        }
    }

    /// Where unit direction `dir` lies in every noise octave the shore
    /// reads, in [`SHORE_OCTAVES`] order, so a sampler near `dir` works from
    /// small offsets however large the planet.
    pub fn lattice_at(&self, dir: DVec3) -> [LatticePlace; SHORE_OCTAVES] {
        let c = &self.continents;
        let salts = self.noise_salts;
        let place = |(cell, within)| LatticePlace { cell, within };
        let mut out = [LatticePlace::default(); SHORE_OCTAVES];
        let continents = octave_cells(
            dir,
            c.base_frequency(self.planet_radius),
            CONTINENT_OCTAVES,
            |o| c.octave_salt(o),
        );
        let phase = octave_cells(dir, self.noise_frequency(), PHASE_NOISE_OCTAVES, |o| {
            salts[usize::saturating_from(o)]
        });
        let swell = octave_cells(dir, self.swell_frequency(), 1, |_| salts[NOISE_SALTS - 1]);
        for (slot, octave) in out.iter_mut().zip(continents.chain(phase).chain(swell)) {
            *slot = place(octave);
        }
        out
    }

    /// The shore's slow variation in unit direction `dir`.
    pub fn noise_at(&self, dir: DVec3) -> ShoreNoise {
        let salts = self.noise_salts;
        let (sum, gradient) = fbm(dir, self.noise_frequency(), PHASE_NOISE_OCTAVES, |o| {
            salts[usize::saturating_from(o)]
        });
        let k = 2.0 * PHASE_NOISE_RADIANS / octave_weight_sum(PHASE_NOISE_OCTAVES);
        let along = dvec3::sub(gradient, dvec3::scale(dir, dvec3::dot(dir, gradient)));
        let (swell, _) = fbm(dir, self.swell_frequency(), 1, |_| salts[NOISE_SALTS - 1]);
        ShoreNoise {
            phase: k * sum - PHASE_NOISE_RADIANS,
            gradient: dvec3::scale(along, k / self.radius),
            swell: 1.0 + SWELL_VARIATION * (2.0 * swell - 1.0),
        }
    }

    /// The most a breaking crest carries the surface toward the land, in
    /// meters: its pinch over the wavenumber, which shoaling only raises above
    /// the slowest train's deep-water one.
    pub fn max_shift(&self) -> f64 {
        let slowest = self
            .trains
            .iter()
            .map(|t| t.omega)
            .fold(f64::INFINITY, f64::min);
        MAX_PINCH * f64::from(GRAVITY) / (slowest * slowest)
    }

    /// The waves at `point`, its slow variation `noise`, the trains at
    /// waterline phases `phases`.
    pub fn surface(
        &self,
        point: &ShorePoint,
        noise: &ShoreNoise,
        phases: [f64; SHORE_TRAINS],
    ) -> ShoreSurface {
        let Some((g, toward, k)) = self.group(point, noise, phases) else {
            return ShoreSurface::default();
        };
        let b = Breaker::of(point, &g);
        let (sc, cc) = libm::sincos(g.carrier);
        let du = 1.0 + b.lean * sc;
        let pinch = MAX_PINCH * smoothstep(0.4, 1.0, b.ratio) / (1.0 + b.lean);
        let (su, cu) = libm::sincos(g.carrier + b.lean * (1.0 - cc));
        let rise = -b.held * su * du * k;
        let spread = -pinch * cu * du;
        let since = wrap_phase(-g.carrier);
        let trail = libm::exp(-since / (TAU * FOAM_TRAIL));
        let lead = smoothstep(TAU - FOAM_LEAD, TAU, since);
        let crest = smoothstep(0.85, 1.15, b.ratio) * trail.max(lead);
        let rms = 2.0 * libm::sqrt(g.energy) / b.cap.max(1e-6);
        ShoreSurface {
            height: b.held * cu,
            shift: dvec3::scale(toward, -pinch * su / k),
            slope: dvec3::scale(toward, rise / (1.0 + spread)),
            foam: crest.max(RESIDUAL_FOAM * smoothstep(0.8, 1.5, rms)),
            celerity: dvec3::scale(toward, self.trains[0].omega / k),
        }
    }

    /// Only the height of [`ShoreWaves::surface`], summed in single
    /// precision as the shader sums it: the phases are wrapped first, so
    /// nothing is lost.
    pub fn height(
        &self,
        point: &ShorePoint,
        noise: &ShoreNoise,
        phases: [f64; SHORE_TRAINS],
    ) -> f64 {
        if point.weight <= 0.0 {
            return 0.0;
        }
        let (mut re, mut im, mut total, mut k_sum) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for (i, t) in self.trains.iter().enumerate() {
            let a = narrow(point.weight * noise.swell * t.amplitude * point.shoaling[i]);
            let p = narrow(-wrap_phase(phases[i] + point.phases[i] + noise.phase));
            let (s, c) = concinnity_core::math::sin_cos(p);
            re += a * c;
            im += a * s;
            total += a;
            k_sum += a * narrow(point.wavenumbers[i]);
        }
        let travel = dvec3::sub(
            dvec3::scale(point.landward, f64::from(k_sum / total.max(1e-12))),
            noise.gradient,
        );
        if dvec3::length(travel) <= 1e-9 {
            return 0.0;
        }
        let group = Group {
            envelope: f64::from(concinnity_core::math::hypot(re, im)),
            carrier: f64::from(concinnity_core::math::atan2(im, re)),
            energy: 0.0,
        };
        let b = Breaker::of(point, &group);
        let carrier = narrow(group.carrier);
        let lean = narrow(b.lean);
        let u = carrier + lean * (1.0 - concinnity_core::math::cos(carrier));
        b.held * f64::from(concinnity_core::math::cos(u))
    }

    // The trains summed at `point` into one group, its direction of travel
    // and its wavenumber; `None` where no train runs.
    fn group(
        &self,
        point: &ShorePoint,
        noise: &ShoreNoise,
        phases: [f64; SHORE_TRAINS],
    ) -> Option<(Group, DVec3, f64)> {
        if point.weight <= 0.0 {
            return None;
        }
        let mut re = 0.0;
        let mut im = 0.0;
        let mut total = 0.0;
        let mut energy = 0.0;
        let mut k_sum = 0.0;
        for (i, t) in self.trains.iter().enumerate() {
            let a = point.weight * noise.swell * t.amplitude * point.shoaling[i];
            let (s, c) = libm::sincos(-wrap_phase(phases[i] + point.phases[i] + noise.phase));
            re += a * c;
            im += a * s;
            total += a;
            energy += a * a;
            k_sum += a * point.wavenumbers[i];
        }
        let travel = dvec3::sub(
            dvec3::scale(point.landward, k_sum / total.max(1e-12)),
            noise.gradient,
        );
        let k = dvec3::length(travel);
        if k <= 1e-9 {
            return None;
        }
        let group = Group {
            envelope: libm::hypot(re, im),
            carrier: libm::atan2(im, re),
            energy,
        };
        Some((group, dvec3::scale(travel, 1.0 / k), k))
    }

    /// The shore waves in unit direction `dir` of `planet` (the one the sea
    /// was built for), `time` seconds into the world: the query a body
    /// floating near the shore reads. Its vectors are in the planet's
    /// authored axes.
    pub fn at(&self, planet: &PlanetShape, dir: DVec3, time: f64) -> ShoreSurface {
        let Some((inland, gradient)) = planet.inland(dir) else {
            return ShoreSurface::default();
        };
        let point = self.point(inland, gradient);
        self.surface(&point, &self.noise_at(dir), self.phases_at(time))
    }
}

// How far offshore the water first reaches `depth` meters, by bisection on
// the profile, which only deepens seaward.
fn offshore_where(coast: &CoastProfile, depth: f64) -> f64 {
    let (mut lo, mut hi) = (0.0, coast.shelf.max(1.0));
    while coast.depth(coast.waterline - hi) < depth && hi < 1.0e7 {
        hi *= 2.0;
    }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if coast.depth(coast.waterline - mid) < depth {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

// `x` radians wrapped into `[0, tau)`.
pub(super) fn wrap_phase(x: f64) -> f64 {
    let r = x - TAU * libm::floor(x / TAU);
    if r >= TAU { 0.0 } else { r }
}

/// How far, in meters, the surf's foam has drifted shoreward `age` periods
/// after its crest passed, on a wave of wavenumber `k`: fresh foam rides in
/// with the bore, then lags, stops a third of the way to the next crest and
/// slips back with the backwash to where it formed as the next one breaks.
pub fn foam_drift(age: f64, k: f64) -> f64 {
    let a = age.clamp(0.0, 1.0);
    FOAM_RIDE * TAU / k.max(1e-4) * a * (1.0 - a) * (1.0 - a)
}

/// Hermite step from 0 at `a` to 1 at `b`.
pub fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn narrow(x: f64) -> f32 {
    concinnity_core::math::narrow_f32(x)
}
