// Land and sea on a planet: a broad noise field split at a threshold into
// continents above the datum and basins below it, and the coast's profile
// across the line between them.

use alloc::vec::Vec;
use concinnity_core::math::dvec3::DVec3;
use concinnity_core::math::noise::lcg_hash;
use concinnity_core::math::{SaturatingFrom, dvec3};

use super::shape::{fbm, octave_weight_sum};

/// Octaves the continent field sums: the finest bends the coast at a
/// thirty-second of a continent's width.
pub const CONTINENT_OCTAVES: u32 = 6;

/// How steeply a beach rises across its width on average, in meters per
/// meter. At the waterline it rises twice as steeply, and the sea floor falls
/// away at that same slope before it levels off.
pub const BEACH_SLOPE: f64 = 0.04;

/// The share of the ocean depth the sea floor's hills may lift it by.
pub const SEABED_RELIEF: f64 = 0.5;

/// How far past the beach the hills take to rise to their full height, as a
/// share of their width.
pub const HILL_RISE: f64 = 0.5;

// Directions the field is sampled at to place the coast.
const CALIBRATION_SAMPLES: usize = 4096;

// The most meters inland a unit of the field may span, in continent widths:
// at an extreme land share the samples nearest the coast sit on the field's
// flat extremes, and their mean steepness says little.
const MAX_COAST_SCALE: f64 = 4.0;

// Separates the continent field's salts from the hills'.
const CONTINENT_SALT: u32 = 0x5bd1_e995;

/// How a planet's ground divides into land and sea around its datum: a broad
/// field raises continents above it and sinks ocean basins below it, and
/// across the coast between them the ground rises gently out of the water.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Continents {
    size: f64,
    land_share: f64,
    ocean_depth: f64,
    beach_width: f64,
    // The field's value along the coast.
    threshold: f64,
    // Meters inland per unit of the field, so a step of a meter across the
    // coast moves the distance a meter on average.
    coast_scale: f64,
    seed: u32,
}

/// The ground across a coast at one point: its height, and how it changes
/// with the distance inland and with the hills' height there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Shore {
    pub height: f64,
    pub per_meter_inland: f64,
    pub per_hill_meter: f64,
}

impl Continents {
    /// No sea: the whole planet is land.
    pub const NONE: Self = Self {
        size: 20_000.0,
        land_share: 1.0,
        ocean_depth: 0.0,
        beach_width: 1.0,
        threshold: 0.0,
        coast_scale: 0.0,
        seed: 0,
    };

    /// Continents `size` meters across covering `land_share` of a planet of
    /// `radius` picked by `seed`, the coast placed by sampling the field over
    /// the whole sphere, so the share holds for every seed.
    pub fn new(
        radius: f64,
        seed: u32,
        size: f64,
        land_share: f64,
        ocean_depth: f64,
        beach_width: f64,
    ) -> Self {
        let mut c = Self {
            size: size.max(1.0),
            land_share: land_share.clamp(0.0, 1.0),
            ocean_depth: ocean_depth.max(0.0),
            beach_width: beach_width.max(1.0),
            seed,
            ..Self::NONE
        };
        if !c.divides() {
            return c;
        }
        let samples: Vec<(f64, f64)> = (0..CALIBRATION_SAMPLES)
            .map(|i| {
                let dir = fibonacci_direction(i, CALIBRATION_SAMPLES);
                let (value, gradient) = c.field(radius, dir);
                let across = dvec3::sub(gradient, dvec3::scale(dir, dvec3::dot(dir, gradient)));
                (value, dvec3::length(across))
            })
            .collect();
        let mut values: Vec<f64> = samples.iter().map(|s| s.0).collect();
        values.sort_unstable_by(f64::total_cmp);
        c.threshold = if c.land_share <= 0.0 {
            1.0 + f64::EPSILON
        } else {
            let at = usize::saturating_from((1.0 - c.land_share) * values.len() as f64);
            values[at.min(values.len() - 1)]
        };
        // The field's mean steepness where it crosses the coast, per unit of
        // direction, from the samples nearest it.
        let mut nearest = samples;
        nearest.sort_unstable_by(|a, b| {
            (a.0 - c.threshold)
                .abs()
                .total_cmp(&(b.0 - c.threshold).abs())
        });
        let near = &nearest[..CALIBRATION_SAMPLES / 16];
        let steepness = near.iter().map(|s| s.1).sum::<f64>() / near.len() as f64;
        let most = MAX_COAST_SCALE * c.size;
        c.coast_scale = if steepness > 0.0 {
            (radius / steepness).min(most)
        } else {
            most
        };
        c
    }

    /// Width of the broadest continents, in meters.
    pub fn size(&self) -> f64 {
        self.size
    }

    /// Share of the surface above the datum, in `[0, 1]`; 1 is all land.
    pub fn land_share(&self) -> f64 {
        self.land_share
    }

    /// Deepest the sea floor lies below the datum, in meters.
    pub fn ocean_depth(&self) -> f64 {
        self.ocean_depth
    }

    /// Width of the gentle rise from the waterline before the hills begin, in
    /// meters.
    pub fn beach_width(&self) -> f64 {
        self.beach_width
    }

    /// Whether any of the planet lies under the datum.
    pub fn divides(&self) -> bool {
        self.land_share < 1.0
    }

    /// How high the beach rises before the hills begin, in meters.
    pub fn beach_rise(&self) -> f64 {
        if self.divides() {
            BEACH_SLOPE * self.beach_width
        } else {
            0.0
        }
    }

    /// The tallest the hills on top of the beach get on a planet whose ground
    /// reaches `amplitude` meters.
    pub fn hill_height(&self, amplitude: f64) -> f64 {
        (amplitude.max(0.0) - self.beach_rise()).max(0.0)
    }

    /// How far past the waterline the sea floor takes to fall most of the
    /// way to the ocean depth, in meters.
    pub fn shelf_length(&self) -> f64 {
        self.ocean_depth / (2.0 * BEACH_SLOPE)
    }

    /// The field's value along the coast.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Meters inland per unit of the field.
    pub fn coast_scale(&self) -> f64 {
        self.coast_scale
    }

    /// Lattice cells of the broadest octave per unit of direction on a planet
    /// of `radius`.
    pub fn base_frequency(&self, radius: f64) -> f64 {
        radius / self.size
    }

    /// Octave `octave`'s hash salt.
    pub fn octave_salt(&self, octave: u32) -> u32 {
        lcg_hash(lcg_hash(self.seed ^ CONTINENT_SALT) ^ octave.wrapping_mul(0x9e37_79b9))
    }

    /// The sea floor's lift per meter of hill under it, so the tallest hill
    /// lifts it [`SEABED_RELIEF`] of the ocean depth.
    pub fn seabed_lift(&self, hills: f64) -> f64 {
        if hills > 0.0 {
            SEABED_RELIEF * self.ocean_depth / hills
        } else {
            0.0
        }
    }

    // The field in [0, 1] in direction `dir` and its gradient over directions.
    fn field(&self, radius: f64, dir: DVec3) -> (f64, DVec3) {
        let (sum, gradient) = fbm(dir, self.base_frequency(radius), CONTINENT_OCTAVES, |o| {
            self.octave_salt(o)
        });
        let k = 1.0 / octave_weight_sum(CONTINENT_OCTAVES);
        (sum * k, dvec3::scale(gradient, k))
    }

    /// The distance inland from the coast in direction `dir` (unit length) on a
    /// planet of `radius`, in meters, negative out to sea, and its gradient
    /// over directions. Exact at the coast, an estimate elsewhere.
    pub(super) fn inland(&self, radius: f64, dir: DVec3) -> (f64, DVec3) {
        let (value, gradient) = self.field(radius, dir);
        (
            (value - self.threshold) * self.coast_scale,
            dvec3::scale(gradient, self.coast_scale),
        )
    }

    /// The ground `inland` meters from the coast where the hills stand `hill`
    /// high, on a planet whose ground reaches `amplitude` meters, the hills
    /// rising in over `rise` meters.
    ///
    /// Inland the beach rises from the waterline over its width, then the
    /// hills grow in on top of it; out to sea the floor falls away at the
    /// beach's slope and levels off toward the ocean depth, lifted by the
    /// hills under it away from the shore.
    pub(super) fn shore(&self, inland: f64, hill: f64, amplitude: f64, rise: f64) -> Shore {
        let rise = rise.max(1e-3);
        if inland >= 0.0 {
            let w = self.beach_width;
            let t = (inland / w).min(1.0);
            let beach = self.beach_rise();
            let (fade, fade_d) = smoothstep((inland - w) / rise);
            return Shore {
                height: beach * t * (2.0 - t) + hill * fade,
                per_meter_inland: beach * 2.0 * (1.0 - t) / w + hill * fade_d / rise,
                per_hill_meter: fade,
            };
        }
        if self.ocean_depth <= 0.0 {
            return Shore {
                height: 0.0,
                per_meter_inland: 0.0,
                per_hill_meter: 0.0,
            };
        }
        let out = -inland;
        let shelf = self.shelf_length();
        let e = libm::exp(-out / shelf);
        let (fall, fall_d) = (1.0 - e, e / shelf);
        let (fade, fade_d) = smoothstep(out / rise);
        let lift = self.seabed_lift(self.hill_height(amplitude));
        let d_out =
            -self.ocean_depth * fall_d + lift * hill * (fall_d * fade + fall * fade_d / rise);
        Shore {
            height: -self.ocean_depth * fall + lift * hill * fall * fade,
            per_meter_inland: -d_out,
            per_hill_meter: lift * fall * fade,
        }
    }

    /// The steepest the ground's height can change per meter of distance
    /// inland, and per meter of hill, anywhere, on a planet whose ground
    /// reaches `amplitude` meters, the hills rising in over `rise` meters.
    pub(super) fn shore_bounds(&self, amplitude: f64, rise: f64) -> (f64, f64) {
        let rise = rise.max(1e-3);
        let beach = self.beach_rise();
        let hills = self.hill_height(amplitude);
        let land = 2.0 * beach / self.beach_width + hills * 1.5 / rise;
        let lift = self.seabed_lift(hills);
        let sea = if self.ocean_depth > 0.0 {
            let shelf = self.shelf_length();
            self.ocean_depth / shelf + lift * hills * (1.0 / shelf + 1.5 / rise)
        } else {
            0.0
        };
        (land.max(sea), lift.max(1.0))
    }

    /// The steepest the distance inland can change per unit of direction on
    /// a planet of `radius`: a bound from each octave's lattice spacing.
    pub(super) fn inland_slope_bound(&self, radius: f64) -> f64 {
        let per_octave = self.base_frequency(radius) * 1.5 * libm::sqrt(3.0);
        self.coast_scale * per_octave * f64::from(CONTINENT_OCTAVES)
            / octave_weight_sum(CONTINENT_OCTAVES)
    }
}

// Smoothstep of `x` clamped to [0, 1], and its slope.
fn smoothstep(x: f64) -> (f64, f64) {
    let t = x.clamp(0.0, 1.0);
    (t * t * (3.0 - 2.0 * t), 6.0 * t * (1.0 - t))
}

// The `i`th of `n` directions spread evenly over the sphere.
pub(super) fn fibonacci_direction(i: usize, n: usize) -> DVec3 {
    let golden = core::f64::consts::PI * (3.0 - libm::sqrt(5.0));
    let y = 1.0 - (i as f64 + 0.5) / n as f64 * 2.0;
    let r = libm::sqrt((1.0 - y * y).max(0.0));
    let a = golden * i as f64;
    [libm::cos(a) * r, y, libm::sin(a) * r]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn continents() -> Continents {
        Continents::new(50_000.0, 4, 20_000.0, 0.4, 60.0, 60.0)
    }

    // Across the coast the ground with no hills only ever rises inland, from
    // the ocean depth to the top of the beach.
    #[test]
    fn the_coast_profile_rises_inland() {
        let c = continents();
        let mut last = f64::NEG_INFINITY;
        for i in -4_000..=1_000 {
            let inland = f64::from(i);
            let s = c.shore(inland, 0.0, 40.0, 1_000.0);
            assert!(s.height >= last, "{inland}: {} < {last}", s.height);
            assert!(s.per_meter_inland >= 0.0);
            last = s.height;
        }
        assert_eq!(c.shore(0.0, 0.0, 40.0, 1_000.0).height, 0.0);
        assert!(c.shore(-4_000.0, 0.0, 40.0, 1_000.0).height < -59.0);
        assert!((c.shore(60.0, 0.0, 40.0, 1_000.0).height - 2.4).abs() < 1e-12);
    }

    // The beach and the sea floor meet the waterline at the same slope.
    #[test]
    fn the_waterline_has_no_kink() {
        let c = continents();
        let land = c.shore(0.0, 0.0, 40.0, 1_000.0).per_meter_inland;
        let sea = c.shore(-1e-9, 0.0, 40.0, 1_000.0).per_meter_inland;
        assert!((land - sea).abs() < 1e-6, "{land} vs {sea}");
        assert!((land - 2.0 * BEACH_SLOPE).abs() < 1e-9);
    }

    #[test]
    fn the_shore_s_slopes_match_its_heights() {
        let c = continents();
        let e = 1e-4;
        for i in -60..=60 {
            let inland = f64::from(i) * 37.0;
            for hill in [0.0, 11.0, 37.6] {
                let s = c.shore(inland, hill, 40.0, 1_000.0);
                let along = (c.shore(inland + e, hill, 40.0, 1_000.0).height
                    - c.shore(inland - e, hill, 40.0, 1_000.0).height)
                    / (2.0 * e);
                let up = (c.shore(inland, hill + e, 40.0, 1_000.0).height
                    - c.shore(inland, hill - e, 40.0, 1_000.0).height)
                    / (2.0 * e);
                assert!((along - s.per_meter_inland).abs() < 1e-5, "{inland} {hill}");
                assert!((up - s.per_hill_meter).abs() < 1e-5, "{inland} {hill}");
                let (along_max, up_max) = c.shore_bounds(40.0, 1_000.0);
                assert!(s.per_meter_inland.abs() <= along_max);
                assert!(s.per_hill_meter.abs() <= up_max);
                assert!(s.height >= -60.0 && s.height <= 40.0, "{}", s.height);
            }
        }
    }

    #[test]
    fn the_coast_is_placed_per_seed_and_none_divides_nothing() {
        let a = continents();
        let b = Continents::new(50_000.0, 5, 20_000.0, 0.4, 60.0, 60.0);
        assert!(a.divides());
        assert_ne!(a.threshold(), b.threshold());
        assert_eq!(a, continents());
        assert!(a.coast_scale() > 0.0);
        assert!(!Continents::NONE.divides());
        assert_eq!(Continents::NONE.beach_rise(), 0.0);
        let sea = Continents::new(50_000.0, 4, 20_000.0, 0.0, 60.0, 60.0);
        for i in 0..64 {
            let dir = fibonacci_direction(i, 64);
            assert!(sea.inland(50_000.0, dir).0 < 0.0);
        }
    }

    // Low hills, or none, do not cut the beach short: the land meets the sea
    // floor at the same slope whatever the hills' height.
    #[test]
    fn low_hills_leave_the_waterline_without_a_kink() {
        let c = continents();
        for amplitude in [0.0, 1.0, 40.0] {
            let land = c.shore(0.0, 0.0, amplitude, 1_000.0);
            let sea = c.shore(-1e-9, 0.0, amplitude, 1_000.0);
            assert!(
                (land.per_meter_inland - sea.per_meter_inland).abs() < 1e-6,
                "{amplitude}"
            );
            assert!((c.shore(60.0, 0.0, amplitude, 1_000.0).height - 2.4).abs() < 1e-12);
        }
    }

    // At an extreme share the samples nearest the coast sit on the field's
    // flat extremes; the coast's scale stays bounded all the same.
    #[test]
    fn an_extreme_share_keeps_a_bounded_coast_scale() {
        for share in [0.001, 0.999] {
            let c = Continents::new(50_000.0, 4, 20_000.0, share, 60.0, 60.0);
            assert!(
                c.coast_scale() > 0.0 && c.coast_scale() <= 80_000.0,
                "{share}"
            );
            assert!(c.inland_slope_bound(50_000.0).is_finite());
        }
    }

    #[test]
    fn fibonacci_directions_are_unit() {
        for i in 0..100 {
            let d = fibonacci_direction(i, 100);
            assert!((dvec3::length(d) - 1.0).abs() < 1e-12);
        }
    }
}
