//! The authored `NightSky` resolved into the numbers its shaders and its
//! moon light are built from.

use super::galaxies::{Galaxy, MAX_GALAXIES, galaxy};
use super::hash::seed_lane;
use super::moon::MoonPath;
use super::stars::stars_per_steradian;
use crate::components::NightSky;
use concinnity_core::math::{SaturatingFrom, cos};

/// Light of a magnitude-0 star, as a fraction of the sunlight, at
/// `star_brightness` 1: the 26.74 magnitudes between a star like Vega and
/// the sun above the air.
pub(crate) const STAR_ZERO_ILLUMINANCE: f32 = 2.0e-11;
/// Radiance at the Milky Way's heart, as a fraction of the sunlight, at
/// `milky_way_brightness` 1: about 20 magnitudes per square arcsecond, its
/// brightest star clouds.
pub(crate) const MILKY_WAY_RADIANCE: f32 = 8.6e-9;
/// How much brighter galaxies are drawn than their magnitudes, so the larger
/// ones show as more than their cores.
pub(crate) const GALAXY_GAIN: f32 = 4.0;
/// The moon's dark side, lit by light from the planet, as a share of its lit
/// face.
pub(crate) const EARTHSHINE: f32 = 0.004;

/// The moon as resolved from the asset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoonSettings {
    /// Its path around the sky.
    pub path: MoonPath,
    /// Cosine of its disc's angular radius.
    pub cos_radius: f32,
    /// The full moon's light as a fraction of the sunlight.
    pub brightness: f32,
    /// Its face's brightness, 1 being a white surface it lights head-on.
    pub disc_brightness: f32,
}

/// A `NightSky` resolved: the stars, the Milky Way, the galaxies and the
/// moon. Built once at init.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NightSkySettings {
    /// Light of a magnitude-0 star, as a fraction of the sunlight.
    pub star_illuminance: f32,
    /// Magnitude of the faintest stars.
    pub star_limit: f32,
    /// Mean stars per steradian down to that magnitude.
    pub star_density: f32,
    /// The 24-bit seed the stars and galaxies are drawn from.
    pub seed: u32,
    /// Radiance at the Milky Way's heart, as a fraction of the sunlight.
    pub milky_way_radiance: f32,
    /// Lean of the Milky Way's pole from the celestial pole, radians.
    pub milky_way_tilt: f32,
    /// Which way it leans, radians east of the sun.
    pub milky_way_turn: f32,
    /// Where along the band the core is, radians.
    pub milky_way_core: f32,
    /// The galaxies; the first `galaxy_count` are drawn.
    pub galaxies: [Galaxy; MAX_GALAXIES],
    /// How many galaxies are drawn.
    pub galaxy_count: usize,
    /// The moon, if the sky has one.
    pub moon: Option<MoonSettings>,
}

impl NightSkySettings {
    /// Resolve an authored night sky.
    pub fn resolve(night: &NightSky) -> Self {
        let n = crate::components::validate::night_sky(night.clone());
        let seed = u32::saturating_from(seed_lane(n.seed));
        Self {
            star_illuminance: STAR_ZERO_ILLUMINANCE * n.star_brightness,
            star_limit: n.star_limit_magnitude,
            star_density: stars_per_steradian(n.star_limit_magnitude),
            seed,
            milky_way_radiance: MILKY_WAY_RADIANCE * n.milky_way_brightness,
            milky_way_tilt: n.milky_way_tilt_deg.to_radians(),
            milky_way_turn: n.milky_way_turn_deg.to_radians(),
            milky_way_core: n.milky_way_core_deg.to_radians(),
            galaxies: core::array::from_fn(|i| galaxy(seed, u32::saturating_from(i))),
            galaxy_count: (n.galaxies as usize).min(MAX_GALAXIES),
            moon: n.moon.then(|| MoonSettings {
                path: MoonPath {
                    start_deg: n.moon_start_deg,
                    period_days: n.moon_period_days,
                    inclination: n.moon_inclination_deg.to_radians(),
                },
                cos_radius: cos((n.moon_angular_diameter_deg * 0.5).to_radians()),
                brightness: n.moon_brightness,
                disc_brightness: n.moon_disc_brightness,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_resolve_to_a_naked_eye_sky_with_a_full_moon() {
        let s = NightSkySettings::resolve(&NightSky::default());
        assert_eq!(s.star_illuminance, STAR_ZERO_ILLUMINANCE);
        assert!((s.star_density * 4.0 * core::f32::consts::PI - 9000.0).abs() < 1.0);
        assert_eq!(s.galaxy_count, 3);
        let moon = s.moon.expect("a moon by default");
        assert_eq!(moon.path.start_deg, 180.0);
        assert!((moon.cos_radius - cos(0.26_f32.to_radians())).abs() < 1e-7);
        assert!((s.milky_way_tilt.to_degrees() - 62.9).abs() < 1e-4);
    }

    #[test]
    fn out_of_range_values_are_clamped() {
        let s = NightSkySettings::resolve(&NightSky {
            star_limit_magnitude: 20.0,
            galaxies: 500,
            moon: false,
            star_brightness: -2.0,
            ..Default::default()
        });
        assert_eq!(s.star_limit, 8.0);
        assert_eq!(s.galaxy_count, MAX_GALAXIES);
        assert_eq!(s.star_illuminance, 0.0);
        assert!(s.moon.is_none());
    }
}
