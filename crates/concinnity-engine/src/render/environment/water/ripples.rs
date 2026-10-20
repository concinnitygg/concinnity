//! The procedural ripples below a body's shortest wave: octaves of value noise
//! from the longest ripple down to about a centimeter, each heading a little
//! off the wind and carried along its heading at its capillary-gravity speed.
//! The shader draws them; this fixes their shape once per body.

use super::spectrum::GRAVITY;
use concinnity_core::math::{atan2, cos, floor, powf, sin, sqrt};
use core::f32::consts::TAU;

/// Ripple octaves. Mirrors `WATER_RIPPLES` in `shaders/water.hlsl`.
pub const RIPPLES: usize = 4;

/// The shortest ripple, in meters.
const SHORTEST_RIPPLE: f32 = 0.012;

/// A crest's length over the ripple's, inverted: how much less the noise
/// varies along a crest than across it. Mirrors `WATER_RIPPLE_CREST`.
pub const RIPPLE_CREST: f32 = 0.4;

/// The lattice period of the ripple noise, in cells. The shader wraps every
/// lattice point into one period centered on zero, so an offset carried
/// modulo it is seamless. Mirrors `WATER_NOISE_PERIOD`.
pub const NOISE_PERIOD: f32 = 4096.0;

/// How fast the domain warp bending the ripple crests drifts downwind, in
/// meters per second.
pub const WARP_DRIFT: f32 = 0.3;

/// The domain warp's noise frequency, per meter. Mirrors
/// `WATER_WARP_FREQUENCY`.
pub const WARP_FREQUENCY: f32 = 2.0;

// The widest a ripple heads off the wind, in radians.
const SPREAD: f32 = 0.6;
// Surface tension over density, m^3 / s^2: capillary ripples outrun gravity
// waves of the same length.
const TENSION: f32 = 7.28e-5;
// Each octave's noise sits this many cells from the last, so no two share a
// lattice.
const OCTAVE_SEED: f32 = 17.31;

/// One ripple octave.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ripple {
    /// Unit horizontal heading `[x, z]`.
    pub direction: [f32; 2],
    /// Distance between crests, in meters.
    pub wavelength: f32,
    /// Speed along `direction`, in meters per second.
    pub speed: f32,
    /// The octave's lattice offset, in cells.
    pub seed: f32,
}

impl Ripple {
    /// The unit direction along its crests, a quarter turn from its heading.
    pub fn across(&self) -> [f32; 2] {
        [-self.direction[1], self.direction[0]]
    }
}

/// The ripple octaves under a wind blowing toward `wind_direction`, from
/// `longest` meters (held to at least `SHORTEST_RIPPLE`) down to
/// `SHORTEST_RIPPLE`, spaced evenly in log wavelength.
pub fn ripples(wind_direction: [f32; 2], longest: f32) -> [Ripple; RIPPLES] {
    let longest = longest.max(SHORTEST_RIPPLE);
    let wind_angle = atan2(wind_direction[1], wind_direction[0]);
    core::array::from_fn(|r| {
        let t = r as f32 / (RIPPLES - 1) as f32;
        let wavelength = longest * powf(SHORTEST_RIPPLE / longest, t);
        let k = TAU / wavelength;
        let heading = wind_angle + (golden_fraction(r) * 2.0 - 1.0) * SPREAD;
        Ripple {
            direction: [cos(heading), sin(heading)],
            wavelength,
            speed: sqrt(GRAVITY / k + TENSION * k),
            seed: r as f32 * OCTAVE_SEED,
        }
    })
}

// The fractional part of `r` golden-ratio steps past 0.2: the headings spread
// evenly however many octaves there are.
fn golden_fraction(r: usize) -> f32 {
    let x = r as f32 * 0.618_034 + 0.2;
    x - floor(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_constants_match_the_shader() {
        let src = crate::render::shader::embedded::WATER;
        assert_eq!(
            crate::render::shader::consts::uint(src, "WATER_RIPPLES"),
            RIPPLES
        );
        assert_eq!(
            crate::render::shader::consts::float(src, "WATER_RIPPLE_CREST"),
            RIPPLE_CREST
        );
        assert_eq!(
            crate::render::shader::consts::float(src, "WATER_NOISE_PERIOD"),
            NOISE_PERIOD
        );
        assert_eq!(
            crate::render::shader::consts::float(src, "WATER_WARP_FREQUENCY"),
            WARP_FREQUENCY
        );
    }

    #[test]
    fn the_octaves_span_longest_to_shortest_around_the_wind() {
        let set = ripples([0.0, 1.0], 0.5);
        assert!((set[0].wavelength - 0.5).abs() < 1e-6);
        assert!((set[RIPPLES - 1].wavelength - SHORTEST_RIPPLE).abs() < 1e-6);
        for pair in set.windows(2) {
            assert!(pair[0].wavelength > pair[1].wavelength);
        }
        for r in &set {
            let len = r.direction[0].hypot(r.direction[1]);
            assert!((len - 1.0).abs() < 1e-5);
            // Within the spread of the wind's heading.
            assert!(r.direction[1] >= cos(SPREAD) - 1e-5, "{r:?}");
            assert!(r.speed > 0.0);
        }
    }

    // A longest ripple below the floor is held to it.
    #[test]
    fn a_short_longest_ripple_is_held_to_the_floor() {
        let set = ripples([1.0, 0.0], 0.001);
        assert!(
            set.iter()
                .all(|r| (r.wavelength - SHORTEST_RIPPLE).abs() < 1e-7)
        );
    }
}
