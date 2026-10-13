//! The waves a wind raises on open water: a fetch-limited JONSWAP spectrum
//! sampled into a fixed set of discrete waves, spread in direction around the
//! wind, each moving at its deep-water speed.

use crate::math::{cos, exp, ln, powf, sin, sqrt};
use alloc::vec::Vec;
use core::f32::consts::{FRAC_PI_2, TAU};

/// Gravitational acceleration, in meters per second squared.
pub const GRAVITY: f32 = 9.81;

/// Waves a wind sea is sampled into.
pub const WIND_SEA_WAVES: usize = 20;

/// The slowest mean wind, in meters per second, that raises a wind sea. Below
/// it the surface keeps only its ripples.
pub const CALM_WIND: f32 = 0.5;

// The dimensionless fetch `g F / U^2` past which the sea is fully developed: a
// longer fetch builds nothing more (Pierson-Moskowitz).
const FULLY_DEVELOPED_FETCH: f32 = 2.2e4;
// JONSWAP peak enhancement and its widths below and above the peak.
const PEAK_GAMMA: f32 = 3.3;
const PEAK_SIGMA: [f32; 2] = [0.07, 0.09];
// The sampled band, as multiples of the peak wavelength.
const LONGEST_PER_PEAK: f32 = 1.8;
const SHORTEST_PER_PEAK: f32 = 1.0 / 12.0;
// The directional spreading exponent `s` of `cos^(2s)(theta / 2)` at the peak;
// it narrows toward the peak and widens away from it (Mitsuyasu's shape).
const PEAK_SPREAD: f32 = 8.0;
// The widest standard deviation a wave's heading strays from the wind by.
const MAX_SPREAD_RADIANS: f32 = 1.2;
// Every this many waves, one runs across the wind rather than with it, at
// this share of its band's amplitude.
const CROSS_WIND_EVERY: usize = 5;
const CROSS_WIND_AMPLITUDE: f32 = 0.5;

/// One wave of a water surface's sum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wave {
    /// Unit horizontal travel direction `[x, z]`.
    pub direction: [f32; 2],
    /// Peak height above rest, in meters.
    pub amplitude: f32,
    /// Distance between crests, in meters.
    pub wavelength: f32,
    /// Phase at the origin at time zero, in radians.
    pub phase: f32,
    /// Crest sharpness in [0, 1].
    pub steepness: f32,
}

impl Wave {
    /// Radians of phase per meter.
    pub fn wavenumber(&self) -> f32 {
        TAU / self.wavelength
    }

    /// Radians of phase per second: deep-water dispersion, `sqrt(g k)`.
    pub fn angular_speed(&self) -> f32 {
        sqrt(GRAVITY * self.wavenumber())
    }
}

/// The angular frequency, in radians per second, of the waves carrying the
/// most energy in a sea a `wind_speed` m/s wind builds over `fetch` meters of
/// open water (JONSWAP; a long enough fetch reaches the fully developed sea).
pub fn peak_angular_frequency(wind_speed: f32, fetch: f32) -> f32 {
    let x = dimensionless_fetch(wind_speed, fetch);
    22.0 * (GRAVITY / wind_speed) * powf(x, -1.0 / 3.0)
}

/// The wavelength, in meters, of a deep-water wave of angular frequency
/// `omega`.
pub fn wavelength_of(omega: f32) -> f32 {
    TAU * GRAVITY / (omega * omega)
}

fn dimensionless_fetch(wind_speed: f32, fetch: f32) -> f32 {
    (GRAVITY * fetch / (wind_speed * wind_speed)).min(FULLY_DEVELOPED_FETCH)
}

// The JONSWAP energy density at `omega` around the peak `omega_p`, in m^2 s.
fn jonswap(omega: f32, omega_p: f32, alpha: f32) -> f32 {
    let r = omega_p / omega;
    let pm = alpha * GRAVITY * GRAVITY * powf(omega, -5.0) * exp(-1.25 * r * r * r * r);
    let sigma = if omega <= omega_p {
        PEAK_SIGMA[0]
    } else {
        PEAK_SIGMA[1]
    };
    let d = (omega - omega_p) / (sigma * omega_p);
    pm * powf(PEAK_GAMMA, exp(-0.5 * d * d))
}

// The standard deviation, in radians, of the heading of a wave of `omega`
// around the wind.
fn spread_radians(omega: f32, omega_p: f32) -> f32 {
    let r = omega / omega_p;
    let s = if r <= 1.0 {
        PEAK_SPREAD * powf(r, 5.0)
    } else {
        PEAK_SPREAD * powf(r, -2.5)
    };
    // cos^(2s)(theta / 2) is close to a Gaussian of variance 2 / s.
    sqrt(2.0 / s.max(1e-3)).min(MAX_SPREAD_RADIANS)
}

// A unit float for wave `i` and a salt, the same on every platform.
pub(super) fn unit_hash(i: u32, salt: u32) -> f32 {
    let mut h = i.wrapping_mul(0x9e37_79b9) ^ salt.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// The waves a `wind_speed` m/s wind blowing toward unit `direction` raises
/// over `fetch` meters of open water, longest first: [`WIND_SEA_WAVES`]
/// wavelengths spread geometrically around the spectrum's peak, each carrying
/// its band's share of the energy. Headings scatter around the wind, wider
/// for waves far from the peak, with every few running across it. The same
/// inputs give the same waves. Empty below [`CALM_WIND`].
pub fn wind_sea(wind_speed: f32, direction: [f32; 2], fetch: f32) -> Vec<Wave> {
    // Written to also refuse a NaN speed or fetch.
    let calm = wind_speed.is_nan() || wind_speed < CALM_WIND;
    if calm || fetch.is_nan() || fetch <= 0.0 {
        return Vec::new();
    }
    let omega_p = peak_angular_frequency(wind_speed, fetch);
    let alpha = 0.076 * powf(dimensionless_fetch(wind_speed, fetch), -0.22);
    let longest = wavelength_of(omega_p) * LONGEST_PER_PEAK;
    let ratio = powf(
        SHORTEST_PER_PEAK / LONGEST_PER_PEAK,
        1.0 / (WIND_SEA_WAVES - 1) as f32,
    );
    // Each wave stands for the band of wavelengths around it; on this
    // geometric spacing that band is `omega * ln(1 / ratio) / 2` wide in
    // angular frequency.
    let band = 0.5 * ln(1.0 / ratio);
    let wind_angle = crate::math::atan2(direction[1], direction[0]);

    (0..WIND_SEA_WAVES)
        .map(|i| {
            let wavelength = longest * powf(ratio, i as f32);
            let omega = sqrt(GRAVITY * TAU / wavelength);
            let mut amplitude = sqrt(2.0 * jonswap(omega, omega_p, alpha) * omega * band);
            let seed = i as u32;
            let heading = if i % CROSS_WIND_EVERY == 2 {
                amplitude *= CROSS_WIND_AMPLITUDE;
                let side = if (i / CROSS_WIND_EVERY).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                };
                side * (FRAC_PI_2 * (0.75 + 0.25 * unit_hash(seed, 3)))
            } else {
                // An even scatter over [-1, 1], which a uniform of this width
                // turns into the band's standard deviation.
                let u = 2.0 * crate::math::fract(i as f32 * 0.618_034 + 0.31) - 1.0;
                u * sqrt(3.0) * spread_radians(omega, omega_p)
            };
            let angle = wind_angle + heading;
            Wave {
                direction: [cos(angle), sin(angle)],
                amplitude,
                wavelength,
                phase: unit_hash(seed, 7) * TAU,
                steepness: 1.0,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total_variance(waves: &[Wave]) -> f32 {
        waves.iter().map(|w| 0.5 * w.amplitude * w.amplitude).sum()
    }

    fn mean_heading(waves: &[Wave]) -> [f32; 2] {
        let mut d = [0.0f32; 2];
        for w in waves {
            d[0] += w.direction[0] * w.amplitude;
            d[1] += w.direction[1] * w.amplitude;
        }
        let len = crate::math::hypot(d[0], d[1]);
        [d[0] / len, d[1] / len]
    }

    #[test]
    fn calm_air_raises_no_waves() {
        assert!(wind_sea(0.0, [1.0, 0.0], 1000.0).is_empty());
        assert!(wind_sea(CALM_WIND * 0.5, [1.0, 0.0], 1000.0).is_empty());
        assert!(wind_sea(f32::NAN, [1.0, 0.0], 1000.0).is_empty());
        assert!(wind_sea(5.0, [1.0, 0.0], 0.0).is_empty());
    }

    // A breeze over a lake: a peak wavelength of around a meter and a
    // significant height of a few centimeters, which is what a lake shows.
    #[test]
    fn a_breeze_over_a_lake_raises_small_waves() {
        let waves = wind_sea(3.0, [1.0, 0.0], 1000.0);
        assert_eq!(waves.len(), WIND_SEA_WAVES);
        let peak = wavelength_of(peak_angular_frequency(3.0, 1000.0));
        assert!((0.8..2.0).contains(&peak), "{peak}");
        let hs = 4.0 * sqrt(total_variance(&waves));
        assert!((0.02..0.12).contains(&hs), "{hs}");
    }

    // A longer fetch and a stronger wind both build a longer, taller sea, up
    // to the fully developed one.
    #[test]
    fn fetch_and_wind_grow_the_sea_until_it_is_fully_developed() {
        let peak = |u: f32, f: f32| wavelength_of(peak_angular_frequency(u, f));
        assert!(peak(3.0, 100.0) < peak(3.0, 1000.0));
        assert!(peak(3.0, 1000.0) < peak(8.0, 1000.0));
        let height = |u: f32, f: f32| total_variance(&wind_sea(u, [1.0, 0.0], f));
        assert!(height(3.0, 100.0) < height(3.0, 1000.0));
        assert!(height(3.0, 1000.0) < height(8.0, 1000.0));
        // Past full development the fetch stops mattering: Pierson-Moskowitz
        // puts the peak of a 10 m/s sea near 0.8 g / U.
        assert_eq!(peak(10.0, 1.0e6), peak(10.0, 1.0e8));
        let omega = peak_angular_frequency(10.0, 1.0e8);
        assert!((omega * 10.0 / GRAVITY - 0.8).abs() < 0.05, "{omega}");
    }

    #[test]
    fn wavelengths_spread_geometrically_longest_first() {
        let waves = wind_sea(5.0, [0.0, 1.0], 5000.0);
        let ratio = waves[1].wavelength / waves[0].wavelength;
        assert!(ratio < 1.0);
        for pair in waves.windows(2) {
            assert!((pair[1].wavelength / pair[0].wavelength - ratio).abs() < 1e-4);
        }
        let peak = wavelength_of(peak_angular_frequency(5.0, 5000.0));
        assert!(waves[0].wavelength > peak && waves[WIND_SEA_WAVES - 1].wavelength < peak);
    }

    // The sea runs downwind on the whole, every wave heads within a right
    // angle or so of it, and some run across it.
    #[test]
    fn headings_spread_around_the_wind() {
        let dir = [0.6, -0.8];
        let waves = wind_sea(6.0, dir, 2000.0);
        let mean = mean_heading(&waves);
        assert!(mean[0] * dir[0] + mean[1] * dir[1] > 0.95, "{mean:?}");
        let along: Vec<f32> = waves
            .iter()
            .map(|w| w.direction[0] * dir[0] + w.direction[1] * dir[1])
            .collect();
        assert!(along.iter().all(|&c| c > -0.4), "{along:?}");
        assert!(along.iter().any(|&c| c.abs() < 0.4), "{along:?}");
        for w in &waves {
            assert!((crate::math::hypot(w.direction[0], w.direction[1]) - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn the_same_wind_raises_the_same_sea() {
        let a = wind_sea(4.0, [1.0, 0.0], 800.0);
        assert_eq!(a, wind_sea(4.0, [1.0, 0.0], 800.0));
        let phases: Vec<f32> = a.iter().map(|w| w.phase).collect();
        assert!(phases.iter().all(|p| (0.0..TAU).contains(p)));
        assert!(phases.windows(2).any(|p| (p[0] - p[1]).abs() > 0.5));
    }

    // Deep water carries a wave at `sqrt(g lambda / 2 pi)`: about 1.25 m/s
    // per square-root meter of wavelength.
    #[test]
    fn dispersion_sets_the_speed() {
        let w = Wave {
            direction: [1.0, 0.0],
            amplitude: 0.1,
            wavelength: 4.0,
            phase: 0.0,
            steepness: 0.0,
        };
        let celerity = w.angular_speed() / w.wavenumber();
        assert!((celerity - 1.25 * 2.0).abs() < 0.02, "{celerity}");
    }
}
