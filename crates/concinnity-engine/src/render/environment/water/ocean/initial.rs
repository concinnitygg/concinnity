//! A cascade's initial spectrum, h0(k): one random complex amplitude per
//! wavenumber its band holds, scaled to the energy the wind sea's spectrum
//! puts there and spread around the wind, plus the whole turns that wave
//! makes per repeat period. The shader evolves it each frame as
//! `h0(k) e^(-i w t) + conj(h0(-k)) e^(i w t)`, which keeps the height field
//! real and carries each wave along `k`.

use super::OceanSea;
use super::cascades::Cascades;
use super::clock::omega_steps;
use crate::render::environment::water::spectrum::{GRAVITY, unit_hash};
use alloc::vec::Vec;
use concinnity_core::math::{SaturatingFrom, cos, exp, ln, sin, sqrt};
use core::f32::consts::{PI, TAU};

/// One texel of an initial spectrum: `[Re h0, Im h0, turns per repeat, 0]`.
pub type SpectrumTexel = [f32; 4];

// Salt for the texels' random amplitudes, so they share no sequence with
// the summed waves' phases.
const SPECTRUM_SALT: u32 = 0x0cea_0001;

/// The signed frequency index texel index `i` of a `size`-point FFT holds:
/// `i` below `size / 2`, `i - size` above.
pub fn frequency_index(i: u32, size: u32) -> i32 {
    if i < size / 2 {
        i32::saturating_from(i)
    } else {
        i32::saturating_from(i) - i32::saturating_from(size)
    }
}

/// Cascade `c`'s initial spectrum for `sea`, row-major: texel `(x, y)` holds
/// the wave of wavenumber `2 pi / length * (fx, fy)`, `fx` and `fy` the
/// [`frequency_index`] of `x` and `y`, with `x` along world X and `y` along
/// world Z. Texels outside the cascade's band, the constant term and the
/// Nyquist row and column hold nothing.
pub fn initial_spectrum(sea: &OceanSea, cascades: &Cascades, c: usize) -> Vec<SpectrumTexel> {
    let size = cascades.size();
    let Some(shape) = sea.shape() else {
        return alloc::vec![[0.0; 4]; (size * size) as usize];
    };
    let length = cascades.lengths()[c];
    let dk = TAU / length;
    let [lo, hi] = cascades.band(c);
    let wind = sea.wind_angle();
    let half = i32::saturating_from(size) / 2;
    let mut texels = Vec::with_capacity((size * size) as usize);
    for y in 0..size {
        for x in 0..size {
            let (fx, fy) = (frequency_index(x, size), frequency_index(y, size));
            let k = [fx as f32 * dk, fy as f32 * dk];
            let kl = sqrt(k[0] * k[0] + k[1] * k[1]);
            let nyquist = fx == -half || fy == -half;
            if nyquist || kl <= 0.0 || kl < lo || kl >= hi {
                texels.push([0.0; 4]);
                continue;
            }
            let omega = sqrt(GRAVITY * kl);
            let heading = wrap_angle(concinnity_core::math::atan2(k[1], k[0]) - wind);
            let sigma = shape.spread(omega);
            let spread = exp(-0.5 * heading * heading / (sigma * sigma)) / (sqrt(TAU) * sigma);
            // S(k) = S(w) D(theta) (dw/dk) / k, with dw/dk = g / (2 w).
            let density = shape.density(omega) * spread * GRAVITY / (2.0 * omega * kl);
            let rms = 0.5 * sqrt(density * dk * dk);
            let [gr, gi] = gaussian_pair(u32::saturating_from(c) * size * size + y * size + x);
            texels.push([gr * rms, gi * rms, omega_steps(omega), 0.0]);
        }
    }
    texels
}

/// The variance, in square meters, of the height a cascade's spectrum
/// raises: every wave's mean square, `|h0(k)|^2 + |h0(-k)|^2` summed over k.
pub fn height_variance(texels: &[SpectrumTexel]) -> f32 {
    2.0 * texels
        .iter()
        .map(|t| t[0] * t[0] + t[1] * t[1])
        .sum::<f32>()
}

// `x` wrapped into [-pi, pi).
fn wrap_angle(x: f32) -> f32 {
    x - TAU * concinnity_core::math::floor((x + PI) / TAU)
}

// Two independent standard normal values for texel `i` (Box-Muller).
fn gaussian_pair(i: u32) -> [f32; 2] {
    let u1 = 1.0 - unit_hash(i, SPECTRUM_SALT);
    let u2 = unit_hash(i, SPECTRUM_SALT ^ 0x5bd1_e995);
    let r = sqrt(-2.0 * ln(u1));
    let a = TAU * u2;
    [r * cos(a), r * sin(a)]
}

#[cfg(test)]
mod tests {
    use super::super::{OceanQuality, OceanSea};
    use super::*;
    use concinnity_core::math::SaturatingFrom;

    fn sea() -> OceanSea {
        OceanSea {
            wind_speed: 9.0,
            direction: [0.8, 0.6],
            fetch: 1.0e5,
        }
    }

    fn all(quality: OceanQuality) -> (Cascades, Vec<Vec<SpectrumTexel>>) {
        let shape = sea().shape().unwrap();
        let c = Cascades::new(shape.peak_wavelength(), quality);
        let spectra = (0..c.count())
            .map(|i| initial_spectrum(&sea(), &c, i))
            .collect();
        (c, spectra)
    }

    // The cascades together raise the sea the spectrum describes: its
    // significant height over their bands, within the scatter of one random
    // draw.
    #[test]
    fn the_cascades_carry_the_spectrums_energy() {
        let (c, spectra) = all(OceanQuality::FULL);
        let total: f32 = spectra.iter().map(|s| height_variance(s)).sum();
        let shape = sea().shape().unwrap();
        let w_lo = (GRAVITY * TAU / c.lengths()[0]).sqrt();
        let w_hi = (GRAVITY * c.band(c.count() - 1)[1]).sqrt();
        let want = shape.variance_between(w_lo, w_hi);
        let hs = 4.0 * total.sqrt();
        let want_hs = 4.0 * want.sqrt();
        assert!((hs / want_hs - 1.0).abs() < 0.15, "{hs} vs {want_hs}");
        // A fresh breeze over a long fetch: a sea of about a meter and a half.
        assert!((0.8..2.5).contains(&hs), "{hs}");
    }

    // The energy runs downwind: the mean wavenumber, weighted by energy,
    // points along the wind.
    #[test]
    fn the_energy_runs_with_the_wind() {
        let (c, spectra) = all(OceanQuality::FULL);
        let size = c.size();
        let mut mean = [0.0f32; 2];
        for (i, t) in spectra[0].iter().enumerate() {
            let e = t[0] * t[0] + t[1] * t[1];
            let fx = frequency_index(u32::saturating_from(i) % size, size) as f32;
            let fy = frequency_index(u32::saturating_from(i) / size, size) as f32;
            let l = (fx * fx + fy * fy).sqrt().max(1.0);
            mean[0] += e * fx / l;
            mean[1] += e * fy / l;
        }
        let l = mean[0].hypot(mean[1]);
        let dir = sea().direction;
        assert!(
            (mean[0] / l * dir[0] + mean[1] / l * dir[1]) > 0.9,
            "{mean:?}"
        );
    }

    // Texels outside a cascade's band, its constant term and its Nyquist
    // lines are empty; inside, each carries its whole turns per repeat.
    #[test]
    fn only_the_band_is_filled() {
        let (c, spectra) = all(OceanQuality::FULL);
        let size = c.size();
        let dk = |b: usize| TAU / c.lengths()[b];
        for (b, s) in spectra.iter().enumerate() {
            let [lo, hi] = c.band(b);
            assert_eq!(s.len(), (size * size) as usize);
            assert_eq!(s[0], [0.0; 4]);
            for (i, t) in s.iter().enumerate() {
                let fx = frequency_index(u32::saturating_from(i) % size, size);
                let fy = frequency_index(u32::saturating_from(i) / size, size);
                let k = dk(b) * ((fx * fx + fy * fy) as f32).sqrt();
                if fx == -i32::saturating_from(size) / 2
                    || fy == -i32::saturating_from(size) / 2
                    || k <= 0.0
                    || k < lo
                    || k >= hi
                {
                    assert_eq!(*t, [0.0; 4], "{b} {i}");
                } else {
                    assert!(t[2] >= 1.0 && t[2] == t[2].round(), "{b} {i}");
                }
            }
        }
    }

    #[test]
    fn the_same_sea_gives_the_same_spectrum() {
        let (_, a) = all(OceanQuality::FULL);
        let (_, b) = all(OceanQuality::FULL);
        assert_eq!(a, b);
    }

    #[test]
    fn frequency_indices_wrap_at_half() {
        assert_eq!(frequency_index(0, 8), 0);
        assert_eq!(frequency_index(3, 8), 3);
        assert_eq!(frequency_index(4, 8), -4);
        assert_eq!(frequency_index(7, 8), -1);
    }
}
