//! Whitecaps on a simulated ocean. Each cascade keeps a foam field across
//! frames: where the surface converges into a sharpening crest, its Jacobian
//! falling below a threshold, the foam is topped up to full, and everywhere
//! it fades over [`FOAM_SECONDS`], so a breaking crest leaves a streak behind
//! it that thins out as the wave runs on.
//!
//! The threshold follows the wind so that the foam covers the share of the
//! sea observed at sea, `3.84e-6 U^3.41` (Monahan and O'Muircheartaigh): next
//! to none below about 4 m/s, about one percent at 10 m/s, several percent
//! in a gale. The foam reads the Jacobian to first order, `1 - chop * xi`,
//! `xi` the band's height curvature: a Gaussian field whose variance is the
//! band's slope variance, so its statistics follow from the spectrum alone.
//! The threshold sits as many of its standard deviations below one as makes
//! the foam injected, plus the wake each injection leaves as it fades (Rice's
//! level-crossing rate times the fade time), cover the band's share of the
//! target. The full Jacobian's second-order term thins that tail by a factor
//! that depends on the grid, which is why it is left out.

use super::cascades::Cascades;
use super::{MAX_OCEAN_CASCADES, OCEAN_CHOPPINESS, OceanSea};
use crate::math::{exp, ln, narrow_f32, powf, sqrt};
use crate::render::water::spectrum::{GRAVITY, SeaShape};
use core::f32::consts::TAU;

/// The time, in seconds, foam takes to fade to `1 / e` of itself.
pub const FOAM_SECONDS: f32 = 2.5;

/// The cascades that whitecap: the longest two, which hold the waves of a
/// meter and up that break. Shorter waves only roughen the surface.
pub const FOAM_CASCADES: usize = 2;

/// The Jacobian span, in its standard deviations, over which the injected
/// foam ramps from none to full: narrow, so a breaking crest turns white
/// rather than gray.
pub const FOAM_RAMP_SIGMAS: f32 = 0.1;

// The foam each crossing leaves behind as it fades, in fade times of full
// foam. Short of one: neighboring crossings overlap, so their wakes do too.
// Fit to the simulated update in this module's tests, which lands within 10%
// of the target at 128 and 256 texels.
const LINGER: f32 = 0.9;

/// The share of the sea whitecaps cover under a `wind_speed` m/s wind,
/// `3.84e-6 U^3.41`.
pub fn whitecap_coverage(wind_speed: f32) -> f32 {
    (3.84e-6 * powf(wind_speed.max(0.0), 3.41)).min(1.0)
}

/// How much of the foam survives `dt` seconds.
pub fn foam_keep(dt: f32) -> f32 {
    exp(-dt.max(0.0) / FOAM_SECONDS)
}

/// The time the foam last advanced to, which turns each frame's time into
/// how much of the foam survives it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FoamClock {
    last: Option<f64>,
}

impl FoamClock {
    /// How much of the foam survives from the last call to `time` seconds:
    /// none on the first call or when time runs backward, which restarts the
    /// foam from clear water.
    pub fn advance(&mut self, time: f64) -> f32 {
        let keep = match self.last {
            Some(last) if time >= last => foam_keep(narrow_f32(time - last)),
            _ => 0.0,
        };
        self.last = Some(time);
        keep
    }
}

/// Where each cascade injects foam: `(threshold - jacobian) * inv_ramp`,
/// clamped to `[0, 1]`. A cascade that never foams holds zero in both.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FoamCalibration {
    /// The Jacobian below which a cascade starts to foam.
    pub thresholds: [f32; MAX_OCEAN_CASCADES],
    /// One over the Jacobian span the injection ramps up over.
    pub inv_ramps: [f32; MAX_OCEAN_CASCADES],
}

impl FoamCalibration {
    /// The thresholds that make `sea`'s whitecaps cover
    /// [`whitecap_coverage`] of it, the first [`FOAM_CASCADES`] of
    /// `cascades` sharing it evenly.
    pub fn new(sea: &OceanSea, cascades: &Cascades) -> Self {
        let mut out = Self::default();
        let Some(shape) = sea.shape() else {
            return out;
        };
        let foaming = cascades.count().min(FOAM_CASCADES);
        let total = whitecap_coverage(sea.wind_speed);
        let share = 1.0 - powf(1.0 - total, 1.0 / foaming as f32);
        if share <= 0.0 {
            return out;
        }
        for c in 0..foaming {
            let band = BandJacobian::of(shape, cascades, c);
            if band.sigma <= 0.0 {
                continue;
            }
            out.thresholds[c] = 1.0 - band.sigma * band.depth_for(share);
            out.inv_ramps[c] = 1.0 / (FOAM_RAMP_SIGMAS * band.sigma);
        }
        out
    }
}

// The statistics of one cascade's Jacobian at a fixed point: its standard
// deviation about one, and how often per second it crosses one going down.
#[derive(Debug, Clone, Copy)]
struct BandJacobian {
    sigma: f32,
    crossings: f32,
}

impl BandJacobian {
    // From the spectral moments of cascade `c`'s band: the slope variance
    // `int k^2 S(w) dw` and its rate of change `int k^2 w^2 S(w) dw`.
    fn of(shape: SeaShape, cascades: &Cascades, c: usize) -> Self {
        const STEPS: usize = 256;
        let [lo, hi] = cascades.band(c);
        let lo = lo.max(TAU / cascades.lengths()[c]);
        let (w_lo, w_hi) = (sqrt(GRAVITY * lo), sqrt(GRAVITY * hi));
        let step = ln(w_hi / w_lo) / STEPS as f32;
        let (mut slope, mut rate) = (0.0f32, 0.0f32);
        for i in 0..STEPS {
            let w = w_lo * exp((i as f32 + 0.5) * step);
            let k = w * w / GRAVITY;
            let m = k * k * shape.density(w) * w * step;
            slope += m;
            rate += m * w * w;
        }
        if slope <= 0.0 {
            return Self {
                sigma: 0.0,
                crossings: 0.0,
            };
        }
        Self {
            sigma: OCEAN_CHOPPINESS * sqrt(slope),
            crossings: sqrt(rate / slope) / TAU,
        }
    }

    // The mean foam with the threshold `z` deviations below one: what the
    // ramp injects at any moment, plus the fading wake of each crossing.
    fn mean_foam(self, z: f32) -> f32 {
        const SAMPLES: usize = 16;
        let injected = (0..SAMPLES)
            .map(|i| below(z + FOAM_RAMP_SIGMAS * (i as f32 + 0.5) / SAMPLES as f32))
            .sum::<f32>()
            / SAMPLES as f32;
        let mid = z + 0.5 * FOAM_RAMP_SIGMAS;
        let wake = LINGER * FOAM_SECONDS * self.crossings * exp(-0.5 * mid * mid);
        (injected + wake).min(1.0)
    }

    // The threshold depth, in deviations, whose mean foam is `target`.
    fn depth_for(self, target: f32) -> f32 {
        let (mut lo, mut hi) = (-4.0f32, 12.0f32);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if self.mean_foam(mid) > target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }
}

// The chance a standard normal value falls below `-z`.
fn below(z: f32) -> f32 {
    0.5 * libm::erfcf(z * core::f32::consts::FRAC_1_SQRT_2)
}

#[cfg(test)]
mod tests {
    use super::super::fields::evolve;
    use super::super::initial::{SpectrumTexel, frequency_index, initial_spectrum};
    use super::super::{OceanQuality, OceanSea};
    use super::*;
    use crate::math::SaturatingFrom;
    use crate::render::water::ocean::clock::time_fraction;
    use alloc::vec;
    use alloc::vec::Vec;

    fn sea(wind_speed: f32) -> OceanSea {
        OceanSea {
            wind_speed,
            direction: [0.8, 0.6],
            fetch: 1.0e5,
        }
    }

    #[test]
    fn coverage_follows_the_wind() {
        assert_eq!(whitecap_coverage(0.0), 0.0);
        assert!(whitecap_coverage(4.0) < 1.0e-3);
        let at_ten = whitecap_coverage(10.0);
        assert!((0.008..0.012).contains(&at_ten), "{at_ten}");
        assert!(whitecap_coverage(14.0) > 0.025);
        assert_eq!(whitecap_coverage(200.0), 1.0);
    }

    #[test]
    fn the_clock_restarts_on_the_first_frame_and_backward_time() {
        let mut clock = FoamClock::default();
        assert_eq!(clock.advance(10.0), 0.0);
        let keep = clock.advance(10.0 + 1.0 / 60.0);
        assert!((keep - foam_keep(1.0 / 60.0)).abs() < 1e-6);
        assert_eq!(clock.advance(10.0 + 1.0 / 60.0), 1.0);
        assert_eq!(clock.advance(3.0), 0.0);
        assert!((foam_keep(FOAM_SECONDS) - libm::expf(-1.0)).abs() < 1e-6);
    }

    // Stronger wind foams more: the threshold climbs toward one. Only the
    // longest cascades foam.
    #[test]
    fn thresholds_rise_with_the_wind() {
        let at = |u: f32| {
            let s = sea(u);
            let c = Cascades::new(s.shape().unwrap().peak_wavelength(), OceanQuality::FULL);
            FoamCalibration::new(&s, &c)
        };
        let (light, fresh, gale) = (at(4.0), at(10.0), at(14.0));
        for c in 0..FOAM_CASCADES {
            assert!(light.thresholds[c] < fresh.thresholds[c], "{c}");
            assert!(fresh.thresholds[c] < gale.thresholds[c], "{c}");
            assert!(fresh.thresholds[c] < 1.0 && fresh.inv_ramps[c] > 0.0, "{c}");
        }
        assert_eq!(gale.thresholds[2], 0.0);
        assert_eq!(gale.inv_ramps[2], 0.0);
    }

    // The columns kernel's per-texel foam update in `water_fft.hlsl`, from
    // the first-order Jacobian.
    fn foam_step(previous: f32, jacobian: f32, threshold: f32, inv_ramp: f32, keep: f32) -> f32 {
        let injected = ((threshold - jacobian) * inv_ramp).clamp(0.0, 1.0);
        (previous * keep).max(injected)
    }

    // An inverse DFT (`sum f e^(+i k x)`, unnormalized), as the shader's
    // Stockham transform computes it.
    fn inverse_fft(a: &mut [[f32; 2]]) {
        let n = a.len();
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                a.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let angle = TAU / len as f32;
            for start in (0..n).step_by(len) {
                for k in 0..len / 2 {
                    let (s, c) = libm::sincosf(angle * k as f32);
                    let [ar, ai] = a[start + k];
                    let [br, bi] = a[start + k + len / 2];
                    let b = [br * c - bi * s, br * s + bi * c];
                    a[start + k] = [ar + b[0], ai + b[1]];
                    a[start + k + len / 2] = [ar - b[0], ai - b[1]];
                }
            }
            len <<= 1;
        }
    }

    // The rows then the columns of `size`-square `data`.
    fn inverse_fft_2d(data: &mut [[f32; 2]], size: usize) {
        for row in data.chunks_mut(size) {
            inverse_fft(row);
        }
        let mut column = vec![[0.0f32; 2]; size];
        for x in 0..size {
            for (y, v) in column.iter_mut().enumerate() {
                *v = data[y * size + x];
            }
            inverse_fft(&mut column);
            for (y, v) in column.iter().enumerate() {
                data[y * size + x] = *v;
            }
        }
    }

    // One cascade's first-order Jacobian over its whole tile at `time`.
    fn jacobian_field(spectrum: &[SpectrumTexel], length: f32, size: u32, time: f64) -> Vec<f32> {
        let n = size as usize;
        let mask = size - 1;
        let fraction = time_fraction(time);
        let dk = TAU / length;
        let mut fields = [vec![[0.0f32; 2]; n * n], vec![[0.0f32; 2]; n * n]];
        for y in 0..size {
            for x in 0..size {
                let i = (y * size + x) as usize;
                let m = (((size - y) & mask) * size + ((size - x) & mask)) as usize;
                let k = [
                    frequency_index(x, size) as f32 * dk,
                    frequency_index(y, size) as f32 * dk,
                ];
                let f = evolve(spectrum[i], spectrum[m], fraction, k);
                fields[0][i] = f.jxx;
                fields[1][i] = f.jzz;
            }
        }
        for f in &mut fields {
            inverse_fft_2d(f, n);
        }
        (0..n * n)
            .map(|i| 1.0 + OCEAN_CHOPPINESS * (fields[0][i][0] + fields[1][i][0]))
            .collect()
    }

    // Steps every cascade's foam through `seconds` at 60 Hz as the shader
    // does, and returns the sea's mean foam over the second half, the
    // cascades layered as the water shader layers them.
    fn simulated_coverage(wind_speed: f32, seconds: f32) -> f32 {
        let s = sea(wind_speed);
        let quality = OceanQuality {
            size: 128,
            cascades: 2,
        };
        let cascades = Cascades::new(s.shape().unwrap().peak_wavelength(), quality);
        let calibration = FoamCalibration::new(&s, &cascades);
        let n = (quality.size * quality.size) as usize;
        let spectra: Vec<_> = (0..cascades.count())
            .map(|c| initial_spectrum(&s, &cascades, c))
            .collect();
        let mut foam = vec![vec![0.0f32; n]; cascades.count()];
        let mut clock = FoamClock::default();
        let frames = usize::saturating_from(seconds * 60.0);
        let (mut sum, mut counted) = (0.0f64, 0usize);
        for frame in 0..frames {
            let time = 1000.0 + frame as f64 / 60.0;
            let keep = clock.advance(time);
            for (c, field) in foam.iter_mut().enumerate() {
                let j = jacobian_field(&spectra[c], cascades.lengths()[c], quality.size, time);
                let (t, r) = (calibration.thresholds[c], calibration.inv_ramps[c]);
                for (f, j) in field.iter_mut().zip(j) {
                    *f = foam_step(*f, j, t, r, keep);
                }
            }
            if frame >= frames / 2 {
                let clear: f64 = (0..n)
                    .map(|i| foam.iter().map(|f| 1.0 - f64::from(f[i])).product::<f64>())
                    .sum::<f64>()
                    / n as f64;
                sum += 1.0 - clear;
                counted += 1;
            }
        }
        narrow_f32(sum / counted as f64)
    }

    // The foam the shader's update builds from the cascades' own evolving
    // Jacobians covers what the calibration aims at, and a light breeze
    // raises next to none.
    #[test]
    fn the_simulated_foam_covers_the_target() {
        for wind in [10.0, 14.0] {
            let got = simulated_coverage(wind, 10.0);
            let want = whitecap_coverage(wind);
            let ratio = got / want;
            assert!((0.8..1.25).contains(&ratio), "{wind} m/s: {got} vs {want}");
        }
        assert!(simulated_coverage(4.0, 6.0) < 1.0e-3);
    }
}
