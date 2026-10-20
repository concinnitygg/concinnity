// The per-frame exponential-moving-average update: it takes the GPU-measured
// mean and highlight log-luminance and produces the next frame's exposure
// multiplier. The histogram itself is built in the backend's compute shader;
// only the reduction and the EMA live here, so both can be unit-tested
// without a GPU.

use super::{AutoExposureSettings, night_vision_weight};
use crate::math::{exp, exp2};
use crate::render::uniforms::ExposureMeasurement;

/// Lowest log2(luminance) the histogram bins span. Pixels darker than this fall
/// in bin 0. Roughly matches a moonlit interior at the dim end.
pub const LUM_LOG2_MIN: f32 = -10.0;

/// Highest log2(luminance) the histogram bins span. Pixels brighter than this
/// fall in the last bin. Roughly matches a direct sun reflection at the bright
/// end. The shader uses `(LUM_LOG2_MAX - LUM_LOG2_MIN)` to convert a bin index
/// back to a log-luminance value during the average pass.
pub const LUM_LOG2_MAX: f32 = 12.0;

/// Number of histogram bins the build kernel writes into and the average kernel
/// reduces. 256 is small enough to fit in threadgroup memory on every Apple GPU
/// (1 KiB at u32) and big enough that the per-bin log-luminance step is fine.
pub const HISTOGRAM_BINS: usize = 256;

/// Running auto-exposure state. The current adapted EV moves toward the target
/// EV (derived from the GPU-measured average log-luminance) via an exponential
/// moving average so a sudden brightness change ramps in over a fraction of a
/// second rather than snapping. One instance lives on each backend that runs
/// auto-exposure.
#[derive(Debug, Clone, Copy)]
pub struct AutoExposureState {
    /// EV currently applied to the scene. Updated each frame by
    /// [`AutoExposureState::update`]; the backend reads it back out to set the
    /// exposure multiplier the post passes push to the GPU.
    pub current_ev: f32,
    /// The scene's mean log2 luminance the eye is adapted to: the measured
    /// mean, eased in at the same rate as `current_ev` and free of its clamp
    /// and biases.
    pub adapted_log_lum: f32,
}

impl AutoExposureState {
    /// Initial state. The current EV is the midpoint of the settings' clamp
    /// range: a neutral starting point before the first GPU measurement
    /// arrives. The first `update` call snaps it toward the real scene EV.
    pub fn new(settings: &AutoExposureSettings) -> Self {
        let current_ev = (settings.min_ev + settings.max_ev) * 0.5;
        Self {
            current_ev,
            adapted_log_lum: settings.target_log_lum - current_ev,
        }
    }

    /// Step the EMA one frame: take the GPU measurement, turn its mean
    /// log2 luminance into a target EV (the offset that maps the scene mean
    /// onto `settings.target_log_lum` plus the authored `ev_bias` and, for a
    /// dark scene, the settings' night bias), hold that under the EV that
    /// keeps the measured highlight within the settings' headroom, then move
    /// `current_ev` toward it by the clamped EMA rate. Returns the new
    /// clamped EV. `dt` is the frame time in seconds; non-finite or
    /// non-positive values short-circuit (the EV stays where it was, no NaN
    /// propagation into the post pass). A non-finite highlight caps nothing.
    pub fn update(
        &mut self,
        measured: ExposureMeasurement,
        ev_bias: f32,
        settings: &AutoExposureSettings,
        dt: f32,
    ) -> f32 {
        let avg_log_lum = measured.mean_log_lum;
        let cap = if measured.highlight_log_lum.is_finite() {
            settings.highlight_cap(measured.highlight_log_lum)
        } else {
            f32::INFINITY
        };
        // The target EV shifts the scene's geometric-mean luminance onto the
        // configured pivot: scene-white on the SDR path (target_log_lum=0,
        // ACES then compresses), perceptual middle-gray on the HDR path
        // (target_log_lum=log2(0.18), no ACES). `exposure = 2^target_ev`
        // then satisfies `avg_lum * exposure = 2^target_log_lum` modulo bias.
        let target =
            (settings.target_log_lum - avg_log_lum + ev_bias + settings.night_bias(avg_log_lum))
                .min(cap)
                .clamp(settings.min_ev, settings.max_ev);
        if !dt.is_finite() || dt <= 0.0 || !target.is_finite() {
            self.current_ev = self.current_ev.clamp(settings.min_ev, settings.max_ev);
            return self.current_ev;
        }
        let blend = 1.0 - exp(-settings.speed * dt);
        self.adapted_log_lum += (avg_log_lum - self.adapted_log_lum) * blend;
        self.current_ev = (self.current_ev + (target - self.current_ev) * blend)
            .clamp(settings.min_ev, settings.max_ev);
        self.current_ev
    }
}

/// A renderer's auto-exposure on the CPU: the tunables, the authored bias, and
/// the adapted EV that each frame's GPU measurement moves.
#[derive(Debug, Clone, Copy)]
pub struct ExposureAdaptation {
    /// The clamped tunables. A live quality change replaces them in place.
    pub settings: AutoExposureSettings,
    /// Authored EV bias, in stops, added to the adapted target.
    pub bias_ev: f32,
    state: AutoExposureState,
}

impl ExposureAdaptation {
    /// Adaptation starting from the midpoint of the settings' clamp range.
    pub fn new(settings: AutoExposureSettings, bias_ev: f32) -> Self {
        Self {
            settings,
            bias_ev,
            state: AutoExposureState::new(&settings),
        }
    }

    /// The EV currently applied.
    pub fn current_ev(&self) -> f32 {
        self.state.current_ev
    }

    /// The scene's mean log2 luminance the eye is currently adapted to.
    #[cfg(test)]
    fn adapted_log_lum(&self) -> f32 {
        self.state.adapted_log_lum
    }

    /// How far, in `[0, 1]`, the composite turns the image to night vision
    /// for a world whose sun has intensity `sunlight`: the settings' strength
    /// times the rods' share at the adapted luminance.
    pub fn night_vision(&self, sunlight: f32) -> f32 {
        night_vision_weight(
            self.settings.night_vision,
            self.state.adapted_log_lum,
            sunlight,
        )
    }

    /// Step one frame of `dt` seconds from the GPU's measurement, returning
    /// the linear exposure multiplier the post passes apply. A non-finite
    /// mean reads as the histogram's floor.
    pub fn step(&mut self, mut measured: ExposureMeasurement, dt: f32) -> f32 {
        if !measured.mean_log_lum.is_finite() {
            measured.mean_log_lum = LUM_LOG2_MIN;
        }
        let ev = self
            .state
            .update(measured, self.bias_ev, &self.settings, dt);
        exp2(ev)
    }
}

// Convert a histogram (256 bin counts) into the weighted-average log-luminance
// the EMA consumes. Mirrors what the average-pass compute kernel does on GPU,
// kept in pure Rust so the math is unit-testable without a device. Bins are
// weighted by their center log-luminance: bin `i` covers
// `[LUM_LOG2_MIN + i*step, LUM_LOG2_MIN + (i+1)*step)`. Bin 0 is treated as
// "below sensor floor", and so is every bin whose center sits more than
// `METERED_RANGE_EV` under the frame's highlight; a frame with nothing left
// reads as the floor, so a mostly-black frame still produces a finite EV.
#[cfg(test)]
pub(crate) fn average_log_luminance(histogram: &[u32; HISTOGRAM_BINS]) -> f32 {
    let step = (LUM_LOG2_MAX - LUM_LOG2_MIN) / HISTOGRAM_BINS as f32;
    let metered_from = highlight_log_luminance(histogram) - super::METERED_RANGE_EV;
    let mut weighted_sum = 0.0f64;
    let mut count = 0u64;
    for (i, &n) in histogram.iter().enumerate().skip(1) {
        let center = LUM_LOG2_MIN + (i as f32 + 0.5) * step;
        if n == 0 || center < metered_from {
            continue;
        }
        weighted_sum += f64::from(center) * f64::from(n);
        count += u64::from(n);
    }
    if count == 0 {
        LUM_LOG2_MIN
    } else {
        crate::math::narrow_f32(weighted_sum / count as f64)
    }
}

// The log-luminance the brightest `HIGHLIGHT_FRACTION` of all pixels starts
// at, bin 0 counted in. Mirrors the average kernel the way
// `average_log_luminance` does: the crossing bin is split as if its pixels
// spread evenly, and a crossing in bin 0 (or an empty histogram) reads as
// the floor.
#[cfg(test)]
pub(crate) fn highlight_log_luminance(histogram: &[u32; HISTOGRAM_BINS]) -> f32 {
    let step = (LUM_LOG2_MAX - LUM_LOG2_MIN) / HISTOGRAM_BINS as f32;
    let total: u32 = histogram.iter().sum();
    if total == 0 {
        return LUM_LOG2_MIN;
    }
    let wanted = super::HIGHLIGHT_FRACTION * total as f32;
    let mut above = 0u32;
    for (i, &n) in histogram.iter().enumerate().rev() {
        let at = above + n;
        if at as f32 >= wanted && (above as f32) < wanted {
            if i == 0 {
                break;
            }
            let inside = (wanted - above as f32) / n as f32;
            return LUM_LOG2_MIN + (i as f32 + 1.0 - inside) * step;
        }
        above = at;
    }
    LUM_LOG2_MIN
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::auto_exposure::{HDR_MIDDLE_GRAY_LOG2, HIGHLIGHT_FRACTION, METERED_RANGE_EV};
    use crate::math::SaturatingFrom;

    // A frame of one luminance throughout: its highlights are its mean.
    fn even(log_lum: f32) -> ExposureMeasurement {
        ExposureMeasurement {
            mean_log_lum: log_lum,
            highlight_log_lum: log_lum,
        }
    }

    #[test]
    fn histogram_constants_match_the_shader() {
        let src = crate::render::shader::embedded::AUTO_EXPOSURE;
        assert_eq!(
            crate::render::shader::consts::uint(src, "HISTOGRAM_BINS"),
            HISTOGRAM_BINS
        );
        assert_eq!(
            crate::render::shader::consts::float(src, "HIGHLIGHT_FRACTION"),
            HIGHLIGHT_FRACTION
        );
        assert_eq!(
            crate::render::shader::consts::float(src, "METERED_RANGE_EV"),
            METERED_RANGE_EV
        );
    }

    #[test]
    fn update_pulls_current_ev_toward_target() {
        let settings = AutoExposureSettings::resolve(-8.0, 8.0, 4.0, false);
        let mut state = AutoExposureState {
            current_ev: 0.0,
            adapted_log_lum: 0.0,
        };
        // avg_log_lum = 2.0 -> target_ev = -2.0 (dim the scene by two stops).
        let ev = state.update(even(2.0), 0.0, &settings, 1.0);
        assert!(ev < 0.0, "ev should move toward target_ev = -2.0, got {ev}");
        assert!(ev > -2.0, "ev should not overshoot in one step, got {ev}");

        // Many steps converge.
        for _ in 0..200 {
            state.update(even(2.0), 0.0, &settings, 1.0 / 60.0);
        }
        assert!((state.current_ev + 2.0).abs() < 1.0e-3);
    }

    #[test]
    fn update_clamps_to_settings_bounds() {
        let settings = AutoExposureSettings::resolve(-1.0, 1.0, 5.0, false);
        let mut state = AutoExposureState {
            current_ev: 0.0,
            adapted_log_lum: 0.0,
        };
        // avg_log_lum = -10 would ask for target_ev = +10, clamped to +1.
        for _ in 0..100 {
            state.update(even(-10.0), 0.0, &settings, 1.0 / 60.0);
        }
        assert!((state.current_ev - 1.0).abs() < 1.0e-3);
    }

    #[test]
    fn update_short_circuits_non_finite_dt() {
        let settings = AutoExposureSettings::resolve(-2.0, 2.0, 1.0, false);
        let mut state = AutoExposureState {
            current_ev: 0.5,
            adapted_log_lum: 0.0,
        };
        let ev = state.update(even(5.0), 0.0, &settings, f32::NAN);
        assert_eq!(ev, 0.5);
        let ev = state.update(even(5.0), 0.0, &settings, -1.0);
        assert_eq!(ev, 0.5);
    }

    #[test]
    fn update_honors_ev_bias() {
        let settings = AutoExposureSettings::resolve(-8.0, 8.0, 10.0, false);
        let mut state = AutoExposureState {
            current_ev: 0.0,
            adapted_log_lum: 0.0,
        };
        // avg_log_lum = 0, bias = +1 -> target_ev = +1 (over-expose by one stop).
        for _ in 0..200 {
            state.update(even(0.0), 1.0, &settings, 1.0 / 60.0);
        }
        assert!((state.current_ev - 1.0).abs() < 1.0e-3);
    }

    // A moonlit scene settles the night bias darker than the plain target; a
    // day scene the same settings meet settles exactly on it.
    #[test]
    fn update_adds_the_night_bias_to_a_dark_scene() {
        let settings = AutoExposureSettings::resolve(-8.0, 24.0, 10.0, false).with_night_bias(-3.0);
        let settle = |avg: f32| {
            let mut state = AutoExposureState {
                current_ev: 0.0,
                adapted_log_lum: 0.0,
            };
            for _ in 0..400 {
                state.update(even(avg), -4.0, &settings, 1.0 / 60.0);
            }
            state.current_ev
        };
        assert!((settle(-23.6) - (23.6 - 4.0 - 3.0)).abs() < 1.0e-3);
        assert!((settle(-4.4) - (4.4 - 4.0)).abs() < 1.0e-3);
    }

    #[test]
    fn resolve_defaults_to_scene_white_target_on_sdr() {
        // SDR worlds keep the legacy "average → 1.0 linear" pivot so existing
        // exposure authoring stays unchanged. ACES then squishes 1.0 down to
        // the display mid-tone band.
        let s = AutoExposureSettings::resolve(-8.0, 8.0, 1.5, false);
        assert_eq!(s.target_log_lum, 0.0);
    }

    #[test]
    fn resolve_shifts_to_middle_gray_on_hdr() {
        // HDR worlds shift AE's pivot to perceptual middle-gray (0.18 linear)
        // because there is no ACES tonemap to compress scene-white down. The
        // pivot is `log2(0.18) ≈ -2.473`.
        let s = AutoExposureSettings::resolve(-8.0, 8.0, 1.5, true);
        assert!((s.target_log_lum - HDR_MIDDLE_GRAY_LOG2).abs() < 1.0e-6);
    }

    #[test]
    fn update_shifts_target_by_target_log_lum_on_hdr() {
        // With HDR-aware settings, the EV the EMA converges on is shifted
        // ~2.47 stops DOWN compared to the SDR default, i.e. the scene gets
        // darker post-exposure so the same input renders at middle-gray
        // instead of scene-white.
        let sdr = AutoExposureSettings::resolve(-8.0, 8.0, 10.0, false);
        let hdr = AutoExposureSettings::resolve(-8.0, 8.0, 10.0, true);
        let mut state_sdr = AutoExposureState {
            current_ev: 0.0,
            adapted_log_lum: 0.0,
        };
        let mut state_hdr = AutoExposureState {
            current_ev: 0.0,
            adapted_log_lum: 0.0,
        };
        for _ in 0..500 {
            state_sdr.update(even(0.0), 0.0, &sdr, 1.0 / 60.0);
            state_hdr.update(even(0.0), 0.0, &hdr, 1.0 / 60.0);
        }
        // SDR settles at 0.0; HDR at log2(0.18) ≈ -2.47.
        assert!(state_sdr.current_ev.abs() < 1.0e-3);
        assert!((state_hdr.current_ev - HDR_MIDDLE_GRAY_LOG2).abs() < 1.0e-3);
    }

    #[test]
    fn adaptation_starts_mid_range_and_returns_linear_exposure() {
        let settings = AutoExposureSettings::resolve(-4.0, 2.0, 10.0, false);
        let mut adaptation = ExposureAdaptation::new(settings, 0.0);
        assert_eq!(adaptation.current_ev(), -1.0);
        let mut exposure = 0.0;
        for _ in 0..500 {
            exposure = adaptation.step(even(1.0), 1.0 / 60.0);
        }
        assert!((adaptation.current_ev() + 1.0).abs() < 1.0e-3);
        assert!((exposure - 0.5).abs() < 1.0e-3, "2^-1, got {exposure}");
    }

    #[test]
    fn adaptation_reads_a_non_finite_measurement_as_the_floor() {
        let settings = AutoExposureSettings::resolve(-20.0, 20.0, 1000.0, false);
        let mut nan = ExposureAdaptation::new(settings, 0.0);
        let mut floor = nan;
        for measured in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let got = nan.step(even(measured), 1.0);
            let want = floor.step(even(LUM_LOG2_MIN), 1.0);
            assert_eq!(got, want);
            assert!(got.is_finite());
        }
    }

    #[test]
    fn adaptation_carries_the_bias_and_live_settings() {
        let settings = AutoExposureSettings::resolve(-8.0, 8.0, 10.0, false);
        let mut adaptation = ExposureAdaptation::new(settings, 1.0);
        for _ in 0..500 {
            adaptation.step(even(0.0), 1.0 / 60.0);
        }
        assert!((adaptation.current_ev() - 1.0).abs() < 1.0e-3);
        adaptation.settings = AutoExposureSettings::resolve(-8.0, 0.5, 10.0, false);
        adaptation.step(even(0.0), 1.0 / 60.0);
        assert!(adaptation.current_ev() <= 0.5);
    }

    // The adapted luminance eases toward the measurement at the exposure's
    // rate, past the EV clamp, and night vision follows it: none by day, all
    // of it once a moonlit night has settled, never a jump in between.
    #[test]
    fn night_vision_follows_the_adapted_luminance() {
        let settings = AutoExposureSettings::resolve(-8.0, 12.0, 2.0, false).with_night_vision(1.0);
        let mut adaptation = ExposureAdaptation::new(settings, -4.0);
        for _ in 0..600 {
            adaptation.step(even(-4.4), 1.0 / 60.0);
        }
        assert!((adaptation.adapted_log_lum() + 4.4).abs() < 1e-3);
        assert_eq!(adaptation.night_vision(3.2), 0.0);
        let mut last = 0.0;
        for _ in 0..1200 {
            adaptation.step(even(-23.6), 1.0 / 60.0);
            let w = adaptation.night_vision(3.2);
            assert!(w >= last && w - last < 0.1, "{w} after {last}");
            last = w;
        }
        assert!((adaptation.adapted_log_lum() + 23.6).abs() < 1e-3);
        assert!(adaptation.current_ev() <= 12.0);
        assert_eq!(last, 1.0);
        let off = ExposureAdaptation {
            settings: settings.with_night_vision(0.0),
            ..adaptation
        };
        assert_eq!(off.night_vision(3.2), 0.0);
    }

    // A dark frame with a small bright subject settles where the subject sits
    // at the headroom over white, not where the mean would put it; a frame
    // whose highlights already fit is untouched.
    #[test]
    fn update_holds_the_highlight_within_the_headroom() {
        let settings = AutoExposureSettings::resolve(-8.0, 24.0, 10.0, false);
        let settle = |measured: ExposureMeasurement, bias: f32| {
            let mut state = AutoExposureState {
                current_ev: 0.0,
                adapted_log_lum: 0.0,
            };
            for _ in 0..400 {
                state.update(measured, bias, &settings, 1.0 / 60.0);
            }
            state.current_ev
        };
        let orbit = ExposureMeasurement {
            mean_log_lum: -19.0,
            highlight_log_lum: 1.5,
        };
        assert!((settle(orbit, -4.0) - (2.0 - 1.5)).abs() < 1e-3);
        let day = ExposureMeasurement {
            mean_log_lum: -1.0,
            highlight_log_lum: 1.5,
        };
        assert!((settle(day, -4.0) - (1.0 - 4.0)).abs() < 1e-3);
        // The EV clamp still has the last word, and an unmeasured highlight
        // caps nothing.
        let blinding = ExposureMeasurement {
            mean_log_lum: 0.0,
            highlight_log_lum: 40.0,
        };
        assert!((settle(blinding, 0.0) + 8.0).abs() < 1e-3);
        let unmeasured = ExposureMeasurement {
            highlight_log_lum: f32::NAN,
            ..orbit
        };
        assert!((settle(unmeasured, -4.0) - 15.0).abs() < 1e-3);
    }

    // A frame darker than the floor throughout cannot hold the EV under its
    // clamp: the floor sits at least two stops under what `max_ev` lifts.
    #[test]
    fn a_sub_floor_highlight_never_caps() {
        for max_ev in [8.0, 16.0, 24.0] {
            let settings = AutoExposureSettings::resolve(-8.0, max_ev, 1.0, false);
            assert!(settings.highlight_cap(settings.histogram_floor()) > settings.max_ev);
        }
    }

    #[test]
    fn highlight_log_luminance_finds_the_brightest_share() {
        let step = (LUM_LOG2_MAX - LUM_LOG2_MIN) / HISTOGRAM_BINS as f32;
        let edge = |bin: usize| LUM_LOG2_MIN + bin as f32 * step;
        assert_eq!(highlight_log_luminance(&[0; HISTOGRAM_BINS]), LUM_LOG2_MIN);

        // Mostly black, some dim, four times the share bright: the share is
        // the top quarter of the bright bin, the black pixels counted toward
        // the total.
        const TOTAL: u32 = 1_000_000;
        let share = u32::saturating_from((HIGHLIGHT_FRACTION * TOTAL as f32).round());
        let mut histogram = [0u32; HISTOGRAM_BINS];
        histogram[0] = 850_000;
        histogram[220] = 4 * share;
        histogram[40] = TOTAL - 850_000 - 4 * share;
        let got = highlight_log_luminance(&histogram);
        assert!((got - (edge(220) + 0.75 * step)).abs() < 1e-4, "{got}");

        // Twice the share: the top half of that bin.
        histogram[220] = 2 * share;
        histogram[40] = TOTAL - 850_000 - 2 * share;
        let got = highlight_log_luminance(&histogram);
        assert!((got - (edge(220) + 0.5 * step)).abs() < 1e-4, "{got}");

        // A bright subject under the share leaves the dim pixels as the
        // highlight.
        histogram[220] = share / 2;
        histogram[40] = TOTAL - 850_000 - histogram[220];
        let got = highlight_log_luminance(&histogram);
        assert!(got > edge(40) && got < edge(41), "{got}");

        // A frame almost all black reads as the floor.
        let mut black = [0u32; HISTOGRAM_BINS];
        black[0] = 100_000_000;
        black[200] = 1;
        assert_eq!(highlight_log_luminance(&black), LUM_LOG2_MIN);
    }

    // A lit planet over black space: the space pixels sit far under the
    // planet's highlight and drop out, so the mean is the planet's own. A
    // frame whose shadows sit within the range keeps them.
    #[test]
    fn average_log_luminance_leaves_out_what_sits_far_under_the_highlight() {
        let step = (LUM_LOG2_MAX - LUM_LOG2_MIN) / HISTOGRAM_BINS as f32;
        let center = |bin: usize| LUM_LOG2_MIN + (bin as f32 + 0.5) * step;
        let bin_at = |log_lum: f32| usize::saturating_from((log_lum - LUM_LOG2_MIN) / step);
        let lit = bin_at(5.0);
        let space = bin_at(5.0 - METERED_RANGE_EV - 1.0);
        let mut histogram = [0u32; HISTOGRAM_BINS];
        histogram[lit] = 200_000;
        histogram[space] = 800_000;
        let avg = average_log_luminance(&histogram);
        assert!((avg - center(lit)).abs() < 1.0e-3, "{avg}");

        let shadow = bin_at(5.0 - METERED_RANGE_EV + 1.0);
        histogram[space] = 0;
        histogram[shadow] = 200_000;
        let avg = average_log_luminance(&histogram);
        let want = 0.5 * (center(lit) + center(shadow));
        assert!((avg - want).abs() < 1.0e-3, "{avg}");
    }

    #[test]
    fn average_log_luminance_empties_to_floor() {
        let histogram = [0u32; HISTOGRAM_BINS];
        assert_eq!(average_log_luminance(&histogram), LUM_LOG2_MIN);
    }

    #[test]
    fn average_log_luminance_single_bin_returns_bin_center() {
        let mut histogram = [0u32; HISTOGRAM_BINS];
        histogram[128] = 10;
        let avg = average_log_luminance(&histogram);
        let step = (LUM_LOG2_MAX - LUM_LOG2_MIN) / HISTOGRAM_BINS as f32;
        let expected = LUM_LOG2_MIN + (128.5) * step;
        assert!(
            (avg - expected).abs() < 1.0e-3,
            "avg={avg} expected={expected}"
        );
    }

    #[test]
    fn average_log_luminance_ignores_underflow_bin() {
        // Pixels in bin 0 are below the sensor floor; they should not pull the
        // average down. The result should equal the center of bin 100.
        let mut histogram = [0u32; HISTOGRAM_BINS];
        histogram[0] = 10_000;
        histogram[100] = 1;
        let avg = average_log_luminance(&histogram);
        let step = (LUM_LOG2_MAX - LUM_LOG2_MIN) / HISTOGRAM_BINS as f32;
        let expected = LUM_LOG2_MIN + (100.5) * step;
        assert!(
            (avg - expected).abs() < 1.0e-3,
            "avg={avg} expected={expected}"
        );
    }
}
