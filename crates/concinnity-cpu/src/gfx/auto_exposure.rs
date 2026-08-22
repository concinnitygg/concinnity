//! Auto-exposure (EV adaptation) state. The per-frame exponential-moving-average
//! update takes a GPU-measured average log-luminance and produces the next
//! frame's exposure multiplier. The histogram itself is built in the backend's
//! compute shader; this module owns only the reduction and the EMA so both can
//! be unit-tested without a GPU. The resolved settings the EMA reads live in
//! concinnity-core, re-exported here under their historical path.

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

pub use concinnity_core::gfx::auto_exposure::{AutoExposureSettings, HDR_MIDDLE_GREY_LOG2};

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
}

impl AutoExposureState {
    /// Initial state. The current EV is the midpoint of the settings' clamp
    /// range: a neutral starting point before the first GPU measurement
    /// arrives. The first `update` call snaps it toward the real scene EV.
    pub fn new(settings: &AutoExposureSettings) -> Self {
        Self {
            current_ev: (settings.min_ev + settings.max_ev) * 0.5,
        }
    }

    /// Step the EMA one frame: take the GPU-measured average log-luminance
    /// (base-2), turn it into a target EV (the offset that maps the scene
    /// mean onto `settings.target_log_lum` plus the authored `ev_bias`),
    /// then move `current_ev` toward it by the clamped EMA rate. Returns
    /// the new clamped EV. `dt` is the frame time in seconds; non-finite or
    /// non-positive values short-circuit (the EV stays where it was, no
    /// NaN propagation into the post pass).
    pub fn update(
        &mut self,
        avg_log_lum: f32,
        ev_bias: f32,
        settings: &AutoExposureSettings,
        dt: f32,
    ) -> f32 {
        // The target EV shifts the scene's geometric-mean luminance onto the
        // configured pivot: scene-white on the SDR path (target_log_lum=0,
        // ACES then compresses), perceptual middle-grey on the HDR path
        // (target_log_lum=log2(0.18), no ACES). `exposure = 2^target_ev`
        // then satisfies `avg_lum * exposure = 2^target_log_lum` modulo bias.
        let target = (settings.target_log_lum - avg_log_lum + ev_bias)
            .clamp(settings.min_ev, settings.max_ev);
        if !dt.is_finite() || dt <= 0.0 || !target.is_finite() {
            self.current_ev = self.current_ev.clamp(settings.min_ev, settings.max_ev);
            return self.current_ev;
        }
        let blend = 1.0 - (-settings.speed * dt).exp();
        self.current_ev = (self.current_ev + (target - self.current_ev) * blend)
            .clamp(settings.min_ev, settings.max_ev);
        self.current_ev
    }
}

// Convert a histogram (256 bin counts) into the weighted-average log-luminance
// the EMA consumes. Mirrors what the average-pass compute kernel does on GPU,
// kept in pure Rust so the math is unit-testable without a device. Bins are
// weighted by their centre log-luminance: bin `i` covers
// `[LUM_LOG2_MIN + i*step, LUM_LOG2_MIN + (i+1)*step)`. Bin 0 is treated as
// "below sensor floor" and weighted-in only when every other bin is empty,
// so a mostly-black frame still produces a finite EV.
#[cfg(test)]
pub(crate) fn average_log_luminance(histogram: &[u32; HISTOGRAM_BINS]) -> f32 {
    let step = (LUM_LOG2_MAX - LUM_LOG2_MIN) / HISTOGRAM_BINS as f32;
    let mut weighted_sum = 0.0f64;
    let mut count = 0u64;
    for (i, &n) in histogram.iter().enumerate().skip(1) {
        if n == 0 {
            continue;
        }
        let centre = LUM_LOG2_MIN + (i as f32 + 0.5) * step;
        weighted_sum += centre as f64 * n as f64;
        count += n as u64;
    }
    if count == 0 {
        LUM_LOG2_MIN
    } else {
        (weighted_sum / count as f64) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_pulls_current_ev_toward_target() {
        let settings = AutoExposureSettings::resolve(-8.0, 8.0, 4.0, false);
        let mut state = AutoExposureState { current_ev: 0.0 };
        // avg_log_lum = 2.0 -> target_ev = -2.0 (dim the scene by two stops).
        let ev = state.update(2.0, 0.0, &settings, 1.0);
        assert!(ev < 0.0, "ev should move toward target_ev = -2.0, got {ev}");
        assert!(ev > -2.0, "ev should not overshoot in one step, got {ev}");

        // Many steps converge.
        for _ in 0..200 {
            state.update(2.0, 0.0, &settings, 1.0 / 60.0);
        }
        assert!((state.current_ev + 2.0).abs() < 1.0e-3);
    }

    #[test]
    fn update_clamps_to_settings_bounds() {
        let settings = AutoExposureSettings::resolve(-1.0, 1.0, 5.0, false);
        let mut state = AutoExposureState { current_ev: 0.0 };
        // avg_log_lum = -10 would ask for target_ev = +10, clamped to +1.
        for _ in 0..100 {
            state.update(-10.0, 0.0, &settings, 1.0 / 60.0);
        }
        assert!((state.current_ev - 1.0).abs() < 1.0e-3);
    }

    #[test]
    fn update_short_circuits_non_finite_dt() {
        let settings = AutoExposureSettings::resolve(-2.0, 2.0, 1.0, false);
        let mut state = AutoExposureState { current_ev: 0.5 };
        let ev = state.update(5.0, 0.0, &settings, f32::NAN);
        assert_eq!(ev, 0.5);
        let ev = state.update(5.0, 0.0, &settings, -1.0);
        assert_eq!(ev, 0.5);
    }

    #[test]
    fn update_honours_ev_bias() {
        let settings = AutoExposureSettings::resolve(-8.0, 8.0, 10.0, false);
        let mut state = AutoExposureState { current_ev: 0.0 };
        // avg_log_lum = 0, bias = +1 -> target_ev = +1 (over-expose by one stop).
        for _ in 0..200 {
            state.update(0.0, 1.0, &settings, 1.0 / 60.0);
        }
        assert!((state.current_ev - 1.0).abs() < 1.0e-3);
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
    fn resolve_shifts_to_middle_grey_on_hdr() {
        // HDR worlds shift AE's pivot to perceptual middle-grey (0.18 linear)
        // because there is no ACES tonemap to compress scene-white down. The
        // pivot is `log2(0.18) ≈ -2.473`.
        let s = AutoExposureSettings::resolve(-8.0, 8.0, 1.5, true);
        assert!((s.target_log_lum - HDR_MIDDLE_GREY_LOG2).abs() < 1.0e-6);
    }

    #[test]
    fn update_shifts_target_by_target_log_lum_on_hdr() {
        // With HDR-aware settings, the EV the EMA converges on is shifted
        // ~2.47 stops DOWN compared to the SDR default, i.e. the scene gets
        // darker post-exposure so the same input renders at middle-grey
        // instead of scene-white.
        let sdr = AutoExposureSettings::resolve(-8.0, 8.0, 10.0, false);
        let hdr = AutoExposureSettings::resolve(-8.0, 8.0, 10.0, true);
        let mut state_sdr = AutoExposureState { current_ev: 0.0 };
        let mut state_hdr = AutoExposureState { current_ev: 0.0 };
        for _ in 0..500 {
            state_sdr.update(0.0, 0.0, &sdr, 1.0 / 60.0);
            state_hdr.update(0.0, 0.0, &hdr, 1.0 / 60.0);
        }
        // SDR settles at 0.0; HDR at log2(0.18) ≈ -2.47.
        assert!(state_sdr.current_ev.abs() < 1.0e-3);
        assert!((state_hdr.current_ev - HDR_MIDDLE_GREY_LOG2).abs() < 1.0e-3);
    }

    #[test]
    fn average_log_luminance_empties_to_floor() {
        let histogram = [0u32; HISTOGRAM_BINS];
        assert_eq!(average_log_luminance(&histogram), LUM_LOG2_MIN);
    }

    #[test]
    fn average_log_luminance_single_bin_returns_bin_centre() {
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
        // average down. The result should equal the centre of bin 100.
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
