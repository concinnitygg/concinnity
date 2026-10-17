//! Auto-exposure: the tunables the post-process config resolves into, the
//! luminance histogram the backends measure a frame with, and the running EMA
//! that turns that measurement into the next frame's exposure.

mod night_vision;
mod pre_exposure;
mod state;

pub use night_vision::{
    METER_CALIBRATION, PHOTOPIC_LUMINANCE, SCOTOPIC_LUMINANCE, SUNLIGHT_LUX, ev100,
    luminance_at_ev100, night_vision_weight, rod_share, scene_ev100,
};
pub use pre_exposure::PreExposure;
pub use state::{
    AutoExposureState, ExposureAdaptation, HISTOGRAM_BINS, LUM_LOG2_MAX, LUM_LOG2_MIN,
};

// Smallest legal EMA speed. A zero or negative speed would freeze adaptation
// at the initial EV, so the authored value is floored here.
const MIN_SPEED: f32 = 1.0e-3;

// Largest legal EMA speed. Anything higher snaps in under a single frame and
// is indistinguishable from "no adaptation" visually, just noisier.
const MAX_SPEED: f32 = 20.0;

/// Largest exposure, in stops either way, that the authored exposure and the
/// auto-exposure bounds clamp to: enough to adapt from daylight to a scene
/// lit by starlight, about 7 orders of magnitude darker.
pub const EV_LIMIT: f32 = 24.0;

/// `log2(0.18)`: perceptual middle-gray in linear light. AE shifts the
/// scene's geometric-mean luminance to this value on the HDR output path so
/// the average pixel reads as a comfortable mid-tone instead of "scene
/// white = SDR reference white = bright" (which only worked on the SDR path
/// because the ACES tonemap implicitly compressed scene-white back down).
pub const HDR_MIDDLE_GRAY_LOG2: f32 = -2.473;

/// Share of a frame's pixels, the black ones counted, that the highlight
/// measurement takes as the frame's brightest.
pub const HIGHLIGHT_FRACTION: f32 = 0.001;

/// How many stops under the highlight the measured mean still counts a
/// pixel: about what a scene lit by one source spans, so the black of space
/// around a lit planet is left out of the mean rather than metered as the
/// scene, while a frame's own shadows stay in.
pub const METERED_RANGE_EV: f32 = 10.0;

/// Default for how many stops over white the brightest
/// [`HIGHLIGHT_FRACTION`] of a frame may sit after exposure: under the ACES
/// tonemap 2 stops (4x white) shows near white without clipping hard.
pub const HIGHLIGHT_HEADROOM_EV: f32 = 2.0;

/// How many stops under a luminance of 1 a scene's mean sits where the night
/// bias starts to ease in: the end of civil dusk under a sun of a few units'
/// intensity.
pub const NIGHT_BIAS_START: f32 = 15.0;

/// How many stops under a luminance of 1 a scene's mean sits where the night
/// bias applies in full: a moonlit night under the same sun.
pub const NIGHT_BIAS_FULL: f32 = 22.0;

/// Clamped auto-exposure tunables resolved from the authored asset fields. Held
/// by the backend; the per-frame EMA in `AutoExposureState::update` reads them
/// to clamp the adapted EV and drive its adaptation rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoExposureSettings {
    /// Lower bound on the adapted EV. Caps how bright a dim scene can ramp.
    pub min_ev: f32,
    /// Upper bound on the adapted EV. Caps how dark a bright scene can ramp.
    pub max_ev: f32,
    /// EMA rate (per second). The exponential `1 - exp(-speed * dt)` step pulls
    /// the current EV toward the target each frame; higher = faster adaptation.
    pub speed: f32,
    /// Log2 of the linear value AE aims the scene's geometric-mean luminance
    /// at. `0.0` = scene-white (legacy SDR + ACES default, ACES then squishes
    /// scene-white back down to a comfortable display mid-tone).
    /// `HDR_MIDDLE_GRAY_LOG2` ≈ -2.47 = perceptual middle-gray, the correct
    /// target on the HDR path where there is no ACES compression. Resolved
    /// from `Display.hdr` at asset time.
    pub target_log_lum: f32,
    /// Stops added to the target of a dark scene, eased in by its measured
    /// mean luminance between [`NIGHT_BIAS_START`] and [`NIGHT_BIAS_FULL`]
    /// stops under 1. Negative keeps a fully adapted night darker than a day.
    pub night_bias_ev: f32,
    /// How far, in `[0, 1]`, the composite turns a dark scene to night
    /// vision: desaturated toward the rods' luminance with a blue cast, by
    /// the rods' share of sight at the adapted scene luminance.
    pub night_vision: f32,
    /// How many stops over white (a linear 1.0 after exposure) the brightest
    /// [`HIGHLIGHT_FRACTION`] of a frame may sit: the adapted EV is held down
    /// to keep it there, so a small bright subject in a dark frame is not
    /// blown out.
    pub highlight_headroom_ev: f32,
}

impl AutoExposureSettings {
    /// Clamp the authored fields into a safe range. `min_ev` is forced to stay
    /// at-or-below `max_ev` so the adapted EV's clamp interval is non-empty.
    /// `hdr_aware` shifts AE's middle-gray pivot down so the average pixel
    /// reads at perceptual middle-gray on the HDR output path; SDR worlds
    /// keep the legacy scene-white pivot to preserve existing exposure
    /// authoring.
    pub fn resolve(min_ev: f32, max_ev: f32, speed: f32, hdr_aware: bool) -> Self {
        let min = min_ev.clamp(-EV_LIMIT, EV_LIMIT);
        let max = max_ev.clamp(-EV_LIMIT, EV_LIMIT);
        let (lo, hi) = if min <= max { (min, max) } else { (max, min) };
        Self {
            min_ev: lo,
            max_ev: hi,
            speed: speed.clamp(MIN_SPEED, MAX_SPEED),
            target_log_lum: if hdr_aware { HDR_MIDDLE_GRAY_LOG2 } else { 0.0 },
            night_bias_ev: 0.0,
            night_vision: 0.0,
            highlight_headroom_ev: HIGHLIGHT_HEADROOM_EV,
        }
    }

    /// These settings with `highlight_headroom_ev` as the highlight headroom,
    /// clamped to `[0, EV_LIMIT]`.
    pub fn with_highlight_headroom(self, highlight_headroom_ev: f32) -> Self {
        Self {
            highlight_headroom_ev: if highlight_headroom_ev.is_nan() {
                HIGHLIGHT_HEADROOM_EV
            } else {
                highlight_headroom_ev.clamp(0.0, EV_LIMIT)
            },
            ..self
        }
    }

    /// The highest EV that keeps a frame whose brightest pixels start at
    /// `highlight_log_lum` within the highlight headroom.
    pub fn highlight_cap(&self, highlight_log_lum: f32) -> f32 {
        self.highlight_headroom_ev - highlight_log_lum
    }

    /// These settings with `night_bias_ev` as the dark-scene bias, clamped to
    /// the exposure range.
    pub fn with_night_bias(self, night_bias_ev: f32) -> Self {
        Self {
            night_bias_ev: night_bias_ev.clamp(-EV_LIMIT, EV_LIMIT),
            ..self
        }
    }

    /// These settings with `night_vision` as the night-vision strength,
    /// clamped to `[0, 1]`.
    pub fn with_night_vision(self, night_vision: f32) -> Self {
        Self {
            night_vision: if night_vision.is_nan() {
                0.0
            } else {
                night_vision.clamp(0.0, 1.0)
            },
            ..self
        }
    }

    /// The night bias for a scene whose measured mean log2 luminance is
    /// `avg_log_lum`: none down to [`NIGHT_BIAS_START`] stops under 1, all
    /// of it from [`NIGHT_BIAS_FULL`] on, a smoothstep between.
    pub fn night_bias(&self, avg_log_lum: f32) -> f32 {
        let t = ((-avg_log_lum - NIGHT_BIAS_START) / (NIGHT_BIAS_FULL - NIGHT_BIAS_START))
            .clamp(0.0, 1.0);
        self.night_bias_ev * t * t * (3.0 - 2.0 * t)
    }

    /// The darkest log2 luminance the histogram measures: two stops under
    /// the darkest scene `max_ev` can lift to scene white, and never above
    /// [`LUM_LOG2_MIN`], so the default range (`max_ev` 8) keeps its bins and
    /// a night world that raises `max_ev` is measured where it sits rather
    /// than dropped into the sub-floor bin.
    pub fn histogram_floor(&self) -> f32 {
        LUM_LOG2_MIN.min(-self.max_ev - 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resolve_clamps_speed_and_orders_ev_bounds() {
        let s = AutoExposureSettings::resolve(8.0, -2.0, 0.0, false);
        // Inverted min/max swap so the clamp interval is non-empty.
        assert_eq!(s.min_ev, -2.0);
        assert_eq!(s.max_ev, 8.0);
        // Zero speed is floored to a tiny positive rate.
        assert!(s.speed >= MIN_SPEED);

        let s = AutoExposureSettings::resolve(-100.0, 100.0, 1.0e9, false);
        assert_eq!(s.min_ev, -EV_LIMIT);
        assert_eq!(s.max_ev, EV_LIMIT);
        assert_eq!(s.speed, MAX_SPEED);
    }

    // The default range keeps the fixed floor; a night range reaches down to
    // a scene two stops under what the clamp can lift.
    #[test]
    fn the_night_bias_eases_in_as_the_scene_darkens() {
        let s = AutoExposureSettings::resolve(-8.0, 24.0, 1.0, false).with_night_bias(-3.0);
        // Day, dusk and the start of the ramp take none of it.
        for lum in [2.0, -4.4, -9.1, -13.3, -NIGHT_BIAS_START] {
            assert_eq!(s.night_bias(lum), 0.0, "{lum}");
        }
        // A moonlit night and darker take all of it.
        for lum in [-NIGHT_BIAS_FULL, -22.8, -23.6, -40.0] {
            assert_eq!(s.night_bias(lum), -3.0, "{lum}");
        }
        let mid = s.night_bias(-0.5 * (NIGHT_BIAS_START + NIGHT_BIAS_FULL));
        assert!((mid + 1.5).abs() < 1e-5, "{mid}");
        let mut last = 0.0;
        for i in 0..=40 {
            let b = s.night_bias(-(5.0 + i as f32 * 0.5));
            assert!(b <= last + 1e-6, "{i}: {b} after {last}");
            last = b;
        }
        // Off by default, and clamped to the exposure range.
        assert_eq!(
            AutoExposureSettings::resolve(-8.0, 24.0, 1.0, false).night_bias(-30.0),
            0.0
        );
        assert_eq!(s.with_night_bias(-100.0).night_bias_ev, -EV_LIMIT);
    }

    #[test]
    fn highlight_headroom_defaults_and_clamps() {
        let s = AutoExposureSettings::resolve(-8.0, 24.0, 1.0, false);
        assert_eq!(s.highlight_headroom_ev, HIGHLIGHT_HEADROOM_EV);
        assert_eq!(s.with_highlight_headroom(3.5).highlight_headroom_ev, 3.5);
        assert_eq!(s.with_highlight_headroom(-1.0).highlight_headroom_ev, 0.0);
        assert_eq!(
            s.with_highlight_headroom(100.0).highlight_headroom_ev,
            EV_LIMIT
        );
        assert_eq!(
            s.with_highlight_headroom(f32::NAN).highlight_headroom_ev,
            HIGHLIGHT_HEADROOM_EV
        );
        // Brightest pixels at 2^3 with two stops of headroom allow EV -1.
        assert_eq!(s.highlight_cap(3.0), -1.0);
    }

    #[test]
    fn night_vision_is_off_by_default_and_clamped() {
        let s = AutoExposureSettings::resolve(-8.0, 24.0, 1.0, false);
        assert_eq!(s.night_vision, 0.0);
        assert_eq!(s.with_night_vision(0.4).night_vision, 0.4);
        assert_eq!(s.with_night_vision(3.0).night_vision, 1.0);
        assert_eq!(s.with_night_vision(-1.0).night_vision, 0.0);
        assert_eq!(s.with_night_vision(f32::NAN).night_vision, 0.0);
    }

    #[test]
    fn the_histogram_floor_follows_the_max_ev() {
        assert_eq!(
            AutoExposureSettings::resolve(-8.0, 8.0, 1.5, false).histogram_floor(),
            LUM_LOG2_MIN
        );
        assert_eq!(
            AutoExposureSettings::resolve(-8.0, 8.0, 1.5, true).histogram_floor(),
            LUM_LOG2_MIN
        );
        assert_eq!(
            AutoExposureSettings::resolve(-8.0, 3.0, 1.5, false).histogram_floor(),
            LUM_LOG2_MIN
        );
        assert_eq!(
            AutoExposureSettings::resolve(-8.0, 16.0, 1.5, false).histogram_floor(),
            -18.0
        );
        assert_eq!(
            AutoExposureSettings::resolve(-8.0, 100.0, 1.5, false).histogram_floor(),
            -EV_LIMIT - 2.0
        );
    }
}
