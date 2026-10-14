//! Auto-exposure: the tunables the post-process config resolves into, the
//! luminance histogram the backends measure a frame with, and the running EMA
//! that turns that measurement into the next frame's exposure.

mod pre_exposure;
mod state;

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
        }
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
