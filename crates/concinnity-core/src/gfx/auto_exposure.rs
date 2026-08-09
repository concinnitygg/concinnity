// src/gfx/auto_exposure.rs
//
// The resolved auto-exposure tunables the post-process config produces and the
// backends hold. The running EMA that consumes them, and the histogram it is
// measured from, are per-frame compute and live above this crate.

// Smallest legal EMA speed. A zero or negative speed would freeze adaptation
// at the initial EV, so the authored value is floored here.
const MIN_SPEED: f32 = 1.0e-3;

// Largest legal EMA speed. Anything higher snaps in under a single frame and
// is indistinguishable from "no adaptation" visually, just noisier.
const MAX_SPEED: f32 = 20.0;

// Clamp range for `min_ev` / `max_ev` so a stray value cannot push exposure to
// `inf` / `0`. Matches the `EXPOSURE_EV_LIMIT` in [`PostProcessConfig`].
const EV_LIMIT: f32 = 16.0;

// `log2(0.18)`: perceptual middle-grey in linear light. AE shifts the
// scene's geometric-mean luminance to this value on the HDR output path so
// the average pixel reads as a comfortable mid-tone instead of "scene
// white = SDR reference white = bright" (which only worked on the SDR path
// because the ACES tonemap implicitly compressed scene-white back down).
pub const HDR_MIDDLE_GREY_LOG2: f32 = -2.473;

// Clamped auto-exposure tunables resolved from the authored asset fields. Held
// by the backend; the per-frame EMA in [`AutoExposureState::update`] reads them
// to clamp the adapted EV and drive its adaptation rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoExposureSettings {
    // Lower bound on the adapted EV. Caps how bright a dim scene can ramp.
    pub min_ev: f32,
    // Upper bound on the adapted EV. Caps how dark a bright scene can ramp.
    pub max_ev: f32,
    // EMA rate (per second). The exponential `1 - exp(-speed * dt)` step pulls
    // the current EV toward the target each frame; higher = faster adaptation.
    pub speed: f32,
    // Log2 of the linear value AE aims the scene's geometric-mean luminance
    // at. `0.0` = scene-white (legacy SDR + ACES default, ACES then squishes
    // scene-white back down to a comfortable display mid-tone).
    // `HDR_MIDDLE_GREY_LOG2` ≈ -2.47 = perceptual middle-grey, the correct
    // target on the HDR path where there is no ACES compression. Resolved
    // from `PostProcessConfig.hdr_display` at asset time.
    pub target_log_lum: f32,
}

impl AutoExposureSettings {
    // Clamp the authored fields into a safe range. `min_ev` is forced to stay
    // at-or-below `max_ev` so the adapted EV's clamp interval is non-empty.
    // `hdr_aware` shifts AE's middle-grey pivot down so the average pixel
    // reads at perceptual middle-grey on the HDR output path; SDR worlds
    // keep the legacy scene-white pivot to preserve existing exposure
    // authoring.
    pub fn resolve(min_ev: f32, max_ev: f32, speed: f32, hdr_aware: bool) -> Self {
        let min = min_ev.clamp(-EV_LIMIT, EV_LIMIT);
        let max = max_ev.clamp(-EV_LIMIT, EV_LIMIT);
        let (lo, hi) = if min <= max { (min, max) } else { (max, min) };
        Self {
            min_ev: lo,
            max_ev: hi,
            speed: speed.clamp(MIN_SPEED, MAX_SPEED),
            target_log_lum: if hdr_aware { HDR_MIDDLE_GREY_LOG2 } else { 0.0 },
        }
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
}
