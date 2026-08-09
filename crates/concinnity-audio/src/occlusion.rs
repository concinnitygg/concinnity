// concinnity-audio/src/occlusion.rs
//
// Per-emitter occlusion response: smooths the physics probe's blocked /
// clear answer over time and maps the smoothed factor to a volume dip and a
// lowpass cutoff, so an emitter passing behind a wall muffles instead of
// popping.

// Smoothing time constant: roughly how long a transition takes to settle.
const TAU_SECONDS: f32 = 0.1;
// Volume at full occlusion (0 dB when clear).
const OCCLUDED_VOLUME_DB: f32 = -9.0;
// Lowpass cutoff sweep endpoints. 20 kHz is acoustically transparent.
const OPEN_CUTOFF_HZ: f64 = 20_000.0;
const OCCLUDED_CUTOFF_HZ: f64 = 1_000.0;

// Exponentially smoothed occlusion factor for one emitter, in [0, 1].
pub(crate) struct OcclusionSmoother {
    current: f32,
}

impl OcclusionSmoother {
    pub(crate) fn new() -> Self {
        Self { current: 0.0 }
    }

    // Advance one tick of `dt` seconds toward blocked (1.0) or clear (0.0),
    // returning the smoothed factor.
    pub(crate) fn step(&mut self, blocked: bool, dt: f32) -> f32 {
        let target = if blocked { 1.0 } else { 0.0 };
        let alpha = 1.0 - (-dt / TAU_SECONDS).exp();
        self.current += (target - self.current) * alpha;
        self.current
    }

    #[cfg(test)]
    pub(crate) fn current(&self) -> f32 {
        self.current
    }
}

// Volume adjustment in decibels for a smoothed occlusion factor.
pub(crate) fn volume_db(occlusion: f32) -> f32 {
    OCCLUDED_VOLUME_DB * occlusion.clamp(0.0, 1.0)
}

// Lowpass cutoff for a smoothed occlusion factor, swept in log-frequency
// space so the muffling sounds even across the range.
pub(crate) fn cutoff_hz(occlusion: f32) -> f64 {
    let t = occlusion.clamp(0.0, 1.0) as f64;
    (OPEN_CUTOFF_HZ.ln() + (OCCLUDED_CUTOFF_HZ.ln() - OPEN_CUTOFF_HZ.ln()) * t).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 60.0;

    #[test]
    fn smoother_converges_without_overshoot() {
        let mut s = OcclusionSmoother::new();
        let mut prev = 0.0;
        for _ in 0..60 {
            let v = s.step(true, DT);
            assert!(v >= prev && v <= 1.0, "monotonic rise, no overshoot");
            prev = v;
        }
        assert!(prev > 0.99, "settled near 1.0 after 1s: {prev}");
        for _ in 0..60 {
            prev = s.step(false, DT);
        }
        assert!(prev < 0.01, "settled near 0.0 after clearing: {prev}");
    }

    #[test]
    fn smoothing_takes_several_ticks() {
        // The whole point is that one blocked answer does not slam the
        // volume: after a single tick the factor is still far from 1.
        let mut s = OcclusionSmoother::new();
        let v = s.step(true, DT);
        assert!(v > 0.0 && v < 0.3, "one tick moves partway: {v}");
    }

    #[test]
    fn volume_maps_clear_to_unity_and_blocked_to_dip() {
        assert_eq!(volume_db(0.0), 0.0);
        assert_eq!(volume_db(1.0), OCCLUDED_VOLUME_DB);
        assert!(volume_db(0.5) < 0.0 && volume_db(0.5) > OCCLUDED_VOLUME_DB);
        // Out-of-range factors clamp instead of amplifying.
        assert_eq!(volume_db(-1.0), 0.0);
        assert_eq!(volume_db(2.0), OCCLUDED_VOLUME_DB);
    }

    #[test]
    fn cutoff_sweeps_between_endpoints() {
        assert!((cutoff_hz(0.0) - OPEN_CUTOFF_HZ).abs() < 1.0);
        assert!((cutoff_hz(1.0) - OCCLUDED_CUTOFF_HZ).abs() < 1.0);
        let mid = cutoff_hz(0.5);
        assert!(mid < OPEN_CUTOFF_HZ && mid > OCCLUDED_CUTOFF_HZ);
        // Log-space midpoint, not arithmetic: sqrt(20k * 1k) ~ 4472 Hz.
        assert!((mid - 4472.0).abs() < 10.0, "log-space midpoint: {mid}");
    }
}
