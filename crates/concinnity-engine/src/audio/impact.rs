// src/audio/impact.rs
//
// Maps a physics contact impulse to a one-shot gain. Physics already gates
// contacts on the world's minimum impulse and debounces repeating pairs, so
// everything arriving here is worth hearing; this only decides how loud.

// Impulse at which an impact reaches full volume. The default contact gate
// is 1.0, so audible impacts span roughly two decades below this.
const FULL_IMPULSE: f32 = 25.0;

// Gain in [0, 1] for a contact impulse. Square-root curve: perceived
// loudness grows quickly for light taps and saturates toward hard hits.
pub(crate) fn gain(impulse: f32) -> f32 {
    (impulse.max(0.0) / FULL_IMPULSE).sqrt().min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gain_is_monotonic_and_clamped() {
        assert_eq!(gain(0.0), 0.0);
        assert_eq!(gain(-5.0), 0.0, "negative impulse clamps silent");
        let mut prev = 0.0;
        for i in 1..100 {
            let g = gain(i as f32);
            assert!(g >= prev && g <= 1.0, "monotonic in [0,1]: {g}");
            prev = g;
        }
        assert_eq!(gain(FULL_IMPULSE), 1.0);
        assert_eq!(gain(1000.0), 1.0, "hard hits saturate");
    }

    #[test]
    fn light_taps_are_audible_but_quiet() {
        // The default contact gate (impulse 1.0) lands well above silence
        // and well below full volume.
        let g = gain(1.0);
        assert!(g > 0.1 && g < 0.4, "gate-level impact: {g}");
    }
}
