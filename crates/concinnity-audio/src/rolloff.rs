// concinnity-audio/src/rolloff.rs
//
// Maps authored AudioEmitter attenuation onto kira's spatial-track
// parameters, sanitizing degenerate values (the cook rejects them, but a
// hand-edited blob or hot-reloaded world must not produce NaN volume).

use concinnity_core::assets::Rolloff;
use kira::Easing;

// Fallbacks matching the AudioEmitter schema defaults.
const DEFAULT_MIN_DISTANCE: f32 = 1.0;
const DEFAULT_MAX_DISTANCE: f32 = 50.0;

// The volume curve between min and max distance. kira interpolates the
// track's volume in decibels from identity down to silence, shaping the
// interpolant with this easing; `Linear` in that domain is the natural,
// steep-near-the-source falloff, and `OutPowi(2)` spends more of the range
// audible, which reads as a gradual, linear-like fade.
pub(crate) fn attenuation(rolloff: Rolloff) -> Option<Easing> {
    match rolloff {
        Rolloff::Logarithmic => Some(Easing::Linear),
        Rolloff::Linear => Some(Easing::OutPowi(2)),
        Rolloff::None => None,
    }
}

// Sanitize an authored distance pair: non-finite values fall back to the
// schema defaults, negatives clamp to zero, and an empty or inverted range
// is widened so kira never divides by zero.
pub(crate) fn clamp_distances(min: f32, max: f32) -> (f32, f32) {
    let min = if min.is_finite() {
        min.max(0.0)
    } else {
        DEFAULT_MIN_DISTANCE
    };
    let max = if max.is_finite() {
        max
    } else {
        DEFAULT_MAX_DISTANCE
    };
    if max > min {
        (min, max)
    } else {
        (min, min + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolloff_kinds_map_to_distinct_curves() {
        assert_eq!(attenuation(Rolloff::Logarithmic), Some(Easing::Linear));
        assert_eq!(attenuation(Rolloff::Linear), Some(Easing::OutPowi(2)));
        assert_eq!(attenuation(Rolloff::None), None);
    }

    #[test]
    fn valid_distances_pass_through() {
        assert_eq!(clamp_distances(2.0, 80.0), (2.0, 80.0));
        assert_eq!(clamp_distances(0.0, 0.5), (0.0, 0.5));
    }

    #[test]
    fn degenerate_distances_are_widened() {
        // Equal and inverted ranges widen instead of dividing by zero.
        assert_eq!(clamp_distances(5.0, 5.0), (5.0, 6.0));
        assert_eq!(clamp_distances(10.0, 3.0), (10.0, 11.0));
    }

    #[test]
    fn non_finite_and_negative_distances_fall_back() {
        assert_eq!(clamp_distances(f32::NAN, 20.0), (1.0, 20.0));
        assert_eq!(clamp_distances(1.0, f32::INFINITY), (1.0, 50.0));
        let (min, max) = clamp_distances(-3.0, -1.0);
        assert_eq!(min, 0.0);
        assert!(max > min);
    }
}
