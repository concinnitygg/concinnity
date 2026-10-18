//! Maps authored AudioEmitter attenuation onto kira's spatial-track
//! parameters; core's `audio::distance` sanitizes the range.

use concinnity_core::components::Rolloff;
use kira::Easing;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolloff_kinds_map_to_distinct_curves() {
        assert_eq!(attenuation(Rolloff::Logarithmic), Some(Easing::Linear));
        assert_eq!(attenuation(Rolloff::Linear), Some(Easing::OutPowi(2)));
        assert_eq!(attenuation(Rolloff::None), None);
    }
}
