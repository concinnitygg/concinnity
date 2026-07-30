// src/editor/behavior/pulse.rs
//
// Execution pulses: while the world simulates, nodes the behavior system
// reports as executed light their chart card / outline row with a warm blend
// that fades over a fixed window, so a single-frame firing stays visible.
// The decay is pure math over an age; the hook owns the clock.

use super::path::Path;

// How long a firing stays visible.
pub(crate) const PULSE_SECS: f32 = 0.6;

// How strongly a fresh pulse leans a card's fill toward the accent.
const BLEND_MAX: f32 = 0.55;

const PULSE_TINT: [f32; 3] = [0.95, 0.72, 0.30];

// One node's last observed execution, held by the hook for the open behavior.
#[derive(Debug, Clone)]
pub(crate) struct NodePulse {
    pub node: u32,
    pub path: Path,
    pub at: std::time::Instant,
}

// The pulse strength for a firing `age` seconds old: full at zero, gone at
// `PULSE_SECS`.
pub(crate) fn alpha(age: f32) -> f32 {
    (1.0 - age / PULSE_SECS).clamp(0.0, 1.0)
}

// A fill leaned toward the pulse accent by `alpha`. The alpha channel is
// lifted at least to the blend strength, so a pulse still shows on an idle
// outline row whose resting tint is fully transparent.
pub(crate) fn blend(base: [f32; 4], alpha: f32) -> [f32; 4] {
    let k = alpha.clamp(0.0, 1.0) * BLEND_MAX;
    [
        base[0] + (PULSE_TINT[0] - base[0]) * k,
        base[1] + (PULSE_TINT[1] - base[1]) * k,
        base[2] + (PULSE_TINT[2] - base[2]) * k,
        base[3].max(k),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_decays_from_full_to_gone() {
        assert_eq!(alpha(0.0), 1.0);
        assert_eq!(alpha(PULSE_SECS), 0.0);
        assert_eq!(alpha(PULSE_SECS * 2.0), 0.0, "never negative");
        let mid = alpha(PULSE_SECS * 0.5);
        assert!(mid > 0.4 && mid < 0.6);
    }

    #[test]
    fn blend_leans_toward_the_accent_and_keeps_base_alpha() {
        let base = [0.1, 0.1, 0.1, 0.85];
        assert_eq!(blend(base, 0.0), base, "no pulse leaves the fill alone");
        let hot = blend(base, 1.0);
        assert!(hot[0] > base[0] && hot[1] > base[1] && hot[2] > base[2]);
        assert_eq!(hot[3], base[3], "an opaque card keeps its own alpha");
        assert!(hot[0] < PULSE_TINT[0], "a pulse tints, never replaces");
    }

    #[test]
    fn blend_lifts_a_transparent_row_into_view() {
        let clear = [0.0, 0.0, 0.0, 0.0];
        assert!(blend(clear, 1.0)[3] > 0.0, "a pulse shows on an idle row");
        assert_eq!(blend(clear, 0.0)[3], 0.0, "no pulse stays invisible");
    }
}
