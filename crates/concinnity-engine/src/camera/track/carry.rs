// Picking a subject up mid-motion: how fast it was going as the track filmed
// it, and the blend that eases it from where it was onto travel's path.

use concinnity_core::math::vec3;

use crate::planet::Rebase;

/// Seconds the blend takes to ease a picked-up subject onto travel's path.
pub(super) const CARRY_BLEND_SECONDS: f32 = 6.0;

/// The subject's position at the last two moments it was filmed, so the
/// speed it is going at can be read off when it is picked up.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct Filmed {
    last: Option<([f32; 3], f32)>,
    velocity: [f32; 3],
}

impl Filmed {
    /// Record the subject `at` track time `seconds`. A moment no later than
    /// the last keeps the speed it had.
    pub(super) fn see(&mut self, at: [f32; 3], seconds: f32) {
        if let Some((before, then)) = self.last
            && seconds > then
        {
            self.velocity = vec3::scale(vec3::sub(at, before), 1.0 / (seconds - then));
        }
        self.last = Some((at, seconds));
    }

    /// How fast the subject was going when last seen, in units per second.
    pub(super) fn velocity(&self) -> [f32; 3] {
        self.velocity
    }

    /// Carry what was seen into another frame.
    pub(super) fn rebase(&mut self, rebase: &Rebase) {
        self.last = self.last.map(|(at, t)| (rebase.apply_point(at), t));
        self.velocity = rebase.apply_vector(self.velocity);
    }
}

/// The gap between where a picked-up subject is and where travel's path has
/// it, closed over [`CARRY_BLEND_SECONDS`] by a cubic that starts at the gap
/// with the speed difference as its slope and ends at nothing with none, so
/// the subject's position and speed both run on without a jump.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct CarryBlend {
    offset: [f32; 3],
    velocity: [f32; 3],
    from_seconds: f32,
}

impl CarryBlend {
    /// The blend from a subject `offset` away from the path and going
    /// `velocity` faster than it, picked up at track time `from_seconds`.
    pub(super) fn new(offset: [f32; 3], velocity: [f32; 3], from_seconds: f32) -> Self {
        Self {
            offset,
            velocity,
            from_seconds,
        }
    }

    /// The gap left at track time `seconds`.
    pub(super) fn at(&self, seconds: f32) -> [f32; 3] {
        let span = CARRY_BLEND_SECONDS;
        let s = ((seconds - self.from_seconds) / span).clamp(0.0, 1.0);
        let h00 = 2.0 * s * s * s - 3.0 * s * s + 1.0;
        let h10 = (s * s * s - 2.0 * s * s + s) * span;
        core::array::from_fn(|i| self.offset[i] * h00 + self.velocity[i] * h10)
    }

    /// Carry the gap into another frame.
    pub(super) fn rebase(&mut self, rebase: &Rebase) {
        self.offset = rebase.apply_vector(self.offset);
        self.velocity = rebase.apply_vector(self.velocity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The gap starts whole with the speed difference as its slope, and is
    // gone with no slope once the blend has run.
    #[test]
    fn the_blend_starts_at_the_gap_and_ends_at_nothing() {
        let blend = CarryBlend::new([2.0, 0.0, 0.0], [0.0, 3.0, 0.0], 10.0);
        assert_eq!(blend.at(10.0), [2.0, 0.0, 0.0]);
        let h = 1e-3;
        let slope = vec3::scale(vec3::sub(blend.at(10.0 + h), blend.at(10.0)), 1.0 / h);
        assert!((slope[1] - 3.0).abs() < 1e-2, "{slope:?}");
        assert!(slope[0].abs() < 1e-2, "{slope:?}");
        let end = 10.0 + CARRY_BLEND_SECONDS;
        assert_eq!(blend.at(end), [0.0; 3]);
        assert_eq!(blend.at(end + 5.0), [0.0; 3]);
        let late = vec3::sub(blend.at(end), blend.at(end - h));
        assert!(vec3::length(late) < 1e-4, "{late:?}");
    }

    // The speed is read off the last two moments seen; a moment that does
    // not advance leaves it.
    #[test]
    fn the_speed_is_read_off_the_last_two_moments() {
        let mut filmed = Filmed::default();
        filmed.see([0.0; 3], 1.0);
        assert_eq!(filmed.velocity(), [0.0; 3]);
        filmed.see([1.0, 0.0, 0.0], 1.5);
        assert_eq!(filmed.velocity(), [2.0, 0.0, 0.0]);
        filmed.see([1.0, 0.0, 0.0], 1.5);
        assert_eq!(filmed.velocity(), [2.0, 0.0, 0.0]);
    }
}
