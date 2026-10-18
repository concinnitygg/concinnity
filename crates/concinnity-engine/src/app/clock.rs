//! The wall-clock side of the simulation clock. Runs at the runtime level, after
//! the frame pacer and before the world steps: each frame's wall delta (or the
//! fixed launch rate's) feeds core's [`TickAccumulator`].

use concinnity_core::ecs::{FrameTime, SimTiming, TickAccumulator};
use std::num::NonZeroU32;
use std::time::Instant;

// One per `Runtime`, advanced once per world step.
#[derive(Debug, Default)]
pub(crate) struct SimClock {
    last: Option<Instant>,
    ticks: TickAccumulator,
}

impl SimClock {
    // The tick budget and real frame time for a frame starting at `now`.
    pub(crate) fn advance(&mut self, now: Instant, paused: bool) -> (SimTiming, FrameTime) {
        let elapsed = self
            .last
            .map(|t| now.duration_since(t).as_secs_f32())
            .unwrap_or(0.0);
        self.last = Some(now);
        self.ticks.advance(elapsed, paused)
    }

    // The tick budget and frame time for a frame that lasts exactly
    // `1 / rate` seconds whatever the wall clock says, so every launch at that
    // rate sees the same clock on the same frame.
    pub(crate) fn advance_fixed(
        &mut self,
        rate: NonZeroU32,
        paused: bool,
    ) -> (SimTiming, FrameTime) {
        self.ticks.advance(1.0 / rate.get() as f32, paused)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_frame_has_no_elapsed_time() {
        let mut clock = SimClock::default();
        let (t, frame) = clock.advance(Instant::now(), false);
        assert_eq!(t.ticks, 0);
        assert_eq!(frame.elapsed, 0.0);
    }

    #[test]
    fn a_fixed_rate_frame_lasts_one_period() {
        let mut clock = SimClock::default();
        let (_, frame) = clock.advance_fixed(NonZeroU32::new(60).unwrap(), false);
        assert_eq!(frame.dt, 1.0 / 60.0);
    }
}
