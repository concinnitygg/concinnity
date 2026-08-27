// src/app/clock.rs
//
// Fixed-timestep accumulator for the simulation. Runs at the App level, after
// the frame pacer and before the world steps: wall-clock time accumulates into
// whole fixed ticks, and the remainder becomes the interpolation alpha the
// simulation systems blend render transforms with. While a menu holds the
// world paused the clock emits zero ticks and stops accumulating, so resuming
// costs nothing -- no catch-up burst by construction.

use crate::ecs::SimTiming;
use std::sync::OnceLock;
use std::time::Instant;

// Monotonic micros since the first call, for the `Clock` resource the world's
// step loop times each system with. The epoch is process-wide because the
// resource carries a plain function pointer, and only differences are read.
pub(crate) fn monotonic_micros() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_micros() as u64
}

// Most fixed ticks one frame may run. Accumulated time past this is dropped,
// so a long hitch degrades to slow motion instead of a tick spiral where each
// frame's catch-up work makes the next frame longer.
const MAX_TICKS_PER_FRAME: u32 = 5;

// Accumulates wall-clock time into fixed simulation ticks. One per `App`,
// advanced once per world step.
#[derive(Debug, Default)]
pub(crate) struct SimClock {
    last: Option<Instant>,
    accumulator: f32,
}

impl SimClock {
    // The tick budget for a frame starting at `now`. `paused` holds the clock:
    // no time accumulates and no ticks are emitted, but the remainder (and so
    // the alpha) is kept, so the frozen frame keeps rendering the same blend.
    pub(crate) fn advance(&mut self, now: Instant, paused: bool) -> SimTiming {
        let elapsed = self
            .last
            .map(|t| now.duration_since(t).as_secs_f32())
            .unwrap_or(0.0);
        self.last = Some(now);
        self.advance_by(elapsed, paused)
    }

    // The accumulator math, split from the wall clock so it is testable with
    // injected durations.
    fn advance_by(&mut self, elapsed: f32, paused: bool) -> SimTiming {
        let tick_dt = SimTiming::TICK_DT;
        if !paused {
            self.accumulator += elapsed.max(0.0);
        }
        let cap = MAX_TICKS_PER_FRAME as f32 * tick_dt;
        self.accumulator = self.accumulator.min(cap);
        let ticks = if paused {
            0
        } else {
            (self.accumulator / tick_dt) as u32
        };
        self.accumulator -= ticks as f32 * tick_dt;
        SimTiming {
            ticks,
            tick_dt,
            alpha: (self.accumulator / tick_dt).clamp(0.0, 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = SimTiming::TICK_DT;

    // The process clock only ever moves forward, which is the whole contract
    // the step loop's per-system deltas rest on.
    #[test]
    fn monotonic_micros_never_goes_backwards() {
        let first = monotonic_micros();
        assert!(monotonic_micros() >= first);
    }

    #[test]
    fn accumulates_whole_ticks_and_keeps_the_remainder() {
        let mut clock = SimClock::default();
        let t = clock.advance_by(DT * 2.5, false);
        assert_eq!(t.ticks, 2);
        assert!((t.alpha - 0.5).abs() < 1.0e-4, "alpha = {}", t.alpha);

        // The half-tick remainder carries into the next frame.
        let t = clock.advance_by(DT * 0.6, false);
        assert_eq!(t.ticks, 1);
        assert!((t.alpha - 0.1).abs() < 1.0e-3, "alpha = {}", t.alpha);
    }

    #[test]
    fn short_frames_emit_zero_ticks_with_growing_alpha() {
        let mut clock = SimClock::default();
        let a = clock.advance_by(DT * 0.4, false);
        assert_eq!(a.ticks, 0);
        let b = clock.advance_by(DT * 0.4, false);
        assert_eq!(b.ticks, 0);
        assert!(b.alpha > a.alpha, "alpha advances between un-ticked frames");
        let c = clock.advance_by(DT * 0.4, false);
        assert_eq!(c.ticks, 1, "the third short frame crosses a tick");
    }

    #[test]
    fn a_hitch_is_clamped_to_the_tick_cap() {
        let mut clock = SimClock::default();
        let t = clock.advance_by(2.0, false);
        assert_eq!(t.ticks, MAX_TICKS_PER_FRAME);
        assert!(t.alpha < 1.0e-4, "the overflow past the cap is dropped");
    }

    #[test]
    fn pause_emits_no_ticks_and_holds_the_accumulator() {
        let mut clock = SimClock::default();
        clock.advance_by(DT * 0.5, false);
        let frozen = clock.advance_by(DT * 20.0, true);
        assert_eq!(frozen.ticks, 0);
        assert!(
            (frozen.alpha - 0.5).abs() < 1.0e-4,
            "the paused blend holds at the pre-pause remainder"
        );
        // Resuming costs one normal frame, not the paused span.
        let resumed = clock.advance_by(DT, false);
        assert_eq!(resumed.ticks, 1);
    }

    #[test]
    fn first_frame_has_no_elapsed_time() {
        let mut clock = SimClock::default();
        let t = clock.advance(Instant::now(), false);
        assert_eq!(t.ticks, 0);
    }
}
