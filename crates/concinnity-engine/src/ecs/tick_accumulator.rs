//! The windowed driver's simulation clock math. Elapsed time accumulates into
//! whole fixed ticks, and the remainder becomes the interpolation alpha the
//! simulation systems blend render transforms with. While a menu holds the world
//! paused it emits zero ticks and stops accumulating, so resuming costs nothing:
//! no catch-up burst by construction. The same delta is published unpaused as
//! the frame's [`FrameTime`].

use concinnity_core::math::{SaturatingFrom, narrow_f32};
use core::num::NonZeroU32;

use super::{FrameTime, MenuActive, SimTiming, World};

/// Accumulates frame time into fixed simulation ticks, advanced once per world
/// step with that frame's elapsed seconds. The caller owns the time source.
#[derive(Debug, Default)]
pub struct TickAccumulator {
    accumulator: f32,
    // Seconds since the first frame, kept wide so a long session stays precise.
    elapsed: f64,
    // Whole ticks run since the first frame.
    ticks_run: u64,
}

impl TickAccumulator {
    /// The tick budget and frame time for a frame that lasted `elapsed` seconds.
    /// `paused` holds the tick clock: no time accumulates and no ticks are
    /// emitted, but the remainder (and so the alpha) is kept, so the frozen
    /// frame keeps rendering the same blend. The frame time runs regardless.
    pub fn advance(&mut self, elapsed: f32, paused: bool) -> (SimTiming, FrameTime) {
        let dt = elapsed.max(0.0);
        self.elapsed += f64::from(dt);
        let frame = FrameTime {
            dt,
            elapsed: narrow_f32(self.elapsed),
        };
        (self.ticks(dt, paused), frame)
    }

    /// Advance by one world frame and publish its [`SimTiming`] and
    /// [`FrameTime`] into `world`. At a fixed launch rate a frame lasts exactly
    /// `1 / rate` seconds whatever the wall clock measured, so every launch at
    /// that rate sees the same clock on the same frame; otherwise it lasts
    /// `wall_elapsed`. Until the simulated time reaches `start_at` seconds, a
    /// frame lasts as many ticks as one may run, so a run that starts late
    /// gets there in fewer frames. An open menu ([`MenuActive`]) pauses the
    /// ticks.
    pub fn publish_frame(
        &mut self,
        world: &mut World,
        fixed_rate: Option<NonZeroU32>,
        wall_elapsed: f32,
        start_at: Option<f32>,
    ) {
        let elapsed = fixed_rate.map_or(wall_elapsed, |rate| 1.0 / rate.get() as f32);
        let elapsed = start_at.map_or(elapsed, |start| self.ahead(elapsed, start));
        let paused = world.resource::<MenuActive>().is_some_and(|m| m.0);
        let (timing, frame) = self.advance(elapsed, paused);
        world.insert_resource(timing);
        world.insert_resource(frame);
    }

    // A frame of `elapsed` seconds lengthened toward simulated time `start`,
    // by at most what one frame may run.
    fn ahead(&self, elapsed: f32, start: f32) -> f32 {
        let cap = SimTiming::MAX_TICKS_PER_FRAME as f32 * SimTiming::TICK_DT;
        let run = self.ticks_run as f64 * f64::from(SimTiming::TICK_DT);
        let behind = narrow_f32(f64::from(start) - run - f64::from(self.accumulator));
        elapsed.max(behind.min(cap))
    }

    fn ticks(&mut self, dt: f32, paused: bool) -> SimTiming {
        let tick_dt = SimTiming::TICK_DT;
        if !paused {
            self.accumulator += dt;
        }
        let cap = SimTiming::MAX_TICKS_PER_FRAME as f32 * tick_dt;
        self.accumulator = self.accumulator.min(cap);
        let ticks = if paused {
            0
        } else {
            u32::saturating_from(self.accumulator / tick_dt)
        };
        self.accumulator -= ticks as f32 * tick_dt;
        self.ticks_run += u64::from(ticks);
        SimTiming {
            ticks,
            tick_dt,
            alpha: (self.accumulator / tick_dt).clamp(0.0, 1.0),
            time: self.ticks_run as f64 * f64::from(tick_dt),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    // A fixed rate ignores the wall clock, and an open menu holds the ticks
    // while the frame time still runs.
    #[test]
    fn a_published_frame_follows_the_fixed_rate_and_the_menu() {
        let mut world = World::new();
        let mut ticks = TickAccumulator::default();
        ticks.publish_frame(&mut world, NonZeroU32::new(60), 5.0, None);
        assert_eq!(world.resource::<FrameTime>().unwrap().dt, 1.0 / 60.0);
        assert_eq!(world.resource::<SimTiming>().unwrap().ticks, 1);

        world.insert_resource(MenuActive(true));
        ticks.publish_frame(&mut world, None, 0.5, None);
        assert_eq!(world.resource::<FrameTime>().unwrap().dt, 0.5);
        assert_eq!(world.resource::<SimTiming>().unwrap().ticks, 0);
    }

    // A late start runs the most ticks a frame may until it reaches its
    // moment, lands on it without overshooting, and then keeps the pace.
    #[test]
    fn a_late_start_runs_ahead_to_its_moment_then_keeps_the_pace() {
        let mut world = World::new();
        let mut ticks = TickAccumulator::default();
        let rate = NonZeroU32::new(60);
        let start = 12.5 * SimTiming::TICK_DT;
        let mut per_frame = Vec::new();
        for _ in 0..5 {
            ticks.publish_frame(&mut world, rate, 0.0, Some(start));
            per_frame.push(world.resource::<SimTiming>().unwrap().ticks);
        }
        let max = SimTiming::MAX_TICKS_PER_FRAME;
        assert_eq!(per_frame[..2], [max, max]);
        let time = world.resource::<SimTiming>().unwrap().time;
        assert!(
            time >= f64::from(start) - f64::from(SimTiming::TICK_DT),
            "{per_frame:?}"
        );
        assert_eq!(per_frame[4], 1, "past the start a frame is one tick again");
    }

    const DT: f32 = SimTiming::TICK_DT;

    // The simulated time counts whole ticks only: it holds through a pause
    // and through the time a hitch drops, and each tick starts where the one
    // before it ended.
    #[test]
    fn simulated_time_counts_the_ticks_run() {
        let mut clock = TickAccumulator::default();
        let t = clock.advance(DT * 2.5, false).0;
        assert_eq!(t.time, 2.0 * f64::from(DT));
        assert_eq!(t.tick_start(0), 0.0);
        assert_eq!(t.tick_start(1), f64::from(DT));
        assert!((t.drawn_time() - 1.5 * f64::from(DT)).abs() < 1e-6);
        let paused = clock.advance(1.0, true).0;
        assert_eq!(paused.time, t.time);
        let hitch = clock.advance(10.0, false).0;
        let cap = f64::from(SimTiming::MAX_TICKS_PER_FRAME);
        assert!((hitch.time - t.time - cap * f64::from(DT)).abs() < 1e-9);
    }

    #[test]
    fn accumulates_whole_ticks_and_keeps_the_remainder() {
        let mut clock = TickAccumulator::default();
        let t = clock.advance(DT * 2.5, false).0;
        assert_eq!(t.ticks, 2);
        assert!((t.alpha - 0.5).abs() < 1.0e-4, "alpha = {}", t.alpha);

        // The half-tick remainder carries into the next frame.
        let t = clock.advance(DT * 0.6, false).0;
        assert_eq!(t.ticks, 1);
        assert!((t.alpha - 0.1).abs() < 1.0e-3, "alpha = {}", t.alpha);
    }

    #[test]
    fn short_frames_emit_zero_ticks_with_growing_alpha() {
        let mut clock = TickAccumulator::default();
        let a = clock.advance(DT * 0.4, false).0;
        assert_eq!(a.ticks, 0);
        let b = clock.advance(DT * 0.4, false).0;
        assert_eq!(b.ticks, 0);
        assert!(b.alpha > a.alpha, "alpha advances between un-ticked frames");
        let c = clock.advance(DT * 0.4, false).0;
        assert_eq!(c.ticks, 1, "the third short frame crosses a tick");
    }

    #[test]
    fn a_hitch_is_clamped_to_the_tick_cap() {
        let mut clock = TickAccumulator::default();
        let t = clock.advance(2.0, false).0;
        assert_eq!(t.ticks, SimTiming::MAX_TICKS_PER_FRAME);
        assert!(t.alpha < 1.0e-4, "the overflow past the cap is dropped");
    }

    #[test]
    fn pause_emits_no_ticks_and_holds_the_accumulator() {
        let mut clock = TickAccumulator::default();
        clock.advance(DT * 0.5, false);
        let frozen = clock.advance(DT * 20.0, true).0;
        assert_eq!(frozen.ticks, 0);
        assert!(
            (frozen.alpha - 0.5).abs() < 1.0e-4,
            "the paused blend holds at the pre-pause remainder"
        );
        // Resuming costs one normal frame, not the paused span.
        let resumed = clock.advance(DT, false).0;
        assert_eq!(resumed.ticks, 1);
    }

    #[test]
    fn frame_time_is_the_real_delta_and_accumulates_elapsed() {
        let mut clock = TickAccumulator::default();
        let (_, a) = clock.advance(0.25, false);
        assert_eq!(
            a,
            FrameTime {
                dt: 0.25,
                elapsed: 0.25
            }
        );
        let (_, b) = clock.advance(0.5, false);
        assert_eq!(
            b,
            FrameTime {
                dt: 0.5,
                elapsed: 0.75
            }
        );
    }

    #[test]
    fn frame_time_keeps_running_while_paused() {
        let mut clock = TickAccumulator::default();
        clock.advance(DT, false);
        let (timing, frame) = clock.advance(2.0, true);
        assert_eq!(timing.ticks, 0);
        assert_eq!(frame.dt, 2.0, "dt is real time even while paused");
        assert!((frame.elapsed - (DT + 2.0)).abs() < 1.0e-6);
    }

    #[test]
    fn a_negative_delta_publishes_no_time() {
        let mut clock = TickAccumulator::default();
        let (_, frame) = clock.advance(-1.0, false);
        assert_eq!(
            frame,
            FrameTime {
                dt: 0.0,
                elapsed: 0.0
            }
        );
    }

    // A fixed delta steps the same clock on every run, frame for frame.
    #[test]
    fn a_fixed_delta_steps_identically_every_run() {
        let run = || {
            let mut clock = TickAccumulator::default();
            (0..300)
                .map(|_| clock.advance(1.0 / 60.0, false))
                .map(|(timing, frame)| (timing.ticks, frame))
                .collect::<Vec<_>>()
        };
        let frames = run();
        assert_eq!(run(), frames);
        let (_, frame) = *frames.last().expect("frames ran");
        assert_eq!(frame.dt, 1.0 / 60.0);
        assert!((frame.elapsed - 5.0).abs() < 1.0e-4, "{}", frame.elapsed);
    }
}
