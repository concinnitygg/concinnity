//! Frame pacing for the FPS cap: when each frame may start, and how close to
//! that deadline a host stops sleeping and spins. Times are offsets from any
//! fixed origin the host keeps.

use core::time::Duration;

use crate::ecs::SimTiming;

// The rate a minimized window is held to whatever the cap: few enough frames
// to idle the GPU, enough that every frame's simulation ticks fit under the
// per-frame tick limit, so the world keeps running at full speed behind it.
const MINIMIZED_FPS: u32 = 30;
const _: () =
    assert!(MINIMIZED_FPS * SimTiming::MAX_TICKS_PER_FRAME >= (1.0 / SimTiming::TICK_DT) as u32);

// The cap a frame is held to: `cap`, lowered to `MINIMIZED_FPS` while the
// window is minimized (0 is uncapped).
fn effective_cap(cap: u32, minimized: bool) -> u32 {
    match (minimized, cap) {
        (false, cap) => cap,
        (true, 0) => MINIMIZED_FPS,
        (true, cap) => cap.min(MINIMIZED_FPS),
    }
}

// Given the current time, the previously scheduled deadline (if any), and the
// target frame interval, returns `(wait_until, next)`: the deadline to hold
// this frame's start to, and the deadline to store for the next frame. A
// deadline still in the future is waited to and accumulated exactly (no
// drift); one already passed (a frame that overran the budget) resets to `now`
// so a slow frame never triggers a catch-up burst of un-paced frames.
fn pace_deadline(now: Duration, prev: Option<Duration>, target: Duration) -> (Duration, Duration) {
    let wait_until = prev.filter(|d| *d > now).unwrap_or(now);
    (wait_until, wait_until + target)
}

// How far ahead of a deadline the pacer stops sleeping and spins. Learned from
// how late the OS actually wakes the thread, so a precise scheduler spins for
// microseconds and a coarse one still meets the deadline.
#[derive(Debug, Clone, Copy)]
struct SleepMargin(Duration);

impl SleepMargin {
    const MIN: Duration = Duration::from_micros(100);
    const MAX: Duration = Duration::from_millis(3);
    const INITIAL: Duration = Duration::from_millis(1);

    // Record one sleep that asked for `requested` and took `actual`. A late
    // wake raises the margin at once, with headroom; prompt wakes let it
    // decay slowly, so one lucky wake never invites the next late one.
    fn observe(&mut self, requested: Duration, actual: Duration) {
        let late = actual.saturating_sub(requested);
        let wanted = (late + late / 2).clamp(Self::MIN, Self::MAX);
        self.0 = if wanted >= self.0 {
            wanted
        } else {
            self.0 - (self.0 - wanted) / 16
        };
    }
}

impl Default for SleepMargin {
    fn default() -> Self {
        Self(Self::INITIAL)
    }
}

/// The cap in force (0 is uncapped): a launch's own cap, else a benchmarked
/// run's, else the one the world publishes.
pub fn frame_cap(launch: Option<u32>, benchmark: Option<u32>, published: Option<u32>) -> u32 {
    launch.or(benchmark).or(published).unwrap_or(0)
}

/// Holds each frame's start to the target interval so a loop runs at most its
/// cap's frames a second. One per loop, planned once per frame.
#[derive(Debug, Default)]
pub struct FramePace {
    // The running target for the next frame's start.
    deadline: Option<Duration>,
    // The cap the deadline was accumulated under. A cap change re-bases the
    // pace (clears the deadline) so switching caps never leaves one stale long
    // wait.
    last_cap: u32,
    margin: SleepMargin,
}

impl FramePace {
    /// Plan the frame starting at `now` under `cap` (0 is uncapped), lowered
    /// to a background rate while `minimized`: the time to hold its start to,
    /// or `None` while no cap is in force.
    pub fn plan(&mut self, cap: u32, minimized: bool, now: Duration) -> Option<Duration> {
        let cap = effective_cap(cap, minimized);
        if cap != self.last_cap {
            self.deadline = None;
            self.last_cap = cap;
        }
        if cap == 0 {
            self.deadline = None;
            return None;
        }
        let target = Duration::from_secs_f64(1.0 / f64::from(cap));
        let (wait_until, next) = pace_deadline(now, self.deadline, target);
        self.deadline = Some(next);
        Some(wait_until)
    }

    /// How far ahead of a deadline to stop sleeping and spin.
    pub fn sleep_margin(&self) -> Duration {
        self.margin.0
    }

    /// Record one sleep that asked for `requested` and took `actual`.
    pub fn observe_sleep(&mut self, requested: Duration, actual: Duration) {
        self.margin.observe(requested, actual);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: Duration = Duration::from_millis(100);
    const TARGET: Duration = Duration::from_millis(16);

    #[test]
    fn pace_deadline_first_frame_does_not_wait() {
        let (wait_until, next) = pace_deadline(NOW, None, TARGET);
        assert_eq!(wait_until, NOW);
        assert_eq!(next, NOW + TARGET);
    }

    // A frame that finished early holds to the previous deadline and
    // accumulates exactly from it.
    #[test]
    fn pace_deadline_accumulates_exactly_when_ahead() {
        let prev = NOW + Duration::from_millis(10);
        let (wait_until, next) = pace_deadline(NOW, Some(prev), TARGET);
        assert_eq!(wait_until, prev);
        assert_eq!(next, prev + TARGET);
    }

    // A frame that overran re-bases to `now`: no wait, and no catch-up burst.
    #[test]
    fn pace_deadline_resets_after_overrun_without_burst() {
        let past = NOW - Duration::from_millis(5);
        let (wait_until, next) = pace_deadline(NOW, Some(past), TARGET);
        assert_eq!(wait_until, NOW);
        assert_eq!(next, NOW + TARGET);
    }

    #[test]
    fn the_launch_outranks_a_benchmark_which_outranks_the_world() {
        assert_eq!(frame_cap(Some(1000), Some(0), Some(60)), 1000);
        assert_eq!(frame_cap(None, Some(0), Some(60)), 0);
        assert_eq!(frame_cap(None, None, Some(60)), 60);
        assert_eq!(frame_cap(None, None, None), 0);
    }

    #[test]
    fn a_minimized_window_runs_at_the_background_rate() {
        assert_eq!(effective_cap(0, true), MINIMIZED_FPS);
        assert_eq!(effective_cap(300, true), MINIMIZED_FPS);
        assert_eq!(effective_cap(10, true), 10, "a lower cap still wins");
        assert_eq!(effective_cap(0, false), 0);
        assert_eq!(effective_cap(300, false), 300);
    }

    // Consecutive frames accumulate deadlines; a cap change re-bases onto
    // `now`, and an uncapped frame is not held at all.
    #[test]
    fn a_cap_change_rebases_the_deadline() {
        let mut pace = FramePace::default();
        assert_eq!(pace.plan(1000, false, NOW), Some(NOW));
        let interval = Duration::from_millis(1);
        assert_eq!(pace.plan(1000, false, NOW), Some(NOW + interval));
        assert_eq!(pace.plan(500, false, NOW), Some(NOW));
        assert_eq!(pace.plan(0, false, NOW), None);
        assert_eq!(pace.plan(500, false, NOW), Some(NOW));
    }

    // Minimizing an uncapped run paces it, and restoring it lifts the pace.
    #[test]
    fn minimizing_paces_an_uncapped_run() {
        let mut pace = FramePace::default();
        assert_eq!(pace.plan(0, true, NOW), Some(NOW));
        assert_eq!(pace.last_cap, MINIMIZED_FPS);
        assert_eq!(pace.plan(0, false, NOW), None);
    }

    #[test]
    fn a_late_wake_raises_the_sleep_margin_with_headroom() {
        let mut pace = FramePace::default();
        let asked = Duration::from_millis(2);
        pace.observe_sleep(asked, asked + Duration::from_millis(2));
        assert_eq!(pace.sleep_margin(), Duration::from_millis(3));
    }

    #[test]
    fn prompt_wakes_let_the_sleep_margin_decay_to_its_floor() {
        let mut margin = SleepMargin::default();
        let asked = Duration::from_millis(2);
        margin.observe(asked, asked + Duration::from_micros(20));
        assert!(margin.0 < SleepMargin::INITIAL, "one prompt wake lowers it");
        assert!(
            margin.0 > SleepMargin::MIN,
            "but only by a step, not to the floor"
        );
        for _ in 0..500 {
            margin.observe(asked, asked);
        }
        assert!(margin.0 - SleepMargin::MIN < Duration::from_micros(1));
    }

    #[test]
    fn the_sleep_margin_never_exceeds_its_ceiling() {
        let mut margin = SleepMargin::default();
        let asked = Duration::from_millis(1);
        margin.observe(asked, asked + Duration::from_millis(20));
        assert_eq!(margin.0, SleepMargin::MAX);
    }
}
