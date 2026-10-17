//! CPU frame pacer for the FPS cap. Runs at the runtime level, before the world
//! steps, so no system pays the sleep inside its own step time and the cap
//! applies whichever systems the world built. The cap value comes from the
//! `FrameRateCap` resource (published by GraphicsSystem from FramePacing +
//! the live settings row), or a benchmarked run's own cap, lowered to a background rate while the
//! `WindowMinimized` resource says nothing drawn can be seen.

use concinnity_core::benchmark::BenchmarkRun;
use concinnity_core::ecs::{FrameRateCap, SimTiming, WindowMinimized, World};
use std::time::{Duration, Instant};

// The rate a minimized window is held to whatever the cap: few enough frames
// to idle the GPU, enough that every frame's simulation ticks fit under the
// per-frame tick limit, so the world keeps running at full speed behind it.
const MINIMIZED_FPS: u32 = 30;
const _: () =
    assert!(MINIMIZED_FPS * SimTiming::MAX_TICKS_PER_FRAME >= (1.0 / SimTiming::TICK_DT) as u32);

// The cap the pacer holds this frame to: the published one, lowered to
// `MINIMIZED_FPS` while the window is minimized (0 is uncapped).
fn effective_cap(cap: u32, minimized: bool) -> u32 {
    match (minimized, cap) {
        (false, cap) => cap,
        (true, 0) => MINIMIZED_FPS,
        (true, cap) => cap.min(MINIMIZED_FPS),
    }
}

// Frame-pacing math for the FPS cap, split from the sleep so the drift / no-burst
// logic is unit-testable. Given the current time, the previously scheduled
// deadline (if any), and the target frame interval, returns `(wait_until, next)`:
// the deadline to hold this frame's start to, and the deadline to store for the
// next frame. When the previous deadline is still in the future the cap is met by
// waiting to it and accumulating exactly (no drift); when it has already passed
// (a frame that overran the budget) it resets to `now` so a slow frame never
// triggers a catch-up burst of un-paced frames.
fn pace_deadline(now: Instant, prev: Option<Instant>, target: Duration) -> (Instant, Instant) {
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

// Holds each frame's start to the target interval so the loop runs at most
// `FrameRateCap` frames a second. One per `Runtime`, driven once per world step.
#[derive(Debug, Default)]
pub(crate) struct FramePacer {
    // The pacer's running target for the next frame's start.
    deadline: Option<Instant>,
    // The cap the deadline was accumulated under. A cap change re-bases
    // the pacer (clears the deadline) so switching caps never leaves one stale
    // long wait, mirroring the old in-step rebase on the settings change.
    last_cap: u32,
    margin: SleepMargin,
}

impl FramePacer {
    // Pace the upcoming world step from the world's published pacing state.
    // No-op (and deadline cleared) while no cap is in force.
    pub(crate) fn pace(&mut self, world: &World) {
        let cap = world
            .resource::<BenchmarkRun>()
            .map(|run| run.config.fps_cap)
            .or_else(|| world.resource::<FrameRateCap>().map(|c| c.0))
            .unwrap_or(0);
        let minimized = world.resource::<WindowMinimized>().is_some_and(|m| m.0);
        let cap = effective_cap(cap, minimized);
        if cap != self.last_cap {
            self.deadline = None;
            self.last_cap = cap;
        }
        if cap == 0 {
            self.deadline = None;
            return;
        }
        let target = Duration::from_secs_f64(1.0 / cap as f64);
        let (wait_until, next) = pace_deadline(Instant::now(), self.deadline, target);
        // Sleep to the learned margin before the deadline, then spin the rest.
        loop {
            let now = Instant::now();
            let remaining = wait_until.saturating_duration_since(now);
            if remaining.is_zero() {
                break;
            }
            if remaining > self.margin.0 {
                let requested = remaining - self.margin.0;
                std::thread::sleep(requested);
                self.margin.observe(requested, now.elapsed());
            } else {
                std::hint::spin_loop();
            }
        }
        self.deadline = Some(next);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pace_deadline_first_frame_does_not_wait() {
        // No previous deadline: wait until now (no sleep), schedule now + target.
        let now = Instant::now();
        let target = Duration::from_millis(16);
        let (wait_until, next) = pace_deadline(now, None, target);
        assert_eq!(wait_until, now);
        assert_eq!(next, now + target);
    }

    #[test]
    fn pace_deadline_accumulates_exactly_when_ahead() {
        // A frame finished early: the previous deadline is still in the future,
        // so hold to it and accumulate exactly (no drift from `now`).
        let now = Instant::now();
        let target = Duration::from_millis(16);
        let prev = now + Duration::from_millis(10);
        let (wait_until, next) = pace_deadline(now, Some(prev), target);
        assert_eq!(wait_until, prev);
        assert_eq!(next, prev + target);
    }

    #[test]
    fn pace_deadline_resets_after_overrun_without_burst() {
        // The frame overran: the previous deadline is in the past, so re-base to
        // `now` (no wait this frame) and schedule from now -- never a catch-up
        // burst of un-paced frames.
        let now = Instant::now();
        let target = Duration::from_millis(16);
        let past = now - Duration::from_millis(5);
        let (wait_until, next) = pace_deadline(now, Some(past), target);
        assert_eq!(wait_until, now);
        assert_eq!(next, now + target);
    }

    // An open menu runs at the published cap like any other frame: with no
    // cap it stays unpaced.
    #[test]
    fn an_open_menu_follows_the_published_cap() {
        let mut world = World::new();
        world.insert_resource(concinnity_core::ecs::MenuActive(true));
        world.insert_resource(FrameRateCap(0));
        let mut pacer = FramePacer::default();
        pacer.pace(&world);
        assert!(pacer.deadline.is_none(), "an uncapped menu is not paced");
        world.insert_resource(FrameRateCap(1000));
        pacer.pace(&world);
        assert_eq!(pacer.last_cap, 1000, "the menu paces at the cap itself");
    }

    // A cap change re-bases the pacer: the accumulated deadline is dropped so
    // the new cap takes over from `now` rather than waiting out a stale long
    // interval.
    #[test]
    fn cap_change_rebases_the_deadline() {
        let mut world = World::new();
        world.insert_resource(FrameRateCap(1000));
        let mut pacer = FramePacer::default();
        pacer.pace(&world);
        assert!(
            pacer.deadline.is_some(),
            "a positive cap schedules a deadline"
        );
        world.insert_resource(FrameRateCap(0));
        pacer.pace(&world);
        assert!(pacer.deadline.is_none(), "cap 0 clears the schedule");
    }

    // A benchmarked run holds its own cap over the published one.
    #[test]
    fn a_benchmarked_run_holds_its_own_cap() {
        use concinnity_core::components::BenchmarkConfig;
        let mut world = World::new();
        world.insert_resource(FrameRateCap(60));
        world.insert_resource(BenchmarkRun::new(BenchmarkConfig::default()));
        let mut pacer = FramePacer::default();
        pacer.pace(&world);
        assert_eq!(pacer.last_cap, 0);
        assert!(pacer.deadline.is_none());
    }

    // A world with no published cap (headless, or graphics failed) never
    // paces.
    #[test]
    fn unpublished_cap_is_unlimited() {
        let world = World::new();
        let mut pacer = FramePacer::default();
        pacer.pace(&world);
        assert!(pacer.deadline.is_none());
    }

    #[test]
    fn a_minimized_window_runs_at_the_background_rate() {
        assert_eq!(effective_cap(0, true), MINIMIZED_FPS);
        assert_eq!(effective_cap(300, true), MINIMIZED_FPS);
        assert_eq!(effective_cap(10, true), 10, "a lower cap still wins");
        assert_eq!(effective_cap(0, false), 0);
        assert_eq!(effective_cap(300, false), 300);
    }

    // Minimizing an uncapped run paces it, and restoring it lifts the pace.
    #[test]
    fn minimizing_paces_an_uncapped_run() {
        let mut world = World::new();
        world.insert_resource(FrameRateCap(0));
        world.insert_resource(WindowMinimized(true));
        let mut pacer = FramePacer::default();
        pacer.pace(&world);
        assert_eq!(pacer.last_cap, MINIMIZED_FPS);
        assert!(pacer.deadline.is_some());
        world.insert_resource(WindowMinimized(false));
        pacer.pace(&world);
        assert!(pacer.deadline.is_none());
    }

    #[test]
    fn a_late_wake_raises_the_sleep_margin_with_headroom() {
        let mut margin = SleepMargin::default();
        let asked = Duration::from_millis(2);
        margin.observe(asked, asked + Duration::from_millis(2));
        assert_eq!(margin.0, Duration::from_millis(3));
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
