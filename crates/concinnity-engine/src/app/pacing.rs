// src/app/pacing.rs
//
// CPU frame pacer for the FPS cap. Runs at the App level, before the world
// steps, so no system pays the sleep inside its own step time and the cap
// applies whichever systems the world built. The cap value comes from the
// `FrameRateCap` resource (published by GraphicsSystem from GraphicsConfig +
// the live settings row); the menu clamp reads the previous frame's
// `MenuActive` resource, which is exactly the one-frame-lagged view the
// in-step pacer used to read from its own field.

use crate::ecs::World;
use std::time::{Duration, Instant};

// Frame-rate ceiling while a menu view is open. A paused menu does not benefit
// from a high refresh rate, and the world render is skipped behind an opaque
// menu anyway, so the pacer holds the loop at this rate (clamping down only).
const MENU_FPS_CAP: u32 = 60;

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

// The cap to pace this frame at: the user cap, clamped down to `MENU_FPS_CAP`
// while a menu view is open (never up -- a user cap already below it stands).
// `0` = unlimited.
fn effective_cap(user_cap: u32, menu_active: bool) -> u32 {
    if !menu_active {
        return user_cap;
    }
    match user_cap {
        0 => MENU_FPS_CAP,
        cap => cap.min(MENU_FPS_CAP),
    }
}

// Holds each frame's start to the target interval so the loop runs at most
// `FrameRateCap` frames a second. One per `App`, driven once per world step.
#[derive(Debug, Default)]
pub struct FramePacer {
    // The pacer's running target for the next frame's start.
    deadline: Option<Instant>,
    // The user cap the deadline was accumulated under. A cap change re-bases
    // the pacer (clears the deadline) so switching caps never leaves one stale
    // long wait, mirroring the old in-step rebase on the settings change.
    last_cap: u32,
}

impl FramePacer {
    // Pace the upcoming world step from the world's published pacing state.
    // No-op (and deadline cleared) while no cap is published or the cap is 0.
    pub fn pace(&mut self, world: &World) {
        let user_cap = world
            .resource::<crate::ecs::FrameRateCap>()
            .map(|c| c.0)
            .unwrap_or(0);
        if user_cap != self.last_cap {
            self.deadline = None;
            self.last_cap = user_cap;
        }
        // The menu state published on the previous step; a menu opening this
        // step is clamped one frame later, which is imperceptible.
        let menu_active = world
            .resource::<crate::ecs::MenuActive>()
            .map(|m| m.0)
            .unwrap_or(false);
        let cap = effective_cap(user_cap, menu_active);
        if cap == 0 {
            self.deadline = None;
            return;
        }
        let target = Duration::from_secs_f64(1.0 / cap as f64);
        let (wait_until, next) = pace_deadline(Instant::now(), self.deadline, target);
        // Coarse-sleep to ~1ms before the deadline, then spin for
        // sub-millisecond precision.
        while let Some(remaining) = wait_until.checked_duration_since(Instant::now()) {
            if remaining > Duration::from_millis(1) {
                std::thread::sleep(remaining - Duration::from_millis(1));
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

    // An open menu clamps the cap down to MENU_FPS_CAP; a user cap already
    // below the clamp stands, and 0 (unlimited) becomes the clamp itself.
    #[test]
    fn menu_clamps_the_cap_down_only() {
        assert_eq!(effective_cap(0, false), 0);
        assert_eq!(effective_cap(144, false), 144);
        assert_eq!(effective_cap(0, true), MENU_FPS_CAP);
        assert_eq!(effective_cap(144, true), MENU_FPS_CAP);
        assert_eq!(effective_cap(30, true), 30);
    }

    // A cap change re-bases the pacer: the accumulated deadline is dropped so
    // the new cap takes over from `now` rather than waiting out a stale long
    // interval.
    #[test]
    fn cap_change_rebases_the_deadline() {
        let mut world = World::new_empty();
        world.insert_resource(crate::ecs::FrameRateCap(1000));
        let mut pacer = FramePacer::default();
        pacer.pace(&world);
        assert!(
            pacer.deadline.is_some(),
            "a positive cap schedules a deadline"
        );
        world.insert_resource(crate::ecs::FrameRateCap(0));
        pacer.pace(&world);
        assert!(pacer.deadline.is_none(), "cap 0 clears the schedule");
    }

    // A world with no published cap (headless, or graphics failed) never
    // paces.
    #[test]
    fn unpublished_cap_is_unlimited() {
        let world = World::new_empty();
        let mut pacer = FramePacer::default();
        pacer.pace(&world);
        assert!(pacer.deadline.is_none());
    }
}
