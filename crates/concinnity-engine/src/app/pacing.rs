//! CPU frame pacer for the FPS cap. Runs at the runtime level, before the world
//! steps, so no system pays the sleep inside its own step time and the cap
//! applies whichever systems the world built. The cap in force is core's
//! `frame_cap`, lowered to a background rate while the `WindowMinimized`
//! resource says nothing drawn can be seen.

use concinnity_core::ecs::{WindowMinimized, World};
use concinnity_core::render::frame_submit::{FramePace, frame_cap};
use std::time::Instant;

// Holds each frame's start to the deadline core's `FramePace` plans: sleeps to
// its margin before the deadline, then spins the rest. One per `Runtime`,
// driven once per world step.
#[derive(Debug)]
pub(crate) struct FramePacer {
    origin: Instant,
    pace: FramePace,
}

impl Default for FramePacer {
    fn default() -> Self {
        Self {
            origin: Instant::now(),
            pace: FramePace::default(),
        }
    }
}

impl FramePacer {
    // Pace the upcoming world step from the world's published pacing state.
    pub(crate) fn pace(&mut self, world: &World) {
        let minimized = world.resource::<WindowMinimized>().is_some_and(|m| m.0);
        let Some(wait_until) = self
            .pace
            .plan(frame_cap(world), minimized, self.origin.elapsed())
        else {
            return;
        };
        loop {
            let now = Instant::now();
            let remaining = wait_until.saturating_sub(now.duration_since(self.origin));
            if remaining.is_zero() {
                break;
            }
            let margin = self.pace.sleep_margin();
            if remaining > margin {
                let requested = remaining - margin;
                std::thread::sleep(requested);
                self.pace.observe_sleep(requested, now.elapsed());
            } else {
                std::hint::spin_loop();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A world with no published cap (headless, or graphics failed) never
    // paces, and returns at once.
    #[test]
    fn unpublished_cap_is_unlimited() {
        let world = World::new();
        let start = Instant::now();
        FramePacer::default().pace(&world);
        assert!(start.elapsed() < std::time::Duration::from_secs(1));
    }
}
