//! CPU frame pacer for the FPS cap. Runs at the runtime level, before the world
//! steps, so no system pays the sleep inside its own step time and the cap
//! applies whichever systems the world built. The cap value comes from the
//! `FrameRateCap` resource (published by GraphicsSystem from FramePacing +
//! the live settings row), under a benchmarked run's own cap and `--fps-cap`
//! over both, lowered to a background rate while the
//! `WindowMinimized` resource says nothing drawn can be seen.

use crate::app::run::LaunchRequest;
use concinnity_core::benchmark::BenchmarkRun;
use concinnity_core::ecs::{FrameRateCap, WindowMinimized, World};
use concinnity_core::render::frame_submit::FramePace;
use std::time::Instant;

// The cap in force, from the world's published pacing state.
fn frame_cap(world: &World) -> u32 {
    concinnity_core::render::frame_submit::frame_cap(
        world
            .resource::<LaunchRequest>()
            .and_then(|launch| launch.args.fps_cap),
        world
            .resource::<BenchmarkRun>()
            .map(|run| run.config.fps_cap),
        world.resource::<FrameRateCap>().map(|c| c.0),
    )
}

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

    // An open menu runs at the published cap like any other frame.
    #[test]
    fn an_open_menu_follows_the_published_cap() {
        let mut world = World::new();
        world.insert_resource(concinnity_core::ecs::MenuActive(true));
        world.insert_resource(FrameRateCap(0));
        assert_eq!(frame_cap(&world), 0);
        world.insert_resource(FrameRateCap(1000));
        assert_eq!(frame_cap(&world), 1000);
    }

    // A benchmarked run holds its own cap over the published one.
    #[test]
    fn a_benchmarked_run_holds_its_own_cap() {
        use concinnity_core::components::BenchmarkConfig;
        let mut world = World::new();
        world.insert_resource(FrameRateCap(60));
        world.insert_resource(BenchmarkRun::new(BenchmarkConfig::default()));
        assert_eq!(frame_cap(&world), 0);
    }

    // `--fps-cap` holds over a benchmark's cap and the published one.
    #[test]
    fn the_launchs_cap_holds_over_a_benchmarks() {
        use concinnity_core::components::BenchmarkConfig;
        use concinnity_core::launch::LaunchArgs;
        let mut world = World::new();
        world.insert_resource(FrameRateCap(60));
        world.insert_resource(BenchmarkRun::new(BenchmarkConfig::default()));
        world.insert_resource(LaunchRequest {
            args: LaunchArgs {
                fps_cap: Some(1000),
                ..LaunchArgs::NONE
            },
            ..Default::default()
        });
        assert_eq!(frame_cap(&world), 1000);
    }

    // A world with no published cap (headless, or graphics failed) never
    // paces, and returns at once.
    #[test]
    fn unpublished_cap_is_unlimited() {
        let world = World::new();
        assert_eq!(frame_cap(&world), 0);
        let start = Instant::now();
        FramePacer::default().pace(&world);
        assert!(start.elapsed() < std::time::Duration::from_secs(1));
    }
}
