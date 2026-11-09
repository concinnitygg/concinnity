// Which screenshots a launch takes: its own `--screenshot-at` shots, and the
// world's in a run that takes them.

use alloc::vec::Vec;

use super::LaunchRequest;
use crate::ecs::{PipelineContext, World};
use crate::presentation::screenshot::{Screenshot, ScreenshotSystem, Shot};

/// The system taking the launch's screenshots in `world`, or `None` for a run
/// with none to take.
pub(crate) fn screenshot_system(world: &World) -> Option<ScreenshotSystem> {
    let launch = world.resource::<LaunchRequest>()?;
    ScreenshotSystem::new(launch.screenshots_dir(), shots(world, launch))
}

/// Whether the run will take screenshots: the launch asks for some, or takes
/// the world's and the world has some.
pub fn takes_screenshots(ctx: &PipelineContext<'_>) -> bool {
    ctx.resource::<LaunchRequest>().is_some_and(|launch| {
        !launch.args.screenshot_at.is_empty()
            || (takes_authored(launch) && ctx.query::<Screenshot>().next().is_some())
    })
}

// Whether a launch takes the world's own screenshots.
fn takes_authored(launch: &LaunchRequest) -> bool {
    launch.args.benchmarks()
}

// The world's screenshots when the launch takes them, then the launch's.
fn shots(world: &World, launch: &LaunchRequest) -> Vec<Shot> {
    let authored = world
        .query::<Screenshot>()
        .filter(|_| takes_authored(launch))
        .map(|s| Shot {
            seconds: s.seconds,
            file: s.file.clone(),
            hud: s.hud,
        });
    let asked = launch.args.screenshot_at.iter().map(|s| Shot {
        seconds: s.seconds(),
        file: s.file.clone(),
        hud: true,
    });
    authored.chain(asked).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launch::{LaunchArgs, LaunchCommand, ScreenshotAt};
    use alloc::string::ToString;
    use alloc::vec;

    fn world(args: LaunchArgs, shots: &[(f32, &str, bool)]) -> World {
        let mut world = World::new();
        world.insert_resource(LaunchRequest {
            args,
            ..Default::default()
        });
        for &(seconds, file, hud) in shots {
            world.add_component(Screenshot {
                seconds,
                file: file.to_string(),
                hud,
            });
        }
        world
    }

    // Plain play takes none of the world's shots; a benchmark takes them, and
    // `--screenshot-at` takes its own alone.
    #[test]
    fn only_a_run_that_asks_takes_screenshots() {
        let shots = [(1.0, "a.png", true)];
        assert!(screenshot_system(&world(LaunchArgs::NONE, &shots)).is_none());
        assert!(!takes_screenshots(
            &world(LaunchArgs::NONE, &shots).context()
        ));
        assert!(screenshot_system(&World::new()).is_none());
        let benchmark = LaunchArgs {
            command: LaunchCommand::Bench,
            ..LaunchArgs::NONE
        };
        assert!(screenshot_system(&world(benchmark.clone(), &shots)).is_some());
        assert!(takes_screenshots(
            &world(benchmark.clone(), &shots).context()
        ));
        assert!(
            screenshot_system(&world(benchmark, &[])).is_none(),
            "nothing to take"
        );
        let asked = LaunchArgs {
            screenshot_at: vec![ScreenshotAt::new(2.0, "b.png")],
            ..LaunchArgs::NONE
        };
        let system = screenshot_system(&world(asked, &shots)).expect("asked for one");
        assert_eq!(system.waiting_files(), ["b.png"]);
    }

    // A benchmark's own shots join the launch's, under the launch's directory.
    #[test]
    fn a_benchmark_takes_the_worlds_shots_and_its_own() {
        let args = LaunchArgs {
            command: LaunchCommand::Bench,
            screenshots_dir: Some("shots".to_string()),
            screenshot_at: vec![ScreenshotAt::new(2.5, "asked.png")],
            ..LaunchArgs::NONE
        };
        let system = screenshot_system(&world(args, &[(1.0, "clean.png", false)])).expect("armed");
        assert_eq!(system.waiting_files(), ["clean.png", "asked.png"]);
    }
}
