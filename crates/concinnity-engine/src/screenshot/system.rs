// Requests each screenshot as the simulation clock reaches it.

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;

use super::Screenshot;
use super::asset::{is_contained_file, normal_file};
use super::schedule::{Shot, ShotSchedule};
use crate::benchmark::BenchmarkRun;
use crate::ecs::{
    FrameCaptureRequest, PipelineContext, SimTiming, StepResult, System, WindowMinimized, World,
};
use crate::launch::{LaunchRequest, screenshot_path};
use concinnity_core::math::narrow_f32;

/// Takes a run's screenshots: the launch's `--screenshot-at` shots, and the
/// world's [`Screenshot`]s in a run that asks for them.
#[derive(Debug)]
pub(crate) struct ScreenshotSystem {
    schedule: ShotSchedule,
    dir: String,
    // The last moment the simulation clock reached, in seconds.
    reached: f32,
}

impl ScreenshotSystem {
    /// The system for a run with screenshots to take.
    pub(crate) fn for_world(world: &World) -> Option<Self> {
        let launch = world.resource::<LaunchRequest>()?;
        let shots = shots(world, launch);
        (!shots.is_empty()).then(|| Self {
            schedule: ShotSchedule::new(shots),
            dir: String::from(launch.screenshots_dir()),
            reached: 0.0,
        })
    }
}

/// Whether the run will take screenshots: the launch asks for some, or takes
/// the world's and the world has some.
pub(crate) fn takes_screenshots(ctx: &PipelineContext<'_>) -> bool {
    ctx.resource::<LaunchRequest>().is_some_and(|launch| {
        !launch.args.screenshot_at.is_empty()
            || (takes_authored(launch) && ctx.query::<Screenshot>().next().is_some())
    })
}

// Whether a launch takes the world's own screenshots.
fn takes_authored(launch: &LaunchRequest) -> bool {
    launch.args.benchmarks()
}

// The world's screenshots when the launch takes them, then the launch's, each
// with a file inside the screenshot directory and none writing a file an
// earlier one writes.
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
    let mut files = BTreeSet::new();
    authored
        .chain(asked)
        .filter(|shot| {
            if !is_contained_file(&shot.file) {
                tracing::warn!(
                    "screenshot: {:?} is not a file inside the screenshot directory, so it is not taken",
                    shot.file
                );
                return false;
            }
            let unique = files.insert(normal_file(&shot.file));
            if !unique {
                tracing::warn!(
                    "screenshot: {:?} at {:.1}s writes the same file as another, so it is not taken",
                    shot.file,
                    shot.seconds
                );
            }
            unique
        })
        .collect()
}

impl System for ScreenshotSystem {
    fn init(&mut self, ctx: &mut PipelineContext<'_>) {
        let Some(end) = ctx
            .resource::<BenchmarkRun>()
            .map(BenchmarkRun::stop_seconds)
        else {
            return;
        };
        for shot in self.schedule.take_after(end) {
            tracing::warn!(
                "screenshot: {} is at {:.1}s, after the benchmark ends at {end:.1}s, so it will not be taken",
                shot.file,
                shot.seconds
            );
        }
    }

    fn step(&mut self, ctx: &mut PipelineContext<'_>) -> StepResult {
        let now = ctx
            .resource::<SimTiming>()
            .map_or(0.0, |t| narrow_f32(t.drawn_time()));
        self.reached = now;
        let Some(shot) = self.schedule.due(now) else {
            return StepResult::Continue;
        };
        if ctx.resource::<WindowMinimized>().is_some_and(|m| m.0) {
            tracing::warn!(
                "screenshot: the window was hidden at {:.1}s, so {} was not taken",
                shot.seconds,
                shot.file
            );
        } else {
            ctx.insert_resource(FrameCaptureRequest {
                path: screenshot_path(&self.dir, &shot.file),
                hud: shot.hud,
            });
        }
        StepResult::Continue
    }
}

impl Drop for ScreenshotSystem {
    fn drop(&mut self) {
        if let Some(line) = untaken(self.reached, self.schedule.waiting()) {
            tracing::warn!("{line}");
        }
    }
}

// One line naming the shots still waiting when the run ended at `ended`
// seconds, if any.
fn untaken<'a>(ended: f32, waiting: impl Iterator<Item = &'a Shot>) -> Option<String> {
    let files: Vec<&str> = waiting.map(|s| s.file.as_str()).collect();
    let (n, s) = match files.len() {
        0 => return None,
        1 => (1, "shot"),
        n => (n, "shots"),
    };
    Some(alloc::format!(
        "screenshot: the run ended at {ended:.1} s; {n} {s} not taken: {}",
        files.join(", ")
    ))
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

    fn at(
        world: &mut World,
        system: &mut ScreenshotSystem,
        seconds: f64,
    ) -> Option<FrameCaptureRequest> {
        world.insert_resource(SimTiming {
            time: seconds,
            ..Default::default()
        });
        system.step(&mut world.context());
        world.remove_resource::<FrameCaptureRequest>()
    }

    // The shots a run ends before name themselves on one line.
    #[test]
    fn the_untaken_shots_share_one_line() {
        let shot = |file: &str| Shot {
            seconds: 70.0,
            file: file.to_string(),
            hud: true,
        };
        assert_eq!(untaken(62.06, [].iter()), None);
        assert_eq!(
            untaken(62.06, [shot("a.png")].iter()).as_deref(),
            Some("screenshot: the run ended at 62.1 s; 1 shot not taken: a.png")
        );
        assert_eq!(
            untaken(62.06, [shot("a.png"), shot("b.png")].iter()).as_deref(),
            Some("screenshot: the run ended at 62.1 s; 2 shots not taken: a.png, b.png")
        );
    }

    // Plain play takes none of the world's shots; a benchmark takes them, and
    // `--screenshot-at` takes its own alone.
    #[test]
    fn only_a_run_that_asks_takes_screenshots() {
        let shots = [(1.0, "a.png", true)];
        assert!(ScreenshotSystem::for_world(&world(LaunchArgs::NONE, &shots)).is_none());
        assert!(ScreenshotSystem::for_world(&World::new()).is_none());
        let benchmark = LaunchArgs {
            command: LaunchCommand::Bench,
            ..LaunchArgs::NONE
        };
        assert!(ScreenshotSystem::for_world(&world(benchmark.clone(), &shots)).is_some());
        assert!(
            ScreenshotSystem::for_world(&world(benchmark, &[])).is_none(),
            "nothing to take"
        );
        let asked = LaunchArgs {
            screenshot_at: vec![ScreenshotAt::new(2.0, "b.png")],
            ..LaunchArgs::NONE
        };
        let system = ScreenshotSystem::for_world(&world(asked, &shots)).expect("asked for one");
        let files: Vec<&str> = system.schedule.waiting().map(|s| s.file.as_str()).collect();
        assert_eq!(files, ["b.png"]);
    }

    // A shot fires on the first frame the simulation clock reaches its moment,
    // under the launch's directory, with its HUD choice; the launch's own
    // shots join the world's, and one leaving the directory or writing a file
    // another writes is dropped.
    #[test]
    fn shots_fire_on_the_simulation_clock() {
        let args = LaunchArgs {
            command: LaunchCommand::Bench,
            screenshots_dir: Some("shots".to_string()),
            screenshot_at: vec![ScreenshotAt::new(2.5, "asked.png")],
            ..LaunchArgs::NONE
        };
        let mut world = world(
            args,
            &[
                (1.0, "clean.png", false),
                (1.5, "../out.png", true),
                (1.8, "./Clean.png", true),
            ],
        );
        let mut system = ScreenshotSystem::for_world(&world).expect("armed");
        system.init(&mut world.context());
        assert_eq!(at(&mut world, &mut system, 0.9), None);
        assert_eq!(
            at(&mut world, &mut system, 1.0),
            Some(FrameCaptureRequest {
                path: "shots/clean.png".to_string(),
                hud: false
            })
        );
        assert_eq!(
            at(&mut world, &mut system, 2.0),
            None,
            "the escaping one and the second clean.png are dropped"
        );
        let asked = at(&mut world, &mut system, 3.0).expect("the launch's shot");
        assert_eq!((asked.path.as_str(), asked.hud), ("shots/asked.png", true));
        assert_eq!(system.schedule.waiting().count(), 0);
    }

    // A shot waits for the moment the frame shows, not the end of its ticks.
    #[test]
    fn a_shot_fires_on_the_drawn_moment() {
        let args = LaunchArgs {
            command: LaunchCommand::Bench,
            ..LaunchArgs::NONE
        };
        let mut world = world(args, &[(1.0, "a.png", true)]);
        let mut system = ScreenshotSystem::for_world(&world).expect("armed");
        let mut drawn = |alpha| {
            world.insert_resource(SimTiming {
                time: 1.01,
                alpha,
                ..Default::default()
            });
            system.step(&mut world.context());
            world.remove_resource::<FrameCaptureRequest>()
        };
        assert_eq!(drawn(0.0), None, "the frame shows 0.993 s");
        assert!(drawn(1.0).is_some());
    }

    // A shot due while the window is hidden is skipped rather than taken late.
    #[test]
    fn a_hidden_window_skips_its_shot() {
        let args = LaunchArgs {
            command: LaunchCommand::Bench,
            ..LaunchArgs::NONE
        };
        let mut world = world(args, &[(1.0, "a.png", true)]);
        let mut system = ScreenshotSystem::for_world(&world).expect("armed");
        world.insert_resource(WindowMinimized(true));
        assert_eq!(at(&mut world, &mut system, 1.0), None);
        world.insert_resource(WindowMinimized(false));
        assert_eq!(at(&mut world, &mut system, 2.0), None);
    }
}
