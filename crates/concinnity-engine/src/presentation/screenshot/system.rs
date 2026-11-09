// Requests each screenshot as the simulation clock reaches it.

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;

use super::asset::{is_contained_file, normal_file, screenshot_path};
use super::schedule::{Shot, ShotSchedule};
use crate::benchmark::BenchmarkRun;
use crate::ecs::{PipelineContext, StepResult, System};
use crate::protocol::{FrameCaptureRequest, SimTiming, WindowMinimized};
use concinnity_core::math::narrow_f32;

/// Takes a run's screenshots as the simulation clock reaches each one.
#[derive(Debug)]
pub(crate) struct ScreenshotSystem {
    schedule: ShotSchedule,
    dir: String,
    // The last moment the simulation clock reached, in seconds.
    reached: f32,
}

impl ScreenshotSystem {
    /// The system taking `shots` into `dir`, or `None` when none of them can
    /// be taken. A shot whose file leaves the directory, or writes the file an
    /// earlier one writes, is dropped with a warning.
    pub(crate) fn new(dir: &str, shots: impl IntoIterator<Item = Shot>) -> Option<Self> {
        let shots = takeable(shots);
        (!shots.is_empty()).then(|| Self {
            schedule: ShotSchedule::new(shots),
            dir: String::from(dir),
            reached: 0.0,
        })
    }

    /// The files of the shots not yet taken, earliest first.
    #[cfg(test)]
    pub(crate) fn waiting_files(&self) -> Vec<&str> {
        self.schedule.waiting().map(|s| s.file.as_str()).collect()
    }
}

// The shots that can be taken, each with a file inside the screenshot
// directory and none writing a file an earlier one writes.
fn takeable(shots: impl IntoIterator<Item = Shot>) -> Vec<Shot> {
    let mut files = BTreeSet::new();
    shots
        .into_iter()
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
    use crate::ecs::World;
    use alloc::string::ToString;

    fn shot(seconds: f32, file: &str, hud: bool) -> Shot {
        Shot {
            seconds,
            file: file.to_string(),
            hud,
        }
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
        assert_eq!(untaken(62.06, [].iter()), None);
        assert_eq!(
            untaken(62.06, [shot(70.0, "a.png", true)].iter()).as_deref(),
            Some("screenshot: the run ended at 62.1 s; 1 shot not taken: a.png")
        );
        assert_eq!(
            untaken(
                62.06,
                [shot(70.0, "a.png", true), shot(70.0, "b.png", true)].iter()
            )
            .as_deref(),
            Some("screenshot: the run ended at 62.1 s; 2 shots not taken: a.png, b.png")
        );
    }

    // A shot fires on the first frame the simulation clock reaches its moment,
    // under the directory, with its HUD choice; one leaving the directory or
    // writing a file another writes is dropped, and none left takes nothing.
    #[test]
    fn shots_fire_on_the_simulation_clock() {
        assert!(ScreenshotSystem::new("shots", [shot(1.0, "../out.png", true)]).is_none());
        let mut world = World::new();
        let mut system = ScreenshotSystem::new(
            "shots",
            [
                shot(1.0, "clean.png", false),
                shot(1.5, "../out.png", true),
                shot(1.8, "./Clean.png", true),
                shot(2.5, "later.png", true),
            ],
        )
        .expect("armed");
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
        let later = at(&mut world, &mut system, 3.0).expect("the later shot");
        assert_eq!((later.path.as_str(), later.hud), ("shots/later.png", true));
        assert!(system.waiting_files().is_empty());
    }

    // A shot waits for the moment the frame shows, not the end of its ticks.
    #[test]
    fn a_shot_fires_on_the_drawn_moment() {
        let mut world = World::new();
        let mut system = ScreenshotSystem::new("", [shot(1.0, "a.png", true)]).expect("armed");
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
        let mut world = World::new();
        let mut system = ScreenshotSystem::new("", [shot(1.0, "a.png", true)]).expect("armed");
        world.insert_resource(WindowMinimized(true));
        assert_eq!(at(&mut world, &mut system, 1.0), None);
        world.insert_resource(WindowMinimized(false));
        assert_eq!(at(&mut world, &mut system, 2.0), None);
    }
}
