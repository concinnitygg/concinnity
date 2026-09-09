// The system a declared FrameReport gates in: records one row per frame, and
// reports what the run cost when it ends.

use std::time::Instant;

use concinnity_core::camera_track::CameraTrackStatus;

use crate::components::{CameraTrack, FrameReport};
use crate::ecs::{PipelineContext, StepResult, System};
use crate::frame_report::reduce::{ReduceOptions, Report};
use crate::frame_report::report::to_text;
use crate::frame_report::sample::{FrameRun, FrameSample};

// Frames a one-minute run at 60Hz would produce. Reserved up front so the
// sample buffer does not grow inside the window it is measuring.
const RESERVED_FRAMES: usize = 60 * 60;

const MICROS_PER_MILLI: f32 = 1_000.0;

/// Records one [`FrameSample`] per frame and prints what the run cost when it
/// ends.
///
/// Runs in the `Late` phase after every engine system, so the render stats and
/// system timings it reads belong to the frame it is recording. The report
/// goes out on the frame the world reports the measurement complete; a run cut
/// short before that reports on the way out instead, and says so.
#[derive(Debug)]
pub struct FrameReportSystem {
    run: FrameRun,
    options: ReduceOptions,
    stop_when_complete: bool,
    // Set once the report has gone out, so the teardown path does not repeat
    // it.
    reported: bool,
    // Wall clock of the previous frame. The first frame has no predecessor to
    // measure against, so it is not recorded.
    previous: Option<Instant>,
    // Fallback clock for a world that publishes no run position of its own, so
    // the warm-up discard and the time column still mean something.
    started: Option<Instant>,
}

impl FrameReportSystem {
    /// The system for a world's declared report.
    pub fn new(report: &FrameReport) -> Self {
        Self {
            run: FrameRun::default(),
            options: ReduceOptions {
                warmup_seconds: report.warmup_seconds,
                budget_us: (report.budget_ms * MICROS_PER_MILLI) as u32,
            },
            stop_when_complete: report.stop_when_complete,
            reported: false,
            previous: None,
            started: None,
        }
    }

    /// Everything recorded so far. What the report is reduced from.
    pub fn run(&self) -> &FrameRun {
        &self.run
    }

    /// Reduce what the run recorded and print it, once. A run with nothing
    /// past the warm-up says that instead of reporting a page of zeroes.
    pub fn emit(&mut self) {
        if self.reported {
            return;
        }
        self.reported = true;
        match Report::of(&self.run, self.options) {
            Some(report) => print!("{}", to_text(&report)),
            None => println!(
                "frame report: no frames past the {:.1}s warm-up, so there is nothing to measure",
                self.options.warmup_seconds,
            ),
        }
    }
}

impl System for FrameReportSystem {
    fn init(&mut self, ctx: &mut PipelineContext) {
        self.run.segments = ctx
            .query::<CameraTrack>()
            .next()
            .map(|track| track.segments.clone())
            .unwrap_or_default();
        self.run.samples.reserve(RESERVED_FRAMES);
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        let now = Instant::now();
        let started = *self.started.get_or_insert(now);
        let Some(previous) = self.previous.replace(now) else {
            return StepResult::Continue;
        };
        let frame_us = now
            .duration_since(previous)
            .as_micros()
            .min(u32::MAX.into()) as u32;

        let status = ctx.resource::<CameraTrackStatus>().copied();
        let run_seconds = status.map_or_else(
            || now.duration_since(started).as_secs_f32(),
            |s| s.elapsed_seconds,
        );
        let render = ctx.profile.render;
        let systems = ctx.profile.system_timings();

        self.run.capture_pass_names(&render);
        self.run.capture_system_names(systems);
        self.run.samples.push(FrameSample::new(
            run_seconds,
            status.and_then(|s| s.segment),
            frame_us,
            &render,
            systems,
        ));

        if status.is_some_and(|s| s.finished) && self.stop_when_complete {
            self.run.completed = true;
            self.emit();
            return StepResult::Stop;
        }
        StepResult::Continue
    }
}

impl Drop for FrameReportSystem {
    // A run that closed before its track finished still has something to say,
    // and the report it prints is marked incomplete rather than passed off as
    // a whole one.
    fn drop(&mut self) {
        self.emit();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::cook::{Camera3D as Camera3DArgs, CameraTrack as CameraTrackArgs};
    use crate::components::{Camera3D, CameraTravel};
    use crate::ecs::World;
    use concinnity_core::camera_track::CameraTrackSystem;

    fn track(travel: Vec<CameraTravel>) -> CameraTrack {
        CameraTrack::bake(CameraTrackArgs {
            travel,
            ..Default::default()
        })
    }

    fn leg(distance: f32, speed: f32, segment: &str) -> CameraTravel {
        CameraTravel {
            direction: [1.0, 0.0, 0.0],
            distance,
            speed,
            segment: segment.to_string(),
            ..Default::default()
        }
    }

    // A world running its camera track beside a report system, stepped by hand.
    // The report is never emitted here: every test drops the system through
    // `forget_report`, which marks it reported first.
    fn world_with(track: CameraTrack) -> (World, CameraTrackSystem, FrameReportSystem) {
        let camera = CameraTrackSystem::new(&track);
        let mut world = World::new();
        world.add_component(Camera3D::bake(Camera3DArgs {
            controller: None,
            ..Default::default()
        }));
        world.add_component(track);
        let system = FrameReportSystem::new(&FrameReport {
            warmup_seconds: 0.0,
            ..Default::default()
        });
        (world, camera, system)
    }

    // Suppress the teardown report, which would otherwise print over the test
    // runner's output.
    fn forget_report(mut system: FrameReportSystem) {
        system.reported = true;
    }

    fn step(world: &mut World, camera: &mut CameraTrackSystem, system: &mut FrameReportSystem) {
        camera.step(&mut world.context());
        system.step(&mut world.context());
    }

    #[test]
    fn the_declared_budget_and_discard_reach_the_reduction() {
        let system = FrameReportSystem::new(&FrameReport {
            warmup_seconds: 4.5,
            budget_ms: 8.0,
            stop_when_complete: false,
            ..Default::default()
        });
        assert_eq!(system.options.warmup_seconds, 4.5);
        assert_eq!(system.options.budget_us, 8_000);
        assert!(!system.stop_when_complete);
        forget_report(system);
    }

    #[test]
    fn the_first_frame_is_not_recorded_because_it_has_nothing_to_be_timed_against() {
        let (mut world, mut camera, mut system) = world_with(track(vec![]));
        system.init(&mut world.context());
        step(&mut world, &mut camera, &mut system);
        assert!(system.run().samples.is_empty());
        step(&mut world, &mut camera, &mut system);
        assert_eq!(system.run().samples.len(), 1);
        forget_report(system);
    }

    #[test]
    fn the_tracks_segment_names_are_captured_at_init() {
        let (mut world, _, mut system) = world_with(track(vec![
            leg(4.0, 2.0, "approach"),
            leg(4.0, 2.0, "rays"),
        ]));
        system.init(&mut world.context());
        assert_eq!(system.run().segments, ["approach", "rays"]);
        forget_report(system);
    }

    #[test]
    fn a_world_with_no_track_still_samples_against_its_own_clock() {
        let mut world = World::new();
        let mut system = FrameReportSystem::new(&FrameReport::default());
        system.init(&mut world.context());
        system.step(&mut world.context());
        system.step(&mut world.context());
        assert_eq!(system.run().samples.len(), 1);
        assert!(system.run().segments.is_empty());
        assert_eq!(system.run().samples[0].segment, None);
        assert!(!system.run().completed);
        forget_report(system);
    }

    #[test]
    fn samples_carry_the_segment_the_frame_was_drawn_in() {
        let (mut world, mut camera, mut system) =
            world_with(track(vec![leg(60.0, 1.0, "approach")]));
        camera.init(&mut world.context());
        system.init(&mut world.context());
        for _ in 0..3 {
            step(&mut world, &mut camera, &mut system);
        }
        let run = system.run();
        assert!(run.samples.iter().all(|s| s.segment == Some(0)));
        // Track time advances on the fixed step, so successive samples differ.
        assert!(run.samples[1].run_seconds > run.samples[0].run_seconds);
        forget_report(system);
    }

    #[test]
    fn reaching_the_end_of_the_track_stops_the_world_and_marks_the_run_complete() {
        // One tick of track, so a later frame is already past its end.
        let (mut world, mut camera, mut system) = world_with(track(vec![leg(1.0, 60.0, "brief")]));
        camera.init(&mut world.context());
        system.init(&mut world.context());
        camera.step(&mut world.context());
        system.step(&mut world.context());

        camera.step(&mut world.context());
        assert_eq!(system.step(&mut world.context()), StepResult::Stop);
        assert!(system.run().completed);
        // The report went out with the stop, so teardown adds nothing.
        assert!(system.reported);
    }

    #[test]
    fn a_world_that_declines_to_stop_keeps_running_past_the_end_of_its_track() {
        let mut system = FrameReportSystem::new(&FrameReport {
            warmup_seconds: 0.0,
            stop_when_complete: false,
            ..Default::default()
        });
        let t = track(vec![leg(1.0, 60.0, "brief")]);
        let mut camera = CameraTrackSystem::new(&t);
        let mut world = World::new();
        world.add_component(Camera3D::bake(Camera3DArgs {
            controller: None,
            ..Default::default()
        }));
        world.add_component(t);
        camera.init(&mut world.context());
        system.init(&mut world.context());
        for _ in 0..4 {
            camera.step(&mut world.context());
            assert_eq!(system.step(&mut world.context()), StepResult::Continue);
        }
        // Still measuring, and the run is not claimed to be a whole one.
        assert!(!system.run().completed);
        assert!(system.run().samples.len() >= 3);
        forget_report(system);
    }

    #[test]
    fn the_report_goes_out_once_however_many_times_it_is_asked_for() {
        let mut system = FrameReportSystem::new(&FrameReport::default());
        system.emit();
        assert!(system.reported);
        // A second call is the teardown path meeting a run that already
        // reported; it must not print the whole thing again.
        system.emit();
        assert!(system.reported);
    }
}
