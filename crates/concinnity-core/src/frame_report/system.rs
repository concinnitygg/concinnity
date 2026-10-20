// The system a benchmarked run gates in: records one row per frame, and
// reports what the run cost when it ends.

use crate::ecs::{
    Clock, FrameCaptureRequest, FrameTime, PipelineContext, StepResult, System, WindowMinimized,
    World,
};
use crate::math::{SaturatingFrom, narrow_f32, round, round_f64};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::{self, Write};

use crate::benchmark::BenchmarkRun;
use crate::camera_track::CameraTrackStatus;
use crate::components::FrameCapture;
use crate::planet::FrameRebases;
use crate::profile::RenderStats;

use crate::frame_report::capture::CaptureSchedule;
use crate::frame_report::hidden::HiddenFrames;
use crate::frame_report::reduce::{ReduceOptions, Report};
use crate::frame_report::report::write_report;
use crate::frame_report::sample::{FrameRun, FrameSample, FrameTiming};
use crate::frame_report::shown::ShownFrames;

const MICROS_PER_MILLI: f32 = 1_000.0;

/// Records one `FrameSample` per frame and writes what the run cost to its
/// output when it ends.
///
/// The fallback clocks read the host's [`Clock`] resource; a world without one
/// reads zero elapsed time.
///
/// Runs in the `PostRender` phase after every engine system, so the render
/// stats it reads were published this step. The system timings are not: the
/// profile rotates them in at the start of the next step, so each frame is
/// held for one step and recorded with its own timings then. The report goes out on the
/// frame the run's last phase ends, which leaves that last frame unrecorded; a
/// run cut short before that reports on the way out instead, and says so.
///
/// A declared capture is requested on the step the run reaches its moment, so
/// the next frame drawn, one step on, is the one saved. That frame and the few
/// after it are not recorded.
///
/// A frame run while the window is minimized or hidden draws nothing, so it
/// and the few after the window shows again are not recorded either. The
/// report says how many there were, and a capture due among them is skipped
/// rather than saving whatever was last shown.
#[derive(Debug)]
pub struct FrameReportSystem {
    run: FrameRun,
    // The frame the previous step measured, waiting for its system timings.
    pending: Option<PendingFrame>,
    benchmark: BenchmarkRun,
    // Where the report text goes.
    out: fn(&str),
    // Set once the report has gone out, so the teardown path does not repeat
    // it.
    reported: bool,
    // Host-clock micros of the previous frame. The first frame has no
    // predecessor to measure against, so it is not recorded.
    previous: Option<u64>,
    // Fallback clock for a world that publishes no run position of its own, so
    // the warm-up discard and the time column still mean something.
    started: Option<u64>,
    // The camera pose time and frame-move count the previous step saw.
    previous_pose_seconds: f32,
    previous_rebases: u64,
    // Each tick's pose, paired with its frame's GPU completion once it lands.
    shown: ShownFrames,
    captures: CaptureSchedule,
    hidden: HiddenFrames,
    // Captures that came due while the window could not show them.
    hidden_captures: Vec<FrameCapture>,
}

// What one step knows about its frame before the profile rotates in the
// step's system timings.
#[derive(Debug, Clone, Copy)]
struct PendingFrame {
    timing: FrameTiming,
    render: RenderStats,
}

impl FrameReportSystem {
    /// The system for a benchmarked run, writing its report through `out`.
    pub fn for_world(world: &World, out: fn(&str)) -> Option<Self> {
        world
            .resource::<BenchmarkRun>()
            .map(|run| Self::new(run, out))
    }

    /// The system that measures `benchmark` and writes its report to `out`.
    pub fn new(benchmark: &BenchmarkRun, out: fn(&str)) -> Self {
        let config = &benchmark.config;
        let options = ReduceOptions {
            warmup_seconds: config.warmup_seconds,
            budget_us: u32::saturating_from(config.budget_ms * MICROS_PER_MILLI),
        };
        Self {
            run: FrameRun::new(benchmark.phase_names().to_vec(), options),
            pending: None,
            benchmark: benchmark.clone(),
            out,
            reported: false,
            previous: None,
            started: None,
            previous_pose_seconds: 0.0,
            previous_rebases: 0,
            shown: ShownFrames::default(),
            captures: CaptureSchedule::new(&config.captures),
            hidden: HiddenFrames::default(),
            hidden_captures: Vec::new(),
        }
    }

    // Everything recorded so far. What the report is reduced from.
    #[cfg(test)]
    fn run(&self) -> &FrameRun {
        &self.run
    }

    // Reduce what the run recorded and write it out, once. A run with nothing
    // past the warm-up says that instead of reporting a page of zeroes.
    fn emit(&mut self) {
        if self.reported {
            return;
        }
        self.reported = true;
        (self.out)(&self.text());
    }

    // The whole report: the reduction, then a line for each thing it left out.
    fn text(&self) -> String {
        fmt::from_fn(|out| self.write_text(out)).to_string()
    }

    fn write_text(&self, text: &mut impl Write) -> fmt::Result {
        match Report::of(&self.run) {
            Some(report) => write_report(text, &report)?,
            None => writeln!(
                text,
                "frame report: no frames past the {:.1}s warm-up, so there is nothing to measure",
                self.run.options.warmup_seconds,
            )?,
        }
        if self.hidden.total() > 0 {
            let phases: Vec<String> = self
                .hidden
                .by_phase()
                .map(|(phase, count)| format!("{} {count}", self.run.phase_name(phase)))
                .collect();
            writeln!(
                text,
                "frame report: the window was hidden, so {} frame(s) past the warm-up ({}) were left out of the measurement",
                self.hidden.total(),
                phases.join(", "),
            )?;
        }
        for skipped in &self.hidden_captures {
            writeln!(
                text,
                "frame report: the window was hidden {:.1}s after the warm-up, so {} was not captured",
                skipped.seconds, skipped.path,
            )?;
        }
        for missed in self.captures.missed() {
            writeln!(
                text,
                "frame report: the run ended before {:.1}s after the warm-up, so {} was not captured",
                missed.seconds, missed.path,
            )?;
        }
        Ok(())
    }
}

impl System for FrameReportSystem {
    fn step(&mut self, ctx: &mut PipelineContext<'_>) -> StepResult {
        let now = ctx.resource::<Clock>().map_or(0, Clock::read);
        let started = *self.started.get_or_insert(now);
        let status = ctx.resource::<CameraTrackStatus>().copied();
        let run_seconds = status.map_or_else(
            || narrow_f32(now.saturating_sub(started) as f64 * 1e-6),
            |s| s.elapsed_seconds,
        );
        let rebases = ctx
            .resource::<FrameRebases>()
            .map_or(0, FrameRebases::count);
        let pose_seconds = status.map_or(run_seconds, |s| s.pose_seconds);
        let run_step_us = u32::saturating_from(round(
            (pose_seconds - self.previous_pose_seconds).max(0.0) * 1_000_000.0,
        ));
        let rebased = rebases != self.previous_rebases;
        self.previous_pose_seconds = pose_seconds;
        self.previous_rebases = rebases;
        if let Some(sampled) = ctx.resource::<crate::ecs::InputSampledAt>() {
            self.shown.posed(sampled.0, pose_seconds);
        }
        let shown = ctx
            .profile
            .render
            .gpu_done
            .and_then(|done| self.shown.completed(done))
            .unwrap_or_default();
        let Some(previous) = self.previous.replace(now) else {
            return StepResult::Continue;
        };
        // The step clock's interval, which is what the frame advanced the
        // world by; a world without one falls back to this system's own.
        let frame_us = ctx.resource::<FrameTime>().map_or_else(
            || u32::saturating_from(now.saturating_sub(previous)),
            |time| {
                u32::saturating_from(round_f64(f64::from(time.dt) * 1e6).min(f64::from(u32::MAX)))
            },
        );

        let render = ctx.profile.render;
        self.run.capture_pass_names(&render);
        let this_frame = PendingFrame {
            timing: FrameTiming {
                run_seconds,
                phase: self.benchmark.phase_at(run_seconds),
                frame_us,
                run_step_us,
                rebased,
                shown,
            },
            render,
        };
        let hidden = ctx.resource::<WindowMinimized>().is_some_and(|m| m.0);
        let past_warmup = run_seconds >= self.run.options.warmup_seconds;
        let unseen = self
            .hidden
            .leaves_out(hidden, this_frame.timing.phase, past_warmup);
        let recorded = (!unseen).then_some(this_frame);
        if let Some(frame) = core::mem::replace(&mut self.pending, recorded) {
            let systems = ctx.profile.system_timings();
            self.run.capture_system_names(systems);
            self.run.record(&FrameSample::new(
                frame.timing,
                &frame.render,
                systems,
                ctx.profile.render_handoffs(),
                ctx.profile.system_fan_outs(),
            ));
        }

        // A capture costs its frame nothing it shows, so the frames around it
        // are measured like any other.
        let measured_seconds = run_seconds - self.benchmark.config.warmup_seconds;
        if let Some(capture) = self.captures.due(measured_seconds) {
            if unseen {
                self.hidden_captures.push(capture);
            } else {
                ctx.insert_resource(FrameCaptureRequest(capture.path));
            }
        }

        if run_seconds >= self.benchmark.end_seconds() {
            self.run.completed = true;
            self.emit();
            return StepResult::Stop;
        }
        StepResult::Continue
    }
}

impl Drop for FrameReportSystem {
    // A run that closed before its last phase ended still has something to say,
    // and the report it prints is marked incomplete rather than passed off as
    // a whole one.
    fn drop(&mut self) {
        self.emit();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera_track::CameraTrackSystem;
    use crate::components::cook::{Camera3D as Camera3DArgs, CameraTrack as CameraTrackArgs};
    use crate::components::{BenchmarkConfig, BenchmarkPhase, Camera3D, CameraTrack, CameraTravel};
    use crate::ecs::World;
    use alloc::string::ToString;
    use alloc::vec;
    use core::cell::Cell;

    std::thread_local! {
        static WRITES: Cell<u32> = const { Cell::new(0) };
    }

    // The report sink every test hands the system: it counts the writes and
    // keeps the text off the test runner's output.
    fn counted(_: &str) {
        WRITES.with(|w| w.set(w.get() + 1));
    }

    fn report(run: &BenchmarkRun) -> FrameReportSystem {
        FrameReportSystem::new(run, counted)
    }

    fn track(travel: Vec<CameraTravel>) -> CameraTrack {
        CameraTrack::bake(CameraTrackArgs {
            travel,
            ..Default::default()
        })
    }

    fn leg(distance: f32, speed: f32) -> CameraTravel {
        CameraTravel {
            direction: [1.0, 0.0, 0.0],
            distance,
            speed,
            ..Default::default()
        }
    }

    // A run with no warm-up over `phases`, each `(name, seconds)`.
    fn benchmark(phases: &[(&str, f32)]) -> BenchmarkRun {
        configured(BenchmarkConfig {
            warmup_seconds: 0.0,
            phases: phases
                .iter()
                .map(|&(name, seconds)| BenchmarkPhase {
                    name: name.to_string(),
                    seconds,
                })
                .collect(),
            ..Default::default()
        })
    }

    fn configured(config: BenchmarkConfig) -> BenchmarkRun {
        BenchmarkRun::new(config)
    }

    // A world running its camera track beside a report system, stepped by hand.
    fn world_with(
        track: CameraTrack,
        run: &BenchmarkRun,
    ) -> (World, CameraTrackSystem, FrameReportSystem) {
        let camera = CameraTrackSystem::new(&track);
        let mut world = World::new();
        world.add_component(Camera3D::bake(Camera3DArgs {
            controller: None,
            ..Default::default()
        }));
        world.add_component(track);
        (world, camera, report(run))
    }

    fn step(world: &mut World, camera: &mut CameraTrackSystem, system: &mut FrameReportSystem) {
        camera.step(&mut world.context());
        system.step(&mut world.context());
    }

    #[test]
    fn the_declared_budget_and_discard_reach_the_reduction() {
        let system = report(&configured(BenchmarkConfig {
            warmup_seconds: 4.5,
            budget_ms: 8.0,
            ..Default::default()
        }));
        assert_eq!(system.run.options.warmup_seconds, 4.5);
        assert_eq!(system.run.options.budget_us, 8_000);
    }

    #[test]
    fn a_frame_is_recorded_one_step_late_and_the_first_not_at_all() {
        // The first frame has nothing to be timed against; the second waits a
        // step for the timings the profile rotates in after it.
        let (mut world, mut camera, mut system) =
            world_with(track(vec![]), &benchmark(&[("a", 60.0)]));
        system.init(&mut world.context());
        step(&mut world, &mut camera, &mut system);
        step(&mut world, &mut camera, &mut system);
        assert!(system.run().recorded.is_empty());
        step(&mut world, &mut camera, &mut system);
        assert_eq!(system.run().recorded.len(), 1);
    }

    // One step of the world as the report sees it: the profile rotated, the
    // graphics system timed with its handoff, this step's render stats
    // published, then the report system last.
    fn profiled_step(
        world: &mut World,
        system: &mut FrameReportSystem,
        graphics: (u32, u32),
        gpu_wait_us: u32,
    ) {
        let mut ctx = world.context();
        ctx.profile.begin_frame();
        ctx.profile.record_render_handoff(graphics.1);
        ctx.profile.record_system("GraphicsSystem", graphics.0);
        ctx.profile.render = RenderStats {
            gpu_wait_us,
            ..Default::default()
        };
        system.step(&mut ctx);
    }

    #[test]
    fn a_frames_wait_and_its_system_timings_come_from_the_same_step() {
        // The second step waited 30 ms on the GPU and handed off for as long;
        // the third did neither. Paired a step apart, the wait would read as 30
        // ms of graphics CPU work in the frame after it.
        let mut world = World::new();
        let mut system = report(&configured(BenchmarkConfig::default()));
        system.init(&mut world.context());
        profiled_step(&mut world, &mut system, (500, 0), 0);
        profiled_step(&mut world, &mut system, (30_400, 30_000), 29_800);
        profiled_step(&mut world, &mut system, (450, 50), 100);
        profiled_step(&mut world, &mut system, (450, 50), 100);

        let samples = &system.run().recorded;
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].gpu_wait_us, 29_800);
        assert_eq!(samples[0].system_us[0], 400);
        assert_eq!(samples[1].gpu_wait_us, 100);
        assert_eq!(samples[1].system_us[0], 400);
    }

    // The frame time is the step clock's interval, not this system's own.
    #[test]
    fn the_frame_time_is_the_step_clocks() {
        let mut world = World::new();
        let mut system = report(&configured(BenchmarkConfig::default()));
        system.init(&mut world.context());
        for _ in 0..3 {
            world.insert_resource(FrameTime {
                dt: 0.025,
                elapsed: 0.0,
            });
            profiled_step(&mut world, &mut system, (500, 0), 0);
        }
        assert_eq!(system.run().recorded[0].frame_us, 25_000);
    }

    // A completed frame is paired with the pose its tick left, and the next
    // completion paces against it.
    #[test]
    fn a_completed_frame_paces_against_the_one_before() {
        use crate::profile::GpuDone;
        let mut world = World::new();
        let mut system = report(&configured(BenchmarkConfig::default()));
        system.init(&mut world.context());
        let ticks = [
            (100, 1.0),
            (22_100, 1.022),
            (44_100, 1.044),
            (66_100, 1.066),
            (88_100, 1.088),
        ];
        for (n, (sampled, pose)) in ticks.into_iter().enumerate() {
            world.insert_resource(crate::ecs::InputSampledAt(sampled));
            world.insert_resource(CameraTrackStatus {
                pose_seconds: pose,
                ..Default::default()
            });
            let mut ctx = world.context();
            ctx.profile.begin_frame();
            // Each step reports the frame two ticks back as finished.
            ctx.profile.render.gpu_done = n.checked_sub(2).map(|back| GpuDone {
                input_sampled_us: ticks[back].0,
                done_us: ticks[back].0 + 45_000 + back as u64 * 3_000,
            });
            system.step(&mut ctx);
        }
        // Samples are recorded a step late: the last one is the fourth tick's,
        // which saw the second completion.
        let samples = &system.run().recorded;
        let last = samples[samples.len() - 1];
        assert_eq!(last.input_latency_us, 48_000);
        assert_eq!(last.shown_us, 25_000);
        assert!(
            (i64::from(last.shown_step_us) - 22_000).abs() <= 1,
            "{}",
            last.shown_step_us
        );
    }

    #[test]
    fn the_runs_phase_names_are_captured_when_it_is_built() {
        let system = report(&benchmark(&[("approach", 2.0), ("rays", 2.0)]));
        assert_eq!(system.run().phases, ["approach", "rays"]);
    }

    #[test]
    fn a_world_with_no_track_still_samples_against_its_own_clock() {
        let mut world = World::new();
        let mut system = report(&configured(BenchmarkConfig::default()));
        system.init(&mut world.context());
        for _ in 0..3 {
            system.step(&mut world.context());
        }
        assert_eq!(system.run().recorded.len(), 1);
        // Inside the default warm-up, so in no phase yet.
        assert_eq!(system.run().recorded[0].phase, None);
        assert!(!system.run().completed);
    }

    #[test]
    fn samples_carry_the_phase_the_frame_was_drawn_in() {
        let (mut world, mut camera, mut system) = world_with(
            track(vec![leg(60.0, 1.0)]),
            &benchmark(&[("approach", 60.0)]),
        );
        camera.init(&mut world.context());
        system.init(&mut world.context());
        for _ in 0..4 {
            step(&mut world, &mut camera, &mut system);
        }
        let run = system.run();
        assert!(run.recorded.iter().all(|s| s.phase == Some(0)));
        // Track time advances on the fixed step, so successive samples differ.
        assert!(run.recorded[1].run_seconds > run.recorded[0].run_seconds);
    }

    #[test]
    fn samples_carry_the_track_time_each_frame_showed_and_the_frame_moves() {
        use crate::ecs::SimTiming;
        use crate::planet::Rebase;

        let (mut world, mut camera, mut system) = world_with(
            track(vec![leg(60.0, 1.0)]),
            &benchmark(&[("approach", 60.0)]),
        );
        world.insert_resource(FrameRebases::default());
        camera.init(&mut world.context());
        system.init(&mut world.context());
        for ticks in [1, 1, 2, 1, 1] {
            if ticks == 2 {
                world.insert_resource(SimTiming {
                    ticks,
                    ..Default::default()
                });
                if let Some(rebases) = world.resource_mut::<FrameRebases>() {
                    rebases.push(Rebase::IDENTITY);
                }
            } else {
                world.insert_resource(SimTiming::default());
            }
            step(&mut world, &mut camera, &mut system);
        }
        let steps: Vec<(u32, bool)> = system
            .run()
            .recorded
            .iter()
            .map(|s| (s.run_step_us, s.rebased))
            .collect();
        // The first frame is never recorded and the last is still pending.
        assert_eq!(steps, [(16_667, false), (33_333, true), (16_667, false)]);
    }

    #[test]
    fn the_end_of_the_last_phase_stops_the_world_and_marks_the_run_complete() {
        // One tick of run, so a later frame is already past its end.
        let (mut world, mut camera, mut system) =
            world_with(track(vec![]), &benchmark(&[("brief", 1.0 / 60.0)]));
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
    fn the_run_outlasts_a_shorter_track_until_its_last_phase_ends() {
        let (mut world, mut camera, mut system) =
            world_with(track(vec![leg(1.0, 60.0)]), &benchmark(&[("long", 60.0)]));
        camera.init(&mut world.context());
        system.init(&mut world.context());
        for _ in 0..4 {
            camera.step(&mut world.context());
            assert_eq!(system.step(&mut world.context()), StepResult::Continue);
        }
        assert!(!system.run().completed);
    }

    #[test]
    fn a_capture_is_requested_once_and_the_frames_around_it_are_measured() {
        use crate::components::FrameCapture;

        let mut world = World::new();
        let mut system = report(&configured(BenchmarkConfig {
            warmup_seconds: 0.0,
            captures: vec![FrameCapture {
                seconds: 0.0,
                path: "a.png".to_string(),
            }],
            ..Default::default()
        }));
        system.init(&mut world.context());
        // The first step only starts the clock; the second is the first frame
        // measured, and the capture is due on it.
        system.step(&mut world.context());
        system.step(&mut world.context());
        let request = world.remove_resource::<FrameCaptureRequest>();
        assert_eq!(request.map(|r| r.0).as_deref(), Some("a.png"));

        // Each frame is recorded once the next step closes it: the capture's
        // and the one after it are both in.
        system.step(&mut world.context());
        system.step(&mut world.context());
        assert_eq!(system.run().recorded.len(), 2);
        assert!(world.resource::<FrameCaptureRequest>().is_none());
    }

    #[test]
    fn a_capture_waits_for_its_moment_after_the_warm_up() {
        use crate::components::FrameCapture;

        let run = configured(BenchmarkConfig {
            warmup_seconds: 0.05,
            captures: vec![FrameCapture {
                seconds: 0.05,
                path: "a.png".to_string(),
            }],
            ..Default::default()
        });
        let (mut world, mut camera, mut system) = world_with(track(vec![leg(60.0, 1.0)]), &run);
        camera.init(&mut world.context());
        system.init(&mut world.context());
        let mut requested_at = None;
        for _ in 0..12 {
            step(&mut world, &mut camera, &mut system);
            if world.remove_resource::<FrameCaptureRequest>().is_some() {
                assert!(requested_at.is_none(), "requested twice");
                requested_at = world
                    .resource::<CameraTrackStatus>()
                    .map(|s| s.elapsed_seconds);
            }
        }
        let at = requested_at.expect("the run passed the capture's moment");
        assert!(at >= 0.1, "requested at {at}s");
    }

    #[test]
    fn frames_run_while_the_window_is_hidden_go_unrecorded_and_are_tallied() {
        use crate::frame_report::hidden::RESTORE_DROPPED_FRAMES;

        let mut world = World::new();
        let mut system = report(&benchmark(&[("a", 60.0)]));
        system.init(&mut world.context());
        system.step(&mut world.context());
        system.step(&mut world.context());
        world.insert_resource(WindowMinimized(true));
        for _ in 0..3 {
            system.step(&mut world.context());
        }
        world.insert_resource(WindowMinimized(false));
        for _ in 0..RESTORE_DROPPED_FRAMES {
            system.step(&mut world.context());
        }
        // Only the frame before the window hid made it in.
        assert_eq!(system.run().recorded.len(), 1);
        system.step(&mut world.context());
        system.step(&mut world.context());
        assert_eq!(system.run().recorded.len(), 2);
        assert_eq!(system.hidden.total(), 3 + RESTORE_DROPPED_FRAMES);
    }

    #[test]
    fn a_capture_due_while_the_window_is_hidden_is_skipped() {
        use crate::components::FrameCapture;

        let mut world = World::new();
        let mut system = report(&configured(BenchmarkConfig {
            warmup_seconds: 0.0,
            captures: vec![FrameCapture {
                seconds: 0.0,
                path: "a.png".to_string(),
            }],
            ..Default::default()
        }));
        system.init(&mut world.context());
        system.step(&mut world.context());
        world.insert_resource(WindowMinimized(true));
        system.step(&mut world.context());
        assert!(world.resource::<FrameCaptureRequest>().is_none());
        assert_eq!(system.hidden_captures.len(), 1);
        assert_eq!(system.captures.missed().count(), 0);
    }

    #[test]
    fn the_report_goes_out_once_however_many_times_it_is_asked_for() {
        let mut system = report(&configured(BenchmarkConfig::default()));
        let before = WRITES.with(Cell::get);
        system.emit();
        // A second call, and the drop, are the teardown path meeting a run
        // that already reported; neither writes the whole thing again.
        system.emit();
        drop(system);
        assert_eq!(WRITES.with(Cell::get) - before, 1);
    }

    // A run with nothing past its warm-up says so instead of a page of zeroes.
    #[test]
    fn an_empty_run_reports_that_there_is_nothing_to_measure() {
        let system = report(&configured(BenchmarkConfig::default()));
        assert!(system.text().contains("there is nothing to measure"));
    }

    // Without a track or a step clock, the run position and the frame time
    // come from the host clock.
    #[test]
    fn a_world_without_a_step_clock_times_frames_on_the_host_clock() {
        use core::sync::atomic::{AtomicU64, Ordering};
        static NOW: AtomicU64 = AtomicU64::new(1_000_000);
        fn tick() -> u64 {
            NOW.fetch_add(20_000, Ordering::Relaxed)
        }

        let mut world = World::new();
        world.insert_resource(Clock(tick));
        let mut system = report(&benchmark(&[("a", 60.0)]));
        system.init(&mut world.context());
        for _ in 0..3 {
            system.step(&mut world.context());
        }
        let sample = system.run().recorded[0];
        assert_eq!(sample.frame_us, 20_000);
        assert!((sample.run_seconds - 0.02).abs() < 1e-6);
    }
}
