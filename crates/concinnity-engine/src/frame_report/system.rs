// The system a declared FrameReport gates in: records one row per frame, and
// reports what the run cost when it ends.

use concinnity_core::camera_track::CameraTrackStatus;
use concinnity_core::components::FrameCapture;
use concinnity_core::components::{CameraTrack, FrameReport};
use concinnity_core::ecs::{FrameTime, PipelineContext, StepResult, System, WindowMinimized};
use concinnity_core::planet::FrameRebases;
use concinnity_core::profile::RenderStats;
use std::time::Instant;

use crate::ecs::FrameCaptureRequest;
use crate::frame_report::capture::{CAPTURE_DROPPED_FRAMES, CaptureSchedule};
use crate::frame_report::hidden::HiddenFrames;
use crate::frame_report::reduce::{ReduceOptions, Report};
use crate::frame_report::report::to_text;
use crate::frame_report::sample::{FrameRun, FrameSample, FrameTiming};
use crate::frame_report::shown::ShownFrames;

const MICROS_PER_MILLI: f32 = 1_000.0;

/// Records one [`FrameSample`] per frame and prints what the run cost when it
/// ends.
///
/// Runs in the `PostRender` phase after every engine system, so the render
/// stats it reads were published this step. The system timings are not: the
/// profile rotates them in at the start of the next step, so each frame is
/// held for one step and recorded with its own timings then. The report goes out on the
/// frame the world reports the measurement complete, which leaves that last
/// frame unrecorded; a run cut short before that reports on the way out
/// instead, and says so.
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
    // The camera pose time and frame-move count the previous step saw.
    previous_pose_seconds: f32,
    previous_rebases: u64,
    // Each tick's pose, paired with its frame's GPU completion once it lands.
    shown: ShownFrames,
    captures: CaptureSchedule,
    // Frames still to leave unrecorded behind the latest capture.
    excluded: u32,
    // Captures requested, and the frames left unrecorded around them.
    captured: u32,
    dropped: u32,
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
    /// The system for a world's declared report.
    pub fn new(report: &FrameReport) -> Self {
        Self {
            run: FrameRun::default(),
            pending: None,
            options: ReduceOptions {
                warmup_seconds: report.warmup_seconds,
                budget_us: (report.budget_ms * MICROS_PER_MILLI) as u32,
            },
            stop_when_complete: report.stop_when_complete,
            reported: false,
            previous: None,
            started: None,
            previous_pose_seconds: 0.0,
            previous_rebases: 0,
            shown: ShownFrames::default(),
            captures: CaptureSchedule::new(&report.captures),
            excluded: 0,
            captured: 0,
            dropped: 0,
            hidden: HiddenFrames::default(),
            hidden_captures: Vec::new(),
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
        if self.captured > 0 {
            println!(
                "frame report: {} frame(s) around {} capture(s) left out of the measurement",
                self.dropped, self.captured,
            );
        }
        if self.hidden.total() > 0 {
            let segments: Vec<String> = self
                .hidden
                .by_segment()
                .map(|(segment, count)| format!("{} {count}", self.run.segment_name(segment)))
                .collect();
            println!(
                "frame report: the window was hidden, so {} frame(s) past the warm-up ({}) were left out of the measurement",
                self.hidden.total(),
                segments.join(", "),
            );
        }
        for skipped in &self.hidden_captures {
            println!(
                "frame report: the window was hidden at {:.1}s, so {} was not captured",
                skipped.seconds, skipped.path,
            );
        }
        for missed in self.captures.missed() {
            println!(
                "frame report: the run ended before {:.1}s, so {} was not captured",
                missed.seconds, missed.path,
            );
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
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        let now = Instant::now();
        let started = *self.started.get_or_insert(now);
        let status = ctx.resource::<CameraTrackStatus>().copied();
        let run_seconds = status.map_or_else(
            || now.duration_since(started).as_secs_f32(),
            |s| s.elapsed_seconds,
        );
        let rebases = ctx
            .resource::<FrameRebases>()
            .map_or(0, FrameRebases::count);
        let pose_seconds = status.map_or(run_seconds, |s| s.pose_seconds);
        let run_step_us =
            ((pose_seconds - self.previous_pose_seconds).max(0.0) * 1_000_000.0).round() as u32;
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
            || {
                now.duration_since(previous)
                    .as_micros()
                    .min(u32::MAX.into()) as u32
            },
            |time| (f64::from(time.dt) * 1e6).round().min(f64::from(u32::MAX)) as u32,
        );

        let render = ctx.profile.render;
        self.run.capture_pass_names(&render);
        let this_frame = PendingFrame {
            timing: FrameTiming {
                run_seconds,
                segment: status.and_then(|s| s.segment),
                frame_us,
                run_step_us,
                rebased,
                shown,
            },
            render,
        };
        let hidden = ctx.resource::<WindowMinimized>().is_some_and(|m| m.0);
        let past_warmup = run_seconds >= self.options.warmup_seconds;
        let unseen = self
            .hidden
            .leaves_out(hidden, this_frame.timing.segment, past_warmup);
        let recorded = if unseen {
            None
        } else if self.excluded > 0 {
            self.excluded -= 1;
            self.dropped += 1;
            None
        } else {
            Some(this_frame)
        };
        if let Some(frame) = std::mem::replace(&mut self.pending, recorded) {
            let systems = ctx.profile.system_timings();
            self.run.capture_system_names(systems);
            self.run.samples.push(FrameSample::new(
                frame.timing,
                &frame.render,
                systems,
                ctx.profile.render_handoffs(),
                ctx.profile.system_fan_outs(),
            ));
        }

        if let Some(capture) = self.captures.due(run_seconds) {
            if unseen {
                self.hidden_captures.push(capture);
            } else {
                ctx.insert_resource(FrameCaptureRequest(capture.path));
                self.excluded = CAPTURE_DROPPED_FRAMES;
                self.captured += 1;
            }
        }

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
    use concinnity_core::camera_track::CameraTrackSystem;
    use concinnity_core::components::cook::{
        Camera3D as Camera3DArgs, CameraTrack as CameraTrackArgs,
    };
    use concinnity_core::components::{Camera3D, CameraTravel};
    use concinnity_core::ecs::World;

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
    fn a_frame_is_recorded_one_step_late_and_the_first_not_at_all() {
        // The first frame has nothing to be timed against; the second waits a
        // step for the timings the profile rotates in after it.
        let (mut world, mut camera, mut system) = world_with(track(vec![]));
        system.init(&mut world.context());
        step(&mut world, &mut camera, &mut system);
        step(&mut world, &mut camera, &mut system);
        assert!(system.run().samples.is_empty());
        step(&mut world, &mut camera, &mut system);
        assert_eq!(system.run().samples.len(), 1);
        forget_report(system);
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
        let mut system = FrameReportSystem::new(&FrameReport::default());
        system.init(&mut world.context());
        profiled_step(&mut world, &mut system, (500, 0), 0);
        profiled_step(&mut world, &mut system, (30_400, 30_000), 29_800);
        profiled_step(&mut world, &mut system, (450, 50), 100);
        profiled_step(&mut world, &mut system, (450, 50), 100);

        let samples = &system.run().samples;
        assert_eq!(samples.len(), 2);
        assert_eq!(samples.get(0).unwrap().gpu_wait_us, 29_800);
        assert_eq!(samples.get(0).unwrap().system_us[0], 400);
        assert_eq!(samples.get(1).unwrap().gpu_wait_us, 100);
        assert_eq!(samples.get(1).unwrap().system_us[0], 400);
        forget_report(system);
    }

    // The frame time is the step clock's interval, not this system's own.
    #[test]
    fn the_frame_time_is_the_step_clocks() {
        let mut world = World::new();
        let mut system = FrameReportSystem::new(&FrameReport::default());
        system.init(&mut world.context());
        for _ in 0..3 {
            world.insert_resource(FrameTime {
                dt: 0.025,
                elapsed: 0.0,
            });
            profiled_step(&mut world, &mut system, (500, 0), 0);
        }
        assert_eq!(system.run().samples.get(0).unwrap().frame_us, 25_000);
        forget_report(system);
    }

    // A completed frame is paired with the pose its tick left, and the next
    // completion paces against it.
    #[test]
    fn a_completed_frame_paces_against_the_one_before() {
        use concinnity_core::profile::GpuDone;
        let mut world = World::new();
        let mut system = FrameReportSystem::new(&FrameReport::default());
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
        let samples = &system.run().samples;
        let last = samples.get(samples.len() - 1).unwrap();
        assert_eq!(last.input_latency_us, 48_000);
        assert_eq!(last.shown_us, 25_000);
        assert!(
            (last.shown_step_us as i64 - 22_000).abs() <= 1,
            "{}",
            last.shown_step_us
        );
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
        for _ in 0..3 {
            system.step(&mut world.context());
        }
        assert_eq!(system.run().samples.len(), 1);
        assert!(system.run().segments.is_empty());
        assert_eq!(system.run().samples.get(0).unwrap().segment, None);
        assert!(!system.run().completed);
        forget_report(system);
    }

    #[test]
    fn samples_carry_the_segment_the_frame_was_drawn_in() {
        let (mut world, mut camera, mut system) =
            world_with(track(vec![leg(60.0, 1.0, "approach")]));
        camera.init(&mut world.context());
        system.init(&mut world.context());
        for _ in 0..4 {
            step(&mut world, &mut camera, &mut system);
        }
        let run = system.run();
        assert!(run.samples.iter().all(|s| s.segment == Some(0)));
        // Track time advances on the fixed step, so successive samples differ.
        assert!(run.samples.get(1).unwrap().run_seconds > run.samples.get(0).unwrap().run_seconds);
        forget_report(system);
    }

    #[test]
    fn samples_carry_the_track_time_each_frame_showed_and_the_frame_moves() {
        use concinnity_core::ecs::SimTiming;
        use concinnity_core::planet::Rebase;

        let (mut world, mut camera, mut system) =
            world_with(track(vec![leg(60.0, 1.0, "approach")]));
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
            .samples
            .iter()
            .map(|s| (s.run_step_us, s.rebased))
            .collect();
        // The first frame is never recorded and the last is still pending.
        assert_eq!(steps, [(16_667, false), (33_333, true), (16_667, false)]);
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
        assert!(system.run().samples.len() >= 2);
        forget_report(system);
    }

    #[test]
    fn a_capture_is_requested_once_and_the_frames_it_stalls_go_unrecorded() {
        use concinnity_core::components::FrameCapture;

        let mut world = World::new();
        let mut system = FrameReportSystem::new(&FrameReport {
            warmup_seconds: 0.0,
            captures: vec![FrameCapture {
                seconds: 0.0,
                path: "a.png".to_string(),
            }],
            ..Default::default()
        });
        system.init(&mut world.context());
        // The first step only starts the clock; the second is the first frame
        // measured, and the capture is due on it.
        system.step(&mut world.context());
        system.step(&mut world.context());
        let request = world.remove_resource::<FrameCaptureRequest>();
        assert_eq!(request.map(|r| r.0).as_deref(), Some("a.png"));

        let stalled = CAPTURE_DROPPED_FRAMES as usize;
        for _ in 0..stalled {
            system.step(&mut world.context());
        }
        // Only the frame the capture was requested on made it in.
        assert_eq!(system.run().samples.len(), 1);
        system.step(&mut world.context());
        system.step(&mut world.context());
        assert_eq!(system.run().samples.len(), 2);
        assert!(world.resource::<FrameCaptureRequest>().is_none());
        assert_eq!(system.dropped, CAPTURE_DROPPED_FRAMES);
        assert_eq!(system.captured, 1);
        forget_report(system);
    }

    #[test]
    fn a_capture_waits_for_the_track_to_reach_its_moment() {
        use concinnity_core::components::FrameCapture;

        let (mut world, mut camera, _) = world_with(track(vec![leg(60.0, 1.0, "approach")]));
        let mut system = FrameReportSystem::new(&FrameReport {
            warmup_seconds: 0.0,
            captures: vec![FrameCapture {
                seconds: 0.05,
                path: "a.png".to_string(),
            }],
            ..Default::default()
        });
        camera.init(&mut world.context());
        system.init(&mut world.context());
        let mut requested_at = None;
        for _ in 0..10 {
            step(&mut world, &mut camera, &mut system);
            if world.remove_resource::<FrameCaptureRequest>().is_some() {
                assert!(requested_at.is_none(), "requested twice");
                requested_at = world
                    .resource::<CameraTrackStatus>()
                    .map(|s| s.elapsed_seconds);
            }
        }
        let at = requested_at.expect("the track passed the capture's moment");
        assert!(at >= 0.05, "requested at {at}s");
        forget_report(system);
    }

    #[test]
    fn frames_run_while_the_window_is_hidden_go_unrecorded_and_are_tallied() {
        use crate::frame_report::hidden::RESTORE_DROPPED_FRAMES;

        let mut world = World::new();
        let mut system = FrameReportSystem::new(&FrameReport {
            warmup_seconds: 0.0,
            ..Default::default()
        });
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
        assert_eq!(system.run().samples.len(), 1);
        system.step(&mut world.context());
        system.step(&mut world.context());
        assert_eq!(system.run().samples.len(), 2);
        assert_eq!(system.hidden.total(), 3 + RESTORE_DROPPED_FRAMES);
        forget_report(system);
    }

    #[test]
    fn a_capture_due_while_the_window_is_hidden_is_skipped() {
        use concinnity_core::components::FrameCapture;

        let mut world = World::new();
        let mut system = FrameReportSystem::new(&FrameReport {
            warmup_seconds: 0.0,
            captures: vec![FrameCapture {
                seconds: 0.0,
                path: "a.png".to_string(),
            }],
            ..Default::default()
        });
        system.init(&mut world.context());
        system.step(&mut world.context());
        world.insert_resource(WindowMinimized(true));
        system.step(&mut world.context());
        assert!(world.resource::<FrameCaptureRequest>().is_none());
        assert_eq!(system.captured, 0);
        assert_eq!(system.hidden_captures.len(), 1);
        assert_eq!(system.captures.missed().count(), 0);
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
