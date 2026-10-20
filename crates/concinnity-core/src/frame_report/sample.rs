// What the sampler records for one frame, and the run it accumulates into.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::fanout::FanOutTiming;
use crate::profile::{MAX_PASS_TIMINGS, RenderStats};

use crate::frame_report::reduce::ReduceOptions;
use crate::frame_report::tally::RunTally;

/// How many per-system CPU timings a sample carries. The engine's own table is
/// well inside this, and a world registering its own systems has headroom.
pub(super) const MAX_SYSTEM_TIMINGS: usize = 32;

/// One measured frame.
///
/// Fixed size on purpose: the per-pass timings ride an array rather than a
/// per-frame `Vec`, so recording a frame costs no allocation and the sampler
/// does not perturb the thing it measures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct FrameSample {
    /// How far into the run the frame was drawn, in seconds. Taken from the
    /// run's own clock where it has one, and from the wall clock since the
    /// first sample otherwise.
    pub run_seconds: f32,
    /// Index into [`FrameRun::phases`], or `None` outside every phase.
    pub phase: Option<u32>,
    /// Microseconds since the previous frame on the step clock, the interval
    /// the simulation advanced by: the frame time a player would feel.
    pub frame_us: u32,
    /// Microseconds of run time the camera's pose advanced this frame, against
    /// `frame_us`. A frame can show more or less motion than the time it took,
    /// and steady frame times over uneven motion read as judder that
    /// `frame_us` alone never shows.
    pub run_step_us: u32,
    /// Whether the simulated frame moved under the camera this frame.
    pub rebased: bool,
    /// For a frame step on which the GPU finished a new frame: run time the
    /// camera's pose advanced between that frame and the one finished before
    /// it. A frame can show more or less motion than the time between
    /// completions, which reads as judder on screen.
    pub shown_step_us: u32,
    /// Microseconds between those two GPU completions; zero on a step that
    /// saw no new completion.
    pub shown_us: u32,
    /// GPU microseconds for the most recently completed frame: its span after
    /// the previous frame's ended.
    pub gpu_frame_us: u32,
    /// Microseconds of that frame's span it shared with the frame before it.
    pub gpu_overlap_us: u32,
    /// Microseconds from the newly finished frame's input sample to the end of
    /// its GPU work, or zero when no new completion could be placed on the host
    /// clock.
    pub input_latency_us: u32,
    /// Microseconds the CPU spent blocked on the GPU.
    pub gpu_wait_us: u32,
    /// Microseconds of CPU work the render side spent submitting the frame,
    /// its wait on the GPU excluded.
    pub render_cpu_us: u32,
    /// The part of `render_cpu_us` spent replaying the frame's recorded
    /// backend effects before the draw.
    pub replay_us: u32,
    /// Wall microseconds the backend spent in its queue submit and present.
    pub submit_present_us: u32,
    /// The render thread's wait on the parallel pass recording, against the
    /// recording work inside it.
    pub recording_fan_out: FanOutTiming,
    /// The render thread's packing of the per-draw cull records, inline or
    /// fanned out.
    pub packing_fan_out: FanOutTiming,
    /// CPU microseconds each pass took to record inside that fan-out, indexed
    /// by `PassId as usize`.
    pub pass_record_us: [u32; MAX_PASS_TIMINGS],
    /// Geometry draw calls issued.
    pub draw_calls: u32,
    /// Renderable objects in the scene.
    pub objects: u32,
    /// GPU memory the device held.
    pub vram_bytes: u64,
    /// Per-pass GPU microseconds, in the backend's slot order. Named by
    /// [`FrameRun::pass_names`], which shares the slot order.
    pub pass_us: [u32; MAX_PASS_TIMINGS],
    /// Bit `i` set when pass slot `i` ran on the async compute queue, beside
    /// `gpu_frame_us` rather than inside it.
    pub async_pass_mask: u64,
    /// Per-system CPU microseconds, in schedule order, with the time a system
    /// spent handing its frame to the renderer taken out: that is the render
    /// side's, and `render_cpu_us` carries its work. Named by
    /// [`FrameRun::system_names`], which shares the order.
    pub system_us: [u32; MAX_SYSTEM_TIMINGS],
    /// Per-system parallel fan-out timing, in the same order as `system_us`.
    /// All zero for a system that fanned nothing out.
    pub system_fan_out: [FanOutTiming; MAX_SYSTEM_TIMINGS],
}

impl FrameSample {
    /// Fold one frame's render stats and system timings into a sample.
    /// `handoffs` and `fan_outs` are the per-system render handoff and fan-out
    /// timing recorded beside `systems`.
    pub(super) fn new(
        timing: FrameTiming,
        render: &RenderStats,
        systems: &[(&'static str, u32)],
        handoffs: &[(&'static str, u32)],
        fan_outs: &[(&'static str, FanOutTiming)],
    ) -> Self {
        let mut pass_us = [0_u32; MAX_PASS_TIMINGS];
        for (slot, (_, us)) in pass_us.iter_mut().zip(render.pass_times_us.iter()) {
            *slot = *us;
        }
        let mut system_us = [0_u32; MAX_SYSTEM_TIMINGS];
        for (slot, (name, us)) in system_us.iter_mut().zip(systems.iter()) {
            let handoff = handoffs
                .iter()
                .find(|(handed, _)| handed == name)
                .map_or(0, |(_, us)| *us);
            *slot = us.saturating_sub(handoff);
        }
        let mut system_fan_out = [FanOutTiming::default(); MAX_SYSTEM_TIMINGS];
        for (slot, (name, _)) in system_fan_out.iter_mut().zip(systems.iter()) {
            if let Some((_, timing)) = fan_outs.iter().find(|(fanned, _)| fanned == name) {
                *slot = *timing;
            }
        }
        Self {
            run_seconds: timing.run_seconds,
            phase: timing.phase,
            frame_us: timing.frame_us,
            run_step_us: timing.run_step_us,
            rebased: timing.rebased,
            shown_step_us: timing.shown.step_us,
            shown_us: timing.shown.interval_us,
            gpu_frame_us: render.gpu_frame_us,
            gpu_overlap_us: render.gpu_overlap_us,
            input_latency_us: timing.shown.latency_us,
            gpu_wait_us: render.gpu_wait_us,
            render_cpu_us: render.render_cpu_us,
            replay_us: render.replay_us,
            submit_present_us: render.submit_present_us,
            recording_fan_out: render.recording_fan_out,
            packing_fan_out: render.packing_fan_out,
            pass_record_us: render.pass_record_us,
            draw_calls: render.draw_calls,
            objects: render.objects,
            vram_bytes: render.vram_bytes,
            pass_us,
            async_pass_mask: render.async_pass_mask,
            system_us,
            system_fan_out,
        }
    }
}

/// Where one frame sat on the run's clocks: what a sample carries beside the
/// render stats and system timings.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct FrameTiming {
    /// See [`FrameSample::run_seconds`].
    pub run_seconds: f32,
    /// See [`FrameSample::phase`].
    pub phase: Option<u32>,
    /// See [`FrameSample::frame_us`].
    pub frame_us: u32,
    /// See [`FrameSample::run_step_us`].
    pub run_step_us: u32,
    /// See [`FrameSample::rebased`].
    pub rebased: bool,
    /// The frame the GPU newly finished this step, if any: see
    /// [`FrameSample::shown_step_us`], [`FrameSample::shown_us`] and
    /// [`FrameSample::input_latency_us`].
    pub(crate) shown: crate::frame_report::shown::Shown,
}

/// Everything one run recorded: the tallies its measured frames went into,
/// the names for the two index spaces they refer to, and whether it reached
/// the end of what it had to measure or was cut short.
#[derive(Debug)]
pub(super) struct FrameRun {
    /// The measured frames, tallied as they arrive.
    pub tally: RunTally,
    /// Frames drawn before the warm-up ended, left out of the tally.
    pub warmup_dropped: usize,
    /// What the frames are measured against.
    pub options: ReduceOptions,
    /// The names of the stretches the run is cut into. A sample's `phase`
    /// indexes this.
    pub phases: Vec<String>,
    /// Pass names in the backend's slot order, captured from the first frame
    /// that reported any. Empty slots stay empty.
    pub pass_names: Vec<String>,
    /// System names in schedule order, captured from the first frame that
    /// reported any.
    pub system_names: Vec<String>,
    /// Whether the run reached the end of what it had to measure. One that
    /// stopped early (the window closed, the frame cap hit) reports what it
    /// has, and says so rather than passing it off as a whole run.
    pub completed: bool,
    /// Every frame handed to `record`, warm-up included.
    #[cfg(test)]
    pub recorded: Vec<FrameSample>,
}

impl Default for FrameRun {
    fn default() -> Self {
        Self::new(Vec::new(), ReduceOptions::default())
    }
}

impl FrameRun {
    /// A run cut into `phases`, with every tally it will fill allocated now.
    pub(super) fn new(phases: Vec<String>, options: ReduceOptions) -> Self {
        Self {
            tally: RunTally::new(phases.len(), options.budget_us),
            warmup_dropped: 0,
            options,
            phases,
            pass_names: Vec::new(),
            system_names: Vec::new(),
            completed: false,
            #[cfg(test)]
            recorded: Vec::new(),
        }
    }

    /// Tally one frame, or count it as warm-up.
    pub(super) fn record(&mut self, sample: &FrameSample) {
        #[cfg(test)]
        self.recorded.push(*sample);
        if sample.run_seconds < self.options.warmup_seconds {
            self.warmup_dropped += 1;
            return;
        }
        let cpu_us = self.cpu_us(sample);
        self.tally.record(sample, cpu_us);
    }

    /// What one frame's CPU work came to: its systems and its render
    /// submission.
    ///
    /// Taken from the work rather than from the wall clock: the wall delta is
    /// measured on the thread that steps the world, while the wait is measured
    /// on the one that submits, so subtracting the second from the first
    /// reports a tick that outran the device as having done no work at all. A
    /// run whose host named no systems has nothing to sum and falls back to
    /// that outside view.
    pub(super) fn cpu_us(&self, sample: &FrameSample) -> u32 {
        if self.system_names.is_empty() {
            return sample.frame_us.saturating_sub(sample.gpu_wait_us);
        }
        self.system_names
            .iter()
            .zip(sample.system_us)
            .filter(|(name, _)| !name.is_empty())
            .map(|(_, us)| us)
            .fold(sample.render_cpu_us, u32::saturating_add)
    }

    /// The name of phase `index`, or a placeholder when the world named
    /// none.
    pub(super) fn phase_name(&self, index: Option<u32>) -> &str {
        match index.and_then(|i| self.phases.get(i as usize)) {
            Some(name) => name,
            None => "(unnamed)",
        }
    }

    /// Record the backend's pass names once, from the first frame that has
    /// them. They are fixed for the process lifetime, so a later frame adds
    /// nothing.
    pub(super) fn capture_pass_names(&mut self, render: &RenderStats) {
        if !self.pass_names.is_empty() {
            return;
        }
        if render.pass_times_us.iter().all(|(name, _)| name.is_empty()) {
            return;
        }
        self.pass_names = render
            .pass_times_us
            .iter()
            .map(|(name, _)| (*name).to_string())
            .collect();
    }

    /// Record the schedule's system names once. Schedule order is fixed for
    /// the run, so a later frame adds nothing.
    pub(super) fn capture_system_names(&mut self, systems: &[(&'static str, u32)]) {
        if self.system_names.is_empty() {
            self.system_names = systems
                .iter()
                .map(|(name, _)| (*name).to_string())
                .collect();
        }
    }
}

/// Frames a test reduces, with the names they refer to: replayed through a
/// real [`FrameRun`] in order, so the names are in place before any frame is.
#[cfg(test)]
#[derive(Debug, Default)]
pub(super) struct Recorded {
    pub samples: Vec<FrameSample>,
    pub phases: Vec<String>,
    pub pass_names: Vec<String>,
    pub system_names: Vec<String>,
    pub completed: bool,
}

#[cfg(test)]
impl Recorded {
    pub(super) fn run(&self, options: ReduceOptions) -> FrameRun {
        let mut run = FrameRun::new(self.phases.clone(), options);
        run.pass_names.clone_from(&self.pass_names);
        run.system_names.clone_from(&self.system_names);
        run.completed = self.completed;
        for sample in &self.samples {
            run.record(sample);
        }
        run
    }

    pub(super) fn reduce(
        &self,
        options: ReduceOptions,
    ) -> Option<crate::frame_report::reduce::Report> {
        crate::frame_report::reduce::Report::of(&self.run(options))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn timing(frame_us: u32) -> FrameTiming {
        FrameTiming {
            frame_us,
            ..Default::default()
        }
    }

    fn render_with(passes: &[(&'static str, u32)]) -> RenderStats {
        let mut r = RenderStats {
            gpu_frame_us: 8_000,
            gpu_wait_us: 500,
            render_cpu_us: 1_200,
            draw_calls: 120,
            objects: 900,
            vram_bytes: 64 << 20,
            ..Default::default()
        };
        for (slot, entry) in r.pass_times_us.iter_mut().zip(passes.iter()) {
            *slot = *entry;
        }
        r
    }

    #[test]
    fn a_sample_carries_the_render_stats_and_the_pass_slots() {
        let mut render = render_with(&[("main", 4_000)]);
        render.async_pass_mask = 0b10;
        render.gpu_overlap_us = 900;
        let s = FrameSample::new(
            FrameTiming {
                run_seconds: 1.5,
                phase: Some(2),
                frame_us: 16_000,
                run_step_us: 16_667,
                rebased: true,
                shown: crate::frame_report::shown::Shown {
                    step_us: 22_000,
                    interval_us: 25_000,
                    latency_us: 45_000,
                },
            },
            &render,
            &[("GraphicsSystem", 700)],
            &[],
            &[],
        );
        assert_eq!(s.run_seconds, 1.5);
        assert_eq!(s.phase, Some(2));
        assert_eq!(s.frame_us, 16_000);
        assert_eq!((s.run_step_us, s.rebased), (16_667, true));
        assert_eq!((s.shown_step_us, s.shown_us), (22_000, 25_000));
        assert_eq!(s.input_latency_us, 45_000);
        assert_eq!(s.async_pass_mask, 0b10);
        assert_eq!(s.gpu_overlap_us, 900);
        assert_eq!((s.gpu_frame_us, s.gpu_wait_us), (8_000, 500));
        assert_eq!(s.render_cpu_us, 1_200);
        assert_eq!((s.draw_calls, s.objects), (120, 900));
        assert_eq!(s.vram_bytes, 64 << 20);
        assert_eq!(s.pass_us[0], 4_000);
        assert_eq!(s.pass_us[1], 0);
        assert_eq!(s.system_us[0], 700);
        assert_eq!(s.system_us[1], 0);
    }

    #[test]
    fn a_systems_render_handoff_is_taken_out_of_its_own_span_only() {
        let s = FrameSample::new(
            timing(16_000),
            &render_with(&[]),
            &[("PhysicsSystem", 400), ("GraphicsSystem", 9_000)],
            &[("GraphicsSystem", 8_700)],
            &[],
        );
        assert_eq!(s.system_us[0], 400);
        assert_eq!(s.system_us[1], 300);
    }

    #[test]
    fn a_handoff_longer_than_its_span_leaves_zero_rather_than_wrapping() {
        let s = FrameSample::new(
            timing(16_000),
            &render_with(&[]),
            &[("GraphicsSystem", 100)],
            &[("GraphicsSystem", 900)],
            &[],
        );
        assert_eq!(s.system_us[0], 0);
    }

    #[test]
    fn a_sample_carries_the_recording_fan_out_and_each_systems_own() {
        let fan_out = |wall_us| FanOutTiming {
            wall_us,
            first_job_us: wall_us / 4,
            job_sum_us: wall_us * 2,
            longest_job_us: wall_us / 2,
            tail_us: wall_us / 8,
        };
        let mut render = render_with(&[]);
        render.recording_fan_out = fan_out(2_000);
        render.packing_fan_out = fan_out(150);
        render.pass_record_us[3] = 1_700;
        let s = FrameSample::new(
            timing(16_000),
            &render,
            &[("PhysicsSystem", 400), ("BehaviorSystem", 900)],
            &[],
            &[("BehaviorSystem", fan_out(800))],
        );
        assert_eq!(s.recording_fan_out, fan_out(2_000));
        assert_eq!(s.packing_fan_out, fan_out(150));
        assert_eq!(s.pass_record_us, render.pass_record_us);
        assert_eq!(s.system_fan_out[0], FanOutTiming::default());
        assert_eq!(s.system_fan_out[1], fan_out(800));
    }

    #[test]
    fn system_names_are_captured_once_in_schedule_order() {
        let mut run = FrameRun::default();
        run.capture_system_names(&[]);
        assert!(run.system_names.is_empty());

        run.capture_system_names(&[("PhysicsSystem", 40), ("GraphicsSystem", 700)]);
        assert_eq!(run.system_names, ["PhysicsSystem", "GraphicsSystem"]);

        run.capture_system_names(&[("Other", 1)]);
        assert_eq!(run.system_names, ["PhysicsSystem", "GraphicsSystem"]);
    }

    #[test]
    fn pass_names_are_captured_once_from_the_first_frame_that_has_them() {
        let mut run = FrameRun::default();
        // A backend with no timestamp support reports every slot empty, so
        // there is nothing to name yet.
        run.capture_pass_names(&render_with(&[]));
        assert!(run.pass_names.is_empty());

        run.capture_pass_names(&render_with(&[("shadow", 1), ("main", 2)]));
        assert_eq!(&run.pass_names[..2], ["shadow", "main"]);

        // A later frame does not rewrite them: slot order is fixed for the
        // process, so a second capture could only disagree by being wrong.
        run.capture_pass_names(&render_with(&[("other", 9)]));
        assert_eq!(&run.pass_names[..2], ["shadow", "main"]);
    }

    #[test]
    fn a_phase_with_no_name_reads_as_a_placeholder_rather_than_panicking() {
        let run = FrameRun::new(vec!["approach".to_string()], ReduceOptions::default());
        assert_eq!(run.phase_name(Some(0)), "approach");
        assert_eq!(run.phase_name(None), "(unnamed)");
        // An index past the end is a world that changed under a stale sample.
        assert_eq!(run.phase_name(Some(7)), "(unnamed)");
    }
}
