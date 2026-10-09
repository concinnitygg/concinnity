// What the sampler records for one frame, and the run it accumulates into.

use concinnity_core::profile::{FanOutTiming, MAX_PASS_TIMINGS, RenderStats};

/// How many per-system CPU timings a sample carries. The engine's own table is
/// well inside this, and a world registering its own systems has headroom.
pub const MAX_SYSTEM_TIMINGS: usize = 32;

/// One measured frame.
///
/// Fixed size on purpose: the per-pass timings ride an array rather than a
/// per-frame `Vec`, so recording a frame costs no allocation and the sampler
/// does not perturb the thing it measures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameSample {
    /// How far into the run the frame was drawn, in seconds. Taken from the
    /// run's own clock where it has one, and from the wall clock since the
    /// first sample otherwise.
    pub run_seconds: f32,
    /// Index into [`FrameRun::segments`], or `None` outside every segment.
    pub segment: Option<u32>,
    /// Wall-clock microseconds since the previous frame: the frame time a
    /// player would feel.
    pub frame_us: u32,
    /// GPU microseconds for the most recently completed frame.
    pub gpu_frame_us: u32,
    /// Microseconds the CPU spent blocked on the GPU.
    pub gpu_wait_us: u32,
    /// Microseconds of CPU work the render side spent submitting the frame,
    /// its wait on the GPU excluded.
    pub render_cpu_us: u32,
    /// The render thread's wait on the parallel pass recording, against the
    /// recording work inside it.
    pub recording_fan_out: FanOutTiming,
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
    pub fn new(
        run_seconds: f32,
        segment: Option<u32>,
        frame_us: u32,
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
            run_seconds,
            segment,
            frame_us,
            gpu_frame_us: render.gpu_frame_us,
            gpu_wait_us: render.gpu_wait_us,
            render_cpu_us: render.render_cpu_us,
            recording_fan_out: render.recording_fan_out,
            pass_record_us: render.pass_record_us,
            draw_calls: render.draw_calls,
            objects: render.objects,
            vram_bytes: render.vram_bytes,
            pass_us,
            system_us,
            system_fan_out,
        }
    }
}

/// Everything one run recorded: the frames, the names for the two index
/// spaces they refer to, and whether it reached the end of what it had to
/// measure or was cut short.
#[derive(Debug, Default)]
pub struct FrameRun {
    /// The frames, in the order they were drawn.
    pub samples: Vec<FrameSample>,
    /// The names of the stretches the run is cut into. A sample's `segment`
    /// indexes this.
    pub segments: Vec<String>,
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
}

impl FrameRun {
    /// The name of segment `index`, or a placeholder when the world named
    /// none.
    pub fn segment_name(&self, index: Option<u32>) -> &str {
        match index.and_then(|i| self.segments.get(i as usize)) {
            Some(name) => name,
            None => "(unnamed)",
        }
    }

    /// Record the backend's pass names once, from the first frame that has
    /// them. They are fixed for the process lifetime, so a later frame adds
    /// nothing.
    pub fn capture_pass_names(&mut self, render: &RenderStats) {
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
    pub fn capture_system_names(&mut self, systems: &[(&'static str, u32)]) {
        if self.system_names.is_empty() {
            self.system_names = systems
                .iter()
                .map(|(name, _)| (*name).to_string())
                .collect();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let s = FrameSample::new(
            1.5,
            Some(2),
            16_000,
            &render_with(&[("main", 4_000)]),
            &[("GraphicsSystem", 700)],
            &[],
            &[],
        );
        assert_eq!(s.run_seconds, 1.5);
        assert_eq!(s.segment, Some(2));
        assert_eq!(s.frame_us, 16_000);
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
            0.0,
            None,
            16_000,
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
            0.0,
            None,
            16_000,
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
        render.pass_record_us[3] = 1_700;
        let s = FrameSample::new(
            0.0,
            None,
            16_000,
            &render,
            &[("PhysicsSystem", 400), ("BehaviorSystem", 900)],
            &[],
            &[("BehaviorSystem", fan_out(800))],
        );
        assert_eq!(s.recording_fan_out, fan_out(2_000));
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
    fn a_segment_with_no_name_reads_as_a_placeholder_rather_than_panicking() {
        let run = FrameRun {
            segments: vec!["approach".to_string()],
            ..Default::default()
        };
        assert_eq!(run.segment_name(Some(0)), "approach");
        assert_eq!(run.segment_name(None), "(unnamed)");
        // An index past the end is a world that changed under a stale sample.
        assert_eq!(run.segment_name(Some(7)), "(unnamed)");
    }
}
