// What a run accumulates per frame: fixed-size tallies sized when the run
// starts, so recording costs no allocation however long the run is.

use alloc::vec::Vec;

use crate::profile::MAX_PASS_TIMINGS;
use concinnity_core::fanout::FanOutTiming;
use concinnity_core::math::SaturatingFrom;

use crate::frame_report::histogram::Histogram;
use crate::frame_report::sample::{FrameSample, MAX_SYSTEM_TIMINGS};
use crate::frame_report::stats::{Column, Mean};

/// How many of the run's slowest frames are kept whole.
pub(super) const SLOW_FRAMES_KEPT: usize = 3;

/// Fan-out slots a tally keeps: the render recording, the render packing,
/// then one per system.
pub(super) const FAN_OUT_SLOTS: usize = 2 + MAX_SYSTEM_TIMINGS;

/// Means of a fan-out's timings over the frames it ran in.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct FanOutMeans {
    pub wall: Mean,
    pub first_job: Mean,
    pub job_sum: Mean,
    pub longest_job: Mean,
    pub tail: Mean,
}

impl FanOutMeans {
    fn record(&mut self, timing: FanOutTiming) {
        if timing.wall_us == 0 {
            return;
        }
        self.wall.record(timing.wall_us);
        self.first_job.record(timing.first_job_us);
        self.job_sum.record(timing.job_sum_us);
        self.longest_job.record(timing.longest_job_us);
        self.tail.record(timing.tail_us);
    }
}

/// Everything one stretch of the run reports, accumulated frame by frame.
#[derive(Debug, Clone)]
pub(super) struct PhaseTally {
    pub frame: Column,
    pub cpu: Column,
    pub gpu: Column,
    pub first_seconds: f32,
    pub last_seconds: f32,
    pub gpu_wait: Mean,
    pub draw_calls: Mean,
    pub objects: Mean,
    pub vram_peak_bytes: u64,
    /// Motion shown against the interval between completions, in
    /// thousandths, over the frames that saw a new completion.
    pub pacing: Histogram,
    pub still_frames: usize,
    /// Input latency over the frames that measured it.
    pub latency: Histogram,
    pub rebases: usize,
    /// GPU microseconds per pass slot, zero for a frame it did not run in.
    pub passes: Vec<Histogram>,
    /// Frames each pass slot ran in, and the ones it ran on the async queue.
    pub pass_ran: [u32; MAX_PASS_TIMINGS],
    pub pass_async: [u32; MAX_PASS_TIMINGS],
    pub overlap: Histogram,
    pub systems: [Mean; MAX_SYSTEM_TIMINGS],
    pub render_cpu: Mean,
    pub fan_outs: [FanOutMeans; FAN_OUT_SLOTS],
}

impl PhaseTally {
    fn new(budget_us: u32) -> Self {
        Self {
            frame: Column::new(budget_us),
            cpu: Column::new(budget_us),
            gpu: Column::new(budget_us),
            first_seconds: 0.0,
            last_seconds: 0.0,
            gpu_wait: Mean::default(),
            draw_calls: Mean::default(),
            objects: Mean::default(),
            vram_peak_bytes: 0,
            pacing: Histogram::new(),
            still_frames: 0,
            latency: Histogram::new(),
            rebases: 0,
            passes: (0..MAX_PASS_TIMINGS).map(|_| Histogram::new()).collect(),
            pass_ran: [0; MAX_PASS_TIMINGS],
            pass_async: [0; MAX_PASS_TIMINGS],
            overlap: Histogram::new(),
            systems: [Mean::default(); MAX_SYSTEM_TIMINGS],
            render_cpu: Mean::default(),
            fan_outs: [FanOutMeans::default(); FAN_OUT_SLOTS],
        }
    }

    /// How many frames the stretch measured.
    pub(super) fn frames(&self) -> usize {
        self.frame.count()
    }

    fn record(&mut self, sample: &FrameSample, cpu_us: u32) {
        if self.frames() == 0 {
            self.first_seconds = sample.run_seconds;
        }
        self.last_seconds = sample.run_seconds;
        self.frame.record(sample.frame_us);
        self.cpu.record(cpu_us);
        self.gpu.record(sample.gpu_frame_us);
        self.gpu_wait.record(sample.gpu_wait_us);
        self.draw_calls.record(sample.draw_calls);
        self.objects.record(sample.objects);
        self.vram_peak_bytes = self.vram_peak_bytes.max(sample.vram_bytes);
        if sample.shown_us > 0 {
            self.pacing.record(u32::saturating_from(
                u64::from(sample.shown_step_us) * 1_000 / u64::from(sample.shown_us),
            ));
            self.still_frames += usize::from(sample.shown_step_us == 0);
        }
        if sample.input_latency_us > 0 {
            self.latency.record(sample.input_latency_us);
        }
        self.rebases += usize::from(sample.rebased);
        for (slot, histogram) in self.passes.iter_mut().enumerate() {
            let us = sample.pass_us[slot];
            histogram.record(us);
            if us > 0 {
                self.pass_ran[slot] += 1;
                self.pass_async[slot] += u32::from(sample.async_pass_mask & (1u64 << slot) != 0);
            }
        }
        self.overlap.record(sample.gpu_overlap_us);
        for (mean, us) in self.systems.iter_mut().zip(sample.system_us) {
            mean.record(us);
        }
        self.render_cpu.record(sample.render_cpu_us);
        self.fan_outs[0].record(sample.recording_fan_out);
        self.fan_outs[1].record(sample.packing_fan_out);
        for (means, timing) in self.fan_outs[2..].iter_mut().zip(sample.system_fan_out) {
            means.record(timing);
        }
    }
}

/// The slowest frames recorded, kept whole, slowest first; the earlier of two
/// equal ones first.
#[derive(Debug, Clone, Default)]
pub(super) struct SlowestFrames {
    frames: [Option<FrameSample>; SLOW_FRAMES_KEPT],
}

impl SlowestFrames {
    fn offer(&mut self, sample: &FrameSample) {
        let Some(at) = self
            .frames
            .iter()
            .position(|kept| kept.is_none_or(|kept| sample.frame_us > kept.frame_us))
        else {
            return;
        };
        self.frames[at..].rotate_right(1);
        self.frames[at] = Some(*sample);
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &FrameSample> {
        self.frames.iter().flatten()
    }
}

/// A whole run's tallies: the run overall, each phase, and the frames outside
/// every phase, with the order the run entered them.
#[derive(Debug, Clone)]
pub(super) struct RunTally {
    pub overall: PhaseTally,
    // One per declared phase, then one for frames outside every phase.
    stretches: Vec<PhaseTally>,
    // Indices into `stretches`, in the order the run first entered them.
    entered: Vec<usize>,
    pub slowest: SlowestFrames,
}

impl RunTally {
    /// Tallies for a run cut into `phases` stretches, every one allocated now.
    pub(super) fn new(phases: usize, budget_us: u32) -> Self {
        Self {
            overall: PhaseTally::new(budget_us),
            stretches: (0..=phases).map(|_| PhaseTally::new(budget_us)).collect(),
            entered: Vec::with_capacity(phases + 1),
            slowest: SlowestFrames::default(),
        }
    }

    /// Record one measured frame whose CPU work came to `cpu_us`.
    pub(super) fn record(&mut self, sample: &FrameSample, cpu_us: u32) {
        let stretch = self.stretch_of(sample.phase);
        if !self.entered.contains(&stretch) {
            self.entered.push(stretch);
        }
        self.stretches[stretch].record(sample, cpu_us);
        self.overall.record(sample, cpu_us);
        self.slowest.offer(sample);
    }

    /// Each stretch the run entered, in that order, with the phase it stands
    /// for: `None` for the frames outside every phase.
    pub(super) fn entered(&self) -> impl Iterator<Item = (Option<u32>, &PhaseTally)> {
        let unphased = self.stretches.len() - 1;
        self.entered.iter().map(move |&stretch| {
            let phase = (stretch != unphased).then(|| u32::saturating_from(stretch));
            (phase, &self.stretches[stretch])
        })
    }

    // A phase index past the declared ones belongs with the unphased frames.
    fn stretch_of(&self, phase: Option<u32>) -> usize {
        let unphased = self.stretches.len() - 1;
        phase.map_or(unphased, |p| (p as usize).min(unphased))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_report::sample::FrameTiming;
    use crate::profile::RenderStats;

    fn frame(run_seconds: f32, phase: Option<u32>, frame_us: u32) -> FrameSample {
        FrameSample::new(
            FrameTiming {
                run_seconds,
                phase,
                frame_us,
                ..Default::default()
            },
            &RenderStats::default(),
            &[],
            &[],
            &[],
        )
    }

    #[test]
    fn the_slowest_frames_are_kept_slowest_first_and_capped() {
        let mut slowest = SlowestFrames::default();
        for (n, us) in [10, 50, 20, 50, 40, 5].into_iter().enumerate() {
            slowest.offer(&frame(n as f32, None, us));
        }
        let kept: Vec<(u32, f32)> = slowest
            .iter()
            .map(|s| (s.frame_us, s.run_seconds))
            .collect();
        // The earlier of the two 50s stays ahead of the later one.
        assert_eq!(kept, [(50, 1.0), (50, 3.0), (40, 4.0)]);
    }

    #[test]
    fn stretches_come_out_in_the_order_the_run_entered_them() {
        let mut tally = RunTally::new(2, 16_667);
        tally.record(&frame(0.0, Some(1), 10), 0);
        tally.record(&frame(1.0, None, 10), 0);
        tally.record(&frame(2.0, Some(0), 10), 0);
        tally.record(&frame(3.0, Some(1), 10), 0);
        let order: Vec<(Option<u32>, usize)> = tally
            .entered()
            .map(|(phase, t)| (phase, t.frames()))
            .collect();
        assert_eq!(order, [(Some(1), 2), (None, 1), (Some(0), 1)]);
        assert_eq!(tally.overall.frames(), 4);
    }

    #[test]
    fn a_phase_past_the_declared_ones_counts_as_unphased() {
        let mut tally = RunTally::new(1, 16_667);
        tally.record(&frame(0.0, Some(7), 10), 0);
        tally.record(&frame(1.0, None, 10), 0);
        let order: Vec<(Option<u32>, usize)> = tally
            .entered()
            .map(|(phase, t)| (phase, t.frames()))
            .collect();
        assert_eq!(order, [(None, 2)]);
    }

    #[test]
    fn a_fan_out_is_averaged_over_the_frames_it_ran_in() {
        let mut tally = PhaseTally::new(16_667);
        let mut ran = frame(0.0, None, 10);
        ran.recording_fan_out.wall_us = 900;
        tally.record(&ran, 0);
        tally.record(&frame(1.0, None, 10), 0);
        assert_eq!(tally.fan_outs[0].wall.get(), 900);
    }
}
