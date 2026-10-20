// Samples in, report out. Pure: no clock, no world, no I/O.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::profile::FanOutTiming;
use crate::render::render_graph::PassId;
use serde::Serialize;

use crate::frame_report::sample::{FrameRun, FrameSample};
use crate::frame_report::stats::Distribution;
use crate::frame_report::tally::{FanOutMeans, PhaseTally};

/// How many passes a phase lists. The tail is noise next to the handful that
/// own the frame.
const PASSES_REPORTED: usize = 8;

/// How many systems a phase lists, on the same reasoning.
const SYSTEMS_REPORTED: usize = 6;

/// The row for the graphics-queue passes that fell below the cut, as the sum
/// of their medians.
const OTHER_PASSES: &str = "other passes";

/// The row for graphics-queue time no timed pass covers: a render pass's
/// vertex and tiling phase where no other pass's fragment phase hides it, a
/// wait on the async queue, and the GPU idling between command buffers.
/// Without it a list of the largest passes reads as if it were the whole frame.
const GAPS: &str = "gaps";

/// The row that closes a pass list, taken away from the rows above it: the
/// part of the frame's span that ran while the previous frame was still on the
/// GPU. Its passes are timed in full, but that time is the previous frame's
/// as much as this one's, so the GPU frame leaves it out.
const OVERLAP: &str = "overlap with previous frame";

/// A system below this share of the stepped CPU work is not what is limiting
/// it. Ranking the rest would rank noise.
const SYSTEM_FLOOR_SHARE: f32 = 0.01;

/// The row that stands for the render side's CPU work beside the systems:
/// replaying the frame's backend effects and recording and submitting its
/// draw. It runs on the stepping thread or on its own render thread, and in
/// neither case inside any one system's share.
const RENDER_SUBMIT: &str = "render submit";

/// The row for the render thread's parallel pass recording among the
/// fan-outs.
const RENDER_RECORDING: &str = "render recording";

/// The row for the render thread's packing of the per-draw cull records among
/// the fan-outs.
const RENDER_PACKING: &str = "render packing";

/// How many CPU rows, and how many passes, a slow frame lists.
const SLOW_FRAME_ROWS: usize = 4;

/// What a run is reduced against.
#[derive(Debug, Clone, Copy)]
pub(super) struct ReduceOptions {
    /// Frames earlier than this many seconds into the run are dropped. Shader
    /// compilation, streaming residency, temporal antialiasing history and
    /// auto-exposure all converge over the opening seconds, so without a
    /// discard two runs of identical code disagree.
    pub warmup_seconds: f32,
    /// The frame budget every distribution is counted against.
    pub budget_us: u32,
}

impl Default for ReduceOptions {
    fn default() -> Self {
        Self {
            warmup_seconds: 2.0,
            budget_us: 16_667,
        }
    }
}

/// One pass's contribution to a phase's GPU frame.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(super) struct PassShare {
    /// The backend's name for the pass.
    pub name: String,
    /// Median GPU microseconds the pass took.
    ///
    /// A median rather than a mean because a backend that mistimes one frame
    /// mistimes it by orders of magnitude, and a handful of those in a phase
    /// of a few hundred frames moves a mean past the slowest frame the phase
    /// actually drew.
    pub median_us: u32,
    /// That median as a share of the phase's median GPU frame, in `0..=1`,
    /// or negative for the overlap row, which is taken away from the rest.
    ///
    /// Reported alongside the microseconds because per-pass timings co-vary
    /// strongly, even between passes that share no work: a pass whose absolute
    /// number moved with every other pass has not changed, and its share says
    /// so.
    pub share: f32,
    /// The listed pass whose time already includes this one's, for a pass the
    /// backend timed inside another. It is listed right below that pass and
    /// left out of the remainder.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub within: Option<String>,
    /// Whether the pass ran on the async compute queue in most of the frames
    /// it ran in. Such a pass overlaps the GPU frame rather than adding to it,
    /// so it is listed after the rows that add up to the frame.
    #[serde(skip_serializing_if = "core::ops::Not::not")]
    pub async_queue: bool,
}

/// One system's CPU contribution to a phase's frame, or the render side's
/// under the `render submit` row.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(super) struct SystemCost {
    /// The schedule's name for the system, or `render submit`.
    pub name: String,
    /// Mean CPU microseconds the system's step took.
    pub mean_us: u32,
    /// That mean as a share of the phase's mean CPU work, in `0..=1`.
    pub share: f32,
}

/// A parallel fan-out's wait against its work: the render side's pass
/// recording, or a system that fanned its own work out. Microseconds, either
/// within one frame or as means over the frames of a phase it ran in.
///
/// A wall time well above the longest job is a fan-out not limited by its
/// work; `first_job_us` and `tail_us` say whether the gap is the workers
/// starting late or the waiting thread resuming late.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(super) struct FanOutCost {
    /// `render recording`, or the schedule's name for the system.
    pub name: String,
    /// How long the waiting thread was held.
    pub wall_us: u32,
    /// From the start of the fan-out to the first job starting.
    pub first_job_us: u32,
    /// Every job's duration, summed.
    pub job_sum_us: u32,
    /// The longest single job.
    pub longest_job_us: u32,
    /// From the last job ending to the waiting thread resuming.
    pub tail_us: u32,
}

impl FanOutCost {
    fn of(name: &str, timing: FanOutTiming) -> Self {
        Self {
            name: name.to_string(),
            wall_us: timing.wall_us,
            first_job_us: timing.first_job_us,
            job_sum_us: timing.job_sum_us,
            longest_job_us: timing.longest_job_us,
            tail_us: timing.tail_us,
        }
    }
}

/// A named cost within one frame.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(super) struct FrameCost {
    /// The system, the `render submit` row, or the pass.
    pub name: String,
    /// Microseconds it took in that frame.
    pub us: u32,
}

/// One of the run's slowest frames, broken down far enough to say which side
/// held it up: CPU work in a system or in the render submission, a wait on
/// the GPU or the display, or neither, which leaves time outside the measured
/// work.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(super) struct SlowFrame {
    /// How far into the run the frame was drawn, in seconds.
    pub run_seconds: f32,
    /// The phase it was drawn in.
    pub phase: String,
    /// Its wall-clock frame time.
    pub frame_us: u32,
    /// The CPU work in it, summed the way the phase table's column is.
    pub cpu_us: u32,
    /// Microseconds the render side spent blocked on the GPU and the display.
    pub gpu_wait_us: u32,
    /// The GPU time it reported. That is the most recently completed GPU
    /// frame, which trails the CPU one, so a GPU spike can land on a later
    /// frame than the wait it caused.
    pub gpu_frame_us: u32,
    /// The largest CPU costs, systems and the render submission together,
    /// largest first. Zero rows are left out.
    pub cpu: Vec<FrameCost>,
    /// The largest graphics-queue passes of the GPU frame it reported,
    /// largest first, leaving out a pass timed inside another.
    pub passes: Vec<FrameCost>,
    /// The part of that frame's span no timed graphics pass covers.
    pub gpu_gap_us: u32,
    /// The part of that frame's span it shared with the frame before it.
    pub gpu_overlap_us: u32,
    /// Microseconds of run time the frame advanced.
    pub run_step_us: u32,
    /// Whether the simulated frame moved under the camera that frame.
    pub rebased: bool,
    /// The fan-outs it waited on, the render recording first. Ones that did
    /// not run are left out.
    pub fan_outs: Vec<FanOutCost>,
    /// The passes that took longest to record on the CPU, largest first: which
    /// jobs made up the render recording's job sum.
    pub recording: Vec<FrameCost>,
    /// The render submission's own spans, largest first: replaying the
    /// frame's backend effects, waiting on the pass recording, and the queue
    /// submit and present. What they leave of the submission was the draw's
    /// other work.
    pub render: Vec<FrameCost>,
}

/// What one stretch of the run measured.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(super) struct PhaseReport {
    /// The phase's authored name, or a placeholder.
    pub name: String,
    /// Seconds of track time the phase covered.
    pub seconds: f32,
    /// Wall-clock frame time, the number a player feels.
    pub frame: Distribution,
    /// The CPU work in that frame: what the tick's systems spent, with the
    /// graphics system's handoff to the renderer taken out, plus the render
    /// side's own submission work. Waits on the GPU are in neither. When the
    /// render side runs on its own thread the two halves overlap, so this is
    /// work done rather than a critical path, and it can exceed `frame`.
    ///
    /// A run whose host named no systems falls back to the wall time less the
    /// wait, which is the same quantity measured from the outside.
    pub cpu: Distribution,
    /// GPU time for the frame.
    pub gpu: Distribution,
    /// Mean microseconds the CPU spent blocked on the GPU. Near the frame time
    /// on a GPU-bound stretch and near zero on a CPU-bound one.
    pub gpu_wait_mean_us: u32,
    /// Mean geometry draw calls per frame.
    pub draw_calls_mean: u32,
    /// Mean renderable objects per frame.
    pub objects_mean: u32,
    /// Largest GPU memory reading seen.
    pub vram_peak_bytes: u64,
    /// How evenly the run's clock advanced against the frames that showed it.
    pub pacing: Pacing,
    /// How stale the input was by the time each frame's GPU work ended.
    pub input_latency: InputLatency,
    /// Frames on which the simulated frame moved under the camera.
    pub rebases: usize,
    /// The passes that owned the GPU frame, largest first.
    pub passes: Vec<PassShare>,
    /// The systems that owned the CPU frame, and the render submission beside
    /// them, largest first. Empty on a stretch where none was a meaningful
    /// share of it.
    pub systems: Vec<SystemCost>,
    /// The fan-outs, the render recording first, each as means over the
    /// frames it ran in: a system that fans out only above a threshold would
    /// otherwise read as faster than any one of its fan-outs. Ones that never
    /// ran are left out.
    pub fan_outs: Vec<FanOutCost>,
}

/// How much run time each finished frame showed against the time since the
/// frame finished before it, as a ratio: 1.0 is a frame whose motion matched
/// the cadence frames reached the end of the GPU at.
///
/// A clock that moves in whole fixed steps under frames that do not divide
/// them evenly, or one sampled at a different moment than frames finish,
/// shows uneven motion at one steady frame time. That reads as judder on
/// screen and never as a frame-time hitch, so it is measured here rather than
/// inferred from the frame times. Frames the backend could not place on the
/// host clock are left out.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub(super) struct Pacing {
    /// 5th percentile of the ratio.
    pub p5: f32,
    /// Median of the ratio.
    pub p50: f32,
    /// 95th percentile of the ratio.
    pub p95: f32,
    /// Finished frames that showed the same pose as the frame before.
    pub still_frames: usize,
}

impl Pacing {
    fn of(tally: &PhaseTally) -> Self {
        if tally.pacing.count() == 0 {
            return Self::default();
        }
        // Thousandths, so the percentiles stay on frames that happened.
        let at = |q| tally.pacing.percentile(q) as f32 / 1_000.0;
        Self {
            p5: at(0.05),
            p50: at(0.50),
            p95: at(0.95),
            still_frames: tally.still_frames,
        }
    }
}

/// How long after a frame's input was sampled its GPU work ended: how stale
/// what reached the screen was. Frames the backend could not place on the
/// host clock are left out, so a backend that measures none reports none.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub(super) struct InputLatency {
    /// Median microseconds.
    pub p50_us: u32,
    /// 95th percentile microseconds.
    pub p95_us: u32,
    /// Frames that measured it.
    pub frames: usize,
}

impl InputLatency {
    fn of(tally: &PhaseTally) -> Self {
        Self {
            p50_us: tally.latency.percentile(0.50),
            p95_us: tally.latency.percentile(0.95),
            frames: tally.latency.count(),
        }
    }
}

/// A whole run, reduced.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(super) struct Report {
    /// Whether the run reached the end of what it had to measure.
    pub completed: bool,
    /// Frames dropped as warm-up.
    pub warmup_dropped: usize,
    /// The budget every distribution was counted against.
    pub budget_us: u32,
    /// Every measured frame, as one summary.
    pub overall: PhaseReport,
    /// The same, cut into the stretches the run named. A cost that shows
    /// only here belongs to one of them; one that shows only in `overall` is
    /// spread across all of them.
    pub phases: Vec<PhaseReport>,
    /// The slowest measured frames, slowest first: a hitch the distributions
    /// can only show as a maximum, named with what it spent.
    pub slowest: Vec<SlowFrame>,
}

impl Report {
    /// Reduce a run. Returns `None` when the warm-up discard left nothing,
    /// which is a run too short to say anything about rather than a run of
    /// zeroes.
    pub(super) fn of(run: &FrameRun) -> Option<Self> {
        let tally = &run.tally;
        if tally.overall.frames() == 0 {
            return None;
        }
        Some(Self {
            completed: run.completed,
            warmup_dropped: run.warmup_dropped,
            budget_us: run.options.budget_us,
            overall: summarize("all".to_string(), &tally.overall, run),
            phases: tally
                .entered()
                .map(|(phase, stretch)| summarize(run.phase_name(phase).to_string(), stretch, run))
                .collect(),
            slowest: tally
                .slowest
                .iter()
                .map(|sample| slow_frame(sample, run))
                .collect(),
        })
    }
}

// One of the slowest frames, broken down.
fn slow_frame(sample: &FrameSample, run: &FrameRun) -> SlowFrame {
    SlowFrame {
        run_seconds: sample.run_seconds,
        phase: run.phase_name(sample.phase).to_string(),
        frame_us: sample.frame_us,
        cpu_us: run.cpu_us(sample),
        gpu_wait_us: sample.gpu_wait_us,
        gpu_frame_us: sample.gpu_frame_us,
        cpu: largest(
            named_slots(&run.system_names)
                .map(|(slot, name)| (name, sample.system_us[slot]))
                .chain(core::iter::once((RENDER_SUBMIT, sample.render_cpu_us))),
        ),
        passes: largest(
            graphics_passes(sample, run).map(|(slot, name)| (name, sample.pass_us[slot])),
        ),
        gpu_gap_us: gap_us(sample, run),
        gpu_overlap_us: sample.gpu_overlap_us,
        run_step_us: sample.run_step_us,
        rebased: sample.rebased,
        fan_outs: fan_out_slots(run)
            .map(|(slot, name)| FanOutCost::of(name, fan_out_in(sample, slot)))
            .filter(|cost| cost.wall_us > 0)
            .collect(),
        recording: largest(
            PassId::ALL
                .iter()
                .map(|id| (id.name(), sample.pass_record_us[*id as usize])),
        ),
        render: largest(
            [
                ("replay", sample.replay_us),
                ("recording", sample.recording_fan_out.wall_us),
                ("submit and present", sample.submit_present_us),
            ]
            .into_iter(),
        ),
    }
}

// Where a sample keeps one fan-out's timing.
#[derive(Clone, Copy)]
enum FanOutSlot {
    Recording,
    Packing,
    System(usize),
}

// Every fan-out a sample can carry, named: the render recording, the render
// packing, then each named system's.
fn fan_out_slots(run: &FrameRun) -> impl Iterator<Item = (FanOutSlot, &str)> {
    [
        (FanOutSlot::Recording, RENDER_RECORDING),
        (FanOutSlot::Packing, RENDER_PACKING),
    ]
    .into_iter()
    .chain(named_slots(&run.system_names).map(|(slot, name)| (FanOutSlot::System(slot), name)))
}

// One sample's reading of a fan-out slot.
fn fan_out_in(sample: &FrameSample, slot: FanOutSlot) -> FanOutTiming {
    match slot {
        FanOutSlot::Recording => sample.recording_fan_out,
        FanOutSlot::Packing => sample.packing_fan_out,
        FanOutSlot::System(slot) => sample.system_fan_out[slot],
    }
}

// A stretch's means for one fan-out slot.
fn fan_out_means_in(tally: &PhaseTally, slot: FanOutSlot) -> &FanOutMeans {
    let index = match slot {
        FanOutSlot::Recording => 0,
        FanOutSlot::Packing => 1,
        FanOutSlot::System(slot) => 2 + slot,
    };
    &tally.fan_outs[index]
}

// The fan-outs over a stretch, each as means over the frames it ran in,
// leaving out the ones that never ran.
fn fan_out_means(tally: &PhaseTally, run: &FrameRun) -> Vec<FanOutCost> {
    fan_out_slots(run)
        .map(|(slot, name)| {
            let means = fan_out_means_in(tally, slot);
            FanOutCost {
                name: name.to_string(),
                wall_us: means.wall.get(),
                first_job_us: means.first_job.get(),
                job_sum_us: means.job_sum.get(),
                longest_job_us: means.longest_job.get(),
                tail_us: means.tail.get(),
            }
        })
        .filter(|cost| cost.wall_us > 0)
        .collect()
}

// The nonzero costs, largest first, cut to what a slow frame lists.
fn largest<'a>(costs: impl Iterator<Item = (&'a str, u32)>) -> Vec<FrameCost> {
    let mut rows: Vec<FrameCost> = costs
        .filter(|(_, us)| *us > 0)
        .map(|(name, us)| FrameCost {
            name: name.to_string(),
            us,
        })
        .collect();
    rows.sort_by_key(|row| core::cmp::Reverse(row.us));
    rows.truncate(SLOW_FRAME_ROWS);
    rows
}

// The pass the named one's time is already counted in: the pass whose timing
// span contains it, when that pass ran. A nested pass whose enclosing pass did
// not run has nothing to be counted twice in, so it stands on its own.
fn counted_in(name: &str, ran: impl Fn(&str) -> bool) -> Option<&'static str> {
    PassId::from_name(name)
        .and_then(PassId::enclosing)
        .map(PassId::name)
        .filter(|outer| ran(outer))
}

// The occupied slots of a name table, with their index.
fn named_slots(names: &[String]) -> impl Iterator<Item = (usize, &str)> {
    names
        .iter()
        .enumerate()
        .filter(|(_, name)| !name.is_empty())
        .map(|(slot, name)| (slot, name.as_str()))
}

// Reduce one stretch of frames.
fn summarize(name: String, tally: &PhaseTally, run: &FrameRun) -> PhaseReport {
    let gpu = tally.gpu.distribution();
    PhaseReport {
        name,
        seconds: tally.last_seconds - tally.first_seconds,
        frame: tally.frame.distribution(),
        cpu: tally.cpu.distribution(),
        gpu,
        gpu_wait_mean_us: tally.gpu_wait.get(),
        draw_calls_mean: tally.draw_calls.get(),
        objects_mean: tally.objects.get(),
        vram_peak_bytes: tally.vram_peak_bytes,
        pacing: Pacing::of(tally),
        input_latency: InputLatency::of(tally),
        rebases: tally.rebases,
        passes: pass_shares(tally, run, gpu.p50_us),
        systems: system_costs(tally, run),
        fan_outs: fan_out_means(tally, run),
    }
}

// The systems and the render submission that owned the frame's CPU work,
// largest first, measured against that work rather than against wall time.
fn system_costs(tally: &PhaseTally, run: &FrameRun) -> Vec<SystemCost> {
    if run.system_names.is_empty() {
        return Vec::new();
    }
    let means: Vec<(&str, u32)> = named_slots(&run.system_names)
        .map(|(slot, name)| (name, tally.systems[slot].get()))
        .chain(core::iter::once((RENDER_SUBMIT, tally.render_cpu.get())))
        .collect();
    let stepped: u32 = means.iter().map(|(_, us)| us).sum();
    if stepped == 0 {
        return Vec::new();
    }
    let mut costs: Vec<SystemCost> = means
        .into_iter()
        .filter_map(|(name, mean_us)| {
            let share = mean_us as f32 / stepped as f32;
            if share < SYSTEM_FLOOR_SHARE {
                return None;
            }
            Some(SystemCost {
                name: name.to_string(),
                mean_us,
                share,
            })
        })
        .collect();
    costs.sort_unstable_by_key(|c| core::cmp::Reverse(c.mean_us));
    costs.truncate(SYSTEMS_REPORTED);
    costs
}

// The passes that owned the GPU frame, largest first, with each one's share of
// it, and a pass timed inside another listed right below it. Passes that never
// ran are left out rather than listed as zero.
//
// The graphics-queue passes come first, closed by the passes below the cut and
// the gaps between them, so those rows add up to the median GPU frame. The
// async-queue passes follow: they ran beside the frame rather than inside it.
fn pass_shares(tally: &PhaseTally, run: &FrameRun, gpu_median_us: u32) -> Vec<PassShare> {
    let share_of = |us: u32| {
        if gpu_median_us == 0 {
            0.0
        } else {
            us as f32 / gpu_median_us as f32
        }
    };
    let mut shares: Vec<PassShare> = named_slots(&run.pass_names)
        .filter_map(|(slot, name)| {
            let median_us = tally.passes[slot].percentile(0.5);
            if median_us == 0 {
                return None;
            }
            Some(PassShare {
                name: name.to_string(),
                median_us,
                share: share_of(median_us),
                within: None,
                async_queue: mostly_async(tally, slot),
            })
        })
        .collect();
    shares.sort_unstable_by_key(|p| core::cmp::Reverse(p.median_us));
    let (async_rows, graphics): (Vec<PassShare>, Vec<PassShare>) =
        shares.into_iter().partition(|p| p.async_queue);
    let (mut rows, other) = nest(graphics);
    // A backend that timed nothing gets no closing rows: they correct a
    // partial list rather than standing in for one.
    if !rows.is_empty() {
        let top: u32 = rows
            .iter()
            .filter(|p| p.within.is_none())
            .map(|p| p.median_us)
            .sum();
        let overlap = tally.overlap.percentile(0.5);
        let gaps = (gpu_median_us + overlap).saturating_sub(top + other);
        for (name, us, sign) in [
            (OTHER_PASSES, other, 1.0),
            (GAPS, gaps, 1.0),
            (OVERLAP, overlap, -1.0),
        ] {
            if us > 0 {
                rows.push(PassShare {
                    name: name.to_string(),
                    median_us: us,
                    share: sign * share_of(us),
                    within: None,
                    async_queue: false,
                });
            }
        }
    }
    rows.extend(async_rows.into_iter().take(PASSES_REPORTED));
    rows
}

// Whether pass `slot` ran on the async queue in more than half the frames it
// ran in.
fn mostly_async(tally: &PhaseTally, slot: usize) -> bool {
    tally.pass_async[slot] * 2 > tally.pass_ran[slot]
}

// The pass slots one frame timed on the graphics queue at the top level: not
// on the async queue, and not inside another pass that ran that frame.
fn graphics_passes<'a>(
    sample: &'a FrameSample,
    run: &'a FrameRun,
) -> impl Iterator<Item = (usize, &'a str)> + 'a {
    let us = |name: &str| {
        named_slots(&run.pass_names)
            .find(|(_, n)| *n == name)
            .map_or(0, |(slot, _)| sample.pass_us[slot])
    };
    named_slots(&run.pass_names)
        .filter(move |(slot, _)| sample.async_pass_mask & (1u64 << slot) == 0)
        .filter(move |(_, name)| counted_in(name, |outer| us(outer) > 0).is_none())
}

// The graphics-queue time one frame's timed passes leave uncovered, over its
// whole span, the part shared with the previous frame included.
fn gap_us(sample: &FrameSample, run: &FrameRun) -> u32 {
    let passes = graphics_passes(sample, run)
        .map(|(slot, _)| sample.pass_us[slot])
        .fold(0, u32::saturating_add);
    (sample.gpu_frame_us + sample.gpu_overlap_us).saturating_sub(passes)
}

// Cut a cost-ordered pass list to the passes a phase reports, each followed
// by the passes timed inside it (see `counted_in`), and sum the medians of the
// top-level passes below the cut.
fn nest(shares: Vec<PassShare>) -> (Vec<PassShare>, u32) {
    let ran: Vec<String> = shares.iter().map(|p| p.name.clone()).collect();
    let within = |name: &str| counted_in(name, |outer| ran.iter().any(|n| n == outer));
    let (mut inner, top): (Vec<PassShare>, Vec<PassShare>) =
        shares.into_iter().partition(|p| within(&p.name).is_some());
    let other = top
        .iter()
        .skip(PASSES_REPORTED)
        .map(|p| p.median_us)
        .fold(0, u32::saturating_add);
    let mut out = Vec::with_capacity(top.len() + inner.len());
    for pass in top.into_iter().take(PASSES_REPORTED) {
        let outer = pass.name.clone();
        out.push(pass);
        for mut p in inner.extract_if(.., |p| within(&p.name) == Some(outer.as_str())) {
            p.within = Some(outer.clone());
            out.push(p);
        }
    }
    (out, other)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_report::sample::MAX_SYSTEM_TIMINGS;
    use crate::frame_report::sample::Recorded;
    use crate::math::SaturatingFrom;
    use crate::profile::MAX_PASS_TIMINGS;
    use alloc::{format, vec};

    // A sample with the fields a test cares about and zeroes elsewhere.
    fn sample(run_seconds: f32, phase: Option<u32>, frame_us: u32) -> FrameSample {
        FrameSample {
            run_seconds,
            phase,
            frame_us,
            run_step_us: frame_us,
            rebased: false,
            shown_step_us: frame_us,
            shown_us: frame_us,
            gpu_frame_us: frame_us / 2,
            gpu_overlap_us: 0,
            input_latency_us: 0,
            gpu_wait_us: 100,
            render_cpu_us: 0,
            replay_us: 0,
            submit_present_us: 0,
            recording_fan_out: FanOutTiming::default(),
            packing_fan_out: FanOutTiming::default(),
            pass_record_us: [0; MAX_PASS_TIMINGS],
            draw_calls: 50,
            objects: 400,
            vram_bytes: 1 << 20,
            pass_us: [0; MAX_PASS_TIMINGS],
            async_pass_mask: 0,
            system_us: [0; MAX_SYSTEM_TIMINGS],
            system_fan_out: [FanOutTiming::default(); MAX_SYSTEM_TIMINGS],
        }
    }

    fn run_of(samples: Vec<FrameSample>) -> Recorded {
        Recorded {
            samples,
            phases: vec!["shadows".to_string(), "rays".to_string()],
            pass_names: Vec::new(),
            system_names: Vec::new(),
            completed: true,
        }
    }

    fn no_warmup() -> ReduceOptions {
        ReduceOptions {
            warmup_seconds: 0.0,
            ..Default::default()
        }
    }

    #[test]
    fn the_default_discard_and_budget_are_the_documented_ones() {
        let o = ReduceOptions::default();
        assert_eq!(o.warmup_seconds, 2.0);
        // 16667us is a 60Hz frame.
        assert_eq!(o.budget_us, 16_667);
    }

    #[test]
    fn a_run_that_is_all_warmup_reduces_to_nothing_rather_than_to_zeroes() {
        // Reporting zeroes here would read as a perfect run instead of an
        // absent one.
        let run = run_of(vec![sample(0.5, Some(0), 16_000)]);
        assert!(run.reduce(ReduceOptions::default()).is_none());
    }

    #[test]
    fn warmup_frames_are_dropped_and_counted() {
        let run = run_of(vec![
            sample(0.5, Some(0), 90_000),
            sample(1.9, Some(0), 80_000),
            sample(2.0, Some(0), 10_000),
            sample(3.0, Some(0), 12_000),
        ]);
        let report = run
            .reduce(ReduceOptions::default())
            .expect("measured frames");
        assert_eq!(report.warmup_dropped, 2);
        assert_eq!(report.overall.frame.count, 2);
        // The slow opening frames are gone, so they cannot drag the summary.
        assert_eq!(report.overall.frame.max_us, 12_000);
    }

    #[test]
    fn phases_come_out_in_the_order_the_path_entered_them() {
        // Declaration order would put shadows first either way, so the track
        // visits them the other way round to tell the two apart.
        let run = run_of(vec![
            sample(0.0, Some(1), 10_000),
            sample(1.0, Some(0), 20_000),
            sample(2.0, Some(1), 11_000),
        ]);
        let report = run.reduce(no_warmup()).expect("measured frames");
        let names: Vec<&str> = report.phases.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["rays", "shadows"]);
    }

    #[test]
    fn a_phase_summarizes_only_its_own_frames() {
        // A cost in one stretch has to be visible there rather than averaged
        // into the run.
        let run = run_of(vec![
            sample(0.0, Some(0), 10_000),
            sample(1.0, Some(0), 10_000),
            sample(2.0, Some(1), 40_000),
            sample(3.0, Some(1), 40_000),
        ]);
        let report = run.reduce(no_warmup()).expect("measured frames");
        let shadows = &report.phases[0];
        let rays = &report.phases[1];
        assert_eq!(shadows.frame.mean_us, 10_000);
        assert_eq!(rays.frame.mean_us, 40_000);
        assert_eq!((shadows.frame.count, rays.frame.count), (2, 2));
        // The overall summary is still over everything.
        assert_eq!(report.overall.frame.count, 4);
        assert_eq!(report.overall.frame.mean_us, 25_000);
    }

    #[test]
    fn a_phases_span_is_its_own_track_time() {
        let run = run_of(vec![
            sample(0.0, Some(0), 10_000),
            sample(4.0, Some(0), 10_000),
            sample(5.0, Some(1), 10_000),
        ]);
        let report = run.reduce(no_warmup()).expect("measured frames");
        assert_eq!(report.phases[0].seconds, 4.0);
        // One frame spans no time, which is honest rather than an error.
        assert_eq!(report.phases[1].seconds, 0.0);
    }

    #[test]
    fn frames_in_no_phase_are_reported_under_a_placeholder() {
        let run = run_of(vec![sample(0.0, None, 10_000)]);
        let report = run.reduce(no_warmup()).expect("measured frames");
        assert_eq!(report.phases[0].name, "(unnamed)");
    }

    #[test]
    fn passes_are_ranked_by_cost_and_carry_their_share_of_the_gpu_frame() {
        let mut first = sample(0.0, Some(0), 20_000);
        // gpu_frame is 10_000us, split across three passes.
        first.pass_us[0] = 1_000;
        first.pass_us[1] = 6_000;
        first.pass_us[2] = 3_000;
        let mut run = run_of(vec![first]);
        run.pass_names = vec!["shadow".to_string(), "main".to_string(), "ssao".to_string()];
        let report = run.reduce(no_warmup()).expect("measured frames");
        let passes = &report.overall.passes;
        assert_eq!(passes[0].name, "main");
        assert_eq!(passes[0].median_us, 6_000);
        assert!((passes[0].share - 0.6).abs() < 1e-4);
        assert_eq!(passes[1].name, "ssao");
        assert_eq!(passes[2].name, "shadow");
        // The three passes account for the whole GPU frame, so nothing is left
        // over to report.
        assert_eq!(passes.len(), 3);
    }

    // A frame whose listed passes cover part of it says so, so the shares add
    // up to the frame rather than to whatever the backend happened to time.
    #[test]
    fn what_the_listed_passes_do_not_cover_is_reported_as_its_own_row() {
        let mut only = sample(0.0, Some(0), 20_000);
        // gpu_frame is 10_000us and one pass covers 2_500 of it.
        only.pass_us[0] = 2_500;
        let mut run = run_of(vec![only]);
        run.pass_names = vec!["main".to_string()];
        let report = run.reduce(no_warmup()).expect("measured frames");
        let passes = &report.overall.passes;
        assert_eq!(passes.len(), 2);
        assert_eq!(passes[1].name, GAPS);
        assert_eq!(passes[1].median_us, 7_500);
        assert!((passes.iter().map(|p| p.share).sum::<f32>() - 1.0).abs() < 1e-4);
    }

    // An async-queue pass runs beside the frame, so it is listed after the
    // rows that add up to the frame and stays out of them.
    #[test]
    fn an_async_pass_is_listed_after_the_frame_and_left_out_of_its_rows() {
        let mut only = sample(0.0, Some(0), 20_000);
        // gpu_frame is 10_000us: main 8_000 on the graphics queue, and a 7_000
        // async pass that would leave nothing over if it were counted.
        only.pass_us[0] = 8_000;
        only.pass_us[1] = 7_000;
        only.async_pass_mask = 0b10;
        let mut run = run_of(vec![only]);
        run.pass_names = vec!["main".to_string(), "light_cull".to_string()];
        let report = run.reduce(no_warmup()).expect("measured frames");
        let rows: Vec<(&str, u32, bool)> = report
            .overall
            .passes
            .iter()
            .map(|p| (p.name.as_str(), p.median_us, p.async_queue))
            .collect();
        assert_eq!(
            rows,
            [
                ("main", 8_000, false),
                (GAPS, 2_000, false),
                ("light_cull", 7_000, true),
            ]
        );
    }

    // The rows above the async ones add up to the median GPU frame: the listed
    // passes, the ones below the cut together, and the gaps between them.
    #[test]
    fn the_graphics_rows_add_up_to_the_gpu_frame() {
        let passes = PASSES_REPORTED + 3;
        let mut frames = Vec::new();
        for n in 0..5 {
            let mut f = sample(n as f32, Some(0), 30_000);
            f.gpu_frame_us = 20_000 + n * 10;
            f.gpu_overlap_us = 4_000;
            for slot in 0..passes {
                f.pass_us[slot] = 2_000 - u32::saturating_from(slot) * 100;
            }
            // One async pass that overlaps the frame.
            f.pass_us[passes] = 5_000;
            f.async_pass_mask = 1 << passes;
            frames.push(f);
        }
        let mut run = run_of(frames);
        run.pass_names = (0..=passes).map(|i| format!("pass{i}")).collect();
        let report = run.reduce(no_warmup()).expect("measured frames");
        let phase = &report.overall;
        let frame: Vec<&PassShare> = phase.passes.iter().filter(|p| !p.async_queue).collect();
        assert_eq!(frame.len(), PASSES_REPORTED + 3);
        let other = frame[PASSES_REPORTED];
        assert_eq!(other.name, OTHER_PASSES);
        // The three passes below the cut: 1_200 + 1_100 + 1_000.
        assert_eq!(other.median_us, 3_300);
        assert_eq!(frame[PASSES_REPORTED + 1].name, GAPS);
        let overlap = frame[PASSES_REPORTED + 2];
        assert_eq!((overlap.name.as_str(), overlap.median_us), (OVERLAP, 4_000));
        let total: i64 = frame
            .iter()
            .map(|p| i64::from(p.median_us) * if p.share < 0.0 { -1 } else { 1 })
            .sum();
        assert_eq!(total, i64::from(phase.gpu.p50_us));
        let shares: f32 = frame.iter().map(|p| p.share).sum();
        assert!((shares - 1.0).abs() < 1e-4, "{shares}");
        assert!(phase.passes.last().is_some_and(|p| p.async_queue));
    }

    // A pass that moves between queues as features toggle is classed by
    // where it ran most.
    #[test]
    fn a_pass_is_async_only_where_it_mostly_ran_async() {
        let mut frames = Vec::new();
        for n in 0..3 {
            let mut f = sample(n as f32, Some(0), 20_000);
            f.pass_us[0] = 1_000;
            f.async_pass_mask = u64::from(n == 0);
            frames.push(f);
        }
        let mut run = run_of(frames);
        run.pass_names = vec!["hiz_build".to_string()];
        let report = run.reduce(no_warmup()).expect("measured frames");
        assert!(!report.overall.passes[0].async_queue);
    }

    #[test]
    fn a_slow_frame_leaves_async_passes_out_and_names_its_gap() {
        let mut slow = sample(0.0, Some(0), 40_000);
        // gpu_frame is 20_000us.
        slow.pass_us[0] = 12_000;
        slow.pass_us[1] = 3_000;
        slow.pass_us[2] = 9_000;
        slow.async_pass_mask = 0b100;
        slow.gpu_overlap_us = 1_000;
        slow.rebased = true;
        slow.run_step_us = 33_333;
        let mut run = run_of(vec![slow]);
        run.pass_names = vec![
            "main".to_string(),
            "shadow".to_string(),
            "ocean".to_string(),
        ];
        let report = run.reduce(no_warmup()).expect("measured frames");
        let frame = &report.slowest[0];
        let names: Vec<&str> = frame.passes.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["main", "shadow"]);
        // 20_000 + 1_000 of span less the 15_000 its graphics passes cover.
        assert_eq!(frame.gpu_gap_us, 6_000);
        assert_eq!(frame.gpu_overlap_us, 1_000);
        assert!(frame.rebased);
        assert_eq!(frame.run_step_us, 33_333);
    }

    // Frames finishing a steady 22 ms apart over a 60 Hz clock show one step,
    // then two: even frame times, uneven motion.
    #[test]
    fn pacing_measures_motion_against_the_gpu_frame_interval() {
        let mut frames = Vec::new();
        for n in 0..20 {
            let mut f = sample(n as f32, Some(0), 22_000);
            f.shown_us = 22_000;
            f.shown_step_us = if n % 3 == 2 { 33_333 } else { 16_667 };
            frames.push(f);
        }
        frames[5].shown_step_us = 0;
        frames[7].rebased = true;
        let report = run_of(frames).reduce(no_warmup()).expect("measured frames");
        let pacing = report.overall.pacing;
        assert_eq!(pacing.p5, 0.0);
        assert!((pacing.p50 - 0.757).abs() < 1e-3, "{pacing:?}");
        assert!((pacing.p95 - 1.515).abs() < 1e-3, "{pacing:?}");
        assert_eq!(pacing.still_frames, 1);
        assert_eq!(report.overall.rebases, 1);
    }

    // Frames the backend could not place on the host clock stay out of the
    // percentiles rather than reading as no latency at all.
    #[test]
    fn input_latency_leaves_out_unmeasured_frames() {
        let frames = (0..20)
            .map(|n| {
                let mut f = sample(n as f32, Some(0), 16_000);
                f.input_latency_us = if n % 2 == 0 { 0 } else { 40_000 + n * 1_000 };
                f
            })
            .collect();
        let latency = run_of(frames)
            .reduce(no_warmup())
            .expect("measured frames")
            .overall
            .input_latency;
        assert_eq!(latency.frames, 10);
        assert_eq!(latency.p50_us, 49_000);
        assert_eq!(latency.p95_us, 59_000);
    }

    #[test]
    fn a_run_with_no_latency_readings_reports_none() {
        let run = run_of((0..10).map(|n| sample(n as f32, Some(0), 16_000)).collect());
        let latency = run
            .reduce(no_warmup())
            .expect("measured frames")
            .overall
            .input_latency;
        assert_eq!(latency, InputLatency::default());
    }

    // A step on which the GPU finished nothing new has no interval to pace.
    #[test]
    fn steps_without_a_new_completion_are_left_out_of_pacing() {
        let frames = (0..10)
            .map(|n| {
                let mut f = sample(n as f32, Some(0), 16_000);
                if n % 2 == 0 {
                    (f.shown_step_us, f.shown_us) = (0, 0);
                }
                f
            })
            .collect();
        let pacing = run_of(frames)
            .reduce(no_warmup())
            .expect("measured frames")
            .overall
            .pacing;
        assert_eq!((pacing.p5, pacing.p50, pacing.p95), (1.0, 1.0, 1.0));
        assert_eq!(pacing.still_frames, 0);
    }

    #[test]
    fn even_motion_paces_at_one() {
        let run = run_of((0..10).map(|n| sample(n as f32, Some(0), 16_000)).collect());
        let pacing = run
            .reduce(no_warmup())
            .expect("measured frames")
            .overall
            .pacing;
        assert_eq!((pacing.p5, pacing.p50, pacing.p95), (1.0, 1.0, 1.0));
        assert_eq!(pacing.still_frames, 0);
    }

    // Vulkan and DirectX time the sky inside the main pass. Its row sits
    // under main and stays out of the remainder, so the top-level rows add up
    // to the frame the way they do on Metal, which reports no sky time.
    #[test]
    fn a_pass_timed_inside_another_is_listed_under_it_and_counted_once() {
        let mut nested = sample(0.0, Some(0), 20_000);
        // gpu_frame is 10_000us: main 6_000 (sky's 1_000 included), shadow 3_000.
        nested.pass_us[0] = 3_000;
        nested.pass_us[1] = 6_000;
        nested.pass_us[2] = 1_000;
        let mut run = run_of(vec![nested]);
        run.pass_names = vec!["shadow".to_string(), "main".to_string(), "sky".to_string()];
        let report = run.reduce(no_warmup()).expect("measured frames");
        let rows: Vec<(&str, Option<&str>, u32)> = report
            .overall
            .passes
            .iter()
            .map(|p| (p.name.as_str(), p.within.as_deref(), p.median_us))
            .collect();
        assert_eq!(
            rows,
            [
                ("main", None, 6_000),
                ("sky", Some("main"), 1_000),
                ("shadow", None, 3_000),
                (GAPS, None, 1_000),
            ]
        );
        let top: f32 = report
            .overall
            .passes
            .iter()
            .filter(|p| p.within.is_none())
            .map(|p| p.share)
            .sum();
        assert!((top - 1.0).abs() < 1e-4);

        // The same frame with the sky folded into main and no sky reading.
        let mut folded = sample(0.0, Some(0), 20_000);
        folded.pass_us[0] = 3_000;
        folded.pass_us[1] = 6_000;
        let mut run = run_of(vec![folded]);
        run.pass_names = vec!["shadow".to_string(), "main".to_string(), "sky".to_string()];
        let metal = run.reduce(no_warmup()).expect("measured frames");
        let gaps = |r: &Report| r.overall.passes.last().map(|p| p.median_us);
        assert_eq!(gaps(&metal), gaps(&report));
    }

    // With no enclosing pass on the list there is nothing to count it twice in.
    #[test]
    fn a_nested_pass_without_its_enclosing_pass_stands_on_its_own() {
        let mut only = sample(0.0, Some(0), 20_000);
        only.pass_us[1] = 4_000;
        let mut run = run_of(vec![only]);
        run.pass_names = vec!["main".to_string(), "sky".to_string()];
        let report = run.reduce(no_warmup()).expect("measured frames");
        let passes = &report.overall.passes;
        assert_eq!(passes[0].name, "sky");
        assert_eq!(passes[0].within, None);
        assert_eq!(passes[1].median_us, 6_000);
    }

    // The slow-frame list follows the phase list's rule: a nested pass whose
    // enclosing pass did not run that frame is listed on its own.
    #[test]
    fn a_slow_frame_lists_a_nested_pass_whose_enclosing_pass_did_not_run() {
        let mut slow = sample(0.0, Some(0), 20_000);
        slow.pass_us[1] = 1_000;
        slow.pass_us[2] = 3_000;
        let mut run = run_of(vec![slow]);
        run.pass_names = vec!["main".to_string(), "sky".to_string(), "shadow".to_string()];
        let report = run.reduce(no_warmup()).expect("measured frames");
        let names: Vec<&str> = report.slowest[0]
            .passes
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(names, ["shadow", "sky"]);
        let phase: Vec<&str> = report
            .overall
            .passes
            .iter()
            .filter(|p| p.within.is_none())
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(phase, ["shadow", "sky", GAPS]);
    }

    #[test]
    fn a_slow_frame_lists_only_the_passes_that_split_its_frame() {
        let mut slow = sample(0.0, Some(0), 20_000);
        slow.pass_us[0] = 6_000;
        slow.pass_us[1] = 1_000;
        let mut run = run_of(vec![slow]);
        run.pass_names = vec!["main".to_string(), "sky".to_string()];
        let report = run.reduce(no_warmup()).expect("measured frames");
        let names: Vec<&str> = report.slowest[0]
            .passes
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(names, ["main"]);
    }

    #[test]
    fn a_pass_that_never_ran_is_left_out_rather_than_listed_as_zero() {
        let mut only = sample(0.0, Some(0), 20_000);
        // The one pass that ran covers the whole 10_000us GPU frame, so the
        // list holds it and nothing else.
        only.pass_us[0] = 10_000;
        let mut run = run_of(vec![only]);
        run.pass_names = vec!["main".to_string(), "fog".to_string()];
        let report = run.reduce(no_warmup()).expect("measured frames");
        let names: Vec<&str> = report
            .overall
            .passes
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(names, ["main"]);
    }

    #[test]
    fn pass_shares_stay_finite_when_the_backend_reports_no_gpu_time() {
        // A backend without timestamp support reports a zero GPU frame; a
        // share is then undefined rather than infinite.
        let mut only = sample(0.0, Some(0), 20_000);
        only.gpu_frame_us = 0;
        only.pass_us[0] = 5_000;
        let mut run = run_of(vec![only]);
        run.pass_names = vec!["main".to_string()];
        let report = run.reduce(no_warmup()).expect("measured frames");
        assert_eq!(report.overall.passes[0].share, 0.0);
    }

    #[test]
    fn systems_are_ranked_by_cost_and_measured_against_what_they_all_stepped() {
        let mut only = sample(0.0, Some(0), 10_000);
        only.gpu_wait_us = 0;
        only.system_us[0] = 400;
        only.system_us[1] = 2_500;
        let mut run = run_of(vec![only]);
        run.system_names = vec!["PhysicsSystem".to_string(), "GraphicsSystem".to_string()];
        let report = run.reduce(no_warmup()).expect("measured frames");
        let systems = &report.overall.systems;
        assert_eq!(systems[0].name, "GraphicsSystem");
        assert_eq!(systems[0].mean_us, 2_500);
        // 2500 of the 2900us the two systems stepped.
        assert!((systems[0].share - 2_500.0 / 2_900.0).abs() < 1e-4);
        assert_eq!(systems[1].name, "PhysicsSystem");
        // Shares are of one measurement, so they account for all of it.
        assert!((systems.iter().map(|s| s.share).sum::<f32>() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn the_render_submission_is_a_row_of_its_own_beside_the_systems() {
        // The submission is CPU work no system's share holds: on its own render
        // thread it is in no system at all, and a system that submits on the
        // stepping thread has that time taken out of its span.
        let mut only = sample(0.0, Some(0), 10_000);
        only.gpu_wait_us = 9_000;
        only.render_cpu_us = 500;
        only.system_us[0] = 300;
        only.system_us[1] = 200;
        let mut run = run_of(vec![only]);
        run.system_names = vec!["PhysicsSystem".to_string(), "GraphicsSystem".to_string()];
        let report = run.reduce(no_warmup()).expect("measured frames");

        // Neither the wait nor the handoff is in it: 1ms of CPU work in a 10ms
        // frame.
        assert_eq!(report.overall.cpu.mean_us, 1_000);
        let systems = &report.overall.systems;
        assert_eq!(systems[0].name, RENDER_SUBMIT);
        assert_eq!(systems[0].mean_us, 500);
        assert!((systems[0].share - 0.5).abs() < 1e-4);
        assert!((systems.iter().map(|s| s.share).sum::<f32>() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn a_run_that_named_no_systems_falls_back_to_the_frame_less_the_wait() {
        let mut fast = sample(0.0, Some(0), 10_000);
        fast.gpu_wait_us = 2_000;
        let mut slow = sample(1.0, Some(0), 20_000);
        slow.gpu_wait_us = 1_000;
        let report = run_of(vec![fast, slow])
            .reduce(no_warmup())
            .expect("measured frames");
        assert!(run_of(vec![]).system_names.is_empty(), "nothing named");
        assert_eq!(report.overall.frame.mean_us, 15_000);
        assert_eq!(report.overall.cpu.mean_us, 13_500);
        assert_eq!(report.overall.cpu.max_us, 19_000);
    }

    // A tick that outran the device is what the column exists to show. Measured
    // from the wall clock it would read as nothing at all: the wait is taken on
    // the thread that submits, not the one that steps.
    #[test]
    fn a_tick_that_outlasts_the_wait_reads_as_cpu_work_rather_than_as_zero() {
        let mut busy = sample(0.0, Some(0), 12_000);
        busy.gpu_wait_us = 11_500;
        busy.system_us[0] = 8_000;
        busy.system_us[1] = 300;
        let mut run = run_of(vec![busy]);
        run.system_names = vec!["BehaviorSystem".to_string(), "GraphicsSystem".to_string()];
        let report = run.reduce(no_warmup()).expect("measured frames");
        assert_eq!(report.overall.cpu.mean_us, 8_300);
        assert_eq!(report.overall.systems[0].name, "BehaviorSystem");
    }

    #[test]
    fn a_system_that_is_a_rounding_error_beside_the_rest_is_left_out() {
        let mut only = sample(0.0, Some(0), 20_000);
        only.gpu_wait_us = 0;
        only.system_us[0] = 5_000;
        only.system_us[1] = 10;
        let mut run = run_of(vec![only]);
        run.system_names = vec!["PhysicsSystem".to_string(), "AudioSystem".to_string()];
        let report = run.reduce(no_warmup()).expect("measured frames");
        let names: Vec<&str> = report
            .overall
            .systems
            .iter()
            .map(|s| s.name.as_str())
            .collect();
        assert_eq!(names, ["PhysicsSystem"]);
    }

    #[test]
    fn a_run_whose_systems_were_never_timed_reports_no_breakdown() {
        // A build without per-system timing reports every span as zero; a
        // ranking of zeroes would be worse than saying nothing.
        let mut run = run_of(vec![sample(0.0, Some(0), 20_000)]);
        run.system_names = vec!["PhysicsSystem".to_string()];
        let report = run.reduce(no_warmup()).expect("measured frames");
        assert!(report.overall.systems.is_empty());
    }

    fn fan_out(wall_us: u32) -> FanOutTiming {
        FanOutTiming {
            wall_us,
            first_job_us: wall_us / 2,
            job_sum_us: wall_us / 4,
            longest_job_us: wall_us / 8,
            tail_us: wall_us / 10,
        }
    }

    #[test]
    fn a_phase_reports_each_fan_outs_means_with_the_recording_first() {
        let mut a = sample(0.0, Some(0), 10_000);
        a.recording_fan_out = fan_out(2_000);
        a.system_fan_out[1] = fan_out(800);
        let mut b = sample(1.0, Some(0), 10_000);
        b.recording_fan_out = fan_out(4_000);
        b.system_fan_out[1] = fan_out(400);
        let mut run = run_of(vec![a, b]);
        run.system_names = vec!["PhysicsSystem".to_string(), "BehaviorSystem".to_string()];
        let report = run.reduce(no_warmup()).expect("measured frames");
        // PhysicsSystem fanned nothing out, so it gets no row.
        assert_eq!(
            report.phases[0].fan_outs,
            [
                FanOutCost::of("render recording", fan_out(3_000)),
                FanOutCost::of("BehaviorSystem", fan_out(600)),
            ]
        );
    }

    // Packing runs every frame, inline or fanned out, so it is listed after
    // the recording and before any system.
    #[test]
    fn the_render_packing_is_listed_after_the_recording() {
        let mut a = sample(0.0, Some(0), 10_000);
        a.recording_fan_out = fan_out(2_000);
        a.packing_fan_out = fan_out(100);
        a.system_fan_out[0] = fan_out(800);
        let mut b = sample(1.0, Some(0), 10_000);
        b.recording_fan_out = fan_out(2_000);
        b.packing_fan_out = fan_out(100);
        let mut run = run_of(vec![a, b]);
        run.system_names = vec!["BehaviorSystem".to_string()];
        let report = run.reduce(no_warmup()).expect("measured frames");
        assert_eq!(
            report.overall.fan_outs,
            [
                FanOutCost::of("render recording", fan_out(2_000)),
                FanOutCost::of("render packing", fan_out(100)),
                FanOutCost::of("BehaviorSystem", fan_out(800)),
            ]
        );
    }

    #[test]
    fn a_fan_out_is_averaged_over_the_frames_it_ran_in() {
        let mut fanned = sample(0.0, Some(0), 10_000);
        fanned.system_fan_out[0] = fan_out(800);
        let serial = sample(1.0, Some(0), 10_000);
        let mut run = run_of(vec![fanned, serial]);
        run.system_names = vec!["BehaviorSystem".to_string()];
        let report = run.reduce(no_warmup()).expect("measured frames");
        assert_eq!(
            report.overall.fan_outs,
            [FanOutCost::of("BehaviorSystem", fan_out(800))]
        );
    }

    #[test]
    fn a_run_with_no_fan_outs_reports_none() {
        let run = run_of(vec![sample(0.0, Some(0), 10_000)]);
        let report = run.reduce(no_warmup()).expect("measured frames");
        assert!(report.overall.fan_outs.is_empty());
        assert!(report.slowest[0].fan_outs.is_empty());
    }

    #[test]
    fn a_slow_frame_carries_its_own_fan_outs() {
        let mut quick = sample(0.0, Some(0), 10_000);
        quick.recording_fan_out = fan_out(300);
        let mut hitch = sample(1.0, Some(0), 40_000);
        hitch.recording_fan_out = fan_out(3_400);
        hitch.system_fan_out[0] = fan_out(2_400);
        let mut run = run_of(vec![quick, hitch]);
        run.system_names = vec!["BehaviorSystem".to_string()];
        let report = run.reduce(no_warmup()).expect("measured frames");
        assert_eq!(
            report.slowest[0].fan_outs,
            [
                FanOutCost::of("render recording", fan_out(3_400)),
                FanOutCost::of("BehaviorSystem", fan_out(2_400)),
            ]
        );
        assert_eq!(
            report.slowest[1].fan_outs,
            [FanOutCost::of("render recording", fan_out(300))]
        );
    }

    #[test]
    fn a_slow_frame_names_the_passes_that_took_longest_to_record() {
        let mut hitch = sample(1.0, Some(0), 40_000);
        hitch.pass_record_us[PassId::Main as usize] = 8_000;
        hitch.pass_record_us[PassId::Shadow as usize] = 3_000;
        hitch.pass_record_us[PassId::Bloom as usize] = 20;
        let run = run_of(vec![sample(0.0, Some(0), 10_000), hitch]);
        let report = run.reduce(no_warmup()).expect("measured frames");
        let cost = |name: &str, us| FrameCost {
            name: name.to_string(),
            us,
        };
        assert_eq!(
            report.slowest[0].recording,
            [
                cost("main", 8_000),
                cost("shadow", 3_000),
                cost("bloom", 20)
            ]
        );
        assert!(report.slowest[1].recording.is_empty());
    }

    #[test]
    fn an_incomplete_run_says_so() {
        let mut run = run_of(vec![sample(0.0, Some(0), 10_000)]);
        run.completed = false;
        let report = run.reduce(no_warmup()).expect("measured frames");
        assert!(!report.completed);
    }

    #[test]
    fn the_slowest_frames_are_named_slowest_first_and_capped() {
        let run = run_of(vec![
            sample(0.0, Some(0), 10_000),
            sample(1.0, Some(1), 300_000),
            sample(2.0, Some(0), 12_000),
            sample(3.0, Some(0), 40_000),
            sample(4.0, Some(1), 40_000),
        ]);
        let report = run.reduce(no_warmup()).expect("measured frames");
        let slowest: Vec<(u32, f32, &str)> = report
            .slowest
            .iter()
            .map(|f| (f.frame_us, f.run_seconds, f.phase.as_str()))
            .collect();
        // Equal frames keep the order they were drawn in.
        assert_eq!(
            slowest,
            [
                (300_000, 1.0, "rays"),
                (40_000, 3.0, "shadows"),
                (40_000, 4.0, "rays")
            ]
        );
    }

    #[test]
    fn warmup_frames_are_never_named_among_the_slowest() {
        let run = run_of(vec![
            sample(0.5, Some(0), 90_000),
            sample(3.0, Some(0), 12_000),
        ]);
        let report = run
            .reduce(ReduceOptions::default())
            .expect("measured frames");
        assert_eq!(report.slowest.len(), 1);
        assert_eq!(report.slowest[0].frame_us, 12_000);
    }

    // A hitch is diagnosable from the report alone when its frame says which
    // side held it: here a 300 ms wait on the GPU, not CPU work.
    #[test]
    fn a_slow_frame_carries_its_own_cpu_wait_and_pass_breakdown() {
        let mut hitch = sample(0.0, Some(0), 300_000);
        hitch.gpu_wait_us = 297_000;
        hitch.gpu_frame_us = 9_000;
        hitch.render_cpu_us = 700;
        hitch.replay_us = 200;
        hitch.recording_fan_out.wall_us = 350;
        hitch.system_us[0] = 1_200;
        hitch.system_us[1] = 0;
        hitch.system_us[2] = 400;
        hitch.pass_us[0] = 1_000;
        hitch.pass_us[1] = 5_000;
        let mut run = run_of(vec![hitch]);
        run.system_names = vec![
            "PhysicsSystem".to_string(),
            "AudioSystem".to_string(),
            "GraphicsSystem".to_string(),
        ];
        run.pass_names = vec!["shadow".to_string(), "main".to_string(), String::new()];
        let report = run.reduce(no_warmup()).expect("measured frames");

        let frame = &report.slowest[0];
        assert_eq!(frame.frame_us, 300_000);
        assert_eq!(frame.gpu_wait_us, 297_000);
        assert_eq!(frame.gpu_frame_us, 9_000);
        assert_eq!(frame.cpu_us, 2_300);
        let cpu: Vec<(&str, u32)> = frame.cpu.iter().map(|c| (c.name.as_str(), c.us)).collect();
        // A system that did nothing this frame is left out.
        assert_eq!(
            cpu,
            [
                ("PhysicsSystem", 1_200),
                (RENDER_SUBMIT, 700),
                ("GraphicsSystem", 400)
            ]
        );
        let passes: Vec<(&str, u32)> = frame
            .passes
            .iter()
            .map(|p| (p.name.as_str(), p.us))
            .collect();
        assert_eq!(passes, [("main", 5_000), ("shadow", 1_000)]);
        // The submission's own spans; a span that took nothing is left out.
        let render: Vec<(&str, u32)> = frame
            .render
            .iter()
            .map(|r| (r.name.as_str(), r.us))
            .collect();
        assert_eq!(render, [("recording", 350), ("replay", 200)]);
    }

    #[test]
    fn a_slow_frame_lists_only_its_largest_costs() {
        let mut busy = sample(0.0, Some(0), 20_000);
        let mut run_names = Vec::new();
        for slot in 0..6 {
            busy.system_us[slot] = 100 * (u32::saturating_from(slot) + 1);
            run_names.push(format!("System{slot}"));
        }
        let mut run = run_of(vec![busy]);
        run.system_names = run_names;
        let report = run.reduce(no_warmup()).expect("measured frames");
        let names: Vec<&str> = report.slowest[0]
            .cpu
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        assert_eq!(names, ["System5", "System4", "System3", "System2"]);
    }

    #[test]
    fn peak_memory_is_the_largest_reading_not_the_last() {
        let mut early = sample(0.0, Some(0), 10_000);
        early.vram_bytes = 900 << 20;
        let run = run_of(vec![early, sample(1.0, Some(0), 10_000)]);
        let report = run.reduce(no_warmup()).expect("measured frames");
        assert_eq!(report.overall.vram_peak_bytes, 900 << 20);
    }
}
