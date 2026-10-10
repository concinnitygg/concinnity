// Samples in, report out. Pure: no clock, no world, no I/O.

use concinnity_core::profile::FanOutTiming;
use concinnity_core::render::render_graph::PassId;
use serde::Serialize;

use crate::frame_report::sample::{FrameRun, FrameSample};
use crate::frame_report::stats::{Distribution, mean_u32};

/// How many passes a segment lists. The tail is noise next to the handful that
/// own the frame.
const PASSES_REPORTED: usize = 8;

/// How many systems a segment lists, on the same reasoning.
const SYSTEMS_REPORTED: usize = 6;

/// The row that closes a pass list: the median GPU frame less every pass listed
/// above it. It covers the passes that fell below the cut, the ones the backend
/// times no part of, and the gaps between them. Without it a list of the eight
/// largest passes reads as if it were the whole frame.
const UNATTRIBUTED: &str = "unattributed";

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

/// How many of the run's slowest frames the report names.
const SLOW_FRAMES_REPORTED: usize = 3;

/// How many CPU rows, and how many passes, a slow frame lists.
const SLOW_FRAME_ROWS: usize = 4;

/// What a run is reduced against.
#[derive(Debug, Clone, Copy)]
pub struct ReduceOptions {
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

/// One pass's contribution to a segment's GPU frame.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PassShare {
    /// The backend's name for the pass.
    pub name: String,
    /// Median GPU microseconds the pass took.
    ///
    /// A median rather than a mean because a backend that mistimes one frame
    /// mistimes it by orders of magnitude, and a handful of those in a segment
    /// of a few hundred frames moves a mean past the slowest frame the segment
    /// actually drew.
    pub median_us: u32,
    /// That median as a share of the segment's median GPU frame, in `0..=1`.
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
}

/// One system's CPU contribution to a segment's frame, or the render side's
/// under the `render submit` row.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SystemCost {
    /// The schedule's name for the system, or `render submit`.
    pub name: String,
    /// Mean CPU microseconds the system's step took.
    pub mean_us: u32,
    /// That mean as a share of the segment's mean CPU work, in `0..=1`.
    pub share: f32,
}

/// A parallel fan-out's wait against its work: the render side's pass
/// recording, or a system that fanned its own work out. Microseconds, either
/// within one frame or as means over the frames of a segment it ran in.
///
/// A wall time well above the longest job is a fan-out not limited by its
/// work; `first_job_us` and `tail_us` say whether the gap is the workers
/// starting late or the waiting thread resuming late.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FanOutCost {
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
pub struct FrameCost {
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
pub struct SlowFrame {
    /// How far into the run the frame was drawn, in seconds.
    pub run_seconds: f32,
    /// The segment it was drawn in.
    pub segment: String,
    /// Its wall-clock frame time.
    pub frame_us: u32,
    /// The CPU work in it, summed the way the segment table's column is.
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
    /// The largest passes of the GPU frame it reported, largest first, leaving
    /// out a pass timed inside another.
    pub passes: Vec<FrameCost>,
    /// The fan-outs it waited on, the render recording first. Ones that did
    /// not run are left out.
    pub fan_outs: Vec<FanOutCost>,
    /// The passes that took longest to record on the CPU, largest first: which
    /// jobs made up the render recording's job sum.
    pub recording: Vec<FrameCost>,
}

/// What one stretch of the run measured.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SegmentReport {
    /// The segment's authored name, or a placeholder.
    pub name: String,
    /// Seconds of track time the segment covered.
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

/// A whole run, reduced.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Report {
    /// Whether the run reached the end of what it had to measure.
    pub completed: bool,
    /// Frames dropped as warm-up.
    pub warmup_dropped: usize,
    /// The budget every distribution was counted against.
    pub budget_us: u32,
    /// Every measured frame, as one summary.
    pub overall: SegmentReport,
    /// The same, cut into the stretches the run named. A cost that shows
    /// only here belongs to one of them; one that shows only in `overall` is
    /// spread across all of them.
    pub segments: Vec<SegmentReport>,
    /// The slowest measured frames, slowest first: a hitch the distributions
    /// can only show as a maximum, named with what it spent.
    pub slowest: Vec<SlowFrame>,
}

impl Report {
    /// Reduce a run. Returns `None` when the warm-up discard left nothing,
    /// which is a run too short to say anything about rather than a run of
    /// zeroes.
    pub fn of(run: &FrameRun, options: ReduceOptions) -> Option<Self> {
        let measured: Vec<&FrameSample> = run
            .samples
            .iter()
            .filter(|s| s.run_seconds >= options.warmup_seconds)
            .collect();
        if measured.is_empty() {
            return None;
        }
        let mut segments = Vec::new();
        for (index, name) in segment_order(&measured, run) {
            let frames: Vec<&FrameSample> = measured
                .iter()
                .copied()
                .filter(|s| s.segment == index)
                .collect();
            segments.push(summarize(name, &frames, run, options));
        }
        Some(Self {
            completed: run.completed,
            warmup_dropped: run.samples.len() - measured.len(),
            budget_us: options.budget_us,
            overall: summarize("all".to_string(), &measured, run, options),
            segments,
            slowest: slowest_frames(&measured, run),
        })
    }
}

// The slowest frames, slowest first; the earlier of two equal ones first.
fn slowest_frames(measured: &[&FrameSample], run: &FrameRun) -> Vec<SlowFrame> {
    let mut by_time: Vec<&FrameSample> = measured.to_vec();
    by_time.sort_by_key(|s| core::cmp::Reverse(s.frame_us));
    by_time.truncate(SLOW_FRAMES_REPORTED);
    by_time
        .into_iter()
        .map(|sample| SlowFrame {
            run_seconds: sample.run_seconds,
            segment: run.segment_name(sample.segment).to_string(),
            frame_us: sample.frame_us,
            cpu_us: stepped_us(sample, run),
            gpu_wait_us: sample.gpu_wait_us,
            gpu_frame_us: sample.gpu_frame_us,
            cpu: largest(
                named_slots(&run.system_names)
                    .map(|(slot, name)| (name, sample.system_us[slot]))
                    .chain(core::iter::once((RENDER_SUBMIT, sample.render_cpu_us))),
            ),
            passes: {
                let us = |name: &str| {
                    named_slots(&run.pass_names)
                        .find(|(_, n)| *n == name)
                        .map_or(0, |(slot, _)| sample.pass_us[slot])
                };
                largest(
                    named_slots(&run.pass_names)
                        .filter(|(_, name)| counted_in(name, |outer| us(outer) > 0).is_none())
                        .map(|(slot, name)| (name, sample.pass_us[slot])),
                )
            },
            fan_outs: fan_out_slots(run)
                .map(|(slot, name)| FanOutCost::of(name, fan_out_in(sample, slot)))
                .filter(|cost| cost.wall_us > 0)
                .collect(),
            recording: largest(
                PassId::ALL
                    .iter()
                    .map(|id| (id.name(), sample.pass_record_us[*id as usize])),
            ),
        })
        .collect()
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

// The fan-outs over a stretch, each as means over the frames it ran in,
// leaving out the ones that never ran.
fn fan_out_means(frames: &[&FrameSample], run: &FrameRun) -> Vec<FanOutCost> {
    fan_out_slots(run)
        .map(|(slot, name)| {
            let mean = |field: fn(FanOutTiming) -> u32| {
                mean_u32(
                    frames
                        .iter()
                        .map(|s| fan_out_in(s, slot))
                        .filter(|t| t.wall_us > 0)
                        .map(field),
                )
            };
            FanOutCost {
                name: name.to_string(),
                wall_us: mean(|t| t.wall_us),
                first_job_us: mean(|t| t.first_job_us),
                job_sum_us: mean(|t| t.job_sum_us),
                longest_job_us: mean(|t| t.longest_job_us),
                tail_us: mean(|t| t.tail_us),
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

// The segments the measured frames fall in, in the order they were first
// entered, so the report reads along the path rather than alphabetically.
fn segment_order(measured: &[&FrameSample], run: &FrameRun) -> Vec<(Option<u32>, String)> {
    let mut seen: Vec<(Option<u32>, String)> = Vec::new();
    for sample in measured {
        if !seen.iter().any(|(index, _)| *index == sample.segment) {
            seen.push((sample.segment, run.segment_name(sample.segment).to_string()));
        }
    }
    seen
}

// Reduce one stretch of frames.
fn summarize(
    name: String,
    frames: &[&FrameSample],
    run: &FrameRun,
    options: ReduceOptions,
) -> SegmentReport {
    let mut frame_us: Vec<u32> = frames.iter().map(|s| s.frame_us).collect();
    let mut cpu_us: Vec<u32> = frames.iter().map(|s| stepped_us(s, run)).collect();
    let mut gpu_us: Vec<u32> = frames.iter().map(|s| s.gpu_frame_us).collect();
    let gpu = Distribution::of(&mut gpu_us, options.budget_us);
    let cpu = Distribution::of(&mut cpu_us, options.budget_us);
    let frame = Distribution::of(&mut frame_us, options.budget_us);
    SegmentReport {
        name,
        seconds: span_seconds(frames),
        frame,
        cpu,
        gpu,
        gpu_wait_mean_us: mean_u32(frames.iter().map(|s| s.gpu_wait_us)),
        draw_calls_mean: mean_u32(frames.iter().map(|s| s.draw_calls)),
        objects_mean: mean_u32(frames.iter().map(|s| s.objects)),
        vram_peak_bytes: frames.iter().map(|s| s.vram_bytes).max().unwrap_or(0),
        passes: pass_shares(frames, run, gpu.p50_us, options.budget_us),
        systems: system_costs(frames, run),
        fan_outs: fan_out_means(frames, run),
    }
}

// What one frame's CPU work came to: its systems and its render submission.
//
// Taken from the work rather than from the wall clock: the wall delta is
// measured on the thread that steps the world, while the wait is measured on
// the one that submits, so subtracting the second from the first reports a
// tick that outran the device as having done no work at all. A run whose host
// named no systems has nothing to sum and falls back to that outside view.
fn stepped_us(sample: &FrameSample, run: &FrameRun) -> u32 {
    if run.system_names.is_empty() {
        return sample.frame_us.saturating_sub(sample.gpu_wait_us);
    }
    named_slots(&run.system_names)
        .map(|(slot, _)| sample.system_us[slot])
        .fold(sample.render_cpu_us, u32::saturating_add)
}

// The systems and the render submission that owned the frame's CPU work,
// largest first, measured against that work rather than against wall time.
fn system_costs(frames: &[&FrameSample], run: &FrameRun) -> Vec<SystemCost> {
    if run.system_names.is_empty() {
        return Vec::new();
    }
    let means: Vec<(&str, u32)> = named_slots(&run.system_names)
        .map(|(slot, name)| (name, mean_u32(frames.iter().map(|s| s.system_us[slot]))))
        .chain(core::iter::once((
            RENDER_SUBMIT,
            mean_u32(frames.iter().map(|s| s.render_cpu_us)),
        )))
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

// Track time from the first frame of a stretch to its last.
fn span_seconds(frames: &[&FrameSample]) -> f32 {
    match (frames.first(), frames.last()) {
        (Some(first), Some(last)) => last.run_seconds - first.run_seconds,
        _ => 0.0,
    }
}

// The passes that owned the GPU frame, largest first, with each one's share of
// it, and a pass timed inside another listed right below it. Passes that never
// ran are left out rather than listed as zero.
fn pass_shares(
    frames: &[&FrameSample],
    run: &FrameRun,
    gpu_median_us: u32,
    budget_us: u32,
) -> Vec<PassShare> {
    let mut column: Vec<u32> = Vec::with_capacity(frames.len());
    let mut shares: Vec<PassShare> = run
        .pass_names
        .iter()
        .enumerate()
        .filter(|(_, name)| !name.is_empty())
        .filter_map(|(slot, name)| {
            column.clear();
            column.extend(frames.iter().map(|s| s.pass_us[slot]));
            let median_us = Distribution::of(&mut column, budget_us).p50_us;
            if median_us == 0 {
                return None;
            }
            Some(PassShare {
                name: name.clone(),
                median_us,
                share: if gpu_median_us == 0 {
                    0.0
                } else {
                    median_us as f32 / gpu_median_us as f32
                },
                within: None,
            })
        })
        .collect();
    shares.sort_unstable_by_key(|p| core::cmp::Reverse(p.median_us));
    let mut shares = nest(shares);

    // Passes on separate queues overlap, so their times can sum past the frame
    // they ran in; there is nothing left over to report when they do. A backend
    // that timed nothing gets no row either: the remainder corrects a partial
    // list rather than standing in for one.
    let listed: u32 = shares
        .iter()
        .filter(|p| p.within.is_none())
        .map(|p| p.median_us)
        .sum();
    let rest = gpu_median_us.saturating_sub(listed);
    if rest > 0 && !shares.is_empty() {
        shares.push(PassShare {
            name: UNATTRIBUTED.to_string(),
            median_us: rest,
            share: rest as f32 / gpu_median_us as f32,
            within: None,
        });
    }
    shares
}

// Cut a cost-ordered pass list to the passes a segment reports, each followed
// by the passes timed inside it (see `counted_in`).
fn nest(shares: Vec<PassShare>) -> Vec<PassShare> {
    let ran: Vec<String> = shares.iter().map(|p| p.name.clone()).collect();
    let within = |name: &str| counted_in(name, |outer| ran.iter().any(|n| n == outer));
    let (mut inner, top): (Vec<PassShare>, Vec<PassShare>) =
        shares.into_iter().partition(|p| within(&p.name).is_some());
    let mut out = Vec::with_capacity(top.len() + inner.len());
    for pass in top.into_iter().take(PASSES_REPORTED) {
        let outer = pass.name.clone();
        out.push(pass);
        for mut p in inner.extract_if(.., |p| within(&p.name) == Some(outer.as_str())) {
            p.within = Some(outer.clone());
            out.push(p);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_report::sample::MAX_SYSTEM_TIMINGS;
    use concinnity_core::profile::MAX_PASS_TIMINGS;

    // A sample with the fields a test cares about and zeroes elsewhere.
    fn sample(run_seconds: f32, segment: Option<u32>, frame_us: u32) -> FrameSample {
        FrameSample {
            run_seconds,
            segment,
            frame_us,
            gpu_frame_us: frame_us / 2,
            gpu_wait_us: 100,
            render_cpu_us: 0,
            recording_fan_out: FanOutTiming::default(),
            packing_fan_out: FanOutTiming::default(),
            pass_record_us: [0; MAX_PASS_TIMINGS],
            draw_calls: 50,
            objects: 400,
            vram_bytes: 1 << 20,
            pass_us: [0; MAX_PASS_TIMINGS],
            system_us: [0; MAX_SYSTEM_TIMINGS],
            system_fan_out: [FanOutTiming::default(); MAX_SYSTEM_TIMINGS],
        }
    }

    fn run_of(samples: Vec<FrameSample>) -> FrameRun {
        FrameRun {
            samples,
            segments: vec!["shadows".to_string(), "rays".to_string()],
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
        assert!(Report::of(&run, ReduceOptions::default()).is_none());
    }

    #[test]
    fn warmup_frames_are_dropped_and_counted() {
        let run = run_of(vec![
            sample(0.5, Some(0), 90_000),
            sample(1.9, Some(0), 80_000),
            sample(2.0, Some(0), 10_000),
            sample(3.0, Some(0), 12_000),
        ]);
        let report = Report::of(&run, ReduceOptions::default()).expect("measured frames");
        assert_eq!(report.warmup_dropped, 2);
        assert_eq!(report.overall.frame.count, 2);
        // The slow opening frames are gone, so they cannot drag the summary.
        assert_eq!(report.overall.frame.max_us, 12_000);
    }

    #[test]
    fn segments_come_out_in_the_order_the_path_entered_them() {
        // Declaration order would put shadows first either way, so the track
        // visits them the other way round to tell the two apart.
        let run = run_of(vec![
            sample(0.0, Some(1), 10_000),
            sample(1.0, Some(0), 20_000),
            sample(2.0, Some(1), 11_000),
        ]);
        let report = Report::of(&run, no_warmup()).expect("measured frames");
        let names: Vec<&str> = report.segments.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["rays", "shadows"]);
    }

    #[test]
    fn a_segment_summarizes_only_its_own_frames() {
        // A cost in one stretch has to be visible there rather than averaged
        // into the run.
        let run = run_of(vec![
            sample(0.0, Some(0), 10_000),
            sample(1.0, Some(0), 10_000),
            sample(2.0, Some(1), 40_000),
            sample(3.0, Some(1), 40_000),
        ]);
        let report = Report::of(&run, no_warmup()).expect("measured frames");
        let shadows = &report.segments[0];
        let rays = &report.segments[1];
        assert_eq!(shadows.frame.mean_us, 10_000);
        assert_eq!(rays.frame.mean_us, 40_000);
        assert_eq!((shadows.frame.count, rays.frame.count), (2, 2));
        // The overall summary is still over everything.
        assert_eq!(report.overall.frame.count, 4);
        assert_eq!(report.overall.frame.mean_us, 25_000);
    }

    #[test]
    fn a_segments_span_is_its_own_track_time() {
        let run = run_of(vec![
            sample(0.0, Some(0), 10_000),
            sample(4.0, Some(0), 10_000),
            sample(5.0, Some(1), 10_000),
        ]);
        let report = Report::of(&run, no_warmup()).expect("measured frames");
        assert_eq!(report.segments[0].seconds, 4.0);
        // One frame spans no time, which is honest rather than an error.
        assert_eq!(report.segments[1].seconds, 0.0);
    }

    #[test]
    fn frames_in_no_segment_are_reported_under_a_placeholder() {
        let run = run_of(vec![sample(0.0, None, 10_000)]);
        let report = Report::of(&run, no_warmup()).expect("measured frames");
        assert_eq!(report.segments[0].name, "(unnamed)");
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
        let report = Report::of(&run, no_warmup()).expect("measured frames");
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
        let report = Report::of(&run, no_warmup()).expect("measured frames");
        let passes = &report.overall.passes;
        assert_eq!(passes.len(), 2);
        assert_eq!(passes[1].name, UNATTRIBUTED);
        assert_eq!(passes[1].median_us, 7_500);
        assert!((passes.iter().map(|p| p.share).sum::<f32>() - 1.0).abs() < 1e-4);
    }

    // Passes on separate queues overlap, so a sum past the frame is ordinary
    // rather than a mistake, and there is nothing left over to name.
    #[test]
    fn overlapping_passes_that_outlast_the_frame_leave_no_remainder() {
        let mut only = sample(0.0, Some(0), 20_000);
        only.pass_us[0] = 8_000;
        only.pass_us[1] = 7_000;
        let mut run = run_of(vec![only]);
        run.pass_names = vec!["main".to_string(), "async".to_string()];
        let report = Report::of(&run, no_warmup()).expect("measured frames");
        let names: Vec<&str> = report
            .overall
            .passes
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(names, ["main", "async"]);
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
        let report = Report::of(&run, no_warmup()).expect("measured frames");
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
                (UNATTRIBUTED, None, 1_000),
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
        let metal = Report::of(&run, no_warmup()).expect("measured frames");
        let unattributed = |r: &Report| r.overall.passes.last().map(|p| p.median_us);
        assert_eq!(unattributed(&metal), unattributed(&report));
    }

    // With no enclosing pass on the list there is nothing to count it twice in.
    #[test]
    fn a_nested_pass_without_its_enclosing_pass_stands_on_its_own() {
        let mut only = sample(0.0, Some(0), 20_000);
        only.pass_us[1] = 4_000;
        let mut run = run_of(vec![only]);
        run.pass_names = vec!["main".to_string(), "sky".to_string()];
        let report = Report::of(&run, no_warmup()).expect("measured frames");
        let passes = &report.overall.passes;
        assert_eq!(passes[0].name, "sky");
        assert_eq!(passes[0].within, None);
        assert_eq!(passes[1].median_us, 6_000);
    }

    // The slow-frame list follows the segment list's rule: a nested pass whose
    // enclosing pass did not run that frame is listed on its own.
    #[test]
    fn a_slow_frame_lists_a_nested_pass_whose_enclosing_pass_did_not_run() {
        let mut slow = sample(0.0, Some(0), 20_000);
        slow.pass_us[1] = 1_000;
        slow.pass_us[2] = 3_000;
        let mut run = run_of(vec![slow]);
        run.pass_names = vec!["main".to_string(), "sky".to_string(), "shadow".to_string()];
        let report = Report::of(&run, no_warmup()).expect("measured frames");
        let names: Vec<&str> = report.slowest[0]
            .passes
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(names, ["shadow", "sky"]);
        let segment: Vec<&str> = report
            .overall
            .passes
            .iter()
            .filter(|p| p.within.is_none())
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(segment, ["shadow", "sky", UNATTRIBUTED]);
    }

    #[test]
    fn a_slow_frame_lists_only_the_passes_that_split_its_frame() {
        let mut slow = sample(0.0, Some(0), 20_000);
        slow.pass_us[0] = 6_000;
        slow.pass_us[1] = 1_000;
        let mut run = run_of(vec![slow]);
        run.pass_names = vec!["main".to_string(), "sky".to_string()];
        let report = Report::of(&run, no_warmup()).expect("measured frames");
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
        let report = Report::of(&run, no_warmup()).expect("measured frames");
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
        let report = Report::of(&run, no_warmup()).expect("measured frames");
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
        let report = Report::of(&run, no_warmup()).expect("measured frames");
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
        let report = Report::of(&run, no_warmup()).expect("measured frames");

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
        let report = Report::of(&run_of(vec![fast, slow]), no_warmup()).expect("measured frames");
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
        let report = Report::of(&run, no_warmup()).expect("measured frames");
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
        let report = Report::of(&run, no_warmup()).expect("measured frames");
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
        let report = Report::of(&run, no_warmup()).expect("measured frames");
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
    fn a_segment_reports_each_fan_outs_means_with_the_recording_first() {
        let mut a = sample(0.0, Some(0), 10_000);
        a.recording_fan_out = fan_out(2_000);
        a.system_fan_out[1] = fan_out(800);
        let mut b = sample(1.0, Some(0), 10_000);
        b.recording_fan_out = fan_out(4_000);
        b.system_fan_out[1] = fan_out(400);
        let mut run = run_of(vec![a, b]);
        run.system_names = vec!["PhysicsSystem".to_string(), "BehaviorSystem".to_string()];
        let report = Report::of(&run, no_warmup()).expect("measured frames");
        // PhysicsSystem fanned nothing out, so it gets no row.
        assert_eq!(
            report.segments[0].fan_outs,
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
        let report = Report::of(&run, no_warmup()).expect("measured frames");
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
        let report = Report::of(&run, no_warmup()).expect("measured frames");
        assert_eq!(
            report.overall.fan_outs,
            [FanOutCost::of("BehaviorSystem", fan_out(800))]
        );
    }

    #[test]
    fn a_run_with_no_fan_outs_reports_none() {
        let run = run_of(vec![sample(0.0, Some(0), 10_000)]);
        let report = Report::of(&run, no_warmup()).expect("measured frames");
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
        let report = Report::of(&run, no_warmup()).expect("measured frames");
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
        let report = Report::of(&run, no_warmup()).expect("measured frames");
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
        let report = Report::of(&run, no_warmup()).expect("measured frames");
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
        let report = Report::of(&run, no_warmup()).expect("measured frames");
        let slowest: Vec<(u32, f32, &str)> = report
            .slowest
            .iter()
            .map(|f| (f.frame_us, f.run_seconds, f.segment.as_str()))
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
        let report = Report::of(&run, ReduceOptions::default()).expect("measured frames");
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
        let report = Report::of(&run, no_warmup()).expect("measured frames");

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
    }

    #[test]
    fn a_slow_frame_lists_only_its_largest_costs() {
        let mut busy = sample(0.0, Some(0), 20_000);
        let mut run_names = Vec::new();
        for slot in 0..6 {
            busy.system_us[slot] = 100 * (slot as u32 + 1);
            run_names.push(format!("System{slot}"));
        }
        let mut run = run_of(vec![busy]);
        run.system_names = run_names;
        let report = Report::of(&run, no_warmup()).expect("measured frames");
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
        let report = Report::of(&run, no_warmup()).expect("measured frames");
        assert_eq!(report.overall.vram_peak_bytes, 900 << 20);
    }
}
