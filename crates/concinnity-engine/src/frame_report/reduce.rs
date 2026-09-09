// Samples in, report out. Pure: no clock, no world, no I/O.

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

/// The system whose measured span contains the frame's blocked-on-GPU wait.
///
/// That wait is wall time inside the graphics system's own step, so a
/// GPU-bound frame reports as CPU-bound unless it is subtracted: without this
/// the graphics system reads as nearly the whole frame on every scene, which
/// says nothing about either.
const GPU_WAIT_BEARER: &str = "GraphicsSystem";

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
}

/// One system's CPU contribution to a segment's frame.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SystemCost {
    /// The schedule's name for the system.
    pub name: String,
    /// Mean CPU microseconds the system's step took.
    pub mean_us: u32,
    /// That mean as a share of everything the segment's systems stepped, in
    /// `0..=1`.
    ///
    /// The denominator is the summed system spans rather than the frame's CPU
    /// time, so numerator and denominator come from one measurement. The two
    /// are sampled a frame apart, and a share across that seam can exceed the
    /// whole.
    pub share: f32,
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
    /// blocked-on-GPU wait taken out of the one that carries it. Larger than
    /// `frame` is impossible; approaching it means the tick, not the device, is
    /// what sets the frame rate.
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
    /// The systems that owned the CPU frame, largest first. Empty on a stretch
    /// where no system was a meaningful share of it.
    pub systems: Vec<SystemCost>,
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
        })
    }
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
    }
}

// What one frame's systems spent between them.
//
// Taken from the systems rather than from the wall clock: the wall delta is
// measured on the thread that steps the world, while the wait is measured on
// the one that submits, so subtracting the second from the first reports a
// tick that outran the device as having done no work at all. A run whose host
// named no systems has nothing to sum and falls back to that outside view.
fn stepped_us(sample: &FrameSample, run: &FrameRun) -> u32 {
    if run.system_names.is_empty() {
        return sample.frame_us.saturating_sub(sample.gpu_wait_us);
    }
    run.system_names
        .iter()
        .enumerate()
        .filter(|(_, name)| !name.is_empty())
        .map(|(slot, name)| system_us(sample, slot, name))
        .sum()
}

// One system's span in one frame. The graphics system's span has the
// blocked-on-GPU wait taken out of it, or it would dominate every scene by
// holding time the GPU spent.
fn system_us(sample: &FrameSample, slot: usize, name: &str) -> u32 {
    let step = sample.system_us[slot];
    if name == GPU_WAIT_BEARER {
        step.saturating_sub(sample.gpu_wait_us)
    } else {
        step
    }
}

// The systems that owned the frame's CPU work, largest first, measured against
// that work rather than against wall time. The graphics system's span has the
// blocked-on-GPU wait taken out of it first, or it would dominate every scene
// by holding time the GPU spent.
fn system_costs(frames: &[&FrameSample], run: &FrameRun) -> Vec<SystemCost> {
    let means: Vec<(&String, u32)> = run
        .system_names
        .iter()
        .enumerate()
        .filter(|(_, name)| !name.is_empty())
        .map(|(slot, name)| {
            let mean_us = mean_u32(frames.iter().map(|s| system_us(s, slot, name)));
            (name, mean_us)
        })
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
                name: name.clone(),
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
// it. Passes that never ran are left out rather than listed as zero.
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
            })
        })
        .collect();
    shares.sort_unstable_by_key(|p| core::cmp::Reverse(p.median_us));
    shares.truncate(PASSES_REPORTED);

    // Passes on separate queues overlap, so their times can sum past the frame
    // they ran in; there is nothing left over to report when they do. A backend
    // that timed nothing gets no row either: the remainder corrects a partial
    // list rather than standing in for one.
    let listed: u32 = shares.iter().map(|p| p.median_us).sum();
    let rest = gpu_median_us.saturating_sub(listed);
    if rest > 0 && !shares.is_empty() {
        shares.push(PassShare {
            name: UNATTRIBUTED.to_string(),
            median_us: rest,
            share: rest as f32 / gpu_median_us as f32,
        });
    }
    shares
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_report::sample::MAX_SYSTEM_TIMINGS;
    use concinnity_core::gfx::profile::MAX_PASS_TIMINGS;

    // A sample with the fields a test cares about and zeroes elsewhere.
    fn sample(run_seconds: f32, segment: Option<u32>, frame_us: u32) -> FrameSample {
        FrameSample {
            run_seconds,
            segment,
            frame_us,
            gpu_frame_us: frame_us / 2,
            gpu_wait_us: 100,
            draw_calls: 50,
            objects: 400,
            vram_bytes: 1 << 20,
            pass_us: [0; MAX_PASS_TIMINGS],
            system_us: [0; MAX_SYSTEM_TIMINGS],
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
    fn the_graphics_systems_span_has_the_blocked_on_gpu_wait_taken_out_of_it() {
        // The wait is wall time inside that system's own step. Left in, it
        // reads as nearly the whole frame on every GPU-bound scene, which says
        // nothing about either side.
        let mut only = sample(0.0, Some(0), 10_000);
        only.gpu_wait_us = 9_000;
        only.system_us[0] = 300;
        only.system_us[1] = 9_400;
        let mut run = run_of(vec![only]);
        run.system_names = vec!["PhysicsSystem".to_string(), "GraphicsSystem".to_string()];
        let report = Report::of(&run, no_warmup()).expect("measured frames");

        // 700us of CPU work in a 10ms frame: what the two systems stepped,
        // once the wait is out of the one that carried it.
        assert_eq!(report.overall.cpu.mean_us, 700);
        let systems = &report.overall.systems;
        assert_eq!(systems[0].name, "GraphicsSystem");
        assert_eq!(systems[0].mean_us, 400);
        // A system that never waits keeps its whole span.
        assert_eq!(systems[1].mean_us, 300);
        assert!((systems[0].share - 400.0 / 700.0).abs() < 1e-4);
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
        busy.system_us[1] = 11_800;
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

    #[test]
    fn an_incomplete_run_says_so() {
        let mut run = run_of(vec![sample(0.0, Some(0), 10_000)]);
        run.completed = false;
        let report = Report::of(&run, no_warmup()).expect("measured frames");
        assert!(!report.completed);
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
