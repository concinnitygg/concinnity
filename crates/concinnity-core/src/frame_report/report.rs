// Rendering a reduced run as the text a person reads.

use alloc::format;
use alloc::string::String;
use core::fmt::Write;

use crate::frame_report::reduce::{FanOutCost, FrameCost, PhaseReport, Report, SlowFrame};

const MICROS_PER_MILLI: f32 = 1_000.0;
const BYTES_PER_MIB: f32 = (1 << 20) as f32;

/// Render a report as plain text.
///
/// Frame time leads every row because it is the number a player feels; the
/// budget count beside it is what a mean would hide.
pub(super) fn to_text(report: &Report) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "frame report: {} measured frame(s), {} warm-up frame(s) dropped, budget {:.2} ms",
        report.overall.frame.count,
        report.warmup_dropped,
        report.budget_us as f32 / MICROS_PER_MILLI,
    );
    if !report.completed {
        let _ = writeln!(
            out,
            "  NOTE: the run ended before its last phase did, so this covers part of it only",
        );
    }
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "{:<16} {:>7} {:>8} {:>8} {:>8} {:>8} {:>9} {:>8} {:>8}",
        "phase", "frames", "p50 ms", "p95 ms", "p99 ms", "max ms", "over bgt", "cpu ms", "gpu ms",
    );
    write_row(&mut out, &report.overall);
    for phase in &report.phases {
        write_row(&mut out, phase);
    }

    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "peak GPU memory {:.0} MiB, mean blocked-on-GPU {:.2} ms, mean {} draw call(s) over {} object(s)",
        report.overall.vram_peak_bytes as f32 / BYTES_PER_MIB,
        report.overall.gpu_wait_mean_us as f32 / MICROS_PER_MILLI,
        report.overall.draw_calls_mean,
        report.overall.objects_mean,
    );
    write_pacing(&mut out, report);
    write_input_latency(&mut out, report);
    write_slowest(&mut out, &report.slowest);

    // The pass block is medians, like the table above it, because a backend
    // that mistimes one frame does it by orders of magnitude. The system block
    // is means, so that its shares add up to the work the systems did. Each
    // heading carries the total its shares are taken against.
    for phase in &report.phases {
        write_block(
            &mut out,
            &format!(
                "{} GPU passes, median {:.2} ms",
                phase.name,
                phase.gpu.p50_us as f32 / MICROS_PER_MILLI,
            ),
            "of the GPU frame",
            phase.passes.iter().map(|p| {
                // A pass timed inside the one above it is indented under it,
                // and one beside the frame on the async queue is marked.
                let name = match (&p.within, p.async_queue) {
                    (Some(_), _) => format!("  {}", p.name),
                    (None, true) => format!("{} (async)", p.name),
                    (None, false) => p.name.clone(),
                };
                (name, p.median_us, p.share)
            }),
        );
        write_block(
            &mut out,
            &format!(
                "{} CPU work, mean {:.2} ms",
                phase.name,
                phase.cpu.mean_us as f32 / MICROS_PER_MILLI,
            ),
            "of the CPU work",
            phase
                .systems
                .iter()
                .map(|s| (s.name.clone(), s.mean_us, s.share)),
        );
        write_fan_outs(&mut out, &phase.name, &phase.fan_outs);
    }
    out
}

// A phase's fan-outs: how long each held the thread waiting on it, against
// the work its jobs contained. Nothing when none ran.
fn write_fan_outs(out: &mut String, phase: &str, fan_outs: &[FanOutCost]) {
    if fan_outs.is_empty() {
        return;
    }
    let ms = |us: u32| us as f32 / MICROS_PER_MILLI;
    let _ = writeln!(out);
    let _ = writeln!(out, "{phase} fan-outs, mean ms waited against work:");
    let _ = writeln!(
        out,
        "  {:<20} {:>8} {:>9} {:>8} {:>8} {:>8}",
        "", "wall", "first job", "job sum", "longest", "tail",
    );
    for f in fan_outs {
        let _ = writeln!(
            out,
            "  {:<20} {:>8.3} {:>9.3} {:>8.3} {:>8.3} {:>8.3}",
            f.name,
            ms(f.wall_us),
            ms(f.first_job_us),
            ms(f.job_sum_us),
            ms(f.longest_job_us),
            ms(f.tail_us),
        );
    }
}

// A named cost breakdown, or nothing at all when there is none to show. An
// empty CPU block is the report saying the CPU was not what limited the
// stretch, which is worth more than a ranking of noise.
fn write_block(
    out: &mut String,
    heading: &str,
    of_what: &str,
    rows: impl Iterator<Item = (String, u32, f32)>,
) {
    let mut started = false;
    for (name, mean_us, share) in rows {
        if !started {
            let _ = writeln!(out);
            let _ = writeln!(out, "{heading}:");
            started = true;
        }
        // A negative share is a row taken away from the others.
        let sign = if share < 0.0 { -1.0 } else { 1.0 };
        let _ = writeln!(
            out,
            "  {:<20} {:>7.3} ms  {:>5.1}% {of_what}",
            name,
            sign * mean_us as f32 / MICROS_PER_MILLI,
            share * 100.0,
        );
    }
}

// How evenly each stretch's motion matched its frame times, and how often the
// simulated frame moved under the camera.
fn write_pacing(out: &mut String, report: &Report) {
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "camera pacing, run time shown per finished frame over GPU frame interval (1.00x is even):"
    );
    let _ = writeln!(
        out,
        "  {:<16} {:>7} {:>7} {:>7} {:>7} {:>8}",
        "phase", "p5", "p50", "p95", "still", "rebases",
    );
    for phase in core::iter::once(&report.overall).chain(&report.phases) {
        let p = &phase.pacing;
        let _ = writeln!(
            out,
            "  {:<16} {:>6.2}x {:>6.2}x {:>6.2}x {:>7} {:>8}",
            phase.name, p.p5, p.p50, p.p95, p.still_frames, phase.rebases,
        );
    }
}

// How stale each stretch's input was by the time its frames' GPU work ended.
// One line when the backend measured none.
fn write_input_latency(out: &mut String, report: &Report) {
    let _ = writeln!(out);
    if report.overall.input_latency.frames == 0 {
        let _ = writeln!(
            out,
            "input latency: not measured (the backend cannot place its GPU work on the host clock)"
        );
        return;
    }
    let _ = writeln!(out, "input latency, input sample to GPU work done:");
    let _ = writeln!(
        out,
        "  {:<16} {:>9} {:>9} {:>7}",
        "phase", "p50 ms", "p95 ms", "frames",
    );
    let ms = |us: u32| us as f32 / MICROS_PER_MILLI;
    for phase in core::iter::once(&report.overall).chain(&report.phases) {
        let l = &phase.input_latency;
        let _ = writeln!(
            out,
            "  {:<16} {:>9.2} {:>9.2} {:>7}",
            phase.name,
            ms(l.p50_us),
            ms(l.p95_us),
            l.frames,
        );
    }
}

// The slowest frames, each with the costs that say where its time went.
fn write_slowest(out: &mut String, frames: &[SlowFrame]) {
    if frames.is_empty() {
        return;
    }
    let ms = |us: u32| us as f32 / MICROS_PER_MILLI;
    let _ = writeln!(out);
    let _ = writeln!(out, "slowest frames:");
    for frame in frames {
        let _ = writeln!(
            out,
            "  {:.2} ms at {:.2} s in {}: cpu {:.2} ms, blocked on GPU {:.2} ms, gpu {:.2} ms (gaps {:.2}, overlap {:.2}), run step {:.2} ms{}",
            ms(frame.frame_us),
            frame.run_seconds,
            frame.phase,
            ms(frame.cpu_us),
            ms(frame.gpu_wait_us),
            ms(frame.gpu_frame_us),
            ms(frame.gpu_gap_us),
            ms(frame.gpu_overlap_us),
            ms(frame.run_step_us),
            if frame.rebased { ", frame moved" } else { "" },
        );
        write_costs(out, "cpu", &frame.cpu);
        write_costs(out, "gpu", &frame.passes);
        write_costs(out, "record", &frame.recording);
        write_costs(out, "render", &frame.render);
        for f in &frame.fan_outs {
            let _ = writeln!(
                out,
                "    fan-out {}: wall {:.2}, first job {:.2}, job sum {:.2}, longest {:.2}, tail {:.2}",
                f.name,
                ms(f.wall_us),
                ms(f.first_job_us),
                ms(f.job_sum_us),
                ms(f.longest_job_us),
                ms(f.tail_us),
            );
        }
    }
}

// One `label: name ms, name ms` line, or nothing when there are no costs.
fn write_costs(out: &mut String, label: &str, costs: &[FrameCost]) {
    if costs.is_empty() {
        return;
    }
    let _ = write!(out, "    {label}:");
    for (i, cost) in costs.iter().enumerate() {
        let separator = if i == 0 { "" } else { "," };
        let _ = write!(
            out,
            "{separator} {} {:.2}",
            cost.name,
            cost.us as f32 / MICROS_PER_MILLI
        );
    }
    let _ = writeln!(out);
}

// One line of the summary table.
fn write_row(out: &mut String, phase: &PhaseReport) {
    let ms = |us: u32| us as f32 / MICROS_PER_MILLI;
    let _ = writeln!(
        out,
        "{:<16} {:>7} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.1}% {:>8.2} {:>8.2}",
        phase.name,
        phase.frame.count,
        ms(phase.frame.p50_us),
        ms(phase.frame.p95_us),
        ms(phase.frame.p99_us),
        ms(phase.frame.max_us),
        phase.frame.over_budget_share() * 100.0,
        ms(phase.cpu.p50_us),
        ms(phase.gpu.p50_us),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_report::reduce::ReduceOptions;
    use crate::frame_report::sample::{FrameRun, FrameSample, MAX_SYSTEM_TIMINGS};
    use crate::profile::{FanOutTiming, MAX_PASS_TIMINGS};
    use crate::render::render_graph::PassId;
    use alloc::string::ToString;
    use alloc::vec;
    use alloc::vec::Vec;

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
            gpu_wait_us: 250,
            render_cpu_us: 0,
            replay_us: 0,
            submit_present_us: 0,
            recording_fan_out: FanOutTiming::default(),
            packing_fan_out: FanOutTiming::default(),
            pass_record_us: [0; MAX_PASS_TIMINGS],
            draw_calls: 64,
            objects: 512,
            vram_bytes: 512 << 20,
            pass_us: [0; MAX_PASS_TIMINGS],
            async_pass_mask: 0,
            system_us: [0; MAX_SYSTEM_TIMINGS],
            system_fan_out: [FanOutTiming::default(); MAX_SYSTEM_TIMINGS],
        }
    }

    fn report_of(run: FrameRun) -> Report {
        Report::of(
            &run,
            ReduceOptions {
                warmup_seconds: 0.0,
                ..Default::default()
            },
        )
        .expect("measured frames")
    }

    fn rendered() -> String {
        let mut first = sample(0.0, Some(0), 10_000);
        first.pass_us[0] = 3_000;
        let run = FrameRun {
            samples: vec![first, sample(1.0, Some(1), 40_000)].into(),
            phases: vec!["approach".to_string(), "rays".to_string()],
            pass_names: vec!["main".to_string()],
            system_names: Vec::new(),
            completed: true,
        };
        to_text(&report_of(run))
    }

    #[test]
    fn the_header_states_what_was_measured_and_what_was_thrown_away() {
        let text = rendered();
        assert!(text.contains("2 measured frame(s)"), "{text}");
        assert!(text.contains("0 warm-up frame(s) dropped"), "{text}");
        assert!(text.contains("budget 16.67 ms"), "{text}");
    }

    #[test]
    fn every_phase_gets_a_row_under_the_overall_one() {
        let text = rendered();
        assert!(text.contains("\nall "), "{text}");
        assert!(text.contains("approach"), "{text}");
        assert!(text.contains("rays"), "{text}");
    }

    #[test]
    fn microseconds_are_rendered_as_milliseconds() {
        // 40_000us is the slowest frame, so the max column reads 40.00.
        assert!(rendered().contains("40.00"), "{}", rendered());
    }

    #[test]
    fn a_pass_is_listed_with_its_share_of_the_gpu_frame() {
        // 3_000us of a 5_000us GPU frame is 60 percent.
        let text = rendered();
        assert!(text.contains("main"), "{text}");
        assert!(text.contains("60.0%"), "{text}");
        // The share is of the mean, which the table above reports the median
        // of, so the heading names the total it was taken against.
        assert!(text.contains("GPU passes, median 5.00 ms"), "{text}");
    }

    #[test]
    fn a_pass_timed_inside_another_is_indented_under_it() {
        let mut first = sample(0.0, Some(0), 10_000);
        first.pass_us[0] = 3_000;
        first.pass_us[1] = 500;
        let run = FrameRun {
            samples: vec![first].into(),
            phases: vec!["approach".to_string()],
            pass_names: vec!["main".to_string(), "sky".to_string()],
            system_names: Vec::new(),
            completed: true,
        };
        let text = to_text(&report_of(run));
        let main = text.find("\n  main ").expect("main row");
        let sky = text.find("\n    sky ").expect("indented sky row");
        assert!(main < sky, "{text}");
    }

    #[test]
    fn an_async_pass_is_marked_and_the_gap_is_named() {
        let mut first = sample(0.0, Some(0), 10_000);
        first.pass_us[0] = 3_000;
        first.pass_us[1] = 1_000;
        first.async_pass_mask = 0b10;
        let run = FrameRun {
            samples: vec![first].into(),
            phases: vec!["approach".to_string()],
            pass_names: vec!["main".to_string(), "light_cull".to_string()],
            system_names: Vec::new(),
            completed: true,
        };
        let text = to_text(&report_of(run));
        assert!(text.contains("\n  gaps "), "{text}");
        assert!(text.contains("\n  light_cull (async) "), "{text}");
    }

    #[test]
    fn the_overlap_with_the_previous_frame_is_taken_away() {
        let mut first = sample(0.0, Some(0), 10_000);
        first.pass_us[0] = 6_000;
        first.gpu_overlap_us = 1_500;
        let run = FrameRun {
            samples: vec![first].into(),
            phases: vec!["approach".to_string()],
            pass_names: vec!["main".to_string()],
            system_names: Vec::new(),
            completed: true,
        };
        let text = to_text(&report_of(run));
        assert!(
            text.contains("  overlap with previous frame  -1.500 ms  -30.0% of the GPU frame"),
            "{text}"
        );
    }

    #[test]
    fn unmeasured_input_latency_says_so() {
        let text = rendered();
        assert!(text.contains("input latency: not measured"), "{text}");
    }

    #[test]
    fn input_latency_is_listed_per_phase() {
        let frames: Vec<FrameSample> = (0..10)
            .map(|n| {
                let mut f = sample(n as f32, Some(0), 16_000);
                f.input_latency_us = 45_000;
                f
            })
            .collect();
        let run = FrameRun {
            samples: frames.into(),
            phases: vec!["approach".to_string()],
            pass_names: Vec::new(),
            system_names: Vec::new(),
            completed: true,
        };
        let text = to_text(&report_of(run));
        assert!(
            text.contains("input latency, input sample to GPU work done:"),
            "{text}"
        );
        assert!(
            text.contains("\n  all                  45.00     45.00      10\n"),
            "{text}"
        );
    }

    #[test]
    fn pacing_is_listed_per_phase() {
        let text = rendered();
        assert!(text.contains("camera pacing"), "{text}");
        assert!(
            text.contains("\n  approach           1.00x   1.00x   1.00x       0        0\n"),
            "{text}"
        );
    }

    #[test]
    fn the_cpu_column_is_rendered_beside_the_wall_one() {
        // The fixture names no systems, so the CPU column falls back to the
        // wall time less the wait: 40ms less 250us on the slower frame.
        let text = rendered();
        assert!(text.contains("cpu ms"), "{text}");
        assert!(text.contains("39.75"), "{text}");
    }

    #[test]
    fn a_run_that_did_not_finish_is_flagged_before_its_numbers() {
        let run = FrameRun {
            samples: vec![sample(0.0, Some(0), 10_000)].into(),
            phases: vec!["approach".to_string()],
            pass_names: Vec::new(),
            system_names: Vec::new(),
            completed: false,
        };
        let text = to_text(&report_of(run));
        let note = text.find("ended before its last phase").expect("the note");
        assert!(note < text.find("\nphase ").expect("the table"), "{text}");
    }

    #[test]
    fn the_slowest_frames_are_listed_with_their_breakdown() {
        let mut hitch = sample(1.0, Some(1), 300_000);
        hitch.gpu_wait_us = 297_000;
        hitch.render_cpu_us = 700;
        hitch.system_us[0] = 1_200;
        hitch.pass_us[0] = 4_000;
        hitch.pass_record_us[PassId::Main as usize] = 2_500;
        hitch.replay_us = 150;
        hitch.submit_present_us = 300;
        let run = FrameRun {
            samples: vec![sample(0.0, Some(0), 10_000), hitch].into(),
            phases: vec!["approach".to_string(), "rays".to_string()],
            pass_names: vec!["main".to_string()],
            system_names: vec!["PhysicsSystem".to_string()],
            completed: true,
        };
        let text = to_text(&report_of(run));
        let heading = text.find("slowest frames:").expect("the list");
        let hitch_line = text
            .find(
                "300.00 ms at 1.00 s in rays: cpu 1.90 ms, blocked on GPU 297.00 ms, gpu 150.00 ms",
            )
            .expect("the slowest frame, first");
        assert!(heading < hitch_line, "{text}");
        assert!(
            text.contains("    cpu: PhysicsSystem 1.20, render submit 0.70\n"),
            "{text}"
        );
        assert!(text.contains("    gpu: main 4.00\n"), "{text}");
        assert!(text.contains("    record: main 2.50\n"), "{text}");
        assert!(
            text.contains("    render: submit and present 0.30, replay 0.15\n"),
            "{text}"
        );
        assert!(text.contains("10.00 ms at 0.00 s in approach"), "{text}");
    }

    #[test]
    fn the_cpu_block_names_the_work_it_shares_out() {
        let mut only = sample(0.0, Some(0), 10_000);
        only.system_us[0] = 600;
        only.render_cpu_us = 400;
        let run = FrameRun {
            samples: vec![only].into(),
            phases: vec!["approach".to_string()],
            pass_names: Vec::new(),
            system_names: vec!["PhysicsSystem".to_string()],
            completed: true,
        };
        let text = to_text(&report_of(run));
        assert!(text.contains("approach CPU work, mean 1.00 ms:"), "{text}");
        assert!(text.contains("40.0% of the CPU work"), "{text}");
    }

    #[test]
    fn fan_outs_are_listed_per_phase_and_per_slow_frame() {
        let recording = FanOutTiming {
            wall_us: 2_400,
            first_job_us: 1_200,
            job_sum_us: 1_300,
            longest_job_us: 350,
            tail_us: 600,
        };
        let mut only = sample(0.0, Some(0), 10_000);
        only.recording_fan_out = recording;
        only.system_fan_out[0] = FanOutTiming {
            wall_us: 900,
            ..recording
        };
        let run = FrameRun {
            samples: vec![only].into(),
            phases: vec!["approach".to_string()],
            pass_names: Vec::new(),
            system_names: vec!["BehaviorSystem".to_string()],
            completed: true,
        };
        let text = to_text(&report_of(run));
        assert!(
            text.contains("approach fan-outs, mean ms waited against work:"),
            "{text}"
        );
        assert!(
            text.contains("  render recording        2.400     1.200    1.300    0.350    0.600\n"),
            "{text}"
        );
        assert!(text.contains("  BehaviorSystem          0.900 "), "{text}");
        assert!(
            text.contains(
                "    fan-out render recording: wall 2.40, first job 1.20, job sum 1.30, longest 0.35, tail 0.60\n"
            ),
            "{text}"
        );
    }

    #[test]
    fn a_run_with_no_fan_outs_lists_no_fan_out_block() {
        assert!(!rendered().contains("fan-out"), "{}", rendered());
    }

    #[test]
    fn a_complete_run_carries_no_note() {
        assert!(!rendered().contains("did not finish"));
    }

    #[test]
    fn a_phase_with_no_pass_timings_lists_no_pass_block() {
        // A backend without timestamp support still gets a table, just no
        // per-pass breakdown under it.
        let run = FrameRun {
            samples: vec![sample(0.0, Some(0), 10_000)].into(),
            phases: vec!["approach".to_string()],
            pass_names: Vec::new(),
            system_names: Vec::new(),
            completed: true,
        };
        let text = to_text(&report_of(run));
        assert!(!text.contains("GPU passes:"), "{text}");
        assert!(text.contains("approach"), "{text}");
    }
}
