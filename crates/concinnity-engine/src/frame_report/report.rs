// Rendering a reduced run as the text a person reads.

use std::fmt::Write;

use crate::frame_report::reduce::{Report, SegmentReport};

const MICROS_PER_MILLI: f32 = 1_000.0;
const BYTES_PER_MIB: f32 = (1 << 20) as f32;

/// Render a report as plain text.
///
/// Frame time leads every row because it is the number a player feels; the
/// budget count beside it is what a mean would hide.
pub fn to_text(report: &Report) -> String {
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
            "  NOTE: the camera track did not finish, so this covers part of the path only",
        );
    }
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "{:<16} {:>7} {:>8} {:>8} {:>8} {:>8} {:>9} {:>8} {:>8}",
        "segment", "frames", "p50 ms", "p95 ms", "p99 ms", "max ms", "over bgt", "cpu ms", "gpu ms",
    );
    write_row(&mut out, &report.overall);
    for segment in &report.segments {
        write_row(&mut out, segment);
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

    // The pass block is medians, like the table above it, because a backend
    // that mistimes one frame does it by orders of magnitude. The system block
    // is means, so that its shares add up to the work the systems did. Each
    // heading carries the total its shares are taken against.
    for segment in &report.segments {
        write_block(
            &mut out,
            &format!(
                "{} GPU passes, median {:.2} ms",
                segment.name,
                segment.gpu.p50_us as f32 / MICROS_PER_MILLI,
            ),
            "of the GPU frame",
            segment
                .passes
                .iter()
                .map(|p| (p.name.as_str(), p.median_us, p.share)),
        );
        write_block(
            &mut out,
            &format!(
                "{} CPU systems, mean {:.2} ms",
                segment.name,
                segment.cpu.mean_us as f32 / MICROS_PER_MILLI,
            ),
            "of system CPU",
            segment
                .systems
                .iter()
                .map(|s| (s.name.as_str(), s.mean_us, s.share)),
        );
    }
    out
}

// A named cost breakdown, or nothing at all when there is none to show. An
// empty CPU block is the report saying the CPU was not what limited the
// stretch, which is worth more than a ranking of noise.
fn write_block<'a>(
    out: &mut String,
    heading: &str,
    of_what: &str,
    rows: impl Iterator<Item = (&'a str, u32, f32)>,
) {
    let mut started = false;
    for (name, mean_us, share) in rows {
        if !started {
            let _ = writeln!(out);
            let _ = writeln!(out, "{heading}:");
            started = true;
        }
        let _ = writeln!(
            out,
            "  {:<20} {:>7.3} ms  {:>5.1}% {of_what}",
            name,
            mean_us as f32 / MICROS_PER_MILLI,
            share * 100.0,
        );
    }
}

// One line of the summary table.
fn write_row(out: &mut String, segment: &SegmentReport) {
    let ms = |us: u32| us as f32 / MICROS_PER_MILLI;
    let _ = writeln!(
        out,
        "{:<16} {:>7} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.1}% {:>8.2} {:>8.2}",
        segment.name,
        segment.frame.count,
        ms(segment.frame.p50_us),
        ms(segment.frame.p95_us),
        ms(segment.frame.p99_us),
        ms(segment.frame.max_us),
        segment.frame.over_budget_share() * 100.0,
        ms(segment.cpu.p50_us),
        ms(segment.gpu.p50_us),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_report::reduce::ReduceOptions;
    use crate::frame_report::sample::MAX_SYSTEM_TIMINGS;
    use crate::frame_report::sample::{FrameRun, FrameSample};
    use concinnity_core::gfx::profile::MAX_PASS_TIMINGS;

    fn sample(run_seconds: f32, segment: Option<u32>, frame_us: u32) -> FrameSample {
        FrameSample {
            run_seconds,
            segment,
            frame_us,
            gpu_frame_us: frame_us / 2,
            gpu_wait_us: 250,
            draw_calls: 64,
            objects: 512,
            vram_bytes: 512 << 20,
            pass_us: [0; MAX_PASS_TIMINGS],
            system_us: [0; MAX_SYSTEM_TIMINGS],
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
            samples: vec![first, sample(1.0, Some(1), 40_000)],
            segments: vec!["approach".to_string(), "rays".to_string()],
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
    fn every_segment_gets_a_row_under_the_overall_one() {
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
            samples: vec![sample(0.0, Some(0), 10_000)],
            segments: vec!["approach".to_string()],
            pass_names: Vec::new(),
            system_names: Vec::new(),
            completed: false,
        };
        let text = to_text(&report_of(run));
        let note = text.find("did not finish").expect("the note");
        assert!(note < text.find("segment").expect("the table"), "{text}");
    }

    #[test]
    fn a_complete_run_carries_no_note() {
        assert!(!rendered().contains("did not finish"));
    }

    #[test]
    fn a_segment_with_no_pass_timings_lists_no_pass_block() {
        // A backend without timestamp support still gets a table, just no
        // per-pass breakdown under it.
        let run = FrameRun {
            samples: vec![sample(0.0, Some(0), 10_000)],
            segments: vec!["approach".to_string()],
            pass_names: Vec::new(),
            system_names: Vec::new(),
            completed: true,
        };
        let text = to_text(&report_of(run));
        assert!(!text.contains("GPU passes:"), "{text}");
        assert!(text.contains("approach"), "{text}");
    }
}
