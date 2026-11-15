// Which passes the view's relevance turned off over a stretch of the run,
// and in how many of its frames.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::{self, Write};

use crate::render::render_graph::{PASS_COUNT, PassId};
use serde::Serialize;

const PERCENT: f32 = 100.0;

/// How often each pass was turned off for adding nothing to the view, over
/// the frames of one stretch.
#[derive(Debug, Clone, Copy)]
pub(super) struct RelevanceTally {
    frames: u32,
    off: [u32; PASS_COUNT],
}

impl Default for RelevanceTally {
    fn default() -> Self {
        Self {
            frames: 0,
            off: [0; PASS_COUNT],
        }
    }
}

impl RelevanceTally {
    /// Count a frame whose relevance turned off the passes in `off`, one bit
    /// per [`PassId`].
    pub(super) fn record(&mut self, off: u64) {
        self.frames += 1;
        for (count, pass) in self.off.iter_mut().zip(PassId::ALL) {
            if off & (1 << pass as u32) != 0 {
                *count += 1;
            }
        }
    }
}

/// One pass the view's relevance turned off.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(super) struct RelevanceRow {
    /// The pass.
    pub pass: String,
    /// Frames it was off in.
    pub frames: u32,
    /// Its share of the stretch's frames, in percent.
    pub share: f32,
}

/// The passes the view's relevance turned off over one stretch of the run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(super) struct RelevanceReport {
    /// Frames the stretch drew.
    pub frames: u32,
    /// Each pass it turned off in some frame, the most often first.
    pub off: Vec<RelevanceRow>,
}

impl RelevanceReport {
    pub(super) fn of(tally: &RelevanceTally) -> Self {
        let mut off: Vec<RelevanceRow> = PassId::ALL
            .iter()
            .zip(tally.off)
            .filter(|(_, frames)| *frames > 0)
            .map(|(pass, frames)| RelevanceRow {
                pass: pass.name().to_string(),
                frames,
                share: frames as f32 * PERCENT / tally.frames.max(1) as f32,
            })
            .collect();
        off.sort_by_key(|row| core::cmp::Reverse(row.frames));
        Self {
            frames: tally.frames,
            off,
        }
    }
}

/// Write the passes one stretch's relevance turned off, under the stretch's
/// `phase` name. Nothing for a stretch that drew no frame.
pub(super) fn write_relevance(
    out: &mut impl Write,
    phase: &str,
    relevance: &RelevanceReport,
) -> fmt::Result {
    if relevance.frames == 0 {
        return Ok(());
    }
    writeln!(out)?;
    if relevance.off.is_empty() {
        return writeln!(out, "{phase} relevance: every asked-for pass ran");
    }
    writeln!(
        out,
        "{phase} relevance, passes off for adding nothing to the view:"
    )?;
    for row in &relevance.off {
        writeln!(
            out,
            "  {:<22} {:>6.1}% of {} frame(s)",
            row.pass, row.share, relevance.frames
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bit(pass: PassId) -> u64 {
        1 << pass as u32
    }

    #[test]
    fn a_pass_off_in_some_frames_reports_its_share() {
        let mut tally = RelevanceTally::default();
        tally.record(bit(PassId::WaterSim) | bit(PassId::Disocclusion));
        tally.record(bit(PassId::WaterSim));
        tally.record(0);
        tally.record(bit(PassId::WaterSim));
        let report = RelevanceReport::of(&tally);
        assert_eq!(report.frames, 4);
        assert_eq!(report.off.len(), 2);
        assert_eq!(report.off[0].pass, PassId::WaterSim.name());
        assert_eq!((report.off[0].frames, report.off[0].share), (3, 75.0));
        assert_eq!((report.off[1].frames, report.off[1].share), (1, 25.0));
        let mut text = String::new();
        write_relevance(&mut text, "orbit", &report).unwrap();
        assert!(text.contains("orbit relevance, passes off"), "{text}");
        assert!(text.contains("water_sim"), "{text}");
        assert!(text.contains("75.0% of 4 frame(s)"), "{text}");
    }

    #[test]
    fn a_stretch_with_nothing_off_says_so_and_an_empty_one_says_nothing() {
        let mut tally = RelevanceTally::default();
        tally.record(0);
        let mut text = String::new();
        write_relevance(&mut text, "roll", &RelevanceReport::of(&tally)).unwrap();
        assert!(
            text.contains("roll relevance: every asked-for pass ran"),
            "{text}"
        );
        let mut empty = String::new();
        write_relevance(
            &mut empty,
            "roll",
            &RelevanceReport::of(&RelevanceTally::default()),
        )
        .unwrap();
        assert!(empty.is_empty());
    }
}
