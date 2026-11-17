// Which passes the view's relevance turned off over a stretch of the run, in
// how many of its frames, and how often each went off or came back.

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
    switches: [u32; PASS_COUNT],
    // When each pass first and last switched, in run seconds.
    switched_at: [Option<(f32, f32)>; PASS_COUNT],
    // The last frame's passes turned off.
    last: Option<u64>,
}

impl Default for RelevanceTally {
    fn default() -> Self {
        Self {
            frames: 0,
            off: [0; PASS_COUNT],
            switches: [0; PASS_COUNT],
            switched_at: [None; PASS_COUNT],
            last: None,
        }
    }
}

impl RelevanceTally {
    /// Count a frame `at` run seconds whose relevance turned off the passes in
    /// `off`, one bit per [`PassId`].
    pub(super) fn record(&mut self, off: u64, at: f32) {
        self.frames += 1;
        let switched = self.last.map_or(0, |last| last ^ off);
        self.last = Some(off);
        for (i, pass) in PassId::ALL.iter().enumerate() {
            let bit = 1 << *pass as u32;
            if off & bit != 0 {
                self.off[i] += 1;
            }
            if switched & bit != 0 {
                self.switches[i] += 1;
                let span = &mut self.switched_at[i];
                *span = Some(span.map_or((at, at), |(first, _)| (first, at)));
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
    /// How many times it went off or came back within the stretch.
    pub switches: u32,
    /// When it first and last went off or came back, in run seconds.
    pub switched_at: Option<(f32, f32)>,
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
            .enumerate()
            .filter(|(i, _)| tally.off[*i] > 0)
            .map(|(i, pass)| RelevanceRow {
                pass: pass.name().to_string(),
                frames: tally.off[i],
                share: tally.off[i] as f32 * PERCENT / tally.frames.max(1) as f32,
                switches: tally.switches[i],
                switched_at: tally.switched_at[i],
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
        write!(
            out,
            "  {:<22} {:>6.1}% of {} frame(s), {} switch(es)",
            row.pass, row.share, relevance.frames, row.switches
        )?;
        match row.switched_at {
            Some((first, last)) if row.switches > 1 => {
                writeln!(out, " from {first:.2} s to {last:.2} s")?;
            }
            Some((at, _)) => writeln!(out, " at {at:.2} s")?,
            None => writeln!(out)?,
        }
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
        tally.record(bit(PassId::WaterSim) | bit(PassId::Disocclusion), 1.0);
        tally.record(bit(PassId::WaterSim), 1.5);
        tally.record(0, 2.0);
        tally.record(bit(PassId::WaterSim), 2.5);
        let report = RelevanceReport::of(&tally);
        assert_eq!(report.frames, 4);
        assert_eq!(report.off.len(), 2);
        assert_eq!(report.off[0].pass, PassId::WaterSim.name());
        assert_eq!((report.off[0].frames, report.off[0].share), (3, 75.0));
        assert_eq!((report.off[1].frames, report.off[1].share), (1, 25.0));
        // The ocean came back once and went off again; the mark came back.
        assert_eq!((report.off[0].switches, report.off[1].switches), (2, 1));
        assert_eq!(report.off[0].switched_at, Some((2.0, 2.5)));
        assert_eq!(report.off[1].switched_at, Some((1.5, 1.5)));
        let mut text = String::new();
        write_relevance(&mut text, "orbit", &report).unwrap();
        assert!(text.contains("orbit relevance, passes off"), "{text}");
        assert!(text.contains("water_sim"), "{text}");
        assert!(
            text.contains("75.0% of 4 frame(s), 2 switch(es) from 2.00 s to 2.50 s"),
            "{text}"
        );
        assert!(text.contains("1 switch(es) at 1.50 s"), "{text}");
    }

    #[test]
    fn a_stretch_with_nothing_off_says_so_and_an_empty_one_says_nothing() {
        let mut tally = RelevanceTally::default();
        tally.record(0, 0.0);
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
