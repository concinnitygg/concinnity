// What the camera saw over a stretch of the run: each view fact's least,
// mean and greatest, so thresholds on them can be read from a tour.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::{self, Write};

use crate::render::view::{Subject, SubjectKind, VIEW_SUBJECTS, ViewFacts};
use concinnity_core::math::narrow_f32;
use serde::Serialize;

const PERCENT: f32 = 100.0;

/// The least, mean and greatest of one fact over the frames it was known in.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Spread {
    min: f32,
    max: f32,
    sum: f64,
    count: u32,
}

impl Spread {
    fn record(&mut self, value: f32) {
        if !value.is_finite() {
            return;
        }
        if self.count == 0 {
            self.min = value;
            self.max = value;
        } else {
            self.min = self.min.min(value);
            self.max = self.max.max(value);
        }
        self.sum += f64::from(value);
        self.count += 1;
    }

    fn record_some(&mut self, value: Option<f32>) {
        if let Some(v) = value {
            self.record(v);
        }
    }

    fn row(&self, name: String) -> Option<ViewRow> {
        (self.count > 0).then(|| ViewRow {
            name,
            frames: self.count,
            min: self.min,
            mean: narrow_f32(self.sum / f64::from(self.count)),
            max: self.max,
        })
    }
}

// One subject slot's facts over the frames it was in the world.
#[derive(Debug, Clone, Copy, Default)]
struct SubjectTally {
    kind: Option<SubjectKind>,
    distance: Spread,
    extent_px: Spread,
    coverage: Spread,
    nearest: Spread,
}

impl SubjectTally {
    fn record(&mut self, subject: &Subject, focal_px: f32, measured: bool) {
        self.kind = Some(subject.kind);
        self.distance.record(subject.distance);
        self.extent_px.record(subject.extent_px(focal_px));
        if measured {
            self.coverage.record(subject.seen.coverage * PERCENT);
            self.nearest.record_some(subject.seen.nearest);
        }
    }
}

/// The view facts of one stretch of the run, accumulated frame by frame.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct ViewTally {
    frames: u32,
    measured: u32,
    ground_height: Spread,
    sea_height: Spread,
    nearest: Spread,
    sky: Spread,
    other: Spread,
    other_nearest: Spread,
    motion: Spread,
    parallax: Spread,
    mover: Spread,
    // 100 for a frame with a mover in view that could not be placed, else 0,
    // so its mean is the share of such frames.
    unplaced: Spread,
    direct_light: Spread,
    subjects: [SubjectTally; VIEW_SUBJECTS],
}

impl ViewTally {
    pub(super) fn record(&mut self, facts: &ViewFacts) {
        self.frames += 1;
        self.ground_height.record_some(facts.ground_height);
        self.sea_height.record_some(facts.sea_height);
        self.motion.record(facts.motion_px);
        self.parallax.record(facts.parallax_px);
        self.mover.record(facts.mover_px);
        self.unplaced.record(if facts.mover_px.is_finite() {
            0.0
        } else {
            PERCENT
        });
        self.direct_light.record(facts.direct_light);
        let measured = facts.seen.is_some();
        if let Some(seen) = facts.seen {
            self.measured += 1;
            self.nearest.record_some(facts.nearest());
            self.sky.record(seen.sky * PERCENT);
            self.other.record(seen.other.coverage * PERCENT);
            self.other_nearest.record_some(seen.other.nearest);
        }
        for (tally, subject) in self.subjects.iter_mut().zip(&facts.subjects) {
            if let Some(subject) = subject {
                tally.record(subject, facts.focal_px, measured);
            }
        }
    }
}

/// One fact over a stretch of the run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(super) struct ViewRow {
    /// What it measures, with its unit.
    pub name: String,
    /// Frames it was known in.
    pub frames: u32,
    /// Its least value.
    pub min: f32,
    /// Its mean.
    pub mean: f32,
    /// Its greatest value.
    pub max: f32,
}

/// What the camera saw over one stretch of the run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(super) struct ViewReport {
    /// Frames the stretch drew.
    pub frames: u32,
    /// Frames a measurement of the depth stood for.
    pub measured: u32,
    /// Each fact known in some frame, the camera's first, then what was seen,
    /// then each subject's.
    pub rows: Vec<ViewRow>,
}

impl ViewReport {
    pub(super) fn of(tally: &ViewTally) -> Self {
        let mut rows: Vec<ViewRow> = [
            (&tally.ground_height, "ground height m"),
            (&tally.sea_height, "sea height m"),
            (&tally.motion, "motion px"),
            (&tally.parallax, "parallax px"),
            (&tally.mover, "mover px"),
            (&tally.unplaced, "unplaced mover %"),
            (&tally.direct_light, "direct light"),
            (&tally.nearest, "nearest surface m"),
            (&tally.sky, "sky %"),
            (&tally.other, "other %"),
            (&tally.other_nearest, "other nearest m"),
        ]
        .into_iter()
        .filter_map(|(spread, name)| spread.row(name.to_string()))
        .collect();
        let mut waters = 0;
        for subject in &tally.subjects {
            let Some(kind) = subject.kind else {
                continue;
            };
            let name = match kind {
                SubjectKind::Water => {
                    waters += 1;
                    if waters == 1 {
                        kind.name().to_string()
                    } else {
                        format!("{} {waters}", kind.name())
                    }
                }
                _ => kind.name().to_string(),
            };
            rows.extend(
                [
                    (&subject.distance, "distance m"),
                    (&subject.extent_px, "extent px"),
                    (&subject.coverage, "%"),
                    (&subject.nearest, "nearest m"),
                ]
                .into_iter()
                .filter_map(|(spread, fact)| spread.row(format!("{name} {fact}"))),
            );
        }
        Self {
            frames: tally.frames,
            measured: tally.measured,
            rows,
        }
    }
}

/// Write one stretch's view facts under the stretch's `phase` name. Nothing
/// for a stretch that drew no frame.
pub(super) fn write_view(out: &mut impl Write, phase: &str, view: &ViewReport) -> fmt::Result {
    if view.frames == 0 {
        return Ok(());
    }
    writeln!(out)?;
    writeln!(
        out,
        "{phase} view, {} of {} frame(s) measured",
        view.measured, view.frames
    )?;
    writeln!(
        out,
        "  {:<24} {:>10} {:>10} {:>10}",
        "fact", "min", "mean", "max"
    )?;
    for row in &view.rows {
        writeln!(
            out,
            "  {:<24} {:>10} {:>10} {:>10}",
            row.name,
            number(row.min),
            number(row.mean),
            number(row.max),
        )?;
    }
    Ok(())
}

// A value with the decimals its size calls for, so meters, kilometers and
// shares read alike.
fn number(value: f32) -> String {
    match value.abs() {
        v if v >= 1_000.0 => format!("{value:.0}"),
        v if v >= 10.0 => format!("{value:.1}"),
        _ => format!("{value:.2}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::view::{Seen, Sighting};

    fn subject(kind: SubjectKind, coverage: f32, nearest: Option<f32>) -> Subject {
        Subject {
            kind,
            size: 2.0,
            distance: 4.0,
            center_distance: 5.0,
            in_view: true,
            seen: Sighting { coverage, nearest },
        }
    }

    fn facts(sky: f32, measured: bool) -> ViewFacts {
        let mut facts = ViewFacts {
            focal_px: 500.0,
            ground_height: Some(10.0),
            motion_px: 2.0,
            seen: measured.then_some(Seen {
                age: 2,
                drift_px: 4.0,
                sky,
                other: Sighting::default(),
            }),
            ..ViewFacts::default()
        };
        facts.subjects[0] = Some(subject(SubjectKind::Planet, 1.0 - sky, Some(12.0)));
        facts.subjects[1] = Some(subject(SubjectKind::Water, 0.0, None));
        facts.subjects[2] = Some(subject(SubjectKind::Water, 0.0, None));
        facts
    }

    fn row<'a>(report: &'a ViewReport, name: &str) -> Option<&'a ViewRow> {
        report.rows.iter().find(|r| r.name == name)
    }

    #[test]
    fn each_fact_reports_its_least_mean_and_greatest_over_the_frames_it_was_known() {
        let mut tally = ViewTally::default();
        tally.record(&facts(0.25, true));
        tally.record(&facts(0.75, true));
        tally.record(&facts(0.5, false));
        let report = ViewReport::of(&tally);
        assert_eq!((report.frames, report.measured), (3, 2));
        let sky = row(&report, "sky %").unwrap();
        assert_eq!(
            (sky.frames, sky.min, sky.mean, sky.max),
            (2, 25.0, 50.0, 75.0)
        );
        let height = row(&report, "ground height m").unwrap();
        assert_eq!((height.frames, height.mean), (3, 10.0));
        assert_eq!(row(&report, "planet nearest m").unwrap().max, 12.0);
        // A fact never known has no row.
        assert!(row(&report, "sea height m").is_none());
        assert!(row(&report, "water nearest m").is_none());
        // A second water surface is numbered.
        assert!(row(&report, "water 2 distance m").is_some());
    }

    // A mover that cannot be placed reads as infinite: it counts toward the
    // share of such frames, not the mover's spread.
    #[test]
    fn unplaced_movers_count_as_a_share_of_frames() {
        let mut tally = ViewTally::default();
        let mut moved = facts(0.5, true);
        moved.mover_px = 3.0;
        tally.record(&moved);
        moved.mover_px = f32::INFINITY;
        tally.record(&moved);
        let report = ViewReport::of(&tally);
        let mover = row(&report, "mover px").unwrap();
        assert_eq!((mover.frames, mover.max), (1, 3.0));
        assert_eq!(row(&report, "unplaced mover %").unwrap().mean, 50.0);
    }

    #[test]
    fn the_view_block_lists_each_row_under_the_phase() {
        let mut tally = ViewTally::default();
        tally.record(&facts(0.5, true));
        let mut out = String::new();
        write_view(&mut out, "orbit", &ViewReport::of(&tally)).unwrap();
        assert!(
            out.contains("orbit view, 1 of 1 frame(s) measured"),
            "{out}"
        );
        assert!(
            out.lines().any(|l| l.trim_start().starts_with("sky %")),
            "{out}"
        );
        let mut none = String::new();
        write_view(&mut none, "x", &ViewReport::of(&ViewTally::default())).unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn numbers_take_the_decimals_their_size_calls_for() {
        assert_eq!(number(12_345.6), "12346");
        assert_eq!(number(12.34), "12.3");
        assert_eq!(number(0.126), "0.13");
    }
}
