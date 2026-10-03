//! What a build's status display draws: one row per step the build has
//! reached, and the messages waiting to scroll out above them.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use concinnity_cook::{BuildProgress, BuildStage};

// A step of a build as the display names it: the world load the CLI runs,
// then the cook's own stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Step {
    Load,
    Cook(BuildStage),
}

impl Step {
    pub(super) fn label(self) -> &'static str {
        match self {
            Step::Load => "Load",
            Step::Cook(BuildStage::Import) => "Import",
            Step::Cook(BuildStage::Compile) => "Compile",
            Step::Cook(BuildStage::Write) => "Write",
            Step::Cook(BuildStage::Thumbnails) => "Thumbnails",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum RowState {
    Running,
    Done { summary: String },
    Failed { detail: String },
}

#[derive(Debug, Clone)]
pub(super) struct Row {
    pub(super) step: Step,
    pub(super) started: Instant,
    pub(super) elapsed: Duration,
    pub(super) total: u32,
    pub(super) done: u32,
    pub(super) reused: u32,
    // Items begun and not yet finished, oldest first.
    pub(super) in_flight: Vec<(String, Instant)>,
    pub(super) last: Option<String>,
    // The distinct items finished, which for an import are its source files.
    pub(super) sources: HashSet<String>,
    pub(super) state: RowState,
}

impl Row {
    fn new(step: Step, total: u32, now: Instant) -> Self {
        Self {
            step,
            started: now,
            elapsed: Duration::ZERO,
            total,
            done: 0,
            reused: 0,
            in_flight: Vec::new(),
            last: None,
            sources: HashSet::new(),
            state: RowState::Running,
        }
    }

    pub(super) fn is_running(&self) -> bool {
        self.state == RowState::Running
    }

    // The item worth naming while the row runs: the oldest one still in
    // flight, else the last one to finish.
    pub(super) fn current_item(&self, now: Instant) -> Option<(&str, Duration)> {
        match self.in_flight.first() {
            Some((item, since)) => Some((item, now.saturating_duration_since(*since))),
            None => self.last.as_deref().map(|item| (item, Duration::ZERO)),
        }
    }

    // What the row reports once it is done, from its own counts.
    fn summary(&self) -> String {
        let fresh = self.done.saturating_sub(self.reused);
        match self.step {
            Step::Load => String::new(),
            Step::Cook(BuildStage::Import) => format!(
                "{} from {}",
                super::render::counted(self.done, "asset"),
                super::render::counted(self.sources.len() as u32, "file"),
            ),
            Step::Cook(BuildStage::Compile) => {
                let assets = super::render::counted(self.done, "asset");
                match (fresh, self.reused) {
                    (_, 0) => assets,
                    (0, _) => format!("{assets}, all cached"),
                    (fresh, reused) => format!(
                        "{assets} · {} compiled, {} cached",
                        super::render::group(fresh),
                        super::render::group(reused),
                    ),
                }
            }
            Step::Cook(BuildStage::Write) => super::render::counted(self.done, "blob"),
            Step::Cook(BuildStage::Thumbnails) => thumbnail_summary(fresh, self.reused),
        }
    }
}

// Thumbnails as a build reports them: those it rendered and those the cache
// already held.
pub(super) fn thumbnail_summary(rendered: u32, reused: u32) -> String {
    match (rendered, reused) {
        (rendered, 0) => format!("{} rendered", super::render::group(rendered)),
        (0, reused) => format!("{} cached", super::render::group(reused)),
        (rendered, reused) => format!(
            "{} rendered · {} cached",
            super::render::group(rendered),
            super::render::group(reused)
        ),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum NoteLevel {
    Error,
    Warning,
    Info,
}

// A message that scrolls out above the rows, such as a warning the cook logs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct Note {
    pub(super) level: NoteLevel,
    pub(super) text: String,
}

// How a finished build ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Outcome {
    Built { written: u64, data_dir: String },
    Failed,
}

#[derive(Debug)]
pub(super) struct Board {
    pub(super) started: Instant,
    pub(super) rows: Vec<Row>,
    pub(super) warnings: usize,
    pub(super) errors: usize,
    pub(super) outcome: Option<(Outcome, Duration)>,
    notes: Vec<Note>,
    // Every note raised so far, so one a build raises twice (a source file
    // read in two steps) shows once.
    raised: HashSet<Note>,
}

impl Board {
    pub(super) fn new(now: Instant) -> Self {
        Self {
            started: now,
            rows: Vec::new(),
            warnings: 0,
            errors: 0,
            outcome: None,
            notes: Vec::new(),
            raised: HashSet::new(),
        }
    }

    // Start `step`, closing whichever step was running.
    pub(super) fn begin(&mut self, step: Step, total: u32, now: Instant) {
        self.close(None, now);
        self.rows.push(Row::new(step, total, now));
    }

    pub(super) fn apply(&mut self, progress: BuildProgress<'_>, now: Instant) {
        match progress {
            BuildProgress::Started { stage, total } => self.begin(Step::Cook(stage), total, now),
            BuildProgress::ItemStarted { stage, item } => {
                if let Some(row) = self.running(stage) {
                    row.in_flight.push((item.to_string(), now));
                }
            }
            BuildProgress::ItemFinished {
                stage,
                item,
                done,
                total,
                reused,
            } => {
                if let Some(row) = self.running(stage) {
                    if let Some(at) = row.in_flight.iter().position(|(i, _)| i == item) {
                        row.in_flight.remove(at);
                    }
                    // Parallel workers report out of order; the count only grows.
                    row.done = row.done.max(done);
                    row.total = total;
                    row.reused += u32::from(reused);
                    if row.step == Step::Cook(BuildStage::Import) {
                        row.sources.insert(item.to_string());
                    }
                    row.last = Some(item.to_string());
                }
            }
        }
    }

    // Close the running step as done, with `summary` in place of the one its
    // counts give.
    pub(super) fn close(&mut self, summary: Option<String>, now: Instant) {
        if let Some(row) = self.rows.last_mut().filter(|r| r.is_running()) {
            row.elapsed = now.saturating_duration_since(row.started);
            row.in_flight.clear();
            let summary = summary.unwrap_or_else(|| row.summary());
            row.state = RowState::Done { summary };
        }
    }

    // End the build: the running step closes as done when it built and as
    // failed when it did not, `summary` standing in for what its counts give.
    pub(super) fn end(&mut self, outcome: Outcome, summary: Option<String>, now: Instant) {
        if outcome == Outcome::Failed {
            if let Some(row) = self.rows.last_mut().filter(|r| r.is_running()) {
                row.elapsed = now.saturating_duration_since(row.started);
                let detail = summary.unwrap_or_else(|| match row.total {
                    0 => "failed".to_string(),
                    total => format!(
                        "failed at {} of {}",
                        super::render::group(row.done),
                        super::render::group(total)
                    ),
                });
                row.state = RowState::Failed { detail };
            }
        } else {
            self.close(summary, now);
        }
        self.outcome = Some((outcome, now.saturating_duration_since(self.started)));
    }

    pub(super) fn note(&mut self, level: NoteLevel, text: String) {
        let note = Note { level, text };
        if !self.raised.insert(note.clone()) {
            return;
        }
        match level {
            NoteLevel::Error => self.errors += 1,
            NoteLevel::Warning => self.warnings += 1,
            NoteLevel::Info => {}
        }
        self.notes.push(note);
    }

    // The notes not yet shown, oldest first.
    pub(super) fn take_notes(&mut self) -> Vec<Note> {
        std::mem::take(&mut self.notes)
    }

    fn running(&mut self, stage: BuildStage) -> Option<&mut Row> {
        self.rows
            .last_mut()
            .filter(|r| r.is_running() && r.step == Step::Cook(stage))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finished<'a>(
        stage: BuildStage,
        item: &'a str,
        done: u32,
        reused: bool,
    ) -> BuildProgress<'a> {
        BuildProgress::ItemFinished {
            stage,
            item,
            done,
            total: 3,
            reused,
        }
    }

    fn summary(row: &Row) -> &str {
        match &row.state {
            RowState::Done { summary } => summary,
            other => panic!("row not done: {other:?}"),
        }
    }

    #[test]
    fn a_new_stage_closes_the_running_one_with_its_elapsed_time() {
        let t0 = Instant::now();
        let mut board = Board::new(t0);
        board.begin(Step::Load, 0, t0);
        let t1 = t0 + Duration::from_millis(250);
        board.apply(
            BuildProgress::Started {
                stage: BuildStage::Compile,
                total: 3,
            },
            t1,
        );
        assert_eq!(board.rows.len(), 2);
        assert_eq!(board.rows[0].elapsed, Duration::from_millis(250));
        assert!(!board.rows[0].is_running());
        assert!(board.rows[1].is_running());
    }

    #[test]
    fn finished_items_count_reuse_and_never_run_backwards() {
        let t0 = Instant::now();
        let mut board = Board::new(t0);
        let compile = BuildStage::Compile;
        board.begin(Step::Cook(compile), 3, t0);
        board.apply(finished(compile, "a", 2, true), t0);
        board.apply(finished(compile, "b", 1, false), t0);
        board.apply(finished(compile, "c", 3, true), t0);
        let row = &board.rows[0];
        assert_eq!((row.done, row.reused), (3, 2));
        assert_eq!(row.last.as_deref(), Some("c"));
        board.close(None, t0);
        assert_eq!(summary(&board.rows[0]), "3 assets · 1 compiled, 2 cached");
    }

    #[test]
    fn the_oldest_item_in_flight_is_the_one_named() {
        let t0 = Instant::now();
        let mut board = Board::new(t0);
        let import = BuildStage::Import;
        board.begin(Step::Cook(import), 2, t0);
        let started = |item| BuildProgress::ItemStarted {
            stage: import,
            item,
        };
        board.apply(started("big.fbx"), t0);
        board.apply(started("small.glb"), t0 + Duration::from_secs(1));
        board.apply(
            finished(import, "small.glb", 1, false),
            t0 + Duration::from_secs(2),
        );
        let now = t0 + Duration::from_secs(5);
        assert_eq!(
            board.rows[0].current_item(now),
            Some(("big.fbx", Duration::from_secs(5)))
        );
    }

    #[test]
    fn an_import_summarizes_its_assets_by_source_file() {
        let t0 = Instant::now();
        let mut board = Board::new(t0);
        let import = BuildStage::Import;
        board.begin(Step::Cook(import), 3, t0);
        board.apply(finished(import, "city.fbx", 1, false), t0);
        board.apply(finished(import, "city.fbx", 2, false), t0);
        board.apply(finished(import, "hero.glb", 3, false), t0);
        board.close(None, t0);
        assert_eq!(summary(&board.rows[0]), "3 assets from 2 files");
    }

    #[test]
    fn progress_for_a_stage_that_is_not_running_is_ignored() {
        let t0 = Instant::now();
        let mut board = Board::new(t0);
        board.begin(Step::Load, 0, t0);
        board.apply(finished(BuildStage::Compile, "stray", 1, false), t0);
        assert_eq!(board.rows[0].done, 0);
    }

    #[test]
    fn a_failed_build_marks_the_running_step_failed() {
        let t0 = Instant::now();
        let mut board = Board::new(t0);
        board.begin(Step::Load, 0, t0);
        board.begin(Step::Cook(BuildStage::Compile), 3, t0);
        board.end(Outcome::Failed, None, t0 + Duration::from_secs(1));
        assert!(matches!(board.rows[0].state, RowState::Done { .. }));
        assert_eq!(
            board.rows[1].state,
            RowState::Failed {
                detail: "failed at 0 of 3".into()
            }
        );
        assert_eq!(
            board.outcome,
            Some((Outcome::Failed, Duration::from_secs(1)))
        );
    }

    #[test]
    fn warnings_and_errors_are_counted_and_notes_drain_once() {
        let mut board = Board::new(Instant::now());
        board.note(NoteLevel::Warning, "w".into());
        board.note(NoteLevel::Info, "i".into());
        board.note(NoteLevel::Error, "e".into());
        assert_eq!((board.warnings, board.errors), (1, 1));
        assert_eq!(board.take_notes().len(), 3);
        assert!(board.take_notes().is_empty());
    }

    #[test]
    fn a_repeated_note_shows_and_counts_once() {
        let mut board = Board::new(Instant::now());
        board.note(NoteLevel::Warning, "odd node".into());
        board.take_notes();
        board.note(NoteLevel::Warning, "odd node".into());
        board.note(NoteLevel::Error, "odd node".into());
        assert_eq!((board.warnings, board.errors), (1, 1));
        assert_eq!(board.take_notes().len(), 1);
    }

    #[test]
    fn compile_and_thumbnail_summaries_say_what_the_cache_held() {
        let row = |done, reused| Row {
            done,
            reused,
            ..Row::new(Step::Cook(BuildStage::Compile), done, Instant::now())
        };
        assert_eq!(row(4, 0).summary(), "4 assets");
        assert_eq!(row(4, 4).summary(), "4 assets, all cached");
        assert_eq!(row(1, 0).summary(), "1 asset");
        assert_eq!(thumbnail_summary(2, 0), "2 rendered");
        assert_eq!(thumbnail_summary(0, 3), "3 cached");
        assert_eq!(thumbnail_summary(1_500, 25), "1,500 rendered · 25 cached");
    }
}
