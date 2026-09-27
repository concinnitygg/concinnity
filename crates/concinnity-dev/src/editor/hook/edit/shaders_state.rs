//! EditorHook: the Shaders and Shader source panels' session state, beside
//! their actions in `shaders.rs` and `shader_source.rs`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use concinnity_core::render::shader_programs::vocabulary::{self, Entry};

use crate::debug::hot_reload::{Latest, ReloadOutcome, ReloadReports, ReloadSubject, ReportBoard};
use crate::editor::panels::sdf_field_list::FieldDecl;
use crate::editor::panels::shader_diagnostics::{self, Status, Tone};
use crate::editor::panels::shader_edit::file_name;
use crate::editor::panels::shader_list::{Row, RowKind, ShaderDecl};
use crate::editor::panels::shader_reference::Reference;
use crate::editor::panels::shader_source::{self, DiskChange, SourceKey};
use crate::editor::text_area::TextArea;
use crate::editor::text_area::highlight::{HLSL, SDF_HLSL};
use crate::editor::text_area::markers::GutterMarker;

// The list's shown state and scroll, the board the hot-reload driver publishes
// each Shader's and volume's latest outcome to, and the file open in the
// source panel.
#[derive(Debug)]
pub(in crate::editor::hook) struct ShadersState {
    pub(in crate::editor::hook) open: bool,
    pub(in crate::editor::hook) scroll: usize,
    // The row whose "..." menu is open, by what it stands for, so a rebuild of
    // the rows keeps it on the same row.
    pub(in crate::editor::hook) menu: Option<RowKind>,
    pub(in crate::editor::hook) reports: ReloadReports,
    pub(in crate::editor::hook) source: Option<SourceState>,
    // Each declared path's on-disk path under `paths_dir`, from `resolve` for
    // a Shader's files and `resolve_field` for a distance field's, which the
    // build looks for differently. Kept, because resolving a bare file name
    // walks the assets tree.
    pub(in crate::editor::hook) paths: HashMap<String, String>,
    pub(in crate::editor::hook) field_paths: HashMap<String, String>,
    pub(in crate::editor::hook) paths_dir: Option<PathBuf>,
    pub(in crate::editor::hook) resolve: fn(&str, Option<&Path>) -> String,
    pub(in crate::editor::hook) resolve_field: fn(&str, Option<&Path>) -> String,
    // The list's rows, rebuilt only when what they show changes.
    pub(in crate::editor::hook) rows: Vec<Row>,
    pub(in crate::editor::hook) rows_key: Option<RowsKey>,
    // The source panel's reference column: shown state, folds and scroll.
    pub(in crate::editor::hook) reference: Reference,
}

// What the list's rows are built from: the declared Shaders and fields, the
// board as of its latest change, and the file the source panel shows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::editor::hook) struct RowsKey {
    pub(in crate::editor::hook) shaders: Vec<ShaderDecl>,
    pub(in crate::editor::hook) fields: Vec<FieldDecl>,
    pub(in crate::editor::hook) seq: u64,
    pub(in crate::editor::hook) open: Option<SourceKey>,
}

impl Default for ShadersState {
    fn default() -> Self {
        Self {
            open: false,
            scroll: 0,
            menu: None,
            reports: ReloadReports::default(),
            source: None,
            paths: HashMap::new(),
            field_paths: HashMap::new(),
            paths_dir: None,
            resolve: shader_source::resolve_path,
            resolve_field: shader_source::resolve_field_path,
            rows: Vec::new(),
            rows_key: None,
            reference: Reference::default(),
        }
    }
}

impl ShadersState {
    // Drop what was read out of the world being left: the open file and the
    // resolved paths. Shown state is not the world's.
    pub(in crate::editor::hook) fn reset_for_world(&mut self) {
        self.scroll = 0;
        self.menu = None;
        self.source = None;
        self.paths.clear();
        self.field_paths.clear();
        self.rows.clear();
        self.rows_key = None;
    }

    // `declared`'s on-disk path, resolved against `dir` once and kept. A
    // different `dir` (another project) drops what was kept.
    pub(in crate::editor::hook) fn resolved(
        &mut self,
        declared: &str,
        dir: Option<&Path>,
    ) -> String {
        self.follow_dir(dir);
        let resolve = self.resolve;
        self.paths
            .entry(declared.to_string())
            .or_insert_with(|| resolve(declared, dir))
            .clone()
    }

    // A distance field's on-disk path, kept as `resolved` keeps a Shader
    // file's.
    pub(in crate::editor::hook) fn resolved_field(
        &mut self,
        declared: &str,
        dir: Option<&Path>,
    ) -> String {
        self.follow_dir(dir);
        let resolve = self.resolve_field;
        self.field_paths
            .entry(declared.to_string())
            .or_insert_with(|| resolve(declared, dir))
            .clone()
    }

    fn follow_dir(&mut self, dir: Option<&Path>) {
        if self.paths_dir.as_deref() != dir {
            self.paths.clear();
            self.field_paths.clear();
            self.paths_dir = dir.map(Path::to_path_buf);
        }
    }
}

// The file open in the source panel.
#[derive(Debug)]
pub(in crate::editor::hook) struct SourceState {
    pub(in crate::editor::hook) key: SourceKey,
    // The resolved path it is read from and saved to.
    pub(in crate::editor::hook) path: String,
    pub(in crate::editor::hook) area: TextArea,
    // Whether the text area holds the keyboard (while the panel is frontmost).
    pub(in crate::editor::hook) focus: bool,
    pub(in crate::editor::hook) status: Option<Status>,
    pub(in crate::editor::hook) markers: Vec<GutterMarker>,
    // Where a click on the status line jumps: the first error marked here.
    pub(in crate::editor::hook) jump: Option<(usize, usize)>,
    // The last board sequence taken in.
    pub(in crate::editor::hook) seen: u64,
    // A save is waiting on its recompile's report.
    pub(in crate::editor::hook) compiling: bool,
    // Whether the Shader is in the running world's reload catalog (`None`
    // until the driver has armed one).
    pub(in crate::editor::hook) live: Option<bool>,
    // What the file held when loaded or last saved, its modification time
    // then, and the changed text a notice was already shown for.
    pub(in crate::editor::hook) known: String,
    pub(in crate::editor::hook) mtime: Option<SystemTime>,
    pub(in crate::editor::hook) noticed: Option<String>,
    // When the file on disk is next compared against `known`, in session
    // seconds.
    pub(in crate::editor::hook) next_check: f64,
}

// The names the engine provides the file `key` opens.
pub(in crate::editor::hook) fn vocabulary_of(key: &SourceKey) -> &'static [Entry] {
    match key {
        SourceKey::Shader { .. } => vocabulary::ENTRIES,
        SourceKey::Field { .. } => vocabulary::sdf::ENTRIES,
    }
}

// The file's text area, highlighted as HLSL with the engine's names for its
// kind set apart.
fn source_area(key: &SourceKey, text: &str) -> TextArea {
    let highlighter = match key {
        SourceKey::Shader { .. } => &HLSL,
        SourceKey::Field { .. } => &SDF_HLSL,
    };
    TextArea::from_text(text).highlighted(highlighter)
}

// The outcome to show out of `newer`, the subjects' latest reports since the
// last look: a failure when any volume reading the file failed, else the
// newest.
fn shown<'a>(newer: &[&'a Latest]) -> Option<&'a Latest> {
    newer
        .iter()
        .find(|l| matches!(l.outcome, ReloadOutcome::Failed(_)))
        .or_else(|| newer.iter().max_by_key(|l| l.seq))
        .copied()
}

// How often the open file is checked for a change on disk.
const DISK_CHECK_S: f64 = 0.5;

impl SourceState {
    pub(in crate::editor::hook) fn new(key: SourceKey, path: String, text: String) -> Self {
        let area = source_area(&key, &text);
        Self {
            key,
            mtime: modified(&path),
            area,
            path,
            focus: false,
            status: None,
            markers: Vec::new(),
            jump: None,
            seen: 0,
            compiling: false,
            live: None,
            known: text,
            noticed: None,
            next_check: 0.0,
        }
    }

    // The title: the Shader and which of its files, or the field's file.
    pub(in crate::editor::hook) fn title(&self) -> String {
        match &self.key {
            SourceKey::Shader { name, stage } => {
                format!("{name} {}", shader_source::stage_name(*stage))
            }
            SourceKey::Field { path } => format!("{} SDF field", file_name(path)),
        }
    }

    // What the file belongs to, as a status line names it.
    fn owner(&self) -> &'static str {
        match self.key {
            SourceKey::Shader { .. } => "Shader",
            SourceKey::Field { .. } => "SDF field",
        }
    }

    // The file name alone, for the unsaved-changes question.
    pub(in crate::editor::hook) fn file_name(&self) -> &str {
        self.path.rsplit(['/', '\\']).next().unwrap_or(&self.path)
    }

    // Take in what the board reports since the last look on `subjects`, what
    // reads the file (its Shader, or the volumes reading the field): a newer
    // outcome, or a catalog armed from a rebuilt world. The file is live once
    // all of them are. `true` when an outcome answering a save of this
    // panel's arrived with an error to jump to.
    pub(in crate::editor::hook) fn take_board(
        &mut self,
        board: &ReportBoard,
        subjects: &[ReloadSubject],
    ) -> bool {
        let live: Vec<Option<bool>> = subjects.iter().map(|s| board.is_live(s)).collect();
        self.live = match live.as_slice() {
            [] => None,
            l if l.contains(&Some(false)) => Some(false),
            l if l.iter().all(|l| *l == Some(true)) => Some(true),
            _ => None,
        };
        let newer: Vec<&Latest> = subjects
            .iter()
            .filter_map(|s| board.latest(s))
            .filter(|l| l.seq > self.seen)
            .collect();
        if let Some(latest) = shown(&newer) {
            self.seen = newer.iter().map(|l| l.seq).max().unwrap_or(latest.seq);
            let view = shader_diagnostics::report_view(&latest.outcome, &self.path);
            let answered = std::mem::take(&mut self.compiling);
            self.markers = view.markers;
            self.status = Some(view.status);
            self.jump = view.jump;
            return answered && self.jump.is_some();
        }
        if let Some(armed) = board.armed_at()
            && armed > self.seen
        {
            self.seen = armed;
            // The rebuilt world compiled every file from disk.
            if std::mem::take(&mut self.compiling) {
                self.markers.clear();
                self.jump = None;
                self.status = Some(Status::new("Compiled with the rebuilt world", Tone::Info));
            }
        }
        false
    }

    // The saved text is on disk now: the buffer is clean against it, and its
    // recompile is awaited (unless what reads it is not in the running world
    // yet, in which case the next rebuild compiles it).
    pub(in crate::editor::hook) fn saved(&mut self, text: String) {
        self.area.mark_saved();
        self.known = text;
        self.noticed = None;
        self.mtime = modified(&self.path);
        match self.live {
            Some(false) => {
                let owner = self.owner();
                self.status = Some(Status::new(
                    format!("Saved; the {owner} joins the running world when it rebuilds"),
                    Tone::Info,
                ))
            }
            _ => {
                self.compiling = true;
                self.status = Some(Status::new("Compiling...", Tone::Info));
            }
        }
    }

    // Compare the file on disk with what the buffer was loaded from, at most
    // every `DISK_CHECK_S` of session time `now`. A clean buffer follows the
    // file; a dirty one is kept, with a notice.
    pub(in crate::editor::hook) fn check_disk(&mut self, now: f64) {
        if now < self.next_check {
            return;
        }
        self.next_check = now + DISK_CHECK_S;
        let mtime = modified(&self.path);
        if mtime == self.mtime {
            return;
        }
        self.mtime = mtime;
        let Ok(disk) = std::fs::read_to_string(&self.path) else {
            return;
        };
        match shader_source::disk_change(
            &self.known,
            &disk,
            self.area.is_dirty(),
            self.noticed.as_deref(),
        ) {
            DiskChange::Unchanged => {}
            DiskChange::Reload => {
                let caret = self.area.caret();
                self.area = source_area(&self.key, &disk);
                self.area.go_to(caret.line, caret.col);
                self.known = disk;
                self.noticed = None;
            }
            DiskChange::Notice => {
                self.status = Some(Status::new(
                    "The file changed on disk; Save overwrites it with these edits",
                    Tone::Warning,
                ));
                self.noticed = Some(disk);
            }
        }
    }
}

fn modified(path: &str) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debug::hot_reload::{ReloadFailure, ReloadOutcome, ReloadReport, ReloadSubject};
    use concinnity_cook::compile::program::{CompileFailure, Diagnostic, Severity};
    use concinnity_core::components::ShaderStage;

    fn state(path: &str, text: &str) -> SourceState {
        SourceState::new(
            SourceKey::shader("water", ShaderStage::Fragment),
            path.to_string(),
            text.to_string(),
        )
    }

    fn water() -> [ReloadSubject; 1] {
        [ReloadSubject::shader("water")]
    }

    fn failed_at(path: &str, line: u32) -> ReloadOutcome {
        ReloadOutcome::Failed(ReloadFailure::Compile(CompileFailure {
            owner: "Shader 'water'".to_string(),
            failures: Vec::new(),
            diagnostics: vec![Diagnostic {
                path: path.to_string(),
                line,
                column: 3,
                severity: Severity::Error,
                message: "bad".to_string(),
                context: String::new(),
            }],
            hint: "",
        }))
    }

    fn publish(reports: &ReloadReports, outcome: ReloadOutcome) {
        reports.publish(&[ReloadReport {
            subject: ReloadSubject::shader("water"),
            outcome,
        }]);
    }

    // A save waits on its report; the report marks the file and, being the
    // answer to that save, asks for the jump. A later clean compile clears it.
    #[test]
    fn a_save_is_answered_by_its_report() {
        let reports = ReloadReports::default();
        reports.arm([ReloadSubject::shader("water")]);
        let mut s = state("/cn-none/water.hlsl", "a\nb\nc");
        assert!(!s.take_board(&reports.snapshot(), &water()));
        assert_eq!(s.live, Some(true));
        s.saved("a\nb\nc".to_string());
        assert!(s.compiling);
        publish(&reports, failed_at("/cn-none/water.hlsl", 2));
        assert!(
            s.take_board(&reports.snapshot(), &water()),
            "the save's error jumps"
        );
        assert!(!s.compiling);
        assert_eq!(s.markers.len(), 1);
        assert_eq!(s.jump, Some((1, 2)));
        assert!(!s.take_board(&reports.snapshot(), &water()), "taken once");

        publish(
            &reports,
            ReloadOutcome::Swapped {
                frame_time: std::time::Duration::ZERO,
                warnings: Vec::new(),
            },
        );
        assert!(!s.take_board(&reports.snapshot(), &water()));
        assert!(s.markers.is_empty());
        assert_eq!(s.jump, None);
    }

    // A report nobody saved for (an external editor's save) marks the file but
    // does not move the caret.
    #[test]
    fn an_unasked_report_marks_without_jumping() {
        let reports = ReloadReports::default();
        reports.arm([ReloadSubject::shader("water")]);
        let mut s = state("/cn-none/water.hlsl", "a\nb");
        publish(&reports, failed_at("/cn-none/water.hlsl", 1));
        assert!(!s.take_board(&reports.snapshot(), &water()));
        assert_eq!(s.markers.len(), 1);
    }

    // A Shader outside the catalog is compiled by the next rebuild, not by a
    // recompile; a rebuild answers a save still waiting.
    #[test]
    fn a_new_shader_waits_for_the_rebuild() {
        let reports = ReloadReports::default();
        reports.arm([ReloadSubject::shader("lit")]);
        let mut s = state("/cn-none/water.hlsl", "a");
        s.take_board(&reports.snapshot(), &water());
        assert_eq!(s.live, Some(false));
        s.saved("a".to_string());
        assert!(!s.compiling);

        reports.arm([ReloadSubject::shader("lit"), ReloadSubject::shader("water")]);
        s.take_board(&reports.snapshot(), &water());
        s.saved("a".to_string());
        assert!(s.compiling);
        reports.arm([ReloadSubject::shader("lit"), ReloadSubject::shader("water")]);
        s.take_board(&reports.snapshot(), &water());
        assert!(!s.compiling);
        assert_eq!(s.status.as_ref().unwrap().tone, Tone::Info);
    }

    // A field is read by several volumes: it is live only once every one is,
    // and a save's answer shows a failure in any of them over a success.
    #[test]
    fn a_field_takes_the_reports_of_every_volume_reading_it() {
        let volumes = ["a", "b"].map(ReloadSubject::sdf_volume);
        let key = SourceKey::Field {
            path: "/cn-none/blob.hlsl".to_string(),
        };
        let mut s = SourceState::new(key, "/cn-none/blob.hlsl".to_string(), "a\nb".to_string());
        assert_eq!(s.title(), "blob.hlsl SDF field");
        let reports = ReloadReports::default();
        reports.arm([ReloadSubject::sdf_volume("a")]);
        s.take_board(&reports.snapshot(), &volumes);
        assert_eq!(s.live, Some(false), "b joins at the rebuild");
        s.saved("a\nb".to_string());
        assert!(!s.compiling);
        assert!(s.status.as_ref().unwrap().text.contains("SDF field"));

        reports.arm(volumes.clone());
        s.take_board(&reports.snapshot(), &volumes);
        assert_eq!(s.live, Some(true));
        s.saved("a\nb".to_string());
        reports.publish(&[
            ReloadReport {
                subject: volumes[0].clone(),
                outcome: failed_at("/cn-none/blob.hlsl", 2),
            },
            ReloadReport {
                subject: volumes[1].clone(),
                outcome: ReloadOutcome::Swapped {
                    frame_time: std::time::Duration::ZERO,
                    warnings: Vec::new(),
                },
            },
        ]);
        assert!(
            s.take_board(&reports.snapshot(), &volumes),
            "the failure jumps"
        );
        assert_eq!(s.jump, Some((1, 2)));
        assert!(!s.take_board(&reports.snapshot(), &volumes), "taken once");
    }

    // A clean buffer follows the file on disk; a dirty one keeps its edits and
    // says so once.
    #[test]
    fn a_change_on_disk_reloads_clean_and_notices_dirty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("water.hlsl");
        std::fs::write(&path, "one").unwrap();
        let mut s = state(&path.to_string_lossy(), "one");

        std::fs::write(&path, "two").unwrap();
        s.mtime = None;
        s.check_disk(0.0);
        assert_eq!(s.area.text(), "two");
        assert!(!s.area.is_dirty());
        assert!(s.status.is_none(), "a clean reload is silent");

        s.area.type_char('x');
        std::fs::write(&path, "three").unwrap();
        s.mtime = None;
        s.check_disk(0.1);
        assert!(s.noticed.is_none(), "throttled");
        s.check_disk(1.0);
        assert_eq!(s.area.text(), "xtwo", "the edits are kept");
        assert_eq!(s.status.as_ref().unwrap().tone, Tone::Warning);
        assert_eq!(s.noticed.as_deref(), Some("three"));

        std::fs::write(&path, s.area.text()).unwrap();
        s.saved(s.area.text());
        s.mtime = None;
        s.check_disk(2.0);
        assert_eq!(s.area.text(), "xtwo", "the save is what the file holds");
    }
}
