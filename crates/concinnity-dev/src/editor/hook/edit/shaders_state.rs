//! EditorHook: the Shaders and Shader source panels' session state, beside
//! their actions in `shaders.rs` and `shader_source.rs`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::debug::hot_reload::{ReportBoard, ShaderReports};
use crate::editor::panels::shader_diagnostics::{self, Status, Tone};
use crate::editor::panels::shader_list::{Row, RowKind, ShaderDecl};
use crate::editor::panels::shader_source::{self, DiskChange, SourceKey};
use crate::editor::text_area::TextArea;
use crate::editor::text_area::markers::GutterMarker;

// The list's shown state and scroll, the board the hot-reload driver publishes
// each Shader's latest outcome to, and the file open in the source panel.
#[derive(Debug)]
pub(in crate::editor::hook) struct ShadersState {
    pub(in crate::editor::hook) open: bool,
    pub(in crate::editor::hook) scroll: usize,
    // The row whose "..." menu is open, by what it stands for, so a rebuild of
    // the rows keeps it on the same row.
    pub(in crate::editor::hook) menu: Option<RowKind>,
    pub(in crate::editor::hook) reports: ShaderReports,
    pub(in crate::editor::hook) source: Option<SourceState>,
    // Each declared path's on-disk path under `paths_dir`, from `resolve`.
    // Kept, because resolving a bare file name walks the assets tree.
    pub(in crate::editor::hook) paths: HashMap<String, String>,
    pub(in crate::editor::hook) paths_dir: Option<PathBuf>,
    pub(in crate::editor::hook) resolve: fn(&str, Option<&Path>) -> String,
    // The list's rows, rebuilt only when what they show changes.
    pub(in crate::editor::hook) rows: Vec<Row>,
    pub(in crate::editor::hook) rows_key: Option<RowsKey>,
}

// What the list's rows are built from: the declared Shaders, the board as of
// its latest change, and the file the source panel shows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::editor::hook) struct RowsKey {
    pub(in crate::editor::hook) shaders: Vec<ShaderDecl>,
    pub(in crate::editor::hook) seq: u64,
    pub(in crate::editor::hook) open: Option<SourceKey>,
}

impl Default for ShadersState {
    fn default() -> Self {
        Self {
            open: false,
            scroll: 0,
            menu: None,
            reports: ShaderReports::default(),
            source: None,
            paths: HashMap::new(),
            paths_dir: None,
            resolve: shader_source::resolve_path,
            rows: Vec::new(),
            rows_key: None,
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
        if self.paths_dir.as_deref() != dir {
            self.paths.clear();
            self.paths_dir = dir.map(Path::to_path_buf);
        }
        let resolve = self.resolve;
        self.paths
            .entry(declared.to_string())
            .or_insert_with(|| resolve(declared, dir))
            .clone()
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

// How often the open file is checked for a change on disk.
const DISK_CHECK_S: f64 = 0.5;

impl SourceState {
    pub(in crate::editor::hook) fn new(key: SourceKey, path: String, text: String) -> Self {
        Self {
            key,
            mtime: modified(&path),
            path,
            area: TextArea::from_text(&text),
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

    // The title: the Shader and which of its files.
    pub(in crate::editor::hook) fn title(&self) -> String {
        format!(
            "{} {}",
            self.key.shader,
            shader_source::stage_name(self.key.stage)
        )
    }

    // The file name alone, for the unsaved-changes question.
    pub(in crate::editor::hook) fn file_name(&self) -> &str {
        self.path.rsplit(['/', '\\']).next().unwrap_or(&self.path)
    }

    // Take in what the board reports since the last look: this Shader's newer
    // outcome, or a catalog armed from a rebuilt world. `true` when an outcome
    // answering a save of this panel's arrived with an error to jump to.
    pub(in crate::editor::hook) fn take_board(&mut self, board: &ReportBoard) -> bool {
        self.live = board.is_live(&self.key.shader);
        if let Some(latest) = board.latest(&self.key.shader)
            && latest.seq > self.seen
        {
            self.seen = latest.seq;
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
            // The rebuilt world compiled every Shader from its files on disk.
            if std::mem::take(&mut self.compiling) {
                self.markers.clear();
                self.jump = None;
                self.status = Some(Status::new("Compiled with the rebuilt world", Tone::Info));
            }
        }
        false
    }

    // The saved text is on disk now: the buffer is clean against it, and its
    // recompile is awaited (unless the Shader is not in the running world yet,
    // in which case the next rebuild compiles it).
    pub(in crate::editor::hook) fn saved(&mut self, text: String) {
        self.area.mark_saved();
        self.known = text;
        self.noticed = None;
        self.mtime = modified(&self.path);
        match self.live {
            Some(false) => {
                self.status = Some(Status::new(
                    "Saved; the Shader joins the running world when it rebuilds",
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
                self.area = TextArea::from_text(&disk);
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
    use crate::debug::hot_reload::{ShaderReloadFailure, ShaderReloadOutcome, ShaderReloadReport};
    use concinnity_cook::compile::program::{CompileFailure, Diagnostic, Severity};
    use concinnity_core::components::ShaderStage;

    fn state(path: &str, text: &str) -> SourceState {
        SourceState::new(
            SourceKey {
                shader: "water".to_string(),
                stage: ShaderStage::Fragment,
            },
            path.to_string(),
            text.to_string(),
        )
    }

    fn failed_at(path: &str, line: u32) -> ShaderReloadOutcome {
        ShaderReloadOutcome::Failed(ShaderReloadFailure::Compile(CompileFailure {
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

    fn publish(reports: &ShaderReports, outcome: ShaderReloadOutcome) {
        reports.publish(&[ShaderReloadReport {
            name: "water".to_string(),
            outcome,
        }]);
    }

    // A save waits on its report; the report marks the file and, being the
    // answer to that save, asks for the jump. A later clean compile clears it.
    #[test]
    fn a_save_is_answered_by_its_report() {
        let reports = ShaderReports::default();
        reports.arm(["water".to_string()]);
        let mut s = state("/cn-none/water.hlsl", "a\nb\nc");
        assert!(!s.take_board(&reports.snapshot()));
        assert_eq!(s.live, Some(true));
        s.saved("a\nb\nc".to_string());
        assert!(s.compiling);
        publish(&reports, failed_at("/cn-none/water.hlsl", 2));
        assert!(s.take_board(&reports.snapshot()), "the save's error jumps");
        assert!(!s.compiling);
        assert_eq!(s.markers.len(), 1);
        assert_eq!(s.jump, Some((1, 2)));
        assert!(!s.take_board(&reports.snapshot()), "taken once");

        publish(
            &reports,
            ShaderReloadOutcome::Swapped {
                frame_time: std::time::Duration::ZERO,
                warnings: Vec::new(),
            },
        );
        assert!(!s.take_board(&reports.snapshot()));
        assert!(s.markers.is_empty());
        assert_eq!(s.jump, None);
    }

    // A report nobody saved for (an external editor's save) marks the file but
    // does not move the caret.
    #[test]
    fn an_unasked_report_marks_without_jumping() {
        let reports = ShaderReports::default();
        reports.arm(["water".to_string()]);
        let mut s = state("/cn-none/water.hlsl", "a\nb");
        publish(&reports, failed_at("/cn-none/water.hlsl", 1));
        assert!(!s.take_board(&reports.snapshot()));
        assert_eq!(s.markers.len(), 1);
    }

    // A Shader outside the catalog is compiled by the next rebuild, not by a
    // recompile; a rebuild answers a save still waiting.
    #[test]
    fn a_new_shader_waits_for_the_rebuild() {
        let reports = ShaderReports::default();
        reports.arm(["lit".to_string()]);
        let mut s = state("/cn-none/water.hlsl", "a");
        s.take_board(&reports.snapshot());
        assert_eq!(s.live, Some(false));
        s.saved("a".to_string());
        assert!(!s.compiling);

        reports.arm(["lit".to_string(), "water".to_string()]);
        s.take_board(&reports.snapshot());
        s.saved("a".to_string());
        assert!(s.compiling);
        reports.arm(["lit".to_string(), "water".to_string()]);
        s.take_board(&reports.snapshot());
        assert!(!s.compiling);
        assert_eq!(s.status.as_ref().unwrap().tone, Tone::Info);
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
