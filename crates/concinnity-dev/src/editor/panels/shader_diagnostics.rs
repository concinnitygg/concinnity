//! What a Shader's reload outcome shows in the source panel open on one of its
//! files: gutter markers on the lines of that file its diagnostics name, a
//! status line, and where a click on the status jumps.
//!
//! A diagnostic belongs to the open file when its path is the file's resolved
//! path (see `shader_source::resolve_path`). An error located elsewhere, in the
//! engine template when `shade` has the wrong signature, has no line here: it
//! reaches the status line with the compile's hint, and a `note:` in its
//! context naming a line of this file marks that line instead.

use concinnity_cook::compile::program::{CompileFailure, Diagnostic, Severity as DxcSeverity};

use super::shader_source::same_file;
use crate::debug::hot_reload::{ShaderReloadFailure, ShaderReloadOutcome};
use crate::editor::text_area::markers::{GutterMarker, Severity};

// How a status line reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tone {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Status {
    pub(crate) text: String,
    pub(crate) tone: Tone,
}

impl Status {
    pub(crate) fn new(text: impl Into<String>, tone: Tone) -> Self {
        Self {
            text: text.into(),
            tone,
        }
    }
}

// An outcome as the panel on one file shows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReportView {
    pub(crate) markers: Vec<GutterMarker>,
    pub(crate) status: Status,
    // The first error marked in this file, as (line, column), zero-based.
    pub(crate) jump: Option<(usize, usize)>,
}

// The view of `outcome` for the panel open on the file at resolved path `file`.
pub(crate) fn report_view(outcome: &ShaderReloadOutcome, file: &str) -> ReportView {
    match outcome {
        ShaderReloadOutcome::Swapped { warnings, .. } => {
            compiled(warnings, file, "pipeline swapped")
        }
        ShaderReloadOutcome::AppliesOnLoad { warnings } => {
            compiled(warnings, file, "applies when its scene loads")
        }
        ShaderReloadOutcome::Failed(ShaderReloadFailure::Compile(failed)) => {
            let markers = markers_for(&failed.diagnostics, file);
            let jump = markers
                .iter()
                .find(|m| m.severity == Severity::Error)
                .map(|m| (m.line, m.column));
            let status = match failed.errors().next() {
                Some(first) => error_status(first, file, failed.hint),
                None => Status::new(unlocated(failed), Tone::Error),
            };
            ReportView {
                markers,
                status,
                jump,
            }
        }
        ShaderReloadOutcome::Failed(other) => ReportView {
            markers: Vec::new(),
            status: Status::new(other.to_string(), Tone::Error),
            jump: None,
        },
    }
}

fn compiled(warnings: &[Diagnostic], file: &str, applied: &str) -> ReportView {
    let status = match warnings.len() {
        0 => Status::new(format!("Compiled; {applied}"), Tone::Info),
        1 => Status::new(format!("Compiled with 1 warning; {applied}"), Tone::Warning),
        n => Status::new(
            format!("Compiled with {n} warnings; {applied}"),
            Tone::Warning,
        ),
    };
    ReportView {
        markers: markers_for(warnings, file),
        status,
        jump: None,
    }
}

// A failure that located no error says why only in the compiler's raw output:
// its first line.
fn unlocated(failed: &CompileFailure) -> String {
    failed
        .failures
        .iter()
        .flat_map(|f| f.output.lines())
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map_or_else(|| failed.to_string(), str::to_string)
}

// The first error as the status line puts it: its line alone when it is in
// this file, else where it is, followed by the compile's hint.
fn error_status(first: &Diagnostic, file: &str, hint: &str) -> Status {
    let text = if same_file(&first.path, file) {
        format!("line {}: {}", first.line, first.message)
    } else {
        let at = file_name(&first.path);
        let hint = hint.trim();
        match hint.is_empty() {
            true => format!("{at}:{}: {}", first.line, first.message),
            false => format!("{at}:{}: {} {hint}", first.line, first.message),
        }
    };
    Status::new(text, Tone::Error)
}

// A marker per distinct (line, message) the diagnostics pin to `file`, in the
// order first reported: those located in it, and for an error located
// elsewhere, each line of `file` one of its notes names.
pub(crate) fn markers_for(diagnostics: &[Diagnostic], file: &str) -> Vec<GutterMarker> {
    let mut out: Vec<GutterMarker> = Vec::new();
    let mut push = |marker: GutterMarker| {
        let seen = out.iter().any(|m| {
            m.line == marker.line && m.severity == marker.severity && m.message == marker.message
        });
        if !seen {
            out.push(marker);
        }
    };
    for d in diagnostics {
        let severity = match d.severity {
            DxcSeverity::Error => Severity::Error,
            DxcSeverity::Warning => Severity::Warning,
        };
        if same_file(&d.path, file) {
            if let Some(line) = (d.line as usize).checked_sub(1) {
                push(GutterMarker {
                    line,
                    column: (d.column as usize).saturating_sub(1),
                    severity,
                    message: d.message.clone(),
                });
            }
            continue;
        }
        if severity != Severity::Error {
            continue;
        }
        for (line, column) in noted_lines(&d.context, file) {
            push(GutterMarker {
                line,
                column,
                severity,
                message: d.message.clone(),
            });
        }
    }
    out
}

// The (line, column), zero-based, of every `path:line:column: note:` in a
// diagnostic's context whose path is `file`.
fn noted_lines(context: &str, file: &str) -> Vec<(usize, usize)> {
    context
        .lines()
        .filter_map(|text| {
            let (location, _) = text.split_once(": note: ")?;
            let (rest, column) = location.rsplit_once(':')?;
            let (path, line) = rest.rsplit_once(':')?;
            let line: usize = line.parse().ok()?;
            let column: usize = column.parse().ok()?;
            same_file(path, file).then_some((line.checked_sub(1)?, column.saturating_sub(1)))
        })
        .collect()
}

fn file_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_cook::compile::program::EntryFailure;
    use std::time::Duration;

    const FILE: &str = "shaders/water.hlsl";

    fn diagnostic(path: &str, line: u32, severity: DxcSeverity, message: &str) -> Diagnostic {
        Diagnostic {
            path: path.to_string(),
            line,
            column: 5,
            severity,
            message: message.to_string(),
            context: String::new(),
        }
    }

    fn failed(diagnostics: Vec<Diagnostic>, hint: &'static str) -> ShaderReloadOutcome {
        ShaderReloadOutcome::Failed(ShaderReloadFailure::Compile(CompileFailure {
            owner: "Shader 'water'".to_string(),
            failures: vec![EntryFailure {
                entry: "fs_main".to_string(),
                output: String::new(),
            }],
            diagnostics,
            hint,
        }))
    }

    // An error in the open file marks its line (zero-based), puts its message
    // on the status line, and is where the status jumps.
    #[test]
    fn an_error_in_the_file_marks_its_line() {
        let view = report_view(
            &failed(
                vec![
                    diagnostic(FILE, 7, DxcSeverity::Error, "undeclared identifier 'x'"),
                    diagnostic("shaders/other.hlsl", 2, DxcSeverity::Error, "elsewhere"),
                ],
                "",
            ),
            FILE,
        );
        assert_eq!(view.markers.len(), 1);
        assert_eq!(view.markers[0].line, 6);
        assert_eq!(view.markers[0].column, 4);
        assert_eq!(view.markers[0].severity, Severity::Error);
        assert_eq!(view.jump, Some((6, 4)));
        assert_eq!(view.status.tone, Tone::Error);
        assert_eq!(view.status.text, "line 7: undeclared identifier 'x'");
    }

    // A path spelled differently from the resolved one still names the file.
    #[test]
    fn a_diagnostic_matches_the_same_file_by_another_spelling() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("lit.hlsl");
        std::fs::write(&file, "").unwrap();
        let spelled = dir.path().join(".").join("lit.hlsl");
        let d = diagnostic(&spelled.to_string_lossy(), 3, DxcSeverity::Error, "bad");
        let markers = markers_for(&[d], &file.to_string_lossy());
        assert_eq!(markers.len(), 1);
    }

    // An error located in the engine template has no line here: the status
    // names where it is and carries the hint, and a note naming a line of this
    // file marks that line.
    #[test]
    fn a_template_error_reaches_the_status_and_its_note_marks_the_line() {
        let mut error = diagnostic(
            "main_bindless.hlsl",
            1540,
            DxcSeverity::Error,
            "no matching function for call to 'shade'",
        );
        error.context = format!(
            "    return shade(v, od);\n           ^~~~~\n\
             {FILE}:2:8: note: candidate function not viable: requires 1 argument\n\
             float4 shade(VertexOut v) {{ return 1.0; }}\n       ^"
        );
        let view = report_view(
            &failed(
                vec![error],
                "\nA Shader's `fragment` file must define `shade`.",
            ),
            FILE,
        );
        assert_eq!(
            view.status.text,
            "main_bindless.hlsl:1540: no matching function for call to 'shade' \
             A Shader's `fragment` file must define `shade`."
        );
        assert_eq!(view.markers.len(), 1);
        assert_eq!((view.markers[0].line, view.markers[0].column), (1, 7));
        assert_eq!(view.jump, Some((1, 7)));
        assert!(view.markers[0].message.contains("'shade'"));
    }

    // A template error whose notes name no line of this file marks nothing.
    #[test]
    fn a_template_error_without_a_note_here_marks_nothing() {
        let mut error = diagnostic("main_bindless.hlsl", 12, DxcSeverity::Error, "boom");
        error.context = "shaders/other.hlsl:4:1: note: declared here".to_string();
        let view = report_view(&failed(vec![error], ""), FILE);
        assert!(view.markers.is_empty());
        assert_eq!(view.jump, None);
        assert_eq!(view.status.text, "main_bindless.hlsl:12: boom");
    }

    // One warning compiled into several programs marks its line once.
    #[test]
    fn a_repeated_warning_marks_its_line_once() {
        let warning = diagnostic(FILE, 3, DxcSeverity::Warning, "implicit truncation");
        let mut moved = warning.clone();
        moved.column = 9;
        let outcome = ShaderReloadOutcome::Swapped {
            frame_time: Duration::from_millis(3),
            warnings: vec![warning.clone(), moved, warning],
        };
        let view = report_view(&outcome, FILE);
        assert_eq!(view.markers.len(), 1);
        assert_eq!(view.markers[0].severity, Severity::Warning);
        assert_eq!(view.status.tone, Tone::Warning);
        assert_eq!(view.jump, None);
    }

    // A clean compile clears every marker.
    #[test]
    fn a_clean_compile_marks_nothing() {
        let outcome = ShaderReloadOutcome::Swapped {
            frame_time: Duration::from_millis(3),
            warnings: Vec::new(),
        };
        let view = report_view(&outcome, FILE);
        assert!(view.markers.is_empty());
        assert_eq!(
            view.status,
            Status::new("Compiled; pipeline swapped", Tone::Info)
        );
    }

    #[test]
    fn an_unloaded_scene_applies_later() {
        let outcome = ShaderReloadOutcome::AppliesOnLoad {
            warnings: vec![diagnostic(
                "shaders/other.hlsl",
                1,
                DxcSeverity::Warning,
                "w",
            )],
        };
        let view = report_view(&outcome, FILE);
        assert!(view.markers.is_empty(), "the warning is in another file");
        assert!(view.status.text.ends_with("applies when its scene loads"));
    }

    #[test]
    fn a_failure_without_diagnostics_shows_its_message() {
        let outcome =
            ShaderReloadOutcome::Failed(ShaderReloadFailure::Rejected("no device".to_string()));
        let view = report_view(&outcome, FILE);
        assert!(view.markers.is_empty());
        assert_eq!(view.status.tone, Tone::Error);
        assert!(view.status.text.contains("no device"));
    }
}
