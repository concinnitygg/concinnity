//! The data half of the Shaders panel: every `Shader` the world declares, in
//! declaration order, with the Materials that name it and its files, and the
//! rows the panel lists them as. A file's status is its Shader's latest reload
//! outcome as it bears on that file.

use concinnity_cook::authoring::world::entry_handles;
use concinnity_core::components::ShaderStage;
use concinnity_core::gfx::render_types::MAX_SHADER_BUCKETS;

use super::shader_diagnostics::Tone;
use super::shader_source::{self, SourceKey, same_file};
use crate::debug::hot_reload::{ReportBoard, ShaderReloadFailure, ShaderReloadOutcome};
use crate::editor::select_related;

// One declared file of a Shader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeclaredFile {
    pub(crate) stage: ShaderStage,
    // As the world line spells it.
    pub(crate) declared: String,
    // Where it is on disk (`shader_source::resolve_path`).
    pub(crate) path: String,
}

// One declared Shader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ShaderDecl {
    pub(crate) name: String,
    // The first declared Shader shades everything no Material assigns.
    pub(crate) default: bool,
    // The Materials naming it, as entry positions and names.
    pub(crate) materials: Vec<(usize, String)>,
    pub(crate) files: Vec<DeclaredFile>,
}

impl ShaderDecl {
    pub(crate) fn file(&self, stage: ShaderStage) -> Option<&DeclaredFile> {
        self.files.iter().find(|f| f.stage == stage)
    }
}

// Every Shader the entries declare, in order, each file's on-disk path from
// `resolve` (`shader_source::resolve_path`, which can walk the assets tree, so
// the caller keeps its answers).
pub(crate) fn declared(
    entries: &[serde_json::Value],
    mut resolve: impl FnMut(&str) -> String,
) -> Vec<ShaderDecl> {
    let names = entry_handles(entries);
    entries
        .iter()
        .enumerate()
        .filter(|(_, e)| entry_type(e) == Some("Shader"))
        .enumerate()
        .filter_map(|(order, (i, e))| {
            let name = names[i].clone()?;
            let args = e.get("args");
            let file = |stage: ShaderStage| {
                let declared = args?.get(shader_source::stage_name(stage))?.as_str()?;
                Some(DeclaredFile {
                    stage,
                    declared: declared.to_string(),
                    path: resolve(declared),
                })
            };
            let files = [ShaderStage::Fragment, ShaderStage::Vertex]
                .into_iter()
                .filter_map(file)
                .collect();
            let materials = select_related::entries_using(entries, &name)
                .into_iter()
                .filter(|&m| entry_type(&entries[m]) == Some("Material"))
                .filter_map(|m| Some((m, names[m].clone()?)))
                .collect();
            Some(ShaderDecl {
                name,
                default: order == 0,
                materials,
                files,
            })
        })
        .collect()
}

// How many rows `rows` lists for `entries`, without resolving any Material:
// the panel sizes itself from this several times a frame.
pub(crate) fn row_count(entries: &[serde_json::Value]) -> usize {
    let files = |e: &serde_json::Value| {
        [ShaderStage::Fragment, ShaderStage::Vertex]
            .into_iter()
            .filter(|&stage| {
                e.get("args")
                    .and_then(|a| a.get(shader_source::stage_name(stage)))
                    .is_some_and(|v| v.is_string())
            })
            .count()
    };
    let per_shader: usize = entries
        .iter()
        .filter(|e| entry_type(e) == Some("Shader"))
        .map(|e| 2 + files(e))
        .sum();
    per_shader + 1
}

// How many Shaders the entries declare.
pub(crate) fn shader_count(entries: &[serde_json::Value]) -> usize {
    entries
        .iter()
        .filter(|e| entry_type(e) == Some("Shader"))
        .count()
}

// Whether a world with `count` Shaders can declare another.
pub(crate) fn can_add_shader(count: usize) -> bool {
    count < MAX_SHADER_BUCKETS
}

// Why "+ New Shader" is unavailable at the limit.
pub(crate) fn limit_reason() -> String {
    format!("{MAX_SHADER_BUCKETS} Shaders is the most a world can declare")
}

fn entry_type(e: &serde_json::Value) -> Option<&str> {
    e.get("type").and_then(|v| v.as_str())
}

// What became of a file the last time its Shader reloaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FileStatus {
    // Nothing reloaded since the world was built: it runs as built.
    Built,
    // The Shader is not in the running world yet; it joins at the next rebuild.
    NotLive,
    // Compiled and swapped in, with this many warnings in the file.
    Ok(usize),
    // Compiled; its scene is not loaded, so it installs when the scene loads.
    AppliesOnLoad(usize),
    // The reload failed on an error in this file, or in no file of its own.
    Failed,
    // The reload failed on an error in another of the Shader's files.
    SiblingFailed,
}

impl FileStatus {
    pub(crate) fn label(self) -> String {
        match self {
            FileStatus::Built => "as built".to_string(),
            FileStatus::NotLive => "after rebuild".to_string(),
            FileStatus::Ok(0) => "ok".to_string(),
            FileStatus::Ok(1) => "1 warning".to_string(),
            FileStatus::Ok(n) => format!("{n} warnings"),
            FileStatus::AppliesOnLoad(_) => "on scene load".to_string(),
            FileStatus::Failed => "failed".to_string(),
            FileStatus::SiblingFailed => "other file failed".to_string(),
        }
    }

    pub(crate) fn tone(self) -> Tone {
        match self {
            FileStatus::Built | FileStatus::NotLive | FileStatus::Ok(0) => Tone::Info,
            FileStatus::AppliesOnLoad(0) => Tone::Info,
            FileStatus::Ok(_) | FileStatus::AppliesOnLoad(_) => Tone::Warning,
            FileStatus::Failed | FileStatus::SiblingFailed => Tone::Error,
        }
    }
}

// The status of `file` of `shader` on `board`.
pub(crate) fn file_status(
    board: &ReportBoard,
    shader: &ShaderDecl,
    file: &DeclaredFile,
) -> FileStatus {
    if board.is_live(&shader.name) == Some(false) {
        return FileStatus::NotLive;
    }
    let Some(latest) = board.latest(&shader.name) else {
        return FileStatus::Built;
    };
    let warnings_here = |warnings: &[concinnity_cook::compile::program::Diagnostic]| {
        warnings
            .iter()
            .filter(|w| same_file(&w.path, &file.path))
            .count()
    };
    match &latest.outcome {
        ShaderReloadOutcome::Swapped { warnings, .. } => FileStatus::Ok(warnings_here(warnings)),
        ShaderReloadOutcome::AppliesOnLoad { warnings } => {
            FileStatus::AppliesOnLoad(warnings_here(warnings))
        }
        ShaderReloadOutcome::Failed(ShaderReloadFailure::Compile(failed)) => {
            let in_sibling = |path: &str| {
                shader
                    .files
                    .iter()
                    .any(|f| f.stage != file.stage && same_file(path, &f.path))
            };
            let here = failed.errors().any(|e| same_file(&e.path, &file.path));
            match !here && failed.errors().any(|e| in_sibling(&e.path)) {
                true => FileStatus::SiblingFailed,
                false => FileStatus::Failed,
            }
        }
        ShaderReloadOutcome::Failed(_) => FileStatus::Failed,
    }
}

// What a row of the panel stands for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RowKind {
    // A Shader's name.
    Header,
    // A line of information under a Shader, or a control that is unavailable.
    Note,
    // The Materials naming Shader `i`; a click selects them.
    Materials(usize),
    // One file of a Shader; a click opens it.
    File(SourceKey),
    // "+ New Shader".
    New,
}

// One row as drawn: its caption, an optional right-hand badge, whether it is
// indented under its Shader, and whether it is the open file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Row {
    pub(crate) kind: RowKind,
    pub(crate) text: String,
    pub(crate) badge: Option<(String, Tone)>,
    pub(crate) indent: bool,
    pub(crate) selected: bool,
}

impl Row {
    // Whether a click on the row does anything.
    pub(crate) fn clickable(&self) -> bool {
        !matches!(self.kind, RowKind::Header | RowKind::Note)
    }
}

// The panel's rows: per Shader its name, the Materials naming it, and a row
// per file, then "+ New Shader" (a note at the Shader limit). `open` is the
// file the source panel shows.
pub(crate) fn rows(
    shaders: &[ShaderDecl],
    board: &ReportBoard,
    open: Option<&SourceKey>,
) -> Vec<Row> {
    let mut out = Vec::new();
    for (i, shader) in shaders.iter().enumerate() {
        out.push(Row {
            kind: RowKind::Header,
            text: shader.name.clone(),
            badge: shader.default.then(|| ("default".to_string(), Tone::Info)),
            indent: false,
            selected: false,
        });
        out.push(materials_row(i, shader));
        for file in &shader.files {
            let key = SourceKey {
                shader: shader.name.clone(),
                stage: file.stage,
            };
            let status = file_status(board, shader, file);
            out.push(Row {
                selected: open == Some(&key),
                kind: RowKind::File(key),
                text: format!(
                    "{}  {}",
                    shader_source::stage_name(file.stage),
                    file.declared
                ),
                badge: Some((status.label(), status.tone())),
                indent: true,
            });
        }
    }
    let (kind, text) = match can_add_shader(shaders.len()) {
        true => (RowKind::New, "+ New Shader".to_string()),
        false => (RowKind::Note, format!("+ New Shader: {}", limit_reason())),
    };
    out.push(Row {
        kind,
        text,
        badge: None,
        indent: false,
        selected: false,
    });
    out
}

fn materials_row(i: usize, shader: &ShaderDecl) -> Row {
    let names: Vec<&str> = shader.materials.iter().map(|(_, n)| n.as_str()).collect();
    let (kind, text) = match (names.is_empty(), shader.default) {
        (false, _) => (
            RowKind::Materials(i),
            format!("used by {}", names.join(", ")),
        ),
        (true, true) => (
            RowKind::Note,
            "shades every Material naming no Shader".to_string(),
        ),
        (true, false) => (RowKind::Note, "no Material names it".to_string()),
    };
    Row {
        kind,
        text,
        badge: None,
        indent: true,
        selected: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debug::hot_reload::{ShaderReloadOutcome, ShaderReports};
    use concinnity_cook::compile::program::{CompileFailure, Diagnostic, Severity};
    use serde_json::json;

    fn entries() -> Vec<serde_json::Value> {
        vec![
            json!({"type": "Shader", "args": {"$id": "lit", "fragment": "/cn-none/lit.hlsl"}}),
            json!({"type": "Material", "args": {"$id": "plain"}}),
            json!({"type": "Shader", "args": {
                "$id": "reeds",
                "fragment": "/cn-none/reeds.hlsl",
                "vertex": "/cn-none/sway.hlsl",
            }}),
            json!({"type": "Material", "args": {"$id": "reed_mat", "shader": "reeds"}}),
            json!({"type": "Material", "args": {"$id": "marsh_mat", "shader": "reeds"}}),
            json!({"type": "Prop", "args": {"$id": "p", "material": "reed_mat"}}),
        ]
    }

    // Declaration order, the default badge on the first, the vertex file only
    // when declared, and the Materials naming each Shader (and nothing else).
    #[test]
    fn shaders_are_listed_in_declaration_order() {
        let shaders = declared(&entries(), str::to_string);
        let names: Vec<&str> = shaders.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["lit", "reeds"]);
        assert!(shaders[0].default && !shaders[1].default);
        assert_eq!(shaders[0].files.len(), 1);
        assert!(shaders[0].file(ShaderStage::Vertex).is_none());
        let vertex = shaders[1].file(ShaderStage::Vertex).unwrap();
        assert_eq!(vertex.declared, "/cn-none/sway.hlsl");
        assert!(shaders[0].materials.is_empty());
        assert_eq!(
            shaders[1].materials,
            [(3, "reed_mat".to_string()), (4, "marsh_mat".to_string())]
        );
    }

    #[test]
    fn an_anonymous_shader_is_listed_under_its_label() {
        let entries = vec![json!({"type": "Shader", "args": {"fragment": "/cn-none/a.hlsl"}})];
        assert_eq!(declared(&entries, str::to_string)[0].name, "Shader#0");
    }

    // The rows: each Shader's header (the default badged), its Materials, its
    // files, then the new-Shader row; the open file is selected.
    #[test]
    fn rows_list_each_shader_then_the_new_row() {
        let shaders = declared(&entries(), str::to_string);
        let open = SourceKey {
            shader: "reeds".to_string(),
            stage: ShaderStage::Vertex,
        };
        let rows = rows(&shaders, &ReportBoard::default(), Some(&open));
        let texts: Vec<&str> = rows.iter().map(|r| r.text.as_str()).collect();
        assert_eq!(
            texts,
            [
                "lit",
                "shades every Material naming no Shader",
                "fragment  /cn-none/lit.hlsl",
                "reeds",
                "used by reed_mat, marsh_mat",
                "fragment  /cn-none/reeds.hlsl",
                "vertex  /cn-none/sway.hlsl",
                "+ New Shader",
            ]
        );
        assert_eq!(rows[0].badge, Some(("default".to_string(), Tone::Info)));
        assert_eq!(rows[3].badge, None);
        assert_eq!(rows[4].kind, RowKind::Materials(1));
        assert!(!rows[1].clickable());
        assert!(rows[6].selected && !rows[5].selected);
        assert_eq!(rows[7].kind, RowKind::New);
        assert!(rows[7].clickable());
        assert_eq!(row_count(&entries()), rows.len());
    }

    // At the Shader limit "+ New Shader" stays listed, unclickable, with why.
    #[test]
    fn the_new_row_is_unavailable_at_the_shader_limit() {
        let full: Vec<serde_json::Value> = (0..MAX_SHADER_BUCKETS)
            .map(|i| json!({"type": "Shader", "args": {"$id": format!("s{i}"), "fragment": "/cn-none/s.hlsl"}}))
            .collect();
        assert!(!can_add_shader(shader_count(&full)));
        assert!(can_add_shader(shader_count(&full[1..])));
        let shaders = declared(&full, str::to_string);
        let rows = rows(&shaders, &ReportBoard::default(), None);
        let last = rows.last().unwrap();
        assert_eq!(last.kind, RowKind::Note);
        assert!(!last.clickable());
        assert!(last.text.contains(&limit_reason()), "{}", last.text);
    }

    fn error_in(path: &str) -> Diagnostic {
        Diagnostic {
            path: path.to_string(),
            line: 1,
            column: 1,
            severity: Severity::Error,
            message: "bad".to_string(),
            context: String::new(),
        }
    }

    fn failed(path: &str) -> ShaderReloadOutcome {
        ShaderReloadOutcome::Failed(ShaderReloadFailure::Compile(CompileFailure {
            owner: "Shader 'reeds'".to_string(),
            failures: Vec::new(),
            diagnostics: vec![error_in(path)],
            hint: "",
        }))
    }

    fn statuses(board: &ReportBoard) -> Vec<FileStatus> {
        let shaders = declared(&entries(), str::to_string);
        shaders[1]
            .files
            .iter()
            .map(|f| file_status(board, &shaders[1], f))
            .collect()
    }

    fn board(name: &str, outcome: ShaderReloadOutcome) -> ReportBoard {
        let reports = ShaderReports::default();
        reports.arm(["lit".to_string(), "reeds".to_string()]);
        reports.publish(&[crate::debug::hot_reload::ShaderReloadReport {
            name: name.to_string(),
            outcome,
        }]);
        reports.snapshot()
    }

    // A failure marks the file its error is in; its sibling reads that another
    // file failed. An error in no file of the Shader fails both.
    #[test]
    fn a_failure_is_charged_to_the_file_it_names() {
        use FileStatus::*;
        assert_eq!(
            statuses(&board("reeds", failed("/cn-none/sway.hlsl"))),
            [SiblingFailed, Failed]
        );
        assert_eq!(
            statuses(&board("reeds", failed("main_bindless.hlsl"))),
            [Failed, Failed]
        );
    }

    #[test]
    fn warnings_count_per_file_and_a_new_shader_waits_for_the_rebuild() {
        use FileStatus::*;
        let warning = Diagnostic {
            severity: Severity::Warning,
            ..error_in("/cn-none/reeds.hlsl")
        };
        let swapped = ShaderReloadOutcome::Swapped {
            frame_time: std::time::Duration::ZERO,
            warnings: vec![warning.clone()],
        };
        assert_eq!(statuses(&board("reeds", swapped)), [Ok(1), Ok(0)]);
        let later = ShaderReloadOutcome::AppliesOnLoad {
            warnings: vec![warning],
        };
        assert_eq!(
            statuses(&board("reeds", later)),
            [AppliesOnLoad(1), AppliesOnLoad(0)]
        );
        assert_eq!(statuses(&ReportBoard::default()), [Built, Built]);
        let reports = ShaderReports::default();
        reports.arm(["lit".to_string()]);
        assert_eq!(statuses(&reports.snapshot()), [NotLive, NotLive]);
    }
}
