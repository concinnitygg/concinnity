use super::*;
use crate::debug::hot_reload::{ReloadFailure, ReloadReport, ReloadReports};
use crate::editor::panels::shader_diagnostics::Tone;
use crate::editor::panels::shader_list::MenuItem;
use concinnity_cook::compile::program::{CompileFailure, Diagnostic, Severity};
use serde_json::json;

fn volume(name: Option<&str>, file: &str, volumetric: bool) -> serde_json::Value {
    let mut args = json!({"fragment_shader": file, "volumetric": volumetric});
    if let Some(name) = name {
        args["$id"] = json!(name);
    }
    json!({"type": "SdfVolume", "args": args})
}

fn entries() -> Vec<serde_json::Value> {
    vec![
        volume(Some("blob_left"), "shaders/blob.hlsl", false),
        json!({"type": "Shader", "args": {"$id": "lit", "fragment": "shaders/blob.hlsl"}}),
        volume(Some("cloud"), "shaders/cloud.hlsl", true),
        volume(None, "./shaders/blob.hlsl", false),
        json!({"type": "SdfVolume", "args": {"$id": "empty"}}),
    ]
}

// Two spellings of one file resolve to one path, so they are one field.
fn resolve(declared: &str) -> String {
    format!("/cn-none/assets/{}", declared.trim_start_matches("./"))
}

fn fields() -> Vec<FieldDecl> {
    declared_fields(&entries(), resolve)
}

// One field per file on disk, in the order a volume first reads it, under the
// first spelling; a Shader reading the same file is not a volume, and a
// volume declaring no file reads no field.
#[test]
fn fields_are_listed_once_per_file_with_their_volumes() {
    let fields = fields();
    let declared: Vec<&str> = fields.iter().map(|f| f.declared.as_str()).collect();
    assert_eq!(declared, ["shaders/blob.hlsl", "shaders/cloud.hlsl"]);
    assert_eq!(fields[0].path, "/cn-none/assets/shaders/blob.hlsl");
    assert_eq!(fields[0].names(), ["blob_left", "SdfVolume#0"]);
    assert_eq!(fields[1].volumes[0].mode(), "volumetric");
    assert_eq!(fields[0].volumes[0].mode(), "surface");
}

// A heading, then each file badged with its status and each volume under it;
// the open field is selected, and a file's menu opens, selects or deletes.
#[test]
fn rows_list_each_file_then_its_volumes() {
    let fields = fields();
    let open = SourceKey::Field {
        path: fields[1].path.clone(),
    };
    let rows = field_rows(&fields, &ReportBoard::default(), Some(&open));
    let texts: Vec<&str> = rows.iter().map(|r| r.text.as_str()).collect();
    assert_eq!(
        texts,
        [
            "SDF fields",
            "shaders/blob.hlsl",
            "blob_left  surface",
            "SdfVolume#0  surface",
            "shaders/cloud.hlsl",
            "cloud  volumetric",
        ]
    );
    assert!(!rows[0].clickable());
    assert_eq!(rows[1].kind, RowKind::Field(0));
    assert_eq!(rows[1].badge, Some(("as built".to_string(), Tone::Info)));
    assert_eq!(rows[5].kind, RowKind::FieldVolume("cloud".to_string()));
    assert!(rows[5].clickable() && rows[5].indent);
    assert!(rows[4].selected && !rows[1].selected);
    assert_eq!(
        rows[1].menu(),
        [MenuItem::Open, MenuItem::SelectVolumes, MenuItem::Delete]
    );
    assert!(rows[2].menu().is_empty());
    assert!(field_rows(&[], &ReportBoard::default(), None).is_empty());
}

fn board(reports: &[(&str, ReloadOutcome)]) -> ReportBoard {
    let board = ReloadReports::default();
    board.arm(["blob_left", "SdfVolume#0", "cloud"].map(ReloadSubject::sdf_volume));
    let reports: Vec<ReloadReport> = reports
        .iter()
        .map(|(name, outcome)| ReloadReport {
            subject: ReloadSubject::sdf_volume(name),
            outcome: outcome.clone(),
        })
        .collect();
    board.publish(&reports);
    board.snapshot()
}

fn swapped(warnings: Vec<Diagnostic>) -> ReloadOutcome {
    ReloadOutcome::Swapped {
        frame_time: std::time::Duration::ZERO,
        warnings,
    }
}

fn warning(path: &str) -> Diagnostic {
    Diagnostic {
        path: path.to_string(),
        line: 1,
        column: 1,
        severity: Severity::Warning,
        message: "w".to_string(),
        context: String::new(),
    }
}

// One volume failing fails the field, whatever the others did; warnings count
// only in the field's own file; a volume the running world lacks waits for
// the rebuild.
#[test]
fn a_field_shows_the_worst_of_its_volumes() {
    let fields = fields();
    let blob = &fields[0];
    let failed = ReloadOutcome::Failed(ReloadFailure::Compile(CompileFailure {
        owner: "SdfVolume 'blob_left'".to_string(),
        failures: Vec::new(),
        diagnostics: Vec::new(),
        hint: "",
    }));
    let both = board(&[("blob_left", failed), ("SdfVolume#0", swapped(Vec::new()))]);
    assert_eq!(field_status(&both, blob), FileStatus::Failed);

    let here = board(&[
        ("blob_left", swapped(vec![warning(&blob.path)])),
        ("SdfVolume#0", swapped(vec![warning("raymarch.hlsl")])),
    ]);
    assert_eq!(field_status(&here, blob), FileStatus::Ok(1));

    let reports = ReloadReports::default();
    reports.arm([ReloadSubject::sdf_volume("blob_left")]);
    assert_eq!(field_status(&reports.snapshot(), blob), FileStatus::NotLive);
    assert_eq!(
        field_status(&ReportBoard::default(), blob),
        FileStatus::Built
    );
}
