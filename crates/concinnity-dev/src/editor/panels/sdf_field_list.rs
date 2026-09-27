//! The Shaders panel's SDF fields: every distance-field file the world's
//! SdfVolumes declare, once per file, with the volumes reading it, and the rows
//! the panel lists them as. A field's status is the worst of its volumes'
//! latest reload outcomes as they bear on that file.

use concinnity_cook::authoring::world::entry_handles;

use super::shader_list::{FileStatus, Row, RowKind};
use super::shader_source::{SourceKey, same_file};
use crate::debug::hot_reload::{ReloadOutcome, ReloadSubject, ReportBoard};

pub(crate) const SDF_VOLUME: &str = "SdfVolume";

// A volume reading a field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FieldVolume {
    pub(crate) name: String,
    pub(crate) volumetric: bool,
}

impl FieldVolume {
    pub(crate) fn mode(&self) -> &'static str {
        match self.volumetric {
            true => "volumetric",
            false => "surface",
        }
    }
}

// One distance-field file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FieldDecl {
    // As the first volume reading it spells it.
    pub(crate) declared: String,
    // Where it is on disk (`shader_source::resolve_field_path`).
    pub(crate) path: String,
    pub(crate) volumes: Vec<FieldVolume>,
}

impl FieldDecl {
    pub(crate) fn names(&self) -> Vec<&str> {
        self.volumes.iter().map(|v| v.name.as_str()).collect()
    }
}

// Every field file the entries' SdfVolumes declare, in the order a volume
// first reads it, grouped by on-disk path from `resolve`
// (`shader_source::resolve_field_path`) as the hot-reload catalog groups them.
// A volume declaring no file reads no field.
pub(crate) fn declared_fields(
    entries: &[serde_json::Value],
    mut resolve: impl FnMut(&str) -> String,
) -> Vec<FieldDecl> {
    let names = entry_handles(entries);
    let mut out: Vec<FieldDecl> = Vec::new();
    for (i, e) in entries.iter().enumerate() {
        if e.get("type").and_then(|t| t.as_str()) != Some(SDF_VOLUME) {
            continue;
        }
        let args = e.get("args");
        let Some(declared) = args
            .and_then(|a| a.get("fragment_shader"))
            .and_then(|v| v.as_str())
            .filter(|d| !d.is_empty())
        else {
            continue;
        };
        let Some(name) = names[i].clone() else {
            continue;
        };
        let volume = FieldVolume {
            name,
            volumetric: args
                .and_then(|a| a.get("volumetric"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        };
        let path = resolve(declared);
        match out.iter_mut().find(|f| f.path == path) {
            Some(field) => field.volumes.push(volume),
            None => out.push(FieldDecl {
                declared: declared.to_string(),
                path,
                volumes: vec![volume],
            }),
        }
    }
    out
}

// The subjects a field's volumes are reported under.
pub(crate) fn subjects(field: &FieldDecl) -> Vec<ReloadSubject> {
    field
        .volumes
        .iter()
        .map(|v| ReloadSubject::sdf_volume(&v.name))
        .collect()
}

// What became of `field` on `board`, as the worst of its volumes' outcomes: a
// surface and a volumetric volume compile one file differently, so either can
// fail alone.
pub(crate) fn field_status(board: &ReportBoard, field: &FieldDecl) -> FileStatus {
    subjects(field)
        .iter()
        .map(|subject| volume_status(board, subject, &field.path))
        .max_by_key(|s| severity(*s))
        .unwrap_or(FileStatus::Built)
}

fn volume_status(board: &ReportBoard, subject: &ReloadSubject, path: &str) -> FileStatus {
    if board.is_live(subject) == Some(false) {
        return FileStatus::NotLive;
    }
    let Some(latest) = board.latest(subject) else {
        return FileStatus::Built;
    };
    let here = |warnings: &[concinnity_cook::compile::program::Diagnostic]| {
        warnings.iter().filter(|w| same_file(&w.path, path)).count()
    };
    match &latest.outcome {
        ReloadOutcome::Swapped { warnings, .. } => FileStatus::Ok(here(warnings)),
        ReloadOutcome::AppliesOnLoad { warnings } => FileStatus::AppliesOnLoad(here(warnings)),
        ReloadOutcome::Failed(_) => FileStatus::Failed,
    }
}

// How much a status needs the author's attention, for picking the worst.
fn severity(status: FileStatus) -> u8 {
    match status {
        FileStatus::Built => 0,
        FileStatus::Ok(0) => 1,
        FileStatus::AppliesOnLoad(0) => 2,
        FileStatus::NotLive => 3,
        FileStatus::Ok(_) | FileStatus::AppliesOnLoad(_) => 4,
        FileStatus::SiblingFailed | FileStatus::Failed => 5,
    }
}

// The SDF section: a heading, then per field its file (badged with its
// status, selected while open) and a row per volume reading it. No fields, no
// section.
pub(crate) fn field_rows(
    fields: &[FieldDecl],
    board: &ReportBoard,
    open: Option<&SourceKey>,
) -> Vec<Row> {
    if fields.is_empty() {
        return Vec::new();
    }
    let mut out = vec![Row {
        kind: RowKind::Section,
        text: "SDF fields".to_string(),
        badge: None,
        indent: false,
        selected: false,
    }];
    for (i, field) in fields.iter().enumerate() {
        let status = field_status(board, field);
        let key = SourceKey::Field {
            path: field.path.clone(),
        };
        out.push(Row {
            kind: RowKind::Field(i),
            text: field.declared.clone(),
            badge: Some((status.label(), status.tone())),
            indent: false,
            selected: open == Some(&key),
        });
        out.extend(field.volumes.iter().map(|v| Row {
            kind: RowKind::FieldVolume(v.name.clone()),
            text: format!("{}  {}", v.name, v.mode()),
            badge: None,
            indent: true,
            selected: false,
        }));
    }
    out
}

#[cfg(test)]
mod tests;
