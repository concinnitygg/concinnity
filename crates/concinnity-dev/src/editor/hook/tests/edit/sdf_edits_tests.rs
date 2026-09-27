//! The Shaders panel's field rows and their edits (`hook/edit/shaders.rs`,
//! `hook/edit/sdf_edits.rs`): a field opening in the source panel with its
//! own vocabulary, selecting its volumes, and deleting it, which removes every
//! volume reading it in one undo step and the file only when asked and when
//! nothing left reads it.

use concinnity_cook::build_only::include::SourcedEntry;
use concinnity_core::render::shader_programs::vocabulary::sdf;
use std::path::Path;

use super::shader_fixtures::{click, message, pick, toasts};
use crate::editor::entry_list::EntryList;
use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::{
    hook, press_modal, press_modal_check, selected, world_with_name_field,
};
use crate::editor::panels::shader_list::{MenuItem, RowKind};
use crate::editor::panels::shader_source::SourceKey;
use crate::editor::panels::shader_source_panel::SourceAction;

fn volume(name: &str, file: &Path) -> serde_json::Value {
    serde_json::json!({"type": "SdfVolume", "args": {
        "$id": name, "fragment_shader": file.to_string_lossy(),
    }})
}

fn volume_names(h: &EditorHook) -> Vec<String> {
    h.entries
        .iter()
        .filter(|e| e["type"] == "SdfVolume")
        .filter_map(|e| e["args"]["$id"].as_str().map(str::to_string))
        .collect()
}

// Two volumes over one field file, and a third over its own.
fn session(dir: &Path) -> EditorHook {
    let blob = dir.join("blob.hlsl");
    let cloud = dir.join("cloud.hlsl");
    std::fs::write(
        &blob,
        "float map(float3 p, SdfParams q, float t) { return 1.0; }",
    )
    .unwrap();
    std::fs::write(&cloud, "cloud").unwrap();
    hook(vec![
        volume("left", &blob),
        volume("right", &blob),
        volume("cloud", &cloud),
    ])
}

// A field row opens its file, titled by the file and highlighted and
// referenced with the field's own names rather than a Shader's.
#[test]
fn a_field_row_opens_the_file_with_the_fields_vocabulary() {
    let dir = tempfile::tempdir().unwrap();
    let mut h = session(dir.path());
    let mut world = world_with_name_field();
    click(&mut h, &mut world, RowKind::Field(0));
    let src = h.shaders.source.as_ref().expect("the field is open");
    assert!(matches!(&src.key, SourceKey::Field { path } if path.ends_with("blob.hlsl")));
    assert_eq!(src.title(), "blob.hlsl SDF field");
    let rows = h.shader_rows().to_vec();
    assert!(
        rows.iter()
            .any(|r| r.kind == RowKind::Field(0) && r.selected)
    );
    // The column lists the field's names: its slot for `map` writes the
    // prototype a field defines.
    let rows = h.shaders.reference.rows(sdf::ENTRIES);
    let slot = rows.iter().position(|r| r.text() == "map()").unwrap();
    let src = h.shaders.source.as_mut().unwrap();
    src.area.go_to(0, 0);
    h.apply_source_action(SourceAction::Reference(slot), 0.0, 0.0);
    let text = h.shaders.source.as_ref().unwrap().area.text();
    assert!(
        text.starts_with("float map(float3 p, SdfParams params, float time)float map"),
        "{text}"
    );
}

#[test]
fn select_volumes_selects_every_volume_reading_the_field() {
    let dir = tempfile::tempdir().unwrap();
    let mut h = session(dir.path());
    let mut world = world_with_name_field();
    pick(
        &mut h,
        &mut world,
        RowKind::Field(0),
        MenuItem::SelectVolumes,
    );
    assert_eq!(selected(&h), ["left", "right"]);
}

// Delete removes both volumes reading the field in one undo step; unchecked,
// the file stays. Checked, it goes after the volumes, and undo brings back
// only the volumes.
#[test]
fn a_field_delete_removes_its_volumes_in_one_step() {
    let dir = tempfile::tempdir().unwrap();
    let mut h = session(dir.path());
    let mut world = world_with_name_field();
    let blob = dir.path().join("blob.hlsl");

    pick(&mut h, &mut world, RowKind::Field(0), MenuItem::Delete);
    let text = message(&h);
    assert!(
        text.contains("Its volumes 'left' and 'right' are deleted."),
        "{text}"
    );
    let check = h.modal.as_ref().unwrap().check.clone().unwrap();
    assert!(check.enabled && !check.on);
    press_modal(&mut h, &mut world, "Delete");
    assert_eq!(volume_names(&h), ["cloud"]);
    assert!(blob.exists(), "unchecked: the file stays");
    h.undo(&mut world);
    assert_eq!(volume_names(&h), ["left", "right", "cloud"]);
    assert!(!h.can_undo(), "the delete was one step");

    pick(&mut h, &mut world, RowKind::Field(0), MenuItem::Delete);
    press_modal_check(&mut h, &mut world);
    press_modal(&mut h, &mut world, "Delete");
    assert_eq!(volume_names(&h), ["cloud"]);
    assert!(!blob.exists(), "checked: the file goes");
    assert!(toasts(&h).iter().any(|t| t.ends_with("not the file")));
    h.undo(&mut world);
    assert_eq!(volume_names(&h), ["left", "right", "cloud"]);
}

// Deleting the field the source panel shows closes it first.
#[test]
fn deleting_the_open_field_closes_it() {
    let dir = tempfile::tempdir().unwrap();
    let mut h = session(dir.path());
    let mut world = world_with_name_field();
    click(&mut h, &mut world, RowKind::Field(1));
    assert!(h.shaders.source.is_some());
    pick(&mut h, &mut world, RowKind::Field(1), MenuItem::Delete);
    press_modal(&mut h, &mut world, "Delete");
    assert!(h.shaders.source.is_none());
    assert_eq!(volume_names(&h), ["left", "right"]);
}

// A volume included from another file is not the editor's to remove, so it
// stays, still reading the file: the box cannot be checked.
#[test]
fn an_included_volume_keeps_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let blob = dir.path().join("blob.hlsl");
    std::fs::write(&blob, "blob").unwrap();
    let mut h = hook(Vec::new());
    h.entries = EntryList::with_includes(vec![
        SourcedEntry {
            entry: volume("own", &blob),
            file: None,
        },
        SourcedEntry {
            entry: volume("shared", &blob),
            file: Some("lib.jsonl".into()),
        },
    ]);
    h.baseline = h.entries.clone();
    let mut world = world_with_name_field();
    pick(&mut h, &mut world, RowKind::Field(0), MenuItem::Delete);
    let check = h.modal.as_ref().unwrap().check.clone().unwrap();
    assert!(!check.enabled);
    assert_eq!(
        check.note.as_deref(),
        Some("blob.hlsl is still read by 'shared'")
    );
    press_modal(&mut h, &mut world, "Delete");
    assert_eq!(volume_names(&h), ["shared"]);
    assert!(blob.exists());
}
