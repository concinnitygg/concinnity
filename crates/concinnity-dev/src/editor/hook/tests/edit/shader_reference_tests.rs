//! The Shader source panel's reference column (`hook/edit/shader_source.rs`):
//! a name clicked goes in at the caret, a heading folds its group, the toggle
//! hides the column, and the wheel over it scrolls its rows.

use concinnity_engine::components::ShaderStage;
use concinnity_engine::render::shader::programs::vocabulary::ENTRIES;
use std::path::Path;

use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::hook;
use crate::editor::panels::registry::{self, PanelKey};
use crate::editor::panels::shader::reference::RefRow;
use crate::editor::panels::shader::source::SourceKey;
use crate::editor::panels::shader::source_panel::SourceAction;

// A hook with one Shader open in the source panel, its file holding "lit".
fn session(dir: &Path) -> EditorHook {
    let lit = dir.join("lit.hlsl");
    std::fs::write(&lit, "lit").unwrap();
    let mut h = hook(vec![serde_json::json!({"type": "Shader", "args": {
        "$id": "lit",
        "fragment": lit.to_string_lossy(),
    }})]);
    h.open_shader_file(SourceKey::shader("lit", ShaderStage::Fragment));
    h
}

fn text(h: &EditorHook) -> String {
    h.shaders.source.as_ref().unwrap().area.text()
}

// The row reading `label`.
fn slot_of(h: &EditorHook, label: &str) -> usize {
    let rows = h.shaders.reference.rows(ENTRIES);
    rows.iter().position(|r| r.text() == label).unwrap()
}

#[test]
fn a_name_goes_in_at_the_caret_as_one_undo_step() {
    let dir = tempfile::tempdir().unwrap();
    let mut h = session(dir.path());
    let src = h.shaders.source.as_mut().unwrap();
    src.area.go_to(0, 3);
    src.focus = false;
    let slot = slot_of(&h, "shade_surface()");
    h.apply_source_action(SourceAction::Reference(slot));
    assert_eq!(text(&h), "litshade_surface(v, od)");
    let src = h.shaders.source.as_mut().unwrap();
    assert!(src.focus, "the text takes the keyboard back");
    assert!(src.area.is_dirty());
    src.area.undo();
    assert_eq!(text(&h), "lit");

    let slot = slot_of(&h, "VIEW.elapsed");
    h.apply_source_action(SourceAction::Reference(slot));
    assert_eq!(text(&h), "litVIEW.elapsed");
}

#[test]
fn a_heading_folds_its_group() {
    let dir = tempfile::tempdir().unwrap();
    let mut h = session(dir.path());
    let before = h.shaders.reference.rows(ENTRIES).len();
    h.apply_source_action(SourceAction::Reference(0));
    let rows = h.shaders.reference.rows(ENTRIES);
    assert!(matches!(rows[0], RefRow::Heading { folded: true, .. }));
    assert!(rows.len() < before);
    assert_eq!(text(&h), "lit", "a heading inserts nothing");
    h.apply_source_action(SourceAction::Reference(0));
    assert_eq!(h.shaders.reference.rows(ENTRIES).len(), before);
}

#[test]
fn the_toggle_hides_the_column_and_narrows_the_panel() {
    let dir = tempfile::tempdir().unwrap();
    let mut h = session(dir.path());
    let panel = registry::panel(PanelKey::ShaderSource);
    let wide = panel.size(&h);
    assert!(h.shaders.reference.open, "shown by default");
    h.apply_source_action(SourceAction::ToggleReference);
    assert!(!h.shaders.reference.open);
    assert!(panel.size(&h)[0] < wide[0]);
}
