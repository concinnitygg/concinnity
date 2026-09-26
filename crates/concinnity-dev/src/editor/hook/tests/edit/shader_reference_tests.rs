//! The Shader source panel's reference column (`hook/edit/shader_source.rs`):
//! a name clicked goes in at the caret, a heading folds its group, the toggle
//! hides the column, and the wheel over it scrolls its rows.

use concinnity_core::components::ShaderStage;
use concinnity_core::ecs::World;
use concinnity_core::render::shader_programs::vocabulary::ENTRIES;
use std::path::Path;

use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::hook;
use crate::editor::panels::registry::{self, PanelKey};
use crate::editor::panels::shader_reference::RefRow;
use crate::editor::panels::shader_source::SourceKey;
use crate::editor::panels::shader_source_panel::{self, SourceAction};

// A hook with one Shader open in the source panel, its file holding "lit".
fn session(dir: &Path) -> EditorHook {
    let lit = dir.join("lit.hlsl");
    std::fs::write(&lit, "lit").unwrap();
    let mut h = hook(vec![serde_json::json!({"type": "Shader", "args": {
        "$id": "lit",
        "fragment": lit.to_string_lossy(),
    }})]);
    h.open_shader_file(SourceKey {
        shader: "lit".to_string(),
        stage: ShaderStage::Fragment,
    });
    h
}

fn text(h: &EditorHook) -> String {
    h.shaders.source.as_ref().unwrap().area.text()
}

// The shown slot of the row reading `label`.
fn slot_of(h: &EditorHook, label: &str) -> usize {
    let rows = h.shaders.reference.rows(ENTRIES);
    let at = rows.iter().position(|r| r.text() == label).unwrap();
    at - h.shaders.reference.scroll
}

#[test]
fn a_name_goes_in_at_the_caret_as_one_undo_step() {
    let dir = tempfile::tempdir().unwrap();
    let mut h = session(dir.path());
    let src = h.shaders.source.as_mut().unwrap();
    src.area.go_to(0, 3);
    src.focus = false;
    let slot = slot_of(&h, "shade_surface()");
    h.apply_source_action(SourceAction::Reference(slot), 0.0, 0.0);
    assert_eq!(text(&h), "litshade_surface(v, od)");
    let src = h.shaders.source.as_mut().unwrap();
    assert!(src.focus, "the text takes the keyboard back");
    assert!(src.area.is_dirty());
    src.area.undo();
    assert_eq!(text(&h), "lit");

    let slot = slot_of(&h, "VIEW.elapsed");
    h.apply_source_action(SourceAction::Reference(slot), 0.0, 0.0);
    assert_eq!(text(&h), "litVIEW.elapsed");
}

#[test]
fn a_heading_folds_its_group() {
    let dir = tempfile::tempdir().unwrap();
    let mut h = session(dir.path());
    let before = h.shaders.reference.rows(ENTRIES).len();
    h.apply_source_action(SourceAction::Reference(0), 0.0, 0.0);
    let rows = h.shaders.reference.rows(ENTRIES);
    assert!(matches!(rows[0], RefRow::Heading { folded: true, .. }));
    assert!(rows.len() < before);
    assert_eq!(text(&h), "lit", "a heading inserts nothing");
    h.apply_source_action(SourceAction::Reference(0), 0.0, 0.0);
    assert_eq!(h.shaders.reference.rows(ENTRIES).len(), before);
}

#[test]
fn the_toggle_hides_the_column_and_narrows_the_panel() {
    let dir = tempfile::tempdir().unwrap();
    let mut h = session(dir.path());
    let panel = registry::panel(PanelKey::ShaderSource);
    let wide = panel.size(&h);
    assert!(h.shaders.reference.open, "shown by default");
    h.apply_source_action(SourceAction::ToggleReference, 0.0, 0.0);
    assert!(!h.shaders.reference.open);
    assert!(panel.size(&h)[0] < wide[0]);
}

#[test]
fn the_wheel_over_the_column_scrolls_its_rows_not_the_text() {
    let dir = tempfile::tempdir().unwrap();
    let mut h = session(dir.path());
    h.viewport = [1600.0, 900.0];
    let o = h.origin(PanelKey::ShaderSource, h.viewport);
    let s = h.effective_size(PanelKey::ShaderSource);
    let column = shader_source_panel::reference_rect(o, s);
    let (mx, my) = (column[0] + 5.0, column[1] + 5.0);
    let panel = registry::panel(PanelKey::ShaderSource);
    assert!(panel.wheel_over(&h, &World::new(), mx, my, o));
    panel.scroll_at(&mut h, &mut World::new(), -1.0, mx, my);
    assert_eq!(h.shaders.reference.scroll, 1);
    panel.scroll_at(&mut h, &mut World::new(), 1.0, mx, my);
    assert_eq!(h.shaders.reference.scroll, 0);
}
