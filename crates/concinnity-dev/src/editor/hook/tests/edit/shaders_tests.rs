//! The Shaders and Shader source panels' actions (`hook/edit/shaders.rs`,
//! `hook/edit/shader_source.rs`): opening a file, the question leaving unsaved
//! edits asks and each of its answers, the close button asking the same, a
//! save writing the file and waiting on its recompile, and the Shader limit.
//! The panel's edits are `shader_edits_tests.rs`, its form
//! `shader_form_tests.rs`.

use concinnity_core::components::ShaderStage;
use concinnity_core::ecs::World;
use concinnity_core::gfx::render_types::MAX_SHADER_BUCKETS;
use std::cell::Cell;
use std::path::Path;

use crate::debug::hot_reload::{ShaderReloadOutcome, ShaderReloadReport};
use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::hook;
use crate::editor::modal;
use crate::editor::panels::registry::{self, PanelKey};
use crate::editor::panels::shader_list::RowKind;
use crate::editor::panels::shader_list_panel::ShadersAction;
use crate::editor::panels::shader_source::{self, Leave, SourceKey};

fn shader(name: &str, fragment: &Path) -> serde_json::Value {
    serde_json::json!({"type": "Shader", "args": {
        "$id": name,
        "fragment": fragment.to_string_lossy(),
    }})
}

fn key(shader: &str) -> SourceKey {
    SourceKey {
        shader: shader.to_string(),
        stage: ShaderStage::Fragment,
    }
}

// A hook over two Shaders whose fragment files hold "lit" and "water".
fn session(dir: &Path) -> EditorHook {
    let lit = dir.join("lit.hlsl");
    let water = dir.join("water.hlsl");
    std::fs::write(&lit, "lit").unwrap();
    std::fs::write(&water, "water").unwrap();
    hook(vec![shader("lit", &lit), shader("water", &water)])
}

fn open_shader(h: &EditorHook) -> Option<String> {
    h.shaders.source.as_ref().map(|s| s.key.shader.clone())
}

fn modal_actions(h: &EditorHook) -> Vec<modal::Action> {
    h.modal
        .as_ref()
        .map(|m| m.buttons.iter().map(|b| b.action.clone()).collect())
        .unwrap_or_default()
}

// A file row opens the source panel on that file, frontmost and ready to type.
#[test]
fn a_file_row_opens_the_source_panel() {
    let dir = tempfile::tempdir().unwrap();
    let mut h = session(dir.path());
    let mut world = World::new();
    let rows = h.shader_rows().to_vec();
    let i = rows
        .iter()
        .position(|r| r.kind == RowKind::File(key("water")))
        .unwrap();
    h.apply_shaders_action(ShadersAction::Row(i), &rows, &mut world);
    let src = h.shaders.source.as_ref().unwrap();
    assert_eq!(src.area.text(), "water");
    assert_eq!(src.title(), "water fragment");
    assert!(src.focus);
    assert_eq!(h.panel_order.last(), Some(&PanelKey::ShaderSource));
    assert!(registry::panel(PanelKey::ShaderSource).is_open(&h));
}

// Leaving unsaved edits asks; Cancel stays, Discard leaves them unwritten, and
// Save writes them and then leaves.
#[test]
fn leaving_unsaved_edits_asks_first() {
    let dir = tempfile::tempdir().unwrap();
    let mut h = session(dir.path());
    h.open_shader_file(key("lit"));
    h.shaders.source.as_mut().unwrap().area.type_char('x');

    h.open_shader_file(key("water"));
    assert_eq!(open_shader(&h).as_deref(), Some("lit"), "not left yet");
    let actions = modal_actions(&h);
    assert_eq!(actions.len(), 3);
    assert_eq!(actions[0], modal::Action::Dismiss);

    // Discard: water opens and lit's file is untouched.
    h.modal = None;
    h.answer_leave_shader_source(false, Leave::Open(key("water")), &mut World::new());
    assert_eq!(open_shader(&h).as_deref(), Some("water"));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("lit.hlsl")).unwrap(),
        "lit"
    );

    // Save: the edits are written, then lit opens.
    h.shaders.source.as_mut().unwrap().area.type_char('y');
    h.answer_leave_shader_source(true, Leave::Open(key("lit")), &mut World::new());
    assert_eq!(
        std::fs::read_to_string(dir.path().join("water.hlsl")).unwrap(),
        "ywater"
    );
    assert_eq!(open_shader(&h).as_deref(), Some("lit"));
}

// Reopening the open file keeps its edits and asks nothing.
#[test]
fn reopening_the_open_file_keeps_its_edits() {
    let dir = tempfile::tempdir().unwrap();
    let mut h = session(dir.path());
    h.open_shader_file(key("lit"));
    h.shaders.source.as_mut().unwrap().area.type_char('x');
    h.open_shader_file(key("lit"));
    assert!(h.modal.is_none());
    assert_eq!(h.shaders.source.as_ref().unwrap().area.text(), "xlit");
}

// The close button asks over unsaved edits and closes a clean file at once.
#[test]
fn closing_asks_only_over_unsaved_edits() {
    let dir = tempfile::tempdir().unwrap();
    let mut h = session(dir.path());
    let mut world = World::new();
    let panel = registry::panel(PanelKey::ShaderSource);
    h.open_shader_file(key("lit"));
    h.shaders.source.as_mut().unwrap().area.type_char('x');
    panel.close(&mut h, &mut world);
    assert!(panel.is_open(&h));
    assert_eq!(
        modal_actions(&h)[1],
        modal::Action::LeaveShaderSource {
            save: false,
            then: Leave::Close
        }
    );
    h.modal = None;
    h.answer_leave_shader_source(false, Leave::Close, &mut world);
    assert!(!panel.is_open(&h));

    h.open_shader_file(key("lit"));
    panel.close(&mut h, &mut world);
    assert!(!panel.is_open(&h));
    assert!(h.modal.is_none());
}

// A save writes the file and waits on the recompile; the report that answers
// it marks the line and moves the caret there.
#[test]
fn a_save_waits_on_its_recompile() {
    let dir = tempfile::tempdir().unwrap();
    let mut h = session(dir.path());
    h.shaders
        .reports
        .arm(["lit".to_string(), "water".to_string()]);
    h.open_shader_file(key("lit"));
    h.drive_shader_source();
    h.shaders.source.as_mut().unwrap().area.type_char('x');
    assert!(h.save_shader_source());
    let src = h.shaders.source.as_ref().unwrap();
    assert!(!src.area.is_dirty());
    assert!(src.compiling);
    assert_eq!(src.status.as_ref().unwrap().text, "Compiling...");

    let path = src.path.clone();
    h.shaders.reports.publish(&[ShaderReloadReport {
        name: "lit".to_string(),
        outcome: failed_at(&path, 1, 3),
    }]);
    h.drive_shader_source();
    let src = h.shaders.source.as_ref().unwrap();
    assert!(!src.compiling);
    assert_eq!(src.markers.len(), 1);
    assert_eq!(src.area.caret().col, 2, "the caret jumped to the error");
}

fn failed_at(path: &str, line: u32, column: u32) -> ShaderReloadOutcome {
    use crate::debug::hot_reload::ShaderReloadFailure;
    use concinnity_cook::compile::program::{CompileFailure, Diagnostic, Severity};
    ShaderReloadOutcome::Failed(ShaderReloadFailure::Compile(CompileFailure {
        owner: "Shader 'lit'".to_string(),
        failures: Vec::new(),
        diagnostics: vec![Diagnostic {
            path: path.to_string(),
            line,
            column,
            severity: Severity::Error,
            message: "bad".to_string(),
            context: String::new(),
        }],
        hint: "",
    }))
}

thread_local! {
    static RESOLVED: Cell<usize> = const { Cell::new(0) };
}

fn counting_resolve(declared: &str, dir: Option<&Path>) -> String {
    RESOLVED.with(|n| n.set(n.get() + 1));
    shader_source::resolve_path(declared, dir)
}

// The list resolves each declared file once, however many frames ask for its
// rows, while a new report still reaches the row it concerns.
#[test]
fn the_list_resolves_each_file_once() {
    let _guard = crate::test_support::lock();
    let dir = tempfile::tempdir().unwrap();
    let mut h = session(dir.path());
    h.shaders.resolve = counting_resolve;
    RESOLVED.with(|n| n.set(0));
    h.shaders
        .reports
        .arm(["lit".to_string(), "water".to_string()]);
    for _ in 0..5 {
        h.shader_rows();
    }
    assert_eq!(RESOLVED.with(Cell::get), 2, "one per declared file");

    let lit = dir.path().join("lit.hlsl").to_string_lossy().into_owned();
    h.shaders.reports.publish(&[ShaderReloadReport {
        name: "lit".to_string(),
        outcome: failed_at(&lit, 1, 1),
    }]);
    let badge = h
        .shader_rows()
        .iter()
        .find(|r| r.kind == RowKind::File(key("lit")))
        .and_then(|r| r.badge.clone())
        .unwrap();
    assert_eq!(badge.0, "failed");
    assert_eq!(RESOLVED.with(Cell::get), 2);
}

// A world at the Shader limit lists "+ New Shader" unclickable; the form's
// own refusal at the limit is `shader_form_tests.rs`.
#[test]
fn new_shader_is_unavailable_at_the_limit() {
    let entries: Vec<serde_json::Value> = (0..MAX_SHADER_BUCKETS)
        .map(|i| shader(&format!("s{i}"), Path::new("/cn-none/s.hlsl")))
        .collect();
    let mut h = hook(entries);
    let last = h.shader_rows().last().cloned().unwrap();
    assert_eq!(last.kind, RowKind::Note);
    assert!(!last.clickable());
    assert!(!h.form_open());
}
