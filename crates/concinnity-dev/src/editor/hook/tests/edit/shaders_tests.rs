//! The Shaders and Shader source panels' actions (`hook/edit/shaders.rs`,
//! `hook/edit/shader_source.rs`): opening a file, the question leaving unsaved
//! edits asks and each of its answers, the close button asking the same, a
//! save writing the file and waiting on its recompile.
//! The panel's edits are `shader_edits_tests.rs`, its form
//! `shader_form_tests.rs`.

use concinnity_engine::components::ShaderStage;
use concinnity_engine::ecs::World;
use std::path::Path;

use crate::debug::hot_reload::{ReloadOutcome, ReloadReport, ReloadSubject};
use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::hook;
use crate::editor::modal;
use crate::editor::panels::registry::{self, PanelKey};
use crate::editor::panels::shader;
use crate::editor::panels::shader::list::RowKind;
use crate::editor::panels::shader::list_panel::ShadersAction;
use crate::editor::panels::shader::source::{Leave, SourceKey};
use crate::frame_hook::FrameHook;

fn shader(name: &str, fragment: &Path) -> serde_json::Value {
    serde_json::json!({"type": "Shader", "args": {
        "$id": name,
        "fragment": fragment.to_string_lossy(),
    }})
}

fn key(shader: &str) -> SourceKey {
    SourceKey::shader(shader, ShaderStage::Fragment)
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
    h.shaders.source.as_ref().and_then(|s| match &s.key {
        SourceKey::Shader { name, .. } => Some(name.clone()),
        SourceKey::Field { .. } => None,
    })
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
    let world = World::new();
    let rows = h.shader_rows().to_vec();
    let i = rows
        .iter()
        .position(|r| r.kind == RowKind::File(key("water")))
        .unwrap();
    h.apply_shaders_action(ShadersAction::Row(i), &world);
    let src = h.shaders.source.as_ref().unwrap();
    assert_eq!(src.area.text(), "water");
    assert_eq!(src.title_parts(), ["water", "fragment"]);
    assert!(src.focus);
    assert_eq!(h.panel_order.last(), Some(&PanelKey::ShaderSource));
    assert!(registry::is_open(PanelKey::ShaderSource, &h));
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
    h.answer_leave_shader_source(false, Leave::Open(key("water")));
    assert_eq!(open_shader(&h).as_deref(), Some("water"));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("lit.hlsl")).unwrap(),
        "lit"
    );

    // Save: the edits are written, then lit opens.
    h.shaders.source.as_mut().unwrap().area.type_char('y');
    h.answer_leave_shader_source(true, Leave::Open(key("lit")));
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
    let source = PanelKey::ShaderSource;
    h.open_shader_file(key("lit"));
    h.shaders.source.as_mut().unwrap().area.type_char('x');
    registry::close(source, &mut h, &mut world);
    assert!(registry::is_open(source, &h));
    assert_eq!(
        modal_actions(&h)[1],
        modal::Action::LeaveShaderSource {
            save: false,
            then: Leave::Close
        }
    );
    h.modal = None;
    h.answer_leave_shader_source(false, Leave::Close);
    assert!(!registry::is_open(source, &h));

    h.open_shader_file(key("lit"));
    registry::close(source, &mut h, &mut world);
    assert!(!registry::is_open(source, &h));
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
        .arm([ReloadSubject::shader("lit"), ReloadSubject::shader("water")]);
    h.open_shader_file(key("lit"));
    h.drive_shader_source();
    h.shaders.source.as_mut().unwrap().area.type_char('x');
    assert!(h.save_shader_source());
    let src = h.shaders.source.as_ref().unwrap();
    assert!(!src.area.is_dirty());
    assert!(src.compiling);
    assert_eq!(src.status.as_ref().unwrap().text, "Compiling...");

    let path = src.path.clone();
    h.shaders.reports.publish(&[ReloadReport {
        subject: ReloadSubject::shader("lit"),
        outcome: failed_at(&path, 1, 3),
    }]);
    h.drive_shader_source();
    let src = h.shaders.source.as_ref().unwrap();
    assert!(!src.compiling);
    assert_eq!(src.markers.len(), 1);
    assert_eq!(src.area.caret().col, 2, "the caret jumped to the error");
}

fn failed_at(path: &str, line: u32, column: u32) -> ReloadOutcome {
    use crate::debug::hot_reload::ReloadFailure;
    use concinnity_cook::compile::program::{CompileFailure, Diagnostic, Severity};
    ReloadOutcome::Failed(ReloadFailure::Compile(CompileFailure {
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

// Tick the hook over `world` on a plain frame, with the source panel's disk
// check due.
fn tick_checking_disk(h: &mut EditorHook, world: &mut World) {
    if let Some(src) = h.shaders.source.as_mut() {
        src.next_check = 0.0;
        src.mtime = None;
    }
    h.tick(world);
}

fn shown_source(dir: &Path) -> (EditorHook, World) {
    let mut world = World::new();
    crate::editor::install::editor_ui(&mut world);
    world.add_component(concinnity_engine::components::FrameInput {
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    let mut h = session(dir);
    h.open_shader_file(key("lit"));
    h.tick(&mut world);
    h.tick(&mut world);
    assert!(h.lent[PanelKey::ShaderSource], "the editor is out on loan");
    (h, world)
}

// A clean file changed on disk while the panel's editor is lent out is
// reloaded into the editor the hook takes back.
#[test]
fn a_clean_file_changed_on_disk_reloads_while_shown() {
    let dir = tempfile::tempdir().unwrap();
    let (mut h, mut world) = shown_source(dir.path());
    std::fs::write(dir.path().join("lit.hlsl"), "lit v2").unwrap();
    tick_checking_disk(&mut h, &mut world);
    h.reclaim_built(&mut world);
    assert_eq!(h.shaders.source.as_ref().unwrap().area.text(), "lit v2");
}

// Unsaved edits are kept over a change on disk, with a notice saying so.
#[test]
fn a_dirty_buffer_notices_a_change_on_disk() {
    let dir = tempfile::tempdir().unwrap();
    let (mut h, mut world) = shown_source(dir.path());
    h.reclaim_built(&mut world);
    h.shaders.source.as_mut().unwrap().area.type_char('x');
    std::fs::write(dir.path().join("lit.hlsl"), "lit v2").unwrap();
    tick_checking_disk(&mut h, &mut world);
    h.reclaim_built(&mut world);
    let src = h.shaders.source.as_ref().unwrap();
    assert_eq!(src.area.text(), "xlit");
    assert_eq!(
        src.status.as_ref().unwrap().tone,
        shader::diagnostics::Tone::Warning
    );
}

// A context menu closes once the Shaders it numbers change, so a pick cannot
// land on another Shader.
#[test]
fn a_menu_closes_when_the_shaders_change() {
    let dir = tempfile::tempdir().unwrap();
    let mut h = session(dir.path());
    let world = World::new();
    let rows = h.shader_rows().to_vec();
    let i = rows
        .iter()
        .position(|r| matches!(r.kind, RowKind::Header(1)))
        .unwrap();
    h.apply_shaders_action(ShadersAction::Context(i), &world);
    assert!(h.shaders.menu.is_some());
    h.entries.remove(0);
    h.apply_shaders_action(ShadersAction::Menu(2), &world);
    assert!(h.shaders.menu.is_none());
    assert_eq!(h.entries.len(), 1, "the pick deleted nothing");
}

// Showing an unchanged file again allocates nothing, with the reference
// column open and a marker in the gutter.
#[test]
fn a_steady_source_sync_allocates_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (mut h, mut world) = shown_source(dir.path());
    h.reclaim_built(&mut world);
    h.shaders.reference.open = true;
    let path = h.shaders.source.as_ref().unwrap().path.clone();
    h.shaders.reports.publish(&[ReloadReport {
        subject: ReloadSubject::shader("lit"),
        outcome: failed_at(&path, 1, 1),
    }]);
    h.drive_shader_source();
    assert_eq!(h.shaders.source.as_ref().unwrap().markers.len(), 1);
    let panel = world
        .resource::<crate::editor::ui::BuiltUi>()
        .and_then(|b| b.panel(PanelKey::ShaderSource))
        .cloned()
        .unwrap();
    h.sync_shader_source(&mut world, &panel);
    let mut min = u64::MAX;
    for _ in 0..16 {
        let before = concinnity_core::memory::alloc_count().unwrap();
        h.sync_shader_source(&mut world, &panel);
        let after = concinnity_core::memory::alloc_count().unwrap();
        min = min.min(after - before);
    }
    assert_eq!(min, 0, "a steady sync allocated {min} times");
}
