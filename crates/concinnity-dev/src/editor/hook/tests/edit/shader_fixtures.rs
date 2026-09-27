//! What the Shaders panel's test companions share: entry literals, a hook's
//! Shader names and args, clicks on the list and on a form's extra rows, and a
//! working directory at a fresh project for the edits that write files under
//! its assets.

use concinnity_core::components::ShaderStage;
use concinnity_core::ecs::World;
use std::path::{Path, PathBuf};

use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::{set_field, world_with_fields};
use crate::editor::modal;
use crate::editor::panels::form_extras::{ExtraControl, ExtraRow};
use crate::editor::panels::form_panel::{self, FormAction};
use crate::editor::panels::shader_list::{MenuItem, RowKind};
use crate::editor::panels::shader_list_panel::ShadersAction;
use crate::editor::panels::shader_source::SourceKey;

pub(super) fn shader(name: &str, fragment: &Path) -> serde_json::Value {
    serde_json::json!({"type": "Shader", "args": {
        "$id": name,
        "fragment": fragment.to_string_lossy(),
    }})
}

pub(super) fn material(name: &str, shader: &str) -> serde_json::Value {
    serde_json::json!({"type": "Material", "args": {"$id": name, "shader": shader}})
}

pub(super) fn key(shader: &str, stage: ShaderStage) -> SourceKey {
    SourceKey::shader(shader, stage)
}

pub(super) fn write(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, name).unwrap();
    path
}

pub(super) fn args<'a>(h: &'a EditorHook, id: &str) -> &'a serde_json::Value {
    let entry = h
        .entries
        .iter()
        .find(|e| e["args"]["$id"] == id)
        .unwrap_or_else(|| panic!("no entry '{id}'"));
    &entry["args"]
}

pub(super) fn shader_names(h: &EditorHook) -> Vec<String> {
    h.entries
        .iter()
        .filter(|e| e["type"] == "Shader")
        .filter_map(|e| e["args"]["$id"].as_str().map(str::to_string))
        .collect()
}

// A world with the form's fields and the dialog's, so a leave question can be
// answered while the form is open.
pub(super) fn form_world() -> World {
    let mut world = world_with_fields();
    for id in modal::all_field_ids() {
        world.push_identified(id, concinnity_core::components::TextInput::default());
    }
    world
}

pub(super) fn plain_material(name: &str) -> serde_json::Value {
    serde_json::json!({"type": "Material", "args": {"$id": name}})
}

pub(super) fn type_name(world: &mut World, name: &str) {
    set_field(world, form_panel::NAME_INPUT, name);
}

pub(super) fn rows(h: &EditorHook, world: &World) -> Vec<ExtraRow> {
    h.form_extras_data(world).0
}

pub(super) fn blocked(h: &EditorHook, world: &World) -> Option<String> {
    h.form_extras_data(world).1
}

// Press the row captioned `caption` that is a checkbox (or, with `choice`, a
// choice).
pub(super) fn press(h: &mut EditorHook, world: &mut World, caption: &str, choice: bool) {
    let row = rows(h, world)
        .into_iter()
        .find(|r| {
            r.caption == caption && matches!(r.control, ExtraControl::Choice { .. }) == choice
        })
        .unwrap_or_else(|| panic!("no '{caption}' row"));
    h.apply_form(FormAction::PressExtra(row.id), world);
}

pub(super) fn detail(h: &EditorHook, world: &World, caption: &str) -> Option<String> {
    rows(h, world)
        .into_iter()
        .find(|r| r.caption == caption)
        .and_then(|r| r.detail)
}

pub(super) fn confirm(h: &mut EditorHook, world: &mut World) {
    h.apply_form(FormAction::Confirm, world);
}

pub(super) fn toasts(h: &EditorHook) -> Vec<String> {
    h.notifier
        .stack()
        .cards
        .into_iter()
        .map(|c| c.message)
        .collect()
}

// Click the row standing for `kind`.
pub(super) fn click(h: &mut EditorHook, world: &mut World, kind: RowKind) {
    let rows = h.shader_rows().to_vec();
    let i = rows.iter().position(|r| r.kind == kind).expect("listed");
    h.apply_shaders_action(ShadersAction::Row(i), &rows, world);
}

// Open the menu on the row standing for `kind` and pick `item`.
pub(super) fn pick(h: &mut EditorHook, world: &mut World, kind: RowKind, item: MenuItem) {
    let rows = h.shader_rows().to_vec();
    let i = rows.iter().position(|r| r.kind == kind).expect("listed");
    h.apply_shaders_action(ShadersAction::OpenMenu(i), &rows, world);
    assert_eq!(h.shaders.menu.as_ref(), Some(&kind));
    h.apply_shaders_action(ShadersAction::Menu(item), &rows, world);
    assert!(h.shaders.menu.is_none());
}

pub(super) fn message(h: &EditorHook) -> String {
    h.modal.as_ref().expect("a dialog is open").message.clone()
}

// Run `test` with the working directory at a fresh project whose assets live
// under it, as `cn editor` runs.
pub(super) fn in_project(test: impl FnOnce(&Path)) {
    let _guard = crate::test_support::lock();
    let tree = concinnity_testing::TempTree::new();
    let old = std::env::current_dir().unwrap();
    std::env::set_current_dir(tree.path()).unwrap();
    let root = std::env::current_dir().unwrap();
    crate::project::open(concinnity_host::store::paths::StateTree::at(&root));
    test(&root);
    std::env::set_current_dir(old).unwrap();
    crate::test_support::isolate_state_dir();
}
