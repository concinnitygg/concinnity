//! The Shaders panel's edits (`hook/edit/shader_edits.rs`): naming a new
//! Shader, renaming one with every reference following, adding and removing a
//! vertex file, and deleting a Shader with or without its files. Each is one
//! undo step; an open source panel on an affected file closes first.

use concinnity_core::components::ShaderStage;
use concinnity_core::ecs::World;
use std::path::{Path, PathBuf};

use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::{
    hook, press_modal, press_modal_check, set_world_name, world_with_name_field,
};
use crate::editor::panels::shader_list::{MenuItem, RowKind};
use crate::editor::panels::shader_list_panel::ShadersAction;
use crate::editor::panels::shader_source::{self, SourceKey};

fn shader(name: &str, fragment: &Path) -> serde_json::Value {
    serde_json::json!({"type": "Shader", "args": {
        "$id": name,
        "fragment": fragment.to_string_lossy(),
    }})
}

fn material(name: &str, shader: &str) -> serde_json::Value {
    serde_json::json!({"type": "Material", "args": {"$id": name, "shader": shader}})
}

fn key(shader: &str, stage: ShaderStage) -> SourceKey {
    SourceKey {
        shader: shader.to_string(),
        stage,
    }
}

fn write(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, name).unwrap();
    path
}

fn args<'a>(h: &'a EditorHook, id: &str) -> &'a serde_json::Value {
    let entry = h
        .entries
        .iter()
        .find(|e| e["args"]["$id"] == id)
        .unwrap_or_else(|| panic!("no entry '{id}'"));
    &entry["args"]
}

fn shader_names(h: &EditorHook) -> Vec<String> {
    h.entries
        .iter()
        .filter(|e| e["type"] == "Shader")
        .filter_map(|e| e["args"]["$id"].as_str().map(str::to_string))
        .collect()
}

// Click the row standing for `kind`.
fn click(h: &mut EditorHook, world: &mut World, kind: RowKind) {
    let rows = h.shader_rows().to_vec();
    let i = rows.iter().position(|r| r.kind == kind).expect("listed");
    h.apply_shaders_action(ShadersAction::Row(i), &rows, world);
}

// Open the menu on the row standing for `kind` and pick `item`.
fn pick(h: &mut EditorHook, world: &mut World, kind: RowKind, item: MenuItem) {
    let rows = h.shader_rows().to_vec();
    let i = rows.iter().position(|r| r.kind == kind).expect("listed");
    h.apply_shaders_action(ShadersAction::OpenMenu(i), &rows, world);
    assert_eq!(h.shaders.menu.as_ref(), Some(&kind));
    h.apply_shaders_action(ShadersAction::Menu(item), &rows, world);
    assert!(h.shaders.menu.is_none());
}

fn message(h: &EditorHook) -> String {
    h.modal.as_ref().expect("a dialog is open").message.clone()
}

// Run `test` with the working directory at a fresh project whose assets live
// under it, as `cn editor` runs.
fn in_project(test: impl FnOnce(&Path)) {
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

// "+ New Shader" asks for a name; the name, made unique, is the `$id` and the
// file's name, numbered past a file already there. A blank name asks again and
// Cancel creates nothing.
#[test]
fn a_new_shader_is_named_by_its_prompt() {
    in_project(|root| {
        let shaders = root.join("assets/shaders");
        std::fs::create_dir_all(&shaders).unwrap();
        std::fs::write(shaders.join("water.hlsl"), "someone else's").unwrap();
        let mut h = hook(Vec::new());
        let mut world = world_with_name_field();

        click(&mut h, &mut world, RowKind::New);
        press_modal(&mut h, &mut world, "Cancel");
        assert!(h.entries.is_empty() && !h.dirty);

        click(&mut h, &mut world, RowKind::New);
        set_world_name(&mut world, "   ");
        press_modal(&mut h, &mut world, "Create");
        assert!(h.prompting(), "asked again");
        assert!(message(&h).contains("Enter a name"), "{}", message(&h));
        assert!(h.entries.is_empty());

        set_world_name(&mut world, " water ");
        press_modal(&mut h, &mut world, "Create");
        assert_eq!(shader_names(&h), ["water"]);
        assert_eq!(args(&h, "water")["fragment"], "assets/shaders/water_1.hlsl");
        let written = std::fs::read_to_string(shaders.join("water_1.hlsl")).unwrap();
        assert_eq!(written, shader_source::STARTER_SHADER);
        assert_eq!(
            std::fs::read_to_string(shaders.join("water.hlsl")).unwrap(),
            "someone else's"
        );
        assert!(h.dirty, "the new entry is a world edit");
        let src = h.shaders.source.as_ref().unwrap();
        assert_eq!(src.key, key("water", ShaderStage::Fragment));
        assert_eq!(src.area.text(), shader_source::STARTER_SHADER);

        click(&mut h, &mut world, RowKind::New);
        set_world_name(&mut world, "water");
        press_modal(&mut h, &mut world, "Create");
        assert_eq!(shader_names(&h), ["water", "water_1"]);
        assert_eq!(
            args(&h, "water_1")["fragment"],
            "assets/shaders/water_1_1.hlsl"
        );
    });
}

// A rename sets the `$id` and every Material naming the Shader follows, in
// one undo step; its file keeps its name, and a source panel open on it
// follows the name through the rename and its undo.
#[test]
fn a_rename_rewrites_every_reference_in_one_step() {
    let dir = tempfile::tempdir().unwrap();
    let lit = write(dir.path(), "lit.hlsl");
    let water = write(dir.path(), "water.hlsl");
    let mut h = hook(vec![
        shader("lit", &lit),
        shader("water", &water),
        material("sea", "water"),
        material("pond", "water"),
        material("stone", "lit"),
    ]);
    let mut world = world_with_name_field();
    h.open_shader_file(key("water", ShaderStage::Fragment));

    pick(&mut h, &mut world, RowKind::Header(1), MenuItem::Rename);
    assert_eq!(
        crate::editor::widget::field_text(&world, crate::editor::modal::NAME_INPUT),
        "water",
        "seeded with the current name"
    );
    set_world_name(&mut world, "lit");
    press_modal(&mut h, &mut world, "Rename");
    assert_eq!(shader_names(&h), ["lit", "lit_1"], "made unique");
    assert_eq!(args(&h, "sea")["shader"], "lit_1");
    assert_eq!(args(&h, "pond")["shader"], "lit_1");
    assert_eq!(args(&h, "stone")["shader"], "lit");
    assert_eq!(
        args(&h, "lit_1")["fragment"],
        water.to_string_lossy().as_ref()
    );
    assert_eq!(h.shaders.source.as_ref().unwrap().key.shader, "lit_1");

    h.undo(&mut world);
    assert_eq!(shader_names(&h), ["lit", "water"]);
    assert_eq!(args(&h, "sea")["shader"], "water");
    assert_eq!(args(&h, "pond")["shader"], "water");
    assert_eq!(h.shaders.source.as_ref().unwrap().key.shader, "water");
    assert!(!h.can_undo(), "the rename was one step");
}

// "+ Add vertex file" writes a starter vertex file, declares it, and opens it;
// Remove stops declaring it and leaves the file, closing it in the source
// panel first (Cancel on the unsaved-changes question keeps everything).
#[test]
fn a_vertex_file_is_added_and_removed() {
    in_project(|root| {
        let lit = write(root, "lit.hlsl");
        let mut h = hook(vec![shader("lit", &lit)]);
        let mut world = world_with_name_field();

        click(&mut h, &mut world, RowKind::AddVertex(0));
        let declared = args(&h, "lit")["vertex"].as_str().unwrap().to_string();
        assert_eq!(declared, "assets/shaders/lit_vertex.hlsl");
        let path = root.join(&declared);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            shader_source::STARTER_VERTEX
        );
        let vertex = key("lit", ShaderStage::Vertex);
        assert_eq!(h.shaders.source.as_ref().unwrap().key, vertex);
        let rows = h.shader_rows().to_vec();
        assert!(!rows.iter().any(|r| matches!(r.kind, RowKind::AddVertex(_))));

        h.shaders.source.as_mut().unwrap().area.type_char('x');
        pick(
            &mut h,
            &mut world,
            RowKind::File(vertex.clone()),
            MenuItem::RemoveVertex,
        );
        press_modal(&mut h, &mut world, "Cancel");
        assert!(args(&h, "lit").get("vertex").is_some(), "Cancel keeps it");
        assert!(h.shaders.source.is_some());

        pick(
            &mut h,
            &mut world,
            RowKind::File(vertex),
            MenuItem::RemoveVertex,
        );
        press_modal(&mut h, &mut world, "Discard");
        assert!(args(&h, "lit").get("vertex").is_none());
        assert!(h.shaders.source.is_none());
        assert!(path.exists(), "the file stays");

        h.undo(&mut world);
        assert_eq!(args(&h, "lit")["vertex"], declared.as_str());
    });
}

// The dialog says what the delete does; Delete removes the entry and the
// references to it in one step, which undo restores. The files stay unless the
// box is checked.
#[test]
fn a_delete_is_one_step_and_keeps_the_files_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let lit = write(dir.path(), "lit.hlsl");
    let water = write(dir.path(), "water.hlsl");
    let mut h = hook(vec![
        shader("lit", &lit),
        shader("water", &water),
        material("sea", "lit"),
        material("pond", "lit"),
    ]);
    let mut world = world_with_name_field();

    pick(&mut h, &mut world, RowKind::Header(0), MenuItem::Delete);
    let text = message(&h);
    assert!(
        text.contains("2 Materials fall back to the default Shader"),
        "{text}"
    );
    assert!(text.contains("'water' becomes the world default"), "{text}");
    let check = h.modal.as_ref().unwrap().check.clone().unwrap();
    assert!(check.enabled && !check.on);

    press_modal(&mut h, &mut world, "Delete");
    assert_eq!(shader_names(&h), ["water"]);
    assert!(args(&h, "sea").get("shader").is_none(), "falls back");
    assert!(args(&h, "pond").get("shader").is_none());
    assert!(lit.exists(), "unchecked: the file stays");

    h.undo(&mut world);
    assert_eq!(shader_names(&h), ["lit", "water"]);
    assert_eq!(args(&h, "sea")["shader"], "lit");
    assert_eq!(args(&h, "pond")["shader"], "lit");
    assert!(!h.can_undo(), "the delete was one step");
}

// Checked, the delete removes the files only this Shader reads, after the
// entry; a file another Shader reads stays. With every file shared, the box
// cannot be checked.
#[test]
fn checked_deletes_only_the_files_no_other_shader_reads() {
    let dir = tempfile::tempdir().unwrap();
    let common = write(dir.path(), "common.hlsl");
    let sway = write(dir.path(), "sway.hlsl");
    let mut reeds = shader("reeds", &common);
    reeds["args"]["vertex"] = serde_json::json!(sway.to_string_lossy());
    let mut h = hook(vec![shader("lit", &common), reeds]);
    let mut world = world_with_name_field();

    pick(&mut h, &mut world, RowKind::Header(0), MenuItem::Delete);
    press_modal_check(&mut h, &mut world);
    let check = h.modal.as_ref().unwrap().check.clone().unwrap();
    assert!(!check.enabled && !check.on, "every file is shared");
    press_modal(&mut h, &mut world, "Cancel");
    assert_eq!(shader_names(&h), ["lit", "reeds"], "Cancel deletes nothing");

    pick(&mut h, &mut world, RowKind::Header(1), MenuItem::Delete);
    press_modal_check(&mut h, &mut world);
    assert!(h.modal.as_ref().unwrap().check.as_ref().unwrap().on);
    press_modal(&mut h, &mut world, "Delete");
    assert_eq!(shader_names(&h), ["lit"]);
    assert!(!sway.exists(), "its own file is deleted");
    assert!(common.exists(), "the shared file stays");

    h.undo(&mut world);
    assert_eq!(
        shader_names(&h),
        ["lit", "reeds"],
        "undo restores the entry"
    );
    assert!(!sway.exists(), "but not the file");
}

// A delete of the Shader the source panel shows closes it first, asking over
// unsaved edits; Cancel there abandons the delete.
#[test]
fn a_delete_closes_the_open_file_first() {
    let dir = tempfile::tempdir().unwrap();
    let lit = write(dir.path(), "lit.hlsl");
    let water = write(dir.path(), "water.hlsl");
    let mut h = hook(vec![shader("lit", &lit), shader("water", &water)]);
    let mut world = world_with_name_field();
    h.open_shader_file(key("water", ShaderStage::Fragment));
    h.shaders.source.as_mut().unwrap().area.type_char('x');

    pick(&mut h, &mut world, RowKind::Header(1), MenuItem::Delete);
    press_modal(&mut h, &mut world, "Delete");
    assert!(message(&h).contains("unsaved changes"), "{}", message(&h));
    press_modal(&mut h, &mut world, "Cancel");
    assert_eq!(shader_names(&h), ["lit", "water"]);
    assert!(h.shaders.source.is_some());

    pick(&mut h, &mut world, RowKind::Header(1), MenuItem::Delete);
    press_modal(&mut h, &mut world, "Delete");
    press_modal(&mut h, &mut world, "Save");
    assert_eq!(shader_names(&h), ["lit"]);
    assert!(h.shaders.source.is_none());
    assert_eq!(std::fs::read_to_string(&water).unwrap(), "xwater.hlsl");

    // A panel on another Shader is left open.
    h.open_shader_file(key("lit", ShaderStage::Fragment));
    h.shaders.source.as_mut().unwrap().area.type_char('y');
    h.delete_shader("nothing", false);
    assert!(h.modal.is_none() && h.shaders.source.is_some());
}
