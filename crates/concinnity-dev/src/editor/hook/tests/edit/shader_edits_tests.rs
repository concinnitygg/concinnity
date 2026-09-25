//! The Shaders panel's edits beyond its form (`hook/edit/shader_edits.rs`):
//! adding and removing a vertex file from the list, duplicating a Shader with
//! its files, and deleting a Shader with or without its files. Each is one
//! undo step; an open source panel on an affected file closes first.

use concinnity_core::components::ShaderStage;

use super::shader_fixtures::{
    args, click, in_project, key, material, message, pick, shader, shader_names, write,
};
use crate::editor::hook::tests::fixtures::{
    hook, press_modal, press_modal_check, world_with_name_field,
};
use crate::editor::panels::shader_list::{MenuItem, RowKind};
use crate::editor::panels::shader_templates;

// A heading's Duplicate copies the Shader under a fresh name with its own
// copies of its files, in one undo step that leaves the copies on disk.
#[test]
fn a_duplicate_owns_copies_of_its_files() {
    let dir = tempfile::tempdir().unwrap();
    let water = write(dir.path(), "water.hlsl");
    let sway = write(dir.path(), "sway.hlsl");
    let mut entry = shader("water", &water);
    entry["args"]["vertex"] = serde_json::json!(sway.to_string_lossy());
    let mut h = hook(vec![entry, material("sea", "water")]);
    let mut world = world_with_name_field();

    pick(&mut h, &mut world, RowKind::Header(0), MenuItem::Duplicate);
    assert_eq!(shader_names(&h), ["water", "water_1"]);
    let copy = dir.path().join("water_copy.hlsl");
    assert_eq!(
        args(&h, "water_1")["fragment"],
        copy.to_string_lossy().as_ref()
    );
    assert_eq!(std::fs::read_to_string(&copy).unwrap(), "water.hlsl");
    assert!(dir.path().join("sway_copy.hlsl").exists());
    assert_eq!(
        args(&h, "water")["fragment"],
        water.to_string_lossy().as_ref()
    );
    assert_eq!(args(&h, "sea")["shader"], "water", "references stay");
    let toast = h.notifier.stack();
    assert!(
        toast
            .cards
            .iter()
            .any(|c| c.message.contains("water.hlsl to water_copy.hlsl")),
        "{toast:?}"
    );

    h.undo(&mut world);
    assert_eq!(shader_names(&h), ["water"]);
    assert!(copy.exists(), "the copy stays");
    assert!(!h.can_undo(), "one step");
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
            shader_templates::VERTEX[0].text
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
