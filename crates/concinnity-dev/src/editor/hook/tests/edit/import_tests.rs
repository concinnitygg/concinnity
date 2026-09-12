// src/editor/hook/tests/edit/import_tests.rs
//
// The Import panel's actions (`hook/edit/import.rs`): the entry a resolved
// file adds, the name collision it works around, what it refuses, the
// environment map an HDR resolves to and the retarget of an existing one, and
// the relative path the native browse fills in.

use concinnity_core::components::InputKey;
use concinnity_core::ecs::World;

use crate::editor::hook::tests::fixtures::{entry, hook, story_key_input};
use crate::editor::hook::{EditorHook, FormTarget};

use crate::editor::inject;

use crate::editor::panels::import_panel::{self, ImportAction};

use crate::editor::panels::registry::PanelKey;
use crate::editor::panels::story;

use crate::editor::widget;

// Import panel

fn import_session() -> (EditorHook, World, concinnity_testing::TempTree) {
    let mut world = World::new();
    inject::editor_hud(&mut world);
    let mut h = hook(Vec::new());
    h.import_open = true;
    h.import_focus = true;
    h.focus_panel(PanelKey::Import);
    (h, world, concinnity_testing::TempTree::new())
}

fn type_path(world: &mut World, path: &str) {
    widget::seed_field(world, import_panel::PATH_INPUT, path);
}

// A scene file resolves through the same dispatch `cn add` uses: one
// SceneImport entry, the world marked changed, and the field cleared.
#[test]
fn import_add_resolves_a_scene_file() {
    let (mut h, mut world, dir) = import_session();
    let glb = dir.join("crate_stack.glb");
    std::fs::write(&glb, b"glb bytes").unwrap();
    type_path(&mut world, &glb.to_string_lossy());
    h.add_import(&mut world);
    assert_eq!(h.import_status, None);
    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["type"], "SceneImport");
    assert_eq!(h.entries[0]["name"], "crate_stack");
    assert_eq!(
        h.entries[0]["args"]["source"],
        serde_json::Value::String(glb.to_string_lossy().to_string())
    );
    assert!(h.dirty && h.rebuild_preview);
    assert_eq!(
        widget::field_text(&world, import_panel::PATH_INPUT),
        "",
        "the path field clears on success"
    );
}

// A story markdown resolves to a StoryImport; a colliding name is uniquified
// instead of erroring.
#[test]
fn import_add_uniquifies_a_colliding_name() {
    let (mut h, mut world, dir) = import_session();
    let md = dir.join("tale.md");
    std::fs::write(&md, story::STARTER_STORY).unwrap();
    h.entries.push(entry("tale", "PointLight"));
    type_path(&mut world, &md.to_string_lossy());
    h.add_import(&mut world);
    assert_eq!(h.import_status, None);
    assert_eq!(h.entries.len(), 2);
    assert_eq!(h.entries[1]["type"], "StoryImport");
    assert_eq!(h.entries[1]["name"], "tale_1", "renamed past the collision");
}

// Failures land on the status line and commit nothing: a missing file, and an
// unknown extension.
#[test]
fn import_add_rejects_missing_files_and_unknown_extensions() {
    let (mut h, mut world, dir) = import_session();
    type_path(&mut world, "/no/such/thing.glb");
    h.add_import(&mut world);
    assert!(
        h.import_status
            .as_ref()
            .unwrap()
            .text()
            .contains("no such file")
    );
    assert!(h.entries.is_empty() && !h.dirty);

    let odd = dir.join("mystery.xyz");
    std::fs::write(&odd, b"?").unwrap();
    type_path(&mut world, &odd.to_string_lossy());
    h.add_import(&mut world);
    assert!(h.import_status.is_some(), "unknown extension rejected");
    assert!(h.entries.is_empty() && !h.dirty);
}

// A `.hdr` into a world with no lighting environment adds one.
#[test]
fn import_add_resolves_an_hdr_to_an_environment_map() {
    let (mut h, mut world, dir) = import_session();
    let hdr = dir.join("studio.hdr");
    std::fs::write(&hdr, b"radiance").unwrap();
    type_path(&mut world, &hdr.to_string_lossy());
    h.add_import(&mut world);
    assert_eq!(h.import_status, None);
    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["type"], "EnvironmentMap");
    assert_eq!(h.entries[0]["name"], "studio");
    assert_eq!(
        h.entries[0]["args"]["source"],
        serde_json::Value::String(hdr.to_string_lossy().to_string())
    );
}

// A second `.hdr` retargets the world's existing map instead of appending one
// the runtime would ignore, and says so on the status line.
#[test]
fn import_add_retargets_an_existing_environment_map() {
    let (mut h, mut world, dir) = import_session();
    h.entries.push(serde_json::json!({
        "name": "env", "type": "EnvironmentMap", "args": {"source": "", "generator": "sky"}
    }));
    let hdr = dir.join("dusk.hdr");
    std::fs::write(&hdr, b"radiance").unwrap();
    type_path(&mut world, &hdr.to_string_lossy());
    h.add_import(&mut world);

    assert_eq!(h.entries.len(), 1, "no second map appended");
    assert_eq!(h.entries[0]["name"], "env", "the existing map is reused");
    assert_eq!(
        h.entries[0]["args"]["source"],
        serde_json::Value::String(hdr.to_string_lossy().to_string())
    );
    assert_eq!(h.entries[0]["args"]["generator"], "");
    assert!(h.dirty && h.rebuild_preview);
    // Reported as a notice, not an error: the Add succeeded.
    assert!(matches!(
        h.import_status,
        Some(import_panel::ImportStatus::Notice(_))
    ));
    assert_eq!(
        widget::field_text(&world, import_panel::PATH_INPUT),
        "",
        "the path field clears on success"
    );
}

// Enter in the focused path field adds, like clicking the Add button.
#[test]
fn import_enter_key_adds() {
    let (mut h, mut world, dir) = import_session();
    let glb = dir.join("prop.glb");
    std::fs::write(&glb, b"glb").unwrap();
    type_path(&mut world, &glb.to_string_lossy());
    h.import_keys(&mut world, &story_key_input(InputKey::Enter));
    assert_eq!(h.entries.len(), 1);
}

// The list shows the world's file-backed entries and opens one in the
// standard edit form (alongside the Assets browse panel).
#[test]
fn import_rows_list_and_open_in_the_edit_form() {
    let (mut h, mut world, _dir) = import_session();
    h.entries = vec![
        entry("lamp", "PointLight"),
        serde_json::json!({"name": "town", "type": "SceneImport", "args": {"source": "town.glb"}}),
        serde_json::json!({"name": "face", "type": "Font", "args": {"path": "face.ttf"}}),
        serde_json::json!({"name": "env", "type": "EnvironmentMap", "args": {"source": "sky.hdr"}}),
    ];
    let rows = h.import_rows();
    assert_eq!(rows.len(), 3, "only file-backed types list");
    assert_eq!(rows[0].entry, 1);
    assert_eq!(rows[0].caption, "town  (SceneImport)  town.glb");
    assert_eq!(rows[1].caption, "face  (Font)  face.ttf");
    assert_eq!(rows[2].caption, "env  (EnvironmentMap)  sky.hdr");

    h.apply_import_action(ImportAction::Open(0), &mut world);
    assert!(h.panel_open, "the Assets UI comes up with the form");
    assert!(h.form_open());
    assert_eq!(
        h.form_target,
        FormTarget::Entry(1),
        "the clicked entry is being edited"
    );
}

// A browsed file lands in the path field as a project-relative path, ready for
// the user to confirm with Add (Browse never commits on its own). The dialog
// itself is not exercised: `browse_import` is a three-line wrapper over it, and
// everything the pick feeds runs through `accept_browsed_path`.
#[test]
fn import_browse_result_fills_the_path_field_relatively() {
    let _guard = crate::test_support::lock();
    let (mut h, mut world, dir) = import_session();
    let old = std::env::current_dir().unwrap();
    std::env::set_current_dir(dir.path()).unwrap();

    let assets = dir.dir("assets");
    let picked = assets.join("hero.glb");
    std::fs::write(&picked, b"glb").unwrap();
    h.import_status = Some(import_panel::ImportStatus::Error("stale error".to_string()));
    h.accept_browsed_path(&mut world, &picked);

    assert_eq!(
        widget::field_text(&world, import_panel::PATH_INPUT),
        "assets/hero.glb",
        "a file inside the project stores relatively"
    );
    assert!(h.import_focus, "the field takes focus, ready to Add");
    assert_eq!(h.import_status, None, "a stale error is cleared");
    assert!(h.entries.is_empty(), "Browse does not commit on its own");

    // Confirming with Add resolves the browsed path like any typed one.
    h.add_import(&mut world);
    assert_eq!(h.import_status, None);
    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["type"], "SceneImport");
    assert_eq!(h.entries[0]["args"]["source"], "assets/hero.glb");

    std::env::set_current_dir(old).unwrap();
}
