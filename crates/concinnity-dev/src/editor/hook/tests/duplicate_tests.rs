//! Duplicating the selection (`hook/duplicate.rs`): the entries a clone appends
//! and selects, and the Ctrl+D that stands down while the Behavior panel owns
//! the keyboard.

use concinnity_core::components::FrameInput;
use concinnity_core::components::InputKey;
use concinnity_core::components::KeyEvent;
use concinnity_core::components::KeyMods;
use concinnity_core::components::KeyPress;
use concinnity_core::components::TextInput;
use concinnity_core::ecs::World;

use super::fixtures::{entry, hook, world_with_input};
use crate::debug_hook::DebugHook;
use crate::editor::behavior;
use crate::editor::hook::declared_id;

use crate::editor::hook::tests::fixtures::{select, selected};
use crate::editor::panels::registry::{self, PanelKey};

// Duplicating the selection clones each authored entry (args included) under
// a unique name, skips singletons, selects the copies, and is one undo step.
#[test]
fn duplicate_selection_clones_entries_and_selects_the_copies() {
    let mut world = World::new();
    let mut h = hook(vec![
        serde_json::json!({
            "type": "Prop", "args": { "$id": "box", "position": [1.0, 2.0, 3.0] }
        }),
        entry("phys", "PhysicsConfig"),
    ]);
    select(&mut h, &["box", "phys"]);

    h.run_console_line(&mut world, "/dup");
    assert_eq!(h.entries.len(), 3, "the singleton is skipped");
    assert_eq!(declared_id(&h.entries[2]), Some("box_1"));
    assert_eq!(
        h.entries[2]["args"]["position"],
        serde_json::json!([1.0, 2.0, 3.0]),
        "the copy keeps the original's args"
    );
    assert_eq!(
        selected(&h),
        vec!["box_1"],
        "the copies become the selection"
    );
    assert!(h.dirty);
    let lines = h.console_sink.window(0, 16);
    assert!(lines.iter().any(|l| l.text == "duplicated 1"));

    h.undo(&mut world);
    assert_eq!(h.entries.len(), 2, "one undo step removes the whole batch");
    assert!(!h.can_undo());
}

// Ctrl+D duplicates through the tick, except while the Behavior panel is
// frontmost (its frame_keys own that shortcut for row duplication).
#[test]
fn ctrl_d_duplicates_unless_the_behavior_panel_owns_it() {
    let mut world = world_with_input(FrameInput {
        ctrl: true,
        key_events: vec![KeyEvent::Press(KeyPress::new(InputKey::D, KeyMods::CTRL))],
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    for id in behavior::panel::all_field_ids() {
        world.push_identified(id, TextInput::default());
    }
    let mut h = hook(vec![entry("box", "Sprite")]);
    select(&mut h, &["box"]);
    h.tick(&mut world);
    assert_eq!(h.entries.len(), 2, "Ctrl+D duplicates the selection");

    registry::panel(PanelKey::Behavior).toggle(&mut h, &mut world);
    select(&mut h, &["box"]);
    let before = h.entries.len();
    h.tick(&mut world);
    assert_eq!(
        h.entries.len(),
        before,
        "a frontmost Behavior panel keeps its own Ctrl+D"
    );
}

// A file an entry's type owns is copied for each clone, never over an existing
// file: two originals sharing one file each get their own copy, and each clone
// names its copy while the originals keep theirs. Undo takes back the clones
// in one step and leaves the copies.
#[test]
fn duplicates_own_copies_of_their_files() {
    let dir = tempfile::tempdir().unwrap();
    let path = |name: &str| dir.path().join(name).to_string_lossy().into_owned();
    std::fs::write(dir.path().join("common.hlsl"), "common").unwrap();
    std::fs::write(dir.path().join("tale.md"), "tale").unwrap();
    std::fs::write(dir.path().join("tale_copy.md"), "someone else's").unwrap();
    let mut world = World::new();
    let mut h = hook(vec![
        serde_json::json!({"type": "Shader", "args": {"$id": "a", "fragment": path("common.hlsl")}}),
        serde_json::json!({"type": "Shader", "args": {"$id": "b", "fragment": path("common.hlsl")}}),
        serde_json::json!({"type": "StoryImport", "args": {"$id": "tale", "source": path("tale.md")}}),
    ]);
    select(&mut h, &["a", "b", "tale"]);
    h.run_console_line(&mut world, "/dup");

    let arg = |id: &str, key: &str| {
        let e = h.entries.iter().find(|e| e["args"]["$id"] == id).unwrap();
        e["args"][key].as_str().unwrap().to_string()
    };
    assert_eq!(arg("a", "fragment"), path("common.hlsl"));
    assert_eq!(arg("a_1", "fragment"), path("common_copy.hlsl"));
    assert_eq!(arg("b_1", "fragment"), path("common_copy_1.hlsl"));
    assert_eq!(arg("tale_1", "source"), path("tale_copy_1.md"));
    let read = |n: &str| std::fs::read_to_string(dir.path().join(n)).unwrap();
    assert_eq!(read("common_copy_1.hlsl"), "common");
    assert_eq!(read("tale_copy.md"), "someone else's", "never overwritten");
    assert_eq!(read("tale_copy_1.md"), "tale");

    h.undo(&mut world);
    assert_eq!(h.entries.len(), 3);
    assert!(!h.can_undo(), "one step");
    assert!(
        dir.path().join("common_copy.hlsl").exists(),
        "the copies stay"
    );
}

// A type at its world limit is not duplicated, and says why.
#[test]
fn a_shader_is_not_duplicated_past_the_limit() {
    let max = concinnity_core::gfx::render_types::MAX_SHADER_BUCKETS;
    let entries: Vec<serde_json::Value> = (0..max)
        .map(|i| serde_json::json!({"type": "Shader", "args": {"$id": format!("s{i}"), "fragment": "/cn-none/s.hlsl"}}))
        .collect();
    let mut world = World::new();
    let mut h = hook(entries);
    select(&mut h, &["s0"]);
    h.run_console_line(&mut world, "/dup");
    assert_eq!(h.entries.len(), max);
    assert!(!h.dirty);
    let cards = h.notifier.stack().cards;
    assert!(
        cards
            .iter()
            .any(|c| c.message.starts_with("Not duplicated")),
        "{cards:?}"
    );
}

// A placed SdfVolume shares its field file with its copies, as a Prop shares
// its mesh: the copy names the same file and nothing is written.
#[test]
fn a_duplicated_sdf_volume_shares_its_field_file() {
    let dir = tempfile::tempdir().unwrap();
    let field = dir.path().join("blob.hlsl");
    std::fs::write(&field, "sdf").unwrap();
    let declared = field.to_string_lossy().into_owned();
    let mut world = World::new();
    let mut h = hook(vec![serde_json::json!({"type": "SdfVolume", "args": {
        "$id": "blob", "fragment_shader": declared,
    }})]);
    select(&mut h, &["blob"]);
    h.run_console_line(&mut world, "/dup");
    assert_eq!(h.entries.len(), 2);
    assert_eq!(h.entries[1]["args"]["fragment_shader"], declared.as_str());
    let files: Vec<_> = std::fs::read_dir(dir.path()).unwrap().collect();
    assert_eq!(files.len(), 1, "no copy written");
}
