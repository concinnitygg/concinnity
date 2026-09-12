// src/editor/hook/tests/duplicate_tests.rs
//
// Duplicating the selection (`hook/duplicate.rs`): the entries a clone appends
// and selects, and the Ctrl+D that stands down while the Behavior panel owns
// the keyboard.

use concinnity_core::components::FrameInput;
use concinnity_core::components::InputKey;
use concinnity_core::components::TextInput;
use concinnity_core::ecs::World;

use super::fixtures::{entry, hook, world_with_input};
use crate::debug_hook::DebugHook;
use crate::editor::behavior;
use crate::editor::hook::entry_name;

use crate::editor::panels::registry::{self, PanelKey};

// Duplicating the selection clones each authored entry (args included) under
// a unique name, skips singletons, selects the copies, and is one undo step.
#[test]
fn duplicate_selection_clones_entries_and_selects_the_copies() {
    let mut world = World::new();
    let mut h = hook(vec![
        serde_json::json!({
            "name": "box", "type": "Prop", "args": { "position": [1.0, 2.0, 3.0] }
        }),
        entry("phys", "PhysicsConfig"),
    ]);
    h.selection.set(vec!["box".to_string(), "phys".to_string()]);

    h.run_console_line(&mut world, "/dup");
    assert_eq!(h.entries.len(), 3, "the singleton is skipped");
    assert_eq!(entry_name(&h.entries[2]), Some("box_1"));
    assert_eq!(
        h.entries[2]["args"]["position"],
        serde_json::json!([1.0, 2.0, 3.0]),
        "the copy keeps the original's args"
    );
    assert_eq!(
        h.selection.iter().collect::<Vec<_>>(),
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
        captured_key: Some(InputKey::D),
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    for id in behavior::panel::all_field_ids() {
        world.add_component(TextInput {
            asset_id: id,
            ..Default::default()
        });
    }
    let mut h = hook(vec![entry("box", "Sprite")]);
    h.selection.replace("box".to_string());
    h.tick(&mut world);
    assert_eq!(h.entries.len(), 2, "Ctrl+D duplicates the selection");

    registry::panel(PanelKey::Behavior).toggle(&mut h, &mut world);
    h.selection.replace("box".to_string());
    let before = h.entries.len();
    h.tick(&mut world);
    assert_eq!(
        h.entries.len(),
        before,
        "a frontmost Behavior panel keeps its own Ctrl+D"
    );
}
