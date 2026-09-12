// src/editor/hook/tests/edits_tests.rs
//
// Committing and persisting an edit (`hook/edits.rs`): the preview rebuild an
// entry change requests, the world a rebuild compiles in memory, the atomic
// world.jsonl write SAVE performs, and the undo / redo stacks over the entry
// list -- what a mark records, what a jump drops, and how the dirty flag
// tracks the saved list across one.

use concinnity_cook::authoring::world::parse_world_jsonl;
use concinnity_core::components::FrameInput;
use concinnity_core::components::TextLabel;
use concinnity_core::ecs::World;

use super::fixtures::{entry, hook, set_field, world_with_fields, world_with_input};
use crate::debug_hook::DebugHook;
use crate::editor::hook::{EditorHook, FormTarget};

use crate::editor::hud::HudAction;

use crate::editor::panels::form_panel::{self, FormAction};

use crate::editor::panels::template_panel::TemplateAction;

use crate::editor::sim;
use crate::test_support::isolate_state_dir;

// Entry changes drive the live preview: a mutation flags a rebuild AND marks
// the world dirty (unsaved); a plain View toggle does neither.
#[test]
fn entry_changes_request_a_preview_rebuild() {
    let mut h = hook(Vec::new());
    h.apply_top(HudAction::ToggleView, &mut World::new());
    assert!(
        !h.rebuild_preview && !h.dirty,
        "a view toggle is not an entry change"
    );
    // Applying a template layers assets: preview rebuild requested + dirty.
    h.open_template_detail(0);
    h.apply_template_detail(TemplateAction::Apply);
    assert!(
        h.rebuild_preview && h.dirty,
        "applying a template updates the live preview and marks unsaved"
    );
}

// The live preview is rebuilt from the in-memory entries with no disk access:
// authored renderable entries build a rendering world directly, and an empty
// world is seeded so a window still shows. This is the swap's source of truth
// now that SAVE only persists.
#[test]
fn build_preview_world_renders_from_in_memory_entries() {
    let _guard = crate::test_support::lock();
    isolate_state_dir();
    // Authored renderable entries (a Room + camera) build a rendering world.
    let h = hook(vec![
        serde_json::json!({"name":"cam","type":"Camera3D","args":{}}),
        serde_json::json!({"name":"room","type":"Room","args":{}}),
    ]);
    assert!(
        concinnity_engine::ecs::renders(
            &h.build_preview_world().expect("authored entries build").0
        ),
        "authored renderable entries render without disk"
    );
    // Empty entries: the seed keeps the preview window from going blank.
    let h = hook(Vec::new());
    assert!(
        concinnity_engine::ecs::renders(&h.build_preview_world().expect("empty world seeds").0),
        "an empty world is seeded so it still renders"
    );
}

// End to end over a really cooked world: an edit to a live type is written into
// the world the build produced -- through the real expansion, name interner and
// entity index -- and no rebuild is asked for. This is the whole point of the
// live path, so it is proved against a real world rather than a stubbed one.
#[test]
fn a_live_edit_reaches_a_really_cooked_world() {
    let _guard = crate::test_support::lock();
    isolate_state_dir();
    let mut h = hook(vec![
        serde_json::json!({"name":"cam","type":"Camera3D","args":{}}),
        serde_json::json!({"name":"room","type":"Room","args":{}}),
        serde_json::json!({"name":"hint","type":"TextLabel","args":{"content":"before"}}),
    ]);
    let (mut world, shadows) = h.build_preview_world().expect("the world builds");
    h.world_shadows = Some(shadows);
    assert_eq!(
        world
            .query::<TextLabel>()
            .next()
            .expect("the label is in the world")
            .content,
        "before"
    );

    h.entries[2]["args"]["content"] = serde_json::json!("after");
    h.mark_changed();
    assert!(
        !h.refresh_preview(&mut world),
        "a TextLabel field is written into the running world"
    );
    assert_eq!(
        world.query::<TextLabel>().next().unwrap().content,
        "after",
        "and the running component carries the edit"
    );

    // Adding a line changes what the expansion produces, so it still rebuilds.
    h.entries.push(serde_json::json!({
        "name":"hint2","type":"TextLabel","args":{"content":"new"}
    }));
    h.mark_changed();
    assert!(h.refresh_preview(&mut world), "a new line rebuilds");
}

#[test]
fn write_jsonl_persists_entries_atomically() {
    let tree = concinnity_testing::TempTree::new();
    let path = tree.join("world.jsonl");
    let path_str = path.to_str().unwrap().to_string();

    let mut h = hook(vec![serde_json::json!({
        "name": "scene", "type": "GraphicsConfig", "args": {}
    })]);
    h.world_path = path_str.clone();
    let mut world = world_with_fields();
    h.selected_type = Some("PointLight".to_string());
    set_field(&mut world, form_panel::NAME_INPUT, "lamp");
    h.apply_form(FormAction::Confirm, &mut world);
    h.write_jsonl().unwrap();

    let content = std::fs::read_to_string(&path).unwrap();
    let parsed = parse_world_jsonl(&content).unwrap();
    assert_eq!(parsed.len(), 2, "both entries written, one line each");
    assert_eq!(parsed[1]["name"], "lamp");
    assert!(!std::path::Path::new(&format!("{path_str}.tmp")).exists());

    let _ = std::fs::remove_file(&path);
}

// A fresh project has no `worlds/` until its first save, so the write creates
// the directory rather than failing on the rename.
#[test]
fn write_jsonl_creates_the_worlds_directory() {
    let tree = concinnity_testing::TempTree::new();
    let path = tree.join("worlds").join("world.jsonl");

    let mut h = hook(vec![serde_json::json!({
        "name": "scene", "type": "GraphicsConfig", "args": {}
    })]);
    h.world_path = path.to_str().unwrap().to_string();
    h.write_jsonl().expect("the write creates its directory");

    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(parse_world_jsonl(&content).unwrap().len(), 1);
}

// A world that does not cook reports why instead of showing an empty tree.
#[test]
fn a_broken_world_reports_its_error_in_the_status_line() {
    let _guard = crate::test_support::lock();
    isolate_state_dir();
    let mut h = hook(vec![serde_json::json!({
        "name": "oops", "type": "NotARealAssetType", "args": {}
    })]);
    h.panel_open = true;
    h.refresh_tree_if_needed();
    assert!(h.tree_groups.is_empty());
    let status = h.tree_status.as_deref().expect("the failure surfaces");
    assert!(status.contains("NotARealAssetType"), "{status}");
}

// A committed edit becomes one undo step: undo restores the pre-edit list (and
// clears dirty when that list matches the on-disk state), redo replays it.
#[test]
fn undo_reverts_a_committed_edit_and_redo_replays_it() {
    let mut world = World::new();
    let mut h = hook(vec![entry("a", "Sprite")]);
    assert!(!h.hud_state().undo && !h.hud_state().redo);

    h.entries.push(entry("b", "Sprite"));
    h.mark_changed();
    assert!(h.dirty && h.hud_state().undo);

    h.undo(&mut world);
    assert_eq!(h.entries, vec![entry("a", "Sprite")]);
    assert!(!h.dirty, "back at the on-disk list: Save chip clears");
    assert!(
        h.rebuild_preview,
        "the restored list drives the live preview"
    );
    assert!(h.hud_state().redo);

    h.redo(&mut world);
    assert_eq!(h.entries, vec![entry("a", "Sprite"), entry("b", "Sprite")]);
    assert!(h.dirty, "the replayed edit is unsaved again");
}

// Editing from an undone state forks the timeline: the redo branch is gone.
#[test]
fn an_edit_after_undo_drops_the_redo_branch() {
    let mut world = World::new();
    let mut h = hook(Vec::new());
    h.entries.push(entry("b", "Sprite"));
    h.mark_changed();
    h.undo(&mut world);
    assert!(h.hud_state().redo);

    h.entries.push(entry("c", "Sprite"));
    h.mark_changed();
    assert!(!h.hud_state().redo, "the new edit invalidates redo");
    h.undo(&mut world);
    assert!(h.entries.is_empty());
}

// A mark_changed that changed nothing (e.g. an Apply that staged no edits)
// records no phantom undo step.
#[test]
fn a_no_change_mark_records_no_undo_step() {
    let mut h = hook(vec![entry("a", "Sprite")]);
    h.mark_changed();
    assert!(!h.hud_state().undo, "nothing changed, nothing to undo");
}

// The open form and row menu index into `entries`; a history jump drops them so
// they can never point at a removed or shifted row.
#[test]
fn undo_drops_entry_indexed_ui_state() {
    let mut world = World::new();
    let mut h = hook(vec![entry("a", "Sprite")]);
    h.entries.push(entry("b", "Sprite"));
    h.mark_changed();
    h.selected_type = Some("Sprite".to_string());
    h.form_target = FormTarget::Entry(1);
    h.row_menu = Some("b".to_string());

    h.undo(&mut world);
    assert_eq!(
        h.form_target,
        FormTarget::New,
        "the form no longer targets a live row"
    );
    assert_eq!(h.selected_type, None);
    assert_eq!(h.row_menu, None);
}

// Ctrl+Z / Ctrl+Y drive the history from the tick, but stand down while a text
// field owns the keyboard or the world holds the cursor (play mode).
#[test]
fn ctrl_z_y_step_history_unless_typing_or_playing() {
    use concinnity_core::components::InputKey;
    let step = |h: &mut EditorHook, key: InputKey| {
        let mut world = world_with_input(FrameInput {
            viewport: [1280.0, 720.0],
            ctrl: true,
            captured_key: Some(key),
            ..Default::default()
        });
        h.tick(&mut world);
    };
    let mut h = hook(Vec::new());
    h.entries.push(entry("b", "Sprite"));
    h.mark_changed();

    // Typing in the Story panel: the shortcut must not fire.
    h.story_focus = true;
    step(&mut h, InputKey::Z);
    assert_eq!(
        h.entries.len(),
        1,
        "suppressed while a text field is focused"
    );
    h.story_focus = false;

    // Play mode: the world owns the keyboard.
    h.sim.state = sim::SimState::Playing;
    step(&mut h, InputKey::Z);
    assert_eq!(h.entries.len(), 1, "suppressed in play mode");
    h.sim.state = sim::SimState::Stopped;

    step(&mut h, InputKey::Z);
    assert!(h.entries.is_empty(), "Ctrl+Z undoes the edit");
    step(&mut h, InputKey::Y);
    assert_eq!(h.entries.len(), 1, "Ctrl+Y redoes it");
}

// SAVE writes world.jsonl and nothing else: the compiled blobs belong to an
// explicit build, so a save leaves the build root as it found it.
#[test]
fn save_writes_the_world_file_and_no_build_output() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    let build_root = dir.path().join(".concinnity");
    crate::project::open(
        concinnity_host::store::paths::StateTree::at(dir.path()).with_build(&build_root),
    );

    let world_path = dir.path().join("worlds").join("world.jsonl");
    let entries = vec![serde_json::json!({"name":"phys","type":"PhysicsConfig","args":{}})];
    let mut h = EditorHook::new(world_path.to_string_lossy().into_owned(), entries.clone());
    h.entries
        .push(serde_json::json!({"name":"hint","type":"TextLabel","args":{}}));
    h.mark_changed();
    h.save();

    let written = std::fs::read_to_string(&world_path).expect("the world file was created");
    assert_eq!(
        parse_world_jsonl(&written).unwrap(),
        h.entries,
        "the working entries are what landed on disk"
    );
    assert!(!h.dirty, "a written save clears the unsaved chip");
    assert_eq!(h.saved, h.entries, "and re-baselines dirty tracking");
    assert!(
        !build_root.exists(),
        "no blobs, no lock, no build root at all"
    );

    crate::test_support::isolate_state_dir();
}

// A save that cannot write leaves the world dirty for the next attempt.
#[test]
fn a_failed_save_leaves_the_world_dirty() {
    let dir = concinnity_testing::TempTree::new();
    // A path under a file, so `create_dir_all` of the parent fails.
    let blocker = dir.path().join("blocker");
    std::fs::write(&blocker, b"not a directory").unwrap();
    let world_path = blocker.join("worlds").join("world.jsonl");

    let mut h = EditorHook::new(world_path.to_string_lossy().into_owned(), Vec::new());
    h.entries
        .push(serde_json::json!({"name":"hint","type":"TextLabel","args":{}}));
    h.mark_changed();
    h.save();

    assert!(h.dirty, "the world stays unsaved");
    assert!(h.saved.is_empty(), "and the saved baseline does not move");
}

// A successful SAVE re-baselines dirty tracking: undoing past it re-dirties,
// redoing back to the saved list cleans the chip again.
#[test]
fn dirty_tracks_the_saved_list_across_history_jumps() {
    let mut world = World::new();
    let mut h = hook(Vec::new());
    h.entries.push(entry("b", "Sprite"));
    h.mark_changed();
    // Stand in for a successful SAVE (which would hit disk).
    h.dirty = false;
    h.saved = h.entries.clone();

    h.undo(&mut world);
    assert!(h.dirty, "behind the saved list is an unsaved state");
    h.redo(&mut world);
    assert!(!h.dirty, "redo back to the saved list clears the chip");
}
