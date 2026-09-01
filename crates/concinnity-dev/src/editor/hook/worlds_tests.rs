// src/editor/hook/worlds_tests.rs
//
// Tests for the Worlds panel's actions: the listing the hook builds, what
// opening a world retargets, the naming rules a New has to pass, and the two
// confirmations (delete, and switching away from unsaved edits).

use super::*;
use crate::components::TextInput;

const VP: [f32; 2] = [1280.0, 720.0];

// A project rooted at `dir`, sharing the machine-wide build cache. The caller
// holds the process lock: this moves the session-wide project.
fn open_project(dir: &std::path::Path) {
    crate::project::open(
        concinnity_host::store::paths::StateTree::at(dir).with_cache(
            concinnity_testing::shared_cache_dir("concinnity-dev-tests-cache"),
        ),
    );
}

fn entry(name: &str) -> serde_json::Value {
    serde_json::json!({"name": name, "type": "Prop", "args": {}})
}

// Write a world file with `entries` and pin its mtime, so a listing's order is
// the one under test rather than whatever the filesystem's resolution gives.
fn write_world(
    dir: &std::path::Path,
    name: &str,
    entries: &[serde_json::Value],
    at_secs: u64,
) -> std::path::PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join(format!("{name}.jsonl"));
    std::fs::write(&path, crate::world::write_world_jsonl(entries).unwrap()).unwrap();
    let file = std::fs::File::options().write(true).open(&path).unwrap();
    file.set_modified(std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(at_secs))
        .unwrap();
    path
}

fn hook_at(path: &std::path::Path, entries: Vec<serde_json::Value>) -> EditorHook {
    let mut h = EditorHook::new(path.to_string_lossy().into_owned(), entries);
    h.refresh_worlds();
    h
}

fn world_with_name_field() -> World {
    let mut world = World::new();
    for id in worlds::all_field_ids() {
        world.add_component(TextInput {
            asset_id: id,
            ..Default::default()
        });
    }
    world
}

fn set_name(world: &mut World, text: &str) {
    for t in world.query_mut::<TextInput>() {
        if t.asset_id == worlds::NAME_INPUT {
            t.content = text.to_string();
        }
    }
}

fn row_index(h: &EditorHook, name: &str) -> usize {
    h.worlds_rows
        .iter()
        .position(|r| r.name == name)
        .unwrap_or_else(|| panic!("{name} is not listed"))
}

fn names(h: &EditorHook) -> Vec<String> {
    h.worlds_rows.iter().map(|r| r.name.clone()).collect()
}

fn button_index(h: &EditorHook, label: &str) -> usize {
    let buttons = &h.modal.as_ref().expect("a dialog is open").buttons;
    buttons
        .iter()
        .position(|b| b.label == label)
        .unwrap_or_else(|| panic!("no '{label}' button"))
}

// Press one of the open dialog's buttons.
fn press_modal(h: &mut EditorHook, world: &mut World, label: &str) {
    let i = button_index(h, label);
    let count = h.modal.as_ref().unwrap().buttons.len();
    let r = modal::button_rect(modal::panel_rect(VP), count, i);
    let input = FrameInput {
        left_click: true,
        mouse_x: r[0] + 2.0,
        mouse_y: r[1] + 2.0,
        viewport: VP,
        ..Default::default()
    };
    assert!(h.route_modal_click(&input, VP, world));
}

// The listing is the project's worlds newest-edited first, the legacy root
// world included, with the session's own world marked.
#[test]
fn the_listing_is_newest_first_and_marks_the_open_world() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let worlds_dir = dir.path().join("worlds");
    let arena = write_world(&worlds_dir, "arena", &[], 1_000);
    write_world(&worlds_dir, "lobby", &[], 3_000);
    write_world(dir.path(), "world", &[], 2_000);

    let h = hook_at(&arena, Vec::new());
    assert_eq!(names(&h), ["lobby", "world", "arena"]);
    assert!(h.worlds_rows[row_index(&h, "arena")].open);
    assert!(!h.worlds_rows[row_index(&h, "lobby")].open);

    crate::test_support::isolate_state_dir();
}

// Opening a world moves the whole session onto it: the path a SAVE writes, the
// working entries with a clean history, and every piece of state that indexed
// the world left behind. The compiled world follows on the rebuild this asks
// for.
#[test]
fn opening_a_world_retargets_the_whole_session() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let worlds_dir = dir.path().join("worlds");
    let arena = write_world(&worlds_dir, "arena", &[entry("crate_a")], 1_000);
    write_world(&worlds_dir, "lobby", &[entry("desk"), entry("lamp")], 2_000);

    let mut h = hook_at(&arena, vec![entry("crate_a")]);
    h.worlds_open = true;
    // A session with history, a selection, and per-world hide state behind it.
    h.entries.push(entry("crate_b"));
    h.mark_changed();
    h.saved = h.entries.clone();
    h.dirty = false;
    h.selection.replace("crate_a".to_string());
    h.hidden_assets.insert("crate_b".to_string());
    h.rebuild_preview = false;
    assert!(h.can_undo());

    let mut world = world_with_name_field();
    let i = row_index(&h, "lobby");
    h.apply_worlds_action(WorldsAction::Open(i), &mut world);

    assert_eq!(
        h.world_path,
        worlds_dir.join("lobby.jsonl").to_string_lossy()
    );
    assert_eq!(h.entries.len(), 2);
    assert_eq!(h.entries[0]["name"], "desk");
    assert_eq!(h.saved, h.entries, "the loaded list is what is on disk");
    assert_eq!(h.baseline, h.entries);
    assert!(!h.dirty);
    assert!(
        !h.can_undo(),
        "the history belonged to the world left behind"
    );
    assert!(
        h.rebuild_preview && h.rebuild_required,
        "the compiled world is swapped on the next frame"
    );
    assert!(h.world_shadows.is_none());
    assert!(h.tree_stale && h.tree_groups.is_empty());
    assert_eq!(h.selection.iter().count(), 0);
    assert!(h.hidden_assets.is_empty());
    assert!(!h.worlds_open, "the panel has done its job");
    assert!(h.worlds_rows[row_index(&h, "lobby")].open);

    crate::test_support::isolate_state_dir();
}

// A world file that will not parse leaves the session on the world it has, and
// says why on the panel's status line.
#[test]
fn opening_an_unparseable_world_keeps_the_open_one() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let worlds_dir = dir.path().join("worlds");
    let arena = write_world(&worlds_dir, "arena", &[entry("crate_a")], 1_000);
    write_world(&worlds_dir, "broken", &[], 2_000);
    std::fs::write(worlds_dir.join("broken.jsonl"), "{not json").unwrap();

    let mut h = hook_at(&arena, vec![entry("crate_a")]);
    let mut world = world_with_name_field();
    let i = row_index(&h, "broken");
    h.apply_worlds_action(WorldsAction::Open(i), &mut world);

    assert_eq!(h.world_path, arena.to_string_lossy());
    assert_eq!(h.entries.len(), 1);
    assert!(h.worlds_status.is_some(), "the failure is reported");

    crate::test_support::isolate_state_dir();
}

// New names the file first: it exists, and lists, before anything is authored
// into it.
#[test]
fn new_creates_the_world_file_and_opens_it() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let worlds_dir = dir.path().join("worlds");
    let arena = write_world(&worlds_dir, "arena", &[entry("crate_a")], 1_000);

    let mut h = hook_at(&arena, vec![entry("crate_a")]);
    h.worlds_open = true;
    let mut world = world_with_name_field();
    set_name(&mut world, " lobby ");
    h.apply_worlds_action(WorldsAction::New, &mut world);

    let created = worlds_dir.join("lobby.jsonl");
    assert_eq!(std::fs::read_to_string(&created).unwrap(), "");
    assert_eq!(h.world_path, created.to_string_lossy());
    assert!(h.entries.is_empty() && !h.dirty);
    assert!(names(&h).contains(&"lobby".to_string()));
    assert!(!h.worlds_open);
    assert_eq!(
        widget::field_text(&world, worlds::NAME_INPUT),
        "",
        "the name field is cleared for the next one"
    );

    crate::test_support::isolate_state_dir();
}

// A name that cannot become a world says so instead of failing silently, and
// nothing is created or retargeted.
#[test]
fn new_rejects_unusable_names_with_a_visible_reason() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let worlds_dir = dir.path().join("worlds");
    let arena = write_world(&worlds_dir, "arena", &[], 1_000);

    let mut h = hook_at(&arena, Vec::new());
    let mut world = world_with_name_field();
    for (typed, expect) in [("   ", "name"), ("arena", "exists"), ("a/b", "/")] {
        set_name(&mut world, typed);
        h.apply_worlds_action(WorldsAction::New, &mut world);
        let status = h.worlds_status.as_deref().unwrap_or_default();
        assert!(
            status.contains(expect),
            "'{typed}' was rejected as: {status}"
        );
        assert_eq!(h.world_path, arena.to_string_lossy());
        assert_eq!(names(&h), ["arena"], "nothing was created");
    }

    crate::test_support::isolate_state_dir();
}

// Delete asks first: the file survives a Cancel, and a confirm takes both it
// and the session state kept under its name.
#[test]
fn delete_asks_first_then_removes_the_file_and_its_session_entry() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let worlds_dir = dir.path().join("worlds");
    let arena = write_world(&worlds_dir, "arena", &[], 1_000);
    let lobby = write_world(&worlds_dir, "lobby", &[], 2_000);

    // A session entry for the world about to go.
    let store_path = session_store::default_path().expect("the temp project has a store");
    let mut store = session_store::SessionStore::default();
    store
        .worlds
        .insert("lobby".to_string(), session_store::WorldSession::default());
    store
        .worlds
        .insert("arena".to_string(), session_store::WorldSession::default());
    session_store::save(&store_path, &store).unwrap();

    let mut h = hook_at(&arena, Vec::new());
    let mut world = world_with_name_field();
    let i = row_index(&h, "lobby");

    // Cancel leaves everything alone.
    h.apply_worlds_action(WorldsAction::Delete(i), &mut world);
    let buttons = &h.modal.as_ref().expect("a dialog is open").buttons;
    assert_eq!(buttons.len(), 2);
    assert!(
        buttons[button_index(&h, "Delete")].danger,
        "the destructive button is marked"
    );
    press_modal(&mut h, &mut world, "Cancel");
    assert!(h.modal.is_none() && lobby.exists());

    // Confirming takes the file and the store entry.
    h.apply_worlds_action(WorldsAction::Delete(i), &mut world);
    press_modal(&mut h, &mut world, "Delete");
    assert!(h.modal.is_none());
    assert!(!lobby.exists());
    assert_eq!(names(&h), ["arena"]);
    let back = session_store::load(&store_path);
    assert!(!back.worlds.contains_key("lobby"));
    assert!(
        back.worlds.contains_key("arena"),
        "only the deleted one goes"
    );

    crate::test_support::isolate_state_dir();
}

// Deleting the world the session has open is allowed: it keeps running on the
// entries it holds, and since none of them are on disk any more it reads as
// unsaved, so a later SAVE writes the file back.
#[test]
fn deleting_the_open_world_keeps_the_session_and_marks_it_unsaved() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let worlds_dir = dir.path().join("worlds");
    let arena = write_world(&worlds_dir, "arena", &[entry("crate_a")], 1_000);

    let mut h = hook_at(&arena, vec![entry("crate_a")]);
    let mut world = world_with_name_field();
    let i = row_index(&h, "arena");
    h.apply_worlds_action(WorldsAction::Delete(i), &mut world);
    press_modal(&mut h, &mut world, "Delete");

    assert!(!arena.exists());
    assert_eq!(
        h.world_path,
        arena.to_string_lossy(),
        "still the edit target"
    );
    assert_eq!(h.entries.len(), 1, "the session keeps its entries");
    assert!(h.dirty, "nothing of the world is on disk any more");
    assert!(h.worlds_rows.is_empty());

    // A save writes it back.
    h.save();
    assert!(arena.exists() && !h.dirty);

    crate::test_support::isolate_state_dir();
}

// Switching away from unsaved edits asks first, with all three answers.
#[test]
fn a_dirty_switch_asks_and_cancel_stays_put() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let worlds_dir = dir.path().join("worlds");
    let arena = write_world(&worlds_dir, "arena", &[entry("crate_a")], 1_000);
    write_world(&worlds_dir, "lobby", &[entry("desk")], 2_000);

    let mut h = hook_at(&arena, vec![entry("crate_a")]);
    h.entries.push(entry("crate_b"));
    h.mark_changed();
    assert!(h.dirty);

    let mut world = world_with_name_field();
    let i = row_index(&h, "lobby");
    h.apply_worlds_action(WorldsAction::Open(i), &mut world);
    assert_eq!(h.modal.as_ref().unwrap().buttons.len(), 3);
    assert!(h.modal.as_ref().unwrap().buttons[button_index(&h, "Discard")].danger);

    press_modal(&mut h, &mut world, "Cancel");
    assert_eq!(h.world_path, arena.to_string_lossy(), "nothing switched");
    assert!(h.dirty, "the edits are still only in memory");
    assert_eq!(h.entries.len(), 2);

    crate::test_support::isolate_state_dir();
}

#[test]
fn a_dirty_switch_saves_before_switching() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let worlds_dir = dir.path().join("worlds");
    let arena = write_world(&worlds_dir, "arena", &[entry("crate_a")], 1_000);
    let lobby = write_world(&worlds_dir, "lobby", &[entry("desk")], 2_000);

    let mut h = hook_at(&arena, vec![entry("crate_a")]);
    h.entries.push(entry("crate_b"));
    h.mark_changed();

    let mut world = world_with_name_field();
    let i = row_index(&h, "lobby");
    h.apply_worlds_action(WorldsAction::Open(i), &mut world);
    press_modal(&mut h, &mut world, "Save");

    let written = crate::world::parse_world_jsonl(&std::fs::read_to_string(&arena).unwrap())
        .expect("the edits were written before the switch");
    assert_eq!(written.len(), 2);
    assert_eq!(h.world_path, lobby.to_string_lossy());
    assert_eq!(h.entries.len(), 1, "and the switch happened");
    assert!(!h.dirty);

    crate::test_support::isolate_state_dir();
}

#[test]
fn a_dirty_switch_can_discard_the_edits() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let worlds_dir = dir.path().join("worlds");
    let arena = write_world(&worlds_dir, "arena", &[entry("crate_a")], 1_000);
    let lobby = write_world(&worlds_dir, "lobby", &[entry("desk")], 2_000);

    let mut h = hook_at(&arena, vec![entry("crate_a")]);
    h.entries.push(entry("crate_b"));
    h.mark_changed();

    let mut world = world_with_name_field();
    let i = row_index(&h, "lobby");
    h.apply_worlds_action(WorldsAction::Open(i), &mut world);
    press_modal(&mut h, &mut world, "Discard");

    let untouched = crate::world::parse_world_jsonl(&std::fs::read_to_string(&arena).unwrap())
        .expect("the world left behind still parses");
    assert_eq!(untouched.len(), 1, "the edits were dropped, not written");
    assert_eq!(h.world_path, lobby.to_string_lossy());
    assert!(!h.dirty);

    crate::test_support::isolate_state_dir();
}

// Creating a world while the open one is dirty takes the same guard.
#[test]
fn a_dirty_new_asks_before_creating() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let worlds_dir = dir.path().join("worlds");
    let arena = write_world(&worlds_dir, "arena", &[entry("crate_a")], 1_000);

    let mut h = hook_at(&arena, vec![entry("crate_a")]);
    h.entries.push(entry("crate_b"));
    h.mark_changed();

    let mut world = world_with_name_field();
    set_name(&mut world, "lobby");
    h.apply_worlds_action(WorldsAction::New, &mut world);
    assert_eq!(h.modal.as_ref().unwrap().buttons.len(), 3);
    assert!(
        !worlds_dir.join("lobby.jsonl").exists(),
        "nothing is created until the guard clears"
    );

    press_modal(&mut h, &mut world, "Discard");
    assert!(worlds_dir.join("lobby.jsonl").exists());
    assert_eq!(
        h.world_path,
        worlds_dir.join("lobby.jsonl").to_string_lossy()
    );

    crate::test_support::isolate_state_dir();
}

// The panel claims presses inside itself and misses everywhere else, so the
// panels behind it stay reachable (`try_panel_press` takes the first claim).
#[test]
fn panel_presses_are_rect_guarded() {
    let mut h = EditorHook::new("unused.jsonl".to_string(), Vec::new());
    h.worlds_open = true;
    h.worlds_rows = vec![WorldRow {
        name: "arena".to_string(),
        path: "/p/worlds/arena.jsonl".to_string(),
        open: false,
    }];
    let mut world = world_with_name_field();
    let o = h.origin(PanelKey::Worlds, VP);
    let s = registry::panel(PanelKey::Worlds).size(&h);

    // Off the panel on every side: the press falls through untouched.
    for (x, y) in [
        (o[0] - 4.0, o[1] + 40.0),
        (o[0] + s[0] + 4.0, o[1] + 40.0),
        (o[0] + 40.0, o[1] + s[1] + 4.0),
    ] {
        assert!(
            !h.try_panel_press(PanelKey::Worlds, x, y, VP, &mut world),
            "({x}, {y}) is off the panel"
        );
    }
    // Body chrome below the rows is claimed and blurs the name field.
    h.worlds_focus = true;
    assert!(h.try_panel_press(
        PanelKey::Worlds,
        o[0] + 40.0,
        o[1] + s[1] - 4.0,
        VP,
        &mut world
    ));
    assert!(!h.worlds_focus);

    // A hidden panel claims nothing.
    h.worlds_open = false;
    let r = worlds::row_rect(o, 0);
    assert!(!h.try_panel_press(PanelKey::Worlds, r[0] + 4.0, r[1] + 4.0, VP, &mut world));
}

// The typed name survives the HUD re-injection a world swap performs, so a
// half-typed world name is not blanked by an unrelated rebuild.
#[test]
fn the_name_field_is_carried_across_a_preview_swap() {
    let mut world = world_with_name_field();
    widget::seed_field(&mut world, worlds::NAME_INPUT, "half-typed");
    let snapshot = EditorHook::field_snapshot(&world);
    let mut fresh = world_with_name_field();
    EditorHook::restore_fields(&mut fresh, &snapshot);
    assert_eq!(widget::field_text(&fresh, worlds::NAME_INPUT), "half-typed");
}
