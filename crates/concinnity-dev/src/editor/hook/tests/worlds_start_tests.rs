//! Tests for the start screen: the chrome it suppresses, the larger presentation
//! it draws, previewing a world without retargeting the session, committing that
//! preview without compiling it twice, and the fallbacks a delete leaves behind.
//! The subject is shared with `edit/worlds_tests.rs`, which covers the switcher the
//! same panel becomes once a world is open.

use concinnity_engine::components::{FrameInput, Sprite, TextLabel, UiEvent, UiValue};
use concinnity_engine::ecs::World;

use super::fixtures::{
    VP, hook_at, open_project, press_modal, prop_entry, set_world_name, world_names,
    world_row_index, write_world,
};
use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::selected;
use crate::editor::panels::registry::{self, PanelKey};
use crate::editor::worlds::{self, WorldsAction};
use crate::editor::{hud, live};
use crate::frame_hook::FrameHook;

// A start-screen session over `dir`'s worlds, showing `previewing` (by name).
// Mirrors what `run_editor` builds -- the session's own path is the
// placeholder a world has not been picked for yet, and it boots on nothing --
// wound forward past the frames the screen spends bringing its pick up, so a
// test starts with that world compiled and showing.
fn start_hook(dir: &std::path::Path, previewing: Option<&str>) -> EditorHook {
    let mut h = booting_hook(dir, previewing);
    h.start_drawn = EditorHook::START_PREVIEW_DELAY;
    h.drive_start_preview();
    settle_rebuild(&mut h);
    h
}

// The same session at the frame the window opens on: the listing is up, the
// pick is preselected, and nothing has been compiled for it yet.
fn booting_hook(dir: &std::path::Path, previewing: Option<&str>) -> EditorHook {
    let placeholder = dir.join("world.jsonl");
    let picked = previewing
        .map(|name| dir.join(format!("{name}.jsonl")))
        .map(|p| p.to_string_lossy().into_owned());
    EditorHook::new(placeholder.to_string_lossy().into_owned(), Vec::new())
        .with_start_screen(picked)
}

// Stand in for the frame loop's swap: the live world is now the one the entries
// describe, and nothing is owed.
fn settle_rebuild(h: &mut EditorHook) {
    h.world_entries = h.entries.clone();
    h.rebuild_preview = false;
    h.rebuild_required = false;
    h.rebuild_countdown = 0;
}

fn entry_names(h: &EditorHook) -> Vec<String> {
    h.entries
        .iter()
        .map(|e| e["args"]["$id"].as_str().unwrap_or_default().to_string())
        .collect()
}

// Click the start screen's panel at `(mx, my)` through the whole tick path, so
// the routing under test is the one a session runs.
fn click_at(h: &mut EditorHook, world: &mut World, mx: f32, my: f32) {
    if world.query::<FrameInput>().next().is_none() {
        world.add_component(FrameInput::default());
    }
    if let Some(i) = world.query_mut::<FrameInput>().last() {
        *i = FrameInput {
            viewport: VP,
            mouse_x: mx,
            mouse_y: my,
            left_click: true,
            left_button_down: true,
            ..Default::default()
        };
    }
    h.tick(world);
}

// Send the panel `event` carrying `i`, through the whole tick path.
fn send(h: &mut EditorHook, world: &mut World, event: &str, i: usize) {
    world.events_mut::<UiEvent>().send(UiEvent {
        name: event.into(),
        element: None,
        value: Some(UiValue::Int(i32::try_from(i).unwrap())),
    });
    click_at(h, world, -100.0, -100.0);
}

// The row the start screen marks, and the one it shows behind the panel.
fn selected_row(h: &EditorHook) -> Option<usize> {
    h.worlds_row_of(h.worlds.selected.as_ref())
}

fn previewing_row(h: &EditorHook) -> Option<usize> {
    h.worlds_row_of(h.worlds.preview.as_ref())
}

// The start screen is the whole session: no top bar, and no other panel draws
// or routes, whatever its own open flag says.
#[test]
fn the_start_screen_suppresses_the_top_bar_and_every_other_panel() {
    let mut h = EditorHook::new("unused.jsonl".to_string(), Vec::new()).with_start_screen(None);
    // Panels a session would have open behind it.
    h.open[PanelKey::Assets] = true;
    h.open[PanelKey::Preview] = true;
    h.open[PanelKey::Console] = true;
    h.open[PanelKey::View] = true;

    assert!(!h.hud_state().visible, "the top bar is not drawn");
    assert!(h.panel_shown(PanelKey::Worlds));
    for key in PanelKey::ALL {
        if key != PanelKey::Worlds {
            assert!(
                !h.panel_shown(key),
                "{key:?} stands down on the start screen"
            );
        }
    }
    assert_eq!(h.frontmost_open_panel(), Some(PanelKey::Worlds));

    // Opening a world hands the session back: the suppressed panels kept their
    // state and come straight back, and so does the bar.
    h.leave_start_screen();
    assert!(h.hud_state().visible);
    assert!(h.panel_shown(PanelKey::Assets) && h.panel_shown(PanelKey::Console));
}

// The cover stands over the render and nothing else: the listing beside it
// stays readable and clickable through the compile, and the cover sits above
// the shot fade it hands over to.
#[test]
fn the_loading_cover_takes_the_render_and_leaves_the_listing() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let root = dir.path();
    write_world(root, "lobby", &[prop_entry("desk")], 3_000);

    let mut h = booting_hook(dir.path(), Some("lobby"));
    h.viewport = VP;
    let mut world = World::new();
    for &id in &worlds::loading::ids().sprites {
        world.push_identified(id, Sprite::default());
    }
    for &id in &worlds::loading::ids().labels {
        world.push_identified(id, TextLabel::default());
    }
    h.drive_loading_draw(&mut world, true);

    let cover = world
        .get_by_id::<Sprite>(worlds::loading::COVER)
        .expect("the cover");
    assert!(cover.visible);
    let sidebar = h.worlds_layout().size()[0];
    assert_eq!(cover.x, sidebar, "it starts where the sidebar ends");
    assert_eq!(cover.width, VP[0] - sidebar);
    assert_eq!(cover.height, VP[1], "and runs the window's full height");
    let caption = world
        .get_by_id::<TextLabel>(worlds::loading::CAPTION)
        .expect("the caption");
    assert_eq!(caption.content, "Loading lobby", "naming what is compiling");

    // Above the shot fade, under the listing.
    h.cinematic = Some(worlds::cinematic::Cinematic::new());
    let layers = h.compute_layers(None);
    let cover_layer = layers[&worlds::loading::COVER];
    assert!(cover_layer > layers[&worlds::cinematic::FADE]);
    for id in registry::panel(PanelKey::Worlds).ids().all() {
        assert!(layers[&id] > cover_layer, "the listing draws over it");
    }

    // Once the world is up, the cover is gone and claims no layer.
    settle_rebuild(&mut h);
    h.start_preview = None;
    h.drive_loading_draw(&mut world, true);
    assert!(
        !world
            .get_by_id::<Sprite>(worlds::loading::COVER)
            .unwrap()
            .visible
    );
    assert!(!h.compute_layers(None).contains_key(&worlds::loading::COVER));

    crate::test_support::isolate_state_dir();
}

// F1 cannot hide the start screen: with no top bar and no other panel, hiding
// it would leave the session with nothing to click.
#[test]
fn f1_cannot_hide_the_start_screen() {
    let mut h = EditorHook::new("unused.jsonl".to_string(), Vec::new()).with_start_screen(None);
    let mut world = World::new();
    world.add_component(FrameInput {
        hud_toggle: true,
        viewport: VP,
        ..Default::default()
    });
    h.tick(&mut world);
    assert!(h.hud_visible, "F1 stands down while the start screen is up");

    // Once a world is open it is an ordinary editor session again.
    h.leave_start_screen();
    h.tick(&mut world);
    assert!(!h.hud_visible);
}

// Two presentations of one panel: the start screen is a narrow sidebar docked
// down the window's left edge at its full height; the switcher keeps the wider
// floating panel under the top bar.
#[test]
fn the_two_presentations_differ_in_size_and_anchor() {
    let mut h = EditorHook::new("unused.jsonl".to_string(), Vec::new()).with_start_screen(None);
    h.viewport = VP;
    let start = registry::panel(PanelKey::Worlds).size(&h);
    let start_o = h.origin(PanelKey::Worlds, VP);
    h.leave_start_screen();
    let session = registry::panel(PanelKey::Worlds).size(&h);
    let session_o = h.origin(PanelKey::Worlds, VP);

    assert!(start[0] < session[0], "the sidebar is the narrower column");
    assert_eq!(start_o, [0.0, 0.0], "docked to the window's top-left");
    assert_eq!(start[1], VP[1], "and running its full height");
    assert!(
        start_o[0] + start[0] < VP[0] * 0.3,
        "the render owns everything to its right"
    );
    // The switcher hangs from its anchor under the top bar.
    assert_eq!(session_o[1], hud::body_top() + 24.0);
    assert!((session_o[0] + session[0] * 0.5 - VP[0] * 0.5).abs() < 0.5);
}

// The window chrome the backend reports reaches the sidebar through the frame's
// input, so its content clears the OS window buttons floating over the render.
#[test]
fn the_reported_top_inset_reaches_the_sidebar() {
    let mut h = EditorHook::new("unused.jsonl".to_string(), Vec::new()).with_start_screen(None);
    let mut world = World::new();
    world.add_component(FrameInput {
        viewport: VP,
        top_inset: 28.0,
        ..Default::default()
    });
    h.tick(&mut world);

    let o = h.origin(PanelKey::Worlds, VP);
    assert_eq!(o, [0.0, 0.0], "the panel itself still reaches the top");
    assert_eq!(h.worlds_layout().content_offset(), 28.0);
}

// The start screen opens on the project's most recent world, preselected. The
// window comes up on the listing alone: nothing of that world is read or
// compiled until the screen has been drawn, and until it is the preview area
// carries the loading cover.
#[test]
fn the_newest_world_is_preselected_and_compiled_once_the_screen_is_up() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let root = dir.path();
    write_world(root, "arena", &[prop_entry("crate_a")], 1_000);
    write_world(root, "lobby", &[prop_entry("desk")], 3_000);

    let newest = worlds::files::newest(Some(root)).expect("a world to open on");
    assert_eq!(newest.name, "lobby", "the most recently edited world");

    let mut h = booting_hook(dir.path(), Some("lobby"));
    let view = h.make_worlds_view();
    assert_eq!(view.layout.mode, worlds::Mode::Start);
    assert_eq!(world_names(&h)[0], "lobby");
    assert_eq!(selected_row(&h), Some(0), "the top row is preselected");
    assert_eq!(
        previewing_row(&h),
        None,
        "but nothing is showing behind it yet"
    );
    assert!(entry_names(&h).is_empty(), "the window opens on no world");
    assert!(!h.rebuild_preview, "and asks for no compile to open");
    assert!(
        h.loading_preview(),
        "the preview area says what it is waiting on"
    );

    // The first frames go to the screen itself; the pick is staged once it has
    // been drawn, and only then is the world read and a rebuild owed.
    let mut world = World::new();
    world.add_component(FrameInput {
        viewport: VP,
        ..Default::default()
    });
    h.tick(&mut world);
    assert!(h.start_preview.is_some(), "still owed after one frame");
    for _ in 0..EditorHook::START_PREVIEW_DELAY {
        h.tick(&mut world);
    }
    assert!(h.start_preview.is_none(), "staged once the screen was up");
    assert_eq!(entry_names(&h), ["desk"]);
    assert!(h.rebuild_preview && h.rebuild_required);
    assert!(
        h.rebuild_countdown > 0,
        "and the cover is given frames before the compile blocks on it"
    );
    assert_eq!(previewing_row(&h), Some(0));
    assert!(
        h.loading_preview(),
        "which the cover stands over until it lands"
    );

    settle_rebuild(&mut h);
    assert!(!h.loading_preview(), "and stands down once it has");

    crate::test_support::isolate_state_dir();
}

// A project with no worlds at all opens on the seeded empty scene with nothing
// picked, which is what the panel's empty listing already says.
#[test]
fn a_project_with_no_worlds_opens_on_the_empty_scene() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());

    let h = start_hook(dir.path(), None);
    let view = h.make_worlds_view();
    assert!(view.rows.is_empty());
    assert_eq!(selected_row(&h), None);
    assert_eq!(previewing_row(&h), None);
    assert!(h.entries.is_empty() && !h.dirty);

    crate::test_support::isolate_state_dir();
}

// Selecting a row shows that world behind the screen without moving the session
// onto it: the path a SAVE would write, the watcher, and the screen itself all
// stay where they are.
#[test]
fn selecting_a_row_previews_it_without_retargeting_the_session() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let root = dir.path();
    write_world(root, "arena", &[prop_entry("crate_a")], 1_000);
    write_world(root, "lobby", &[prop_entry("desk")], 3_000);

    let mut h = start_hook(dir.path(), Some("lobby"));
    settle_rebuild(&mut h);
    let placeholder = h.world_path.clone();

    let mut world = World::new();
    let i = world_row_index(&h, "arena");
    h.apply_worlds_action(WorldsAction::Select(i), &mut world);

    assert_eq!(entry_names(&h), ["crate_a"], "the picked world is staged");
    assert!(
        h.rebuild_preview && h.rebuild_required,
        "and swapped in on the next frame"
    );
    assert_eq!(h.world_path, placeholder, "the session was not retargeted");
    assert!(
        h.start_mode && h.open[PanelKey::Worlds],
        "the screen stays up over it"
    );
    assert!(!h.dirty && !h.can_undo(), "a preview is not an edit");
    assert_eq!(h.saved, h.entries, "and never reads as unsaved");
    assert_eq!(selected_row(&h), Some(i));
    assert_eq!(previewing_row(&h), Some(i));

    crate::test_support::isolate_state_dir();
}

// A world that will not parse cannot be shown: the screen says why and keeps
// what was already behind it.
#[test]
fn selecting_an_unparseable_world_reports_it_and_keeps_the_preview() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let root = dir.path();
    write_world(root, "lobby", &[prop_entry("desk")], 3_000);
    write_world(root, "broken", &[], 2_000);
    std::fs::write(root.join("broken.jsonl"), "{not json").unwrap();

    let mut h = start_hook(dir.path(), Some("lobby"));
    settle_rebuild(&mut h);
    let mut world = World::new();
    let i = world_row_index(&h, "broken");
    h.apply_worlds_action(WorldsAction::Select(i), &mut world);

    assert!(h.worlds.status.is_some(), "the failure is reported");
    assert_eq!(entry_names(&h), ["desk"], "lobby is still what shows");
    assert!(!h.rebuild_preview, "nothing was staged to rebuild");

    crate::test_support::isolate_state_dir();
}

// Opening the world already showing commits it: the session retargets onto it
// and the screen closes, with no second compile of a world that is already
// running.
#[test]
fn opening_the_previewed_world_commits_it_without_rebuilding() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let root = dir.path();
    write_world(root, "arena", &[prop_entry("crate_a")], 1_000);
    let lobby = write_world(root, "lobby", &[prop_entry("desk")], 3_000);

    let mut h = start_hook(dir.path(), Some("lobby"));
    settle_rebuild(&mut h);
    h.world_shadows = Some(live::ShadowBaselines::default());

    let mut world = World::new();
    let i = world_row_index(&h, "lobby");
    h.apply_worlds_action(WorldsAction::Open(i), &mut world);

    assert_eq!(
        h.world_path,
        lobby.to_string_lossy(),
        "the session moved on"
    );
    assert_eq!(h.saved, h.entries);
    assert!(!h.dirty && !h.can_undo());
    assert!(
        !h.rebuild_preview && !h.rebuild_required,
        "the running world already is this world"
    );
    assert!(
        h.world_shadows.is_some(),
        "the adopted world keeps its template baselines"
    );
    assert!(!h.start_mode, "the start screen is over for the session");
    assert!(!h.open[PanelKey::Worlds]);
    assert!(h.worlds.selected.is_none() && h.worlds.preview.is_none());
    assert!(h.hud_state().visible && h.panel_shown(PanelKey::Preview));

    crate::test_support::isolate_state_dir();
}

// Opening a row that is not the one showing (the selection a delete moved, say)
// reads it and asks for the rebuild that swaps it in.
#[test]
fn opening_a_row_that_is_not_showing_compiles_it() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let root = dir.path();
    let arena = write_world(root, "arena", &[prop_entry("crate_a")], 1_000);
    write_world(root, "lobby", &[prop_entry("desk")], 3_000);

    let mut h = start_hook(dir.path(), Some("lobby"));
    settle_rebuild(&mut h);
    // A selection with no preview behind it, as a delete leaves.
    h.worlds.preview = None;

    let mut world = World::new();
    let i = world_row_index(&h, "arena");
    h.apply_worlds_action(WorldsAction::Open(i), &mut world);

    assert_eq!(h.world_path, arena.to_string_lossy());
    assert_eq!(entry_names(&h), ["crate_a"]);
    assert!(
        h.rebuild_preview && h.rebuild_required,
        "the world showing was another one, so it is compiled"
    );
    assert!(!h.start_mode);

    crate::test_support::isolate_state_dir();
}

// Deleting the world being previewed drops the background back to the seeded
// empty scene and moves the selection to the row that took its place.
#[test]
fn deleting_the_previewed_world_falls_back_to_the_empty_scene() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let root = dir.path();
    write_world(root, "arena", &[prop_entry("crate_a")], 1_000);
    let lobby = write_world(root, "lobby", &[prop_entry("desk")], 3_000);

    let mut h = start_hook(dir.path(), Some("lobby"));
    settle_rebuild(&mut h);
    let mut world = World::new();
    let i = world_row_index(&h, "lobby");
    h.apply_worlds_action(WorldsAction::Delete(i), &mut world);
    press_modal(&mut h, &mut world, "Delete");

    assert!(!lobby.exists());
    assert!(h.entries.is_empty(), "the scene behind it is empty again");
    assert!(h.rebuild_preview, "and swaps in on the next frame");
    assert!(h.worlds.preview.is_none());
    assert_eq!(world_names(&h), ["arena"]);
    assert_eq!(
        selected_row(&h),
        Some(0),
        "the next row takes the selection"
    );
    assert_eq!(previewing_row(&h), None);
    assert!(h.start_mode && !h.dirty, "an empty scene is not an edit");

    // Deleting the last world leaves nothing selected rather than a stale row.
    let i = world_row_index(&h, "arena");
    h.apply_worlds_action(WorldsAction::Delete(i), &mut world);
    press_modal(&mut h, &mut world, "Delete");
    assert!(world_names(&h).is_empty());
    assert_eq!(selected_row(&h), None);

    crate::test_support::isolate_state_dir();
}

// Deleting a world the screen is not showing leaves the preview alone.
#[test]
fn deleting_another_world_leaves_the_preview_standing() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let root = dir.path();
    write_world(root, "arena", &[prop_entry("crate_a")], 1_000);
    write_world(root, "lobby", &[prop_entry("desk")], 3_000);

    let mut h = start_hook(dir.path(), Some("lobby"));
    settle_rebuild(&mut h);
    let mut world = World::new();
    let i = world_row_index(&h, "arena");
    h.apply_worlds_action(WorldsAction::Delete(i), &mut world);
    press_modal(&mut h, &mut world, "Delete");

    assert_eq!(entry_names(&h), ["desk"], "lobby is still what shows");
    assert!(!h.rebuild_preview, "and it was not rebuilt to say so");
    assert_eq!(previewing_row(&h), Some(0));

    crate::test_support::isolate_state_dir();
}

// A burst of row clicks costs one rebuild of the world last picked: the request
// is a flag the frame loop consumes once, not a queue of compiles.
#[test]
fn rapid_selections_coalesce_into_one_rebuild() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let root = dir.path();
    write_world(root, "arena", &[prop_entry("crate_a")], 1_000);
    write_world(root, "gym", &[prop_entry("bench")], 2_000);
    write_world(root, "lobby", &[prop_entry("desk")], 3_000);

    let mut h = start_hook(dir.path(), Some("lobby"));
    settle_rebuild(&mut h);
    let mut world = World::new();
    for name in ["arena", "gym", "arena"] {
        let i = world_row_index(&h, name);
        h.apply_worlds_action(WorldsAction::Select(i), &mut world);
    }

    assert_eq!(entry_names(&h), ["crate_a"], "the last pick is what shows");
    assert!(h.rebuild_preview && h.rebuild_required);
    // One swap answers the whole burst, and leaves nothing owed behind it.
    settle_rebuild(&mut h);
    assert!(!h.rebuild_preview);
    assert_eq!(previewing_row(&h), Some(world_row_index(&h, "arena")));

    crate::test_support::isolate_state_dir();
}

// The sidebar has no field of its own to lose: previewing a world is pure
// browsing, so a row click leaves the naming prompt's text (the only world name
// being typed anywhere) exactly where it was, whatever world it is shown in.
#[test]
fn previewing_a_world_leaves_a_half_typed_name_alone() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let root = dir.path();
    write_world(root, "arena", &[prop_entry("crate_a")], 1_000);
    write_world(root, "lobby", &[prop_entry("desk")], 3_000);

    let mut h = start_hook(dir.path(), Some("lobby"));
    settle_rebuild(&mut h);
    let mut world = World::new();
    h.open_prompt("Name it", Vec::new());
    set_world_name(&mut h, "half-typed");

    let i = world_row_index(&h, "arena");
    h.apply_worlds_action(WorldsAction::Select(i), &mut world);
    assert_eq!(h.modal.as_ref().unwrap().text, "half-typed");

    crate::test_support::isolate_state_dir();
}

// The switcher never previews: browsing the list must not swap the world the
// user is editing out from under them, so a row click opens behind the
// unsaved-changes guard.
#[test]
fn an_in_session_row_click_opens_behind_the_guard_and_never_previews() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let root = dir.path();
    let arena = write_world(root, "arena", &[prop_entry("crate_a")], 1_000);
    write_world(root, "lobby", &[prop_entry("desk")], 3_000);

    let mut h = hook_at(&arena, vec![prop_entry("crate_a")]);
    h.open[PanelKey::Worlds] = true;
    h.entries.push(prop_entry("crate_b"));
    h.mark_changed();
    h.rebuild_preview = false;
    assert!(!h.start_mode && h.dirty);

    // The switcher resolves a row press as an open, never as a selection.
    let i = world_row_index(&h, "lobby");
    assert_eq!(h.make_worlds_view().layout.mode, worlds::Mode::Session);
    assert_eq!(
        selected_row(&h),
        None,
        "the switcher has no selection model"
    );
    let row = UiEvent {
        name: worlds::panel::ROW.into(),
        element: None,
        value: Some(UiValue::Int(i32::try_from(i).unwrap())),
    };
    assert_eq!(h.worlds_event(&row), Some(WorldsAction::Open(i)));

    let mut world = World::new();
    h.apply_worlds_action(WorldsAction::Open(i), &mut world);
    assert!(h.modal.is_some(), "the unsaved edits are guarded first");
    assert_eq!(
        entry_names(&h),
        ["crate_a", "crate_b"],
        "nothing was staged"
    );
    assert!(!h.rebuild_preview, "and the live world was not touched");
    assert!(h.worlds.preview.is_none() && h.worlds.selected.is_none());

    press_modal(&mut h, &mut world, "Cancel");
    assert_eq!(h.world_path, arena.to_string_lossy());

    crate::test_support::isolate_state_dir();
}

// Routing on the start screen: a press on a row previews it, and a press
// anywhere off the panel reaches nothing behind it.
#[test]
fn start_mode_routing_reaches_the_panel_and_nothing_else() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let root = dir.path();
    write_world(root, "arena", &[prop_entry("crate_a")], 1_000);
    write_world(root, "lobby", &[prop_entry("desk")], 3_000);

    let mut h = start_hook(dir.path(), Some("lobby"));
    settle_rebuild(&mut h);
    let mut world = World::new();

    // Off the panel entirely, over the render: nothing selects, nothing is
    // staged, and no pick reaches the previewed world.
    click_at(&mut h, &mut world, VP[0] - 4.0, VP[1] - 4.0);
    assert_eq!(selected_row(&h), Some(0));
    assert!(!h.rebuild_preview);
    assert_eq!(selected(&h).len(), 0);

    // Where the top bar would be: it is not drawn, so its chips resolve nothing.
    click_at(&mut h, &mut world, VP[0] - 20.0, hud::BAR_H * 0.5);
    assert!(
        !h.open[PanelKey::View],
        "no View panel behind a bar that is not there"
    );
    assert!(h.start_mode && h.open[PanelKey::Worlds]);

    // A row press previews that world.
    let arena = world_row_index(&h, "arena");
    send(&mut h, &mut world, worlds::panel::ROW, arena);
    assert_eq!(entry_names(&h), ["crate_a"]);
    assert_eq!(previewing_row(&h), Some(arena));

    // A second press on the row now showing opens it, and normal routing comes
    // back with the session.
    settle_rebuild(&mut h);
    send(&mut h, &mut world, worlds::panel::ROW, arena);
    assert!(!h.start_mode);
    assert!(h.world_path.ends_with("arena.jsonl"));
    assert!(h.hud_state().visible);

    crate::test_support::isolate_state_dir();
}

// The row menu's Open is the other way to commit: it opens the row whose menu
// it belongs to, whether or not that row is the one showing.
#[test]
fn the_row_menus_open_commits_that_row() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let root = dir.path();
    write_world(root, "arena", &[prop_entry("crate_a")], 1_000);
    let lobby = write_world(root, "lobby", &[prop_entry("desk")], 3_000);

    let mut h = start_hook(dir.path(), Some("lobby"));
    settle_rebuild(&mut h);
    let mut world = World::new();

    send(&mut h, &mut world, worlds::panel::CONTEXT, 0);
    assert!(h.worlds.menu.is_some(), "a right click opened the menu");

    // Its first entry is Open.
    send(&mut h, &mut world, "editor.worlds.menu_pick", 0);

    assert_eq!(h.world_path, lobby.to_string_lossy());
    assert!(!h.start_mode && !h.open[PanelKey::Worlds]);
    assert!(
        !h.rebuild_preview,
        "the chip commits what is already showing"
    );

    crate::test_support::isolate_state_dir();
}

// `+` on the start screen drops straight into an untitled session: nothing is
// guarded on the way out (a preview has no unsaved edits to lose), and nothing
// is written until that world is saved and named.
#[test]
fn plus_leaves_the_start_screen_for_an_untitled_world() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let root = dir.path();
    write_world(root, "lobby", &[prop_entry("desk")], 3_000);

    let mut h = start_hook(dir.path(), Some("lobby"));
    settle_rebuild(&mut h);
    // A staged preview leaves entries standing that are not on the session's
    // own path: the guard must still not read that as unsaved work.
    let mut world = World::new();
    h.apply_worlds_action(WorldsAction::New, &mut world);

    assert!(h.modal.is_none(), "nothing to guard on the start screen");
    assert!(h.untitled && h.entries.is_empty() && !h.dirty);
    assert!(!h.start_mode, "the whole editor comes up on it");
    assert!(h.hud_state().visible);
    assert_eq!(world_names(&h), ["lobby"], "and nothing new is on disk");

    // Naming it at the first save is what puts it there.
    h.save();
    set_world_name(&mut h, "gym");
    press_modal(&mut h, &mut world, "Save");
    let created = root.join("gym.jsonl");
    assert!(created.exists());
    assert_eq!(h.world_path, created.to_string_lossy());
    assert!(!h.untitled);

    crate::test_support::isolate_state_dir();
}

// A preview whose compile failed is not showing, so opening its row compiles
// it the usual way instead of adopting the world the failure left up.
#[test]
fn a_failed_preview_is_not_adopted_when_its_row_is_opened() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let root = dir.path();
    write_world(root, "arena", &[prop_entry("crate_a")], 1_000);
    write_world(root, "lobby", &[prop_entry("desk")], 3_000);

    let mut h = start_hook(dir.path(), Some("lobby"));
    let lobby = world_row_index(&h, "lobby");
    h.preview_failed("bad shader\nsecond line");
    assert_eq!(previewing_row(&h), None, "the row loses its showing mark");
    assert_eq!(selected_row(&h), Some(lobby), "but stays selected");
    assert_eq!(h.make_worlds_view().status, Some("bad shader"));

    let mut world = World::new();
    h.open_from_start(lobby, &mut world);
    assert!(!h.start_mode);
    assert!(
        h.rebuild_required,
        "the open owes the compile the preview failed"
    );
    assert_eq!(entry_names(&h), ["desk"]);
    crate::test_support::isolate_state_dir();
}

// Opening a row whose preview has not landed yet (a second press before the
// compile ran) commits the session but still owes the compile: the world on
// screen stands for some other entries.
#[test]
fn opening_a_pending_preview_still_compiles_it() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let root = dir.path();
    write_world(root, "arena", &[prop_entry("crate_a")], 1_000);
    write_world(root, "lobby", &[prop_entry("desk")], 3_000);

    let mut h = start_hook(dir.path(), Some("lobby"));
    let arena = world_row_index(&h, "arena");
    h.select_world(arena);
    assert!(h.rebuild_preview, "the arena compile is pending");
    assert_ne!(h.world_entries, h.entries);

    let mut world = World::new();
    h.open_from_start(arena, &mut world);
    assert!(!h.start_mode);
    assert!(h.rebuild_required, "the live world is still the lobby");
    assert_eq!(entry_names(&h), ["crate_a"]);
    crate::test_support::isolate_state_dir();
}

// Deleting the world the screen opened on, before its deferred compile ran,
// drops that pick along with the row.
#[test]
fn deleting_the_boot_pick_before_it_compiles_drops_the_pick() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let root = dir.path();
    write_world(root, "arena", &[prop_entry("crate_a")], 1_000);
    write_world(root, "lobby", &[prop_entry("desk")], 3_000);

    let mut h = booting_hook(dir.path(), Some("lobby"));
    let lobby_path = h.worlds.rows[world_row_index(&h, "lobby")].path.clone();
    h.delete_world(&lobby_path);
    assert!(h.start_preview.is_none());

    h.start_drawn = EditorHook::START_PREVIEW_DELAY;
    h.drive_start_preview();
    assert_eq!(
        h.worlds.preview, None,
        "nothing is staged for a deleted world"
    );
    assert_eq!(world_names(&h), ["arena"]);
    assert_eq!(selected_row(&h), Some(0));
    crate::test_support::isolate_state_dir();
}
