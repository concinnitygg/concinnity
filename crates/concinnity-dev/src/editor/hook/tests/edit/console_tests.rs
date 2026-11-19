//! The Console panel's drive (`hook/edit/console.rs`): the log's pinned tail,
//! the command line's focus, the editing keys (submit and the /del ghost
//! completion), the build command's worker handoff, and what the /add, /del,
//! /snap and /dup commands do to the working entries. The command parsers
//! themselves are tested beside them in `editor/panels/console.rs`.

use concinnity_cook::authoring::registry::RegisteredType;
use concinnity_engine::components::{
    FrameInput, InputKey, KeyEvent, KeyMods, KeyPress, List, TextField,
};
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;

use crate::editor::hook::tests::fixtures::{entry, hook, select, world_with_fields};
use crate::editor::hook::{EditorHook, declared_id, entry_type};
use crate::editor::panels::console;
use crate::editor::panels::console_panel;
use crate::editor::panels::registry::PanelKey;
use crate::editor::ui::{BuiltUi, UiBuilder, field};
use crate::frame_hook::FrameHook;
use crate::test_support::isolate_state_dir;
use std::sync::atomic::Ordering;

// A world holding the built Console panel.
fn console_world() -> World {
    let mut world = World::new();
    let panel = {
        let mut ui = UiBuilder::new(&mut world, AssetId(1), None);
        console_panel::build(&mut ui).unwrap()
    };
    let mut built = BuiltUi::default();
    built.panels[PanelKey::Console] = Some(panel);
    world.insert_resource(built);
    world
}

fn field(world: &World) -> Option<AssetId> {
    EditorHook::console_input(world)
}

fn seed(world: &mut World, text: &str) {
    let id = field(world);
    field::set_text(world, id, text);
}

fn line(world: &World) -> String {
    field::text(world, field(world))
}

fn fill_log(h: &EditorHook, n: usize) {
    for i in 0..n {
        h.console_sink.info(&format!("line {i}"));
    }
}

fn log_text(h: &EditorHook) -> String {
    h.console_sink
        .window(0, 512)
        .iter()
        .map(|l| l.text.clone())
        .collect::<Vec<_>>()
        .join("\n")
}

// A pinned console shows the newest lines however long the log grows; a
// scroll away from the tail unpins it, and one back onto it re-pins.
#[test]
fn the_log_follows_its_tail_while_pinned() {
    let mut h = hook(Vec::new());
    let mut world = console_world();
    let panel = world
        .resource::<BuiltUi>()
        .unwrap()
        .panel(PanelKey::Console)
        .unwrap()
        .clone();
    fill_log(&h, 40);
    let sync = |h: &EditorHook, world: &mut World| {
        let view = h.console_view("");
        let log = || h.console_sink.read(console_panel::items);
        console_panel::sync(&view, log, world, &panel);
    };
    sync(&h, &mut world);
    let list = |w: &World| w.get_by_id::<List>(panel.slots[0]).unwrap().clone();
    let tail = list(&world).max_scroll(40);
    assert!(
        tail > 0.0 && list(&world).scroll == tail,
        "the list sits at the tail"
    );

    world.get_mut_by_id::<List>(panel.slots[0]).unwrap().scroll = 0.0;
    h.observe_console(&world, &panel);
    assert!(!h.console.pinned, "scrolling back leaves the tail");
    fill_log(&h, 1);
    sync(&h, &mut world);
    assert_eq!(list(&world).scroll, 0.0, "an unpinned log stays put");

    world.get_mut_by_id::<List>(panel.slots[0]).unwrap().scroll = list(&world).max_scroll(41);
    h.observe_console(&world, &panel);
    assert!(h.console.pinned, "returning to the tail re-pins");
}

// A click that focuses the command line, or leaves it, reaches the hook.
#[test]
fn the_command_lines_focus_follows_the_widget() {
    let mut h = hook(Vec::new());
    let mut world = console_world();
    let panel = world
        .resource::<BuiltUi>()
        .unwrap()
        .panel(PanelKey::Console)
        .unwrap()
        .clone();
    h.focus_panel(PanelKey::Console);
    let id = field(&world).unwrap();
    world.get_mut_by_id::<TextField>(id).unwrap().focused = true;
    h.observe_console(&world, &panel);
    assert!(h.console.focus);
    world.get_mut_by_id::<TextField>(id).unwrap().focused = false;
    h.observe_console(&world, &panel);
    assert!(!h.console.focus);
}

// Enter submits the trimmed line, echoes it, and clears the field ready for
// the next command.
#[test]
fn enter_submits_the_line_and_clears_the_field() {
    let mut h = hook(Vec::new());
    let mut world = console_world();
    h.console.focus = true;
    seed(&mut world, "  /help  ");

    h.console_keys(
        &mut world,
        &FrameInput {
            key_events: vec![KeyEvent::press(InputKey::Enter)],
            ..Default::default()
        },
    );

    assert_eq!(line(&world), "");
    let log = log_text(&h);
    assert!(log.contains("> /help"), "the line is echoed: {log}");
    assert!(log.lines().count() > 1, "/help answered: {log}");
}

// A blank submission is not a command: the echo would be noise.
#[test]
fn enter_on_a_blank_line_submits_nothing() {
    let mut h = hook(Vec::new());
    let mut world = console_world();
    h.console.focus = true;
    seed(&mut world, "   ");

    h.console_keys(
        &mut world,
        &FrameInput {
            key_events: vec![KeyEvent::press(InputKey::Enter)],
            ..Default::default()
        },
    );
    assert_eq!(h.console_sink.len(), 0);
}

// The keys are only the focused command line's; an unfocused console must not
// eat the editor's shortcuts.
#[test]
fn the_keys_are_ignored_while_the_command_line_is_unfocused() {
    let mut h = hook(Vec::new());
    let mut world = console_world();
    h.console.focus = false;
    seed(&mut world, "/help");

    h.console_keys(
        &mut world,
        &FrameInput {
            key_events: vec![KeyEvent::press(InputKey::Enter)],
            ..Default::default()
        },
    );
    assert_eq!(h.console_sink.len(), 0);
    assert_eq!(line(&world), "/help");
}

// Tab accepts the /del name completion, so a long asset name is one keypress.
#[test]
fn tab_accepts_the_del_ghost() {
    let mut h = hook(vec![
        serde_json::json!({"type":"PointLight","args":{"$id":"lantern_post"}}),
    ]);
    let mut world = console_world();
    h.console.focus = true;
    seed(&mut world, "/del lant");
    assert_eq!(h.console_ghost(&world), "ern_post");

    h.console_keys(
        &mut world,
        &FrameInput {
            key_events: vec![KeyEvent::press(InputKey::Tab)],
            ..Default::default()
        },
    );
    assert_eq!(line(&world), "/del lantern_post");
}

// Right accepts the ghost only from the end of the line; mid-line it is an
// ordinary caret move the text system owns.
#[test]
fn right_accepts_the_ghost_only_at_the_end_of_the_line() {
    let entries = vec![serde_json::json!({"type":"PointLight","args":{"$id":"lantern_post"}})];
    let key = FrameInput {
        key_events: vec![KeyEvent::press(InputKey::Right)],
        ..Default::default()
    };

    let mut at_end = hook(entries.clone());
    let mut world = console_world();
    at_end.console.focus = true;
    seed(&mut world, "/del lant");
    at_end.console_keys(&mut world, &key);
    assert_eq!(line(&world), "/del lantern_post");

    let mut mid = hook(entries);
    let mut world = console_world();
    mid.console.focus = true;
    seed(&mut world, "/del lant");
    if let Some(t) = field(&world).and_then(|id| world.get_mut_by_id::<TextField>(id)) {
        t.caret = 2;
    }
    mid.console_keys(&mut world, &key);
    assert_eq!(line(&world), "/del lant");
}

// With no name to complete there is no ghost, and the accept keys leave the
// line exactly as typed.
#[test]
fn the_accept_keys_are_inert_without_a_ghost() {
    let mut h = hook(Vec::new());
    let mut world = console_world();
    h.console.focus = true;
    seed(&mut world, "/del nothing");
    assert_eq!(h.console_ghost(&world), "");

    h.console_keys(
        &mut world,
        &FrameInput {
            key_events: vec![KeyEvent::press(InputKey::Tab)],
            ..Default::default()
        },
    );
    assert_eq!(line(&world), "/del nothing");
}

// One build at a time: a second /cook while the worker is still writing blobs
// is refused rather than racing it.
#[test]
fn a_second_build_is_refused_while_one_is_running() {
    let mut h = hook(vec![
        serde_json::json!({"type":"PhysicsConfig","args":{"$id":"phys"}}),
    ]);
    h.console_build_running.store(true, Ordering::SeqCst);

    let mut world = console_world();
    h.run_console_line(&mut world, "/cook");

    assert!(log_text(&h).contains("cook already running"));
    assert!(
        h.console_build_running.load(Ordering::SeqCst),
        "the refusal leaves the running flag alone"
    );
}

// The build command hands off to a worker: it reports the start on the frame
// thread, and the worker clears the running flag and reports the outcome so
// the frame loop never stalls on a cook.
#[test]
fn the_build_command_runs_on_a_worker_and_reports_back() {
    let _state = crate::test_support::in_temp_cwd();
    isolate_state_dir();

    let mut h = hook(vec![
        serde_json::json!({"type":"PhysicsConfig","args":{"$id":"phys"}}),
    ]);
    let mut world = console_world();
    h.run_console_line(&mut world, "/cook");
    assert!(log_text(&h).contains("cook started"));

    // Wait out the worker rather than the wall clock, so the cwd is restored
    // only once nothing is still writing through it.
    let deadline = std::time::Instant::now() + std::time::Duration::from_mins(2);
    while h.console_build_running.load(Ordering::SeqCst) {
        assert!(std::time::Instant::now() < deadline, "the cook worker hung");
        std::thread::yield_now();
    }
    let log = log_text(&h);

    assert!(
        log.contains("cook finished") || log.contains("cook failed"),
        "the worker reported an outcome: {log}"
    );
}

// The prism world the export command tests drive: an inline two-joint skinned
// mesh with one morph target and a live shape.
fn export_entries() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({"type":"SkinnedMesh","args":{
            "$id":"prism",
            "vertices":[
                {"pos":[0.0,0.0,0.0],"joints":[0,0,0,0],"weights":[1.0,0.0,0.0,0.0]},
                {"pos":[1.0,0.0,0.0],"joints":[0,0,0,0],"weights":[1.0,0.0,0.0,0.0]},
                {"pos":[0.0,1.0,0.0],"joints":[1,0,0,0],"weights":[1.0,0.0,0.0,0.0]}],
            "indices":[0,1,2],
            "skeleton":[{"name":"root","parent":-1},
                        {"name":"tip","parent":0,"translation":[0.0,1.0,0.0]}],
            "morph_target_names":["wide"],
            "morph_deltas":[{"position":[1.0,0.0,0.0]},{"position":[1.0,0.0,0.0]},
                            {"position":[1.0,0.0,0.0]}],
            "scale":[1.0,1.0,1.0]}}),
        serde_json::json!({"type":"CharacterShape","args":{
            "$id":"shape",
            "target":"prism","sliders":[{"name":"wide","value":0.5}]}}),
    ]
}

// /export with nothing selected and no name has nothing to act on.
#[test]
fn export_without_a_name_or_selection_reports_an_error() {
    let mut h = hook(export_entries());
    let mut world = console_world();
    h.run_console_line(&mut world, "/export");

    assert!(log_text(&h).contains("select a skinned mesh"));
    assert!(
        !h.console_build_running.load(Ordering::SeqCst),
        "no worker was spawned"
    );
}

// The one-cook-at-a-time guard also refuses an export mid-cook.
#[test]
fn an_export_is_refused_while_a_cook_runs() {
    let mut h = hook(export_entries());
    h.console_build_running.store(true, Ordering::SeqCst);

    let mut world = console_world();
    h.run_console_line(&mut world, "/export prism");

    assert!(log_text(&h).contains("cook already running"));
    assert!(h.console_build_running.load(Ordering::SeqCst));
}

// The export command hands off to a worker like the build command, resolves
// the selection when no name is given, and writes `<name>.glb` beside the
// world file.
#[test]
fn the_export_command_runs_on_a_worker_and_writes_the_file() {
    let _guard = crate::test_support::lock();
    isolate_state_dir();
    let dir = tempfile::tempdir().expect("temp dir");
    let world_path = dir.path().join("world.jsonl");

    let mut h = EditorHook::new(world_path.to_string_lossy().into_owned(), export_entries());
    select(&mut h, &["prism"]);
    let mut world = console_world();
    h.run_console_line(&mut world, "/export");
    assert!(log_text(&h).contains("export of 'prism' started"));

    let deadline = std::time::Instant::now() + std::time::Duration::from_mins(2);
    while h.console_build_running.load(Ordering::SeqCst) {
        assert!(
            std::time::Instant::now() < deadline,
            "the export worker hung"
        );
        std::thread::yield_now();
    }
    let log = log_text(&h);
    assert!(log.contains("Exported"), "{log}");
    let out = dir.path().join("prism.glb");
    let bytes = std::fs::read(&out).expect("the .glb landed beside the world file");
    assert_eq!(&bytes[0..4], b"glTF");
}

// /snap drives the same settings the Preview panel rows toggle.
#[test]
fn console_snap_adjusts_and_reports_the_settings() {
    let mut world = World::new();
    let mut h = hook(vec![]);
    h.run_console_line(&mut world, "/snap 0.25");
    assert!(h.snap.translate.enabled, "a step also enables the family");
    assert_eq!(h.snap.translate.step, 0.25);
    h.run_console_line(&mut world, "/snap rot 45");
    assert!(h.snap.rotate.enabled);
    assert_eq!(h.snap.rotate.step, 45.0);
    h.run_console_line(&mut world, "/snap off");
    assert!(!h.snap.translate.enabled && !h.snap.rotate.enabled);
    assert_eq!(h.snap.translate.step, 0.25, "off keeps the steps");
    let lines = h.console_sink.window(0, 100);
    assert!(
        lines
            .last()
            .is_some_and(|l| l.text == "snap: move 0.25 m (off), rotate 45 deg (off)"),
        "each /snap reports the resulting state"
    );
}

// Console commands mutate the working entries like their panel counterparts,
// and everything they do lands in the log sink.
#[test]
fn console_commands_edit_the_working_entries() {
    let mut world = World::new();
    let mut h = hook(Vec::new());

    h.run_console_line(&mut world, "/add PhysicsConfig phys");
    assert_eq!(h.entries.len(), 1);
    assert_eq!(declared_id(&h.entries[0]), Some("phys"));
    assert_eq!(
        entry_type(&h.entries[0]),
        Some(RegisteredType::PhysicsConfig)
    );
    assert!(h.dirty, "an added entry marks the world dirty");

    h.run_console_line(&mut world, "/del phys");
    assert!(h.entries.is_empty());

    h.run_console_line(&mut world, "/del ghost");
    h.run_console_line(&mut world, "just a note");
    let lines = h.console_sink.window(0, 64);
    let texts: Vec<&str> = lines.iter().map(|l| l.text.as_str()).collect();
    assert!(
        texts.contains(&"added 'phys' (PhysicsConfig)"),
        "got: {texts:?}"
    );
    assert!(
        texts.contains(&"removed 'phys' (PhysicsConfig)"),
        "got: {texts:?}"
    );
    assert!(
        texts.contains(&"no authored asset named 'ghost'"),
        "got: {texts:?}"
    );
    // A bare line echoes (as does every submitted command line).
    assert!(texts.contains(&"> just a note"), "got: {texts:?}");
    assert!(
        lines.iter().any(|l| l.severity == console::Severity::Error),
        "the missing name reports as an error"
    );
}

// Backtick opens the console focused-but-blurred for that frame (so the text
// system cannot type the backtick into the command line), closes it again on
// the next press, and stands down entirely while another field is typing.
#[test]
fn backtick_toggles_the_console_with_a_one_frame_blur() {
    let mut h = hook(Vec::new());
    let mut world = world_with_fields();
    let input = FrameInput {
        key_events: vec![KeyEvent::press(InputKey::Backtick)],
        ..Default::default()
    };

    h.drive_console_toggle(&input, &mut world);
    assert!(h.open[PanelKey::Console] && h.console.focus && h.console.blur);
    assert_eq!(h.panel_order.last(), Some(&PanelKey::Console));
    assert!(
        !h.console_view("").focus,
        "the opening frame never asserts field focus"
    );
    // The next frame clears the blur (the tick does this) and focus asserts.
    h.console.blur = false;
    assert!(h.console_view("").focus);

    h.drive_console_toggle(&input, &mut world);
    assert!(
        !h.open[PanelKey::Console] && !h.console.focus,
        "second press closes"
    );

    // While another text field is focused, backtick is just a character.
    crate::editor::hook::tests::fixtures::focus_story(&mut h);
    h.drive_console_toggle(&input, &mut world);
    assert!(!h.open[PanelKey::Console]);
}

// A held backtick opens the console once; its auto-repeat does not close it.
#[test]
fn a_held_backtick_toggles_once() {
    let mut h = hook(Vec::new());
    let mut world = world_with_fields();
    let press = |repeat| FrameInput {
        key_events: vec![KeyEvent::Press(KeyPress {
            repeat,
            ..KeyPress::new(InputKey::Backtick, KeyMods::NONE)
        })],
        ..Default::default()
    };
    h.drive_console_toggle(&press(false), &mut world);
    assert!(h.open[PanelKey::Console]);
    h.drive_console_toggle(&press(true), &mut world);
    assert!(h.open[PanelKey::Console], "the repeat leaves it open");
}

// The /del ghost completes against authored names, and Tab accepts it into
// the command line.
#[test]
fn console_ghost_completes_del_names_and_tab_accepts() {
    let mut h = hook(vec![entry("cube_red", "Prop")]);
    let mut world = console_world();
    seed(&mut world, "/del cu");

    assert_eq!(h.console_ghost(&world), "be_red");
    h.console.focus = true;
    let tab = FrameInput {
        key_events: vec![KeyEvent::press(InputKey::Tab)],
        ..Default::default()
    };
    h.console_keys(&mut world, &tab);
    assert_eq!(line(&world), "/del cube_red");
    // With the name complete there is nothing left to ghost.
    assert_eq!(h.console_ghost(&world), "");
}

// The full tick path of a backtick open: the first frame shows the command
// line unfocused (so the text system cannot type the '`' into it), the next
// frame asserts focus.
#[test]
fn tick_opens_the_console_blurred_then_focuses() {
    let _guard = crate::test_support::lock();
    isolate_state_dir();
    let mut world = World::new();
    crate::editor::install::editor_ui(&mut world);
    world.add_component(FrameInput {
        key_events: vec![KeyEvent::press(InputKey::Backtick)],
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    let mut h = hook(Vec::new());

    h.tick(&mut world);
    assert!(h.open[PanelKey::Console]);
    let input = world
        .get_by_id::<TextField>(field(&world).unwrap())
        .unwrap();
    assert!(!input.focused, "the opening frame leaves the field blurred");

    // Next frame, no key held: focus asserts.
    for i in world.query_mut::<FrameInput>() {
        i.key_events.clear();
    }
    h.tick(&mut world);
    let input = world
        .get_by_id::<TextField>(field(&world).unwrap())
        .unwrap();
    assert!(input.focused);
}

// The completion follows the command line and the authored names as they stand
// this frame: none once Tab has taken it, and a fresh one when a name arrives.
#[test]
fn the_ghost_follows_the_line_and_the_names_this_frame() {
    let mut h = hook(vec![entry("cube_red", "Prop")]);
    let mut world = console_world();
    h.console.focus = true;
    seed(&mut world, "/del cu");
    assert_eq!(h.console_ghost_now(&world), "be_red");
    h.console_keys(
        &mut world,
        &FrameInput {
            key_events: vec![KeyEvent::press(InputKey::Tab)],
            ..Default::default()
        },
    );
    assert_eq!(h.console_ghost_now(&world), "", "taken by the Tab");

    seed(&mut world, "/del wa");
    assert_eq!(h.console_ghost_now(&world), "");
    h.entries.push(entry("wall", "Prop"));
    assert_eq!(h.console_ghost_now(&world), "ll", "a new name completes");
}
