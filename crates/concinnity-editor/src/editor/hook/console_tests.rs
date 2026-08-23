// src/editor/hook/console_tests.rs
//
// The Console panel's drive (`hook/console_edit.rs`): the log window's pinned
// tail and its scroll, the focus actions, the editing keys (submit and the
// /del ghost completion), and the build command's worker handoff. The command
// parsers themselves are tested in `editor/console.rs`, and the /add, /del,
// /snap, /dup, /floor dispatches beside the rest of the drive in `tests.rs`.

use super::*;
use crate::assets::InputKey;
use crate::test_support::isolate_state_dir;
use std::sync::atomic::Ordering;

fn hook(entries: Vec<serde_json::Value>) -> EditorHook {
    EditorHook::new("unused.jsonl".to_string(), entries)
}

// A world holding the console's command-line field.
fn console_world() -> World {
    let mut world = World::new();
    for id in console_panel::all_field_ids() {
        world.add_component(crate::assets::TextInput {
            asset_id: id,
            ..Default::default()
        });
    }
    world
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

// A fresh console is pinned to the tail, so the newest lines are the ones on
// screen however long the log grows.
#[test]
fn the_log_window_follows_the_tail_while_pinned() {
    let h = hook(Vec::new());
    let shown = console_panel::visible_lines(h.effective_size(PanelKey::Console)[1]);
    fill_log(&h, shown + 20);

    let (lines, total, first) = h.console_window();
    assert_eq!(total, shown + 20);
    assert_eq!(first, 20, "the window sits at the tail");
    assert_eq!(lines.len(), shown);
    assert_eq!(lines.last().unwrap().text, format!("line {}", shown + 19));
}

// Scrolling back through the log unpins the window; stepping forward onto the
// last line re-pins it, so the log resumes following new output on its own.
#[test]
fn scrolling_off_the_tail_unpins_and_returning_re_pins() {
    let mut h = hook(Vec::new());
    let shown = console_panel::visible_lines(h.effective_size(PanelKey::Console)[1]);
    fill_log(&h, shown + 10);
    assert!(h.console_pinned);

    h.scroll_console(-1.0);
    assert!(!h.console_pinned, "stepping back leaves the tail");
    let (_, _, first) = h.console_window();
    assert_eq!(first, 9);

    h.scroll_console(1.0);
    assert!(h.console_pinned, "stepping onto the last line re-pins");
    assert_eq!(h.console_window().2, 10);
}

// The scroll clamps at both ends of a log shorter than the window, where
// there is nothing to scroll at all.
#[test]
fn scrolling_stays_in_bounds_on_a_short_log() {
    let mut h = hook(Vec::new());
    fill_log(&h, 3);
    for _ in 0..8 {
        h.scroll_console(1.0);
    }
    assert_eq!(h.console_window().2, 0);
    for _ in 0..8 {
        h.scroll_console(-1.0);
    }
    assert_eq!(h.console_window().2, 0);
    assert!(h.console_pinned);
}

// Clicking the command line focuses it; clicking the surrounding chrome blurs
// it, so the keys stop being swallowed by a field the user left.
#[test]
fn the_panel_actions_focus_and_blur_the_command_line() {
    let mut h = hook(Vec::new());
    let mut world = console_world();

    h.apply_console_action(ConsoleAction::FocusInput, &mut world);
    assert!(h.console_focus);

    h.apply_console_action(ConsoleAction::Consume, &mut world);
    assert!(!h.console_focus);
}

// Enter submits the trimmed line, echoes it, and clears the field ready for
// the next command.
#[test]
fn enter_submits_the_line_and_clears_the_field() {
    let mut h = hook(Vec::new());
    let mut world = console_world();
    h.console_focus = true;
    widget::seed_field(&mut world, console_panel::INPUT, "  /help  ");

    h.console_keys(
        &mut world,
        &FrameInput {
            captured_key: Some(InputKey::Enter),
            ..Default::default()
        },
    );

    assert_eq!(widget::field_text(&world, console_panel::INPUT), "");
    let log = log_text(&h);
    assert!(log.contains("> /help"), "the line is echoed: {log}");
    assert!(log.lines().count() > 1, "/help answered: {log}");
}

// A blank submission is not a command: the echo would be noise.
#[test]
fn enter_on_a_blank_line_submits_nothing() {
    let mut h = hook(Vec::new());
    let mut world = console_world();
    h.console_focus = true;
    widget::seed_field(&mut world, console_panel::INPUT, "   ");

    h.console_keys(
        &mut world,
        &FrameInput {
            captured_key: Some(InputKey::Enter),
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
    h.console_focus = false;
    widget::seed_field(&mut world, console_panel::INPUT, "/help");

    h.console_keys(
        &mut world,
        &FrameInput {
            captured_key: Some(InputKey::Enter),
            ..Default::default()
        },
    );
    assert_eq!(h.console_sink.len(), 0);
    assert_eq!(widget::field_text(&world, console_panel::INPUT), "/help");
}

// Tab accepts the /del name completion, so a long asset name is one keypress.
#[test]
fn tab_accepts_the_del_ghost() {
    let mut h = hook(vec![
        serde_json::json!({"name":"lantern_post","type":"PointLight","args":{}}),
    ]);
    let mut world = console_world();
    h.console_focus = true;
    widget::seed_field(&mut world, console_panel::INPUT, "/del lant");
    assert_eq!(h.console_ghost(&world), "ern_post");

    h.console_keys(
        &mut world,
        &FrameInput {
            captured_key: Some(InputKey::Tab),
            ..Default::default()
        },
    );
    assert_eq!(
        widget::field_text(&world, console_panel::INPUT),
        "/del lantern_post"
    );
}

// Right accepts the ghost only from the end of the line; mid-line it is an
// ordinary caret move the text system owns.
#[test]
fn right_accepts_the_ghost_only_at_the_end_of_the_line() {
    let entries = vec![serde_json::json!({"name":"lantern_post","type":"PointLight","args":{}})];
    let key = FrameInput {
        captured_key: Some(InputKey::Right),
        ..Default::default()
    };

    let mut at_end = hook(entries.clone());
    let mut world = console_world();
    at_end.console_focus = true;
    widget::focus_field_with(&mut world, console_panel::INPUT, "/del lant");
    at_end.console_keys(&mut world, &key);
    assert_eq!(
        widget::field_text(&world, console_panel::INPUT),
        "/del lantern_post"
    );

    let mut mid = hook(entries);
    let mut world = console_world();
    mid.console_focus = true;
    widget::focus_field_with(&mut world, console_panel::INPUT, "/del lant");
    if let Some(t) = widget::input_mut(&mut world, console_panel::INPUT) {
        t.caret = 2;
    }
    mid.console_keys(&mut world, &key);
    assert_eq!(
        widget::field_text(&world, console_panel::INPUT),
        "/del lant"
    );
}

// With no name to complete there is no ghost, and the accept keys leave the
// line exactly as typed.
#[test]
fn the_accept_keys_are_inert_without_a_ghost() {
    let mut h = hook(Vec::new());
    let mut world = console_world();
    h.console_focus = true;
    widget::focus_field_with(&mut world, console_panel::INPUT, "/del nothing");
    assert_eq!(h.console_ghost(&world), "");

    h.console_keys(
        &mut world,
        &FrameInput {
            captured_key: Some(InputKey::Tab),
            ..Default::default()
        },
    );
    assert_eq!(
        widget::field_text(&world, console_panel::INPUT),
        "/del nothing"
    );
}

// One build at a time: a second /cook while the worker is still writing blobs
// is refused rather than racing it.
#[test]
fn a_second_build_is_refused_while_one_is_running() {
    let mut h = hook(vec![
        serde_json::json!({"name":"phys","type":"PhysicsConfig","args":{}}),
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
    let _guard = crate::test_support::lock();
    isolate_state_dir();
    let dir = tempfile::tempdir().expect("temp dir");
    let prev = std::env::current_dir().expect("cwd");
    std::env::set_current_dir(dir.path()).expect("enter temp cwd");

    let mut h = hook(vec![
        serde_json::json!({"name":"phys","type":"PhysicsConfig","args":{}}),
    ]);
    let mut world = console_world();
    h.run_console_line(&mut world, "/cook");
    assert!(log_text(&h).contains("cook started"));

    // Wait out the worker rather than the wall clock, so the cwd is restored
    // only once nothing is still writing through it.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    while h.console_build_running.load(Ordering::SeqCst) {
        assert!(std::time::Instant::now() < deadline, "the cook worker hung");
        std::thread::yield_now();
    }
    let log = log_text(&h);
    std::env::set_current_dir(prev).expect("restore cwd");

    assert!(
        log.contains("cook finished") || log.contains("cook failed"),
        "the worker reported an outcome: {log}"
    );
}

// The prism world the export command tests drive: an inline two-joint skinned
// mesh with one morph target and a live shape.
fn export_entries() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({"name":"prism","type":"SkinnedMesh","args":{
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
        serde_json::json!({"name":"shape","type":"CharacterShape","args":{
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
    h.selection.set(vec!["prism".to_string()]);
    let mut world = console_world();
    h.run_console_line(&mut world, "/export");
    assert!(log_text(&h).contains("export of 'prism' started"));

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
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
