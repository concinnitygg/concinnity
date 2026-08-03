// src/editor/hook/palette_tests.rs
//
// The command palette's drive (`hook/palette_edit.rs`): the shortcut that opens
// it under either platform modifier, the query mirrored off its field, the
// keyboard walk over the matches, and what committing a row does. The ranking
// and the providers are tested beside them in `editor/palette/`.

use super::*;
use crate::assets::Key;
use crate::editor::palette::{Category, PaletteAction};

fn hook(entries: Vec<serde_json::Value>) -> EditorHook {
    EditorHook::new("unused.jsonl".to_string(), entries)
}

// A world holding the palette's query field.
fn palette_world() -> World {
    let mut world = World::new_empty();
    for id in palette_panel::all_field_ids() {
        world.add_component(crate::assets::TextInput {
            asset_id: id,
            ..Default::default()
        });
    }
    world
}

fn chord(key: Key, ctrl: bool, cmd: bool) -> FrameInput {
    FrameInput {
        captured_key: Some(key),
        ctrl,
        cmd,
        ..Default::default()
    }
}

// Both platform modifiers open the palette: Ctrl on Windows and Linux, Command
// on macOS, where `cmd` is the only one the backend sets.
#[test]
fn either_platform_modifier_opens_the_palette() {
    for (ctrl, cmd) in [(true, false), (false, true)] {
        let mut h = hook(Vec::new());
        let mut world = palette_world();
        h.drive_palette_toggle(&chord(Key::K, ctrl, cmd), &mut world);
        assert!(h.palette_open, "ctrl={ctrl} cmd={cmd} did not open it");
        // The same chord closes it again.
        h.drive_palette_toggle(&chord(Key::K, ctrl, cmd), &mut world);
        assert!(!h.palette_open);
    }
}

// A bare K types a letter; only the modified chord is the shortcut.
#[test]
fn an_unmodified_k_leaves_the_palette_closed() {
    let mut h = hook(Vec::new());
    let mut world = palette_world();
    h.drive_palette_toggle(&chord(Key::K, false, false), &mut world);
    assert!(!h.palette_open);
}

// Opening blurs the field for one frame, so the keypress that opened it cannot
// also be typed into the fresh query.
#[test]
fn opening_blurs_the_query_for_one_frame() {
    let mut h = hook(Vec::new());
    let mut world = palette_world();
    h.drive_palette_toggle(&chord(Key::K, true, false), &mut world);
    assert!(h.palette_blur);
    assert!(!h.make_palette_view([0.0, 0.0]).focus, "blurred this frame");
}

// The empty query offers the panels and commands; typing narrows across the
// categories and re-homes the highlight to the best answer.
#[test]
fn typing_reranks_and_rehomes_the_highlight() {
    let mut h = hook(Vec::new());
    let mut world = palette_world();
    h.drive_palette_toggle(&chord(Key::K, true, false), &mut world);
    let opened = h.palette_matches.len();
    assert!(opened > 0, "the launch list is not empty");

    h.palette_pick = 3;
    widget::seed_field(&mut world, palette_panel::INPUT, "cook");
    h.sample_palette_query(&world);
    assert_eq!(
        h.palette_pick, 0,
        "a narrowed list starts at its best answer"
    );
    assert!(h.palette_matches.len() < opened);
    let first = &h.palette_items[h.palette_matches[0]];
    assert_eq!(first.label, "/cook");
}

// Up / Down walk the matches and stop at either end rather than wrapping.
#[test]
fn the_arrows_walk_the_matches_and_stop_at_the_ends() {
    let mut h = hook(Vec::new());
    let mut world = palette_world();
    h.drive_palette_toggle(&chord(Key::K, true, false), &mut world);

    h.palette_keys(&mut world, &chord(Key::Down, false, false));
    assert_eq!(h.palette_pick, 1);
    h.palette_keys(&mut world, &chord(Key::Up, false, false));
    assert_eq!(h.palette_pick, 0);
    h.palette_keys(&mut world, &chord(Key::Up, false, false));
    assert_eq!(h.palette_pick, 0, "stops at the top");
}

// Committing a panel row opens that panel and closes the palette.
#[test]
fn committing_a_panel_row_opens_it() {
    let mut h = hook(Vec::new());
    let mut world = palette_world();
    h.drive_palette_toggle(&chord(Key::K, true, false), &mut world);
    let at = h
        .palette_matches
        .iter()
        .position(|&i| h.palette_items[i].action == PaletteAction::OpenPanel(PanelKey::Variables))
        .expect("the Variables panel is a palette row");
    h.palette_pick = at;

    h.palette_keys(&mut world, &chord(Key::Enter, false, false));
    assert!(!h.palette_open, "committing closes the palette");
    assert!(h.variables_open, "the panel opened");
    assert_eq!(
        h.panel_order.last().copied(),
        Some(PanelKey::Variables),
        "and it is frontmost"
    );
}

// A command taking arguments seeds command mode instead of running: the
// palette stays up with the name typed, ready for its arguments.
#[test]
fn committing_an_argument_command_seeds_command_mode() {
    let mut h = hook(Vec::new());
    let mut world = palette_world();
    h.drive_palette_toggle(&chord(Key::K, true, false), &mut world);
    let at = h
        .palette_matches
        .iter()
        .position(|&i| h.palette_items[i].label == "/add")
        .expect("/add is a palette row");
    h.palette_pick = at;

    h.palette_keys(&mut world, &chord(Key::Enter, false, false));
    assert!(h.palette_open, "the palette stays up for the arguments");
    assert_eq!(widget::field_text(&world, palette_panel::INPUT), "/add ");
}

// A committed row leads the next empty-query list, so repeating recent work is
// one keystroke away.
#[test]
fn a_commit_is_remembered_for_the_next_launch_list() {
    let mut h = hook(Vec::new());
    let mut world = palette_world();
    h.drive_palette_toggle(&chord(Key::K, true, false), &mut world);
    let at = h
        .palette_matches
        .iter()
        .position(|&i| h.palette_items[i].action == PaletteAction::OpenPanel(PanelKey::Variables))
        .expect("the Variables panel is a palette row");
    let label = h.palette_items[h.palette_matches[at]].label.clone();
    h.palette_pick = at;
    h.palette_keys(&mut world, &chord(Key::Enter, false, false));

    h.drive_palette_toggle(&chord(Key::K, true, false), &mut world);
    let first = &h.palette_items[h.palette_matches[0]];
    assert_eq!(first.label, label, "the last commit leads the list");
}

// Escape closes the palette through the global escape drive, and a click
// outside it dismisses while claiming the press, so nothing underneath is
// picked by the dismissal.
#[test]
fn a_press_outside_dismisses_without_reaching_the_world() {
    let mut h = hook(Vec::new());
    let mut world = palette_world();
    let vp = [1280.0, 720.0];
    h.drive_palette_toggle(&chord(Key::K, true, false), &mut world);

    let o = h.origin(PanelKey::Palette, vp);
    let inside = FrameInput {
        mouse_x: o[0] + 20.0,
        mouse_y: o[1] + 10.0,
        ..Default::default()
    };
    assert!(
        !h.route_palette_dismiss(&inside, vp),
        "a press on the palette is left to its own hit test"
    );
    assert!(h.palette_open);

    let outside = FrameInput {
        mouse_x: o[0] - 40.0,
        mouse_y: o[1] + 10.0,
        ..Default::default()
    };
    assert!(
        h.route_palette_dismiss(&outside, vp),
        "the press is claimed"
    );
    assert!(!h.palette_open);
}

// The world's assets reach the palette: a behavior is offered as an asset row
// that opens its own panel, everything else as an entity row.
#[test]
fn world_assets_become_rows_routed_by_type() {
    let entries = vec![
        serde_json::json!({"name": "greeter", "type": "Behavior", "args": {"on": "tick", "do": []}}),
    ];
    let mut h = hook(entries);
    let mut world = palette_world();
    h.drive_palette_toggle(&chord(Key::K, true, false), &mut world);

    let row = h
        .palette_items
        .iter()
        .find(|it| it.label == "greeter")
        .expect("the authored behavior is a palette row");
    assert_eq!(row.category, Category::Asset);
    assert_eq!(row.action, PaletteAction::OpenAsset("greeter".to_string()));
}
