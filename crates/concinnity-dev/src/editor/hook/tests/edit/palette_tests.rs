//! The command palette's drive (`hook/edit/palette.rs`): the shortcut that opens
//! it under either platform modifier, the query typed into its field, the
//! keyboard walk over the matches, and what committing a row does. The ranking
//! and the providers are tested beside them in `editor/palette/`.

use concinnity_engine::components::{
    FrameInput, InputKey, KeyEvent, KeyMods, KeyPress, Layout, TextField, UiEvent,
};
use concinnity_engine::ecs::{Ref, World};

use crate::editor::hook::EditorHook;
use crate::editor::install;
use crate::editor::palette::{Category, PaletteAction};
use crate::editor::panels::registry::{self, PanelKey};
use crate::editor::ui::BuiltUi;

fn hook(entries: Vec<serde_json::Value>) -> EditorHook {
    EditorHook::new("unused.jsonl".to_string(), entries)
}

fn chord(key: InputKey, ctrl: bool, cmd: bool) -> FrameInput {
    FrameInput {
        key_events: vec![KeyEvent::Press(KeyPress::new(
            key,
            KeyMods {
                ctrl,
                cmd,
                ..KeyMods::NONE
            },
        ))],
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
        h.drive_palette_toggle(&chord(InputKey::K, ctrl, cmd));
        assert!(
            h.open[PanelKey::Palette],
            "ctrl={ctrl} cmd={cmd} did not open it"
        );
        // The same chord closes it again.
        h.drive_palette_toggle(&chord(InputKey::K, ctrl, cmd));
        assert!(!h.open[PanelKey::Palette]);
    }
}

// A bare K types a letter; only the modified chord is the shortcut.
#[test]
fn an_unmodified_k_leaves_the_palette_closed() {
    let mut h = hook(Vec::new());
    h.drive_palette_toggle(&chord(InputKey::K, false, false));
    assert!(!h.open[PanelKey::Palette]);
}

// Opening blurs the field for one frame, so the keypress that opened it cannot
// also be typed into the fresh query.
#[test]
fn opening_blurs_the_query_for_one_frame() {
    let mut h = hook(Vec::new());
    h.drive_palette_toggle(&chord(InputKey::K, true, false));
    assert!(h.palette.blur);
    assert!(!h.make_palette_view().focus, "blurred this frame");
}

// The empty query offers the panels and commands; typing narrows across the
// categories and re-homes the highlight to the best answer.
#[test]
fn typing_reranks_and_rehomes_the_highlight() {
    let mut h = hook(Vec::new());
    h.drive_palette_toggle(&chord(InputKey::K, true, false));
    let opened = h.palette.matches.len();
    assert!(opened > 0, "the launch list is not empty");

    h.palette.pick = 3;
    let mut built = World::new();
    install::editor_ui(&mut built);
    let panel = built
        .resource::<BuiltUi>()
        .and_then(|b| b.panel(PanelKey::Palette))
        .cloned()
        .unwrap();
    if let Some(f) = built.get_mut_by_id::<TextField>(panel.slots[0]) {
        f.content = "cook".into();
    }
    h.observe_palette(&built, &panel);
    assert_eq!(
        h.palette.pick, 0,
        "a narrowed list starts at its best answer"
    );
    assert!(h.palette.matches.len() < opened);
    let first = &h.palette.items[h.palette.matches[0]];
    assert_eq!(first.label, "/cook");
}

// Up / Down walk the matches and stop at either end rather than wrapping.
#[test]
fn the_arrows_walk_the_matches_and_stop_at_the_ends() {
    let mut h = hook(Vec::new());
    let mut world = World::new();
    h.drive_palette_toggle(&chord(InputKey::K, true, false));

    h.palette_keys(&mut world, &chord(InputKey::Down, false, false));
    assert_eq!(h.palette.pick, 1);
    h.palette_keys(&mut world, &chord(InputKey::Up, false, false));
    assert_eq!(h.palette.pick, 0);
    h.palette_keys(&mut world, &chord(InputKey::Up, false, false));
    assert_eq!(h.palette.pick, 0, "stops at the top");
}

// Committing a panel row opens that panel and closes the palette.
#[test]
fn committing_a_panel_row_opens_it() {
    let mut h = hook(Vec::new());
    let mut world = World::new();
    h.drive_palette_toggle(&chord(InputKey::K, true, false));
    let at = h
        .palette
        .matches
        .iter()
        .position(|&i| h.palette.items[i].action == PaletteAction::OpenPanel(PanelKey::Variables))
        .expect("the Variables panel is a palette row");
    h.palette.pick = at;

    h.palette_keys(&mut world, &chord(InputKey::Enter, false, false));
    assert!(!h.open[PanelKey::Palette], "committing closes the palette");
    assert!(h.open[PanelKey::Variables], "the panel opened");
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
    let mut world = World::new();
    h.drive_palette_toggle(&chord(InputKey::K, true, false));
    let at = h
        .palette
        .matches
        .iter()
        .position(|&i| h.palette.items[i].label == "/add")
        .expect("/add is a palette row");
    h.palette.pick = at;

    h.palette_keys(&mut world, &chord(InputKey::Enter, false, false));
    assert!(
        h.open[PanelKey::Palette],
        "the palette stays up for the arguments"
    );
    assert_eq!(h.palette.query, "/add ", "the field shows it next frame");
}

// A committed row leads the next empty-query list, so repeating recent work is
// one keystroke away.
#[test]
fn a_commit_is_remembered_for_the_next_launch_list() {
    let mut h = hook(Vec::new());
    let mut world = World::new();
    h.drive_palette_toggle(&chord(InputKey::K, true, false));
    let at = h
        .palette
        .matches
        .iter()
        .position(|&i| h.palette.items[i].action == PaletteAction::OpenPanel(PanelKey::Variables))
        .expect("the Variables panel is a palette row");
    let label = h.palette.items[h.palette.matches[at]].label.clone();
    h.palette.pick = at;
    h.palette_keys(&mut world, &chord(InputKey::Enter, false, false));

    h.drive_palette_toggle(&chord(InputKey::K, true, false));
    let first = &h.palette.items[h.palette.matches[0]];
    assert_eq!(first.label, label, "the last commit leads the list");
}

// A press outside the open palette is the engine's dismissal: the hook routes
// it to nothing, and the event it sends closes the palette.
#[test]
fn a_press_outside_dismisses_without_reaching_the_world() {
    let mut h = hook(Vec::new());
    let mut world = World::new();
    install::editor_ui(&mut world);
    h.drive_palette_toggle(&chord(InputKey::K, true, false));
    let root = world
        .resource::<BuiltUi>()
        .and_then(|b| b.panel(PanelKey::Palette))
        .unwrap()
        .root;
    if let Some(l) = world.get_mut_by_id::<Layout>(root) {
        l.visible = true;
        l.rect = Some([300.0, 100.0, 400.0, 300.0]);
    }
    // A second transient layout of the editor's, and one the world authored,
    // which is the world's own business.
    let screen = world
        .resource::<BuiltUi>()
        .and_then(|b| b.panel(PanelKey::Palette))
        .and_then(|p| p.screen)
        .unwrap();
    world.push(Layout {
        screen: Some(Ref::new(screen)),
        on_dismiss: "other".into(),
        rect: Some([800.0, 500.0, 100.0, 100.0]),
        ..Default::default()
    });
    world.push(Layout {
        on_dismiss: "the world's".into(),
        rect: Some([0.0, 0.0, 1280.0, 720.0]),
        ..Default::default()
    });
    let press = |x, y| FrameInput {
        mouse_x: x,
        mouse_y: y,
        left_click: true,
        ..Default::default()
    };
    let dismisses = |x, y| h.press_dismisses(&world, &press(x, y));
    assert!(
        !dismisses(320.0, 120.0),
        "a press on the palette is its own"
    );
    assert!(
        !dismisses(850.0, 550.0),
        "and one on the other is the other's"
    );
    assert!(
        dismisses(260.0, 120.0),
        "one away from both reaches nothing under them"
    );

    // A panel in front of the palette covers it where it lies, so a press
    // there is outside the palette, as the engine sees it.
    h.open[PanelKey::Variables] = true;
    h.focus_panel(PanelKey::Variables);
    let front = world
        .resource::<BuiltUi>()
        .and_then(|b| b.panel(PanelKey::Variables))
        .unwrap()
        .root;
    if let Some(l) = world.get_mut_by_id::<Layout>(front) {
        l.visible = true;
        l.rect = Some([300.0, 100.0, 100.0, 100.0]);
    }
    let dismisses = |x, y| h.press_dismisses(&world, &press(x, y));
    assert!(dismisses(320.0, 120.0), "covered by the panel in front");
    assert!(!dismisses(500.0, 300.0), "still the palette's past it");

    let dismissed = UiEvent {
        name: "editor.palette.dismiss".into(),
        element: None,
        value: None,
    };
    assert!(registry::panel(PanelKey::Palette).on_event(&mut h, &mut world, &dismissed));
    assert!(!h.open[PanelKey::Palette]);
}

// The world's assets reach the palette: a behavior is offered as an asset row
// that opens its own panel, everything else as an entity row.
#[test]
fn world_assets_become_rows_routed_by_type() {
    let entries = vec![
        serde_json::json!({"type": "Behavior", "args": {"$id": "greeter", "on": "tick", "do": []}}),
    ];
    let mut h = hook(entries);
    h.drive_palette_toggle(&chord(InputKey::K, true, false));

    let row = h
        .palette
        .items
        .iter()
        .find(|it| it.label == "greeter")
        .expect("the authored behavior is a palette row");
    assert_eq!(row.category, Category::Asset);
    assert_eq!(row.action, PaletteAction::OpenAsset("greeter".to_string()));
}
