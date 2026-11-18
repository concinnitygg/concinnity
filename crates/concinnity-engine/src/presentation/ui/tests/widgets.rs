// Widgets: lists, tabs, checkboxes, text areas, menus, dialogs, toasts, and
// floating panels driven by the frame's input.

use super::*;
use crate::components::{
    Checkbox, KeyBinding, KeyEvent, Layout, List, Menu, MenuEntry, Modal, ModalButton, Tabs,
    TextArea, Toast, ToastTone, Toasts, UiEvent, UiItem,
};
use crate::presentation::text_edit::Metrics;
use crate::protocol::FrameTime;
use alloc::string::String;

// The events the system sent, read with a fresh cursor.
fn events(world: &World) -> Vec<(String, Option<UiValue>)> {
    let mut cursor = EventCursor::default();
    world
        .events::<UiEvent>()
        .map(|e| {
            e.read(&mut cursor)
                .map(|e| (e.name.clone(), e.value.clone()))
                .collect()
        })
        .unwrap_or_default()
}

fn state(world: &World, name: &str) -> Option<UiValue> {
    world
        .resource::<UiState>()
        .and_then(|s| s.get(name).cloned())
}

// A world holding the UI state `values` and the list `files` of `n` rows.
fn with_state(values: &[(&str, UiValue)], n: usize) -> World {
    let mut world = World::new();
    let mut s = UiState::default();
    for (name, value) in values {
        s.set(name, value.clone());
    }
    let items = (0..n)
        .map(|i| UiItem {
            expandable: i == 0,
            ..UiItem::new(&alloc::format!("row {i}"))
        })
        .collect();
    s.set_list("files", items);
    world.insert_resource(s);
    world.insert_resource(FrameTime::default());
    world
}

const VIEWPORT: [f32; 2] = [1280.0, 720.0];

fn click(world: &mut World, x: f32, y: f32) {
    world.add_component(FrameInput {
        mouse_x: x,
        mouse_y: y,
        left_click: true,
        left_button_down: true,
        viewport: VIEWPORT,
        ..Default::default()
    });
    world.step();
}

fn hover(world: &mut World, x: f32, y: f32) {
    world.add_component(FrameInput {
        viewport: VIEWPORT,
        ..make_frame_input(x, y, false)
    });
    world.step();
}

fn press(world: &mut World, key: InputKey) {
    world.add_component(FrameInput {
        key_events: vec![KeyEvent::press(key)],
        viewport: VIEWPORT,
        ..Default::default()
    });
    world.step();
}

// A 200x100 list of 20-pixel rows at the origin, showing `files`.
fn list() -> List {
    List {
        items: "files".into(),
        selected: Some(Binding::State("row".into())),
        width: 200.0,
        height: 100.0,
        row_height: 20.0,
        on_select: "pick".into(),
        on_activate: "open".into(),
        on_expand: "expand".into(),
        on_context: "menu".into(),
        ..Default::default()
    }
}

#[test]
fn a_list_click_selects_and_a_second_activates() {
    let mut world = with_state(&[("row", UiValue::Int(-1))], 3);
    world.add_component(list());
    start(&mut world);

    click(&mut world, 60.0, 30.0);
    assert_eq!(state(&world, "row"), Some(UiValue::Int(1)));
    assert_eq!(events(&world), [("pick".into(), Some(UiValue::Int(1)))]);

    click(&mut world, 60.0, 30.0);
    let names: Vec<String> = events(&world).into_iter().map(|e| e.0).collect();
    assert_eq!(names, ["pick", "pick", "open"], "a double click activates");
}

#[test]
fn a_list_expander_expands_and_a_right_click_asks_for_a_menu() {
    let mut world = with_state(&[("row", UiValue::Int(-1))], 3);
    world.add_component(list());
    start(&mut world);

    click(&mut world, 12.0, 10.0);
    assert_eq!(events(&world), [("expand".into(), Some(UiValue::Int(0)))]);
    assert_eq!(state(&world, "row"), Some(UiValue::Int(-1)), "no selection");

    world.add_component(FrameInput {
        mouse_x: 60.0,
        mouse_y: 50.0,
        right_click: true,
        ..Default::default()
    });
    world.step();
    assert_eq!(state(&world, "row"), Some(UiValue::Int(2)));
    assert_eq!(events(&world)[1], ("menu".into(), Some(UiValue::Int(2))));
}

#[test]
fn the_wheel_scrolls_a_list_within_its_rows() {
    let mut world = with_state(&[], 20);
    world.add_component(list());
    start(&mut world);
    let scroll = |world: &World| world.query::<List>().next().unwrap().scroll;

    world.add_component(FrameInput {
        mouse_x: 60.0,
        mouse_y: 50.0,
        scroll_delta: -10.0,
        ..Default::default()
    });
    world.step();
    assert_eq!(scroll(&world), 30.0);
    world.add_component(FrameInput {
        mouse_x: 60.0,
        mouse_y: 50.0,
        scroll_delta: -1000.0,
        ..Default::default()
    });
    world.step();
    assert_eq!(scroll(&world), 300.0, "20 rows of 20 in a view of 100");
}

#[test]
fn the_keys_move_the_selection_of_the_list_clicked_last() {
    let mut world = with_state(&[("row", UiValue::Int(-1))], 5);
    world.add_component(list());
    start(&mut world);

    press(&mut world, InputKey::Down);
    assert_eq!(
        state(&world, "row"),
        Some(UiValue::Int(-1)),
        "not active yet"
    );
    click(&mut world, 60.0, 30.0);
    press(&mut world, InputKey::Down);
    press(&mut world, InputKey::Down);
    assert_eq!(state(&world, "row"), Some(UiValue::Int(3)));
    press(&mut world, InputKey::Enter);
    assert_eq!(
        events(&world).last().cloned(),
        Some(("open".into(), Some(UiValue::Int(3))))
    );
}

#[test]
fn a_tab_click_selects_the_tab() {
    let mut world = with_state(&[("tab", UiValue::Int(0))], 0);
    world.add_component(Tabs {
        tabs: vec!["A".into(), "B".into(), "C".into()],
        selected: Some(Binding::State("tab".into())),
        width: 300.0,
        on_select: "tab".into(),
        ..Default::default()
    });
    start(&mut world);
    click(&mut world, 250.0, 10.0);
    assert_eq!(state(&world, "tab"), Some(UiValue::Int(2)));
    assert_eq!(events(&world), [("tab".into(), Some(UiValue::Int(2)))]);
}

#[test]
fn a_checkbox_flips_its_value_unless_disabled() {
    let mut world = with_state(&[("grid", UiValue::Bool(false))], 0);
    world.add_component(Checkbox {
        checked: Some(Binding::State("grid".into())),
        on_change: "grid".into(),
        ..Default::default()
    });
    world.add_component(Checkbox {
        checked: Some(Binding::State("grid".into())),
        disabled: true,
        y: 100.0,
        ..Default::default()
    });
    start(&mut world);
    click(&mut world, 10.0, 10.0);
    assert_eq!(state(&world, "grid"), Some(UiValue::Bool(true)));
    assert_eq!(events(&world), [("grid".into(), Some(UiValue::Bool(true)))]);
    click(&mut world, 10.0, 110.0);
    assert_eq!(state(&world, "grid"), Some(UiValue::Bool(true)), "disabled");
}

// A context menu of one entry emitting `picked`, opened by `menu`.
fn context_menu() -> Menu {
    Menu {
        open: Some(Binding::State("menu".into())),
        at_cursor: true,
        entries: vec![MenuEntry {
            label: "Pick".into(),
            action: Some(act(serde_json::json!({"emit": "picked"}))),
            ..Default::default()
        }],
        ..Default::default()
    }
}

// A region at the origin emitting `under`, to see what a popup leaves it.
fn region_under() -> HitRegion {
    HitRegion {
        width: 1000.0,
        height: 1000.0,
        action: Some(act(serde_json::json!({"emit": "under"}))),
        ..Default::default()
    }
}

#[test]
fn a_menu_opens_at_the_cursor_and_a_pick_fires_and_closes_it() {
    let mut world = with_state(&[("menu", UiValue::Bool(true))], 0);
    world.add_component(context_menu());
    world.add_component(region_under());
    start(&mut world);
    hover(&mut world, 100.0, 100.0);
    assert_eq!(
        world.query::<Menu>().next().unwrap().at,
        Some([100.0, 100.0])
    );

    click(&mut world, 120.0, 110.0);
    let names: Vec<String> = events(&world).into_iter().map(|e| e.0).collect();
    assert_eq!(names, ["picked"], "the region under the menu stays quiet");
    assert_eq!(state(&world, "menu"), Some(UiValue::Bool(false)));
}

#[test]
fn a_click_outside_a_menu_closes_it_and_reaches_nothing_else() {
    let mut world = with_state(&[("menu", UiValue::Bool(true))], 0);
    world.add_component(context_menu());
    world.add_component(region_under());
    start(&mut world);
    hover(&mut world, 100.0, 100.0);
    click(&mut world, 500.0, 500.0);
    assert!(events(&world).is_empty());
    assert_eq!(state(&world, "menu"), Some(UiValue::Bool(false)));
}

fn dialog() -> Modal {
    let mut modal = Modal {
        open: Some(Binding::State("ask".into())),
        title: "Quit?".into(),
        buttons: vec![
            ModalButton {
                label: "No".into(),
                ..Default::default()
            },
            ModalButton {
                label: "Yes".into(),
                action: Some(act(serde_json::json!({"emit": "yes"}))),
                primary: true,
            },
        ],
        ..Default::default()
    };
    modal.lay_out(UI_REFERENCE_SIZE, 16.0, 0.0);
    modal
}

#[test]
fn an_open_dialog_shuts_out_its_screen_and_escape_closes_it() {
    let mut world = with_state(&[("ask", UiValue::Bool(true))], 0);
    world.add_component(dialog());
    world.add_component(region_under());
    start(&mut world);
    click(&mut world, 5.0, 5.0);
    assert!(events(&world).is_empty(), "the region behind is shut out");
    world.add_component(FrameInput {
        escape: true,
        ..Default::default()
    });
    world.step();
    assert_eq!(state(&world, "ask"), Some(UiValue::Bool(false)));
}

#[test]
fn a_dialog_button_fires_its_action_and_closes_the_dialog() {
    let mut world = with_state(&[("ask", UiValue::Bool(true))], 0);
    let modal = dialog();
    let yes = modal.drawn.buttons[1];
    world.add_component(modal);
    start(&mut world);
    click(&mut world, yes[0] + 4.0, yes[1] + 4.0);
    assert_eq!(events(&world), [("yes".into(), None)]);
    assert_eq!(state(&world, "ask"), Some(UiValue::Bool(false)));
}

#[test]
fn toasts_arrive_count_down_and_dismiss_on_a_click() {
    let mut world = with_state(&[], 0);
    world.add_component(Toasts::default());
    start(&mut world);
    let shown = |world: &World| world.query::<Toasts>().next().unwrap().shown.len();

    world
        .events_mut::<Toast>()
        .send(Toast::new("saved", ToastTone::Success));
    world
        .events_mut::<Toast>()
        .send(Toast::new("built", ToastTone::Info));
    hover(&mut world, 0.0, 0.0);
    assert_eq!(shown(&world), 2);

    // The newest card sits nearest the bottom-right corner.
    let corner = world
        .query::<Toasts>()
        .next()
        .unwrap()
        .card_rect(0, [1280.0, 720.0]);
    world.add_component(FrameInput {
        mouse_x: corner[0] + 10.0,
        mouse_y: corner[1] + 10.0,
        left_click: true,
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    world.step();
    let toasts = world.query::<Toasts>().next().unwrap();
    assert_eq!(toasts.shown.len(), 1);
    assert_eq!(toasts.shown[0].text, "saved", "the newest was dismissed");

    world.insert_resource(FrameTime {
        dt: 10.0,
        elapsed: 10.0,
    });
    hover(&mut world, 0.0, 0.0);
    assert_eq!(shown(&world), 0, "spent");
}

// A 400x200 text area at the origin editing `notes`, on a 10x20 grid.
fn text_area() -> TextArea {
    TextArea {
        text: Some(Binding::State("notes".into())),
        width: 400.0,
        height: 200.0,
        on_change: "edited".into(),
        metrics: Some(Metrics {
            line_h: 20.0,
            advance: 10.0,
            text_px: 16.0,
            label_scale: 1.0,
        }),
        ..Default::default()
    }
}

#[test]
fn a_text_area_takes_the_keyboard_and_writes_its_edits_back() {
    let mut world = with_state(&[("notes", UiValue::Text("hi".into()))], 0);
    world.add_component(text_area());
    start(&mut world);
    click(&mut world, 390.0, 5.0);
    assert!(world.query::<TextArea>().next().unwrap().focused);

    world.add_component(FrameInput {
        key_events: vec![KeyEvent::Text('!')],
        ..Default::default()
    });
    world.step();
    assert_eq!(state(&world, "notes"), Some(UiValue::Text("hi!".into())));
    assert_eq!(events(&world), [("edited".into(), None)]);

    world.add_component(FrameInput {
        escape: true,
        ..Default::default()
    });
    world.step();
    assert!(!world.query::<TextArea>().next().unwrap().focused);
}

#[test]
fn the_save_shortcut_sends_the_text_areas_save_event() {
    let mut world = with_state(&[("notes", UiValue::Text("hi".into()))], 0);
    world.add_component(TextArea {
        on_save: "saved".into(),
        ..text_area()
    });
    start(&mut world);
    click(&mut world, 390.0, 5.0);
    world.add_component(FrameInput {
        key_events: vec![KeyEvent::Press(KeyPress::new(InputKey::S, KeyMods::CTRL))],
        ..Default::default()
    });
    world.step();
    assert_eq!(events(&world), [("saved".into(), None)]);
}

#[test]
fn a_text_area_shows_a_change_to_its_binding_while_unfocused() {
    let mut world = with_state(&[("notes", UiValue::Text("a".into()))], 0);
    world.add_component(text_area());
    start(&mut world);
    hover(&mut world, 0.0, 0.0);
    world
        .resource_mut::<UiState>()
        .unwrap()
        .set("notes", UiValue::Text("b\nc".into()));
    hover(&mut world, 0.0, 0.0);
    let area = world.query::<TextArea>().next().unwrap();
    assert_eq!(area.editor.line_count(), 2);
    assert_eq!(area.editor.line(1), "c");
}

#[test]
fn a_panel_drags_by_its_top_band_and_sizes_by_its_corner() {
    let mut world = with_state(&[], 0);
    world.add_component(Layout {
        movable: true,
        resizable: true,
        rect: Some([100.0, 100.0, 300.0, 200.0]),
        ..Default::default()
    });
    start(&mut world);
    let root = |world: &World| world.query::<Layout>().next().unwrap().clone();

    click(&mut world, 150.0, 110.0);
    world.add_component(FrameInput {
        mouse_x: 200.0,
        mouse_y: 130.0,
        left_button_down: true,
        viewport: VIEWPORT,
        ..Default::default()
    });
    world.step();
    assert_eq!(root(&world).offset, [50.0, 20.0]);
    hover(&mut world, 200.0, 130.0);

    click(&mut world, 398.0, 298.0);
    world.add_component(FrameInput {
        mouse_x: 300.0,
        mouse_y: 200.0,
        left_button_down: true,
        viewport: VIEWPORT,
        ..Default::default()
    });
    world.step();
    assert_eq!(
        root(&world).size,
        Some([202.0, 102.0]),
        "dragged in by 98 each way"
    );
}

#[test]
fn an_open_dialog_shuts_out_its_screens_key_bindings_and_fields() {
    let mut world = with_state(&[("ask", UiValue::Bool(true))], 0);
    world.add_component(dialog());
    world.add_component(KeyBinding {
        key: "Q".into(),
        action: Some(act(serde_json::json!({"emit": "key"}))),
        ..Default::default()
    });
    world.add_component(TextField {
        width: 100.0,
        height: 30.0,
        ..Default::default()
    });
    start(&mut world);
    press(&mut world, InputKey::Q);
    click(&mut world, 10.0, 10.0);
    assert!(events(&world).is_empty(), "no key binding fires");
    let field = world.query::<TextField>().next().unwrap();
    assert!(!field.focused, "no field takes the keyboard");
}

#[test]
fn a_click_focuses_the_field_under_it_and_blurs_the_rest() {
    let mut world = with_state(&[], 0);
    for x in [0.0, 200.0] {
        world.add_component(TextField {
            x,
            width: 100.0,
            height: 30.0,
            content: "ab".into(),
            ..Default::default()
        });
    }
    start(&mut world);
    click(&mut world, 250.0, 10.0);
    let focus: Vec<(bool, usize)> = world
        .query::<TextField>()
        .map(|f| (f.focused, f.caret))
        .collect();
    assert_eq!(focus, [(false, 0), (true, 2)], "caret at the end");
    click(&mut world, 500.0, 500.0);
    assert!(!world.query::<TextField>().any(|f| f.focused));
}

#[test]
fn a_button_fires_its_action_unless_disabled() {
    let mut world = with_state(&[], 0);
    for (x, disabled) in [(0.0, false), (200.0, true)] {
        world.add_component(Button {
            label: "Go".into(),
            action: Some(act(serde_json::json!({"emit": "go"}))),
            x,
            disabled,
            ..Default::default()
        });
    }
    start(&mut world);
    click(&mut world, 210.0, 10.0);
    assert!(events(&world).is_empty(), "the disabled button");
    click(&mut world, 10.0, 10.0);
    assert_eq!(events(&world), [("go".into(), None)]);
    assert!(world.query::<Button>().next().unwrap().hovered);
}

#[test]
fn a_button_shows_the_value_it_binds() {
    let mut world = with_state(&[("step", UiValue::Text("0.5 m".into()))], 0);
    world.add_component(Button {
        text_binding: Some(Binding::State("step".into())),
        ..Default::default()
    });
    start(&mut world);
    hover(&mut world, 500.0, 500.0);
    assert_eq!(world.query::<Button>().next().unwrap().label, "0.5 m");
    if let Some(s) = world.resource_mut::<UiState>() {
        s.set("step", UiValue::Text("1 m".into()));
    }
    hover(&mut world, 500.0, 500.0);
    assert_eq!(world.query::<Button>().next().unwrap().label, "1 m");
}

// A floating panel over `rect` holding the element `holds`.
fn floating(rect: [f32; 4], holds: u32) -> Layout {
    Layout {
        movable: true,
        children: vec![crate::components::LayoutChild {
            item: Some(Ref::new(AssetId(holds))),
            ..Default::default()
        }],
        rect: Some(rect),
        ..Default::default()
    }
}

#[test]
fn a_press_reaches_only_the_front_panel_and_brings_its_panel_forward() {
    let mut world = with_state(
        &[("a", UiValue::Bool(false)), ("b", UiValue::Bool(false))],
        0,
    );
    for (id, x, name) in [(10, 10.0, "a"), (20, 60.0, "b")] {
        world.push_identified(
            AssetId(id),
            Checkbox {
                checked: Some(Binding::State(name.into())),
                x,
                y: 40.0,
                width: 100.0,
                ..Default::default()
            },
        );
    }
    world.push_identified(AssetId(1), floating([0.0, 0.0, 200.0, 200.0], 10));
    world.push_identified(AssetId(2), floating([50.0, 0.0, 200.0, 200.0], 20));
    start(&mut world);
    click(&mut world, 80.0, 50.0);
    assert_eq!(state(&world, "a"), Some(UiValue::Bool(false)), "covered");
    assert_eq!(state(&world, "b"), Some(UiValue::Bool(true)));
    click(&mut world, 20.0, 50.0);
    assert_eq!(state(&world, "a"), Some(UiValue::Bool(true)));
    let ranks: Vec<u32> = world.query::<Layout>().map(|l| l.rank).collect();
    assert!(ranks[0] > ranks[1], "the pressed panel came forward");
    click(&mut world, 80.0, 50.0);
    assert_eq!(state(&world, "a"), Some(UiValue::Bool(false)));
    assert_eq!(state(&world, "b"), Some(UiValue::Bool(true)), "now covered");
}

#[test]
fn a_panel_on_a_screen_above_a_capture_keeps_the_press_from_it() {
    let mut world = with_state(&[], 0);
    world.push_identified(
        AssetId(90),
        Screen {
            initial: true,
            fade_in_secs: 0.0,
            ..Default::default()
        },
    );
    world.push_identified(
        AssetId(91),
        Screen {
            persistent: true,
            layer: 5,
            pauses_world: false,
            ..Default::default()
        },
    );
    world.add_component(HitRegion {
        width: 300.0,
        height: 300.0,
        action: Some(act(serde_json::json!({"emit": "under"}))),
        screen: Some(Ref::new(AssetId(90))),
        ..Default::default()
    });
    world.add_component(Layout {
        movable: true,
        rect: Some([0.0, 0.0, 200.0, 200.0]),
        screen: Some(Ref::new(AssetId(91))),
        ..Default::default()
    });
    start(&mut world);
    click(&mut world, 100.0, 100.0);
    assert!(events(&world).is_empty(), "the panel covers the region");
    click(&mut world, 250.0, 250.0);
    assert_eq!(events(&world), [("under".into(), None)]);
}

#[test]
fn a_field_a_hidden_layout_holds_gives_up_the_keyboard() {
    let mut world = with_state(&[], 0);
    world.push_identified(
        AssetId(10),
        TextField {
            focused: true,
            ..Default::default()
        },
    );
    world.add_component(Layout {
        children: vec![crate::components::LayoutChild {
            item: Some(Ref::new(AssetId(10))),
            ..Default::default()
        }],
        visible: false,
        ..Default::default()
    });
    start(&mut world);
    hover(&mut world, 500.0, 500.0);
    assert!(!world.query::<TextField>().next().unwrap().focused);
}

#[test]
fn a_second_context_press_moves_the_menu_and_reselects_the_row() {
    let mut world = with_state(
        &[("row", UiValue::Int(1)), ("menu", UiValue::Bool(true))],
        5,
    );
    world.add_component(list());
    world.add_component(context_menu());
    start(&mut world);
    hover(&mut world, 60.0, 30.0);
    world.add_component(FrameInput {
        mouse_x: 60.0,
        mouse_y: 70.0,
        right_click: true,
        viewport: VIEWPORT,
        ..Default::default()
    });
    world.step();
    assert_eq!(
        world.query::<Menu>().next().unwrap().at,
        Some([60.0, 70.0]),
        "the context menu moved to the press"
    );
    assert_eq!(state(&world, "row"), Some(UiValue::Int(3)));
    assert_eq!(events(&world), [("menu".into(), Some(UiValue::Int(3)))]);
}

#[test]
fn a_context_press_on_the_menu_itself_stays_with_the_menu() {
    let mut world = with_state(
        &[("row", UiValue::Int(1)), ("menu", UiValue::Bool(true))],
        5,
    );
    world.add_component(list());
    world.add_component(context_menu());
    start(&mut world);
    hover(&mut world, 60.0, 30.0);
    world.add_component(FrameInput {
        mouse_x: 70.0,
        mouse_y: 35.0,
        right_click: true,
        viewport: VIEWPORT,
        ..Default::default()
    });
    world.step();
    assert_eq!(world.query::<Menu>().next().unwrap().at, Some([60.0, 30.0]));
    assert_eq!(state(&world, "row"), Some(UiValue::Int(1)));
    assert!(events(&world).is_empty(), "the list beneath heard nothing");
}

#[test]
fn an_enter_the_active_list_uses_confirms_nothing_else() {
    let mut world = with_state(&[("row", UiValue::Int(-1))], 5);
    world.push_identified(
        AssetId(90),
        Screen {
            initial: true,
            fade_in_secs: 0.0,
            ..Default::default()
        },
    );
    world.add_component(List {
        screen: Some(Ref::new(AssetId(90))),
        ..list()
    });
    world.add_component(HitRegion {
        y: 300.0,
        width: 200.0,
        height: 40.0,
        action: Some(act(serde_json::json!({"emit": "under"}))),
        screen: Some(Ref::new(AssetId(90))),
        ..Default::default()
    });
    start(&mut world);
    click(&mut world, 60.0, 30.0);
    world.add_component(FrameInput {
        nav: Some(NavDirection::Right),
        viewport: VIEWPORT,
        ..Default::default()
    });
    world.step();
    press(&mut world, InputKey::Enter);
    assert_eq!(
        events(&world).last().cloned(),
        Some(("open".into(), Some(UiValue::Int(1))))
    );
    assert!(
        events(&world).iter().all(|e| e.0 != "under"),
        "the focused region did not fire too"
    );
}

#[test]
fn a_lists_scrollbar_drags_and_its_keys_stay_with_it() {
    let mut world = with_state(&[("row", UiValue::Int(-1))], 20);
    world.add_component(list());
    world.add_component(KeyBinding {
        key: "Down".into(),
        action: Some(act(serde_json::json!({"emit": "key"}))),
        ..Default::default()
    });
    world.add_component(KeyBinding {
        key: "Q".into(),
        action: Some(act(serde_json::json!({"emit": "q"}))),
        ..Default::default()
    });
    start(&mut world);
    // The thumb is a quarter of the 100-tall track at its top; grab it at 10
    // and drag it down by half the travel.
    click(&mut world, 197.0, 10.0);
    assert_eq!(
        state(&world, "row"),
        Some(UiValue::Int(-1)),
        "no row picked"
    );
    world.add_component(FrameInput {
        mouse_x: 197.0,
        mouse_y: 10.0 + 37.5,
        left_button_down: true,
        viewport: VIEWPORT,
        ..Default::default()
    });
    world.step();
    assert_eq!(world.query::<List>().next().unwrap().scroll, 150.0);

    press(&mut world, InputKey::Down);
    assert_eq!(state(&world, "row"), Some(UiValue::Int(0)));
    assert!(
        events(&world).iter().all(|e| e.0 != "key"),
        "the active list took Down"
    );
    press(&mut world, InputKey::Q);
    assert!(
        events(&world).iter().any(|e| e.0 == "q"),
        "a key the list does not use still fires"
    );
}

#[test]
fn a_binding_change_while_focused_lands_once_the_area_lets_go() {
    let mut world = with_state(&[("notes", UiValue::Text("a".into()))], 0);
    world.add_component(text_area());
    start(&mut world);
    click(&mut world, 390.0, 5.0);
    world
        .resource_mut::<UiState>()
        .unwrap()
        .set("notes", UiValue::Text("b".into()));
    hover(&mut world, 390.0, 5.0);
    assert_eq!(
        world.query::<TextArea>().next().unwrap().editor.line(0),
        "a"
    );
    world.add_component(FrameInput {
        escape: true,
        ..Default::default()
    });
    world.step();
    hover(&mut world, 390.0, 5.0);
    assert_eq!(
        world.query::<TextArea>().next().unwrap().editor.line(0),
        "b"
    );
}
