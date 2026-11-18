// Widgets: lists, tabs, checkboxes, text areas, menus, dialogs, toasts, and
// floating panels driven by the frame's input.

use super::*;
use crate::components::{
    Button, Canvas, Checkbox, Grid, KeyBinding, KeyEvent, Layout, LayoutChild, List, ListToggle,
    Menu, MenuEntry, Modal, ModalButton, Tabs, TextArea, Toast, ToastTone, Toasts, UiEvent, UiItem,
    ValueSlider,
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

// With no binding to keep a selection in, the keys move nothing.
#[test]
fn the_keys_leave_a_list_with_no_selection_binding_alone() {
    let mut world = with_state(&[], 5);
    world.add_component(List {
        selected: None,
        ..list()
    });
    start(&mut world);
    click(&mut world, 60.0, 30.0);
    assert_eq!(events(&world), [("pick".into(), Some(UiValue::Int(1)))]);
    for key in [
        InputKey::Down,
        InputKey::Up,
        InputKey::Home,
        InputKey::End,
        InputKey::Enter,
    ] {
        press(&mut world, key);
        let sent = events(&world);
        assert!(
            sent.iter()
                .all(|e| *e == ("pick".into(), Some(UiValue::Int(1)))),
            "{key:?} sent {sent:?}"
        );
    }
}

#[test]
fn a_modified_click_extends_and_a_toggle_cell_flips_its_flag() {
    let mut world = with_state(&[("row", UiValue::Int(-1))], 3);
    if let Some(s) = world.resource_mut::<UiState>() {
        s.edit_list("files", |items| {
            for item in items.iter_mut() {
                item.expandable = false;
                item.toggles = Some(0);
            }
        });
    }
    world.add_component(List {
        on_extend: "extend".into(),
        toggles: vec![ListToggle {
            on_toggle: "hide".into(),
            ..Default::default()
        }],
        ..list()
    });
    start(&mut world);

    click(&mut world, 12.0, 30.0);
    assert_eq!(events(&world), [("hide".into(), Some(UiValue::Int(1)))]);
    assert_eq!(state(&world, "row"), Some(UiValue::Int(-1)), "no selection");

    world.add_component(FrameInput {
        mouse_x: 120.0,
        mouse_y: 50.0,
        left_click: true,
        left_button_down: true,
        shift: true,
        viewport: VIEWPORT,
        ..Default::default()
    });
    world.step();
    assert_eq!(events(&world)[1], ("extend".into(), Some(UiValue::Int(2))));
    assert_eq!(
        state(&world, "row"),
        Some(UiValue::Int(-1)),
        "the app selects"
    );
}

#[test]
fn a_press_dragged_off_an_item_drags_it_out_once() {
    let mut world = with_state(&[("row", UiValue::Int(-1))], 3);
    world.add_component(List {
        on_drag: "drag".into(),
        ..list()
    });
    start(&mut world);
    let held = |x: f32| FrameInput {
        mouse_x: x,
        mouse_y: 30.0,
        left_button_down: true,
        viewport: VIEWPORT,
        ..Default::default()
    };

    click(&mut world, 60.0, 30.0);
    world.add_component(held(62.0));
    world.step();
    assert_eq!(events(&world).len(), 1, "two pixels is still a click");
    world.add_component(held(400.0));
    world.step();
    world.add_component(held(500.0));
    world.step();
    let drags: Vec<_> = events(&world)
        .into_iter()
        .filter(|e| e.0 == "drag")
        .collect();
    assert_eq!(drags, [("drag".into(), Some(UiValue::Int(1)))]);
}

#[test]
fn a_grid_selects_its_cells_and_steps_them_by_rows() {
    let mut world = with_state(&[("cell", UiValue::Int(-1))], 7);
    world.add_component(Grid {
        items: "files".into(),
        selected: Some(Binding::State("cell".into())),
        width: 300.0,
        height: 300.0,
        cell_width: 90.0,
        cell_height: 90.0,
        gap: 10.0,
        on_select: "pick".into(),
        ..Default::default()
    });
    start(&mut world);

    click(&mut world, 150.0, 150.0);
    assert_eq!(state(&world, "cell"), Some(UiValue::Int(4)));
    press(&mut world, InputKey::Down);
    assert_eq!(
        state(&world, "cell"),
        Some(UiValue::Int(6)),
        "clamped to the last"
    );
    press(&mut world, InputKey::Up);
    press(&mut world, InputKey::Left);
    assert_eq!(state(&world, "cell"), Some(UiValue::Int(2)));
}

// A scrolling layout over a 300-pixel column: a list at its top and a button
// laid out below its 100-pixel view.
fn scrolling_world(list_rows: usize) -> World {
    let mut world = with_state(&[], list_rows);
    world.push_identified(
        AssetId(21),
        List {
            height: 60.0,
            ..list()
        },
    );
    world.push_identified(
        AssetId(22),
        Button {
            y: 150.0,
            width: 100.0,
            height: 30.0,
            action: Some(act(serde_json::json!({"emit": "pressed"}))),
            ..Default::default()
        },
    );
    world.push_identified(
        AssetId(20),
        Layout {
            children: vec![LayoutChild::fit(AssetId(21)), LayoutChild::fit(AssetId(22))],
            scroll: true,
            rect: Some([0.0, 0.0, 200.0, 100.0]),
            content_height: 300.0,
            ..Default::default()
        },
    );
    world
}

fn scrolled(world: &World) -> f32 {
    world.query::<Layout>().next().unwrap().scrolled
}

fn wheel(world: &mut World, x: f32, y: f32) {
    world.add_component(FrameInput {
        mouse_x: x,
        mouse_y: y,
        scroll_delta: -10.0,
        viewport: VIEWPORT,
        ..Default::default()
    });
    world.step();
}

#[test]
fn the_wheel_scrolls_a_layout_unless_a_list_in_it_scrolls() {
    let mut world = scrolling_world(20);
    start(&mut world);
    wheel(&mut world, 60.0, 30.0);
    assert_eq!(scrolled(&world), 0.0, "the overflowing list took the wheel");
    wheel(&mut world, 150.0, 80.0);
    assert_eq!(scrolled(&world), 30.0, "below the list, the layout scrolls");

    let mut world = scrolling_world(2);
    start(&mut world);
    wheel(&mut world, 60.0, 30.0);
    assert_eq!(scrolled(&world), 30.0, "a list that fits leaves the wheel");
}

#[test]
fn an_element_scrolled_out_of_its_layout_takes_no_press() {
    let mut world = scrolling_world(2);
    start(&mut world);
    click(&mut world, 50.0, 160.0);
    assert!(events(&world).is_empty(), "the button is below the view");
    for layout in world.query_mut::<Layout>() {
        layout.rect = Some([0.0, 0.0, 200.0, 200.0]);
    }
    click(&mut world, 50.0, 160.0);
    assert_eq!(events(&world).len(), 1, "a taller view shows it");
}

// A press on a value slider sets the number under it, a drag follows the
// cursor even off the slider, and letting go commits what it settled on.
#[test]
fn a_value_slider_follows_a_drag_and_commits_on_release() {
    let mut world = with_state(&[("radius", UiValue::Float(0.0))], 0);
    world.add_component(ValueSlider {
        value: Some(Binding::State("radius".into())),
        max: 10.0,
        step: 1.0,
        width: 144.0,
        value_width: 44.0,
        on_change: "radius".into(),
        on_commit: "radius_set".into(),
        ..Default::default()
    });
    start(&mut world);
    // The track runs from x 5 to 95.
    click(&mut world, 50.0, 12.0);
    assert_eq!(state(&world, "radius"), Some(UiValue::Float(5.0)));
    world.add_component(FrameInput {
        mouse_x: 400.0,
        mouse_y: 300.0,
        left_button_down: true,
        viewport: VIEWPORT,
        ..Default::default()
    });
    world.step();
    assert_eq!(state(&world, "radius"), Some(UiValue::Float(10.0)));
    hover(&mut world, 400.0, 300.0);
    let sent = events(&world);
    assert_eq!(
        sent.last(),
        Some(&("radius_set".into(), Some(UiValue::Float(10.0))))
    );
}

// The number's column is no part of the track: a press there grabs nothing.
#[test]
fn a_value_slider_takes_no_press_on_its_number() {
    let mut world = with_state(&[("radius", UiValue::Float(2.0))], 0);
    world.add_component(ValueSlider {
        value: Some(Binding::State("radius".into())),
        max: 10.0,
        width: 144.0,
        value_width: 44.0,
        ..Default::default()
    });
    start(&mut world);
    click(&mut world, 130.0, 12.0);
    assert_eq!(state(&world, "radius"), Some(UiValue::Float(2.0)));
}

// A slider hidden mid-drag lets go, committing what it set.
#[test]
fn a_value_slider_hidden_mid_drag_commits() {
    let mut world = with_state(&[("radius", UiValue::Float(0.0))], 0);
    world.add_component(ValueSlider {
        value: Some(Binding::State("radius".into())),
        max: 10.0,
        step: 1.0,
        width: 144.0,
        value_width: 44.0,
        on_commit: "radius_set".into(),
        ..Default::default()
    });
    start(&mut world);
    click(&mut world, 50.0, 12.0);
    for slider in world.query_mut::<ValueSlider>() {
        slider.visible = false;
    }
    world.add_component(FrameInput {
        mouse_x: 50.0,
        mouse_y: 12.0,
        left_button_down: true,
        viewport: VIEWPORT,
        ..Default::default()
    });
    world.step();
    assert_eq!(
        events(&world).last(),
        Some(&("radius_set".into(), Some(UiValue::Float(5.0))))
    );
}

// Of nested scrolling layouts under the pointer, the innermost scrolls.
#[test]
fn the_innermost_scrolling_layout_takes_the_wheel() {
    let mut world = with_state(&[], 0);
    let layout = |rect: [f32; 4], children: Vec<LayoutChild>| Layout {
        children,
        scroll: true,
        rect: Some(rect),
        content_height: 1000.0,
        ..Default::default()
    };
    world.push_identified(AssetId(31), layout([0.0, 0.0, 100.0, 50.0], Vec::new()));
    world.push_identified(
        AssetId(30),
        layout(
            [0.0, 0.0, 200.0, 200.0],
            vec![LayoutChild::fit(AssetId(31))],
        ),
    );
    start(&mut world);
    wheel(&mut world, 20.0, 20.0);
    let scrolled_of =
        |world: &World, id: u32| world.get_by_id::<Layout>(AssetId(id)).unwrap().scrolled;
    assert_eq!(scrolled_of(&world, 31), 30.0);
    assert_eq!(scrolled_of(&world, 30), 0.0, "the outer one stays put");
}

// While a modal root is shown only what it holds takes a click, and a field
// outside it gives up the keyboard.
#[test]
fn a_modal_root_takes_its_screens_input() {
    let mut world = with_state(&[], 0);
    let button = |x: f32, name: &str| Button {
        label: name.into(),
        x,
        y: 10.0,
        action: Some(act(serde_json::json!({"emit": name}))),
        ..Default::default()
    };
    world.push_identified(AssetId(40), button(10.0, "inside"));
    world.add_component(button(400.0, "outside"));
    world.add_component(TextField {
        x: 400.0,
        y: 100.0,
        focused: true,
        ..Default::default()
    });
    world.push_identified(
        AssetId(41),
        Layout {
            modal: true,
            children: vec![LayoutChild::fit(AssetId(40))],
            rect: Some([0.0, 0.0, 200.0, 100.0]),
            ..Default::default()
        },
    );
    start(&mut world);
    click(&mut world, 410.0, 15.0);
    assert!(events(&world).is_empty(), "the button outside is shut out");
    assert!(!world.query::<TextField>().next().unwrap().focused);
    click(&mut world, 20.0, 15.0);
    assert_eq!(events(&world), [("inside".into(), None)]);
}

// A modal root shuts out the toasts and menus behind it too.
#[test]
fn a_modal_root_shuts_out_toasts_and_menus() {
    let mut world = with_state(&[("menu", UiValue::Bool(true))], 0);
    world.push_identified(AssetId(40), Button::default());
    world.push_identified(
        AssetId(41),
        Layout {
            modal: true,
            children: vec![LayoutChild::fit(AssetId(40))],
            rect: Some([0.0, 0.0, 10.0, 10.0]),
            ..Default::default()
        },
    );
    world.add_component(Toasts {
        on_dismiss: "gone".into(),
        ..Default::default()
    });
    world.add_component(context_menu());
    start(&mut world);
    world
        .events_mut::<Toast>()
        .send(Toast::new("saved", ToastTone::Success));
    hover(&mut world, 100.0, 100.0);
    let card = world
        .query::<Toasts>()
        .next()
        .unwrap()
        .card_rect(0, [1280.0, 720.0]);
    click(&mut world, card[0] + 10.0, card[1] + 10.0);
    assert_eq!(world.query::<Toasts>().next().unwrap().shown.len(), 1);
    click(&mut world, 120.0, 110.0);
    assert!(
        events(&world).is_empty(),
        "neither the toast nor the menu acted"
    );
}

// A 200x100 pannable canvas at the origin over a 400-wide chart: a tagged
// card at its left, untagged ground under it.
fn canvas_world() -> World {
    let mut world = with_state(&[], 0);
    if let Some(s) = world.resource_mut::<UiState>() {
        s.edit_shapes("chart", |shapes| {
            shapes.push(crate::components::UiShape {
                rect: [0.0, 0.0, 400.0, 100.0],
                ..Default::default()
            });
            shapes.push(crate::components::UiShape {
                rect: [10.0, 10.0, 50.0, 30.0],
                tag: Some(7),
                ..Default::default()
            });
        });
    }
    world.add_component(Canvas {
        shapes: "chart".into(),
        on_press: "card".into(),
        on_extend: "card.extend".into(),
        pannable: true,
        width: 200.0,
        height: 100.0,
        ..Default::default()
    });
    world
}

fn pan(world: &World) -> [f32; 2] {
    world.query::<Canvas>().next().unwrap().pan
}

// A press on a tagged shape reports its tag, with the modifiers its own
// event; a drag on empty space pans, never past the shapes; the wheel scrolls.
#[test]
fn a_canvas_reports_its_shapes_and_pans() {
    let mut world = canvas_world();
    start(&mut world);
    click(&mut world, 20.0, 20.0);
    assert_eq!(events(&world), [("card".into(), Some(UiValue::Int(7)))]);
    world.add_component(FrameInput {
        ctrl: true,
        ..make_frame_input(20.0, 20.0, true)
    });
    world.step();
    assert!(events(&world).contains(&("card.extend".into(), Some(UiValue::Int(7)))));

    click(&mut world, 150.0, 50.0);
    for x in [120.0, -500.0] {
        world.add_component(FrameInput {
            mouse_x: x,
            mouse_y: 50.0,
            left_button_down: true,
            viewport: VIEWPORT,
            ..Default::default()
        });
        world.step();
    }
    assert_eq!(
        pan(&world),
        [200.0, 0.0],
        "dragged left, stopped at the edge"
    );
    hover(&mut world, 150.0, 50.0);
    assert_eq!(pan(&world), [200.0, 0.0], "let go");
    world.add_component(FrameInput {
        shift: true,
        ..make_frame_input(150.0, 50.0, false)
    });
    if let Some(i) = world.query_mut::<FrameInput>().last() {
        i.scroll_delta = 2.0;
    }
    world.step();
    assert!(pan(&world)[0] < 200.0, "shift and the wheel pan across");
}

// Shapes that shrink pull the pan back within them on the next frame.
#[test]
fn a_canvas_pan_follows_its_shapes_shrinking() {
    let mut world = canvas_world();
    start(&mut world);
    for c in world.query_mut::<Canvas>() {
        c.pan = [200.0, 0.0];
    }
    if let Some(s) = world.resource_mut::<UiState>() {
        s.edit_shapes("chart", |shapes| shapes[0].rect[2] = 250.0);
    }
    hover(&mut world, 500.0, 500.0);
    assert_eq!(pan(&world), [50.0, 0.0]);
}

// A passthrough canvas takes no press.
#[test]
fn a_passthrough_canvas_takes_nothing() {
    let mut world = canvas_world();
    for c in world.query_mut::<Canvas>() {
        c.passthrough = true;
    }
    world.add_component(region_under());
    start(&mut world);
    click(&mut world, 20.0, 20.0);
    let names: Vec<String> = events(&world).into_iter().map(|e| e.0).collect();
    assert_eq!(names, ["under"]);
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

// A menu placed by `x` and `y` follows them while open.
#[test]
fn an_open_menu_follows_its_place() {
    let mut world = with_state(&[("menu", UiValue::Bool(true))], 0);
    world.add_component(Menu {
        at_cursor: false,
        x: 10.0,
        y: 20.0,
        ..context_menu()
    });
    start(&mut world);
    hover(&mut world, 0.0, 0.0);
    if let Some(m) = world.query_mut::<Menu>().next() {
        m.x = 300.0;
    }
    hover(&mut world, 0.0, 0.0);
    assert_eq!(
        world.query::<Menu>().next().unwrap().at,
        Some([300.0, 20.0])
    );
}

// An entry that stays open, and an expandable list item, leave the menu up
// as they fire.
#[test]
fn a_toggle_entry_and_a_disclosure_item_leave_the_menu_open() {
    let mut world = with_state(&[("menu", UiValue::Bool(true))], 0);
    let mut menu = context_menu();
    menu.entries[0].stay_open = true;
    world.add_component(menu);
    start(&mut world);
    hover(&mut world, 100.0, 100.0);
    click(&mut world, 120.0, 110.0);
    assert_eq!(state(&world, "menu"), Some(UiValue::Bool(true)));

    let mut world = with_state(&[("menu", UiValue::Bool(true))], 0);
    if let Some(s) = world.resource_mut::<UiState>() {
        s.set_list(
            "rows",
            vec![UiItem {
                expandable: true,
                ..UiItem::new("Prefabs")
            }],
        );
    }
    world.add_component(Menu {
        items: "rows".into(),
        on_pick: "pick".into(),
        ..context_menu()
    });
    start(&mut world);
    hover(&mut world, 100.0, 100.0);
    click(&mut world, 120.0, 110.0);
    assert_eq!(events(&world), [("pick".into(), Some(UiValue::Int(0)))]);
    assert_eq!(state(&world, "menu"), Some(UiValue::Bool(true)));
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

// A click away reports the card's key; a card showing progress stays; the
// line counting the waiting cards reports a click on it.
#[test]
fn toast_clicks_report_the_card_and_the_waiting_line() {
    let mut world = with_state(&[], 0);
    world.add_component(Toasts {
        max_visible: 1,
        on_dismiss: "toast.gone".into(),
        on_more: "toast.more".into(),
        ..Default::default()
    });
    start(&mut world);
    world.events_mut::<Toast>().send(Toast {
        key: "saved".into(),
        ..Toast::new("saved", ToastTone::Success)
    });
    world.events_mut::<Toast>().send(Toast {
        key: "cook".into(),
        progress: Some(crate::components::ToastProgress::Ongoing),
        ..Toast::new("cooking", ToastTone::Info)
    });
    hover(&mut world, 0.0, 0.0);
    let rect = |slot| {
        world
            .query::<Toasts>()
            .next()
            .unwrap()
            .card_rect(slot, [1280.0, 720.0])
    };
    let (card, more) = (rect(0), rect(1));
    click(&mut world, card[0] + 10.0, card[1] + 10.0);
    assert_eq!(
        world.query::<Toasts>().next().unwrap().shown.len(),
        2,
        "the card showing progress stays"
    );
    click(&mut world, more[0] + 10.0, more[1] + 10.0);
    assert!(events(&world).contains(&("toast.more".into(), None)));
    if let Some(t) = world.query_mut::<Toasts>().next() {
        t.shown.swap(0, 1);
    }
    click(&mut world, card[0] + 10.0, card[1] + 10.0);
    assert!(events(&world).contains(&("toast.gone".into(), Some(UiValue::Text("saved".into())))));
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
        children: vec![LayoutChild {
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
        children: vec![LayoutChild {
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
