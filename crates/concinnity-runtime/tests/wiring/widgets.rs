// Widgets driven through the runtime's whole table, under the debug-build
// access validator: every context touch the UI systems make while popups,
// lists, text areas, and panels take input must be one they declared. A
// steady frame over them allocates nothing.

use concinnity_engine::components::{
    Binding, Checkbox, FrameInput, KeyEvent, Layout, List, Menu, MenuEntry, Modal, ModalButton,
    Screen, Tabs, TextArea, TextField, Toast, ToastTone, Toasts, UiItem, UiState, UiValue,
};
use concinnity_engine::ecs::Ref;
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;
use concinnity_engine::presentation::text_edit::Metrics;
use concinnity_runtime::SYSTEMS;

const SCREEN: AssetId = AssetId(900);
const VIEWPORT: [f32; 2] = [1280.0, 720.0];

fn on_screen() -> Option<Ref<Screen>> {
    Ref::new(SCREEN).into()
}

// A persistent screen holding one of every widget, a floating panel, and a
// text field, over UI state that fills the list and opens the menu.
fn widget_world() -> World {
    let mut world = World::new();
    world.push_identified(
        SCREEN,
        Screen {
            persistent: true,
            pauses_world: false,
            ..Default::default()
        },
    );
    let mut state = UiState::default();
    state.set("row", UiValue::Int(-1));
    state.set("grid", UiValue::Bool(false));
    state.set("notes", UiValue::Text("fn main".into()));
    state.set("menu", UiValue::Bool(false));
    state.set("ask", UiValue::Bool(false));
    state.set("tab", UiValue::Int(0));
    let items = (0..50).map(|i| UiItem::new(&format!("row {i}"))).collect();
    state.set_list("files", items);
    world.insert_resource(state);
    world.add_component(List {
        items: "files".into(),
        selected: Some(Binding::State("row".into())),
        width: 200.0,
        height: 200.0,
        screen: on_screen(),
        ..Default::default()
    });
    world.add_component(Tabs {
        tabs: vec!["A".into(), "B".into()],
        selected: Some(Binding::State("tab".into())),
        y: 300.0,
        screen: on_screen(),
        ..Default::default()
    });
    world.add_component(Checkbox {
        checked: Some(Binding::State("grid".into())),
        y: 340.0,
        screen: on_screen(),
        ..Default::default()
    });
    world.add_component(TextArea {
        text: Some(Binding::State("notes".into())),
        x: 300.0,
        metrics: Some(Metrics {
            line_h: 20.0,
            advance: 10.0,
            text_px: 16.0,
            label_scale: 1.0,
        }),
        screen: on_screen(),
        ..Default::default()
    });
    world.add_component(TextField {
        x: 300.0,
        y: 400.0,
        width: 200.0,
        height: 30.0,
        screen: on_screen(),
        ..Default::default()
    });
    world.add_component(Menu {
        open: Some(Binding::State("menu".into())),
        at_cursor: true,
        entries: vec![MenuEntry {
            label: "Pick".into(),
            ..Default::default()
        }],
        screen: on_screen(),
        ..Default::default()
    });
    world.add_component(Modal {
        open: Some(Binding::State("ask".into())),
        buttons: vec![ModalButton {
            label: "OK".into(),
            ..Default::default()
        }],
        screen: on_screen(),
        ..Default::default()
    });
    world.add_component(Toasts {
        screen: on_screen(),
        ..Default::default()
    });
    world.add_component(Layout {
        movable: true,
        resizable: true,
        rect: Some([800.0, 100.0, 300.0, 200.0]),
        screen: on_screen(),
        ..Default::default()
    });
    world
}

fn frame(x: f32, y: f32) -> FrameInput {
    FrameInput {
        mouse_x: x,
        mouse_y: y,
        viewport: VIEWPORT,
        ..Default::default()
    }
}

fn press(x: f32, y: f32) -> FrameInput {
    FrameInput {
        left_click: true,
        left_button_down: true,
        ..frame(x, y)
    }
}

fn state(world: &World, name: &str) -> Option<UiValue> {
    world
        .resource::<UiState>()
        .and_then(|s| s.get(name).cloned())
}

#[test]
fn widgets_take_input_within_their_declared_access() {
    let mut world = widget_world();
    world.start(SYSTEMS).unwrap();
    let step = |world: &mut World, input: FrameInput| {
        world.add_component(input);
        world.step();
    };

    step(&mut world, press(50.0, 40.0));
    assert_eq!(state(&world, "row"), Some(UiValue::Int(1)), "a row selects");
    step(&mut world, press(10.0, 345.0));
    assert_eq!(state(&world, "grid"), Some(UiValue::Bool(true)));

    step(&mut world, press(400.0, 20.0));
    step(
        &mut world,
        FrameInput {
            key_events: vec![KeyEvent::Text('!')],
            ..frame(400.0, 20.0)
        },
    );
    assert!(
        matches!(state(&world, "notes"), Some(UiValue::Text(t)) if t.contains('!')),
        "typing into the focused text area writes its binding"
    );

    step(&mut world, press(810.0, 105.0));
    step(
        &mut world,
        FrameInput {
            left_button_down: true,
            ..frame(860.0, 125.0)
        },
    );
    step(&mut world, frame(860.0, 125.0));
    let panel = world.query::<Layout>().next().unwrap();
    assert_eq!(panel.offset, [50.0, 20.0], "the panel moved");

    if let Some(s) = world.resource_mut::<UiState>() {
        s.set("menu", UiValue::Bool(true));
        s.set("ask", UiValue::Bool(true));
    }
    world
        .events_mut::<Toast>()
        .send(Toast::new("done", ToastTone::Success));
    step(&mut world, frame(600.0, 600.0));
    step(&mut world, press(600.0, 600.0));
    step(
        &mut world,
        FrameInput {
            escape: true,
            ..frame(600.0, 600.0)
        },
    );
    assert_eq!(state(&world, "ask"), Some(UiValue::Bool(false)));
}

#[test]
fn a_steady_frame_over_widgets_allocates_nothing() {
    let mut world = widget_world();
    world.start(SYSTEMS).unwrap();
    world.add_component(frame(50.0, 40.0));
    for _ in 0..64 {
        world.step();
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut min = u64::MAX;
    while min > 0 && std::time::Instant::now() < deadline {
        let before = concinnity_core::memory::alloc_count().expect("the binary tracks its heap");
        world.step();
        let after = concinnity_core::memory::alloc_count().expect("the allocator stays installed");
        min = min.min(after - before);
    }
    assert_eq!(
        min,
        0,
        "a steady widget frame allocated {min} times: {:?}",
        world.profile().system_allocs()
    );
}
