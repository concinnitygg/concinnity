// The screen stack and the focus field it owns.

use super::*;

// A screen's `focus` TextInput gains keyboard focus when the screen
// reaches the top of the stack, and loses it when the screen closes.
#[test]
fn focus_field_follows_the_top_screen() {
    let mut world = World::new();
    world.push_identified(
        AssetId(90),
        Screen {
            toggle_key: "Backtick".to_string(),
            focus: Some(Ref::new(AssetId(91))),
            ..Default::default()
        },
    );
    world.push_identified(
        AssetId(91),
        TextInput {
            visible: true,
            screen: Some(Ref::new(AssetId(90))),
            ..Default::default()
        },
    );
    start(&mut world);
    let focused = |w: &World| {
        w.get_by_id::<TextInput>(AssetId(91))
            .map(|t| (t.visible, t.focused))
            .unwrap()
    };
    assert_eq!(focused(&world), (false, false), "hidden until shown");

    let backtick = |w: &mut World| {
        w.add_component(FrameInput {
            key_events: vec![KeyEvent::press(InputKey::Backtick)],
            ..Default::default()
        });
        w.step();
        w.add_component(FrameInput::default());
        w.step();
    };
    backtick(&mut world);
    assert_eq!(focused(&world), (true, true), "shown and focused");
    backtick(&mut world);
    assert_eq!(focused(&world), (false, false), "blurred on close");
}

// While a visible TextInput has keyboard focus, ordinary KeyBindings are
// suspended so typing cannot fire actions.
#[test]
fn keybindings_are_suspended_while_typing() {
    let mut world = World::new();
    world.push_identified(AssetId(100), Screen::default());
    world.add_component(KeyBinding {
        key: "T".to_string(),
        action: Some(act(serde_json::json!({"toggle": 100}))),
        ..Default::default()
    });
    let mut field = TextInput {
        visible: true,
        ..Default::default()
    };
    field.focused = true;
    world.push_identified(AssetId(101), field);
    start(&mut world);

    world.add_component(FrameInput {
        key_events: vec![KeyEvent::press(InputKey::T)],
        ..Default::default()
    });
    world.step();
    assert!(
        produced_screen_command(&world).is_none(),
        "typed key does not fire the binding"
    );

    // Blur the field: the same key now fires the binding.
    for ti in world.query_mut::<TextInput>() {
        ti.focused = false;
    }
    world.add_component(FrameInput {
        key_events: vec![KeyEvent::press(InputKey::T)],
        ..Default::default()
    });
    world.step();
    assert!(matches!(
        produced_screen_command(&world),
        Some(ScreenCommand::Toggle(AssetId(100)))
    ));
}

// P saves one frame and Shift+P a burst, each press asking once.
#[test]
fn screenshot_bindings_request_their_frame_counts() {
    let mut world = World::new();
    world.add_component(KeyBinding {
        key: "P".to_string(),
        action: Some(act(serde_json::json!({"screenshot": 1}))),
        ..Default::default()
    });
    world.add_component(KeyBinding {
        key: "P".to_string(),
        mods: Some(KeyMods::SHIFT),
        action: Some(act(serde_json::json!({"screenshot": 8}))),
        ..Default::default()
    });
    start(&mut world);
    let mut press = |mods| {
        world.add_component(FrameInput {
            key_events: vec![KeyEvent::Press(KeyPress::new(InputKey::P, mods))],
            ..Default::default()
        });
        world.step();
        world
            .remove_resource::<crate::protocol::ScreenshotRequest>()
            .map(|r| r.0)
    };
    assert_eq!(press(KeyMods::NONE), Some(1));
    assert_eq!(press(KeyMods::SHIFT), Some(8));
    world.add_component(FrameInput::default());
    world.step();
    assert!(
        world
            .resource::<crate::protocol::ScreenshotRequest>()
            .is_none()
    );
}

// Each press of a `toggle_hud` binding flips the stats HUD's visibility.
#[test]
fn a_toggle_hud_binding_flips_the_hud() {
    let mut world = World::new();
    world.add_component(KeyBinding {
        key: "H".to_string(),
        action: Some(act(serde_json::json!("toggle_hud"))),
        ..Default::default()
    });
    start(&mut world);
    let mut press = || {
        world.add_component(FrameInput {
            key_events: vec![KeyEvent::press(InputKey::H)],
            ..Default::default()
        });
        world.step();
        world.resource::<crate::protocol::HudHidden>().copied()
    };
    assert_eq!(press(), Some(crate::protocol::HudHidden(true)));
    assert_eq!(press(), Some(crate::protocol::HudHidden(false)));
}

// A KeyBinding scoped to a screen fires only while that screen is on top.
#[test]
fn scoped_keybinding_fires_only_while_its_screen_is_top() {
    let mut world = World::new();
    world.push_identified(AssetId(110), Screen::default());
    world.push_identified(AssetId(111), Screen::default());
    world.add_component(KeyBinding {
        key: "Space".to_string(),
        action: Some(act(serde_json::json!({"show": 111}))),
        screen: Some(Ref::new(AssetId(110))),
        ..Default::default()
    });
    start(&mut world);

    // No screen on top: the scoped binding stays quiet.
    world.add_component(FrameInput {
        key_events: vec![KeyEvent::press(InputKey::Space)],
        ..Default::default()
    });
    world.step();
    assert!(produced_screen_command(&world).is_none());

    // Open its screen; the binding now fires.
    world
        .events_mut::<ScreenCommand>()
        .send(ScreenCommand::Show(AssetId(110)));
    world.add_component(FrameInput::default());
    world.step();
    world.add_component(FrameInput {
        key_events: vec![KeyEvent::press(InputKey::Space)],
        ..Default::default()
    });
    world.step();
    let mut cursor = EventCursor::default();
    let sent: Vec<ScreenCommand> = world
        .events::<ScreenCommand>()
        .map(|e| e.read(&mut cursor).copied().collect())
        .unwrap_or_default();
    assert!(
        sent.iter()
            .any(|c| matches!(c, ScreenCommand::Show(AssetId(111)))),
        "scoped binding fires on top: {sent:?}"
    );
}

// `push` stacks a screen over the current one: both stay visible,
// and hiding the pushed screen reveals the one beneath.
#[test]
fn push_stacks_over_the_current_screen() {
    let mut world = World::new();
    for (screen, sprite) in [(120u32, 130u32), (121, 131)] {
        world.push_identified(AssetId(screen), Screen::default());
        world.push_identified(
            AssetId(sprite),
            Sprite {
                width: 10.0,
                height: 10.0,
                visible: true,
                screen: Some(Ref::new(AssetId(screen))),
                ..Default::default()
            },
        );
    }
    start(&mut world);
    let shown = |w: &World, id: u32| w.get_by_id::<Sprite>(AssetId(id)).unwrap().visible;
    let settle = |w: &mut World| {
        w.add_component(FrameInput::default());
        w.step();
    };
    world
        .events_mut::<ScreenCommand>()
        .send(ScreenCommand::Show(AssetId(120)));
    settle(&mut world);
    world
        .events_mut::<ScreenCommand>()
        .send(ScreenCommand::Push(AssetId(121)));
    settle(&mut world);
    assert!(
        shown(&world, 130) && shown(&world, 131),
        "both screens visible while stacked"
    );
    // The stack resource carries both layers, in stack order.
    {
        let stack = world.resource::<ScreenStack>().unwrap();
        assert_eq!(stack.layers[&AssetId(120)], 1);
        assert_eq!(stack.layers[&AssetId(121)], 2);
        assert!(stack.pauses_world && stack.captures_input);
    }
    world
        .events_mut::<ScreenCommand>()
        .send(ScreenCommand::Hide);
    settle(&mut world);
    assert!(
        shown(&world, 130) && !shown(&world, 131),
        "hide pops the pushed screen, revealing the one beneath"
    );
}
