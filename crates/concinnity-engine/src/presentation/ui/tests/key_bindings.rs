// Key bindings and the views they show.

use super::*;

// The ScreenShown announcements the cursor has not yet consumed, read the
// way a real consumer (AudioCue) does: incrementally, before the queue's
// two-frame retention retires them.
fn shown_views(world: &World, cursor: &mut EventCursor) -> Vec<AssetId> {
    world
        .events::<ScreenShown>()
        .map(|e| e.read(cursor).map(|s| s.screen).collect())
        .unwrap_or_default()
}

// Both the initial screen at start and a Show navigation announce the newly
// active screen, so screen-triggered consumers (AudioCue) hear every screen.
#[test]
fn view_activation_emits_view_shown() {
    let mut world = World::new();
    let first = AssetId(80);
    let second = AssetId(81);
    for (id, initial) in [(first, true), (second, false)] {
        world.push_identified(
            id,
            Screen {
                initial,
                fade_in_secs: 0.0,
                ..Default::default()
            },
        );
    }
    start(&mut world);
    let mut cursor = EventCursor::default();
    assert_eq!(shown_views(&world, &mut cursor), vec![first]);

    world
        .events_mut::<ScreenCommand>()
        .send(ScreenCommand::Show(second));
    world.step();
    assert_eq!(shown_views(&world, &mut cursor), vec![second]);

    // Showing the already-active screen is a no-op: no repeat announcement.
    world
        .events_mut::<ScreenCommand>()
        .send(ScreenCommand::Show(second));
    world.step();
    assert!(shown_views(&world, &mut cursor).is_empty());
}

#[test]
fn escape_key_binding_fires_action() {
    let mut world = World::new();

    let screen_id = AssetId(50);
    world.push_identified(
        screen_id,
        Screen {
            initial: false,
            fade_in_secs: 0.0,
            ..Default::default()
        },
    );
    world.add_component(KeyBinding {
        key: "Escape".to_string(),
        action: Some(act(serde_json::json!({"toggle": 50}))),
        ..Default::default()
    });
    start(&mut world);

    // Press Escape.
    world.add_component(FrameInput {
        escape: true,
        ..Default::default()
    });
    world.step();

    let cmd = produced_screen_command(&world);
    assert!(matches!(cmd, Some(ScreenCommand::Toggle(AssetId(50)))));
}

// Every StoryCommand the system sent this step, in send order.
fn produced_story_commands(world: &World) -> Vec<StoryCommand> {
    let mut cursor = EventCursor::default();
    world
        .events::<StoryCommand>()
        .map(|e| e.read(&mut cursor).cloned().collect())
        .unwrap_or_default()
}

// A non-Escape KeyBinding fires when its key is pressed (surfaced as the
// frame's key events), so a story's Space / Enter advance bindings work.
#[test]
fn pressed_key_binding_fires_action() {
    for key in [InputKey::Space, InputKey::Enter] {
        let mut world = World::new();
        world.add_component(KeyBinding {
            key: key.name().to_string(),
            action: Some(act(serde_json::json!({"story": "advance"}))),
            ..Default::default()
        });
        start(&mut world);

        world.add_component(FrameInput {
            key_events: vec![KeyEvent::press(key)],
            ..Default::default()
        });
        world.step();

        assert_eq!(
            produced_story_commands(&world),
            vec![StoryCommand::Advance],
            "{key:?} should advance",
        );
    }
}

// A pressed key with no matching binding fires nothing (the arrow keys a
// scrollable list reads must not be swallowed by a phantom action).
#[test]
fn pressed_key_without_a_binding_fires_nothing() {
    let mut world = World::new();
    world.add_component(KeyBinding {
        key: "Space".to_string(),
        action: Some(act(serde_json::json!({"story": "advance"}))),
        ..Default::default()
    });
    start(&mut world);

    world.add_component(FrameInput {
        key_events: vec![KeyEvent::press(InputKey::Down)],
        ..Default::default()
    });
    world.step();

    assert!(produced_story_commands(&world).is_empty());
}

// Escape toggles the menu; Settings is reached by a Show (a sub-screen). After
// visiting Settings, escaping back to the menu and escaping again must return
// to the world, not back into Settings: a Toggle that opens a screen must not
// record the outgoing screen as the dismiss target.
#[test]
fn escape_from_menu_returns_to_world_after_visiting_a_subview() {
    let mut world = World::new();
    let menu = AssetId(60);
    let settings = AssetId(61);
    world.push_identified(
        menu,
        Screen {
            initial: false,
            fade_in_secs: 0.0,
            ..Default::default()
        },
    );
    world.push_identified(
        settings,
        Screen {
            initial: false,
            fade_in_secs: 0.0,
            ..Default::default()
        },
    );
    // One sprite per screen, to observe which screen is active by visibility.
    for (id, screen) in [(70u32, menu), (71u32, settings)] {
        world.push_identified(
            AssetId(id),
            Sprite {
                x: 0.0,
                y: 0.0,
                width: 10.0,
                height: 10.0,
                texture: None,
                tint: [0.0, 0.0, 0.0, 1.0],
                follow_cursor: false,
                visible: false,
                screen: Some(Ref::new(screen)),
                fit: SpriteFit::Fit,
                corner_radius: 0.0,
                border_width: 0.0,
                border_color: [0.0, 0.0, 0.0, 1.0],
            },
        );
    }
    world.add_component(KeyBinding {
        key: "Escape".to_string(),
        action: Some(act(serde_json::json!({"toggle": 60}))),
        ..Default::default()
    });
    start(&mut world);

    let shown = |w: &World, id: u32| {
        w.get_by_id::<Sprite>(AssetId(id))
            .map(|s| s.visible)
            .unwrap()
    };
    // A frame that presses Escape (fires the toggle keybinding).
    fn esc(w: &mut World) {
        w.add_component(FrameInput {
            escape: true,
            ..Default::default()
        });
        w.step();
    }
    // A frame that applies the screen command queued the previous frame.
    fn settle(w: &mut World) {
        w.add_component(FrameInput::default());
        w.step();
    }

    // World -> Escape -> menu.
    esc(&mut world);
    settle(&mut world);
    assert!(shown(&world, 70) && !shown(&world, 71), "menu opens");

    // Menu -> click Settings (a Show) -> settings sub-screen.
    world
        .events_mut::<ScreenCommand>()
        .send(ScreenCommand::Show(settings));
    settle(&mut world);
    assert!(!shown(&world, 70) && shown(&world, 71), "settings shown");

    // Settings -> Escape -> back to the menu.
    esc(&mut world);
    settle(&mut world);
    assert!(shown(&world, 70) && !shown(&world, 71), "back to the menu");

    // Menu -> Escape -> world (regression: previously returned to Settings).
    esc(&mut world);
    settle(&mut world);
    assert!(
        !shown(&world, 70) && !shown(&world, 71),
        "escape from the menu returns to the world, not back into Settings"
    );
}

// A Screen's `toggle_key` opens it from anywhere and closes it again,
// without any KeyBinding.
#[test]
fn toggle_key_opens_and_closes_a_screen() {
    let mut world = World::new();
    world.push_identified(
        AssetId(80),
        Screen {
            toggle_key: "Backtick".to_string(),
            ..Default::default()
        },
    );
    world.push_identified(
        AssetId(81),
        Sprite {
            width: 10.0,
            height: 10.0,
            visible: true,
            screen: Some(Ref::new(AssetId(80))),
            ..Default::default()
        },
    );
    start(&mut world);
    let shown = |w: &World| w.get_by_id::<Sprite>(AssetId(81)).unwrap().visible;
    assert!(!shown(&world), "screens start hidden");

    let backtick = |w: &mut World| {
        w.add_component(FrameInput {
            key_events: vec![KeyEvent::press(InputKey::Backtick)],
            ..Default::default()
        });
        w.step();
        // Settle frame: apply the command queued above.
        w.add_component(FrameInput::default());
        w.step();
    };
    backtick(&mut world);
    assert!(shown(&world), "toggle key opens the screen");
    backtick(&mut world);
    assert!(!shown(&world), "toggle key closes it again");
}

// A held toggle key's auto-repeat does not flip the screen back and forth:
// only the fresh press toggles.
#[test]
fn a_held_toggle_key_toggles_once() {
    let mut world = World::new();
    world.push_identified(
        AssetId(80),
        Screen {
            toggle_key: "Backtick".to_string(),
            ..Default::default()
        },
    );
    world.push_identified(
        AssetId(81),
        Sprite {
            width: 10.0,
            height: 10.0,
            visible: true,
            screen: Some(Ref::new(AssetId(80))),
            ..Default::default()
        },
    );
    start(&mut world);
    let shown = |w: &World| w.get_by_id::<Sprite>(AssetId(81)).unwrap().visible;
    let frame = |w: &mut World, repeat: bool| {
        let press = KeyPress {
            repeat,
            ..KeyPress::new(InputKey::Backtick, KeyMods::NONE)
        };
        w.add_component(FrameInput {
            key_events: vec![KeyEvent::Press(press)],
            ..Default::default()
        });
        w.step();
        w.add_component(FrameInput::default());
        w.step();
    };
    frame(&mut world, false);
    assert!(shown(&world), "the fresh press opens the screen");
    frame(&mut world, true);
    frame(&mut world, true);
    assert!(shown(&world), "the held key's repeats leave it open");
}
