// Keyboard and gamepad focus across a menu.

use super::*;

// A menu screen (id 90) with two buttons at y 100 / 200 whose labels take a
// hover color, plus a second screen (91) the first button navigates to.
fn focus_menu_world() -> World {
    let mut world = World::new();
    let menu = AssetId(90);
    world.push_identified(
        menu,
        Screen {
            initial: true,
            fade_in_secs: 0.0,
            ..Default::default()
        },
    );
    world.push_identified(
        AssetId(91),
        Screen {
            initial: false,
            fade_in_secs: 0.0,
            ..Default::default()
        },
    );
    for (id, y, action) in [
        (1u32, 100.0f32, serde_json::json!({"show": 91})),
        (2, 200.0, serde_json::json!("hide")),
    ] {
        world.push_identified(AssetId(id), panel_label(y, menu, "Btn"));
        world.add_component(HitRegion {
            x: 0.0,
            y,
            width: 200.0,
            height: 40.0,
            label: Some(Ref::new(AssetId(id))),
            hover_color: Some([1.0, 0.85, 0.3]),
            hover_scale: Some(1.0),
            action: Some(act(action)),
            drag_handle: None,
            screen: Some(Ref::new(menu)),
            disabled: false,
            follow_label: false,
            fit: SpriteFit::Fit,
        });
    }
    start(&mut world);
    world
}

// A nav pulse focuses the first button (styled like hover), further pulses
// walk the list, and confirm fires the focused button's action.
#[test]
fn nav_focuses_buttons_and_confirm_fires_the_focused_one() {
    let mut world = focus_menu_world();

    world.add_component(FrameInput {
        nav: Some(NavDirection::Down),
        ..Default::default()
    });
    world.step();
    assert_eq!(
        label_field(&world, AssetId(1), |l| l.color),
        [1.0, 0.85, 0.3],
        "first pulse focuses + styles the top button"
    );

    // The next pulse moves focus to the second button; the first restores.
    world.add_component(FrameInput {
        nav: Some(NavDirection::Down),
        ..Default::default()
    });
    world.step();
    assert_eq!(
        label_field(&world, AssetId(1), |l| l.color),
        [1.0, 1.0, 1.0]
    );
    assert_eq!(
        label_field(&world, AssetId(2), |l| l.color),
        [1.0, 0.85, 0.3]
    );

    // Back up to the first, then confirm: its action fires.
    world.add_component(FrameInput {
        nav: Some(NavDirection::Up),
        ..Default::default()
    });
    world.step();
    world.add_component(FrameInput {
        confirm: true,
        ..Default::default()
    });
    world.step();
    assert!(matches!(
        produced_screen_command(&world),
        Some(ScreenCommand::Show(AssetId(91)))
    ));
}

// Mouse movement dismisses the focus cursor: the focused label restores
// and a later confirm no longer fires anything.
#[test]
fn mouse_movement_dismisses_focus() {
    let mut world = focus_menu_world();

    world.add_component(FrameInput {
        nav: Some(NavDirection::Down),
        ..Default::default()
    });
    world.step();
    assert_eq!(
        label_field(&world, AssetId(1), |l| l.color),
        [1.0, 0.85, 0.3]
    );

    // The cursor moves (well away from both buttons): focus clears.
    world.add_component(make_frame_input(600.0, 600.0, false));
    world.step();
    assert_eq!(
        label_field(&world, AssetId(1), |l| l.color),
        [1.0, 1.0, 1.0],
        "moving the mouse restores the focused label"
    );

    world.add_component(FrameInput {
        mouse_x: 600.0,
        mouse_y: 600.0,
        confirm: true,
        ..Default::default()
    });
    world.step();
    assert!(
        produced_screen_command(&world).is_none(),
        "no focus, nothing fires"
    );
}

// A settings screen with a stepper row and a slider row: Left/Right on the
// focused row send Prev/Next for its setting instead of moving focus.
#[test]
fn focus_left_right_adjusts_value_rows() {
    let mut world = World::new();
    let screen = AssetId(95);
    world.push_identified(
        screen,
        Screen {
            initial: true,
            fade_in_secs: 0.0,
            ..Default::default()
        },
    );
    world.push_identified(AssetId(1), panel_label(100.0, screen, "Vsync"));
    world.push_identified(AssetId(2), panel_label(200.0, screen, "50"));
    // The stepper's two regions (prev + next) share the row.
    for (x, suffix) in [(200.0, "prev"), (260.0, "next")] {
        world.add_component(HitRegion {
            x,
            y: 100.0,
            width: 50.0,
            height: 40.0,
            label: Some(Ref::new(AssetId(1))),
            hover_color: Some([1.0, 0.85, 0.3]),
            hover_scale: Some(1.0),
            action: Some(act(
                serde_json::json!({"setting": {"key": "vsync", "verb": suffix}}),
            )),
            drag_handle: None,
            screen: Some(Ref::new(screen)),
            disabled: false,
            follow_label: false,
            fit: SpriteFit::Fit,
        });
    }
    world.add_component(HitRegion {
        x: 200.0,
        y: 200.0,
        width: 110.0,
        height: 40.0,
        label: Some(Ref::new(AssetId(2))),
        hover_color: Some([1.0, 0.85, 0.3]),
        hover_scale: Some(1.0),
        action: Some(act(
            serde_json::json!({"setting": {"key": "exposure", "verb": "drag"}}),
        )),
        drag_handle: None,
        screen: Some(Ref::new(screen)),
        disabled: false,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    start(&mut world);

    // Focus the stepper row (its two regions read as one target).
    world.add_component(FrameInput {
        nav: Some(NavDirection::Down),
        ..Default::default()
    });
    world.step();
    assert_eq!(
        label_field(&world, AssetId(1), |l| l.color),
        [1.0, 0.85, 0.3]
    );

    // Right cycles forward, Left back; focus stays on the row.
    for (dir, expect_prev) in [(NavDirection::Right, false), (NavDirection::Left, true)] {
        world.add_component(FrameInput {
            nav: Some(dir),
            ..Default::default()
        });
        world.step();
        let cmds = produced_setting_commands(&world);
        let cmd = cmds.last().unwrap();
        assert_eq!(cmd.setting, SettingKey::Vsync);
        assert_eq!(
            matches!(cmd.op, SettingOp::Prev),
            expect_prev,
            "Left sends Prev, Right sends Next"
        );
        assert!(cmd.persist);
    }
    assert_eq!(
        label_field(&world, AssetId(1), |l| l.color),
        [1.0, 0.85, 0.3],
        "adjusting keeps the row focused"
    );

    // Down reaches the slider row (styled too); Right steps its setting.
    world.add_component(FrameInput {
        nav: Some(NavDirection::Down),
        ..Default::default()
    });
    world.step();
    assert_eq!(
        label_field(&world, AssetId(2), |l| l.color),
        [1.0, 0.85, 0.3]
    );
    world.add_component(FrameInput {
        nav: Some(NavDirection::Right),
        ..Default::default()
    });
    world.step();
    let cmds = produced_setting_commands(&world);
    let cmd = cmds.last().unwrap();
    assert_eq!(cmd.setting, SettingKey::Exposure);
    assert!(matches!(cmd.op, SettingOp::Next));
}

// The pad's back pulse mirrors Escape only while a screen is active: with
// nothing open it must NOT open an Escape-toggled menu, and with the menu
// open it closes it.
#[test]
fn back_mirrors_escape_only_while_a_screen_is_active() {
    let mut world = World::new();
    let menu = AssetId(97);
    world.push_identified(
        menu,
        Screen {
            initial: false,
            fade_in_secs: 0.0,
            toggle_key: "Escape".to_string(),
            ..Default::default()
        },
    );
    start(&mut world);

    // Back with no screen active: nothing opens.
    world.add_component(FrameInput {
        back: true,
        ..Default::default()
    });
    world.step();
    assert!(produced_screen_command(&world).is_none());

    // Escape opens the menu (toggle); a frame applies it.
    world.add_component(FrameInput {
        escape: true,
        ..Default::default()
    });
    world.step();
    world.add_component(FrameInput::default());
    world.step();
    assert!(
        world
            .resource::<ScreenStack>()
            .is_some_and(|s| s.captures_input),
        "escape toggles the menu open"
    );

    // Back now mirrors Escape: the toggle pops the menu.
    world.add_component(FrameInput {
        back: true,
        ..Default::default()
    });
    world.step();
    world.add_component(FrameInput::default());
    world.step();
    assert!(
        world
            .resource::<ScreenStack>()
            .is_some_and(|s| !s.captures_input),
        "back closes the open menu"
    );
}

// Confirm with no focus fires a full-canvas "press anywhere" region (a
// story stage's advance region), which is itself excluded from focus.
#[test]
fn confirm_without_focus_fires_a_full_canvas_region() {
    let mut world = World::new();
    let stage = AssetId(98);
    world.push_identified(
        stage,
        Screen {
            initial: true,
            fade_in_secs: 0.0,
            ..Default::default()
        },
    );
    world.add_component(HitRegion {
        x: 0.0,
        y: 0.0,
        width: UI_REFERENCE_SIZE[0],
        height: UI_REFERENCE_SIZE[1],
        label: None,
        hover_color: None,
        hover_scale: None,
        action: Some(act(serde_json::json!({"story": "advance"}))),
        drag_handle: None,
        screen: Some(Ref::new(stage)),
        disabled: false,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    start(&mut world);

    world.add_component(FrameInput {
        confirm: true,
        ..Default::default()
    });
    world.step();
    let mut cursor = EventCursor::default();
    let advanced = world.events::<StoryCommand>().is_some_and(|e| {
        e.read(&mut cursor)
            .into_iter()
            .any(|c| *c == StoryCommand::Advance)
    });
    assert!(advanced, "South advances the stage without a focus cursor");
}

// While a pad rebind row is capturing, the East button is a bindable
// button, not a back pulse: it binds instead of canceling.
#[test]
fn pad_capture_binds_east_instead_of_backing_out() {
    let mut world = World::new();
    let mut label = panel_label(0.0, AssetId(0), "South");
    label.screen = None;
    world.push_identified(AssetId(7), label);
    world.add_component(HitRegion {
        x: 0.0,
        y: 0.0,
        width: 100.0,
        height: 40.0,
        label: Some(Ref::new(AssetId(7))),
        hover_color: None,
        hover_scale: None,
        action: Some(act(
            serde_json::json!({"setting": {"key": "pad_jump", "verb": "rebind"}}),
        )),
        drag_handle: None,
        screen: None,
        disabled: false,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    start(&mut world);

    // Click the row: capture begins.
    world.add_component(make_frame_input(50.0, 20.0, true));
    world.step();

    // Press East (which also raises the back pulse): it binds.
    world.add_component(FrameInput {
        captured_button: Some(GamepadButton::East),
        back: true,
        ..Default::default()
    });
    world.step();
    let cmds = produced_setting_commands(&world);
    assert_eq!(cmds.len(), 1);
    assert_eq!(cmds[0].setting, SettingKey::PadRebind(GamepadAction::Jump));
    assert!(matches!(
        cmds[0].op,
        SettingOp::RebindButton(GamepadButton::East)
    ));
}
