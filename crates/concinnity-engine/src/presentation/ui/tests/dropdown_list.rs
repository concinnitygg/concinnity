// A settings dropdown's option list: opening, picking, scrolling, and
// dismissing it.

use super::*;

// A screen-owned dropdown row (window_mode has three options, so its
// `:open` region opens a floating list). Clicking the control opens the list
// (published as an OpenDropdown resource, no command yet); clicking an option
// sends a SetIndex command and closes.
fn dropdown_world() -> (World, AssetId) {
    let screen = AssetId(9);
    let mut world = World::new();
    world.push_identified(
        screen,
        Screen {
            initial: true,
            ..Default::default()
        },
    );
    // The row's value label (screen-owned), currently "Windowed" (option 0).
    world.push_identified(
        AssetId(1),
        TextLabel {
            font: None,
            content: "Windowed".to_string(),
            x: 0.0,
            y: 0.0,
            color: [0.85, 0.85, 0.85],
            color_runs: Vec::new(),
            scale: 1.0,
            centered: false,
            align: TextAlign::Left,
            fit: SpriteFit::Fit,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            visible: true,
            screen: Some(Ref::new(screen)),
            wrap_width: 0.0,
            max_lines: 0,
        },
    );
    // The control button whose click opens the list.
    world.add_component(HitRegion {
        x: 400.0,
        y: 100.0,
        width: 200.0,
        height: 40.0,
        label: Some(Ref::new(AssetId(1))),
        hover_color: Some([1.0, 0.85, 0.3]),
        hover_scale: Some(1.0),
        action: Some(act(
            serde_json::json!({"setting": {"key": "window_mode", "verb": "open"}}),
        )),
        drag_handle: None,
        screen: Some(Ref::new(screen)),
        disabled: false,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    start(&mut world);
    (world, screen)
}

fn dropdown_is_open(world: &World) -> bool {
    world
        .resource::<OpenDropdown>()
        .and_then(|d| d.0.as_ref())
        .is_some()
}

#[test]
fn dropdown_opens_then_picks_an_option() {
    let (mut world, _) = dropdown_world();

    // Click the control button (default viewport is identity, so window
    // coords are reference coords): the list opens, no command yet.
    world.add_component(make_frame_input(500.0, 120.0, true));
    world.step();
    assert!(produced_setting_commands(&world).is_empty());
    let open = world.resource::<OpenDropdown>().unwrap();
    let dv = open.0.as_ref().expect("dropdown should be open");
    assert_eq!(dv.options.len(), 3);
    assert_eq!(dv.selected, 0, "current value 'Windowed' is option 0");

    // The list opens below the button (rows at y = 140, 180, 220, height 40).
    // Click the second option ("Borderless") at y = 200.
    world.add_component(make_frame_input(500.0, 200.0, true));
    world.step();
    let cmds = produced_setting_commands(&world);
    assert_eq!(cmds.len(), 1);
    assert_eq!(cmds[0].setting, SettingKey::WindowMode);
    assert!(matches!(cmds[0].op, SettingOp::SetIndex(1)));
    assert!(!dropdown_is_open(&world), "picking closes the list");
}

// A dropdown over a runtime-enumerated list longer than the shown window
// (a `resolution` row with 20 display modes): opening centers the current
// selection, the wheel scrolls the window instead of dismissing, and a
// click picks the OPTION under the row (not the raw row index).
fn scrolled_dropdown_world() -> World {
    let screen = AssetId(9);
    let mut world = World::new();
    world.push_identified(
        screen,
        Screen {
            initial: true,
            ..Default::default()
        },
    );
    // 20 modes, 1000x100 (0Hz) .. 1000x2000 (0Hz); the row's value label
    // currently shows the 11th (index 10).
    let modes: Vec<display_mode::DisplayMode> = (1..=20)
        .map(|i| display_mode::DisplayMode {
            width: 1000,
            height: i * 100,
            refresh_hz: 0,
        })
        .collect();
    world.push_identified(
        AssetId(1),
        TextLabel {
            font: None,
            content: modes[10].label(),
            x: 0.0,
            y: 0.0,
            color: [0.85, 0.85, 0.85],
            color_runs: Vec::new(),
            scale: 1.0,
            centered: false,
            align: TextAlign::Left,
            fit: SpriteFit::Fit,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            visible: true,
            screen: Some(Ref::new(screen)),
            wrap_width: 0.0,
            max_lines: 0,
        },
    );
    world.add_component(HitRegion {
        x: 400.0,
        y: 100.0,
        width: 200.0,
        height: 40.0,
        label: Some(Ref::new(AssetId(1))),
        hover_color: Some([1.0, 0.85, 0.3]),
        hover_scale: Some(1.0),
        action: Some(act(
            serde_json::json!({"setting": {"key": "resolution", "verb": "open"}}),
        )),
        drag_handle: None,
        screen: Some(Ref::new(screen)),
        disabled: false,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    start(&mut world);
    world.insert_resource(display_mode::DisplayModes(modes));
    world
}

#[test]
fn dropdown_scrolls_instead_of_dismissing() {
    let mut world = scrolled_dropdown_world();

    // Open: the full option list is carried, the window starts centered on
    // the selection (10 - MAX_VISIBLE/2).
    world.add_component(make_frame_input(500.0, 120.0, true));
    world.step();
    let center = 10 - dropdown::MAX_VISIBLE / 2;
    {
        let open = world.resource::<OpenDropdown>().unwrap();
        let dv = open.0.as_ref().expect("dropdown should be open");
        assert_eq!(dv.options.len(), 20);
        assert_eq!(dv.selected, 10);
        assert_eq!(dv.first, center);
    }

    // Wheel: the list stays open and the window moves (40px * speed 2.0 /
    // 40px rows = +2 rows), clamped to the scrollable range.
    world.add_component(FrameInput {
        mouse_x: 500.0,
        mouse_y: 200.0,
        scroll_delta: 40.0,
        ..Default::default()
    });
    world.step();
    {
        let open = world.resource::<OpenDropdown>().unwrap();
        let dv = open.0.as_ref().expect("scrolling must not dismiss");
        assert_eq!(dv.first, center + 2);
    }

    // Click the top shown row (y 140..180): the pick is the OPTION at the
    // scrolled window's top, not row 0 of the full list.
    world.add_component(make_frame_input(500.0, 150.0, true));
    world.step();
    let cmds = produced_setting_commands(&world);
    assert_eq!(cmds.len(), 1);
    assert_eq!(cmds[0].setting, SettingKey::Resolution);
    assert!(matches!(cmds[0].op, SettingOp::SetIndex(i) if i == center + 2));
    assert!(!dropdown_is_open(&world));
}

// The Graphics API row lists the platforms the host published as compiled
// in, selecting the one its value label shows; with none published the
// row stays closed.
#[test]
fn graphics_api_dropdown_lists_the_compiled_platforms() {
    use concinnity_core::platform::Platform;

    let screen = AssetId(9);
    let mut world = World::new();
    world.push_identified(
        screen,
        Screen {
            initial: true,
            ..Default::default()
        },
    );
    world.push_identified(AssetId(1), panel_label(0.0, screen, "Vulkan"));
    world.add_component(HitRegion {
        x: 400.0,
        y: 100.0,
        width: 200.0,
        height: 40.0,
        label: Some(Ref::new(AssetId(1))),
        hover_color: None,
        hover_scale: None,
        action: Some(act(
            serde_json::json!({"setting": {"key": "graphics_api", "verb": "open"}}),
        )),
        drag_handle: None,
        screen: Some(Ref::new(screen)),
        disabled: false,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    start(&mut world);

    world.add_component(make_frame_input(500.0, 120.0, true));
    world.step();
    assert!(!dropdown_is_open(&world), "no compiled platforms published");

    world.insert_resource(CompiledPlatforms(&[Platform::Metal, Platform::Vulkan]));
    world.add_component(make_frame_input(500.0, 120.0, true));
    world.step();
    let open = world.resource::<OpenDropdown>().unwrap();
    let dv = open.0.as_ref().expect("dropdown should be open");
    assert_eq!(dv.options, ["Metal", "Vulkan"]);
    assert_eq!(dv.selected, 1);
}

#[test]
fn dropdown_outside_click_dismisses_without_command() {
    let (mut world, _) = dropdown_world();

    world.add_component(make_frame_input(500.0, 120.0, true));
    world.step();
    assert!(dropdown_is_open(&world));

    // Click far from both the list and the button: it dismisses, no command.
    world.add_component(make_frame_input(50.0, 600.0, true));
    world.step();
    assert!(produced_setting_commands(&world).is_empty());
    assert!(!dropdown_is_open(&world));
}

// The open list's first shown option, from the published OpenDropdown
// resource.
fn dropdown_first(world: &World) -> usize {
    world
        .resource::<OpenDropdown>()
        .and_then(|d| d.0.as_ref())
        .map(|dv| dv.first)
        .expect("dropdown should be open")
}

// Dragging the open list's scrollbar thumb scrolls the window: a press on
// the thumb neither picks nor dismisses, the held cursor's y drives the
// window, and after release a click picks again.
#[test]
fn dropdown_thumb_drag_scrolls_window_without_picking() {
    let mut world = scrolled_dropdown_world();

    // Open: 20 options, window starts at first = 6 (selection centered).
    // The list is [400, 140, 200, 320]; the 28px thumb sits at y = 286
    // (halfway along its 292px travel, first 6 of max 12).
    world.add_component(make_frame_input(500.0, 120.0, true));
    world.step();
    assert_eq!(dropdown_first(&world), 6);

    // Press on the thumb (thumb x 584..598): the drag begins, nothing is
    // picked even though an option row sits under the cursor.
    world.add_component(FrameInput {
        mouse_x: 590.0,
        mouse_y: 290.0,
        left_click: true,
        left_button_down: true,
        ..Default::default()
    });
    world.step();
    assert!(dropdown_is_open(&world), "a thumb press must not pick");
    assert!(produced_setting_commands(&world).is_empty());
    assert_eq!(dropdown_first(&world), 6, "grab does not jump the window");

    // Drag past the bottom of the list: the window clamps to the end.
    world.add_component(FrameInput {
        mouse_x: 590.0,
        mouse_y: 600.0,
        left_button_down: true,
        ..Default::default()
    });
    world.step();
    assert!(dropdown_is_open(&world));
    assert_eq!(dropdown_first(&world), 12);

    // Release, then click the top shown row: picking works again and maps
    // to the dragged window (option 12).
    world.add_component(make_frame_input(590.0, 400.0, false));
    world.step();
    world.add_component(make_frame_input(500.0, 150.0, true));
    world.step();
    let cmds = produced_setting_commands(&world);
    assert_eq!(cmds.len(), 1);
    assert!(matches!(cmds[0].op, SettingOp::SetIndex(12)));
    assert!(!dropdown_is_open(&world));
}

// A press on the scrollbar strip beside the thumb jumps the window there
// (and starts a drag) instead of picking the option row behind it.
#[test]
fn dropdown_track_click_jumps_instead_of_picking() {
    let mut world = scrolled_dropdown_world();
    world.add_component(make_frame_input(500.0, 120.0, true));
    world.step();
    assert_eq!(dropdown_first(&world), 6);

    // Press near the top of the strip (x 582..600), well above the thumb.
    world.add_component(FrameInput {
        mouse_x: 590.0,
        mouse_y: 145.0,
        left_click: true,
        left_button_down: true,
        ..Default::default()
    });
    world.step();
    assert!(dropdown_is_open(&world), "a track press must not pick");
    assert!(produced_setting_commands(&world).is_empty());
    assert_eq!(dropdown_first(&world), 0, "the window jumps to the press");
}

// The open list's highlighted option, from the published resource.
fn dropdown_hovered(world: &World) -> Option<usize> {
    world
        .resource::<OpenDropdown>()
        .and_then(|d| d.0.as_ref())
        .expect("dropdown should be open")
        .hovered
}

// Nav pulses (arrow keys / pad) move the open list's selection highlight,
// scrolling the shown window along with it, and Enter picks the
// highlighted option. The cursor stays put throughout, so the idle hover
// never re-asserts the row under it.
#[test]
fn dropdown_nav_moves_selection_and_enter_picks() {
    let mut world = scrolled_dropdown_world();
    world.add_component(make_frame_input(500.0, 120.0, true));
    world.step();
    assert_eq!(dropdown_first(&world), 6);

    let pulse_down = || FrameInput {
        mouse_x: 500.0,
        mouse_y: 120.0,
        key_events: vec![KeyEvent::press(InputKey::Down)],
        ..Default::default()
    };

    // Down steps from the current value (10); the window (6..14) holds.
    world.add_component(pulse_down());
    world.step();
    assert!(dropdown_is_open(&world), "nav must not dismiss");
    assert_eq!(dropdown_hovered(&world), Some(11));
    assert_eq!(dropdown_first(&world), 6);

    // Walking to option 14 crosses the window's bottom edge: the window
    // follows the selection.
    for _ in 0..3 {
        world.add_component(pulse_down());
        world.step();
    }
    assert_eq!(dropdown_hovered(&world), Some(14));
    assert_eq!(dropdown_first(&world), 7);

    // Enter picks the highlighted option and closes the list.
    world.add_component(FrameInput {
        mouse_x: 500.0,
        mouse_y: 120.0,
        key_events: vec![KeyEvent::press(InputKey::Enter)],
        ..Default::default()
    });
    world.step();
    let cmds = produced_setting_commands(&world);
    assert_eq!(cmds.len(), 1);
    assert!(matches!(cmds[0].op, SettingOp::SetIndex(14)));
    assert!(!dropdown_is_open(&world));
}
