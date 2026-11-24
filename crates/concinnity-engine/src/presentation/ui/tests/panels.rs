// Scroll panels: sliders, collapsible groups, the wheel and the scrollbar.

use super::*;

#[test]
fn slider_drag_pushes_set_fraction_then_persists_on_release() {
    let mut world = World::new();
    world.add_component(HitRegion {
        x: 100.0,
        y: 0.0,
        width: 200.0,
        height: 40.0,
        label: Some(Ref::new(AssetId(7))),
        hover_color: None,
        hover_scale: None,
        action: None,
        slider: Some(Binding::Setting(SettingKey::Exposure)),
        drag_handle: Some(Ref::new(AssetId(8))),
        screen: None,
        disabled: false,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    start(&mut world);

    // Press at x=150 (25% across the [100, 300) track) with the button held:
    // a live, non-persisting fraction.
    world.add_component(FrameInput {
        mouse_x: 150.0,
        mouse_y: 20.0,
        left_click: true,
        left_button_down: true,
        ..Default::default()
    });
    world.step();
    let cmd = produced_setting_commands(&world)
        .into_iter()
        .last()
        .unwrap();
    assert_eq!(cmd.setting, SettingKey::Exposure);
    assert!(matches!(cmd.op, SettingOp::SetFraction(f) if (f - 0.25).abs() < 1.0e-5));
    assert!(
        !cmd.persist,
        "an in-progress drag applies live but does not persist"
    );

    // Release at x=250 (75%): the button up commits the final value and persists.
    world.add_component(FrameInput {
        mouse_x: 250.0,
        mouse_y: 20.0,
        left_click: false,
        left_button_down: false,
        ..Default::default()
    });
    world.step();
    let cmd = produced_setting_commands(&world)
        .into_iter()
        .last()
        .unwrap();
    assert!(matches!(cmd.op, SettingOp::SetFraction(f) if (f - 0.75).abs() < 1.0e-5));
    assert!(cmd.persist, "release commits and persists the final value");
}

// A header flipping the group's bound state collapses the group's body rows (hiding their
// elements) and flips the header's `+`/`-` prefix; the body's click region
// then goes inert.
#[test]
fn group_toggle_collapses_body_and_updates_header() {
    let mut world = World::new();
    let screen = AssetId(50);
    let (header, body) = (AssetId(51), AssetId(52));
    world.push_identified(
        screen,
        Screen {
            initial: true,
            fade_in_secs: 0.0,
            ..Default::default()
        },
    );
    world.push_identified(AssetId(51), panel_label(100.0, screen, "- Adv"));
    world.push_identified(AssetId(52), panel_label(140.0, screen, "Body"));
    // Header click region (toggles group 0).
    world.add_component(HitRegion {
        x: 0.0,
        y: 100.0,
        width: 300.0,
        height: 40.0,
        label: Some(Ref::new(header)),
        hover_color: None,
        hover_scale: None,
        action: Some(act(serde_json::json!({"flip": "state:adv"}))),
        slider: None,
        drag_handle: None,
        screen: Some(Ref::new(screen)),
        disabled: false,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    // Body click region (a settings action; a content region, so it is
    // bucketed into its row and gated by the collapse).
    world.add_component(HitRegion {
        x: 0.0,
        y: 140.0,
        width: 300.0,
        height: 40.0,
        label: None,
        hover_color: None,
        hover_scale: None,
        action: Some(act(
            serde_json::json!({"step": {"binding": "setting:vsync", "by": 1}}),
        )),
        slider: None,
        drag_handle: None,
        screen: Some(Ref::new(screen)),
        disabled: false,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    world.add_component(ScrollPanel {
        screen: Some(Ref::new(screen)),
        x: 0.0,
        y: 100.0,
        width: 300.0,
        height: 100.0,
        rows: vec![
            ScrollRow {
                elements: vec![Ref::new(header)],
                base_y: 100.0,
                height: 40.0,
                group: -1,
            },
            ScrollRow {
                elements: vec![Ref::new(body)],
                base_y: 140.0,
                height: 40.0,
                group: 0,
            },
        ],
        groups: vec![ScrollGroup {
            collapsed: false,
            header: Some(Ref::new(header)),
            title: "Adv".to_string(),
            collapsed_binding: Some(Binding::State("adv".to_string())),
        }],
        thumb: None,
        track: None,
        track_x: 0.0,
        track_y: 0.0,
        track_w: 0.0,
        track_h: 0.0,
    });
    start(&mut world);

    // Expanded after init: body shown, header reads "- Adv".
    assert!(label_field(&world, body, |l| l.visible));
    assert_eq!(label_field(&world, header, |l| l.content.clone()), "- Adv");

    // Click the header to collapse.
    world.add_component(make_frame_input(10.0, 120.0, true));
    world.step();
    assert!(!label_field(&world, body, |l| l.visible), "body hides");
    assert_eq!(label_field(&world, header, |l| l.content.clone()), "+ Adv");

    // The body's region is now inert: clicking where it was fires nothing.
    world.add_component(make_frame_input(10.0, 160.0, true));
    world.step();
    assert!(
        produced_setting_commands(&world).is_empty(),
        "a collapsed row's region does not fire"
    );
}

// The mouse wheel, with the cursor over the panel band, scrolls the content
// up: the top row's element moves up by wheel-delta * speed (clamped).
#[test]
fn wheel_scrolls_panel_content() {
    let mut world = World::new();
    let screen = AssetId(60);
    let e0 = AssetId(61);
    world.push_identified(
        screen,
        Screen {
            initial: true,
            fade_in_secs: 0.0,
            ..Default::default()
        },
    );
    world.push_identified(AssetId(61), panel_label(0.0, screen, "Row0"));
    // Three 40px rows (120px) in a 60px band -> overflows by 60px.
    world.add_component(ScrollPanel {
        screen: Some(Ref::new(screen)),
        x: 0.0,
        y: 0.0,
        width: 300.0,
        height: 60.0,
        rows: vec![
            ScrollRow {
                elements: vec![Ref::new(e0)],
                base_y: 0.0,
                height: 40.0,
                group: -1,
            },
            ScrollRow {
                elements: vec![],
                base_y: 40.0,
                height: 40.0,
                group: -1,
            },
            ScrollRow {
                elements: vec![],
                base_y: 80.0,
                height: 40.0,
                group: -1,
            },
        ],
        groups: vec![],
        thumb: None,
        track: None,
        track_x: 0.0,
        track_y: 0.0,
        track_w: 0.0,
        track_h: 0.0,
    });
    start(&mut world);
    assert_eq!(label_field(&world, e0, |l| l.y), 0.0);

    // Wheel down with the cursor inside the band: scroll = 10 * speed (2.0)
    // = 20 (within the 60px max), so the top row moves up by 20.
    world.add_component(FrameInput {
        mouse_x: 10.0,
        mouse_y: 10.0,
        scroll_delta: 10.0,
        ..Default::default()
    });
    world.step();
    assert_eq!(label_field(&world, e0, |l| l.y), -20.0);

    // Wheel far past the end clamps to the max (60px up), not further.
    world.add_component(FrameInput {
        mouse_x: 10.0,
        mouse_y: 10.0,
        scroll_delta: 1000.0,
        ..Default::default()
    });
    world.step();
    assert_eq!(label_field(&world, e0, |l| l.y), -60.0);
}

// A panel whose content overflows its band, with scrollbar-track geometry
// so the thumb is grabbable: three 40px rows (120px) in a 60px band, the
// track beside it (thumb = 30px, travel = 30px, max scroll = 60px). Each
// row carries a setting region so focus navigation has targets.
fn scrollbar_panel_world() -> (World, AssetId) {
    let mut world = World::new();
    let screen = AssetId(60);
    let e0 = AssetId(61);
    world.push_identified(
        screen,
        Screen {
            initial: true,
            fade_in_secs: 0.0,
            ..Default::default()
        },
    );
    world.push_identified(AssetId(61), panel_label(0.0, screen, "Row0"));
    for (key, y) in ["vsync", "window_mode", "fps_cap"]
        .into_iter()
        .zip([0.0, 40.0, 80.0])
    {
        world.add_component(HitRegion {
            x: 0.0,
            y,
            width: 300.0,
            height: 40.0,
            label: None,
            hover_color: None,
            hover_scale: None,
            action: Some(act(
                serde_json::json!({"step": {"binding": alloc::format!("setting:{key}"), "by": 1}}),
            )),
            slider: None,
            drag_handle: None,
            screen: Some(Ref::new(screen)),
            disabled: false,
            follow_label: false,
            fit: SpriteFit::Fit,
        });
    }
    world.add_component(ScrollPanel {
        screen: Some(Ref::new(screen)),
        x: 0.0,
        y: 0.0,
        width: 300.0,
        height: 60.0,
        rows: vec![
            ScrollRow {
                elements: vec![Ref::new(e0)],
                base_y: 0.0,
                height: 40.0,
                group: -1,
            },
            ScrollRow {
                elements: vec![],
                base_y: 40.0,
                height: 40.0,
                group: -1,
            },
            ScrollRow {
                elements: vec![],
                base_y: 80.0,
                height: 40.0,
                group: -1,
            },
        ],
        groups: vec![],
        thumb: None,
        track: None,
        track_x: 305.0,
        track_y: 0.0,
        track_w: 8.0,
        track_h: 60.0,
    });
    start(&mut world);
    (world, e0)
}

// Dragging the panel's scrollbar thumb scrolls the content: a press on the
// thumb grabs it and the held cursor's y maps onto the scroll range.
#[test]
fn thumb_drag_scrolls_panel_content() {
    let (mut world, e0) = scrollbar_panel_world();
    assert_eq!(label_field(&world, e0, |l| l.y), 0.0);

    // Press on the thumb (at rest: [305, 0, 8, 30]).
    world.add_component(FrameInput {
        mouse_x: 306.0,
        mouse_y: 5.0,
        left_click: true,
        left_button_down: true,
        ..Default::default()
    });
    world.step();
    assert_eq!(label_field(&world, e0, |l| l.y), 0.0);

    // Drag halfway along the 30px travel: scroll = half of the 60px max.
    world.add_component(FrameInput {
        mouse_x: 306.0,
        mouse_y: 20.0,
        left_button_down: true,
        ..Default::default()
    });
    world.step();
    assert_eq!(label_field(&world, e0, |l| l.y), -30.0);

    // Drag far past the end: the scroll clamps to the max.
    world.add_component(FrameInput {
        mouse_x: 306.0,
        mouse_y: 500.0,
        left_button_down: true,
        ..Default::default()
    });
    world.step();
    assert_eq!(label_field(&world, e0, |l| l.y), -60.0);

    // Release: the drag ends and the cursor no longer moves the content.
    world.add_component(make_frame_input(306.0, 5.0, false));
    world.step();
    assert_eq!(label_field(&world, e0, |l| l.y), -60.0);
}

// Nav pulses (arrow keys / pad) walk the panel's rows and scroll the
// focused row into the clip band, wrapping back to the top past the end.
#[test]
fn nav_focus_walks_panel_rows_and_scrolls_them_into_view() {
    let (mut world, e0) = scrollbar_panel_world();
    let pulse_down = || FrameInput {
        key_events: vec![KeyEvent::press(InputKey::Down)],
        ..Default::default()
    };

    // First pulse focuses the top row: everything already in view.
    world.add_component(pulse_down());
    world.step();
    assert_eq!(label_field(&world, e0, |l| l.y), 0.0);

    // The second row's bottom (80) pokes past the 60px band: the panel
    // scrolls the overflow (20px) into view.
    world.add_component(pulse_down());
    world.step();
    assert_eq!(label_field(&world, e0, |l| l.y), -20.0);

    // The third row scrolls fully in (its bottom sat 40px past the band).
    world.add_component(pulse_down());
    world.step();
    assert_eq!(label_field(&world, e0, |l| l.y), -60.0);

    // Past the end the focus wraps to the top row, scrolling back up.
    world.add_component(pulse_down());
    world.step();
    assert_eq!(label_field(&world, e0, |l| l.y), 0.0);
}

// A panel whose content fits its track has no thumb to draw.
fn scroll_panel(scroll: f32, content_height: f32, thumb_h: f32) -> PanelState {
    PanelState {
        screen: None,
        band: [0.0, 0.0, 100.0, 200.0],
        rows: Vec::new(),
        groups: Vec::new(),
        thumb: None,
        track: None,
        track_x: 300.0,
        track_y: 50.0,
        track_w: 8.0,
        track_h: 200.0,
        scroll,
        content_height,
        thumb_h,
    }
}

// The thumb sits proportionally down its track, at the panel's scroll
// fraction of the content.
#[test]
fn thumb_rect_tracks_the_scroll_fraction() {
    let top = UiInputSystem::thumb_rect(&scroll_panel(0.0, 400.0, 100.0)).unwrap();
    assert_eq!(
        top,
        [300.0, 50.0, 8.0, 100.0],
        "unscrolled sits at track top"
    );

    // Scrolled a quarter of the content: a quarter down the 200px track.
    let mid = UiInputSystem::thumb_rect(&scroll_panel(100.0, 400.0, 100.0)).unwrap();
    assert_eq!(mid, [300.0, 100.0, 8.0, 100.0]);
}

// Content that fits (a thumb as tall as its track) draws no thumb, and
// neither does a panel that has not been solved yet.
#[test]
fn thumb_rect_absent_when_there_is_nothing_to_scroll() {
    assert!(UiInputSystem::thumb_rect(&scroll_panel(0.0, 400.0, 200.0)).is_none());
    assert!(UiInputSystem::thumb_rect(&scroll_panel(0.0, 0.0, 50.0)).is_none());
}

// A scroll offset past the content pins the thumb to the track's end rather
// than sliding it off.
#[test]
fn thumb_rect_clamps_an_overscrolled_offset() {
    let rect = UiInputSystem::thumb_rect(&scroll_panel(9999.0, 400.0, 100.0)).unwrap();
    assert_eq!(rect[1], 250.0, "pinned at track_y + track_h");
}

// A window-space screen open at full size over a 2560x1440 window, so the
// reference-canvas mapping would halve the cursor.
fn window_space_screen(world: &mut World, screen: AssetId) {
    world.push_identified(
        screen,
        Screen {
            initial: true,
            fade_in_secs: 0.0,
            space: crate::components::ScreenSpace::Window,
            ..Default::default()
        },
    );
}

#[test]
fn a_window_space_slider_reads_the_cursor_in_window_pixels() {
    let mut world = World::new();
    let screen = AssetId(70);
    window_space_screen(&mut world, screen);
    world.add_component(HitRegion {
        x: 100.0,
        y: 0.0,
        width: 200.0,
        height: 40.0,
        slider: Some(Binding::Setting(SettingKey::Exposure)),
        screen: Some(Ref::new(screen)),
        ..Default::default()
    });
    start(&mut world);

    world.add_component(FrameInput {
        mouse_x: 150.0,
        mouse_y: 20.0,
        left_click: true,
        left_button_down: true,
        viewport: [2560.0, 1440.0],
        ..Default::default()
    });
    world.step();
    let cmd = produced_setting_commands(&world)
        .into_iter()
        .last()
        .expect("the press lands on the track");
    assert!(matches!(cmd.op, SettingOp::SetFraction(f) if (f - 0.25).abs() < 1.0e-5));
}

#[test]
fn a_window_space_panel_scrolls_under_the_cursor_in_window_pixels() {
    let mut world = World::new();
    let screen = AssetId(80);
    let e0 = AssetId(81);
    window_space_screen(&mut world, screen);
    world.push_identified(e0, panel_label(0.0, screen, "Row0"));
    let row = |base_y: f32, elements| ScrollRow {
        elements,
        base_y,
        height: 40.0,
        group: -1,
    };
    world.add_component(ScrollPanel {
        screen: Some(Ref::new(screen)),
        x: 600.0,
        y: 0.0,
        width: 300.0,
        height: 60.0,
        rows: vec![
            row(0.0, vec![Ref::new(e0)]),
            row(40.0, vec![]),
            row(80.0, vec![]),
        ],
        groups: vec![],
        thumb: None,
        track: None,
        track_x: 0.0,
        track_y: 0.0,
        track_w: 0.0,
        track_h: 0.0,
    });
    start(&mut world);

    // Inside the band in window pixels; the canvas mapping would put the
    // cursor at x=305, left of the band.
    world.add_component(FrameInput {
        mouse_x: 610.0,
        mouse_y: 10.0,
        scroll_delta: 10.0,
        viewport: [2560.0, 1440.0],
        ..Default::default()
    });
    world.step();
    assert_eq!(label_field(&world, e0, |l| l.y), -20.0);
}

// A slider region over x 100..300 placing a 20-wide handle sprite, bound to
// `binding`.
fn slider_world(binding: Binding) -> World {
    let mut world = World::new();
    world.push_identified(
        AssetId(8),
        Sprite {
            width: 20.0,
            ..Default::default()
        },
    );
    world.add_component(HitRegion {
        x: 100.0,
        width: 200.0,
        slider: Some(binding),
        drag_handle: Some(Ref::new(AssetId(8))),
        ..Default::default()
    });
    world
}

fn handle_x(world: &World) -> f32 {
    world
        .get_by_id::<Sprite>(AssetId(8))
        .map(|s| s.x)
        .expect("handle present")
}

// A slider's handle sits at its setting's published fraction, and follows it
// when the setting changes.
#[test]
fn a_slider_handle_follows_its_setting() {
    let mut world = slider_world(Binding::Setting(SettingKey::Exposure));
    let mut values = SettingValues::default();
    values.set_fraction(SettingKey::Exposure, 0.5);
    world.insert_resource(values);
    start(&mut world);
    assert_eq!(handle_x(&world), 190.0, "100 + 0.5 * (200 - 20)");

    if let Some(values) = world.resource_mut::<SettingValues>() {
        values.set_fraction(SettingKey::Exposure, 1.0);
    }
    world.add_component(FrameInput::default());
    world.step();
    assert_eq!(handle_x(&world), 280.0);
}

// Dragging a `state:` slider sets the state value to the fraction under the
// cursor, and the handle follows it.
#[test]
fn a_state_slider_drags_its_value() {
    let mut world = slider_world(Binding::State("zoom".to_string()));
    let mut state = UiState::default();
    state.set("zoom", UiValue::Float(0.0));
    world.insert_resource(state);
    start(&mut world);
    assert_eq!(handle_x(&world), 100.0);

    world.add_component(FrameInput {
        mouse_x: 150.0,
        mouse_y: 20.0,
        left_click: true,
        left_button_down: true,
        ..Default::default()
    });
    world.step();
    let zoom = world
        .resource::<UiState>()
        .and_then(|s| s.get("zoom").cloned());
    assert_eq!(zoom, Some(UiValue::Float(0.25)));
    assert!(produced_setting_commands(&world).is_empty());

    world.add_component(FrameInput::default());
    world.step();
    assert_eq!(handle_x(&world), 145.0, "100 + 0.25 * 180");
}

// A slider whose setting is disabled at runtime does not start a drag.
#[test]
fn a_disabled_slider_does_not_drag() {
    let mut world = slider_world(Binding::Setting(SettingKey::Exposure));
    world.insert_resource(DisabledSettingRows(
        [SettingKey::Exposure].into_iter().collect(),
    ));
    start(&mut world);
    world.add_component(FrameInput {
        mouse_x: 150.0,
        mouse_y: 20.0,
        left_click: true,
        left_button_down: true,
        ..Default::default()
    });
    world.step();
    assert!(produced_setting_commands(&world).is_empty());
}
