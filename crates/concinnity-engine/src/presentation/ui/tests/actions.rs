// Hover styling and the actions a click fires: scene commands, quit, and
// showing and hiding screens.

use super::*;

#[test]
fn hover_applies_and_restores_label_style() {
    let mut world = World::new();

    world.push_identified(
        AssetId(1),
        TextLabel {
            font: None,
            content: "Hello".to_string(),
            x: 0.0,
            y: 0.0,
            color: [1.0, 1.0, 1.0],
            color_runs: Vec::new(),
            scale: 1.0,
            centered: false,
            align: TextAlign::Left,
            fit: SpriteFit::Fit,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            visible: true,
            screen: None,
            wrap_width: 0.0,
            max_lines: 0,
        },
    );
    world.add_component(HitRegion {
        x: 10.0,
        y: 10.0,
        width: 100.0,
        height: 40.0,
        label: Some(Ref::new(AssetId(1))),
        hover_color: Some([1.0, 0.0, 0.0]),
        hover_scale: Some(2.0),
        action: None,
        drag_handle: None,
        screen: None,
        disabled: false,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    start(&mut world);

    // Hover over the region.
    world.add_component(make_frame_input(50.0, 30.0, false));
    world.step();

    // Label should be styled.
    let lbl_color = world
        .get_by_id::<TextLabel>(AssetId(1))
        .map(|l| l.color)
        .unwrap();
    assert_eq!(lbl_color, [1.0, 0.0, 0.0]);

    // Move cursor away.
    world.add_component(make_frame_input(0.0, 0.0, false));
    world.step();

    let lbl_color_after = world
        .get_by_id::<TextLabel>(AssetId(1))
        .map(|l| l.color)
        .unwrap();
    assert_eq!(lbl_color_after, [1.0, 1.0, 1.0]);
}

// Clicking a menu button hovers its label (hover color) and switches away to
// another screen the same frame. The next frame the button's screen is hidden;
// its hover color must be restored, not stranded, so it is not still
// highlighted when its screen is shown again.
#[test]
fn hover_style_restored_when_region_screen_is_hidden() {
    let mut world = World::new();
    let menu = AssetId(80);
    let settings = AssetId(81);
    world.push_identified(
        menu,
        Screen {
            initial: true,
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
    // The menu's "Settings" label + its hit region (screen-owned).
    world.push_identified(
        AssetId(1),
        TextLabel {
            font: None,
            content: "Settings".to_string(),
            x: 0.0,
            y: 0.0,
            color: [1.0, 1.0, 1.0],
            color_runs: Vec::new(),
            scale: 1.0,
            centered: false,
            align: TextAlign::Left,
            fit: SpriteFit::Fit,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            visible: true,
            screen: Some(Ref::new(menu)),
            wrap_width: 0.0,
            max_lines: 0,
        },
    );
    world.add_component(HitRegion {
        x: 10.0,
        y: 10.0,
        width: 100.0,
        height: 40.0,
        label: Some(Ref::new(AssetId(1))),
        hover_color: Some([1.0, 0.85, 0.3]),
        hover_scale: Some(1.0),
        action: Some(act(serde_json::json!({"show": 81}))),
        drag_handle: None,
        screen: Some(Ref::new(menu)),
        disabled: false,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    start(&mut world);

    // Hover + click the Settings button (identity overlay at viewport [0,0]):
    // the label takes the hover color and the click sends Show(settings).
    world.add_component(FrameInput {
        mouse_x: 50.0,
        mouse_y: 30.0,
        left_click: true,
        ..Default::default()
    });
    world.step();
    assert_eq!(
        label_field(&world, AssetId(1), |l| l.color),
        [1.0, 0.85, 0.3],
        "hovered button takes the hover color"
    );

    // Next frame applies Show(settings): the menu (and its Settings label) is
    // hidden. The hover color must be restored despite the screen being hidden.
    world.add_component(FrameInput::default());
    world.step();
    assert_eq!(
        label_field(&world, AssetId(1), |l| l.color),
        [1.0, 1.0, 1.0],
        "hover color restored when the region's screen is hidden"
    );
}

// When a region's hover_scale equals its label's scale (what the generated
// settings menu emits), hovering changes only the color: the label keeps its
// size, so it does not grow or shift out of its row. This is the runtime end
// of the build-side `default_menu_hover_is_color_only` guarantee.
#[test]
fn hover_with_matching_scale_changes_color_only() {
    let mut world = World::new();

    world.push_identified(
        AssetId(1),
        TextLabel {
            font: None,
            content: "Vsync".to_string(),
            x: 0.0,
            y: 0.0,
            color: [0.85, 0.85, 0.85],
            color_runs: Vec::new(),
            scale: 0.66,
            centered: false,
            align: TextAlign::Left,
            fit: SpriteFit::Fit,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            visible: true,
            screen: None,
            wrap_width: 0.0,
            max_lines: 0,
        },
    );
    world.add_component(HitRegion {
        x: 10.0,
        y: 10.0,
        width: 100.0,
        height: 40.0,
        label: Some(Ref::new(AssetId(1))),
        hover_color: Some([1.0, 0.85, 0.3]),
        // Matches the label's scale, so hover must not resize it.
        hover_scale: Some(0.66),
        action: None,
        drag_handle: None,
        screen: None,
        disabled: false,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    start(&mut world);

    world.add_component(make_frame_input(50.0, 30.0, false));
    world.step();

    let lbl = world
        .get_by_id::<TextLabel>(AssetId(1))
        .map(|l| (l.color, l.scale))
        .unwrap();
    assert_eq!(lbl.0, [1.0, 0.85, 0.3], "hover should change color");
    assert_eq!(lbl.1, 0.66, "hover must not change the label scale");
}

#[test]
fn click_pushes_scene_command() {
    let mut world = World::new();

    world.add_component(HitRegion {
        x: 0.0,
        y: 0.0,
        width: 100.0,
        height: 100.0,
        label: None,
        hover_color: None,
        hover_scale: None,
        action: Some(act(serde_json::json!({"scene": 3}))),
        drag_handle: None,
        screen: None,
        disabled: false,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    start(&mut world);

    world.add_component(make_frame_input(50.0, 50.0, true));
    world.step();

    let has_cmd = world
        .events::<SceneCommand>()
        .is_some_and(|e| !e.is_empty());
    assert!(has_cmd);
}

#[test]
fn quit_action_returns_stop() {
    let mut world = World::new();

    world.add_component(HitRegion {
        x: 0.0,
        y: 0.0,
        width: 100.0,
        height: 100.0,
        label: None,
        hover_color: None,
        hover_scale: None,
        action: Some(act(serde_json::json!("quit"))),
        drag_handle: None,
        screen: None,
        disabled: false,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    start(&mut world);

    world.add_component(make_frame_input(50.0, 50.0, true));
    let result = world.step();
    assert_eq!(result, StepResult::Stop);
}

// Showing a screen makes its sprites visible and hides them again on Hide.
#[test]
fn screen_show_and_hide_toggles_sprite_visibility() {
    let mut world = World::new();

    let screen_id = AssetId(10);
    world.push_identified(
        screen_id,
        Screen {
            initial: false,
            fade_in_secs: 0.0,
            ..Default::default()
        },
    );
    world.push_identified(
        AssetId(11),
        Sprite {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            texture: None,
            tint: [0.0, 0.0, 0.0, 0.5],
            follow_cursor: false,
            visible: true, // intentionally true to confirm init hides it
            screen: Some(Ref::new(screen_id)),
            fit: SpriteFit::Fit,
            corner_radius: 0.0,
            border_width: 0.0,
            border_color: [0.0, 0.0, 0.0, 1.0],
        },
    );
    start(&mut world);

    // init() hides screen elements.
    let visible_after_init = world
        .get_by_id::<Sprite>(AssetId(11))
        .map(|s| s.visible)
        .unwrap();
    assert!(!visible_after_init, "screen starts hidden after init");

    // Show the screen.
    world
        .events_mut::<ScreenCommand>()
        .send(ScreenCommand::Show(screen_id));
    world.add_component(FrameInput::default());
    world.step();

    let visible_after_show = world
        .get_by_id::<Sprite>(AssetId(11))
        .map(|s| s.visible)
        .unwrap();
    assert!(visible_after_show, "screen sprite is visible after Show");

    // Hide it again.
    world
        .events_mut::<ScreenCommand>()
        .send(ScreenCommand::Hide);
    world.add_component(FrameInput::default());
    world.step();

    let visible_after_hide = world
        .get_by_id::<Sprite>(AssetId(11))
        .map(|s| s.visible)
        .unwrap();
    assert!(!visible_after_hide, "screen sprite is hidden after Hide");
}
