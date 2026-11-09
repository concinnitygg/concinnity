// Hit testing in the overlay's reference space, and which regions may fire.

use super::*;

// A screen-owned region is overlay UI authored in the reference canvas; when
// the window differs from the reference the region is scaled, and the live
// cursor must be mapped back into reference space to hit it. At a 2x
// viewport, a click at the scaled on-screen rect fires; a click at the raw
// reference coordinates (which no longer overlap the scaled rect) does not.
fn frame_input_at(mx: f32, my: f32, viewport: [f32; 2]) -> FrameInput {
    FrameInput {
        mouse_x: mx,
        mouse_y: my,
        left_click: true,
        viewport,
        ..Default::default()
    }
}

fn overlay_region_world() -> World {
    let mut world = World::new();
    let screen_id = AssetId(30);
    world.push_identified(
        screen_id,
        Screen {
            initial: true,
            fade_in_secs: 0.0,
            ..Default::default()
        },
    );
    // Reference-space rect [200,400] x [200,260].
    world.add_component(HitRegion {
        x: 200.0,
        y: 200.0,
        width: 200.0,
        height: 60.0,
        label: None,
        hover_color: None,
        hover_scale: None,
        action: Some(act(serde_json::json!({"scene": 7}))),
        drag_handle: None,
        screen: Some(Ref::new(screen_id)),
        disabled: false,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    start(&mut world);
    world
}

#[test]
fn screen_owned_region_hit_tests_in_reference_space_when_scaled() {
    // 2x reference viewport: the reference center (300,230) maps on-screen
    // to (600,460). A click there inverse-maps back inside the rect → fires.
    let mut world = overlay_region_world();
    world.add_component(frame_input_at(600.0, 460.0, [2560.0, 1440.0]));
    world.step();
    assert!(
        world
            .events::<SceneCommand>()
            .is_some_and(|e| !e.is_empty()),
        "click at the scaled rect should fire the region"
    );

    // A click at the raw reference coordinates lands outside the scaled
    // rect at 2x, so it must not fire.
    let mut world = overlay_region_world();
    world.add_component(frame_input_at(300.0, 230.0, [2560.0, 1440.0]));
    world.step();
    assert!(
        world
            .events::<SceneCommand>()
            .is_none_or(crate::ecs::Events::is_empty),
        "click at the unscaled coords should miss the scaled region"
    );
}

// While a screen is active, underlying scene HitRegions don't fire.
#[test]
fn hit_region_filtered_when_view_is_active() {
    let mut world = World::new();

    let screen_id = AssetId(20);
    world.push_identified(
        screen_id,
        Screen {
            initial: false,
            fade_in_secs: 0.0,
            ..Default::default()
        },
    );
    // A scene-level region (no screen) that would normally fire.
    world.add_component(HitRegion {
        x: 0.0,
        y: 0.0,
        width: 100.0,
        height: 100.0,
        label: None,
        hover_color: None,
        hover_scale: None,
        action: Some(act(serde_json::json!({"scene": 7}))),
        drag_handle: None,
        screen: None,
        disabled: false,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    start(&mut world);

    // Show the screen, then click where the scene-region is.
    world
        .events_mut::<ScreenCommand>()
        .send(ScreenCommand::Show(screen_id));
    world.add_component(make_frame_input(50.0, 50.0, true));
    world.step();

    let has_cmd = world
        .events::<SceneCommand>()
        .is_some_and(|e| !e.is_empty());
    assert!(
        !has_cmd,
        "scene-level region should not fire while screen is active"
    );
}

#[test]
fn fire_action_dispatches_view_variants() {
    // "hide" → ScreenCommand::Hide
    let mut world = World::new();
    world.add_component(HitRegion {
        x: 0.0,
        y: 0.0,
        width: 100.0,
        height: 100.0,
        label: None,
        hover_color: None,
        hover_scale: None,
        action: Some(act(serde_json::json!("hide"))),
        drag_handle: None,
        screen: None,
        disabled: false,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    start(&mut world);
    world.add_component(make_frame_input(50.0, 50.0, true));
    world.step();
    assert!(matches!(
        produced_screen_command(&world),
        Some(ScreenCommand::Hide)
    ));

    // {"show": 42} → ScreenCommand::Show(42)
    let mut world = World::new();
    world.add_component(HitRegion {
        x: 0.0,
        y: 0.0,
        width: 100.0,
        height: 100.0,
        label: None,
        hover_color: None,
        hover_scale: None,
        action: Some(act(serde_json::json!({"show": 42}))),
        drag_handle: None,
        screen: None,
        disabled: false,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    start(&mut world);
    world.add_component(make_frame_input(50.0, 50.0, true));
    world.step();
    let cmd = produced_screen_command(&world);
    assert!(matches!(cmd, Some(ScreenCommand::Show(AssetId(42)))));

    // {"toggle": 43} → ScreenCommand::Toggle(43)
    let mut world = World::new();
    world.add_component(HitRegion {
        x: 0.0,
        y: 0.0,
        width: 100.0,
        height: 100.0,
        label: None,
        hover_color: None,
        hover_scale: None,
        action: Some(act(serde_json::json!({"toggle": 43}))),
        drag_handle: None,
        screen: None,
        disabled: false,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    start(&mut world);
    world.add_component(make_frame_input(50.0, 50.0, true));
    world.step();
    let cmd = produced_screen_command(&world);
    assert!(matches!(cmd, Some(ScreenCommand::Toggle(AssetId(43)))));
}

#[test]
fn fire_action_dispatches_setting_with_value_label() {
    // setting:vsync:next → SettingCommand carrying the region's label as the
    // value-label to update, and the parsed direction.
    let mut world = World::new();
    let value_label = AssetId(99);
    world.add_component(HitRegion {
        x: 0.0,
        y: 0.0,
        width: 100.0,
        height: 100.0,
        label: Some(Ref::new(value_label)),
        hover_color: None,
        hover_scale: None,
        action: Some(act(
            serde_json::json!({"setting": {"key": "vsync", "verb": "next"}}),
        )),
        drag_handle: None,
        screen: None,
        disabled: false,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    start(&mut world);
    world.add_component(make_frame_input(50.0, 50.0, true));
    world.step();
    let cmd = produced_setting_commands(&world)
        .into_iter()
        .next()
        .unwrap();
    assert_eq!(cmd.setting, SettingKey::Vsync);
    assert_eq!(cmd.op, SettingOp::Next);
    assert_eq!(cmd.value_label, Some(value_label));

    // The :prev suffix parses to the reverse direction. The default
    // HitRegion is 100x40, so click within those bounds.
    let mut world = World::new();
    world.add_component(HitRegion {
        action: Some(act(
            serde_json::json!({"setting": {"key": "vsync", "verb": "prev"}}),
        )),
        ..Default::default()
    });
    start(&mut world);
    world.add_component(make_frame_input(50.0, 20.0, true));
    world.step();
    let cmd = produced_setting_commands(&world)
        .into_iter()
        .next()
        .unwrap();
    assert_eq!(cmd.op, SettingOp::Prev);
}

// A region published as disabled (e.g. a capability-gated settings row grayed
// out at init) is inert: clicking where it sits fires nothing.
#[test]
fn disabled_region_does_not_fire() {
    let mut world = World::new();
    world.add_component(HitRegion {
        x: 0.0,
        y: 0.0,
        width: 100.0,
        height: 100.0,
        label: None,
        hover_color: None,
        hover_scale: None,
        action: Some(act(
            serde_json::json!({"setting": {"key": "ray_traced_reflections", "verb": "next"}}),
        )),
        drag_handle: None,
        screen: None,
        disabled: true,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    start(&mut world);

    world.add_component(make_frame_input(50.0, 50.0, true));
    world.step();
    assert!(
        produced_setting_commands(&world).is_empty(),
        "a disabled region must not fire its action"
    );
}

// A row disabled at runtime via the `DisabledSettingRows` resource (e.g. the
// show_fps row while the "Display performance stats" master is off) is inert,
// even though its HitRegion was enabled at init. This is the runtime twin of
// the init-time capability gating above.
#[test]
fn runtime_disabled_setting_row_does_not_fire() {
    let mut world = World::new();
    world.add_component(HitRegion {
        x: 0.0,
        y: 0.0,
        width: 100.0,
        height: 100.0,
        label: None,
        hover_color: None,
        hover_scale: None,
        action: Some(act(
            serde_json::json!({"setting": {"key": "show_fps", "verb": "next"}}),
        )),
        drag_handle: None,
        screen: None,
        disabled: false,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    start(&mut world);

    // Master off: the show_fps row is in the runtime-disabled set, so a click
    // over it fires nothing.
    world.insert_resource(DisabledSettingRows(
        [SettingKey::ShowFps].into_iter().collect(),
    ));
    world.add_component(make_frame_input(50.0, 50.0, true));
    world.step();
    assert!(
        produced_setting_commands(&world).is_empty(),
        "a runtime-disabled row must not fire its action"
    );
}
