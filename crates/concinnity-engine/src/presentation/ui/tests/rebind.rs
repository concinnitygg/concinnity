// Rebinding a key from the settings menu.

use super::*;

// A rebind row: a value TextLabel showing the current key + a HitRegion over
// it whose action enters capture mode.
fn rebind_world() -> (World, AssetId) {
    let mut world = World::new();
    let value = AssetId(7);
    world.push_identified(
        value,
        TextLabel {
            font: None,
            content: "W".to_string(),
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
        x: 0.0,
        y: 0.0,
        width: 100.0,
        height: 40.0,
        label: Some(Ref::new(value)),
        hover_color: None,
        hover_scale: None,
        action: Some(act(
            serde_json::json!({"setting": {"key": "key_forward", "verb": "rebind"}}),
        )),
        drag_handle: None,
        screen: None,
        disabled: false,
        follow_label: false,
        fit: SpriteFit::Fit,
    });
    start(&mut world);
    (world, value)
}

// Clicking a rebind row enters capture (the value shows the prompt and no
// command fires); the next pressed key binds it via a Rebind SettingCommand.
#[test]
fn rebind_click_captures_then_binds_next_key() {
    use crate::components::InputKey;
    let (mut world, value) = rebind_world();

    // Click the rebind row: enters capture, value shows the prompt, and no
    // command is pushed yet.
    world.add_component(make_frame_input(50.0, 20.0, true));
    world.step();
    assert_eq!(
        label_field(&world, value, |l| l.content.clone()),
        REBIND_PROMPT
    );
    assert!(
        produced_setting_commands(&world).is_empty(),
        "no command until a key is pressed"
    );

    // Press a key: it binds, pushing a Rebind command carrying the key.
    world.add_component(FrameInput {
        key_events: vec![KeyEvent::press(InputKey::Q)],
        ..Default::default()
    });
    world.step();
    let cmd = produced_setting_commands(&world)
        .into_iter()
        .next()
        .unwrap();
    assert_eq!(cmd.setting, SettingKey::KeyRebind(Bindable::Forward));
    assert_eq!(cmd.value_label, Some(value));
    assert!(matches!(cmd.op, SettingOp::Rebind(InputKey::Q)));
    assert!(cmd.persist);
}

// A capture binds a fresh press only: a key still auto-repeating from
// before the capture began does not bind.
#[test]
fn rebind_capture_ignores_a_repeat() {
    let (mut world, _value) = rebind_world();
    world.add_component(make_frame_input(50.0, 20.0, true));
    world.step();
    let repeat = KeyPress {
        repeat: true,
        ..KeyPress::new(InputKey::E, KeyMods::NONE)
    };
    world.add_component(FrameInput {
        key_events: vec![KeyEvent::Press(repeat)],
        ..Default::default()
    });
    world.step();
    assert!(
        produced_setting_commands(&world).is_empty(),
        "a repeat does not bind"
    );
    world.add_component(FrameInput {
        key_events: vec![KeyEvent::press(InputKey::Q)],
        ..Default::default()
    });
    world.step();
    let cmd = produced_setting_commands(&world)
        .into_iter()
        .next()
        .unwrap();
    assert!(matches!(cmd.op, SettingOp::Rebind(InputKey::Q)));
}

// Escape while capturing cancels and restores the row's previous value text.
#[test]
fn rebind_escape_cancels_and_restores() {
    let (mut world, value) = rebind_world();
    world.add_component(make_frame_input(50.0, 20.0, true));
    world.step();
    assert_eq!(
        label_field(&world, value, |l| l.content.clone()),
        REBIND_PROMPT
    );

    // Escape cancels: the previous text returns and nothing is bound.
    world.add_component(FrameInput {
        escape: true,
        ..Default::default()
    });
    world.step();
    assert_eq!(label_field(&world, value, |l| l.content.clone()), "W");
    assert!(produced_setting_commands(&world).is_empty());
}

// A captured key with no active capture binds nothing.
#[test]
fn a_key_press_without_capture_is_ignored() {
    use crate::components::InputKey;
    let (mut world, _value) = rebind_world();
    world.add_component(FrameInput {
        key_events: vec![KeyEvent::press(InputKey::Q)],
        ..Default::default()
    });
    world.step();
    assert!(produced_setting_commands(&world).is_empty());
}
