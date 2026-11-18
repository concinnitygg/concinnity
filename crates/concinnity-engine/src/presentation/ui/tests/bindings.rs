// Actions on bindings: UI state flips, steps of a setting, variable changes,
// emitted events, and the labels that show a bound value.

use super::*;
use crate::components::{Binding, UiEvent, UiState, UiValue, ValueOp, VariableCommand};

fn region(action: serde_json::Value) -> HitRegion {
    HitRegion {
        x: 0.0,
        y: 0.0,
        width: 100.0,
        height: 40.0,
        action: Some(act(action)),
        ..Default::default()
    }
}

fn click(world: &mut World) {
    world.add_component(make_frame_input(10.0, 10.0, true));
    world.step();
}

fn state(world: &World, name: &str) -> Option<UiValue> {
    world
        .resource::<UiState>()
        .and_then(|s| s.get(name).cloned())
}

#[test]
fn a_flip_turns_ui_state_over_and_a_bound_label_follows() {
    let mut world = World::new();
    world.add_component(region(serde_json::json!({"flip": "state:grid"})));
    world.push_identified(
        AssetId(1),
        TextLabel {
            text_binding: Some(Binding::State("grid".into())),
            ..Default::default()
        },
    );
    start(&mut world);

    click(&mut world);
    assert_eq!(state(&world, "grid"), Some(UiValue::Bool(true)));
    // The label shows the value the next step.
    world.add_component(make_frame_input(500.0, 500.0, false));
    world.step();
    assert_eq!(
        label_field(&world, AssetId(1), |l| l.content.clone()),
        "true"
    );

    click(&mut world);
    assert_eq!(state(&world, "grid"), Some(UiValue::Bool(false)));
}

// A label and a button added once the UI runs show their bound values too.
#[test]
fn elements_added_later_show_their_bound_values() {
    let mut world = World::new();
    world.insert_resource(UiState::default());
    start(&mut world);
    world.push_identified(
        AssetId(1),
        TextLabel {
            text_binding: Some(Binding::State("name".into())),
            ..Default::default()
        },
    );
    world.push_identified(
        AssetId(2),
        Button {
            text_binding: Some(Binding::State("name".into())),
            ..Default::default()
        },
    );
    if let Some(s) = world.resource_mut::<UiState>() {
        s.set("name", UiValue::Text("sea".into()));
    }
    world.add_component(make_frame_input(500.0, 500.0, false));
    world.step();
    assert_eq!(
        label_field(&world, AssetId(1), |l| l.content.clone()),
        "sea"
    );
    let button = world.get_by_id::<Button>(AssetId(2)).unwrap();
    assert_eq!(button.label, "sea");
}

#[test]
fn a_step_back_on_a_setting_sends_a_previous_option_command() {
    let mut world = World::new();
    world.add_component(region(
        serde_json::json!({"step": {"binding": "setting:window_mode", "by": -1}}),
    ));
    start(&mut world);
    click(&mut world);
    let commands = produced_setting_commands(&world);
    assert_eq!(commands.len(), 1);
    assert!(
        commands
            .iter()
            .all(|c| c.setting == SettingKey::WindowMode && c.op == SettingOp::Prev)
    );
}

#[test]
fn a_var_action_and_an_emit_reach_their_systems() {
    let mut world = World::new();
    world.push_identified(
        AssetId(3),
        region(serde_json::json!({"emit": {"name": "pick", "value": 3}})),
    );
    world.add_component(HitRegion {
        y: 50.0,
        ..region(serde_json::json!({"add": {"binding": "var:score", "by": 1}}))
    });
    start(&mut world);

    click(&mut world);
    let mut cursor = EventCursor::default();
    let events: Vec<UiEvent> = world
        .events::<UiEvent>()
        .map(|e| e.read(&mut cursor).cloned().collect())
        .unwrap_or_default();
    assert_eq!(
        events,
        vec![UiEvent {
            name: "pick".into(),
            element: Some(AssetId(3)),
            value: Some(UiValue::Int(3)),
        }]
    );

    world.add_component(make_frame_input(10.0, 60.0, true));
    world.step();
    let mut cursor = EventCursor::default();
    let commands: Vec<VariableCommand> = world
        .events::<VariableCommand>()
        .map(|e| e.read(&mut cursor).cloned().collect())
        .unwrap_or_default();
    assert_eq!(
        commands,
        vec![VariableCommand {
            name: "score".into(),
            op: ValueOp::Add(1.0),
        }]
    );
}
