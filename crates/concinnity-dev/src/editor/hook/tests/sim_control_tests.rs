// src/editor/hook/tests/sim_control_tests.rs
//
// The simulation transport (`hook/sim_control.rs`): the keys and the top bar's
// chips that play, pause, step and stop, and which edits the transport
// survives -- a body or table edit written into the running world leaves it
// running, one that moves a reference rebuilds and stops it. Also the fly
// camera's exchange with play, and the trace request the live panels ask for.

use concinnity_core::components::Behavior;
use concinnity_core::components::BehaviorLiteral;
use concinnity_core::components::FrameInput;
use concinnity_core::components::InputKey;
use concinnity_core::components::Sprite;
use concinnity_core::components::Variables;
use concinnity_core::ecs::ComponentAsset;
use concinnity_core::ecs::TraceRequest;
use concinnity_core::ecs::World;
use concinnity_host::thread::asset_id;

use super::fixtures::{behavior, entry, entry_with_args, hook, playing_hook};

use crate::editor::hud::HudAction;

use crate::editor::sim;

#[test]
fn transport_keys_play_pause_stop_and_step() {
    let mut h = hook(Vec::new());
    let key = |k, shift| FrameInput {
        ctrl: true,
        shift,
        captured_key: Some(k),
        ..Default::default()
    };
    h.sim_keys(&key(InputKey::P, false));
    assert!(h.sim.playing(), "Ctrl+P plays");
    h.sim_keys(&key(InputKey::P, false));
    assert_eq!(h.sim.state, sim::SimState::Paused, "Ctrl+P again pauses");
    h.sim_keys(&key(InputKey::Period, false));
    assert!(h.sim.take_run_frame(), "Ctrl+Period queues one step");
    h.sim_keys(&key(InputKey::P, true));
    assert_eq!(h.sim.state, sim::SimState::Stopped, "Ctrl+Shift+P stops");
    assert!(
        h.rebuild_preview,
        "Stop restores through the preview rebuild"
    );

    // A focused text field owns the keyboard.
    h.story_focus = true;
    h.sim_keys(&key(InputKey::P, false));
    assert_eq!(h.sim.state, sim::SimState::Stopped);
}

#[test]
fn transport_chips_drive_the_transport() {
    let mut h = hook(Vec::new());
    let mut world = World::new();
    h.apply_top(HudAction::PlayPause, &mut world);
    assert!(h.sim.playing());
    h.apply_top(HudAction::Step, &mut world);
    assert_eq!(
        h.sim.state,
        sim::SimState::Paused,
        "Step while playing pauses"
    );
    h.apply_top(HudAction::Stop, &mut world);
    assert_eq!(h.sim.state, sim::SimState::Stopped);
    assert!(h.rebuild_preview);
}

#[test]
fn an_edit_that_rebuilds_stops_the_simulation() {
    let mut h = playing_hook(vec![entry("box", "Prop")]);
    let mut world = World::new();
    h.entries.push(entry("box2", "Prop"));
    h.mark_changed();
    assert!(
        h.refresh_preview(&mut world),
        "a new line needs the world rebuilt"
    );
    assert_eq!(
        h.sim.state,
        sim::SimState::Stopped,
        "the rebuild discards the run, so the transport says so"
    );
}

// An edit written straight into the running world discards nothing, so the
// transport keeps running rather than reporting a stop that did not happen.
#[test]
fn an_edit_applied_live_leaves_the_simulation_running() {
    let mut h = playing_hook(vec![entry_with_args(
        "badge",
        "Sprite",
        serde_json::json!({ "width": 4.0 }),
    )]);
    h.world_shadows = Some(Default::default());
    let mut world = World::new();
    let e = world.push(Sprite::default());
    let mut by_name = std::collections::BTreeMap::new();
    by_name.insert(asset_id::intern("badge"), e);
    world.insert_resource(concinnity_core::ecs::EntityByName(by_name));

    h.entries[0]["args"]["width"] = serde_json::json!(16.0);
    h.mark_changed();
    assert!(
        !h.refresh_preview(&mut world),
        "a Sprite field is written into the running world"
    );
    assert!(!h.rebuild_preview, "and the preview is current again");
    assert_eq!(h.sim.state, sim::SimState::Playing);
    assert_eq!(world.query::<Sprite>().next().unwrap().width, 16.0);
}

// The Behavior and Variables panels commit as each change is made. Both types
// are live, so the running world takes the edit and the run it is being edited
// against survives.

// A world holding one component of each edited type, indexed by the names the
// entries use.
fn named_world(assets: Vec<(&str, ComponentAsset)>) -> World {
    let mut world = World::new();
    let mut by_name = std::collections::BTreeMap::new();
    for (name, asset) in assets {
        let entity = world.add(asset);
        by_name.insert(asset_id::intern(name), entity);
    }
    world.insert_resource(concinnity_core::ecs::EntityByName(by_name));
    world
}

fn behavior_args(nodes: serde_json::Value) -> serde_json::Value {
    serde_json::json!({ "on": "tick", "do": nodes })
}

fn set_node(var: &str) -> serde_json::Value {
    serde_json::json!({ "set": { "var": var, "value": { "int": 1 } } })
}

#[test]
fn a_behavior_body_edit_is_written_into_the_running_world() {
    let mut h = playing_hook(vec![entry_with_args(
        "counter",
        "Behavior",
        behavior_args(serde_json::json!([set_node("n")])),
    )]);
    h.world_shadows = Some(Default::default());
    let mut world = named_world(vec![(
        "counter",
        ComponentAsset::Behavior(Behavior::default()),
    )]);

    h.entries[0]["args"] = behavior_args(serde_json::json!([set_node("n"), set_node("m")]));
    h.mark_changed();
    assert!(
        !h.refresh_preview(&mut world),
        "an edited body is written into the running world"
    );
    assert!(!h.rebuild_preview);
    assert_eq!(h.sim.state, sim::SimState::Playing);
    assert_eq!(
        world.query::<Behavior>().next().unwrap().body.len(),
        2,
        "the column holds the edited body"
    );
}

// The names a body reaches other assets by are the build's to resolve, so
// moving one rebuilds even though the type is live.
#[test]
fn a_behavior_edit_that_moves_a_reference_rebuilds() {
    let spawning = |template: &str| {
        behavior_args(serde_json::json!([
            { "spawn": { "template": template } },
        ]))
    };
    let mut h = playing_hook(vec![entry_with_args(
        "spawner",
        "Behavior",
        spawning("crate"),
    )]);
    h.world_shadows = Some(Default::default());
    let mut world = named_world(vec![(
        "spawner",
        ComponentAsset::Behavior(Behavior::default()),
    )]);

    h.entries[0]["args"] = spawning("barrel");
    h.mark_changed();
    assert!(
        h.refresh_preview(&mut world),
        "a retargeted spawn needs the world rebuilt"
    );
}

#[test]
fn a_variables_edit_is_written_into_the_running_world() {
    let table = |value: i64| serde_json::json!({ "vars": [{ "name": "score", "value": { "int": value } }] });
    let mut h = playing_hook(vec![entry_with_args("world_vars", "Variables", table(0))]);
    h.world_shadows = Some(Default::default());
    let mut world = named_world(vec![(
        "world_vars",
        ComponentAsset::Variables(Variables::default()),
    )]);

    h.entries[0]["args"] = table(7);
    h.mark_changed();
    assert!(
        !h.refresh_preview(&mut world),
        "a declaration is written into the running world"
    );
    assert!(!h.rebuild_preview);
    assert_eq!(h.sim.state, sim::SimState::Playing);
    let declared = world.query::<Variables>().next().unwrap();
    assert_eq!(declared.vars.len(), 1);
    assert_eq!(declared.vars[0].value, BehaviorLiteral::Int(7));
}

// Stop restores the authored state, which means undoing whatever the run did:
// no entry diff describes that, so it must rebuild even though the entries are
// untouched.
#[test]
fn stop_rebuilds_even_though_nothing_was_authored() {
    let mut h = playing_hook(vec![entry_with_args(
        "badge",
        "Sprite",
        serde_json::json!({ "width": 4.0 }),
    )]);
    h.world_shadows = Some(Default::default());
    let mut world = World::new();
    h.sim_stop();
    assert!(h.refresh_preview(&mut world), "Stop always rebuilds");
}

#[test]
fn entering_play_ends_the_fly_camera_and_vice_versa() {
    let mut h = hook(Vec::new());
    h.toggle_fly();
    assert!(h.fly);
    h.sim_toggle_play();
    assert!(h.sim.playing() && !h.fly, "play takes the cursor from fly");
    h.toggle_fly();
    assert!(h.fly);
    assert_eq!(
        h.sim.state,
        sim::SimState::Paused,
        "fly pauses a running world"
    );
}

#[test]
fn the_trace_request_follows_the_live_debug_panels() {
    let mut h = hook(vec![behavior(
        "b",
        serde_json::json!({
            "on": "start", "do": [{"save": {}}],
        }),
    )]);
    let mut world = World::new();
    h.drive_trace(&mut world);
    assert!(
        world.resource::<TraceRequest>().is_none(),
        "no panel open, no request"
    );
    h.behavior_open = true;
    h.drive_trace(&mut world);
    assert!(world.resource::<TraceRequest>().is_some());
    h.behavior_open = false;
    h.drive_trace(&mut world);
    assert!(
        world.resource::<TraceRequest>().is_none(),
        "closing the panels withdraws it"
    );
}
