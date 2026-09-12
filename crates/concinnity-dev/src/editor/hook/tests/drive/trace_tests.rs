// src/editor/hook/tests/drive/trace_tests.rs
//
// The execution-trace exchange (`hook/drive/trace.rs`): the pulses and live
// values a frame of trace events becomes, the pause a breakpoint hit lands on
// its node, the state a stop clears, and the Ctrl+click that toggles a card's
// breakpoint.

use concinnity_core::ecs::World;
use concinnity_host::thread::asset_id;

use crate::editor::behavior::graph::CardKind;
use crate::editor::behavior::panel::BehaviorAction;
use crate::editor::behavior::path;
use crate::editor::hook::tests::fixtures::{behavior, hook, playing_hook};

use crate::editor::sim;

// A world carrying one published trace tick for behavior `b`'s first node.
fn traced_world(id: asset_id::AssetId, hit: bool) -> World {
    use concinnity_core::ecs::{ExecutionTrace, TraceEvent, TracePaths, TraceStep, TraceVal};
    let mut world = World::new();
    let event = TraceEvent {
        behavior: id,
        node: 0,
    };
    world.insert_resource(TracePaths(vec![(
        id,
        vec![vec![TraceStep::Field("do"), TraceStep::Index(0)]],
    )]));
    world.insert_resource(ExecutionTrace {
        frame: 1,
        events: vec![event],
        vars: vec![("n".to_string(), TraceVal::Int(3))],
        locals: Vec::new(),
        hit: hit.then_some(event),
    });
    world
}

#[test]
fn trace_events_become_pulses_and_live_values() {
    asset_id::reset_interner();
    let id = asset_id::intern("b");
    let mut h = playing_hook(vec![behavior(
        "b",
        serde_json::json!({
            "on": "start", "do": [{"save": {}}],
        }),
    )]);
    h.behavior_open = true;
    let mut world = traced_world(id, false);
    h.drive_trace(&mut world);

    assert_eq!(h.behavior_pulses.len(), 1);
    assert_eq!(
        h.behavior_pulses[0].path,
        vec![path::field("do"), path::Step::Index(0)],
        "the pulse addresses the node the way a checker fault would"
    );
    assert_eq!(
        h.live_vars,
        vec![("n".to_string(), "int".to_string(), "3".to_string())]
    );
    let data = h.behavior_data();
    assert_eq!(
        data.pulse_cards.len(),
        1,
        "the node's card carries the pulse"
    );
    assert_eq!(data.pulse_rows.len(), 1, "so does its outline row");
    // The same frame again reports nothing new; the pulse just decays.
    h.drive_trace(&mut world);
    assert_eq!(h.behavior_pulses.len(), 1);

    // Live values reach the Variables panel and retitle its value column.
    let vdata = h.variables_data();
    assert!(vdata.live);
    assert!(
        vdata.rows.iter().any(|r| r.name == "n" && r.value == "3"),
        "{:?}",
        vdata.rows
    );
}

#[test]
fn a_breakpoint_hit_pauses_and_lands_on_the_node() {
    asset_id::reset_interner();
    let id = asset_id::intern("b");
    let mut h = playing_hook(vec![behavior(
        "b",
        serde_json::json!({
            "on": "start", "do": [{"save": {}}],
        }),
    )]);
    h.behavior_open = true;
    let mut world = traced_world(id, true);
    h.drive_trace(&mut world);
    assert_eq!(h.sim.state, sim::SimState::Paused, "the hit froze the run");
    let row = h.behavior_row.expect("the panel landed on the node");
    assert_eq!(
        h.behavior_rows()[row].path,
        vec![path::field("do"), path::Step::Index(0)]
    );
}

#[test]
fn stopping_clears_the_live_state() {
    asset_id::reset_interner();
    let id = asset_id::intern("b");
    let mut h = playing_hook(vec![behavior(
        "b",
        serde_json::json!({
            "on": "start", "do": [{"save": {}}],
        }),
    )]);
    h.behavior_open = true;
    let mut world = traced_world(id, false);
    h.drive_trace(&mut world);
    assert!(!h.behavior_pulses.is_empty() && !h.live_vars.is_empty());

    assert!(h.sim.stop());
    h.drive_trace(&mut world);
    assert!(h.behavior_pulses.is_empty(), "Stop shows authored data");
    assert!(h.live_vars.is_empty());
    assert!(!h.variables_data().live);
}

#[test]
fn ctrl_click_toggles_a_card_breakpoint() {
    let mut h = hook(vec![behavior(
        "b",
        serde_json::json!({
            "on": "start", "do": [{"save": {}}],
        }),
    )]);
    let mut world = World::new();
    h.behavior_open = true;
    let data = h.behavior_data();
    let card = data
        .chart
        .cards
        .iter()
        .position(|c| c.kind == CardKind::Node)
        .expect("the body has a node card");

    h.ctrl_held = true;
    h.apply_behavior_action(BehaviorAction::SelectCard(card), &mut world, [0.0, 0.0]);
    assert_eq!(h.behavior_breakpoints.len(), 1);
    assert_eq!(h.behavior_breakpoints[0].0, "b", "held by behavior name");
    assert_eq!(
        h.behavior_data().break_cards,
        vec![card],
        "the card shows its marker"
    );
    h.apply_behavior_action(BehaviorAction::SelectCard(card), &mut world, [0.0, 0.0]);
    assert!(
        h.behavior_breakpoints.is_empty(),
        "a second toggle removes it"
    );

    // A plain click still selects.
    h.ctrl_held = false;
    h.apply_behavior_action(BehaviorAction::SelectCard(card), &mut world, [0.0, 0.0]);
    assert!(h.behavior_row.is_some());
}
