// src/gfx/animation/tests.rs

use std::time::{Duration, Instant};

use super::resumed_origin;
use crate::assets::{AnimGraph, AnimParams, Animation};
use crate::ecs::asset_id::{AssetId, intern};
use crate::ecs::{SystemAsset, World};

// Resuming after a pause must leave clip time `t = now - origin` exactly
// where it was when the pause began, so playback continues from the frozen
// pose with no jump, no matter how long the menu was open.
#[test]
fn resumed_origin_freezes_clip_time_across_pause() {
    let start = Instant::now();
    // Paused at t = 5s, menu held open for 30s of real time.
    let anchor = start + Duration::from_secs(5);
    let now = anchor + Duration::from_secs(30);

    let t_at_pause = (anchor - start).as_secs_f32();
    let new_origin = resumed_origin(start, anchor, now);
    let t_on_resume = (now - new_origin).as_secs_f32();

    assert!(
        (t_on_resume - t_at_pause).abs() < 1e-6,
        "clip time jumped across the pause: {t_at_pause} -> {t_on_resume}"
    );
}

// An `Animation` in the world implies the internal AnimationSystem: it is
// constructed by `World::start`, not declared as an asset.
#[test]
fn animation_component_spawns_internal_system() {
    let mut world = World::new_empty();
    world.add_component(Animation::default());
    world.start().unwrap();

    let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
    assert_eq!(names, ["AnimationSystem"]);
}

// No `Animation` means no AnimationSystem; the gate keys purely off world
// content.
#[test]
fn no_animation_no_internal_system() {
    let mut world = World::new_empty();
    world.start().unwrap();
    assert!(world.systems().is_empty());
}

// An `AnimGraph` alone also implies the system (a graph without clips is a
// build error, but the runtime gate must not depend on validation).
#[test]
fn anim_graph_component_spawns_internal_system() {
    let mut world = World::new_empty();
    world.add_component(AnimGraph::default());
    world.start().unwrap();

    let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
    assert_eq!(names, ["AnimationSystem"]);
}

// A clip targeting `hero`, named so graph states can reference it.
fn clip(name: &str, duration: f32) -> Animation {
    let mut a: Animation = serde_json::from_value(serde_json::json!({
        "target": "hero",
        "duration": duration,
        "looping": true,
    }))
    .unwrap();
    a.asset_id = intern(name);
    a
}

// idle/run graph on `hero`, transitioning on `speed` in both directions with
// snap fades (duration 0) so state flips are visible after one step.
fn hero_graph() -> AnimGraph {
    let mut g: AnimGraph = serde_json::from_value(serde_json::json!({
        "target": "hero",
        "parameters": [{"name": "speed", "default": 0.0}],
        "initial": "idle",
        "states": [
            {"name": "idle", "clip": "idle_clip"},
            {"name": "run", "clip": "run_clip"}
        ],
        "transitions": [
            {"from": "idle", "to": "run",
             "conditions": [{"parameter": "speed", "op": "gt", "value": 0.5}]},
            {"from": "run", "to": "idle",
             "conditions": [{"parameter": "speed", "op": "le", "value": 0.5}]}
        ]
    }))
    .unwrap();
    g.asset_id = intern("hero_graph");
    g
}

fn graph_world() -> World {
    let mut world = World::new_empty();
    world.add_component(clip("idle_clip", 1.0));
    world.add_component(clip("run_clip", 0.8));
    world.add_component(hero_graph());
    world.start().unwrap();
    world
}

fn hero() -> AssetId {
    intern("hero")
}

// Match the system back out of the world for report assertions.
fn with_anim<R>(world: &mut World, f: impl FnOnce(&mut super::AnimationSystem) -> R) -> R {
    for system in world.systems_mut() {
        if let SystemAsset::AnimationSystem(anim) = system {
            return f(anim);
        }
    }
    panic!("AnimationSystem not constructed");
}

// Init publishes one `AnimParams` per graph, seeded with the declared
// defaults, and the graph starts in its initial state.
#[test]
fn graph_init_seeds_params_and_initial_state() {
    let mut world = graph_world();
    world.step();

    let params: Vec<&AnimParams> = world.query::<AnimParams>().collect();
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].target, hero());
    assert_eq!(params[0].values, vec![0.0]);

    let report = with_anim(&mut world, |anim| anim.graph_report(hero()).unwrap());
    assert_eq!(report.state, "idle");
    assert_eq!(report.params, vec![("speed".to_string(), 0.0)]);
    assert!(
        report.blend_weights.is_none(),
        "single-clip states report no blend weights"
    );
}

// A blendspace state's reported member weights track the parameter: parked
// on the first point at default, split across the bracketing pair mid-range.
#[test]
fn blendspace_weights_follow_the_parameter() {
    let target = intern("hero_blend");
    let mut world = World::new_empty();
    for (name, duration) in [("bl_idle", 1.0), ("bl_walk", 0.8), ("bl_run", 0.6)] {
        let mut a = clip(name, duration);
        a.target = Some(target);
        world.add_component(a);
    }
    let mut g: AnimGraph = serde_json::from_value(serde_json::json!({
        "target": "hero_blend",
        "parameters": [{"name": "speed", "default": 0.0}],
        "states": [
            {"name": "locomotion", "blend": {"kind": "blend1d", "parameter": "speed",
             "sync": true,
             "points": [
                 {"value": 0.0, "clip": "bl_idle"},
                 {"value": 1.6, "clip": "bl_walk"},
                 {"value": 5.0, "clip": "bl_run"}
             ]}}
        ]
    }))
    .unwrap();
    g.asset_id = intern("hero_blend_graph");
    world.add_component(g);
    world.start().unwrap();
    world.step();

    let report = with_anim(&mut world, |anim| anim.graph_report(target).unwrap());
    assert_eq!(report.state, "locomotion");
    assert_eq!(report.blend_weights, Some(vec![1.0, 0.0, 0.0]));

    // Half-way between the walk (1.6) and run (5.0) points.
    with_anim(&mut world, |anim| {
        anim.queue_param(target, "speed", 3.3).unwrap();
    });
    world.step();
    let w = with_anim(&mut world, |anim| anim.graph_report(target).unwrap())
        .blend_weights
        .unwrap();
    assert_eq!(w[0], 0.0);
    assert!(
        (w[1] - 0.5).abs() < 1e-4 && (w[2] - 0.5).abs() < 1e-4,
        "{w:?}"
    );
}

// Writing the `AnimParams` component (the gameplay surface) drives a
// transition on the next step; writing it back returns to the source state.
#[test]
fn graph_transitions_on_component_write() {
    let mut world = graph_world();
    world.step();

    for p in world.query_mut::<AnimParams>() {
        p.set(0, 2.0);
    }
    world.step();
    let report = with_anim(&mut world, |anim| anim.graph_report(hero()).unwrap());
    assert_eq!(report.state, "run");

    for p in world.query_mut::<AnimParams>() {
        p.set(0, 0.0);
    }
    world.step();
    let report = with_anim(&mut world, |anim| anim.graph_report(hero()).unwrap());
    assert_eq!(report.state, "idle");
}

// The `anim-param` path: a queued write lands in the component on the next
// step and the graph reacts to it; unknown names and flat targets fail.
#[test]
fn queued_param_writes_component_and_drives_graph() {
    let mut world = graph_world();
    world.step();

    with_anim(&mut world, |anim| {
        anim.queue_param(hero(), "speed", 3.0).unwrap();
        assert!(
            anim.queue_param(hero(), "nope", 1.0)
                .unwrap_err()
                .contains("no parameter"),
        );
    });
    world.step();

    let params: Vec<&AnimParams> = world.query::<AnimParams>().collect();
    assert_eq!(params[0].values, vec![3.0], "write landed in the component");
    let report = with_anim(&mut world, |anim| anim.graph_report(hero()).unwrap());
    assert_eq!(report.state, "run");
}

// Crossfade commands are the flat-bucket surface; a graph target must reject
// them (and point at anim-param), and a flat target must reject param
// commands (and point at anim-crossfade).
#[test]
fn mode_mismatched_commands_are_rejected() {
    let mut world = graph_world();
    world.step();
    with_anim(&mut world, |anim| {
        let err = anim
            .apply_crossfade(hero(), vec![1.0, 0.0], 0.0, 0.0)
            .unwrap_err();
        assert!(err.contains("graph-driven"));
        assert!(err.contains("anim-param"));
    });

    let mut flat_world = World::new_empty();
    let mut a = clip("solo_clip", 1.0);
    a.target = Some(intern("flat_hero"));
    flat_world.add_component(a);
    flat_world.start().unwrap();
    flat_world.step();
    with_anim(&mut flat_world, |anim| {
        let target = intern("flat_hero");
        anim.apply_crossfade(target, vec![0.5], 0.0, 0.0).unwrap();
        let err = anim.queue_param(target, "speed", 1.0).unwrap_err();
        assert!(err.contains("anim-crossfade"));
        let err = anim.graph_report(target).unwrap_err();
        assert!(err.contains("no AnimGraph"));
    });
}

// While a menu is open the animation step returns early, so graph clocks and
// transitions freeze with the pose.
#[test]
fn graph_freezes_while_menu_open() {
    let mut world = graph_world();
    world.step();

    world.insert_resource(crate::ecs::MenuActive(true));
    for p in world.query_mut::<AnimParams>() {
        p.set(0, 2.0);
    }
    world.step();
    let report = with_anim(&mut world, |anim| anim.graph_report(hero()).unwrap());
    assert_eq!(
        report.state, "idle",
        "paused step must not take transitions"
    );

    world.insert_resource(crate::ecs::MenuActive(false));
    world.step();
    let report = with_anim(&mut world, |anim| anim.graph_report(hero()).unwrap());
    assert_eq!(report.state, "run", "resumed step sees the parameter");
}
