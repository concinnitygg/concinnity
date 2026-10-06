//! The verbs that address one skinned mesh's animation by asset name, through
//! the `AnimationSystem`'s name-addressed methods on the engine thread.
//!
//! `anim-crossfade` re-weights a flat clip bucket and ramps the live blend
//! toward the new weights over `duration_secs` (zero snaps); the weight vector
//! must match the target's clip count. `anim-param` writes one graph parameter,
//! and `anim-state` reports the graph's state as of the last animation step.

use concinnity_core::ecs::World;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_engine::AnimationSystem;
use concinnity_host::thread::asset_id;
use serde_json::{Map, Value, json};

use crate::debug::call::Call;
use crate::debug::verb::{Access, Args, Kind, Reply, Verb, optional, queued, required};

const TARGET: &str = "Skinned mesh asset name, as reported by the names command.";

pub(in crate::debug) const VERBS: &[Verb] = &[
    Verb {
        name: "anim-crossfade",
        description: "Ramp a skinned mesh's clip blend weights toward new values over a duration.",
        access: Access::Mutating,
        params: &[
            required("target", Kind::Name, TARGET),
            optional(
                "weights",
                Kind::NumberList,
                "Per-clip blend weights; the length must match the target's clip count.",
            ),
            optional(
                "duration_secs",
                Kind::Number,
                "Ramp duration in seconds. Zero snaps to the new weights.",
            ),
        ],
        run: anim_crossfade,
    },
    Verb {
        name: "anim-param",
        description: "Set one animation graph parameter on a skinned mesh.",
        access: Access::Mutating,
        params: &[
            required("target", Kind::Name, TARGET),
            required("name", Kind::Name, "Graph parameter name."),
            optional(
                "value",
                Kind::Number,
                "New parameter value. Defaults to zero.",
            ),
        ],
        run: anim_param,
    },
    Verb {
        name: "anim-state",
        description: "Report a skinned mesh's animation state, clock, fade progress, blend weights, and parameters.",
        access: Access::ReadOnly,
        params: &[required("target", Kind::Name, TARGET)],
        run: anim_state,
    },
];

#[derive(serde::Deserialize)]
struct Crossfade {
    target: String,
    #[serde(default)]
    weights: Vec<f32>,
    #[serde(default)]
    duration_secs: f32,
}

#[derive(serde::Deserialize)]
struct Param {
    target: String,
    name: String,
    #[serde(default)]
    value: f32,
}

#[derive(serde::Deserialize)]
struct Target {
    target: String,
}

fn anim_crossfade(call: &Call, args: Args) -> Reply {
    let Crossfade {
        target,
        weights,
        duration_secs,
    } = args.parse()?;
    call.on_world(move |world, _| {
        on_target(world, "anim-crossfade", &target, |anim, id| {
            anim.crossfade(id, weights, duration_secs)
        })
    })?;
    queued()
}

fn anim_param(call: &Call, args: Args) -> Reply {
    let Param {
        target,
        name,
        value,
    } = args.parse()?;
    call.on_world(move |world, _| {
        on_target(world, "anim-param", &target, |anim, id| {
            anim.set_param(id, &name, value)
        })
    })?;
    queued()
}

fn anim_state(call: &Call, args: Args) -> Reply {
    let Target { target } = args.parse()?;
    let report = call.on_world(move |world, _| {
        on_target(world, "anim-state", &target, |anim, id| {
            anim.graph_state(id)
        })
    })?;
    let params: Map<String, Value> = report
        .params
        .into_iter()
        .map(|(name, value)| (name, json!(value)))
        .collect();
    Ok(json!({
        "state": report.state,
        "clock_secs": report.clock_secs,
        "fading_from": report.fading_from,
        "fade_progress": report.fade_progress,
        "blend_weights": report.blend_weights,
        "params": params,
    }))
}

// Resolve `target` to its interned id and hand it to the world's animation
// system. A world without one answers with an error, so a client never waits
// out its timeout.
fn on_target<T>(
    world: &mut World,
    verb: &str,
    target: &str,
    apply: impl FnOnce(&mut AnimationSystem, AssetId) -> Result<T, String>,
) -> Result<T, String> {
    let id =
        asset_id::lookup(target).ok_or_else(|| format!("{verb}: unknown asset name '{target}'"))?;
    let anim = concinnity_engine::ecs::animation_system_mut(world)
        .ok_or_else(|| format!("{verb}: no AnimationSystem in this world"))?;
    apply(anim, id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debug::verbs::testing::Engine;
    use crate::test_support;
    use concinnity_core::components::{Animation, AnimationGraph};

    fn anim_clip(name: &str, duration: f32) -> (AssetId, Animation) {
        asset_id::ensure_name_resolver();
        let a: Animation = serde_json::from_value(json!({
            "target": "hero",
            "duration": duration,
            "looping": true,
        }))
        .unwrap();
        (asset_id::intern(name), a)
    }

    fn hero_graph() -> (AssetId, AnimationGraph) {
        asset_id::ensure_name_resolver();
        let g: AnimationGraph = serde_json::from_value(json!({
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
        (asset_id::intern("hero_graph"), g)
    }

    fn add<C: concinnity_core::ecs::ComponentSlot>(world: &mut World, (id, c): (AssetId, C)) {
        world.push_identified(id, c);
    }

    // A started world whose `hero` mesh is driven by an idle/run graph.
    fn graph_world() -> World {
        asset_id::reset_interner();
        let mut world = World::new();
        add(&mut world, anim_clip("idle_clip", 1.0));
        add(&mut world, anim_clip("run_clip", 0.8));
        add(&mut world, hero_graph());
        world.start(concinnity_engine::ecs::SYSTEMS).unwrap();
        world
    }

    // A started world whose `hero` mesh blends two clips by weight.
    fn flat_world() -> World {
        asset_id::reset_interner();
        let mut world = World::new();
        add(&mut world, anim_clip("wave_clip", 1.0));
        add(&mut world, anim_clip("bow_clip", 0.5));
        world.start(concinnity_engine::ecs::SYSTEMS).unwrap();
        world
    }

    #[test]
    fn anim_state_reports_the_live_graph_state() {
        let _guard = test_support::lock();
        let mut engine = Engine::new(graph_world());
        let reply = engine.call("anim-state", json!({ "target": "hero" }));
        let reply = reply.expect("the graph target reports");
        assert_eq!(reply["state"], "idle");
        assert_eq!(reply["params"], json!({ "speed": 0.0 }));
    }

    #[test]
    fn anim_param_queues_a_graph_parameter_write() {
        let _guard = test_support::lock();
        let mut engine = Engine::new(graph_world());
        let reply = engine.call(
            "anim-param",
            json!({ "target": "hero", "name": "speed", "value": 1.0 }),
        );
        assert_eq!(reply, Ok(json!({ "queued": true })));
    }

    #[test]
    fn anim_param_surfaces_an_unknown_parameter() {
        let _guard = test_support::lock();
        let mut engine = Engine::new(graph_world());
        let reply = engine.call(
            "anim-param",
            json!({ "target": "hero", "name": "altitude", "value": 1.0 }),
        );
        assert!(reply.unwrap_err().contains("no parameter 'altitude'"));
    }

    #[test]
    fn anim_crossfade_queues_on_a_flat_target() {
        let _guard = test_support::lock();
        let mut engine = Engine::new(flat_world());
        let reply = engine.call(
            "anim-crossfade",
            json!({ "target": "hero", "weights": [0.0, 1.0], "duration_secs": 0.5 }),
        );
        assert_eq!(reply, Ok(json!({ "queued": true })));
    }

    #[test]
    fn anim_crossfade_is_rejected_on_a_graph_target() {
        let _guard = test_support::lock();
        let mut engine = Engine::new(graph_world());
        let reply = engine.call(
            "anim-crossfade",
            json!({ "target": "hero", "weights": [1.0, 0.0] }),
        );
        assert!(reply.unwrap_err().contains("graph-driven"));
    }

    #[test]
    fn an_unknown_target_name_is_refused() {
        let _guard = test_support::lock();
        let mut engine = Engine::new(graph_world());
        for verb in ["anim-crossfade", "anim-state"] {
            assert_eq!(
                engine.call(verb, json!({ "target": "villain" })),
                Err(format!("{verb}: unknown asset name 'villain'"))
            );
        }
    }

    #[test]
    fn a_world_without_an_animation_system_is_an_error() {
        let _guard = test_support::lock();
        asset_id::reset_interner();
        asset_id::intern_all(&["hero"]);
        let mut engine = Engine::new(World::new());
        assert_eq!(
            engine.call("anim-state", json!({ "target": "hero" })),
            Err("anim-state: no AnimationSystem in this world".to_string())
        );
    }
}
