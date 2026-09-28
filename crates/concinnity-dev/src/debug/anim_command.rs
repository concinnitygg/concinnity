//! The `anim-crossfade`, `anim-param` and `anim-state` verbs as queued runtime
//! commands. The handlers in `super::commands` push them onto the debug
//! server's `RuntimeQueue`; the per-frame debug drive applies each through the
//! `AnimationSystem`'s name-addressed methods and answers its reply channel.

use concinnity_core::ecs::asset_id::AssetId;
use concinnity_engine::animation::{AnimationSystem, GraphStateReport};
use std::sync::mpsc::SyncSender;

// One animation command. `target` is the interned name id of the
// `SkinnedMesh` it addresses.
pub(crate) enum AnimCommand {
    Crossfade {
        target: AssetId,
        weights: Vec<f32>,
        duration_secs: f32,
        reply: SyncSender<Result<(), String>>,
    },
    SetParam {
        target: AssetId,
        name: String,
        value: f32,
        reply: SyncSender<Result<(), String>>,
    },
    QueryState {
        target: AssetId,
        reply: SyncSender<Result<GraphStateReport, String>>,
    },
}

// Apply one command and reply. A world without an `AnimationSystem` answers
// with an error, so a client never waits out its timeout. Reply-send failures
// are dropped: the handler may already have given up waiting.
pub(crate) fn dispatch_anim_command(cmd: AnimCommand, anim: Option<&mut AnimationSystem>) {
    match cmd {
        AnimCommand::Crossfade {
            target,
            weights,
            duration_secs,
            reply,
        } => {
            let result = with_system(anim, "anim-crossfade", |a| {
                a.crossfade(target, weights, duration_secs)
            });
            let _ = reply.send(result);
        }
        AnimCommand::SetParam {
            target,
            name,
            value,
            reply,
        } => {
            let result = with_system(anim, "anim-param", |a| a.set_param(target, &name, value));
            let _ = reply.send(result);
        }
        AnimCommand::QueryState { target, reply } => {
            let _ = reply.send(with_system(anim, "anim-state", |a| a.graph_state(target)));
        }
    }
}

fn with_system<T>(
    anim: Option<&mut AnimationSystem>,
    verb: &str,
    apply: impl FnOnce(&mut AnimationSystem) -> Result<T, String>,
) -> Result<T, String> {
    match anim {
        Some(anim) => apply(anim),
        None => Err(format!("{verb}: no AnimationSystem in this world")),
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use concinnity_core::components::{Animation, AnimationGraph};
    use concinnity_core::ecs::World;
    use concinnity_host::thread::asset_id;

    fn anim_clip(name: &str, duration: f32) -> (AssetId, Animation) {
        asset_id::ensure_name_resolver();
        let a: Animation = serde_json::from_value(serde_json::json!({
            "target": "hero",
            "duration": duration,
            "looping": true,
        }))
        .unwrap();
        (asset_id::intern(name), a)
    }

    fn hero_graph() -> (AssetId, AnimationGraph) {
        asset_id::ensure_name_resolver();
        let g: AnimationGraph = serde_json::from_value(serde_json::json!({
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

    // A started world whose `hero` mesh is driven by an idle/run graph. Interns
    // names, so the calling test holds the shared test lock.
    pub(in crate::debug) fn graph_world() -> World {
        let mut world = World::new();
        add(&mut world, anim_clip("idle_clip", 1.0));
        add(&mut world, anim_clip("run_clip", 0.8));
        add(&mut world, hero_graph());
        world.start(concinnity_engine::ecs::SYSTEMS).unwrap();
        world
    }

    // A started world whose `hero` mesh blends two clips by weight. Interns
    // names, so the calling test holds the shared test lock.
    pub(in crate::debug) fn flat_world() -> World {
        let mut world = World::new();
        add(&mut world, anim_clip("wave_clip", 1.0));
        add(&mut world, anim_clip("bow_clip", 0.5));
        world.start(concinnity_engine::ecs::SYSTEMS).unwrap();
        world
    }

    #[test]
    fn a_world_without_an_animation_system_replies_with_an_error() {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dispatch_anim_command(
            AnimCommand::QueryState {
                target: AssetId(0),
                reply: tx,
            },
            None,
        );
        let err = rx.try_recv().unwrap().unwrap_err();
        assert!(err.contains("anim-state"), "{err}");
        assert!(err.contains("no AnimationSystem"), "{err}");
    }

    #[test]
    fn a_state_query_reports_the_live_graph() {
        let _guard = crate::test_support::lock();
        let mut world = graph_world();
        let hero = asset_id::lookup("hero").expect("the graph interned its target");
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dispatch_anim_command(
            AnimCommand::QueryState {
                target: hero,
                reply: tx,
            },
            concinnity_engine::ecs::animation_system_mut(&mut world),
        );
        let report = rx.try_recv().unwrap().expect("the graph target reports");
        assert_eq!(report.state, "idle");
        assert_eq!(report.params, vec![("speed".to_string(), 0.0)]);
    }
}
