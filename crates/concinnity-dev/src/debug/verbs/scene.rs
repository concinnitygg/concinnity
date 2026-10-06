//! The verbs that change what the running world holds: spawning, removing and
//! re-parenting authored placements, and driving the story system. Each sends
//! the same event the gameplay systems do, so the next world step applies it
//! through the real path; the reply fires once the event is sent, and a name
//! that does not resolve is a clean error.

use concinnity_core::components::{
    DespawnRequest, ReparentRequest, SpawnRequest, StoryCommand, Transform,
};
use concinnity_core::ecs::World;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_host::thread::asset_id;

use crate::debug::call::Call;
use crate::debug::verb::{Access, Args, Kind, Reply, Verb, optional, queued, required};

pub(in crate::debug) const VERBS: &[Verb] = &[
    Verb {
        name: "despawn",
        description: "Remove an authored placement and its descendants from the running world.",
        access: Access::Mutating,
        params: &[required(
            "target",
            Kind::Name,
            "Placement to remove, by `$id` or `<Type>#<ordinal>`.",
        )],
        run: despawn,
    },
    Verb {
        name: "reparent",
        description: "Move an authored placement under a new parent, or detach it to a root.",
        access: Access::Mutating,
        params: &[
            required(
                "target",
                Kind::Name,
                "Placement to move, by `$id` or `<Type>#<ordinal>`.",
            ),
            optional(
                "parent",
                Kind::TextOrNull,
                "New parent placement name; omit to detach the target to a root.",
            ),
        ],
        run: reparent,
    },
    Verb {
        name: "spawn",
        description: "Instantiate a runtime copy of an authored placement at a given pose.",
        access: Access::Mutating,
        params: &[
            required(
                "template",
                Kind::Name,
                "Placement to copy, by `$id` or `<Type>#<ordinal>`.",
            ),
            required(
                "name",
                Kind::Name,
                "Name for the new instance; one no asset is known by.",
            ),
            optional(
                "position",
                Kind::Vec3,
                "World-space position. Defaults to the origin.",
            ),
            optional(
                "rotation_deg",
                Kind::Vec3,
                "Euler rotation in degrees. Defaults to no rotation.",
            ),
            optional(
                "scale",
                Kind::Vec3,
                "Scale on each axis. Defaults to unit scale.",
            ),
            optional(
                "lifetime",
                Kind::NumberOrNull,
                "Seconds before the instance despawns itself; omit to keep it alive.",
            ),
        ],
        run: spawn,
    },
    Verb {
        name: "story",
        description: "Drive the story system with one control action, the way a stage click or key press does.",
        access: Access::Mutating,
        params: &[
            required(
                "action",
                Kind::Choice(&StoryCommand::VERBS),
                "Story control action.",
            ),
            optional(
                "option",
                Kind::Count,
                "Index for the choose and slot actions. Defaults to zero.",
            ),
        ],
        run: story,
    },
];

#[derive(serde::Deserialize)]
struct Despawn {
    target: String,
}

#[derive(serde::Deserialize)]
struct Reparent {
    target: String,
    #[serde(default)]
    parent: Option<String>,
}

#[derive(serde::Deserialize)]
struct Spawn {
    template: String,
    name: String,
    #[serde(default)]
    position: [f32; 3],
    #[serde(default)]
    rotation_deg: [f32; 3],
    #[serde(default)]
    scale: [f32; 3],
    #[serde(default)]
    lifetime: Option<f32>,
}

#[derive(serde::Deserialize)]
struct Story {
    action: String,
    #[serde(default)]
    option: usize,
}

fn despawn(call: &Call, args: Args) -> Reply {
    let Despawn { target } = args.parse()?;
    call.on_world(move |world, _| {
        let id = asset_id::lookup(&target)
            .ok_or_else(|| format!("despawn: name '{target}' not found"))?;
        world
            .events_mut::<DespawnRequest>()
            .send(DespawnRequest { target: id.into() });
        Ok(())
    })?;
    queued()
}

// An empty or whitespace parent detaches the target to a root, like an omitted
// one.
fn reparent(call: &Call, args: Args) -> Reply {
    let Reparent { target, parent } = args.parse()?;
    let parent = parent.filter(|p| !p.trim().is_empty());
    call.on_world(move |world, _| reparent_by_name(world, &target, parent.as_deref()))?;
    queued()
}

// A non-null `lifetime` makes the instance despawn itself after that many
// seconds, which is what exercises draw-slot recycling.
fn spawn(call: &Call, args: Args) -> Reply {
    let request: Spawn = args.parse()?;
    call.on_world(move |world, _| {
        let template = spawn_template(&request.template, &request.name)?;
        world.events_mut::<SpawnRequest>().send(SpawnRequest {
            template,
            name: Some(asset_id::intern(&request.name)),
            transform: Transform {
                position: request.position,
                rotation_deg: request.rotation_deg,
                // A zero scale would make the instance invisible.
                scale: if request.scale == [0.0; 3] {
                    [1.0; 3]
                } else {
                    request.scale
                },
            },
            lifetime_secs: request.lifetime,
        });
        Ok(())
    })?;
    queued()
}

fn story(call: &Call, args: Args) -> Reply {
    let Story { action, option } = args.parse()?;
    let command = StoryCommand::from_verb(&action, Some(option))
        .ok_or_else(|| format!("story: unknown action '{action}'"))?;
    call.on_world(move |world, _| {
        world.events_mut::<StoryCommand>().send(command);
        Ok(())
    })?;
    queued()
}

// Resolve a spawn's template to its id, refusing a new-instance name some
// asset is already known by: interning it would hand back that asset's id.
fn spawn_template(template: &str, name: &str) -> Result<AssetId, String> {
    let template_id = asset_id::lookup(template)
        .ok_or_else(|| format!("spawn: template '{template}' not found"))?;
    if asset_id::lookup(name).is_some() {
        return Err(format!("spawn: '{name}' already names an asset"));
    }
    Ok(template_id)
}

fn reparent_by_name(world: &mut World, child: &str, parent: Option<&str>) -> Result<(), String> {
    let child_id =
        asset_id::lookup(child).ok_or_else(|| format!("reparent: child '{child}' not found"))?;
    let parent_id = parent
        .map(|p| asset_id::lookup(p).ok_or_else(|| format!("reparent: parent '{p}' not found")))
        .transpose()?;
    world.events_mut::<ReparentRequest>().send(ReparentRequest {
        child: child_id.into(),
        parent: parent_id.map(Into::into),
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debug::verbs::testing::Engine;
    use crate::test_support;
    use concinnity_core::ecs::EventCursor;
    use serde_json::json;

    fn queued_reply() -> Reply {
        Ok(json!({ "queued": true }))
    }

    // Every event of type `E` sent since `cursor` last read.
    fn sent<E: Clone + 'static>(world: &World, cursor: &mut EventCursor) -> Vec<E> {
        world
            .events::<E>()
            .map(|events| events.read(cursor).cloned().collect())
            .unwrap_or_default()
    }

    fn engine_with(names: &[&str]) -> Engine {
        asset_id::reset_interner();
        asset_id::intern_all(names);
        Engine::new(World::new())
    }

    #[test]
    fn despawn_resolves_the_name_and_reports_an_unknown_one() {
        let _guard = test_support::lock();
        let mut engine = engine_with(&["crate_a", "crate_b"]);
        let mut cursor = EventCursor::default();

        assert_eq!(
            engine.call("despawn", json!({ "target": "crate_b" })),
            queued_reply()
        );
        let seen = sent::<DespawnRequest>(&engine.world, &mut cursor);
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].target.name(), Some(AssetId(1)));

        assert_eq!(
            engine.call("despawn", json!({ "target": "ghost" })),
            Err("despawn: name 'ghost' not found".to_string())
        );
    }

    #[test]
    fn reparent_resolves_both_names_and_detaches_without_a_parent() {
        let _guard = test_support::lock();
        let mut engine = engine_with(&["box_a", "frame"]);
        let mut cursor = EventCursor::default();

        let call = json!({ "target": "box_a", "parent": "frame" });
        assert_eq!(engine.call("reparent", call), queued_reply());
        let seen = sent::<ReparentRequest>(&engine.world, &mut cursor);
        assert_eq!(seen[0].child.name(), Some(AssetId(0)));
        assert_eq!(seen[0].parent.and_then(|p| p.name()), Some(AssetId(1)));

        for detach in [
            json!({ "target": "box_a" }),
            json!({ "target": "box_a", "parent": "  " }),
        ] {
            assert_eq!(engine.call("reparent", detach), queued_reply());
            let seen = sent::<ReparentRequest>(&engine.world, &mut cursor);
            assert!(
                seen[0].parent.is_none(),
                "an absent or blank parent detaches"
            );
        }

        assert_eq!(
            engine.call("reparent", json!({ "target": "ghost" })),
            Err("reparent: child 'ghost' not found".to_string())
        );
        assert_eq!(
            engine.call("reparent", json!({ "target": "box_a", "parent": "void" })),
            Err("reparent: parent 'void' not found".to_string())
        );
    }

    #[test]
    fn spawn_interns_the_new_name_and_treats_a_zero_scale_as_unit() {
        let _guard = test_support::lock();
        let mut engine = engine_with(&["template_a"]);
        let mut cursor = EventCursor::default();

        let call = json!({
            "template": "template_a",
            "name": "instance_1",
            "position": [1.0, 2.0, 3.0],
            "rotation_deg": [0.0, 90.0, 0.0],
            "scale": [2.0, 2.0, 2.0],
            "lifetime": 5.0,
        });
        assert_eq!(engine.call("spawn", call), queued_reply());
        let seen = sent::<SpawnRequest>(&engine.world, &mut cursor);
        assert_eq!(seen[0].template, AssetId(0));
        assert_eq!(seen[0].name, Some(AssetId(1)));
        assert_eq!(seen[0].transform.position, [1.0, 2.0, 3.0]);
        assert_eq!(seen[0].transform.rotation_deg, [0.0, 90.0, 0.0]);
        assert_eq!(seen[0].transform.scale, [2.0, 2.0, 2.0]);
        assert_eq!(seen[0].lifetime_secs, Some(5.0));

        let call = json!({ "template": "template_a", "name": "instance_2" });
        assert_eq!(engine.call("spawn", call), queued_reply());
        let seen = sent::<SpawnRequest>(&engine.world, &mut cursor);
        assert_eq!(seen[0].transform.scale, [1.0, 1.0, 1.0]);
        assert_eq!(seen[0].lifetime_secs, None);
    }

    // A spawn resolves its template by handle, labels included, and refuses a
    // new name that an asset is already known by.
    #[test]
    fn spawn_template_resolves_labels_and_refuses_a_taken_name() {
        let _guard = test_support::lock();
        asset_id::reset_interner();
        asset_id::prime_name_table(&[(0, "crate".to_string()), (1, "Prop#0".to_string())]);
        assert_eq!(spawn_template("Prop#0", "crate_copy"), Ok(AssetId(1)));
        assert!(
            spawn_template("ghost", "x")
                .unwrap_err()
                .contains("not found")
        );
        assert!(
            spawn_template("crate", "Prop#0")
                .unwrap_err()
                .contains("already names")
        );
    }

    #[test]
    fn story_maps_every_action_to_its_command() {
        let _guard = test_support::lock();
        let mut engine = engine_with(&[]);
        let mut cursor = EventCursor::default();
        for action in StoryCommand::VERBS {
            let call = json!({ "action": action, "option": 2 });
            assert_eq!(engine.call("story", call), queued_reply(), "{action}");
            let seen = sent::<StoryCommand>(&engine.world, &mut cursor);
            assert_eq!(seen, [StoryCommand::from_verb(action, Some(2)).unwrap()]);
        }
        assert_eq!(
            engine.call("story", json!({ "action": "choose" })),
            queued_reply()
        );
        let seen = sent::<StoryCommand>(&engine.world, &mut cursor);
        assert_eq!(seen, [StoryCommand::Choose(0)], "option defaults to zero");
    }
}
