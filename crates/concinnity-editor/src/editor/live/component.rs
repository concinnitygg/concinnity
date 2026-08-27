// src/editor/live/component.rs
//
// The general live path: rebake one component from its edited args and write
// it over the running world's copy. Open only to types the registry marks
// `live` -- the running world re-reads those columns every frame, so the
// overwrite shows on the next draw. A type whose column is read once at load
// would swallow the write silently, which is worse than the rebuild it
// replaces, so everything else declines here and rebuilds instead.
//
// A live type may still name other assets, and those names are the build's to
// resolve: they are edges in the reference graph that decides which scene each
// referenced payload packs into, and a name resolving to nothing is a build
// error rather than a reference that quietly does nothing. An in-place write
// runs neither, so an edit that moves one declines and only the values between
// them apply live.

use crate::ecs::World;
use crate::ecs::asset_id::AssetId;
use concinnity_core::blob::{AssetKind, BlobAssetDef};
use concinnity_core::ecs::ComponentAsset;
use concinnity_world::refs::referenced_names;
use concinnity_world::registry::{self, RegisteredType};
use concinnity_world::world::WorldJsonlAsset;
use serde_json::{Map, Value};

use super::Apply;

/// Plan the overwrite for `name`'s component, or `None` when the type is not
/// live-readable, an asset reference moved, or the args do not bake.
pub(super) fn plan(
    world: &World,
    ct: RegisteredType,
    name: &str,
    before: &Map<String, Value>,
    args: &Map<String, Value>,
) -> Option<Apply> {
    if !ct.live() || changes_references(ct, name, before, args) {
        return None;
    }
    let id = crate::ecs::asset_id::lookup(name)?;
    let entity = world
        .resource::<concinnity_core::ecs::EntityByName>()?
        .get(id)?;
    let asset = bake(ct, id, args).ok()?;
    Some(Apply::Component {
        entity,
        asset: Box::new(asset),
    })
}

// Whether the edit moved which assets this one names, over both sources the
// build resolves: the registry's flat `refs:` fields and the structured
// `CrossReferenced` impls.
fn changes_references(
    ct: RegisteredType,
    name: &str,
    before: &Map<String, Value>,
    after: &Map<String, Value>,
) -> bool {
    names(ct, name, before) != names(ct, name, after)
}

fn names(ct: RegisteredType, name: &str, args: &Map<String, Value>) -> Vec<String> {
    referenced_names(&WorldJsonlAsset {
        name: name.to_string(),
        asset_type: ct.as_str().to_string(),
        args: Value::Object(args.clone()),
    })
}

// The runtime component the args bake to, by the same translation the cook
// runs: a divergent type through its `bake`, a pass-through type through its
// args schema (which is the component).
pub(super) fn bake(
    ct: RegisteredType,
    id: AssetId,
    args: &Map<String, Value>,
) -> Result<ComponentAsset, concinnity_core::result::CnResult> {
    let value = Value::Object(args.clone());
    let args_bytes = match registry::bake_divergent(ct, &value)? {
        Some(bytes) => bytes,
        None => ct.reserialize_args(&value)?,
    };
    let def = BlobAssetDef {
        name: Some(id),
        kind: AssetKind::Component,
        discriminant: ct
            .discriminant()
            .ok_or(concinnity_core::result::CnResult::AssetInvalidType)?,
        args_bytes,
        payload: None,
    };
    ComponentAsset::from_baked(&def)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn args(v: Value) -> Map<String, Value> {
        v.as_object().cloned().unwrap()
    }

    // A world holding one default-valued `ct` under `name`, so a planned
    // overwrite has somewhere to land.
    fn world_with(ct: RegisteredType, name: &str) -> World {
        let mut world = World::new();
        let id = crate::ecs::asset_id::intern(name);
        let entity = world.add(bake(ct, id, &Map::new()).expect("default args bake"));
        let mut by_name = std::collections::BTreeMap::new();
        by_name.insert(id, entity);
        world.insert_resource(concinnity_core::ecs::EntityByName(by_name));
        world
    }

    // A live type bakes to the component its args describe, under the edited
    // asset's own identity.
    #[test]
    fn a_live_type_bakes_its_component() {
        let id = crate::ecs::asset_id::intern("hud_dot");
        let asset = bake(
            RegisteredType::Sprite,
            id,
            &args(json!({ "width": 8.0, "height": 4.0 })),
        )
        .expect("Sprite args bake");
        let ComponentAsset::Sprite(sprite) = asset else {
            panic!("baked the wrong variant");
        };
        assert_eq!((sprite.width, sprite.height), (8.0, 4.0));
        assert_eq!(sprite.asset_id, id, "the asset keeps its identity");
    }

    // A type whose authored schema diverges from its component routes through
    // the same `bake` translation the cook uses.
    #[test]
    fn a_divergent_type_bakes_through_its_translation() {
        let id = crate::ecs::asset_id::intern("cam");
        let asset = bake(
            RegisteredType::Camera3D,
            id,
            &args(json!({ "position": [0.0, 2.0, 5.0] })),
        )
        .expect("Camera3D args bake");
        let ComponentAsset::Camera3D(camera) = asset else {
            panic!("baked the wrong variant");
        };
        assert_eq!(camera.position, [0.0, 2.0, 5.0]);
    }

    // Only types the registry marks live are eligible; the rest decline so the
    // caller rebuilds.
    #[test]
    fn only_live_types_are_eligible() {
        let world = world_with(RegisteredType::Sprite, "solo");

        assert!(
            plan(
                &world,
                RegisteredType::Sprite,
                "solo",
                &args(json!({ "width": 1.0 })),
                &args(json!({ "width": 2.0 })),
            )
            .is_some(),
            "Sprite is live-readable"
        );
        assert!(
            plan(
                &world,
                RegisteredType::PointLight,
                "solo",
                &args(json!({ "intensity": 1.0 })),
                &args(json!({ "intensity": 2.0 })),
            )
            .is_none(),
            "a light's data is built once at load"
        );
    }

    // A retargeted flat reference may name an asset the running world never
    // loaded, so it declines even on a live type.
    #[test]
    fn a_changed_reference_declines() {
        let world = world_with(RegisteredType::Sprite, "badge");
        assert!(
            plan(
                &world,
                RegisteredType::Sprite,
                "badge",
                &args(json!({ "texture": "gold", "width": 2.0 })),
                &args(json!({ "texture": "silver", "width": 2.0 })),
            )
            .is_none()
        );
        assert!(
            plan(
                &world,
                RegisteredType::Sprite,
                "badge",
                &args(json!({ "texture": "gold", "width": 1.0 })),
                &args(json!({ "texture": "gold", "width": 2.0 })),
            )
            .is_some(),
            "the reference stayed put, so the size applies live"
        );
    }

    // A reference a flat `refs:` pair cannot express is guarded the same way:
    // Camera3D reaches its follow target through a nested controller field.
    #[test]
    fn a_changed_structured_reference_declines() {
        let follow = |target: &str, distance: f64| {
            args(json!({
                "controller": { "follow": { "target": target, "distance": distance } },
            }))
        };
        let world = world_with(RegisteredType::Camera3D, "eye");
        assert!(
            plan(
                &world,
                RegisteredType::Camera3D,
                "eye",
                &follow("hero", 4.0),
                &follow("villain", 4.0),
            )
            .is_none()
        );
        assert!(
            plan(
                &world,
                RegisteredType::Camera3D,
                "eye",
                &follow("hero", 4.0),
                &follow("hero", 6.0),
            )
            .is_some(),
            "the orbit distance is not a reference"
        );
    }

    fn spawning(template: &str, count: i64) -> Map<String, Value> {
        args(json!({
            "on": "tick",
            "do": [{ "repeat": { "times": count, "do": [
                { "spawn": { "template": template } },
            ] } }],
        }))
    }

    fn behavior_moves(before: &Map<String, Value>, after: &Map<String, Value>) -> bool {
        changes_references(RegisteredType::Behavior, "spawner", before, after)
    }

    #[test]
    fn behavior_logic_between_the_names_applies_live() {
        assert!(!behavior_moves(
            &spawning("crate", 1),
            &spawning("crate", 4)
        ));
    }

    #[test]
    fn a_retargeted_behavior_reference_declines() {
        assert!(behavior_moves(
            &spawning("crate", 1),
            &spawning("barrel", 1)
        ));
    }

    #[test]
    fn a_behavior_reference_the_edit_added_declines() {
        let before = args(json!({ "on": "tick", "do": [] }));
        assert!(behavior_moves(&before, &spawning("crate", 1)));
    }

    // The firing source names assets too, and it is not part of the body.
    #[test]
    fn a_retargeted_behavior_source_declines() {
        let volume = |name: &str| args(json!({ "on": { "enter": name }, "do": [] }));
        assert!(behavior_moves(&volume("porch"), &volume("hallway")));
        assert!(!behavior_moves(&volume("porch"), &volume("porch")));
    }
}
