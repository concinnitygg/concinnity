// src/editor/live/mod.rs
//
// Applying an authored edit to the running preview world instead of rebuilding
// it. The rebuild recompiles every asset and stands the GPU context back up
// from scratch, which is the right answer when the world's shape changed and
// far too much when a slider moved: the data being edited is ECS data, and the
// running world already draws it.
//
// The path is all-or-nothing and planned before anything is written. Every
// change in the edit is turned into an `Apply` first; a single one that cannot
// be expressed against the live world abandons the whole attempt and the
// caller rebuilds. So the world is never left holding half of an edit, and a
// type this module does not understand degrades to exactly the old behaviour.

mod component;
mod diff;
mod draw;
mod lighting;
mod placement;
mod shape;

use crate::components::{
    CharacterCapsule, CharacterShape, DirectionalLight, GraphicsConfig, PostProcessConfig,
    Transform, VolumetricFog,
};
use crate::ecs::{Entity, World};
use concinnity_cook::authoring::registry::{RegisteredType, ScopeResolution};
use concinnity_core::ecs::ComponentAsset;
use serde_json::Value;
use std::collections::BTreeMap;

pub(crate) use diff::{args_changes, same_assets};

/// The pre-merge args of each generated asset a world line patches, keyed by
/// asset name, as of the build that produced the running world. An authored
/// line over a generated asset is a sparse patch, so its own args are not the
/// effective ones; these are the baseline they merge over.
pub(crate) type ShadowBaselines = BTreeMap<String, Value>;

/// One planned write against the running world.
pub(crate) enum Apply {
    /// Overwrite a component with the one its edited args bake to. Boxed:
    /// the component enum is as wide as its widest variant, which dwarfs the
    /// rest of this one.
    Component {
        entity: Entity,
        asset: Box<ComponentAsset>,
    },
    /// Move, turn, or resize a placement.
    Transform {
        entity: Entity,
        transform: Transform,
    },
    /// Re-seed every pose a character shape deforms.
    Shape {
        shape: CharacterShape,
        capsule: Option<CharacterCapsule>,
    },
    /// Replace one directional light and re-light the running world.
    Sun {
        entity: Entity,
        light: DirectionalLight,
    },
    /// Push an edited render-config asset at the renderer. These are consumed
    /// at load, so there is no column to write.
    RenderConfig(RenderConfig),
    /// Rewrite what a placement's draw slots render with.
    Draw(draw::DrawChange),
}

/// A render-config asset an edit can push at the running renderer.
pub(crate) enum RenderConfig {
    Fog(VolumetricFog),
    Graphics(GraphicsConfig),
    /// Boxed: it is several times the width of the other two.
    Post(Box<PostProcessConfig>),
}

/// Plan every change as a write against the running world, or `None` when any
/// one of them needs the world rebuilt. An empty plan means the edit changed
/// nothing the world holds.
pub(crate) fn plan(
    world: &World,
    entries: &[Value],
    changes: &[diff::ArgsChange],
) -> Option<Vec<Apply>> {
    changes
        .iter()
        .map(|change| plan_one(world, entries, change))
        .collect()
}

/// Perform a plan. Every write is against a component the running world's
/// systems read afresh, so the next draw shows the edit.
pub(crate) fn commit(world: &mut World, plan: Vec<Apply>) {
    for apply in plan {
        match apply {
            Apply::Component { entity, asset } => {
                world.replace_component(entity, *asset);
            }
            Apply::Transform { entity, transform } => {
                if let Some(slot) = world.get_mut::<Transform>(entity) {
                    *slot = transform;
                }
            }
            Apply::Shape { shape, capsule } => {
                crate::gfx::shape_preview::apply(world, &shape, capsule.as_ref());
            }
            Apply::Sun { entity, light } => lighting::commit_sun(world, entity, light),
            Apply::RenderConfig(config) => lighting::commit(world, config),
            Apply::Draw(change) => draw::commit(world, change),
        }
    }
}

// The one change, as a write. Each path is tried against the type it knows;
// none claiming it means the build has to run.
fn plan_one(world: &World, entries: &[Value], change: &diff::ArgsChange) -> Option<Apply> {
    let ct = RegisteredType::parse(&change.ty)?;
    // A build-only type is what the expansion reads to produce other assets,
    // so editing one moves the very baselines the diff measures patches
    // against. The build owns that.
    if ct.scope_resolution() == ScopeResolution::Expanded {
        return None;
    }
    if ct == RegisteredType::CharacterShape {
        return shape::plan(world, entries, &change.args, &change.keys);
    }
    lighting::plan(world, ct, &change.name, &change.args, &change.keys)
        .or_else(|| component::plan(world, ct, &change.name, &change.before, &change.args))
        .or_else(|| placement::plan(world, &change.name, &change.args, &change.keys))
        .or_else(|| draw::plan(world, ct, &change.name, &change.args, &change.keys))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Sprite;
    use serde_json::json;

    fn change(name: &str, ty: &str, args: Value, keys: &[&str]) -> diff::ArgsChange {
        diff::ArgsChange {
            name: name.to_string(),
            ty: ty.to_string(),
            before: serde_json::Map::new(),
            args: args.as_object().cloned().unwrap(),
            keys: keys.iter().map(|k| k.to_string()).collect(),
        }
    }

    fn world_with_sprite(name: &str) -> World {
        let mut world = World::new();
        let entity = world.push(Sprite::default());
        let mut by_name = BTreeMap::new();
        by_name.insert(crate::ecs::asset_id::intern(name), entity);
        world.insert_resource(concinnity_core::ecs::EntityByName(by_name));
        world
    }

    // A live type's edit reaches the running component.
    #[test]
    fn a_live_edit_is_written_to_the_world() {
        let mut world = world_with_sprite("badge");
        let changes = [change(
            "badge",
            "Sprite",
            json!({ "width": 16.0, "height": 8.0 }),
            &["width"],
        )];
        let plan = plan(&world, &[], &changes).expect("plans");
        commit(&mut world, plan);
        let sprite = world.query::<Sprite>().next().unwrap();
        assert_eq!((sprite.width, sprite.height), (16.0, 8.0));
    }

    // One unplannable change abandons the whole edit, so the world never holds
    // half of it. A point light's data is baked into the scene's light buffer at
    // load, which nothing here can rewrite.
    #[test]
    fn one_unplannable_change_abandons_the_batch() {
        let world = world_with_sprite("badge");
        let changes = [
            change("badge", "Sprite", json!({ "width": 16.0 }), &["width"]),
            change(
                "lamp",
                "PointLight",
                json!({ "intensity": 3.0 }),
                &["intensity"],
            ),
        ];
        assert!(plan(&world, &[], &changes).is_none());
    }

    // A live type whose reference moved rebuilds, and no path downstream of the
    // component one claims the edit instead.
    #[test]
    fn a_moved_reference_declines_through_every_path() {
        let world = world_with_sprite("spawner");
        let spawning =
            |template: &str| json!({ "on": "tick", "do": [{ "spawn": { "template": template } }] });
        let object = |v: Value| v.as_object().cloned().unwrap();
        let changes = [diff::ArgsChange {
            name: "spawner".to_string(),
            ty: "Behavior".to_string(),
            before: object(spawning("crate")),
            args: object(spawning("barrel")),
            keys: vec!["do".to_string()],
        }];
        assert!(plan(&world, &[], &changes).is_none());
    }

    #[test]
    fn an_unknown_type_declines() {
        let world = world_with_sprite("badge");
        let changes = [change("badge", "NotAType", json!({ "a": 1 }), &["a"])];
        assert!(plan(&world, &[], &changes).is_none());
    }

    // A build-only type is what the expansion reads, so editing one always
    // rebuilds -- even when the running world happens to hold a placement
    // under the same name.
    #[test]
    fn a_build_only_type_declines() {
        let mut world = World::new();
        let e = world.push(Transform::default());
        let mut by_name = BTreeMap::new();
        by_name.insert(crate::ecs::asset_id::intern("hero"), e);
        world.insert_resource(concinnity_core::ecs::EntityByName(by_name));
        let changes = [change(
            "hero",
            "CharacterModel",
            json!({ "position": [1.0, 0.0, 0.0] }),
            &["position"],
        )];
        assert!(plan(&world, &[], &changes).is_none());
    }
}
