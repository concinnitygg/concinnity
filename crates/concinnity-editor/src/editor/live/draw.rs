// src/editor/live/draw.rs
//
// What a placement draws with. A Prop's `material` and `cull_distance` are
// consumed at load, one level further out than the placement fields beside
// them: the draw list bakes them into the GPU draw object and the entity keeps
// only the handle, so there is no live column to write. The engine's
// `gfx::draw_preview` seam is what reaches the draw slots; this module decides
// whether an edit can go through it.
//
// A material the running world never loaded is left to the build, the way a
// changed asset reference is on every other type: resolving one is the cook's
// job, not this path's.

use crate::ecs::{Entity, World};
use crate::gfx::draw_preview::{self, DrawMaterial};
use concinnity_world::registry::RegisteredType;
use serde_json::{Map, Value};

use super::Apply;

// The two placement args that land in the draw slot rather than in a component.
const KEYS: [&str; 2] = ["material", "cull_distance"];

/// One planned rewrite of a placement's draw slots.
pub(crate) struct DrawChange {
    entity: Entity,
    material: Option<DrawMaterial>,
    cull_distance: Option<f32>,
}

/// Plan the draw rewrite for `name`, or `None` when the edit touches anything
/// but a placement's draw args, the running world has no draws to rewrite, or
/// the material it names cannot be swapped in live.
pub(super) fn plan(
    world: &World,
    ct: RegisteredType,
    name: &str,
    args: &Map<String, Value>,
    keys: &[String],
) -> Option<Apply> {
    if ct != RegisteredType::Prop || !keys.iter().all(|k| KEYS.contains(&k.as_str())) {
        return None;
    }
    if !draw_preview::is_available(world) {
        return None;
    }
    let id = crate::ecs::asset_id::lookup(name)?;
    let entity = world
        .resource::<concinnity_core::ecs::EntityByName>()?
        .get(id)?;
    let mut change = DrawChange {
        entity,
        material: None,
        cull_distance: None,
    };
    for key in keys {
        match key.as_str() {
            "material" => change.material = Some(swap(world, entity, args)?),
            _ => change.cull_distance = Some(args.get(key)?.as_f64()? as f32),
        }
    }
    Some(Apply::Draw(change))
}

/// Perform a planned draw rewrite.
pub(super) fn commit(world: &mut World, change: DrawChange) {
    if let Some(material) = change.material {
        draw_preview::apply_material(world, change.entity, material);
    }
    if let Some(cull_distance) = change.cull_distance {
        draw_preview::apply_cull_distance(world, change.entity, cull_distance);
    }
}

// The material the edit names, or `None` when the running world cannot be
// swapped to it: the reference was cleared (what the draw falls back to is the
// build's to resolve), the world never loaded a material under that name, the
// placement draws from a Model (whose sub-meshes carry their own materials), or
// the swap would move the draw to another pass or pipeline.
fn swap(world: &World, entity: Entity, args: &Map<String, Value>) -> Option<DrawMaterial> {
    let name = args.get("material")?.as_str()?;
    let next = draw_preview::material(world, crate::ecs::asset_id::lookup(name)?)?;
    let current = draw_preview::drawn_material(world, entity)?;
    current.swappable_with(&next).then_some(next)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn args(v: Value) -> Map<String, Value> {
        v.as_object().cloned().unwrap()
    }

    fn keys(names: &[&str]) -> Vec<String> {
        names.iter().map(|k| k.to_string()).collect()
    }

    // Only a placement's own draw args route here; the placement fields beside
    // them belong to the transform path, and another type's material (a Model
    // sub-mesh's, say) is not a draw rewrite at all.
    #[test]
    fn a_non_draw_edit_is_not_claimed() {
        let world = World::new();
        for (ty, changed) in [
            ("Prop", "position"),
            ("Prop", "texture"),
            ("Model", "material"),
            ("Material", "roughness"),
        ] {
            let ct = RegisteredType::parse(ty).expect("a registered type");
            assert!(
                plan(
                    &world,
                    ct,
                    "crate_a",
                    &args(json!({ changed: "x" })),
                    &keys(&[changed]),
                )
                .is_none(),
                "{ty}.{changed} is not this path's"
            );
        }
    }

    // A world with no renderer has no draw slots to rewrite, so the edit falls
    // back to the rebuild rather than being swallowed.
    #[test]
    fn a_world_without_a_renderer_declines() {
        let world = World::new();
        assert!(!draw_preview::is_available(&world));
        assert!(
            plan(
                &world,
                RegisteredType::Prop,
                "crate_a",
                &args(json!({ "material": "steel" })),
                &keys(&["material"]),
            )
            .is_none()
        );
        assert!(
            plan(
                &world,
                RegisteredType::Prop,
                "crate_a",
                &args(json!({ "cull_distance": 40.0 })),
                &keys(&["cull_distance"]),
            )
            .is_none()
        );
    }
}
