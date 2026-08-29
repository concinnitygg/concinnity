// src/editor/live/shape.rs
//
// A CharacterShape edit. GraphicsSystem resolves every shape once at load, so
// overwriting the component alone would change nothing; the running world is
// re-seeded through the same narrow seam a slider drag already uses
// (`gfx::shape_preview`), which re-resolves the shape against each live
// `SkeletonPose` and sizes the rig capsule from the authored one through the
// new proportions.
//
// The physics capsule the rig sweeps its own motion against is resized from
// that component every tick, so it follows. The body other actors collide with
// keeps the dimensions it was created with until the world is rebuilt, which
// SAVE does.

use crate::components::{CharacterCapsule, CharacterShape};
use crate::ecs::{SkinnedMeshHandle, World};
use crate::gfx::shape_preview;
use serde_json::{Map, Value};

use super::Apply;

// The shape fields that re-seed a live pose. `target` picks a different mesh
// (which would leave the old one deformed) and `bake` is a build-time
// flattening, so neither belongs here.
const KEYS: [&str; 2] = ["sliders", "proportions"];

/// Plan the re-seed for the shape named by `args.target`, or `None` when the
/// edit reaches past the shape values or the target mesh is not in the running
/// world.
pub(super) fn plan(
    world: &World,
    entries: &[Value],
    args: &Map<String, Value>,
    keys: &[String],
) -> Option<Apply> {
    if !keys.iter().all(|k| KEYS.contains(&k.as_str())) {
        return None;
    }
    let mesh = args.get("target")?.as_str()?;
    let target = shape_preview::mesh_handle(world, crate::ecs::asset_id::lookup(mesh)?)?;
    let shape = shape_of(args, target)?;
    Some(Apply::Shape {
        shape,
        capsule: capsule_of(entries, mesh),
    })
}

// The shape component the args describe, deforming `target`. An absent list
// is the empty one; a malformed list belongs to the build's validation.
fn shape_of(args: &Map<String, Value>, target: SkinnedMeshHandle) -> Option<CharacterShape> {
    fn list<T: serde::de::DeserializeOwned + Default>(
        args: &Map<String, Value>,
        key: &str,
    ) -> Option<T> {
        match args.get(key) {
            None | Some(Value::Null) => Some(T::default()),
            Some(v) => serde_json::from_value(v.clone()).ok(),
        }
    }
    Some(CharacterShape {
        target: Some(target),
        sliders: list(args, "sliders")?,
        proportions: list(args, "proportions")?,
        ..Default::default()
    })
}

// The authored capsule of the shape's target: the base dimensions the new
// proportions scale. Absent for a mesh that declares none.
fn capsule_of(entries: &[Value], mesh: &str) -> Option<CharacterCapsule> {
    let entry = entries
        .iter()
        .find(|e| e.get("name").and_then(|v| v.as_str()) == Some(mesh))?;
    serde_json::from_value(entry.pointer("/args/capsule")?.clone()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn args(v: Value) -> Map<String, Value> {
        v.as_object().cloned().unwrap()
    }

    #[test]
    fn shape_values_build_the_component_against_its_target() {
        let shape = shape_of(
            &args(json!({
                "target": "hero_mesh",
                "sliders": [{ "name": "weight", "value": 0.4 }],
                "proportions": [{ "joint": "spine", "scale": 1.1, "length": 0.0 }],
            })),
            SkinnedMeshHandle(3),
        )
        .expect("shape values build");
        assert_eq!(shape.target, Some(SkinnedMeshHandle(3)));
        assert_eq!(shape.sliders.len(), 1);
        assert_eq!(shape.proportions[0].scale, 1.1);
        assert!(!shape.bake, "a live re-seed never flattens");
    }

    // A shape that authors neither list is the identity shape, which is what
    // Reset commits.
    #[test]
    fn absent_lists_build_an_empty_shape() {
        let shape = shape_of(
            &args(json!({ "target": "hero_mesh" })),
            SkinnedMeshHandle(0),
        )
        .expect("an empty shape builds");
        assert!(shape.sliders.is_empty() && shape.proportions.is_empty());
    }

    #[test]
    fn a_malformed_list_declines() {
        assert!(
            shape_of(
                &args(json!({ "sliders": "not a list" })),
                SkinnedMeshHandle(0)
            )
            .is_none()
        );
    }

    #[test]
    fn the_capsule_comes_from_the_target_mesh() {
        let entries = vec![
            json!({ "name": "plain_mesh", "type": "SkinnedMesh", "args": {} }),
            json!({
                "name": "hero_mesh",
                "type": "SkinnedMesh",
                "args": { "capsule": { "half_height": 0.8, "radius": 0.3 } },
            }),
        ];
        assert_eq!(
            capsule_of(&entries, "hero_mesh").expect("declared").radius,
            0.3
        );
        assert!(
            capsule_of(&entries, "plain_mesh").is_none(),
            "a mesh may declare none"
        );
        assert!(capsule_of(&entries, "absent").is_none());
    }

    // Retargeting and baking are not shape values, so they decline before any
    // world lookup.
    #[test]
    fn retargeting_and_baking_decline() {
        let world = World::new();
        let shape = args(json!({ "target": "hero_mesh", "sliders": [] }));
        assert!(plan(&world, &[], &shape, &["target".to_string()]).is_none());
        assert!(plan(&world, &[], &shape, &["bake".to_string()]).is_none());
    }

    // A target the running world never loaded has no pose to re-seed.
    #[test]
    fn an_absent_target_declines() {
        let world = World::new();
        assert!(
            plan(
                &world,
                &[],
                &args(json!({ "target": "hero_mesh", "sliders": [] })),
                &["sliders".to_string()],
            )
            .is_none()
        );
    }
}
