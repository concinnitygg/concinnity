// src/editor/live/placement.rs
//
// Where an asset sits. A placement type's own column is drained at load (the
// decomposition turns a Prop into a Transform plus its renderer and physics
// parts), so its `position` / `rotation_deg` / `scale` args are applied to the
// live `Transform` the decomposition left behind -- the same component the
// gizmo writes while a drag is in flight. Propagation, the pick index, the
// selection rings, and the renderer all follow it.

use crate::components::Transform;
use crate::ecs::World;
use serde_json::{Map, Value};

use super::Apply;

// The three arg keys that land in a `Transform`, paired with the field each
// one fills.
const KEYS: [&str; 3] = ["position", "rotation_deg", "scale"];

/// Plan the transform write for `name`, or `None` when the edit touches
/// anything but placement or the asset has no live transform.
pub(super) fn plan(
    world: &World,
    name: &str,
    args: &Map<String, Value>,
    keys: &[String],
) -> Option<Apply> {
    if !keys.iter().all(|k| KEYS.contains(&k.as_str())) {
        return None;
    }
    let id = crate::ecs::asset_id::lookup(name)?;
    let entity = world
        .resource::<concinnity_core::ecs::EntityByName>()?
        .get(id)?;
    let mut transform = *world.get::<Transform>(entity)?;
    for key in keys {
        let value = vec3(args.get(key))?;
        match key.as_str() {
            "position" => transform.position = value,
            "rotation_deg" => transform.rotation_deg = value,
            _ => transform.scale = value,
        }
    }
    Some(Apply::Transform { entity, transform })
}

// A three-element numeric array, or `None` for anything else (an authored
// null, a shorter array, a non-number).
fn vec3(value: Option<&Value>) -> Option<[f32; 3]> {
    let array = value?.as_array()?;
    if array.len() != 3 {
        return None;
    }
    let mut out = [0.0; 3];
    for (slot, v) in out.iter_mut().zip(array) {
        *slot = v.as_f64()? as f32;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn world_with(name: &str, transform: Transform) -> World {
        let mut world = World::new();
        let entity = world.push(transform);
        let mut by_name = std::collections::BTreeMap::new();
        by_name.insert(crate::ecs::asset_id::intern(name), entity);
        world.insert_resource(concinnity_core::ecs::EntityByName(by_name));
        world
    }

    fn args(v: Value) -> Map<String, Value> {
        v.as_object().cloned().unwrap()
    }

    // Only the edited fields move; the rest keep what the live transform holds.
    #[test]
    fn one_key_leaves_the_other_fields_alone() {
        let start = Transform {
            position: [1.0, 2.0, 3.0],
            rotation_deg: [0.0, 90.0, 0.0],
            scale: [2.0, 2.0, 2.0],
        };
        let world = world_with("crate_a", start);
        let plan = plan(
            &world,
            "crate_a",
            &args(json!({ "position": [4.0, 5.0, 6.0] })),
            &["position".to_string()],
        )
        .expect("a placement edit plans");
        let Apply::Transform { transform, .. } = plan else {
            panic!("planned the wrong kind");
        };
        assert_eq!(transform.position, [4.0, 5.0, 6.0]);
        assert_eq!(transform.rotation_deg, start.rotation_deg);
        assert_eq!(transform.scale, start.scale);
    }

    #[test]
    fn all_three_keys_apply_together() {
        let world = world_with("crate_b", Transform::default());
        let plan = plan(
            &world,
            "crate_b",
            &args(json!({
                "position": [1.0, 0.0, 0.0],
                "rotation_deg": [0.0, 45.0, 0.0],
                "scale": [3.0, 3.0, 3.0],
            })),
            &[
                "position".to_string(),
                "rotation_deg".to_string(),
                "scale".to_string(),
            ],
        )
        .expect("a placement edit plans");
        let Apply::Transform { transform, .. } = plan else {
            panic!("planned the wrong kind");
        };
        assert_eq!(transform.position, [1.0, 0.0, 0.0]);
        assert_eq!(transform.rotation_deg, [0.0, 45.0, 0.0]);
        assert_eq!(transform.scale, [3.0, 3.0, 3.0]);
    }

    // Anything beyond placement is not this path's to apply.
    #[test]
    fn a_non_placement_key_declines() {
        let world = world_with("crate_c", Transform::default());
        assert!(
            plan(
                &world,
                "crate_c",
                &args(json!({ "position": [1.0, 0.0, 0.0], "material": "steel" })),
                &["position".to_string(), "material".to_string()],
            )
            .is_none()
        );
    }

    // An asset the running world never placed has nothing to write to.
    #[test]
    fn an_unplaced_asset_declines() {
        let world = world_with("crate_d", Transform::default());
        assert!(
            plan(
                &world,
                "not_in_the_world",
                &args(json!({ "position": [1.0, 0.0, 0.0] })),
                &["position".to_string()],
            )
            .is_none()
        );
    }

    // A malformed authored value is left to the build's validation.
    #[test]
    fn a_malformed_vector_declines() {
        let world = world_with("crate_e", Transform::default());
        assert!(
            plan(
                &world,
                "crate_e",
                &args(json!({ "position": [1.0, 0.0] })),
                &["position".to_string()],
            )
            .is_none()
        );
    }
}
