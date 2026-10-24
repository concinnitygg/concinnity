//! The planet live path: an edit to the world's `Planet` that only reshapes
//! its ground or changes its pull goes to the running planet through core's
//! `live_preview::planet`; any other field is read once at load, so the edit
//! declines and the world rebuilds.

use concinnity_cook::authoring::registry::RegisteredType;
use concinnity_engine::components::Planet;
use concinnity_engine::ecs::{ComponentAsset, World};
use concinnity_engine::render::live_preview::planet as planet_preview;
use serde_json::{Map, Value};

use super::{Apply, component};

/// Plan the live reshape of the running planet, or `None` when this is not a
/// planet edit, a changed field is not live, or the world has no planet.
pub(super) fn plan(
    world: &World,
    ct: RegisteredType,
    args: &Map<String, Value>,
    keys: &[String],
) -> Option<Apply> {
    if ct != RegisteredType::Planet || !planet_preview::is_live(keys.iter().map(String::as_str)) {
        return None;
    }
    let (entity, _) = world.query_with_entity::<Planet>().next()?;
    let ComponentAsset::Planet(planet) = component::bake(ct, args).ok()? else {
        return None;
    };
    Some(Apply::Planet {
        entity,
        planet: Box::new(planet),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn args(v: &Value) -> Map<String, Value> {
        v.as_object().cloned().unwrap()
    }

    fn keys(k: &[&str]) -> Vec<String> {
        k.iter().map(ToString::to_string).collect()
    }

    fn world_with_planet() -> World {
        let mut world = World::new();
        world.push(Planet::default());
        world
    }

    #[test]
    fn a_shape_edit_plans_a_live_reshape() {
        let world = world_with_planet();
        let plan = plan(
            &world,
            RegisteredType::Planet,
            &args(&json!({"seed": 7, "amplitude": 12.0})),
            &keys(&["amplitude", "seed"]),
        );
        let Some(Apply::Planet { planet, .. }) = plan else {
            panic!("planned no reshape");
        };
        assert_eq!((planet.seed, planet.amplitude), (7, 12.0));
    }

    // The radius sizes what the renderer built at load, and covers fill its
    // tables: both rebuild. So does a world without a running planet.
    #[test]
    fn a_load_time_field_or_a_missing_planet_declines() {
        let world = world_with_planet();
        let radius = args(&json!({"radius": 9000.0}));
        assert!(plan(&world, RegisteredType::Planet, &radius, &keys(&["radius"])).is_none());
        let covers = args(&json!({"covers": []}));
        assert!(plan(&world, RegisteredType::Planet, &covers, &keys(&["covers"])).is_none());
        let seed = args(&json!({"seed": 1}));
        assert!(
            plan(
                &World::new(),
                RegisteredType::Planet,
                &seed,
                &keys(&["seed"])
            )
            .is_none()
        );
        assert!(plan(&world, RegisteredType::Sprite, &seed, &keys(&["seed"])).is_none());
    }
}
