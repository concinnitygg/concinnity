//! Live edits of the world's planet. Its ground is built around the camera
//! from its shape as the world runs, so an edit to the shape needs no rebuild:
//! the planet system takes it up on its next step, building the ground again
//! under the camera, and the renderer's tiles stream in for the new shape.

use crate::components::Planet;
use crate::ecs::{Entity, World};
use crate::planet::PlanetEdit;

/// Whether an edit changing `fields` of a running planet can be applied live:
/// every one is in [`Planet::LIVE_FIELDS`].
pub fn is_live<'a>(mut fields: impl Iterator<Item = &'a str>) -> bool {
    fields.all(|f| Planet::LIVE_FIELDS.contains(&f))
}

/// Write `planet` over the running world's at `entity` and hand it to the
/// planet system. Only [`Planet::LIVE_FIELDS`] may differ from the running
/// planet. Returns whether `entity` holds a planet.
pub fn apply(world: &mut World, entity: Entity, planet: Planet) -> bool {
    let Some(slot) = world.get_mut::<Planet>(entity) else {
        return false;
    };
    slot.clone_from(&planet);
    world.insert_resource(PlanetEdit(planet));
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_shape_and_the_pull_are_live() {
        assert!(is_live(["seed", "gravity", "land_share"].into_iter()));
        assert!(!is_live(["seed", "radius"].into_iter()));
        assert!(!is_live(["covers"].into_iter()));
        assert!(is_live(core::iter::empty()));
    }

    #[test]
    fn an_applied_edit_reaches_the_component_and_the_planet_system() {
        let mut world = World::new();
        let entity = world.push(Planet::default());
        let edited = Planet {
            seed: 12,
            ..Planet::default()
        };
        assert!(apply(&mut world, entity, edited));
        assert_eq!(world.get::<Planet>(entity).unwrap().seed, 12);
        assert_eq!(world.resource::<PlanetEdit>().unwrap().0.seed, 12);
        let other = world.push(crate::components::Sprite::default());
        assert!(!apply(&mut world, other, Planet::default()));
    }
}
