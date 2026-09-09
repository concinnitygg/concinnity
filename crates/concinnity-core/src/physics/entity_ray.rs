// Casting a ray at a list of entities rather than at the simulation.
//
// The simulation's own `raycast` answers with a surface point, against every
// body it holds. A caller that addresses the world by entity wants the other
// shape of answer: which of *these* entities the ray meets first. That is what
// this is, and it lives here because the ray-versus-shape primitives do -- the
// caller stays free of them, and a hit means the same thing either way.
//
// Only an entity carrying a `Collider` can be hit. Terrain and heightfields
// belong to the simulation, not to any entity, so they are not tested here.

use crate::components::{Collider, Transform};
use crate::ecs::{ComponentStorage, Entity};
use crate::math::sqrt;
use crate::physics::convert::collider_shape;
use crate::physics::sim::ray_hit_distance;

/// The first entity of `candidates` the ray meets, or `None`.
///
/// `dir` need not be unit length; a zero or non-finite direction, a
/// non-finite origin, and a non-positive `distance` all miss. `exclude` leaves
/// one entity out, which is what keeps a ray fired from inside an entity's own
/// collider from meeting it at zero distance.
pub(crate) fn nearest_hit(
    components: &ComponentStorage,
    candidates: &[Entity],
    from: [f32; 3],
    dir: [f32; 3],
    distance: f32,
    exclude: Option<Entity>,
) -> Option<Entity> {
    if !(distance.is_finite() && distance > 0.0) {
        return None;
    }
    let length = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    if !(length.is_finite() && length > 0.0) {
        return None;
    }
    let unit = [dir[0] / length, dir[1] / length, dir[2] / length];
    if !from.iter().all(|c| c.is_finite()) {
        return None;
    }

    // Shrinks to the nearest hit so far, the way the simulation's own cast
    // does: everything past it is out of the running.
    let mut reach = distance;
    let mut hit = None;
    for &entity in candidates {
        if Some(entity) == exclude {
            continue;
        }
        let (Some(collider), Some(transform)) = (
            components.get::<Collider>(entity),
            components.get::<Transform>(entity),
        ) else {
            continue;
        };
        let shape = collider_shape(&collider.0, transform.scale);
        if let Some(met) = ray_hit_distance(
            from,
            unit,
            &shape,
            transform.position,
            transform.rotation_deg,
            reach,
        ) {
            reach = met;
            hit = Some(entity);
        }
    }
    hit
}

#[cfg(test)]
mod tests {
    use super::nearest_hit;
    use crate::components::{Collider, PropCollider, Transform};
    use crate::ecs::{ComponentStorage, Entity};
    use alloc::vec::Vec;

    // A unit box at `x`, on its own entity.
    fn boxed_at(components: &mut ComponentStorage, x: f32) -> Entity {
        let entity = components.spawn();
        components.insert_typed(
            entity,
            Transform {
                position: [x, 0.0, 0.0],
                ..Default::default()
            },
        );
        components.insert_typed(entity, Collider(PropCollider::default()));
        entity
    }

    // A ray down +x meets the nearer box, not the one behind it.
    #[test]
    fn the_nearest_candidate_wins() {
        let mut components = ComponentStorage::default();
        let near = boxed_at(&mut components, 5.0);
        let far = boxed_at(&mut components, 10.0);

        let hit = nearest_hit(
            &components,
            &[far, near],
            [0.0; 3],
            [1.0, 0.0, 0.0],
            20.0,
            None,
        );
        assert_eq!(
            hit,
            Some(near),
            "candidate order does not decide the answer"
        );
    }

    // `exclude` leaves one candidate out, which is what keeps a ray fired from
    // inside an entity's own collider from meeting it at zero distance.
    #[test]
    fn the_excluded_candidate_is_never_hit() {
        let mut components = ComponentStorage::default();
        let self_entity = boxed_at(&mut components, 0.0);
        let other = boxed_at(&mut components, 5.0);

        let candidates = [self_entity, other];
        assert_eq!(
            nearest_hit(
                &components,
                &candidates,
                [0.0; 3],
                [1.0, 0.0, 0.0],
                20.0,
                None
            ),
            Some(self_entity),
            "without the exclusion the ray meets the box it starts inside"
        );
        assert_eq!(
            nearest_hit(
                &components,
                &candidates,
                [0.0; 3],
                [1.0, 0.0, 0.0],
                20.0,
                Some(self_entity)
            ),
            Some(other),
        );
    }

    // Only an entity carrying a collider can be hit: a transform alone is not a
    // shape, so it is passed through.
    #[test]
    fn an_entity_without_a_collider_is_passed_through() {
        let mut components = ComponentStorage::default();
        let bare = components.spawn();
        components.insert_typed(
            bare,
            Transform {
                position: [2.0, 0.0, 0.0],
                ..Default::default()
            },
        );
        let solid = boxed_at(&mut components, 5.0);

        let hit = nearest_hit(
            &components,
            &[bare, solid],
            [0.0; 3],
            [1.0, 0.0, 0.0],
            20.0,
            None,
        );
        assert_eq!(hit, Some(solid));
    }

    // A candidate past the ray's reach is not a hit.
    #[test]
    fn a_candidate_past_the_reach_is_missed() {
        let mut components = ComponentStorage::default();
        let far = boxed_at(&mut components, 10.0);
        assert_eq!(
            nearest_hit(&components, &[far], [0.0; 3], [1.0, 0.0, 0.0], 2.0, None),
            None
        );
    }

    // A degenerate cast meets nothing rather than dividing by a zero length or
    // sweeping the whole world.
    #[test]
    fn a_degenerate_cast_meets_nothing() {
        let mut components = ComponentStorage::default();
        let target = boxed_at(&mut components, 5.0);
        let candidates: Vec<Entity> = alloc::vec![target];

        let cases: [([f32; 3], f32); 5] = [
            ([0.0, 0.0, 0.0], 20.0),
            ([f32::NAN, 0.0, 0.0], 20.0),
            ([1.0, 0.0, 0.0], 0.0),
            ([1.0, 0.0, 0.0], -1.0),
            ([1.0, 0.0, 0.0], f32::NAN),
        ];
        for (dir, distance) in cases {
            assert_eq!(
                nearest_hit(&components, &candidates, [0.0; 3], dir, distance, None),
                None,
                "dir {dir:?} distance {distance}",
            );
        }
        assert_eq!(
            nearest_hit(
                &components,
                &candidates,
                [f32::NAN, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                20.0,
                None
            ),
            None,
            "a non-finite origin misses",
        );
    }

    // A direction need not be unit length: the reach is in distance, so a long
    // direction vector does not extend it.
    #[test]
    fn a_direction_need_not_be_unit_length() {
        let mut components = ComponentStorage::default();
        let target = boxed_at(&mut components, 5.0);
        assert_eq!(
            nearest_hit(
                &components,
                &[target],
                [0.0; 3],
                [100.0, 0.0, 0.0],
                6.0,
                None
            ),
            Some(target)
        );
        assert_eq!(
            nearest_hit(
                &components,
                &[target],
                [0.0; 3],
                [100.0, 0.0, 0.0],
                2.0,
                None
            ),
            None
        );
    }
}
