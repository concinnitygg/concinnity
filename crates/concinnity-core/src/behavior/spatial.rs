// The spatial half of what a behavior may read: nearest, within, and the ray
// test behind `raycast`.
//
// The candidates are always a declared query's entities, so what a body may
// reach is exactly what it declared, the way its component queries already
// work. Nearest and within are transform arithmetic and are here; the ray test
// belongs to physics, where the ray-versus-shape primitives are, and this only
// calls it.
//
// Nothing here reads the physics world itself, so a behavior sees no terrain
// and no heightfield: an entity with no collider cannot be hit.

use crate::behavior::position;
use crate::ecs::{ComponentStorage, Entity};

/// Squared distance between two points, for comparisons that never need the
/// root.
fn distance_sq(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

/// The candidate nearest `point`, skipping `exclude` and anything that says
/// nowhere it is. Ties go to the earlier entity, and the candidate order is the
/// query's own stable order, so the answer does not depend on despawns
/// elsewhere.
pub(crate) fn nearest(
    components: &ComponentStorage,
    candidates: &[Entity],
    point: [f32; 3],
    exclude: Option<Entity>,
) -> Option<Entity> {
    let mut best: Option<(Entity, f32)> = None;
    for &entity in candidates {
        if Some(entity) == exclude {
            continue;
        }
        let Some(candidate) = position::of(components, entity) else {
            continue;
        };
        let d = distance_sq(candidate, point);
        if best.is_none_or(|(_, best)| d < best) {
            best = Some((entity, d));
        }
    }
    best.map(|(entity, _)| entity)
}

/// How many candidates lie within `radius` of `point`, skipping `exclude`. A
/// negative radius counts nothing.
pub(crate) fn count_within(
    components: &ComponentStorage,
    candidates: &[Entity],
    point: [f32; 3],
    radius: f32,
    exclude: Option<Entity>,
) -> i32 {
    if !(radius.is_finite() && radius >= 0.0) {
        return 0;
    }
    let limit = radius * radius;
    candidates
        .iter()
        .filter(|entity| Some(**entity) != exclude)
        .filter_map(|entity| position::of(components, *entity))
        .filter(|candidate| distance_sq(*candidate, point) <= limit)
        .count() as i32
}

/// The candidate a ray meets first, skipping `exclude` and anything with no
/// collider.
///
/// `dir` need not be unit length; a zero or non-finite direction, or a
/// non-positive `distance`, meets nothing. A ray starting inside a candidate
/// meets it at once, which is why the behavior's own entity is excluded.
pub(crate) fn raycast(
    components: &ComponentStorage,
    candidates: &[Entity],
    from: [f32; 3],
    dir: [f32; 3],
    distance: f32,
    exclude: Option<Entity>,
) -> Option<Entity> {
    crate::physics::entity_ray::nearest_hit(components, candidates, from, dir, distance, exclude)
}

#[cfg(test)]
mod tests {
    use super::{count_within, nearest};
    use crate::components::Transform;
    use crate::ecs::{ComponentStorage, Entity};

    fn at(components: &mut ComponentStorage, x: f32) -> Entity {
        let entity = components.spawn();
        components.insert_typed(
            entity,
            Transform {
                position: [x, 0.0, 0.0],
                ..Default::default()
            },
        );
        entity
    }

    #[test]
    fn nearest_picks_the_closest_candidate() {
        let mut components = ComponentStorage::default();
        let near = at(&mut components, 1.0);
        let far = at(&mut components, 9.0);
        assert_eq!(
            nearest(&components, &[far, near], [0.0; 3], None),
            Some(near),
            "candidate order does not decide the answer",
        );
    }

    // An entity is never its own nearest, which is what makes
    // `nearest(q, self)` mean "the nearest other".
    #[test]
    fn nearest_skips_the_excluded_entity() {
        let mut components = ComponentStorage::default();
        let me = at(&mut components, 0.0);
        let other = at(&mut components, 5.0);
        assert_eq!(
            nearest(&components, &[me, other], [0.0; 3], Some(me)),
            Some(other)
        );
    }

    // An entity with no transform has no position to compare, so it is passed
    // over rather than treated as sitting at the origin.
    #[test]
    fn nearest_passes_over_a_candidate_with_no_transform() {
        let mut components = ComponentStorage::default();
        let bare = components.spawn();
        let placed = at(&mut components, 5.0);
        assert_eq!(
            nearest(&components, &[bare, placed], [0.0; 3], None),
            Some(placed)
        );
    }

    // A camera has no transform but does have a pose, so a query that names
    // one searches it rather than passing it over.
    #[test]
    fn a_camera_is_a_candidate_through_its_own_pose() {
        use crate::components::Camera3D;
        use crate::components::cook::Camera3D as Camera3DArgs;

        let mut components = ComponentStorage::default();
        let camera = components.spawn();
        components.insert_typed(
            camera,
            Camera3D::bake(Camera3DArgs {
                position: [2.0, 0.0, 0.0],
                ..Default::default()
            }),
        );
        let placed = at(&mut components, 5.0);

        assert_eq!(
            nearest(&components, &[placed, camera], [0.0; 3], None),
            Some(camera),
        );
        assert_eq!(
            count_within(&components, &[placed, camera], [0.0; 3], 3.0, None),
            1,
        );
    }

    #[test]
    fn nearest_of_nothing_is_none() {
        let components = ComponentStorage::default();
        assert_eq!(nearest(&components, &[], [0.0; 3], None), None);
    }

    // The radius is inclusive at its edge, and the excluded entity is not
    // counted.
    #[test]
    fn count_within_counts_the_candidates_inside_the_radius() {
        let mut components = ComponentStorage::default();
        let me = at(&mut components, 0.0);
        let inside = at(&mut components, 3.0);
        let edge = at(&mut components, 5.0);
        let outside = at(&mut components, 6.0);

        let all = [me, inside, edge, outside];
        assert_eq!(count_within(&components, &all, [0.0; 3], 5.0, None), 3);
        assert_eq!(count_within(&components, &all, [0.0; 3], 5.0, Some(me)), 2);
        assert_eq!(count_within(&components, &all, [0.0; 3], 0.0, Some(me)), 0);
    }

    // A radius that is not a real length counts nothing rather than everything.
    #[test]
    fn count_within_rejects_a_degenerate_radius() {
        let mut components = ComponentStorage::default();
        let entity = at(&mut components, 1.0);
        assert_eq!(
            count_within(&components, &[entity], [0.0; 3], -1.0, None),
            0
        );
        assert_eq!(
            count_within(&components, &[entity], [0.0; 3], f32::NAN, None),
            0
        );
    }
}
