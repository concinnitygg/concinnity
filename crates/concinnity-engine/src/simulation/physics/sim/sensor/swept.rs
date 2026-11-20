// The pass-through a region's boundary test misses: a body outside at both ends
// of the step but inside part way, answered off the continuous-collision
// sweep. A mover that ends inside is left to the boundary test.

use crate::simulation::physics::sim::math::Vec3;
use crate::simulation::physics::sim::query::gjk::Support;
use crate::simulation::physics::sim::query::sweep::sweep;

use super::overlap::shapes_overlap;

/// Whether a shape swept along `motion` passed clean through `region`.
pub(crate) fn passed_through(moving: &Support, motion: Vec3, region: &Support) -> bool {
    let Some(hit) = sweep(moving, motion, region) else {
        return false;
    };
    if hit.started_touching {
        return false;
    }
    let mut ended = *moving;
    ended.pose.position = moving.pose.position + motion;
    !shapes_overlap(&ended, region)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::physics::ColliderShape;
    use crate::simulation::physics::sim::collide::Pose;
    use crate::simulation::physics::sim::math::{Quat, vec3};

    const BULLET: ColliderShape = ColliderShape::Ball { radius: 0.1 };
    // A doorway-sized region, two units across the x axis it is crossed on.
    const REGION: ColliderShape = ColliderShape::Cuboid {
        half_extents: [1.0, 2.0, 1.0],
    };

    fn at(shape: &ColliderShape, position: Vec3) -> Support {
        Support::new(
            shape,
            Pose {
                position,
                rotation: Quat::IDENTITY,
            },
        )
    }

    #[test]
    fn a_body_that_crosses_the_whole_region_in_one_step_is_reported() {
        let start = at(&BULLET, vec3(-4.0, 0.0, 0.0));
        assert!(passed_through(
            &start,
            vec3(8.0, 0.0, 0.0),
            &at(&REGION, Vec3::ZERO)
        ));
    }

    // The boundary test owns a mover that stops inside, and would report the
    // entry itself on the next step.
    #[test]
    fn a_body_that_ends_up_inside_is_left_to_the_boundary_test() {
        let start = at(&BULLET, vec3(-4.0, 0.0, 0.0));
        assert!(!passed_through(
            &start,
            vec3(4.0, 0.0, 0.0),
            &at(&REGION, Vec3::ZERO)
        ));
    }

    #[test]
    fn a_body_that_began_inside_is_left_to_the_boundary_test() {
        let start = at(&BULLET, vec3(-0.5, 0.0, 0.0));
        assert!(!passed_through(
            &start,
            vec3(8.0, 0.0, 0.0),
            &at(&REGION, Vec3::ZERO)
        ));
    }

    #[test]
    fn a_body_that_stops_short_and_one_that_goes_past_report_nothing() {
        let start = at(&BULLET, vec3(-4.0, 0.0, 0.0));
        let region = at(&REGION, Vec3::ZERO);
        assert!(!passed_through(&start, vec3(1.0, 0.0, 0.0), &region));
        assert!(!passed_through(&start, vec3(-8.0, 0.0, 0.0), &region));
    }

    #[test]
    fn a_body_passing_beside_the_region_reports_nothing() {
        let start = at(&BULLET, vec3(-4.0, 0.0, 6.0));
        assert!(!passed_through(
            &start,
            vec3(8.0, 0.0, 0.0),
            &at(&REGION, Vec3::ZERO)
        ));
    }

    #[test]
    fn a_standing_body_reports_nothing_however_it_is_placed() {
        let region = at(&REGION, Vec3::ZERO);
        assert!(!passed_through(
            &at(&BULLET, vec3(-4.0, 0.0, 0.0)),
            Vec3::ZERO,
            &region
        ));
        assert!(!passed_through(
            &at(&BULLET, Vec3::ZERO),
            Vec3::ZERO,
            &region
        ));
    }
}
