// concinnity-physics/src/sim/collide/mod.rs
//
// Narrow phase: a candidate pair in, a contact manifold out.
//
// Each shape pair is written once, in one order -- sphere first, then capsule,
// then box -- and the dispatcher reverses the normal when a pair arrives the
// other way round. Contact points are midway between the two surfaces and so
// need no reversing themselves, which is what makes one implementation per
// pair enough.
//
// Terrain does not go through the dispatcher at all. A height grid is not
// convex and answers along its own face normals rather than along a least
// separating axis, so `heightfield` and the `triangle` test it is built on sit
// beside the six pairs rather than inside them.

mod box_box;
mod capsule;
pub(crate) mod heightfield;
mod sphere;
mod support;
pub(crate) mod triangle;

pub(crate) use support::{OrientedBox, Pose, Sphere, support_vertex};

use crate::ColliderShape;
use crate::sim::contact::Manifold;
use crate::sim::math::Vec3;

use capsule::capsule_segment;

fn ball(pose: Pose, radius: f32) -> Sphere {
    Sphere {
        center: pose.position,
        radius,
    }
}

fn cuboid(half_extents: [f32; 3], pose: Pose) -> OrientedBox {
    OrientedBox {
        half: Vec3::from_array(half_extents).abs(),
        pose,
    }
}

/// Build the contact manifold for one pair, writing into `out`, whose bodies
/// are already set. Returns whether the pair touches within `margin`.
///
/// The manifold normal points from the first shape toward the second.
pub(crate) fn collide(
    shape_a: &ColliderShape,
    pose_a: Pose,
    shape_b: &ColliderShape,
    pose_b: Pose,
    margin: f32,
    out: &mut Manifold,
) -> bool {
    out.count = 0;
    let (touched, reversed) = match (shape_a, shape_b) {
        (ColliderShape::Ball { radius: ra }, ColliderShape::Ball { radius: rb }) => (
            sphere::spheres(ball(pose_a, *ra), ball(pose_b, *rb), Vec3::Y, margin, out),
            false,
        ),
        (ColliderShape::Ball { radius }, ColliderShape::Cuboid { half_extents }) => (
            sphere::sphere_box(
                ball(pose_a, *radius),
                cuboid(*half_extents, pose_b),
                margin,
                out,
            ),
            false,
        ),
        (ColliderShape::Cuboid { half_extents }, ColliderShape::Ball { radius }) => (
            sphere::sphere_box(
                ball(pose_b, *radius),
                cuboid(*half_extents, pose_a),
                margin,
                out,
            ),
            true,
        ),
        (
            ColliderShape::Ball { radius },
            ColliderShape::Capsule {
                half_height,
                radius: capsule_radius,
            },
        ) => (
            sphere::sphere_capsule(
                ball(pose_a, *radius),
                capsule_segment(pose_b, *half_height),
                *capsule_radius,
                margin,
                out,
            ),
            false,
        ),
        (
            ColliderShape::Capsule {
                half_height,
                radius: capsule_radius,
            },
            ColliderShape::Ball { radius },
        ) => (
            sphere::sphere_capsule(
                ball(pose_b, *radius),
                capsule_segment(pose_a, *half_height),
                *capsule_radius,
                margin,
                out,
            ),
            true,
        ),
        (
            ColliderShape::Capsule {
                half_height: ha,
                radius: ra,
            },
            ColliderShape::Capsule {
                half_height: hb,
                radius: rb,
            },
        ) => (
            capsule::capsules(
                capsule_segment(pose_a, *ha),
                *ra,
                capsule_segment(pose_b, *hb),
                *rb,
                margin,
                out,
            ),
            false,
        ),
        (
            ColliderShape::Cuboid { half_extents },
            ColliderShape::Capsule {
                half_height,
                radius,
            },
        ) => (
            capsule::box_capsule(
                cuboid(*half_extents, pose_a),
                capsule_segment(pose_b, *half_height),
                *radius,
                margin,
                out,
            ),
            false,
        ),
        (
            ColliderShape::Capsule {
                half_height,
                radius,
            },
            ColliderShape::Cuboid { half_extents },
        ) => (
            capsule::box_capsule(
                cuboid(*half_extents, pose_b),
                capsule_segment(pose_a, *half_height),
                *radius,
                margin,
                out,
            ),
            true,
        ),
        (
            ColliderShape::Cuboid { half_extents: ha },
            ColliderShape::Cuboid { half_extents: hb },
        ) => (
            box_box::box_box(cuboid(*ha, pose_a), cuboid(*hb, pose_b), margin, out),
            false,
        ),
    };
    if touched && reversed {
        out.normal = -out.normal;
    }
    touched && out.count > 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::math::{Quat, vec3};

    const BALL: ColliderShape = ColliderShape::Ball { radius: 0.5 };
    const BOX: ColliderShape = ColliderShape::Cuboid {
        half_extents: [0.5, 0.5, 0.5],
    };
    const CAPSULE: ColliderShape = ColliderShape::Capsule {
        half_height: 0.5,
        radius: 0.25,
    };

    fn pose(position: Vec3) -> Pose {
        Pose {
            position,
            rotation: Quat::IDENTITY,
        }
    }

    fn shapes() -> [ColliderShape; 3] {
        [BALL, BOX, CAPSULE]
    }

    // Whichever way round a pair arrives, the same contact must come back
    // with the normal reversed and the points in the same place.
    #[test]
    fn every_pair_is_symmetric_under_swapping_its_shapes() {
        for a in shapes() {
            for b in shapes() {
                let mut forward = Manifold::new(0, 1);
                let mut backward = Manifold::new(1, 0);
                // Offset on every axis: two capsules stacked exactly on one
                // line have no contact direction of their own, and the test
                // would be asking about a tie rather than about symmetry.
                let low = pose(Vec3::ZERO);
                let high = pose(vec3(0.3, 0.8, 0.1));
                let hit_forward = collide(&a, low, &b, high, 0.0, &mut forward);
                let hit_backward = collide(&b, high, &a, low, 0.0, &mut backward);
                assert_eq!(hit_forward, hit_backward, "{a:?} vs {b:?}");
                if !hit_forward {
                    continue;
                }
                assert!(
                    (forward.normal + backward.normal).length() < 1.0e-4,
                    "{a:?} vs {b:?}: {:?} {:?}",
                    forward.normal,
                    backward.normal
                );
                assert_eq!(forward.count, backward.count, "{a:?} vs {b:?}");
            }
        }
    }

    // Whatever the pair, the normal points from the first body to the second,
    // and overlap reads as a negative separation.
    #[test]
    fn every_pair_agrees_on_the_normal_direction_and_the_sign_of_overlap() {
        for a in shapes() {
            for b in shapes() {
                let mut m = Manifold::new(0, 1);
                assert!(
                    collide(
                        &a,
                        pose(Vec3::ZERO),
                        &b,
                        pose(vec3(0.0, 0.6, 0.0)),
                        0.0,
                        &mut m
                    ),
                    "{a:?} vs {b:?} must touch"
                );
                assert!(m.normal.y > 0.5, "{a:?} vs {b:?}: {:?}", m.normal);
                assert!(
                    m.points().iter().all(|p| p.separation < 0.0),
                    "{a:?} vs {b:?}: {m:?}"
                );
                assert!((m.normal.length() - 1.0).abs() < 1.0e-4, "{a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn every_pair_misses_when_the_shapes_are_far_apart() {
        for a in shapes() {
            for b in shapes() {
                let mut m = Manifold::new(0, 1);
                assert!(
                    !collide(
                        &a,
                        pose(Vec3::ZERO),
                        &b,
                        pose(vec3(0.0, 20.0, 0.0)),
                        0.0,
                        &mut m
                    ),
                    "{a:?} vs {b:?}"
                );
                assert_eq!(m.count, 0);
            }
        }
    }

    // Manifolds are reused between steps, so a miss has to leave no points
    // behind from the hit before it.
    #[test]
    fn a_miss_clears_the_points_a_previous_hit_left() {
        let mut m = Manifold::new(0, 1);
        assert!(collide(
            &BOX,
            pose(Vec3::ZERO),
            &BOX,
            pose(vec3(0.0, 0.9, 0.0)),
            0.0,
            &mut m
        ));
        assert!(m.count > 0);
        assert!(!collide(
            &BOX,
            pose(Vec3::ZERO),
            &BOX,
            pose(vec3(0.0, 9.0, 0.0)),
            0.0,
            &mut m
        ));
        assert_eq!(m.count, 0);
    }
}
