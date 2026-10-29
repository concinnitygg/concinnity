// The one question a sensor asks: are these two shapes in the same place.
//
// It is the distance query rather than the narrow phase because that is all
// the answer needs to be. A manifold is points, normals and separations built
// so a solver can push something out; a region pushes nothing out, and a
// yes-or-no read off the gap between the two surfaces costs one descent
// instead.
//
// Terrain never reaches here. A grid is immovable and so is a region, and two
// immovable things cannot have started overlapping.

use crate::physics::ColliderShape;
use crate::physics::sim::body::Body;
use crate::physics::sim::collide::Pose;
use crate::physics::sim::math::Vec3;
use crate::physics::sim::query::gjk::{self, Support};

/// Whether two bodies occupy any of the same space.
pub(crate) fn overlapping(a: &Body, b: &Body) -> bool {
    let (Some(shape_a), Some(shape_b)) = (a.convex(), b.convex()) else {
        return false;
    };
    if let Some(answer) =
        ball_in_box(shape_a, a, shape_b, b).or_else(|| ball_in_box(shape_b, b, shape_a, a))
    {
        return answer;
    }
    shapes_overlap(
        &Support::new(shape_a, pose_of(a)),
        &Support::new(shape_b, pose_of(b)),
    )
}

/// A ball against a box, answered in closed form rather than by a descent:
/// the point of the box nearest the ball's center is inside the ball or it is
/// not, and a center strictly inside the box overlaps whatever the radius.
/// Extents are taken as magnitudes, the way the descent reads them. `None` for
/// any other pair of shapes.
fn ball_in_box(
    ball: &ColliderShape,
    ball_at: &Body,
    cuboid: &ColliderShape,
    box_at: &Body,
) -> Option<bool> {
    let (ColliderShape::Ball { radius }, ColliderShape::Cuboid { half_extents }) = (ball, cuboid)
    else {
        return None;
    };
    let half = Vec3::from_array(*half_extents).abs();
    let radius = radius.abs();
    let local = box_at
        .orientation
        .inverse_rotate(ball_at.position - box_at.position);
    let inside = local
        .abs()
        .to_array()
        .iter()
        .zip(half.to_array())
        .all(|(c, h)| *c < h);
    let nearest = local.clamp(-half, half);
    Some(inside || (local - nearest).length_squared() < radius * radius)
}

/// The same question of two shapes at poses no body holds, which is what the
/// swept test asks of a mover part way along its path.
pub(crate) fn shapes_overlap(a: &Support, b: &Support) -> bool {
    let separation = gjk::separation(a, b);
    // A pair with no separating direction is inside itself; one with a
    // direction overlaps only while the surfaces have crossed.
    separation.is_entangled() || separation.gap < 0.0
}

fn pose_of(body: &Body) -> Pose {
    Pose {
        position: body.position,
        rotation: body.orientation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::LayerMask;
    use crate::physics::sim::math::{Quat, vec3};

    const UNIT: ColliderShape = ColliderShape::Cuboid {
        half_extents: [1.0, 1.0, 1.0],
    };

    fn at(shape: ColliderShape, position: Vec3) -> Body {
        Body::fixed(shape, position, Quat::IDENTITY, 0.0, LayerMask::ALL)
    }

    #[test]
    fn boxes_sharing_space_overlap_and_boxes_beside_each_other_do_not() {
        let region = at(UNIT, Vec3::ZERO);
        assert!(overlapping(&region, &at(UNIT, vec3(1.5, 0.0, 0.0))));
        assert!(overlapping(&region, &at(UNIT, Vec3::ZERO)));
        assert!(!overlapping(&region, &at(UNIT, vec3(2.5, 0.0, 0.0))));
    }

    // A rounded shape's surface stands off its core, so the answer has to be
    // about the surfaces and not about the cores.
    #[test]
    fn a_ball_is_measured_by_its_surface() {
        let region = at(UNIT, Vec3::ZERO);
        let ball = ColliderShape::Ball { radius: 0.5 };
        assert!(overlapping(&region, &at(ball, vec3(1.4, 0.0, 0.0))));
        assert!(!overlapping(&region, &at(ball, vec3(1.6, 0.0, 0.0))));
    }

    #[test]
    fn a_capsule_crossing_a_corner_is_found() {
        let region = at(UNIT, Vec3::ZERO);
        let capsule = ColliderShape::Capsule {
            half_height: 1.0,
            radius: 0.25,
        };
        assert!(overlapping(&region, &at(capsule, vec3(1.1, 1.5, 0.0))));
        assert!(!overlapping(&region, &at(capsule, vec3(1.5, 2.5, 0.0))));
    }

    // A turned region covers different space than an unturned one, and the
    // test has to read the pose rather than the bounds.
    #[test]
    fn a_turned_region_is_measured_where_it_actually_is() {
        let slab = ColliderShape::Cuboid {
            half_extents: [2.0, 0.2, 2.0],
        };
        let mut region = at(slab, Vec3::ZERO);
        let probe = at(ColliderShape::Ball { radius: 0.1 }, vec3(0.0, 1.5, 0.0));
        assert!(!overlapping(&region, &probe));
        region.orientation = Quat::from_euler_deg([0.0, 0.0, 90.0]);
        assert!(overlapping(&region, &probe), "the slab now stands upright");
    }

    // The descent reads a box's extents as magnitudes, so the closed form has
    // to as well, or a box authored with a negative extent answers one way at
    // its boundary and the other in the pass-through test.
    #[test]
    fn a_negative_extent_is_read_as_its_magnitude() {
        let flipped = at(
            ColliderShape::Cuboid {
                half_extents: [-1.0, 1.0, -1.0],
            },
            Vec3::ZERO,
        );
        let ball = ColliderShape::Ball { radius: 0.25 };
        for x in [0.0, 1.1, 1.3] {
            let probe = at(ball, vec3(x, 0.0, 0.0));
            let descended = shapes_overlap(
                &Support::new(&probe_shape(&flipped), pose_of(&flipped)),
                &Support::new(&ball, pose_of(&probe)),
            );
            assert_eq!(overlapping(&flipped, &probe), descended, "at x = {x}");
        }
        assert!(overlapping(&flipped, &at(ball, Vec3::ZERO)));
    }

    // A point inside a box is inside it, as the descent says, though no
    // distance from the box is below a zero radius.
    #[test]
    fn a_zero_radius_ball_inside_a_box_overlaps_it() {
        let region = at(UNIT, Vec3::ZERO);
        let point = ColliderShape::Ball { radius: 0.0 };
        assert!(overlapping(&region, &at(point, vec3(0.5, 0.0, 0.0))));
        assert!(!overlapping(&region, &at(point, vec3(1.5, 0.0, 0.0))));
    }

    fn probe_shape(body: &Body) -> ColliderShape {
        *body.convex().expect("a convex body")
    }

    // Terrain is the world rather than something in it, and has no convex
    // shape to measure against in any case.
    #[test]
    fn terrain_never_overlaps_a_region() {
        use crate::physics::sim::aabb::Aabb;
        let terrain = Body::terrain(0, Aabb::EMPTY, Vec3::ZERO, 1.0, LayerMask::ALL);
        assert!(!overlapping(&at(UNIT, Vec3::ZERO), &terrain));
        assert!(!overlapping(&terrain, &at(UNIT, Vec3::ZERO)));
    }

    // A turned box and a ball placed just inside and just outside reach of
    // every face, edge and corner, built in the box's own frame so the
    // expected answer is exact.
    #[test]
    fn a_ball_reaches_a_turned_box_exactly_as_far_as_its_radius() {
        let half = vec3(1.5, 0.5, 0.8);
        let radius = 0.4;
        let mut region = at(
            ColliderShape::Cuboid {
                half_extents: half.to_array(),
            },
            vec3(0.3, -0.2, 0.1),
        );
        region.orientation = Quat::from_euler_deg([20.0, 50.0, 70.0]);
        let mut checked = 0;
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    let outward = vec3(x as f32, y as f32, z as f32);
                    if outward == Vec3::ZERO {
                        continue;
                    }
                    let surface = half * outward;
                    let away = outward.normalize_or_zero();
                    for (reach, inside) in [(radius - 0.01, true), (radius + 0.01, false)] {
                        let local = surface + away * reach;
                        let world = region.position + region.orientation.rotate(local);
                        let probe = at(ColliderShape::Ball { radius }, world);
                        assert_eq!(overlapping(&probe, &region), inside, "{outward:?} {reach}");
                        assert_eq!(overlapping(&region, &probe), inside, "{outward:?} {reach}");
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 52);
        let center = at(ColliderShape::Ball { radius }, region.position);
        assert!(overlapping(&center, &region), "a ball inside the box");
    }
}
