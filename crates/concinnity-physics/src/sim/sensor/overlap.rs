// concinnity-physics/src/sim/sensor/overlap.rs
//
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

use crate::sim::body::Body;
use crate::sim::collide::Pose;
use crate::sim::query::gjk::{self, Support};

/// Whether two bodies occupy any of the same space.
pub(crate) fn overlapping(a: &Body, b: &Body) -> bool {
    let (Some(shape_a), Some(shape_b)) = (a.convex(), b.convex()) else {
        return false;
    };
    shapes_overlap(
        &Support::new(shape_a, pose_of(a)),
        &Support::new(shape_b, pose_of(b)),
    )
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
    use crate::sim::math::{Quat, Vec3, vec3};
    use crate::{ColliderShape, LayerMask};

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

    // Terrain is the world rather than something in it, and has no convex
    // shape to measure against in any case.
    #[test]
    fn terrain_never_overlaps_a_region() {
        use crate::sim::aabb::Aabb;
        let terrain = Body::terrain(0, Aabb::EMPTY, Vec3::ZERO, 1.0, LayerMask::ALL);
        assert!(!overlapping(&at(UNIT, Vec3::ZERO), &terrain));
        assert!(!overlapping(&terrain, &at(UNIT, Vec3::ZERO)));
    }
}
