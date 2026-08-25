// concinnity-physics/src/sim/aabb.rs
//
// Axis-aligned bounds, the only thing the broad phase looks at. Bounds are
// grown by a fixed margin when they are stored so a body that moves a little
// does not invalidate them, and the sweep compares the grown boxes.

use crate::ColliderShape;

use super::math::{Mat3, Quat, Vec3, vec3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Aabb {
    pub(crate) min: Vec3,
    pub(crate) max: Vec3,
}

impl Aabb {
    /// Bounds that contain nothing, so a union with anything is that thing.
    pub(crate) const EMPTY: Aabb = Aabb {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
    };

    pub(crate) fn from_center_half_extents(center: Vec3, half_extents: Vec3) -> Self {
        Aabb {
            min: center - half_extents,
            max: center + half_extents,
        }
    }

    pub(crate) fn expanded(self, margin: f32) -> Self {
        let m = Vec3::splat(margin);
        Aabb {
            min: self.min - m,
            max: self.max + m,
        }
    }

    pub(crate) fn contains(self, other: Aabb) -> bool {
        self.min.x <= other.min.x
            && self.min.y <= other.min.y
            && self.min.z <= other.min.z
            && self.max.x >= other.max.x
            && self.max.y >= other.max.y
            && self.max.z >= other.max.z
    }

    pub(crate) fn overlaps(self, other: Aabb) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    pub(crate) fn center(self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// The smallest bounds containing both.
    pub(crate) fn union(self, other: Aabb) -> Self {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

/// World-space bounds of a shape at a pose, with no margin.
///
/// It is a free function rather than a body method because the query path
/// bounds shapes that no body owns: a swept capsule has a start pose and an
/// end pose, and neither is anybody's.
pub(crate) fn shape_bounds(shape: &ColliderShape, position: Vec3, rotation: Quat) -> Aabb {
    match *shape {
        ColliderShape::Ball { radius } => {
            Aabb::from_center_half_extents(position, Vec3::splat(libm::fabsf(radius)))
        }
        ColliderShape::Cuboid { half_extents } => {
            let h = Vec3::from_array(half_extents).abs();
            let r = Mat3::from_quat(rotation);
            // Extent along a world axis is the sum of each rotated local
            // axis's contribution to it.
            let extent = vec3(
                libm::fabsf(r.cols[0].x) * h.x
                    + libm::fabsf(r.cols[1].x) * h.y
                    + libm::fabsf(r.cols[2].x) * h.z,
                libm::fabsf(r.cols[0].y) * h.x
                    + libm::fabsf(r.cols[1].y) * h.y
                    + libm::fabsf(r.cols[2].y) * h.z,
                libm::fabsf(r.cols[0].z) * h.x
                    + libm::fabsf(r.cols[1].z) * h.y
                    + libm::fabsf(r.cols[2].z) * h.z,
            );
            Aabb::from_center_half_extents(position, extent)
        }
        ColliderShape::Capsule {
            half_height,
            radius,
        } => {
            let axis = rotation.rotate(Vec3::Y) * libm::fabsf(half_height);
            let r = Vec3::splat(libm::fabsf(radius));
            let a = position - axis;
            let b = position + axis;
            Aabb {
                min: a.min(b) - r,
                max: a.max(b) + r,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::math::vec3;

    #[test]
    fn overlap_is_inclusive_at_the_touching_face() {
        let a = Aabb::from_center_half_extents(Vec3::ZERO, Vec3::splat(1.0));
        let touching = Aabb::from_center_half_extents(vec3(2.0, 0.0, 0.0), Vec3::splat(1.0));
        let apart = Aabb::from_center_half_extents(vec3(2.01, 0.0, 0.0), Vec3::splat(1.0));
        assert!(a.overlaps(touching));
        assert!(!a.overlaps(apart));
        assert!(!apart.overlaps(a));
    }

    #[test]
    fn separation_on_any_single_axis_is_enough_to_miss() {
        let a = Aabb::from_center_half_extents(Vec3::ZERO, Vec3::splat(1.0));
        for axis in 0..3 {
            let mut center = Vec3::ZERO;
            center.set(axis, 3.0);
            let b = Aabb::from_center_half_extents(center, Vec3::splat(1.0));
            assert!(!a.overlaps(b), "axis {axis}");
        }
    }

    #[test]
    fn expansion_grows_every_face_and_containment_follows() {
        let a = Aabb::from_center_half_extents(Vec3::ZERO, Vec3::splat(1.0));
        let grown = a.expanded(0.5);
        assert_eq!(grown.min, Vec3::splat(-1.5));
        assert_eq!(grown.max, Vec3::splat(1.5));
        assert!(grown.contains(a));
        assert!(!a.contains(grown));
        assert_eq!(grown.center(), Vec3::ZERO);
    }

    #[test]
    fn a_union_covers_both_operands_and_absorbs_empty() {
        let a = Aabb::from_center_half_extents(Vec3::ZERO, Vec3::splat(1.0));
        let b = Aabb::from_center_half_extents(vec3(4.0, 0.0, 0.0), Vec3::splat(1.0));
        let joined = a.union(b);
        assert!(joined.contains(a) && joined.contains(b));
        assert_eq!(joined.min, vec3(-1.0, -1.0, -1.0));
        assert_eq!(joined.max, vec3(5.0, 1.0, 1.0));
        assert_eq!(Aabb::EMPTY.union(a), a);
    }

    #[test]
    fn shape_bounds_cover_every_shape_at_its_pose() {
        let ball = shape_bounds(
            &ColliderShape::Ball { radius: 0.5 },
            Vec3::Y,
            Quat::IDENTITY,
        );
        assert_eq!(ball.min, vec3(-0.5, 0.5, -0.5));
        assert_eq!(ball.max, vec3(0.5, 1.5, 0.5));

        // Rolled onto its side, a capsule is long in x and thin in y.
        let capsule = shape_bounds(
            &ColliderShape::Capsule {
                half_height: 1.0,
                radius: 0.25,
            },
            Vec3::ZERO,
            Quat::from_euler_deg([0.0, 0.0, 90.0]),
        );
        assert!((capsule.max.x - 1.25).abs() < 1.0e-5, "{capsule:?}");
        assert!((capsule.max.y - 0.25).abs() < 1.0e-5, "{capsule:?}");

        // A box turned 45 degrees about Y bounds wider than the box itself.
        let turned = shape_bounds(
            &ColliderShape::Cuboid {
                half_extents: [1.0, 0.5, 1.0],
            },
            Vec3::ZERO,
            Quat::from_euler_deg([0.0, 45.0, 0.0]),
        );
        assert!(
            (turned.max.x - libm::sqrtf(2.0)).abs() < 1.0e-5,
            "{turned:?}"
        );
        assert!((turned.max.y - 0.5).abs() < 1.0e-5, "{turned:?}");
    }

    #[test]
    fn empty_bounds_contain_nothing_and_overlap_nothing() {
        let a = Aabb::from_center_half_extents(Vec3::ZERO, Vec3::splat(1.0));
        assert!(!Aabb::EMPTY.overlaps(a));
        assert!(!Aabb::EMPTY.contains(a));
    }
}
