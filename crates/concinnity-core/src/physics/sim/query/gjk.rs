// How far apart two shapes are, and along which direction.
//
// Every shape the simulation has is a box rounded by a radius: a ball is a
// zero-extent box rounded by its radius, a capsule is a box with extent on one
// axis only, and a cuboid is a box rounded by nothing. That collapses three
// support functions into one, and it keeps the polytope GJK iterates over
// finite -- rounding is subtracted from the answer at the end rather than
// being built into a curved support function that GJK converges on slowly.
//
// The distance is between the cores. Subtracting the two radii turns it into
// the gap between the surfaces, and that gap goes negative before the cores
// ever meet, which is what lets a sweep stop at a touch rather than at an
// overlap.

use crate::physics::ColliderShape;

use crate::physics::sim::collide::{Pose, support_vertex};
use crate::physics::sim::math::Vec3;

use super::simplex::closest_to_origin;

/// A shape as the distance query sees it: a box core, a rounding radius, and
/// where the pair of them sits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Support {
    /// Half-extents of the core box, in the shape's own frame.
    half: Vec3,
    /// How far the surface stands off the core.
    pub(crate) radius: f32,
    pub(crate) pose: Pose,
    /// What the core and radius were derived from, kept so a caller that
    /// needs the narrow phase does not have to carry the shape alongside.
    pub(crate) shape: ColliderShape,
}

impl Support {
    pub(crate) fn new(shape: &ColliderShape, pose: Pose) -> Self {
        let (half, radius) = match *shape {
            ColliderShape::Ball { radius } => (Vec3::ZERO, radius.abs()),
            ColliderShape::Capsule {
                half_height,
                radius,
            } => (
                Vec3::from_array([0.0, half_height.abs(), 0.0]),
                radius.abs(),
            ),
            ColliderShape::Cuboid { half_extents } => (Vec3::from_array(half_extents).abs(), 0.0),
        };
        Support {
            half,
            radius,
            pose,
            shape: *shape,
        }
    }

    /// The core point furthest along a world-space direction.
    fn core_support(&self, direction: Vec3) -> Vec3 {
        let local = self.pose.rotation.inverse_rotate(direction);
        self.pose.to_world(support_vertex(self.half, local))
    }

    /// Whether a world-space point is inside the rounded shape.
    pub(crate) fn contains(&self, point: Vec3) -> bool {
        let local = self.pose.to_local(point);
        let nearest = local.clamp(-self.half, self.half);
        (local - nearest).length_squared() <= self.radius * self.radius
    }
}

/// How far apart two shapes are, and which way they are apart.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Separation {
    /// Gap between the two surfaces. Negative when they overlap.
    pub(crate) gap: f32,
    /// Unit vector pointing from `b` toward `a`, or zero when the cores
    /// overlap and there is no direction to give.
    pub(crate) direction: Vec3,
    /// Closest point on `b`'s surface. Meaningless with a zero direction.
    pub(crate) on_b: Vec3,
}

impl Separation {
    /// Whether the cores overlap, leaving no separating direction to report.
    pub(crate) fn is_entangled(&self) -> bool {
        self.direction == Vec3::ZERO
    }
}

/// Support points past this many have stopped buying accuracy.
const MAX_ITERATIONS: usize = 32;

/// Relative slack on the distance bound. A support point that improves the
/// answer by less than this fraction is the answer.
const TOLERANCE: f32 = 1.0e-6;

/// The gap between two shapes, with the direction and witness point that go
/// with it.
///
/// Both shapes are convex, so the closest point on the Minkowski difference of
/// their cores to the origin is the whole answer, and that is what the
/// iteration below converges on.
pub(crate) fn separation(a: &Support, b: &Support) -> Separation {
    let radii = a.radius + b.radius;

    let mut cso = [Vec3::ZERO; 4];
    let mut witness_b = [Vec3::ZERO; 4];
    let start = (a.pose.position - b.pose.position).normalize_or(Vec3::X);
    (cso[0], witness_b[0]) = support(a, b, start);
    let mut count = 1usize;
    let mut closest = closest_to_origin(&cso[..count]);
    // Whether the last support point put the origin outside a supporting
    // plane. Without that certificate the simplex's answer is an upper bound
    // on a distance that may well be zero, and reporting it as a separation
    // would hand a caller a direction pointing the wrong way out of an
    // overlap.
    let mut proven_apart = false;

    for _ in 0..MAX_ITERATIONS {
        let v = closest.point;
        let v_dot_v = v.length_squared();
        if closest.encloses_origin || v_dot_v <= f32::MIN_POSITIVE {
            return entangled(radii);
        }
        // The far side of the difference along -v bounds how much closer the
        // true answer can be than the simplex's.
        let (point, on_b) = support(a, b, -v);
        let bound = v.dot(point);
        proven_apart = bound > 0.0;
        // A support point that buys nothing, or one the simplex already
        // holds, means the iteration has arrived.
        if v_dot_v - bound <= TOLERANCE * v_dot_v || cso[..count].contains(&point) {
            break;
        }

        cso[count] = point;
        witness_b[count] = on_b;
        count += 1;
        closest = closest_to_origin(&cso[..count]);
        if closest.encloses_origin {
            return entangled(radii);
        }
        // Keep only the vertices carrying the answer, so the simplex always
        // has room for the next support point.
        count = closest.count;
        let (mut kept_cso, mut kept_witness) = ([Vec3::ZERO; 4], [Vec3::ZERO; 4]);
        for i in 0..count {
            kept_cso[i] = cso[closest.keep[i]];
            kept_witness[i] = witness_b[closest.keep[i]];
            closest.keep[i] = i;
        }
        (cso, witness_b) = (kept_cso, kept_witness);
    }

    let core_distance = closest.point.length();
    if !proven_apart || core_distance <= f32::MIN_POSITIVE {
        return entangled(radii);
    }
    let direction = closest.point * (1.0 / core_distance);

    let mut on_b_core = Vec3::ZERO;
    for i in 0..closest.count {
        on_b_core += witness_b[closest.keep[i]] * closest.weights[i];
    }

    Separation {
        gap: core_distance - radii,
        direction,
        on_b: on_b_core + direction * b.radius,
    }
}

/// The Minkowski-difference support point along `direction`, with the point
/// on `b`'s core that produced it.
fn support(a: &Support, b: &Support, direction: Vec3) -> (Vec3, Vec3) {
    let on_a = a.core_support(direction);
    let on_b = b.core_support(-direction);
    (on_a - on_b, on_b)
}

fn entangled(radii: f32) -> Separation {
    Separation {
        gap: -radii,
        direction: Vec3::ZERO,
        on_b: Vec3::ZERO,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::sim::math::{Quat, vec3};

    const BALL: ColliderShape = ColliderShape::Ball { radius: 0.5 };
    const CUBE: ColliderShape = ColliderShape::Cuboid {
        half_extents: [0.5, 0.5, 0.5],
    };
    const CAPSULE: ColliderShape = ColliderShape::Capsule {
        half_height: 0.5,
        radius: 0.25,
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

    fn turned(shape: &ColliderShape, position: Vec3, euler_deg: [f32; 3]) -> Support {
        Support::new(
            shape,
            Pose {
                position,
                rotation: Quat::from_euler_deg(euler_deg),
            },
        )
    }

    #[test]
    fn two_balls_are_apart_by_the_gap_between_their_surfaces() {
        let a = at(&BALL, vec3(0.0, 3.0, 0.0));
        let b = at(&BALL, Vec3::ZERO);
        let s = separation(&a, &b);
        assert!((s.gap - 2.0).abs() < 1.0e-4, "{s:?}");
        assert!((s.direction - Vec3::Y).length() < 1.0e-4, "{s:?}");
        assert!((s.on_b - vec3(0.0, 0.5, 0.0)).length() < 1.0e-4, "{s:?}");
    }

    #[test]
    fn overlapping_balls_report_a_negative_gap_and_still_give_a_direction() {
        let s = separation(&at(&BALL, vec3(0.0, 0.6, 0.0)), &at(&BALL, Vec3::ZERO));
        assert!((s.gap - (-0.4)).abs() < 1.0e-4, "{s:?}");
        assert!(!s.is_entangled(), "the cores are two distinct points");
        assert!((s.direction - Vec3::Y).length() < 1.0e-4, "{s:?}");
    }

    // Two boxes have no rounding, so an overlap is an overlap of the cores
    // themselves and there is no direction left to report.
    #[test]
    fn overlapping_boxes_report_themselves_entangled() {
        let s = separation(&at(&CUBE, vec3(0.0, 0.2, 0.0)), &at(&CUBE, Vec3::ZERO));
        assert!(s.is_entangled(), "{s:?}");
        assert!(s.gap <= 0.0, "{s:?}");
    }

    // A small box buried in a much larger one is the case the iteration is
    // most likely to stall on, and a stall must read as an overlap rather
    // than as some direction it happened to be holding.
    #[test]
    fn a_box_buried_in_a_far_larger_one_is_entangled_whatever_the_disparity() {
        let slab = ColliderShape::Cuboid {
            half_extents: [3.0, 3.0, 0.5],
        };
        for offset in [0.0, 0.1, -0.2, 0.35] {
            let s = separation(&at(&CUBE, vec3(0.0, 0.0, offset)), &at(&slab, Vec3::ZERO));
            assert!(s.is_entangled(), "offset {offset}: {s:?}");
        }
        // Clear of it again, and the direction comes back.
        let clear = separation(&at(&CUBE, vec3(0.0, 0.0, -2.0)), &at(&slab, Vec3::ZERO));
        assert!(!clear.is_entangled(), "{clear:?}");
        assert!((clear.gap - 1.0).abs() < 1.0e-4, "{clear:?}");
    }

    // A capsule buried in a wall keeps a direction: the cores are a segment
    // and a box, and the segment is still outside the box.
    #[test]
    fn a_rounded_shape_overlapping_a_box_still_reports_which_way_out() {
        let slab = ColliderShape::Cuboid {
            half_extents: [3.0, 3.0, 0.5],
        };
        let s = separation(&at(&BALL, vec3(0.0, 0.0, -0.8)), &at(&slab, Vec3::ZERO));
        assert!(!s.is_entangled(), "{s:?}");
        assert!((s.gap - (-0.2)).abs() < 1.0e-4, "{s:?}");
        assert!((s.direction + Vec3::Z).length() < 1.0e-4, "{s:?}");
    }

    #[test]
    fn a_box_and_a_ball_agree_with_the_distance_worked_out_by_hand() {
        // Ball centre 3 up, box top face at 0.5, ball radius 0.5.
        let s = separation(&at(&BALL, vec3(0.0, 3.0, 0.0)), &at(&CUBE, Vec3::ZERO));
        assert!((s.gap - 2.0).abs() < 1.0e-4, "{s:?}");
        assert!((s.on_b - vec3(0.0, 0.5, 0.0)).length() < 1.0e-4, "{s:?}");

        // Off the corner: the diagonal from the box corner to the ball centre.
        let corner = separation(&at(&BALL, vec3(3.0, 3.0, 3.0)), &at(&CUBE, Vec3::ZERO));
        let expected = (vec3(3.0, 3.0, 3.0) - vec3(0.5, 0.5, 0.5)).length() - 0.5;
        assert!((corner.gap - expected).abs() < 1.0e-3, "{corner:?}");
    }

    #[test]
    fn two_boxes_face_to_face_are_apart_by_the_gap_between_their_faces() {
        let s = separation(&at(&CUBE, vec3(0.0, 4.0, 0.0)), &at(&CUBE, Vec3::ZERO));
        assert!((s.gap - 3.0).abs() < 1.0e-4, "{s:?}");
        assert!((s.direction - Vec3::Y).length() < 1.0e-3, "{s:?}");
    }

    #[test]
    fn a_capsule_is_measured_from_its_segment_not_its_centre() {
        // Upright capsule: the cap reaches 0.75 above centre.
        let s = separation(&at(&CAPSULE, vec3(0.0, 4.0, 0.0)), &at(&CUBE, Vec3::ZERO));
        assert!((s.gap - (4.0 - 0.75 - 0.5)).abs() < 1.0e-4, "{s:?}");

        // Laid on its side, the same capsule reaches only 0.25 down.
        let sideways = separation(
            &turned(&CAPSULE, vec3(0.0, 4.0, 0.0), [0.0, 0.0, 90.0]),
            &at(&CUBE, Vec3::ZERO),
        );
        assert!(
            (sideways.gap - (4.0 - 0.25 - 0.5)).abs() < 1.0e-4,
            "{sideways:?}"
        );
    }

    #[test]
    fn two_crossed_capsules_are_apart_by_the_gap_between_their_segments() {
        let a = turned(&CAPSULE, vec3(0.0, 2.0, 0.0), [0.0, 0.0, 90.0]);
        let b = turned(&CAPSULE, Vec3::ZERO, [90.0, 0.0, 0.0]);
        let s = separation(&a, &b);
        assert!((s.gap - (2.0 - 0.5)).abs() < 1.0e-3, "{s:?}");
    }

    // The gap does not depend on which shape is asked about first, and the
    // direction simply turns around.
    #[test]
    fn swapping_the_pair_reverses_the_direction_and_keeps_the_gap() {
        for pair in [(BALL, CUBE), (CAPSULE, CUBE), (CAPSULE, BALL), (CUBE, CUBE)] {
            let a = turned(&pair.0, vec3(1.5, 2.5, 0.5), [10.0, 20.0, 30.0]);
            let b = turned(&pair.1, Vec3::ZERO, [0.0, 45.0, 0.0]);
            let forward = separation(&a, &b);
            let backward = separation(&b, &a);
            assert!(
                (forward.gap - backward.gap).abs() < 1.0e-3,
                "{pair:?}: {forward:?} {backward:?}"
            );
            assert!(
                (forward.direction + backward.direction).length() < 1.0e-3,
                "{pair:?}: {forward:?} {backward:?}"
            );
        }
    }

    // The witness point has to sit on the far shape's surface: a sweep uses it
    // as the contact point.
    #[test]
    fn the_witness_point_lands_on_the_far_shapes_surface() {
        for shape in [BALL, CUBE, CAPSULE] {
            let a = at(&BALL, vec3(0.0, 5.0, 0.0));
            let b = turned(&shape, Vec3::ZERO, [0.0, 30.0, 0.0]);
            let s = separation(&a, &b);
            assert!(b.contains(s.on_b + s.direction * -1.0e-3), "{shape:?}");
            assert!(!b.contains(s.on_b + s.direction * 1.0e-2), "{shape:?}");
        }
    }

    #[test]
    fn containment_answers_for_every_shape() {
        assert!(at(&BALL, Vec3::ZERO).contains(vec3(0.0, 0.4, 0.0)));
        assert!(!at(&BALL, Vec3::ZERO).contains(vec3(0.0, 0.6, 0.0)));
        assert!(at(&CUBE, Vec3::ZERO).contains(vec3(0.4, -0.4, 0.4)));
        assert!(!at(&CUBE, Vec3::ZERO).contains(vec3(0.6, 0.0, 0.0)));
        // The capsule reaches 0.75 up the y axis but only 0.25 out on x.
        assert!(at(&CAPSULE, Vec3::ZERO).contains(vec3(0.0, 0.7, 0.0)));
        assert!(!at(&CAPSULE, Vec3::ZERO).contains(vec3(0.0, 0.8, 0.0)));
        assert!(!at(&CAPSULE, Vec3::ZERO).contains(vec3(0.3, 0.0, 0.0)));
    }

    // Two runs of the same query must land on the same bits: a query whose
    // answer drifted would take the whole simulation with it.
    #[test]
    fn the_same_query_returns_the_same_bits() {
        let a = turned(&CAPSULE, vec3(0.3, 2.7, -1.1), [15.0, 40.0, 5.0]);
        let b = turned(&CUBE, Vec3::ZERO, [0.0, 22.5, 0.0]);
        let first = separation(&a, &b);
        let second = separation(&a, &b);
        assert_eq!(first.gap.to_bits(), second.gap.to_bits());
        assert_eq!(first.direction.to_array(), second.direction.to_array());
    }
}
