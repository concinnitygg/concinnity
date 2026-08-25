// concinnity-physics/src/sim/query/sweep.rs
//
// How far a shape gets before it meets another one.
//
// Conservative advancement over the distance query next door, rather than a
// ray cast against the Minkowski sum. Both are correct; this one is chosen
// because the sweep is a pure translation. With no rotation to bound, the rate
// the gap can close at is exactly the motion projected on the current
// separating direction, so each step advances by the whole gap and is
// guaranteed not to step over the contact. That makes the advancement a dozen
// lines over a distance routine that is separately testable, where the ray-cast
// formulation would fold the search into the simplex iteration and be testable
// only as a whole.
//
// What it costs is linear convergence: a shape arriving almost tangentially
// takes many small steps. The iteration is capped, and a sweep that runs out
// reports the contact where it had got to. That is early rather than late,
// which is the direction a character controller can survive being wrong in.

use crate::sim::collide;
use crate::sim::contact::Manifold;
use crate::sim::math::Vec3;

use super::gjk::{Support, separation};

/// Where a swept shape first met another one.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SweepImpact {
    /// Fraction of the motion covered before the contact.
    pub(crate) toi: f32,
    /// World-space point on the shape that was hit.
    pub(crate) point: Vec3,
    /// Unit-length normal on the shape that was hit, pointing back toward the
    /// shape that swept into it.
    pub(crate) normal: Vec3,
    /// Distance between the two surfaces where the sweep stopped. Negative
    /// only when the sweep began overlapping, and then it is how far along
    /// `normal` the shape has to move to be clear.
    pub(crate) gap: f32,
    /// Whether the two were already touching before the sweep began.
    pub(crate) started_touching: bool,
}

/// Advances past this many have stopped closing the gap usefully.
const MAX_ADVANCES: usize = 32;

/// Gap at which the two count as touching. Small enough not to stop a shape
/// short of somewhere it fits, large enough that the advancement terminates.
pub(crate) const TOUCH_GAP: f32 = 1.0e-4;

/// One shape swept along `motion` against another that stays put.
///
/// `motion` is the whole displacement, so the returned time of impact is a
/// fraction of it. A zero motion reports only whether the two already touch.
pub(crate) fn sweep(moving: &Support, motion: Vec3, target: &Support) -> Option<SweepImpact> {
    let start = moving.pose.position;
    let mut probe = *moving;
    let mut toi = 0.0f32;

    for _ in 0..MAX_ADVANCES {
        probe.pose.position = start + motion * toi;
        let apart = separation(&probe, target);

        if apart.is_entangled() {
            // The cores overlap, so the distance query has no direction left.
            // The narrow phase does, and it is the same geometry.
            let overlap = entangled_contact(&probe, target, motion);
            return Some(SweepImpact {
                toi,
                point: overlap.point,
                normal: overlap.normal,
                gap: overlap.gap,
                started_touching: toi <= 0.0,
            });
        }
        if apart.gap <= TOUCH_GAP {
            return Some(SweepImpact {
                toi,
                point: apart.on_b,
                normal: apart.direction,
                gap: apart.gap,
                started_touching: toi <= 0.0,
            });
        }

        // The gap closes at most this fast, so advancing by the whole gap
        // cannot step past the contact.
        let closing = -motion.dot(apart.direction);
        if closing <= 0.0 {
            return None;
        }
        toi += apart.gap / closing;
        if toi > 1.0 {
            return None;
        }
    }

    // Out of advances with the gap still open: stop where the search got to
    // rather than let the shape through.
    let apart = separation(&probe, target);
    Some(SweepImpact {
        toi,
        point: apart.on_b,
        normal: apart.direction.normalize_or(-motion.normalize_or(Vec3::Y)),
        gap: apart.gap,
        started_touching: toi <= 0.0,
    })
}

/// The contact two overlapping shapes are read off the narrow phase as.
struct Overlap {
    normal: Vec3,
    point: Vec3,
    gap: f32,
}

/// The contact for two shapes whose cores already overlap, read off the narrow
/// phase. The target goes in first, so the manifold normal already points back
/// at the shape that swept into it.
fn entangled_contact(moving: &Support, target: &Support, motion: Vec3) -> Overlap {
    let mut scratch = Manifold::new(0, 0);
    let touched = collide::collide(
        &target.shape,
        target.pose,
        &moving.shape,
        moving.pose,
        0.0,
        &mut scratch,
    );
    // Nothing to read the geometry off: back along the motion is the only
    // direction the caller told us about.
    let fallback = (-motion).normalize_or(Vec3::Y);
    match (touched, scratch.points().first()) {
        (true, Some(first)) => Overlap {
            normal: scratch.normal,
            point: first.point,
            // The deepest point is what has to clear the surface.
            gap: scratch
                .points()
                .iter()
                .fold(first.separation, |deepest, p| deepest.min(p.separation)),
        },
        // No geometry to read a depth off, so report none: a caller that
        // separates by it stays put rather than being thrown somewhere.
        _ => Overlap {
            normal: fallback,
            point: moving.pose.position - fallback * moving.radius,
            gap: 0.0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ColliderShape;
    use crate::sim::collide::Pose;
    use crate::sim::math::{Quat, vec3};

    const BALL: ColliderShape = ColliderShape::Ball { radius: 0.5 };
    const CUBE: ColliderShape = ColliderShape::Cuboid {
        half_extents: [0.5, 0.5, 0.5],
    };
    const CAPSULE: ColliderShape = ColliderShape::Capsule {
        half_height: 0.5,
        radius: 0.25,
    };
    // A wall standing in the xy plane, faces at z = +/- 0.5.
    const WALL: ColliderShape = ColliderShape::Cuboid {
        half_extents: [3.0, 3.0, 0.5],
    };
    // A slab lying in the xz plane, faces at y = +/- 0.5.
    const FLOOR: ColliderShape = ColliderShape::Cuboid {
        half_extents: [5.0, 0.5, 5.0],
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

    fn run(
        shape: &ColliderShape,
        from: Vec3,
        motion: Vec3,
        target_shape: &ColliderShape,
        target: Vec3,
    ) -> Option<SweepImpact> {
        sweep(&at(shape, from), motion, &at(target_shape, target))
    }

    #[test]
    fn a_capsule_driven_at_a_wall_stops_against_its_face() {
        // Capsule radius 0.25 starting 4 back from a wall face at z = -0.5.
        let hit = run(
            &CAPSULE,
            vec3(0.0, 0.0, -4.5),
            vec3(0.0, 0.0, 8.0),
            &WALL,
            Vec3::ZERO,
        )
        .expect("a hit");
        let travelled = hit.toi * 8.0;
        assert!((travelled - 3.75).abs() < 1.0e-2, "{hit:?}");
        assert!((hit.normal + Vec3::Z).length() < 1.0e-2, "{hit:?}");
        assert!(!hit.started_touching);
        assert!((hit.point.z + 0.5).abs() < 1.0e-2, "{hit:?}");
    }

    #[test]
    fn every_shape_stops_where_its_own_reach_meets_the_floor() {
        // Half-extent along y toward the floor: ball 0.5, cube 0.5,
        // capsule 0.5 + 0.25.
        for (shape, reach) in [(BALL, 0.5), (CUBE, 0.5), (CAPSULE, 0.75)] {
            let hit = run(
                &shape,
                vec3(0.0, 6.0, 0.0),
                vec3(0.0, -10.0, 0.0),
                &FLOOR,
                Vec3::ZERO,
            )
            .expect("a hit");
            let landed = 6.0 - hit.toi * 10.0;
            assert!(
                (landed - (0.5 + reach)).abs() < 1.0e-2,
                "{shape:?} landed at {landed}"
            );
            assert!(
                (hit.normal - Vec3::Y).length() < 1.0e-2,
                "{shape:?} {hit:?}"
            );
        }
    }

    #[test]
    fn a_sweep_that_stops_short_of_the_target_reports_nothing() {
        assert!(
            run(
                &CAPSULE,
                vec3(0.0, 0.0, -8.0),
                vec3(0.0, 0.0, 2.0),
                &WALL,
                Vec3::ZERO
            )
            .is_none()
        );
    }

    #[test]
    fn a_sweep_pointed_away_reports_nothing() {
        assert!(
            run(
                &CAPSULE,
                vec3(0.0, 0.0, -4.0),
                vec3(0.0, 0.0, -20.0),
                &WALL,
                Vec3::ZERO
            )
            .is_none()
        );
    }

    #[test]
    fn a_sweep_beside_the_target_never_meets_it() {
        assert!(
            run(
                &BALL,
                vec3(20.0, 0.0, -4.0),
                vec3(0.0, 0.0, 8.0),
                &WALL,
                Vec3::ZERO
            )
            .is_none()
        );
    }

    // Sliding along a face, exactly touching it: the motion never closes the
    // gap, so it must not read as a block.
    #[test]
    fn a_sweep_grazing_along_a_face_is_not_a_block() {
        let hit = run(
            &BALL,
            vec3(-4.0, 1.01, 0.0),
            vec3(8.0, 0.0, 0.0),
            &FLOOR,
            Vec3::ZERO,
        );
        assert!(hit.is_none(), "{hit:?}");
    }

    #[test]
    fn a_sweep_that_begins_in_contact_says_so_at_once() {
        let hit = run(
            &BALL,
            vec3(0.0, 0.0, -0.9),
            vec3(0.0, 0.0, 4.0),
            &WALL,
            Vec3::ZERO,
        )
        .expect("a hit");
        assert_eq!(hit.toi, 0.0);
        assert!(hit.started_touching);
        assert!((hit.normal + Vec3::Z).length() < 1.0e-2, "{hit:?}");
    }

    // Deep overlap leaves the distance query nothing to work with, so the
    // narrow phase has to supply the direction out.
    #[test]
    fn a_sweep_beginning_deep_inside_still_reports_a_usable_normal() {
        let hit = run(
            &CUBE,
            vec3(0.0, 0.0, -0.2),
            vec3(0.0, 0.0, 4.0),
            &WALL,
            Vec3::ZERO,
        )
        .expect("a hit");
        assert_eq!(hit.toi, 0.0);
        assert!(hit.started_touching);
        assert!((hit.normal.length() - 1.0).abs() < 1.0e-3, "{hit:?}");
        assert!(hit.normal.z < -0.5, "out of the wall, not into it: {hit:?}");
    }

    #[test]
    fn a_zero_motion_reports_a_touch_and_nothing_else() {
        assert!(
            run(&BALL, vec3(0.0, 0.0, -4.0), Vec3::ZERO, &WALL, Vec3::ZERO).is_none(),
            "far away and going nowhere"
        );
        let touching =
            run(&BALL, vec3(0.0, 0.0, -0.9), Vec3::ZERO, &WALL, Vec3::ZERO).expect("a hit");
        assert!(touching.started_touching);
    }

    // The time of impact is a fraction of the motion, so doubling the motion
    // must halve it and leave the contact where it was.
    #[test]
    fn the_time_of_impact_scales_with_the_motion_it_is_a_fraction_of() {
        let short = run(
            &BALL,
            vec3(0.0, 0.0, -4.0),
            vec3(0.0, 0.0, 8.0),
            &WALL,
            Vec3::ZERO,
        )
        .expect("a hit");
        let long = run(
            &BALL,
            vec3(0.0, 0.0, -4.0),
            vec3(0.0, 0.0, 16.0),
            &WALL,
            Vec3::ZERO,
        )
        .expect("a hit");
        assert!((short.toi * 8.0 - long.toi * 16.0).abs() < 1.0e-3);
        assert!((short.point - long.point).length() < 1.0e-2);
    }

    #[test]
    fn a_sweep_at_a_corner_still_lands_on_the_surface() {
        let hit = run(
            &BALL,
            vec3(-6.0, 6.0, 0.0),
            vec3(6.0, -6.0, 0.0),
            &CUBE,
            Vec3::ZERO,
        )
        .expect("a hit");
        // The ball's centre at contact is 0.5 out from the box's corner.
        let centre = vec3(-6.0, 6.0, 0.0) + vec3(6.0, -6.0, 0.0) * hit.toi;
        let corner = vec3(-0.5, 0.5, 0.0);
        assert!(((centre - corner).length() - 0.5).abs() < 1.0e-2, "{hit:?}");
    }

    // The depth is what a caller separating from a surface it started inside
    // moves by, so it has to be the real overlap and not the touch tolerance.
    #[test]
    fn a_sweep_reports_how_far_it_has_to_move_to_be_clear() {
        // Ball radius 0.5 with its centre 0.4 from the wall face: 0.1 in.
        let inside = run(
            &BALL,
            vec3(0.0, 0.0, -0.9),
            vec3(0.0, 0.0, 4.0),
            &WALL,
            Vec3::ZERO,
        )
        .expect("a hit");
        assert!((inside.gap + 0.1).abs() < 1.0e-3, "{inside:?}");

        // Cores overlapping, so the depth comes off the narrow phase instead.
        let buried = run(
            &CUBE,
            vec3(0.0, 0.0, -0.2),
            vec3(0.0, 0.0, 4.0),
            &WALL,
            Vec3::ZERO,
        )
        .expect("a hit");
        assert!(buried.gap < -0.1, "{buried:?}");

        // A sweep that stopped short of the body is not inside anything.
        let stopped = run(
            &CAPSULE,
            vec3(0.0, 0.0, -4.5),
            vec3(0.0, 0.0, 8.0),
            &WALL,
            Vec3::ZERO,
        )
        .expect("a hit");
        assert!(
            stopped.gap >= 0.0 && stopped.gap <= TOUCH_GAP,
            "{stopped:?}"
        );
    }

    #[test]
    fn the_same_sweep_returns_the_same_bits() {
        let go = || {
            run(
                &CAPSULE,
                vec3(0.3, 2.0, -4.0),
                vec3(0.1, -1.0, 8.0),
                &WALL,
                Vec3::ZERO,
            )
            .expect("a hit")
        };
        assert_eq!(go().toi.to_bits(), go().toi.to_bits());
        assert_eq!(go().normal.to_array(), go().normal.to_array());
    }
}
