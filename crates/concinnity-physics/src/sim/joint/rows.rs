// concinnity-physics/src/sim/joint/rows.rs
//
// The linear algebra a joint row is solved through, with no joint in sight.
//
// Every joint kind reduces to the same two effective-mass blocks: one for a
// point held between two bodies, one for their relative rotation. A row is
// then that block read along one direction, two of them, or all three, which
// is why the kinds share this module rather than each carrying their own
// arithmetic. Splitting it out is also what makes the blocks testable against
// hand-worked answers instead of only through a swinging pendulum.
//
// The two scalar solves at the bottom mirror the contact solver's: an
// inequality row pushes softly once it is passed and speculatively while it is
// being approached, and a motor row is an equality row with a ceiling.

use crate::sim::config::Softness;
use crate::sim::math::{Mat3, Vec3};

/// One body's contribution to a block: how freely it moves, and about what.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Arm {
    pub(crate) inv_mass: f32,
    pub(crate) inv_inertia: Mat3,
    /// Offset from the body's centre to the point the row acts through.
    pub(crate) lever: Vec3,
}

/// The mass a point constraint between two bodies is seen through.
///
/// `(1/ma + 1/mb) I - S(ra) Ia S(ra) - S(rb) Ib S(rb)`.
pub(crate) fn point_block(a: Arm, b: Arm) -> Mat3 {
    let linear = a.inv_mass + b.inv_mass;
    let (ka, kb) = (lever_block(a), lever_block(b));
    let mut cols = [
        ka.cols[0] + kb.cols[0],
        ka.cols[1] + kb.cols[1],
        ka.cols[2] + kb.cols[2],
    ];
    cols[0].x += linear;
    cols[1].y += linear;
    cols[2].z += linear;
    Mat3::from_cols(cols[0], cols[1], cols[2])
}

/// One arm's `-S(r) Iinv S(r)`: the mass its lever adds across itself.
///
/// Taken through the columns of `Iinv S(r)`, which are three scalings of the
/// tensor's own columns, rather than by pushing each basis vector through two
/// cross products and the tensor. This block is rebuilt for every joint on
/// every solver pass, so what it costs is what a joint costs.
fn lever_block(arm: Arm) -> Mat3 {
    // A body that cannot turn adds nothing, and every joint anchored to the
    // world has one.
    if arm.inv_inertia == Mat3::ZERO {
        return Mat3::ZERO;
    }
    let (c, r) = (arm.inv_inertia.cols, arm.lever);
    // `S(r)`'s columns are `(0, z, -y)`, `(-z, 0, x)` and `(y, -x, 0)`.
    let through = [
        c[1] * r.z - c[2] * r.y,
        c[2] * r.x - c[0] * r.z,
        c[0] * r.y - c[1] * r.x,
    ];
    // `-S(r) v` is `v x r`.
    Mat3::from_cols(
        through[0].cross(r),
        through[1].cross(r),
        through[2].cross(r),
    )
}

/// The mass a relative-rotation constraint is seen through.
pub(crate) fn angular_block(inv_inertia_a: Mat3, inv_inertia_b: Mat3) -> Mat3 {
    inv_inertia_a.add(inv_inertia_b)
}

/// The impulse that cancels `rhs` across all three rows of a block.
pub(crate) fn solve_block(block: &Mat3, rhs: Vec3) -> Vec3 {
    -block.inverse().mul_vec3(rhs)
}

/// The impulse that cancels `rhs` across the two rows `t1` and `t2` span,
/// leaving the direction they are perpendicular to free.
pub(crate) fn solve_plane(block: &Mat3, t1: Vec3, t2: Vec3, rhs: [f32; 2]) -> Vec3 {
    let (k1, k2) = (block.mul_vec3(t1), block.mul_vec3(t2));
    let (m00, m01) = (t1.dot(k1), t1.dot(k2));
    let (m10, m11) = (t2.dot(k1), t2.dot(k2));
    let determinant = m00 * m11 - m01 * m10;
    if libm::fabsf(determinant) <= f32::MIN_POSITIVE {
        return Vec3::ZERO;
    }
    let inv = 1.0 / determinant;
    let x = -(m11 * rhs[0] - m01 * rhs[1]) * inv;
    let y = -(m00 * rhs[1] - m10 * rhs[0]) * inv;
    t1 * x + t2 * y
}

/// The mass a single row along `axis` is seen through. Zero when nothing along
/// it can move.
pub(crate) fn axis_mass(block: &Mat3, axis: Vec3) -> f32 {
    let k = axis.dot(block.mul_vec3(axis));
    if k > f32::MIN_POSITIVE { 1.0 / k } else { 0.0 }
}

/// One inequality row: how far inside its bound the joint is, how fast that is
/// changing, and the mass it is seen through.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LimitRow {
    /// Distance left before the bound is passed. Negative once it is.
    pub(crate) separation: f32,
    /// Rate `separation` is changing at.
    pub(crate) rate: f32,
    pub(crate) mass: f32,
}

/// How stiffly a bound is enforced, and whether this pass corrects position at
/// all.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Push {
    pub(crate) soft: Softness,
    pub(crate) inv_h: f32,
    /// Ceiling on the speed a passed bound is pushed back out at.
    pub(crate) max_push: f32,
    pub(crate) use_bias: bool,
}

/// Advance an inequality row, returning the impulse to apply along its axis.
///
/// The accumulated impulse may only push, so a joint inside its limits carries
/// nothing and a joint on one holds itself there.
pub(crate) fn solve_limit(row: LimitRow, total: &mut f32, push: &Push) -> f32 {
    let (bias, mass_scale, impulse_scale) = if row.separation > 0.0 {
        // Still short of the bound: allow exactly enough approach to reach it
        // this substep and no more.
        (row.separation * push.inv_h, 1.0, 0.0)
    } else if push.use_bias {
        (
            (push.soft.bias_rate * row.separation).max(-push.max_push),
            push.soft.mass_scale,
            push.soft.impulse_scale,
        )
    } else {
        (0.0, 1.0, 0.0)
    };
    let delta = -row.mass * mass_scale * (row.rate + bias) - impulse_scale * *total;
    let next = (*total + delta).max(0.0);
    let applied = next - *total;
    *total = next;
    applied
}

/// Advance a motor row, returning the impulse to apply along its axis.
///
/// `budget` is the most the motor may have applied by the end of the substep,
/// which is what turns an authored force ceiling into something a velocity
/// solve can honour.
pub(crate) fn solve_motor(error: f32, mass: f32, total: &mut f32, budget: f32) -> f32 {
    let next = (*total - mass * error).clamp(-budget, budget);
    let applied = next - *total;
    *total = next;
    applied
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::math::vec3;

    fn free_point(inv_mass: f32) -> Arm {
        Arm {
            inv_mass,
            inv_inertia: Mat3::ZERO,
            lever: Vec3::ZERO,
        }
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1.0e-5
    }

    // Two point masses with no lever arms see the reduced mass, whichever
    // direction the row is read along.
    #[test]
    fn two_point_masses_reduce_to_their_summed_inverse_mass() {
        let block = point_block(free_point(0.5), free_point(0.25));
        for axis in [Vec3::X, Vec3::Y, Vec3::Z, vec3(0.6, 0.8, 0.0)] {
            assert!(
                close(axis_mass(&block, axis), 1.0 / 0.75),
                "{axis:?} -> {}",
                axis_mass(&block, axis)
            );
        }
    }

    // A lever arm makes a body harder to shift across the arm and no harder
    // along it, which is the whole reason the block is a matrix.
    #[test]
    fn a_lever_arm_stiffens_the_rows_across_it_and_leaves_the_one_along_it() {
        let arm = Arm {
            inv_mass: 1.0,
            inv_inertia: Mat3::from_diagonal(Vec3::splat(1.0)),
            lever: Vec3::X,
        };
        let block = point_block(arm, free_point(0.0));
        assert!(close(axis_mass(&block, Vec3::X), 1.0), "along the arm");
        assert!(
            axis_mass(&block, Vec3::Y) < 1.0,
            "across it: {}",
            axis_mass(&block, Vec3::Y)
        );
    }

    #[test]
    fn an_immovable_pair_has_no_mass_to_solve_through() {
        let block = point_block(free_point(0.0), free_point(0.0));
        assert_eq!(solve_block(&block, vec3(1.0, 2.0, 3.0)), Vec3::ZERO);
        assert_eq!(
            solve_plane(&block, Vec3::X, Vec3::Y, [1.0, 1.0]),
            Vec3::ZERO
        );
        assert_eq!(axis_mass(&block, Vec3::Y), 0.0);
    }

    // The point of the block solve: one impulse cancels all three rows at
    // once, where three separate rows would each disturb the others.
    #[test]
    fn a_block_solve_cancels_every_row_it_covers() {
        let a = Arm {
            inv_mass: 1.0,
            inv_inertia: Mat3::from_diagonal(vec3(2.0, 1.0, 0.5)),
            lever: vec3(0.3, -0.7, 0.2),
        };
        let b = Arm {
            inv_mass: 0.5,
            inv_inertia: Mat3::from_diagonal(vec3(1.5, 0.8, 1.2)),
            lever: vec3(-0.1, 0.4, 0.6),
        };
        let block = point_block(a, b);
        let velocity = vec3(0.7, -1.3, 0.4);
        let impulse = solve_block(&block, velocity);
        let after = velocity + block.mul_vec3(impulse);
        assert!(after.length() < 1.0e-4, "{after:?}");
    }

    // The plane solve has to cancel the two rows it covers and leave the third
    // alone, or a hinge would stop turning.
    #[test]
    fn a_plane_solve_cancels_its_two_rows_and_touches_no_others() {
        let block = angular_block(
            Mat3::from_diagonal(vec3(2.0, 1.0, 3.0)),
            Mat3::from_diagonal(vec3(0.5, 1.5, 0.25)),
        );
        let (t1, t2) = (Vec3::X, Vec3::Y);
        let spin = vec3(0.4, -0.9, 2.0);
        let impulse = solve_plane(&block, t1, t2, [spin.dot(t1), spin.dot(t2)]);
        let after = spin + block.mul_vec3(impulse);
        assert!(
            close(after.dot(t1), 0.0) && close(after.dot(t2), 0.0),
            "{after:?}"
        );
        assert!(
            close(after.z, 2.0),
            "the free axis kept its spin: {after:?}"
        );
    }

    #[test]
    fn an_angular_block_is_the_two_inverse_tensors_together() {
        let block = angular_block(
            Mat3::from_diagonal(vec3(1.0, 2.0, 4.0)),
            Mat3::from_diagonal(vec3(1.0, 2.0, 4.0)),
        );
        assert!(close(axis_mass(&block, Vec3::X), 0.5));
        assert!(close(axis_mass(&block, Vec3::Z), 0.125));
    }

    fn push(use_bias: bool) -> Push {
        Push {
            soft: Softness::new(60.0, 2.0, 1.0 / 240.0),
            inv_h: 240.0,
            max_push: 3.0,
            use_bias,
        }
    }

    // A limit row is an inequality: it pushes out of a bound it has passed and
    // never pulls back toward one it has not.
    #[test]
    fn a_limit_row_pushes_out_of_its_bound_and_never_pulls_into_it() {
        let mut total = 0.0;
        let applied = solve_limit(
            LimitRow {
                separation: -0.1,
                rate: 0.0,
                mass: 1.0,
            },
            &mut total,
            &push(true),
        );
        assert!(applied > 0.0 && total > 0.0, "{applied} {total}");

        let mut idle = 0.0;
        let none = solve_limit(
            LimitRow {
                separation: 0.5,
                rate: 0.0,
                mass: 1.0,
            },
            &mut idle,
            &push(true),
        );
        assert_eq!((none, idle), (0.0, 0.0), "a bound in the distance is free");
    }

    // Approaching a bound is allowed up to the speed that lands exactly on it,
    // which is what stops a limit from being overshot in one substep.
    #[test]
    fn a_limit_row_allows_only_the_approach_that_reaches_it() {
        let mut total = 0.0;
        let row = LimitRow {
            separation: 0.01,
            rate: -10.0,
            mass: 1.0,
        };
        // The bound is 0.01 away and the row closes at 10/s over a 1/240s
        // substep, so it would overshoot by more than half of it.
        let applied = solve_limit(row, &mut total, &push(true));
        assert!(applied > 0.0, "the approach has to be slowed: {applied}");
        assert!(close(-10.0 + applied, -0.01 * 240.0), "{applied}");
    }

    // The relax pass takes the correction's velocity back out without letting
    // the bound reopen.
    #[test]
    fn a_relax_pass_leaves_a_bound_that_is_already_still_alone() {
        let mut total = 4.0;
        let applied = solve_limit(
            LimitRow {
                separation: -0.1,
                rate: 0.0,
                mass: 1.0,
            },
            &mut total,
            &push(false),
        );
        assert_eq!((applied, total), (0.0, 4.0));
    }

    #[test]
    fn a_motor_row_drives_toward_its_target_and_stops_at_its_ceiling() {
        let mut total = 0.0;
        // Ten units of error against a budget of one.
        let applied = solve_motor(-10.0, 1.0, &mut total, 1.0);
        assert_eq!((applied, total), (1.0, 1.0));
        // Once saturated it applies nothing more.
        assert_eq!(solve_motor(-10.0, 1.0, &mut total, 1.0), 0.0);
        // And it drives the other way when the error reverses, down to the
        // ceiling on that side.
        let back = solve_motor(10.0, 1.0, &mut total, 1.0);
        assert_eq!((back, total), (-2.0, -1.0));
    }
}
