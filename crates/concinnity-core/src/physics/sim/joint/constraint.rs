// The joint rows, prepared once a step and solved every substep beside the
// contacts.
//
// The shape of it follows the contact solver deliberately: the errors are
// re-measured against where the bodies now are on every substep, and each
// impulse is split between correcting now and being remembered. A joint solved
// any other way would be stiffer or softer than the contacts it shares bodies
// with, and the two would trade the difference back and forth as jitter.
//
// Where it parts from the contact solver is the effective-mass blocks. A
// contact keeps the ones it was prepared with and rotates its anchors; a joint
// rebuilds them every time a pose moves. A contact's block is read across a
// normal that barely turns, while a joint's is read along a lever, and the
// mass along a lever and the mass across it differ by whatever the arm is
// worth -- so a block that is a substep out of date does not slightly mistune
// a joint, it makes the solve diverge.
//
// Every kind runs through the same four row groups -- motor, bounds, angular,
// linear -- and the kind only decides which of them exist and how many
// directions each covers. That is why there is one solve here rather than one
// per joint kind.

use alloc::vec::Vec;

use crate::physics::JointMotor;
use crate::physics::sim::math::{Mat3, Quat, Vec3};
use crate::physics::sim::solver::{Bodies, SolverBody};

use super::rows::{self, Arm, LimitRow, Push};
use super::{Joint, JointFrame, JointImpulses, JointKind};

/// Ceiling on the speed a joint's positional error is corrected at, matching
/// the one the contact solver pushes penetration out at.
const MAX_CORRECTION: f32 = 3.0;

/// One joint the step is solving, and what it is carrying.
pub(crate) struct Prepared {
    joint: u32,
    a: u32,
    b: u32,
    frame: JointFrame,
    anchor_a: Vec3,
    anchor_b: Vec3,
    /// Where the bodies are and what that makes the rows weigh. Both follow
    /// from the two poses alone, so both are rebuilt wherever a pose moves.
    place: Placement,
    masses: Masses,
    impulses: JointImpulses,
}

/// The masses a joint's rows are solved through.
#[derive(Clone, Copy)]
struct Masses {
    /// Mass block for the rows holding the anchors together.
    linear: Mat3,
    /// Mass block for the rows holding the orientations together.
    angular: Mat3,
    /// Mass for the single row the joint's own axis carries.
    axis: f32,
}

/// Where a joint's two bodies are, gathered once so the four row groups do not
/// each re-derive it.
#[derive(Clone, Copy)]
struct Placement {
    /// Offset from the second body's centre to its anchor.
    lever_b: Vec3,
    /// Offset the linear rows act through, which is the anchor for a joint
    /// that holds a point and the whole separation for one that slides.
    drive_a: Vec3,
    /// Anchor separation: zero when the joint is holding.
    error: Vec3,
    /// World joint axis, taken from the first body.
    axis: Vec3,
}

/// The joint half of a step.
pub(crate) struct JointSolver {
    prepared: Vec<Prepared>,
}

impl JointSolver {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        JointSolver {
            prepared: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn reserved_bytes(&self) -> u64 {
        (self.prepared.capacity() * size_of::<Prepared>()) as u64
    }

    /// Build this step's rows, one per joint the partition kept, in the order
    /// it grouped them.
    ///
    /// A joint neither of whose bodies the step can move never reaches here:
    /// the partition owns that filter, the same way it does for contacts.
    pub(crate) fn prepare(&mut self, source: &[u32], joints: &[Joint], bodies: &[SolverBody]) {
        self.prepared.clear();
        for &index in source {
            let joint = &joints[index as usize];
            let (body_a, body_b) = (&bodies[joint.a as usize], &bodies[joint.b as usize]);
            let place = placement(
                &joint.frame,
                (joint.anchor_a, joint.anchor_b),
                body_a,
                body_b,
            );
            self.prepared.push(Prepared {
                joint: index,
                a: joint.a,
                b: joint.b,
                frame: joint.frame,
                anchor_a: joint.anchor_a,
                anchor_b: joint.anchor_b,
                place,
                masses: masses_of(&joint.frame, &place, body_a, body_b),
                impulses: joint.impulses,
            });
        }
    }

    /// The prepared rows, for a caller splitting them into runs of work.
    pub(crate) fn rows_mut(&mut self) -> &mut [Prepared] {
        &mut self.prepared
    }

    /// Re-read every joint against the poses its bodies now hold.
    ///
    /// Called wherever a pose moves and nowhere else, which is once a substep.
    /// The masses are the reason it has to happen at all: a lever that has
    /// turned since is a different mass, and an impulse solved through the old
    /// one overshoots the row it was aimed at instead of cancelling it. A few
    /// degrees of turn per substep is enough for that overshoot to grow rather
    /// than settle, which is a joint given a speed rather than released from
    /// rest.
    pub(crate) fn refresh(rows: &mut [Prepared], bodies: &Bodies<'_>) {
        for prepared in rows.iter_mut() {
            let (body_a, body_b) = (bodies.get(prepared.a), bodies.get(prepared.b));
            let place = placement(
                &prepared.frame,
                (prepared.anchor_a, prepared.anchor_b),
                body_a,
                body_b,
            );
            prepared.place = place;
            prepared.masses = masses_of(&prepared.frame, &place, body_a, body_b);
        }
    }

    /// Re-apply what each joint was holding at the end of the last substep, so
    /// the solve starts from a joint that is already carrying its load.
    pub(crate) fn warm_start(rows: &[Prepared], bodies: &mut Bodies<'_>) {
        for prepared in rows {
            let place = prepared.place;
            let carried = prepared.impulses;
            let along_axis = carried.lower - carried.upper + carried.motor;
            let (linear, angular) = if prepared.frame.kind.is_linear_axis() {
                (carried.linear + place.axis * along_axis, carried.angular)
            } else {
                (carried.linear, carried.angular + place.axis * along_axis)
            };
            apply(bodies, prepared, &place, linear, angular);
        }
    }

    /// Advance every joint's rows by one pass, in joint order.
    pub(crate) fn solve(rows: &mut [Prepared], bodies: &mut Bodies<'_>, push: &Push, h: f32) {
        for prepared in rows.iter_mut() {
            let (place, masses) = (prepared.place, prepared.masses);
            solve_axis(prepared, &place, &masses, bodies, push, h);
            solve_angular(prepared, &place, &masses, bodies, push);
            solve_linear(prepared, &place, &masses, bodies, push);
        }
    }

    /// Hand each joint back what it ended the step holding.
    pub(crate) fn store(&self, joints: &mut [Joint]) {
        for prepared in &self.prepared {
            joints[prepared.joint as usize].impulses = prepared.impulses;
        }
    }
}

/// Where a joint's bodies are, from the poses they hold right now.
fn placement(
    frame: &JointFrame,
    anchors: (Vec3, Vec3),
    body_a: &SolverBody,
    body_b: &SolverBody,
) -> Placement {
    let lever_a = body_a.rotation.rotate(anchors.0);
    let lever_b = body_b.rotation.rotate(anchors.1);
    let error = (body_b.position + lever_b) - (body_a.position + lever_a);
    Placement {
        lever_b,
        // A slider's anchors are meant to be apart along the axis, so the row
        // acting on the first body has to reach the second body's anchor
        // rather than its own.
        drive_a: if frame.kind.is_linear_axis() {
            lever_a + error
        } else {
            lever_a
        },
        error,
        axis: body_a.rotation.rotate(frame.axis_a),
    }
}

/// The masses a joint's rows are solved through, from where its bodies are
/// right now. Only [`JointSolver::refresh`] and the first read in `prepare`
/// call this, which is what keeps them matched to a pose.
fn masses_of(
    frame: &JointFrame,
    place: &Placement,
    body_a: &SolverBody,
    body_b: &SolverBody,
) -> Masses {
    let linear = rows::point_block(
        Arm {
            inv_mass: body_a.inv_mass,
            inv_inertia: body_a.inv_inertia,
            lever: place.drive_a,
        },
        Arm {
            inv_mass: body_b.inv_mass,
            inv_inertia: body_b.inv_inertia,
            lever: place.lever_b,
        },
    );
    let angular = rows::angular_block(body_a.inv_inertia, body_b.inv_inertia);
    // Only a bounded or driven joint reads a mass along its own axis.
    let axis = if frame.limits.is_none() && frame.motor.is_none() {
        0.0
    } else if frame.kind.is_linear_axis() {
        rows::axis_mass(&linear, place.axis)
    } else {
        rows::axis_mass(&angular, place.axis)
    };
    Masses {
        linear,
        angular,
        axis,
    }
}

/// Apply one linear and one angular impulse across a joint's two bodies.
fn apply(
    bodies: &mut Bodies<'_>,
    prepared: &Prepared,
    place: &Placement,
    linear: Vec3,
    angular: Vec3,
) {
    let (a, b) = (prepared.a, prepared.b);
    if linear != Vec3::ZERO {
        bodies.get_mut(a).apply_impulse(-linear, place.drive_a);
        bodies.get_mut(b).apply_impulse(linear, place.lever_b);
    }
    if angular != Vec3::ZERO {
        bodies.get_mut(a).apply_angular_impulse(-angular);
        bodies.get_mut(b).apply_angular_impulse(angular);
    }
}

/// Rate the joint's own coordinate is changing at: the slide speed along the
/// axis for a slider, the turn rate about it otherwise.
fn axis_rate(prepared: &Prepared, place: &Placement, bodies: &Bodies<'_>) -> f32 {
    let (a, b) = (prepared.a, prepared.b);
    if prepared.frame.kind.is_linear_axis() {
        let relative =
            bodies.get(b).velocity_at(place.lever_b) - bodies.get(a).velocity_at(place.drive_a);
        relative.dot(place.axis)
    } else {
        (bodies.get(b).angular_velocity - bodies.get(a).angular_velocity).dot(place.axis)
    }
}

/// Push an impulse along the joint's axis, whichever kind of axis it is.
fn apply_along_axis(
    bodies: &mut Bodies<'_>,
    prepared: &Prepared,
    place: &Placement,
    magnitude: f32,
) {
    let impulse = place.axis * magnitude;
    if prepared.frame.kind.is_linear_axis() {
        apply(bodies, prepared, place, impulse, Vec3::ZERO);
    } else {
        apply(bodies, prepared, place, Vec3::ZERO, impulse);
    }
}

/// The motor and the two bounds, all of which act along the joint's own axis.
fn solve_axis(
    prepared: &mut Prepared,
    place: &Placement,
    masses: &Masses,
    bodies: &mut Bodies<'_>,
    push: &Push,
    h: f32,
) {
    if prepared.frame.limits.is_none() && prepared.frame.motor.is_none() {
        return;
    }
    let mass = masses.axis;

    if let Some(JointMotor {
        target_velocity,
        max_force,
    }) = prepared.frame.motor
    {
        let error = axis_rate(prepared, place, bodies) - target_velocity;
        let applied = rows::solve_motor(error, mass, &mut prepared.impulses.motor, max_force * h);
        apply_along_axis(bodies, prepared, place, applied);
    }

    let Some([low, high]) = prepared.frame.limits else {
        return;
    };
    let coordinate = axis_coordinate(prepared, place, bodies);

    let applied = rows::solve_limit(
        LimitRow {
            separation: coordinate - low,
            rate: axis_rate(prepared, place, bodies),
            mass,
        },
        &mut prepared.impulses.lower,
        push,
    );
    apply_along_axis(bodies, prepared, place, applied);

    let applied = rows::solve_limit(
        LimitRow {
            separation: high - coordinate,
            rate: -axis_rate(prepared, place, bodies),
            mass,
        },
        &mut prepared.impulses.upper,
        push,
    );
    apply_along_axis(bodies, prepared, place, -applied);
}

/// Where the joint sits along its own axis: radians turned for a hinge,
/// world units slid for a slider.
fn axis_coordinate(prepared: &Prepared, place: &Placement, bodies: &Bodies<'_>) -> f32 {
    if prepared.frame.kind.is_linear_axis() {
        place.error.dot(place.axis)
    } else {
        prepared.frame.hinge_angle(
            bodies.get(prepared.a).rotation,
            bodies.get(prepared.b).rotation,
        )
    }
}

/// The rows holding the two orientations together: all three for a rigid
/// assembly or a slider, the two across the hinge for a revolute, none for a
/// ball and socket.
fn solve_angular(
    prepared: &mut Prepared,
    place: &Placement,
    masses: &Masses,
    bodies: &mut Bodies<'_>,
    push: &Push,
) {
    let (a, b) = (prepared.a, prepared.b);
    let rate = bodies.get(b).angular_velocity - bodies.get(a).angular_velocity;
    let (mass_scale, impulse_scale) = scales(push);

    let delta = match prepared.frame.kind {
        JointKind::Spherical => return,
        JointKind::Fixed | JointKind::Prismatic => {
            let error = twist_error(
                &prepared.frame,
                bodies.get(a).rotation,
                bodies.get(b).rotation,
            );
            let biased = (rate + bias(error, push)) * mass_scale;
            rows::solve_block(&masses.angular, biased) - prepared.impulses.angular * impulse_scale
        }
        JointKind::Revolute => {
            let hinge_b = bodies.get(b).rotation.rotate(prepared.frame.axis_b);
            // The turn that carries the first body's hinge onto the second's:
            // perpendicular to both, so it never fights the free axis.
            let error = place.axis.cross(hinge_b);
            let biased = (rate + bias(error, push)) * mass_scale;
            let (t1, t2) = perpendiculars(place.axis);
            rows::solve_plane(&masses.angular, t1, t2, [biased.dot(t1), biased.dot(t2)])
                - prepared.impulses.angular * impulse_scale
        }
    };
    prepared.impulses.angular += delta;
    apply(bodies, prepared, place, Vec3::ZERO, delta);
}

/// The rows holding the two anchors together: all three unless the joint
/// slides, which leaves the axis free.
fn solve_linear(
    prepared: &mut Prepared,
    place: &Placement,
    masses: &Masses,
    bodies: &mut Bodies<'_>,
    push: &Push,
) {
    let (a, b) = (prepared.a, prepared.b);
    let rate = bodies.get(b).velocity_at(place.lever_b) - bodies.get(a).velocity_at(place.drive_a);
    let (mass_scale, impulse_scale) = scales(push);

    let delta = if prepared.frame.kind.is_linear_axis() {
        let biased = (rate + bias(across_axis(place.error, place.axis), push)) * mass_scale;
        let (t1, t2) = perpendiculars(place.axis);
        rows::solve_plane(&masses.linear, t1, t2, [biased.dot(t1), biased.dot(t2)])
            - prepared.impulses.linear * impulse_scale
    } else {
        let biased = (rate + bias(place.error, push)) * mass_scale;
        rows::solve_block(&masses.linear, biased) - prepared.impulses.linear * impulse_scale
    };
    prepared.impulses.linear += delta;
    apply(bodies, prepared, place, delta, Vec3::ZERO);
}

/// How much of an impulse corrects the error now, and how much of it is
/// remembered instead. The relax pass corrects none of it.
fn scales(push: &Push) -> (f32, f32) {
    if push.use_bias {
        (push.soft.mass_scale, push.soft.impulse_scale)
    } else {
        (1.0, 0.0)
    }
}

/// The velocity an error is corrected at, capped so a joint built badly out of
/// place eases together rather than launching.
fn bias(error: Vec3, push: &Push) -> Vec3 {
    if !push.use_bias {
        return Vec3::ZERO;
    }
    let wanted = error * push.soft.bias_rate;
    let speed = wanted.length();
    if speed > push.max_push.min(MAX_CORRECTION) {
        wanted * (push.max_push.min(MAX_CORRECTION) / speed)
    } else {
        wanted
    }
}

/// The part of an offset that is not along the axis, which is the part a
/// slider still has to hold at zero.
fn across_axis(offset: Vec3, axis: Vec3) -> Vec3 {
    offset - axis * offset.dot(axis)
}

/// Two unit directions spanning the plane across `axis`.
fn perpendiculars(axis: Vec3) -> (Vec3, Vec3) {
    let t1 = axis.any_perpendicular();
    (t1, axis.cross(t1))
}

/// The world-space rotation carrying the second body from where the joint
/// holds it to where it actually is.
fn twist_error(frame: &JointFrame, rotation_a: Quat, rotation_b: Quat) -> Vec3 {
    let held = rotation_a.mul(frame.rest);
    let delta = rotation_b.mul(held.conjugate());
    // A quaternion and its negation name the same rotation; the half with a
    // non-negative scalar part is the short way round.
    let sign = if delta.w < 0.0 { -2.0 } else { 2.0 };
    Vec3::from_array([delta.x, delta.y, delta.z]) * sign
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::sim::config::Softness;
    use crate::physics::sim::math::vec3;

    fn identity_error() -> Vec3 {
        twist_error(
            &JointFrame::new(
                crate::physics::JointSpec::Fixed,
                Quat::IDENTITY,
                Quat::IDENTITY,
            ),
            Quat::IDENTITY,
            Quat::IDENTITY,
        )
    }

    #[test]
    fn a_joint_at_the_pose_it_was_made_at_has_no_twist_to_correct() {
        assert!(identity_error().length() < 1.0e-6);
        let a = Quat::from_euler_deg([12.0, -35.0, 60.0]);
        let b = Quat::from_euler_deg([5.0, 20.0, -10.0]);
        let frame = JointFrame::new(crate::physics::JointSpec::Fixed, a, b);
        assert!(twist_error(&frame, a, b).length() < 1.0e-5);
    }

    // The error has to point along the turn that puts the second body back,
    // and grow with it. It is `2 sin(angle/2)` rather than the angle itself,
    // which is the same thing where a joint is meant to live and never has the
    // wrong sign anywhere else.
    #[test]
    fn the_twist_error_is_the_turn_that_puts_the_body_back() {
        let frame = JointFrame::new(
            crate::physics::JointSpec::Fixed,
            Quat::IDENTITY,
            Quat::IDENTITY,
        );
        for degrees in [-90.0f32, -30.0, -5.0, 5.0, 30.0, 90.0] {
            let radians = degrees.to_radians();
            let turned = Quat::from_axis_angle(Vec3::Z, radians);
            let error = twist_error(&frame, Quat::IDENTITY, turned);
            let expected = 2.0 * (radians * 0.5).sin();
            assert!((error.z - expected).abs() < 1.0e-5, "{degrees}: {error:?}");
            assert!(
                error.x.abs() < 1.0e-5 && error.y.abs() < 1.0e-5,
                "{error:?}"
            );
        }
        // Small angles are the angle itself, which is where a held joint sits.
        let small = 2.0f32.to_radians();
        let error = twist_error(
            &frame,
            Quat::IDENTITY,
            Quat::from_axis_angle(Vec3::Z, small),
        );
        assert!((error.z - small).abs() < 1.0e-4, "{error:?}");
    }

    // Past half a turn the short way round is the other way, or a fixed joint
    // would be driven the long way home.
    #[test]
    fn a_twist_past_half_a_turn_corrects_the_short_way() {
        let frame = JointFrame::new(
            crate::physics::JointSpec::Fixed,
            Quat::IDENTITY,
            Quat::IDENTITY,
        );
        let turned = Quat::from_axis_angle(Vec3::Z, 350.0f32.to_radians());
        assert!(twist_error(&frame, Quat::IDENTITY, turned).z < 0.0);
    }

    #[test]
    fn the_part_across_an_axis_drops_the_part_along_it() {
        let across = across_axis(vec3(1.0, 2.0, 3.0), Vec3::Y);
        assert!((across - vec3(1.0, 0.0, 3.0)).length() < 1.0e-6);
        let (t1, t2) = perpendiculars(Vec3::Y);
        assert!(t1.dot(Vec3::Y).abs() < 1.0e-6 && t2.dot(Vec3::Y).abs() < 1.0e-6);
        assert!((t1.cross(t2) - Vec3::Y).length() < 1.0e-5);
    }

    fn push(use_bias: bool) -> Push {
        Push {
            soft: Softness::new(60.0, 5.0, 1.0 / 240.0),
            inv_h: 240.0,
            max_push: 3.0,
            use_bias,
        }
    }

    // A joint built badly out of place must ease together rather than being
    // thrown at the speed the error alone would ask for.
    #[test]
    fn a_large_error_is_corrected_at_a_capped_speed() {
        let far = bias(vec3(0.0, -50.0, 0.0), &push(true));
        assert!(far.length() <= MAX_CORRECTION + 1.0e-4, "{far:?}");
        assert!(far.y < 0.0, "and still in the right direction: {far:?}");
        let near = bias(vec3(0.0, -0.001, 0.0), &push(true));
        assert!(near.length() < MAX_CORRECTION, "{near:?}");
        assert_eq!(bias(vec3(0.0, -50.0, 0.0), &push(false)), Vec3::ZERO);
    }

    #[test]
    fn the_relax_pass_corrects_nothing_and_remembers_nothing() {
        assert_eq!(scales(&push(false)), (1.0, 0.0));
        let (mass_scale, impulse_scale) = scales(&push(true));
        assert!((mass_scale + impulse_scale - 1.0).abs() < 1.0e-5);
    }
}
