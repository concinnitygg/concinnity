// The authored joint specification, reduced once to the frame the solver
// works in.
//
// A joint is authored as an axis, a pair of limits, and a motor, none of which
// the solver can use as given: an axis may be zero-length, limits may arrive
// the wrong way round, and a motor with no force ceiling drives nothing. All
// of that is settled here, at construction, so the substep loop never branches
// on degenerate input.
//
// The rest rotation is the other half of it. A joint holds the bodies at the
// relative orientation they had when it was made, so what "zero" means for a
// hinge angle, and what a fixed joint holds, is captured once rather than
// re-derived from an authored pose the simulation never sees.

use crate::math::atan2;
use crate::physics::sim::math::{Quat, Vec3};
use crate::physics::{JointMotor, JointSpec};

/// Which degrees of freedom a joint leaves open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JointKind {
    /// Nothing moves: three linear rows and three angular ones.
    Fixed,
    /// Rotation about one axis: three linear rows and two angular ones.
    Revolute,
    /// Rotation about any axis: three linear rows and none angular.
    Spherical,
    /// Translation along one axis: two linear rows and three angular ones.
    Prismatic,
}

impl JointKind {
    /// Whether the joint measures a limit and drives a motor along its axis
    /// rather than about it.
    pub(crate) fn is_linear_axis(self) -> bool {
        self == JointKind::Prismatic
    }
}

/// One joint's specification, in the form the solver reads it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct JointFrame {
    pub(crate) kind: JointKind,
    /// Unit joint axis in each body's own frame. Both are the authored axis;
    /// keeping them apart is what lets the two bodies be turned relative to
    /// each other and still name the same hinge.
    pub(crate) axis_a: Vec3,
    pub(crate) axis_b: Vec3,
    /// Relative orientation the bodies had when the joint was made. A fixed
    /// joint holds it, and a hinge measures its angle from it.
    pub(crate) rest: Quat,
    /// Ordered low-to-high, in radians for a hinge and world units for a
    /// slider.
    pub(crate) limits: Option<[f32; 2]>,
    pub(crate) motor: Option<JointMotor>,
}

impl JointFrame {
    /// Reduce an authored spec against the poses the two bodies hold now.
    pub(crate) fn new(spec: JointSpec, rotation_a: Quat, rotation_b: Quat) -> Self {
        let (kind, axis, limits, motor) = match spec {
            JointSpec::Fixed => (JointKind::Fixed, [0.0, 1.0, 0.0], None, None),
            JointSpec::Spherical => (JointKind::Spherical, [0.0, 1.0, 0.0], None, None),
            JointSpec::Revolute {
                axis,
                limits,
                motor,
            } => (JointKind::Revolute, axis, limits, motor),
            JointSpec::Prismatic {
                axis,
                limits,
                motor,
            } => (JointKind::Prismatic, axis, limits, motor),
        };
        let axis = normalize_axis(axis);
        JointFrame {
            kind,
            axis_a: axis,
            axis_b: axis,
            rest: rotation_a.conjugate().mul(rotation_b).normalize(),
            limits: order_limits(limits),
            motor: usable_motor(motor),
        }
    }

    /// Whether a motor is actively driving this joint, which is what stops the
    /// island it belongs to from settling.
    pub(crate) fn is_driven(&self) -> bool {
        self.motor
            .is_some_and(|m| m.target_velocity != 0.0 && m.max_force > 0.0)
    }

    /// How far the joint has turned about its hinge from the rest pose, in
    /// radians in `[-pi, pi]`.
    pub(crate) fn hinge_angle(&self, rotation_a: Quat, rotation_b: Quat) -> f32 {
        // What is left of b's orientation once a's and the rest pose are taken
        // out: for a hinge holding, a rotation about the axis and nothing else.
        let delta = self
            .rest
            .conjugate()
            .mul(rotation_a.conjugate().mul(rotation_b));
        // A quaternion and its negation name the same rotation, so the half
        // with a non-negative scalar part is the one whose angle is in range.
        let (sin_part, cos_part) = if delta.w < 0.0 {
            (
                -Vec3::from_array([delta.x, delta.y, delta.z]).dot(self.axis_b),
                -delta.w,
            )
        } else {
            (
                Vec3::from_array([delta.x, delta.y, delta.z]).dot(self.axis_b),
                delta.w,
            )
        };
        2.0 * atan2(sin_part, cos_part)
    }
}

/// A unit axis, or `+Y` when the authored one names no direction.
pub(crate) fn normalize_axis(axis: [f32; 3]) -> Vec3 {
    let v = Vec3::from_array(axis);
    if !v.is_finite() {
        return Vec3::Y;
    }
    let length = v.length();
    if length > 1.0e-6 {
        v * (1.0 / length)
    } else {
        Vec3::Y
    }
}

/// Limits low-to-high, or none when they name no range the solver can hold.
fn order_limits(limits: Option<[f32; 2]>) -> Option<[f32; 2]> {
    let [low, high] = limits?;
    if !low.is_finite() || !high.is_finite() {
        return None;
    }
    Some(if low <= high {
        [low, high]
    } else {
        [high, low]
    })
}

/// A motor, or none when it has no force to drive with.
fn usable_motor(motor: Option<JointMotor>) -> Option<JointMotor> {
    let m = motor?;
    if !m.target_velocity.is_finite() || !m.max_force.is_finite() || m.max_force <= 0.0 {
        return None;
    }
    Some(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::sim::math::vec3;

    const HINGE: [f32; 3] = [0.0, 0.0, 1.0];

    fn revolute(limits: Option<[f32; 2]>, motor: Option<JointMotor>) -> JointSpec {
        JointSpec::Revolute {
            axis: HINGE,
            limits,
            motor,
        }
    }

    #[test]
    fn each_spec_selects_the_rows_its_kind_leaves_open() {
        let at_rest = |spec| JointFrame::new(spec, Quat::IDENTITY, Quat::IDENTITY).kind;
        assert_eq!(at_rest(JointSpec::Fixed), JointKind::Fixed);
        assert_eq!(at_rest(JointSpec::Spherical), JointKind::Spherical);
        assert_eq!(at_rest(revolute(None, None)), JointKind::Revolute);
        assert_eq!(
            at_rest(JointSpec::Prismatic {
                axis: HINGE,
                limits: None,
                motor: None,
            }),
            JointKind::Prismatic
        );
        assert!(JointKind::Prismatic.is_linear_axis());
        assert!(!JointKind::Revolute.is_linear_axis());
    }

    // The degenerate input a caller is allowed to hand over: none of it may
    // reach the solver as it arrived.
    #[test]
    fn a_zero_length_axis_falls_back_to_up() {
        assert_eq!(normalize_axis([0.0, 0.0, 0.0]), Vec3::Y);
        assert_eq!(normalize_axis([f32::NAN, 1.0, 0.0]), Vec3::Y);
        assert_eq!(normalize_axis([1.0e-9, 0.0, 0.0]), Vec3::Y);
        let n = normalize_axis([3.0, 0.0, 4.0]);
        assert!((n.x - 0.6).abs() < 1.0e-6 && (n.z - 0.8).abs() < 1.0e-6);
        assert!((normalize_axis([0.0, -2.0, 0.0]) - vec3(0.0, -1.0, 0.0)).length() < 1.0e-6);
    }

    #[test]
    fn limits_the_wrong_way_round_are_swapped_and_unusable_ones_dropped() {
        assert_eq!(order_limits(Some([0.5, -0.5])), Some([-0.5, 0.5]));
        assert_eq!(order_limits(Some([-0.5, 0.5])), Some([-0.5, 0.5]));
        assert_eq!(order_limits(Some([0.25, 0.25])), Some([0.25, 0.25]));
        assert_eq!(order_limits(Some([f32::NAN, 1.0])), None);
        assert_eq!(order_limits(None), None);
    }

    #[test]
    fn a_motor_with_no_force_ceiling_drives_nothing() {
        let driving = JointMotor {
            target_velocity: 2.0,
            max_force: 10.0,
        };
        assert_eq!(usable_motor(Some(driving)), Some(driving));
        assert_eq!(
            usable_motor(Some(JointMotor {
                target_velocity: 2.0,
                max_force: 0.0,
            })),
            None
        );
        assert_eq!(
            usable_motor(Some(JointMotor {
                target_velocity: f32::NAN,
                max_force: 1.0,
            })),
            None
        );
    }

    // A motor with a ceiling but no target holds the joint still rather than
    // driving it, so it must not keep an island awake.
    #[test]
    fn only_a_motor_with_somewhere_to_go_counts_as_driving() {
        let frame = |motor| JointFrame::new(revolute(None, motor), Quat::IDENTITY, Quat::IDENTITY);
        assert!(!frame(None).is_driven());
        assert!(
            !frame(Some(JointMotor {
                target_velocity: 0.0,
                max_force: 5.0,
            }))
            .is_driven()
        );
        assert!(
            frame(Some(JointMotor {
                target_velocity: 3.0,
                max_force: 5.0,
            }))
            .is_driven()
        );
    }

    // The angle is measured from the pose the joint was made at, so a joint
    // built on two turned bodies still reads zero.
    #[test]
    fn a_hinge_reads_zero_at_the_pose_it_was_made_at() {
        let a = Quat::from_euler_deg([10.0, 25.0, -40.0]);
        let b = Quat::from_euler_deg([0.0, 0.0, 30.0]);
        let frame = JointFrame::new(revolute(None, None), a, b);
        assert!(frame.hinge_angle(a, b).abs() < 1.0e-5);
    }

    #[test]
    fn a_hinge_measures_the_turn_it_has_taken_about_its_axis() {
        let frame = JointFrame::new(revolute(None, None), Quat::IDENTITY, Quat::IDENTITY);
        for degrees in [-170.0f32, -90.0, -1.0, 0.0, 45.0, 90.0, 179.0] {
            let radians = degrees.to_radians();
            let turned = Quat::from_axis_angle(Vec3::Z, radians);
            let measured = frame.hinge_angle(Quat::IDENTITY, turned);
            assert!(
                (measured - radians).abs() < 1.0e-4,
                "{degrees} degrees read as {measured}"
            );
        }
    }

    // Past half a turn the angle wraps rather than running away, which is what
    // keeps a limit from being pushed the long way round.
    #[test]
    fn a_hinge_angle_stays_inside_half_a_turn_either_way() {
        let frame = JointFrame::new(revolute(None, None), Quat::IDENTITY, Quat::IDENTITY);
        for degrees in [181.0f32, 270.0, 359.0] {
            let turned = Quat::from_axis_angle(Vec3::Z, degrees.to_radians());
            let measured = frame.hinge_angle(Quat::IDENTITY, turned);
            assert!(
                measured.abs() <= core::f32::consts::PI + 1.0e-4,
                "{degrees} degrees read as {measured}"
            );
            assert!(measured < 0.0, "{degrees} degrees must read as negative");
        }
    }
}
