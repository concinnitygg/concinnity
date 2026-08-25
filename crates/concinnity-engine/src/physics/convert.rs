// src/physics/convert.rs
//
// Conversions from the engine's authored representation to the simulation
// vocabulary: the asset data types (PhysicsJoint, Prop, PropBody) turned into
// the shapes, joint specs and body parameters the simulation is built from.

use concinnity_core::assets::{BodyDynamics, PhysicsJoint, PhysicsJointKind, PropCollider};
use concinnity_physics::{ColliderShape, DynamicParams, JointMotor, JointSpec};

// The `JointSpec` a `PhysicsJoint` asset describes, converting authored degrees
// to the radians a revolute joint is specified in.
pub(crate) fn joint_spec(joint: &PhysicsJoint) -> JointSpec {
    let limits = if joint.limits_enabled {
        Some(joint.limits)
    } else {
        None
    };
    let motor = if joint.motor_max_force > 0.0 {
        Some(JointMotor {
            target_velocity: joint.motor_target_velocity,
            max_force: joint.motor_max_force,
        })
    } else {
        None
    };
    match joint.parsed_kind() {
        PhysicsJointKind::Fixed => JointSpec::Fixed,
        PhysicsJointKind::Spherical => JointSpec::Spherical,
        PhysicsJointKind::Revolute => JointSpec::Revolute {
            axis: joint.axis,
            limits: limits.map(|[a, b]| [a.to_radians(), b.to_radians()]),
            motor: motor.map(|m| JointMotor {
                target_velocity: m.target_velocity.to_radians(),
                max_force: m.max_force,
            }),
        },
        PhysicsJointKind::Prismatic => JointSpec::Prismatic {
            axis: joint.axis,
            limits,
            motor,
        },
    }
}

// The collision shape for a `PropCollider`, baking in the prop's `scale` (the
// simulation has no separate scale concept).
pub(crate) fn collider_shape(collider: &PropCollider, scale: [f32; 3]) -> ColliderShape {
    let [sx, sy, sz] = [scale[0].abs(), scale[1].abs(), scale[2].abs()];
    match collider.shape.as_str() {
        "ball" | "sphere" => ColliderShape::Ball {
            radius: collider.radius * sx,
        },
        "capsule" => ColliderShape::Capsule {
            half_height: collider.half_height * sy,
            radius: collider.radius * sx,
        },
        // "aabb", "cuboid", and anything unrecognised fall back to a box.
        _ => ColliderShape::Cuboid {
            half_extents: [
                collider.half_extents[0] * sx,
                collider.half_extents[1] * sy,
                collider.half_extents[2] * sz,
            ],
        },
    }
}

// The dynamic-body parameters a `BodyDynamics` component describes.
pub(crate) fn dynamic_params(body: &BodyDynamics) -> DynamicParams {
    DynamicParams {
        mass: body.mass.max(0.0),
        friction: body.friction.max(0.0),
        restitution: body.restitution.clamp(0.0, 1.0),
        gravity_scale: body.gravity_scale,
        linear_damping: body.linear_damping.max(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_spec_converts_revolute_units_to_radians() {
        let j = PhysicsJoint {
            kind: "revolute".to_string(),
            axis: [0.0, 0.0, 1.0],
            limits_enabled: true,
            limits: [-90.0, 90.0],
            motor_target_velocity: 180.0,
            motor_max_force: 5.0,
            ..Default::default()
        };
        match joint_spec(&j) {
            JointSpec::Revolute {
                axis,
                limits,
                motor,
            } => {
                assert_eq!(axis, [0.0, 0.0, 1.0]);
                let lim = limits.expect("limits set");
                assert!((lim[0] - (-std::f32::consts::FRAC_PI_2)).abs() < 1.0e-5);
                assert!((lim[1] - std::f32::consts::FRAC_PI_2).abs() < 1.0e-5);
                let m = motor.expect("motor set");
                assert!((m.target_velocity - std::f32::consts::PI).abs() < 1.0e-5);
                assert_eq!(m.max_force, 5.0);
            }
            other => panic!("expected Revolute, got {other:?}"),
        }
    }

    #[test]
    fn joint_spec_prismatic_keeps_units() {
        let j = PhysicsJoint {
            kind: "prismatic".to_string(),
            axis: [1.0, 0.0, 0.0],
            limits_enabled: true,
            limits: [-0.5, 0.5],
            ..Default::default()
        };
        match joint_spec(&j) {
            JointSpec::Prismatic {
                axis,
                limits,
                motor,
            } => {
                assert_eq!(axis, [1.0, 0.0, 0.0]);
                assert_eq!(limits, Some([-0.5, 0.5]));
                assert!(motor.is_none());
            }
            other => panic!("expected Prismatic, got {other:?}"),
        }
    }

    #[test]
    fn joint_motor_inactive_when_max_force_zero() {
        let j = PhysicsJoint {
            kind: "revolute".to_string(),
            motor_target_velocity: 30.0,
            motor_max_force: 0.0,
            ..Default::default()
        };
        match joint_spec(&j) {
            JointSpec::Revolute { motor, .. } => assert!(motor.is_none()),
            _ => unreachable!(),
        }
    }
}
