// concinnity-physics/src/joints.rs
//
// The constraint shapes the simulation can be asked to build between two
// bodies.
// Angles are radians and velocities are per-second: the authored degrees are
// converted once, by whoever reads the asset.

/// Constraint shape connecting two bodies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointSpec {
    /// All six degrees of freedom locked: the bodies move and rotate as one
    /// rigid assembly relative to their anchors.
    Fixed,
    /// Hinge: rotation is allowed only around one axis.
    Revolute {
        /// Hinge axis, in each body's local frame.
        axis: [f32; 3],
        /// Clamps the hinge angle, in radians.
        limits: Option<[f32; 2]>,
        /// Drives the hinge at a target angular velocity.
        motor: Option<JointMotor>,
    },
    /// Ball-and-socket: translation locked, all three rotational axes free.
    Spherical,
    /// Slider: translation is allowed only along one axis.
    Prismatic {
        /// Slide axis, in each body's local frame.
        axis: [f32; 3],
        /// Clamps the slide distance, in world units.
        limits: Option<[f32; 2]>,
        /// Drives the slide at a target linear velocity.
        motor: Option<JointMotor>,
    },
}

/// Velocity-driven motor parameters for a revolute or prismatic joint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointMotor {
    /// Target velocity: radians/second for revolute, units/second for prismatic.
    pub target_velocity: f32,
    /// Maximum force the motor may apply to reach the target.
    pub max_force: f32,
}
