// Grounded character-body schema.

/// Gives a player [Camera3D](#camera3d) gravity, jumping, and a grounded
/// character body.
///
/// Every [Camera3D](#camera3d) already collides with the world as a capsule.
/// Adding a RigidBody upgrades that camera from a free-flying spectator to a
/// grounded character: it falls under gravity, lands on surfaces, climbs steps,
/// slides off steep slopes, and can jump. The capsule size is configured here
/// too.
///
/// ```json
/// ["RigidBody", { "jump_height": 1.4 }]
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct RigidBody {
    /// Multiplier applied to the global gravity constant. 1.0 = normal gravity.
    #[asset(default = 1.0)]
    pub gravity_scale: f32,
    /// Radius of the player capsule used for collision, in world units.
    #[asset(default = 0.3)]
    pub capsule_radius: f32,
    /// Total height of the player capsule. The camera eye sits at the top.
    #[asset(default = 1.7)]
    pub capsule_height: f32,
    /// Apex height of a jump in world units. 0 disables jumping.
    #[asset(default = 1.1)]
    pub jump_height: f32,
    /// Steepest slope the player can walk up, in degrees.
    #[asset(default = 50.0)]
    pub max_slope_deg: f32,
    /// Tallest obstacle the controller auto-steps over, in world units.
    #[asset(default = 0.3)]
    pub step_height: f32,
    /// True when the capsule is resting on a surface this frame.
    /// Written by PhysicsSystem.
    // Starting grounded keeps the first frame from playing a fall.
    #[serde(skip)]
    #[asset(default = true)]
    pub is_grounded: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ground_state_is_runtime_only_and_never_rides_the_wire() {
        let b: RigidBody = serde_json::from_str(r#"{"is_grounded":false}"#).unwrap();
        // The authored `is_grounded` is skipped, so it keeps its default.
        assert!(b.is_grounded);

        let airborne = RigidBody {
            is_grounded: false,
            ..Default::default()
        };
        let bytes = postcard::to_allocvec(&airborne).unwrap();
        let back: RigidBody = postcard::from_bytes(&bytes).unwrap();
        assert!(back.is_grounded);
    }
}
