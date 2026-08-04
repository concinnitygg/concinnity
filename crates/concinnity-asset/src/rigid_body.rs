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
/// { "name": "player_body", "type": "RigidBody", "args": { "jump_height": 1.4 } }
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RigidBody {
    /// Multiplier applied to the global gravity constant. 1.0 = normal gravity.
    pub gravity_scale: f32,
    /// Radius of the player capsule used for collision, in world units.
    pub capsule_radius: f32,
    /// Total height of the player capsule. The camera eye sits at the top.
    pub capsule_height: f32,
    /// Apex height of a jump in world units. 0 disables jumping.
    pub jump_height: f32,
    /// Steepest slope the player can walk up, in degrees.
    pub max_slope_deg: f32,
    /// Tallest obstacle the controller auto-steps over, in world units.
    pub step_height: f32,
    /// True when the capsule is resting on a surface this frame.
    /// Written by PhysicsSystem.
    #[serde(skip)]
    pub is_grounded: bool,
}

impl Default for RigidBody {
    fn default() -> Self {
        Self {
            gravity_scale: 1.0,
            capsule_radius: 0.3,
            capsule_height: 1.7,
            jump_height: 1.1,
            max_slope_deg: 50.0,
            step_height: 0.3,
            is_grounded: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_capsule_is_a_person_sized_walker() {
        let b = RigidBody::default();
        assert_eq!(b.capsule_radius, 0.3);
        assert_eq!(b.capsule_height, 1.7);
        assert_eq!(b.jump_height, 1.1);
        assert_eq!(b.max_slope_deg, 50.0);
        assert_eq!(b.step_height, 0.3);
        assert_eq!(b.gravity_scale, 1.0);
        // Starting grounded keeps the first frame from playing a fall.
        assert!(b.is_grounded);
    }

    #[test]
    fn ground_state_is_runtime_only_and_never_rides_the_wire() {
        let b: RigidBody = serde_json::from_str(
            r#"{"gravity_scale":2,"capsule_radius":0.4,"capsule_height":2,"jump_height":0,
                "max_slope_deg":35,"step_height":0.5,"is_grounded":false}"#,
        )
        .unwrap();
        // The authored `is_grounded` is skipped, so it keeps its default.
        assert!(b.is_grounded);
        assert_eq!(b.jump_height, 0.0);

        let bytes = postcard::to_allocvec(&b).unwrap();
        let back: RigidBody = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.gravity_scale, 2.0);
        assert_eq!(back.capsule_radius, 0.4);
        assert_eq!(back.capsule_height, 2.0);
        assert_eq!(back.max_slope_deg, 35.0);
        assert_eq!(back.step_height, 0.5);
        assert!(back.is_grounded);
    }
}
