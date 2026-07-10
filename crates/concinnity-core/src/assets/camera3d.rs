// src/assets/camera3d.rs
//
// Runtime 3D camera component. Its authored args and controller config live in
// the schema crate (concinnity_asset::camera3d).

use crate::assets::{Camera3DArgs, CameraController};
use crate::ecs::{AssetOrigin, Component};

/// Declares the 3D camera. One per scene.
///
/// ```jsonl
/// {
///   "name": "main_camera",
///   "type": "Camera3D",
///   "args": {
///     "fov_y_degrees": 80.0,
///     "near": 0.05,
///     "far": 500.0,
///     "position": [0.0, 4.0, 0.0]
///   }
/// }
/// ```
#[derive(Debug)]
pub struct Camera3D {
    pub fov_y_degrees: f32,
    pub near: f32,
    pub far: f32,
    /// Current view matrix, written each step by the active camera system.
    /// Column-major, matching the GLSL mat4 convention.
    pub view_matrix: [[f32; 4]; 4],
    /// Current world-space eye position, kept in sync with view_matrix.
    pub position: [f32; 3],
    /// Current yaw in radians.
    pub yaw: f32,
    /// Current pitch in radians.
    pub pitch: f32,
    /// World-space horizontal movement intent (units/second). Written by
    /// Camera3DSystem each frame, consumed by PhysicsSystem. Runtime-only.
    pub desired_move: [f32; 3],
    /// Set for one frame when the jump key is pressed. Runtime-only.
    pub jump_requested: bool,
    /// Set for one frame when the interact key is pressed. Runtime-only.
    pub interact_requested: bool,
    /// Controller settings, or `None` for an uncontrolled (cutscene) camera.
    /// Read once by the internal camera controller at init.
    pub controller: Option<CameraController>,
}

impl Component for Camera3D {
    const NAME: &'static str = "Camera3D";

    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Camera3DArgs;

    fn to_args(&self) -> Camera3DArgs {
        Camera3DArgs {
            fov_y_degrees: self.fov_y_degrees,
            near: self.near,
            far: self.far,
            position: self.position,
            yaw: self.yaw,
            pitch: self.pitch,
            controller: self.controller.clone(),
        }
    }

    fn from_args(args: Camera3DArgs) -> Self {
        Self {
            fov_y_degrees: args.fov_y_degrees,
            near: args.near,
            far: args.far,
            view_matrix: crate::gfx::camera::view_matrix(args.position, args.yaw, args.pitch),
            position: args.position,
            yaw: args.yaw,
            pitch: args.pitch,
            desired_move: [0.0; 3],
            jump_requested: false,
            interact_requested: false,
            controller: args.controller,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::FollowDrive;
    use crate::ecs::asset_id::AssetId;

    #[test]
    fn follow_block_deserializes_names_and_defaults() {
        crate::ecs::asset_id::reset_interner();
        crate::ecs::asset_id::intern_all(&["hero"]);
        let args: Camera3DArgs = serde_json::from_value(serde_json::json!({
            "controller": {"follow": {"target": "hero", "drive": "direct"}}
        }))
        .unwrap();
        let follow = args.controller.unwrap().follow.unwrap();
        assert_eq!(follow.target, Some(AssetId(0)));
        assert_eq!(follow.drive, FollowDrive::Direct);
        // Omitted fields keep the documented defaults.
        assert_eq!(follow.speed_parameter, "speed");
        assert!((follow.distance - 4.0).abs() < 1e-6);
        assert!((follow.height - 1.5).abs() < 1e-6);
        assert_eq!(follow.jump_height, 0.0);

        // No follow block keeps the first-person modes.
        let bare: Camera3DArgs =
            serde_json::from_value(serde_json::json!({"controller": {}})).unwrap();
        assert!(bare.controller.unwrap().follow.is_none());
    }
}
