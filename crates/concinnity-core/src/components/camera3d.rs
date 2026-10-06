//! Runtime 3D camera component. Its authored args and controller config live in
//! this file, alongside the runtime component they bake into.

use crate::components::Vocabulary;
use crate::ecs::SkinnedMeshHandle;
use alloc::string::String;

/// How a followed character converts movement input into displacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Vocabulary)]
#[serde(rename_all = "snake_case")]
pub enum FollowDrive {
    /// The controller only writes the speed parameter and the facing; the
    /// character moves by the displacement its animation clips carry (clips
    /// baked with [root_motion](animation.md)). Clips must travel along
    /// local -Z so the facing yaw and the travel direction agree.
    #[vocab("root_motion")]
    RootMotion,
    /// The controller moves the character capsule directly at the camera
    /// controller's `move_speed`, for characters whose clips animate in
    /// place. The speed parameter is still written, so a locomotion
    /// blendspace matches the visual gait to the travel speed.
    #[vocab("direct")]
    Direct,
}

/// Third-person follow settings carried on a [CameraController](#cameracontroller).
///
/// When `follow` is set the camera becomes a third-person orbit camera: the
/// mouse orbits around the followed character, and WASD steers the character
/// itself (camera-relative). The character must be a
/// [SkinnedMesh](skinned_mesh.md) with a `capsule`, so it has a kinematic
/// character capsule to move.
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct FollowController {
    /// Name of the followed [SkinnedMesh](skinned_mesh.md). It must declare a
    /// `capsule`.
    pub target: Option<SkinnedMeshHandle>,
    /// Orbit distance from the pivot to the camera, in world units.
    #[asset(default = 4.0)]
    pub distance: f32,
    /// Pivot height above the character's feet, in world units.
    #[asset(default = 1.5)]
    pub height: f32,
    /// How the character moves; see [FollowDrive](#followdrive).
    #[asset(default = FollowDrive::RootMotion)]
    pub drive: FollowDrive,
    /// Character turn rate toward the input heading, in radians per second.
    #[asset(default = 10.0)]
    pub turn_speed: f32,
    /// Name of the character's [AnimationGraph](anim_graph.md) float parameter
    /// that receives the current travel speed in world units per second
    /// (drives a locomotion blendspace). Empty disables parameter writes,
    /// leaving the graph externally driven.
    #[asset(default = "speed")]
    pub speed_parameter: String,
    /// Jump apex height in world units when the jump key is pressed while
    /// grounded. `0` disables jumping.
    pub jump_height: f32,
}

/// First-person / fly-through controller settings carried on a `Camera3D`.
///
/// A `Camera3D` whose `controller` is set (the default) is driven each frame by
/// the internal camera controller, which turns mouse/keyboard input into a
/// camera orientation and a movement intent. Set `controller` to `null` for a
/// camera driven by something else (a `CameraShot` / `Scene` cutscene).
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct CameraController {
    /// Direct 6-DoF flight mode. WASD moves along the camera's full forward
    /// vector (yaw + pitch) and jump rises along world +Y; the controller
    /// writes the new position straight onto Camera3D, bypassing the physics
    /// step and the bounds box. Used for inspector / fly-through cameras (the
    /// default, e.g. the `cn add foo.glb` scaffold). Set `false` for the
    /// FPS-style ground walker.
    // The `cn add foo.glb` scaffold relies on a bare `Camera3D` being a free-fly inspector.
    #[asset(default = true)]
    pub free_fly: bool,
    /// Walk / fly speed in world units per second.
    #[asset(default = 1.0)]
    pub move_speed: f32,
    /// Sprint multiplier applied when the sprint key is held.
    #[asset(default = 3.0)]
    pub sprint_multiplier: f32,
    /// Mouse look sensitivity in radians per pixel.
    #[asset(default = 0.0015)]
    pub mouse_sensitivity: f32,
    /// Margin kept between the camera and the bounds box (world units).
    #[asset(default = 0.3)]
    pub player_radius: f32,
    /// AABB minimum corner the camera center must stay inside [x, y, z].
    #[asset(default = [-1.0e9; 3])]
    pub bounds_min: [f32; 3],
    /// AABB maximum corner the camera center must stay inside [x, y, z].
    #[asset(default = [1.0e9; 3])]
    pub bounds_max: [f32; 3],
    /// Third-person follow settings; see [FollowController](#followcontroller).
    /// When set, the camera orbits the followed character and WASD steers the
    /// character instead of the camera (`free_fly` and the bounds box are
    /// ignored). `null` (the default) keeps the first-person / fly modes.
    pub follow: Option<FollowController>,
}

// A `Camera3D` with no explicit `controller` gets the default inspector
// controller, so an authored scene is navigable out of the box.
fn default_controller() -> Option<CameraController> {
    Some(CameraController::default())
}

/// Declares the 3D camera. One per scene.
///
/// ```rust
/// # use concinnity_core::components::cook::Camera3D as Camera3DArgs;
/// Camera3DArgs {
///     fov_y_degrees: 80.0,
///     near: 0.05,
///     view_distance: Some(500.0),
///     position: [0.0, 4.0, 0.0],
///     ..Default::default()
/// };
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
pub struct Camera3DArgs {
    /// Vertical field-of-view in degrees.
    #[asset(default = 75.0)]
    pub fov_y_degrees: f32,
    /// Near clip plane distance.
    #[asset(default = 0.05)]
    pub near: f32,
    /// How far the camera sees, in world units. Objects lying wholly beyond
    /// this distance along the view direction are not drawn, and shadows reach
    /// no farther; an object that straddles it is drawn whole, never cut.
    /// `null` (the default) sees without limit: there is no far clip plane.
    pub view_distance: Option<f32>,
    /// Initial eye position in world space [x, y, z].
    #[asset(default = [0.0, 1.7, 0.0])]
    pub position: [f32; 3],
    /// Initial yaw in radians (0 = looking toward -Z).
    pub yaw: f32,
    /// Initial pitch in radians.
    pub pitch: f32,
    /// Input controller settings, or `null` to leave the camera uncontrolled
    /// (driven by a [CameraShot](#camerashot) / [Scene](#scene)
    /// cutscene). Omitted defaults to a free-fly inspector controller.
    #[asset(default = default_controller())]
    pub controller: Option<CameraController>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_view_distance_is_unlimited_unless_set() {
        let bare: Camera3DArgs = crate::test_support::from_json("{}");
        assert_eq!(bare.view_distance, None);
        let set: Camera3DArgs = crate::test_support::from_json(r#"{"view_distance":750.0}"#);
        assert_eq!(set.view_distance, Some(750.0));
        assert_eq!(Camera3D::bake(set).view_distance, Some(750.0));
    }

    #[test]
    fn an_explicit_null_controller_leaves_the_camera_undriven() {
        let args: Camera3DArgs = crate::test_support::from_json(r#"{"controller":null}"#);
        assert!(args.controller.is_none());
    }

    #[test]
    fn a_ground_walker_turns_free_fly_off() {
        let args: Camera3DArgs = crate::test_support::from_json(
            r#"{"controller":{"free_fly":false,"player_radius":0.4}}"#,
        );
        let c = args.controller.expect("controller");
        assert!(!c.free_fly);
        assert_eq!(c.player_radius, 0.4);
        assert_eq!(
            c.mouse_sensitivity,
            CameraController::default().mouse_sensitivity
        );
    }

    #[test]
    fn a_partial_follow_block_resolves_its_target_and_fills_the_rest() {
        let args: Camera3DArgs = crate::test_support::from_json(
            r#"{"controller":{"follow":{"target":"hero","drive":"direct"}}}"#,
        );
        let follow = args.controller.expect("controller").follow.expect("follow");
        assert_eq!(follow.target, Some(SkinnedMeshHandle::new(4)));
        assert_eq!(follow.drive, FollowDrive::Direct);
        let defaults = FollowController::default();
        assert_eq!(follow.speed_parameter, defaults.speed_parameter);
        assert_eq!(
            (follow.distance, follow.height),
            (defaults.distance, defaults.height)
        );

        // No follow block keeps the first-person modes.
        let bare: Camera3DArgs = crate::test_support::from_json(r#"{"controller":{}}"#);
        assert!(bare.controller.expect("controller").follow.is_none());
    }

    #[test]
    fn drive_names_parse_in_snake_case() {
        let d = |s: &str| serde_json::from_str::<FollowDrive>(s).unwrap();
        assert_eq!(d(r#""root_motion""#), FollowDrive::RootMotion);
        assert_eq!(d(r#""direct""#), FollowDrive::Direct);
        assert_eq!(
            serde_json::to_string(&FollowDrive::RootMotion).unwrap(),
            r#""root_motion""#
        );
    }
}

/// The runtime `Camera3D`: the authored fields of
/// [`cook::Camera3D`](crate::components::cook::Camera3D) plus the view matrix and
/// per-frame input intent, which are not declared.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Camera3D {
    /// Vertical field of view in degrees.
    pub fov_y_degrees: f32,
    /// Near clip distance in world units.
    pub near: f32,
    /// How far the camera sees in world units, or `None` for no limit.
    pub view_distance: Option<f32>,
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

impl Camera3D {
    /// Translate the authored args into the runtime camera: compose the initial
    /// view matrix and zero the runtime state. Run by cook at build time (the
    /// baked blob record carries the result) and by tests that need a camera.
    pub fn bake(args: Camera3DArgs) -> Self {
        Self {
            fov_y_degrees: args.fov_y_degrees,
            near: args.near,
            view_distance: args.view_distance,
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
