// Camera authoring schema: the 3D camera's authored args and controller config.
// The runtime `Camera3D` component (view matrix, per-frame intent) lives in core.

use crate::{AssetId, de_opt_asset_ref};
use alloc::string::{String, ToString};

/// How a followed character converts movement input into displacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FollowDrive {
    /// The controller only writes the speed parameter and the facing; the
    /// character moves by the displacement its animation clips carry (clips
    /// baked with [root_motion](animation.md)). Clips must travel along
    /// local -Z so the facing yaw and the travel direction agree.
    RootMotion,
    /// The controller moves the character capsule directly at the camera
    /// controller's `move_speed`, for characters whose clips animate in
    /// place. The speed parameter is still written, so a locomotion
    /// blendspace matches the visual gait to the travel speed.
    Direct,
}

/// Third-person follow settings carried on a [CameraController](#cameracontroller).
///
/// When `follow` is set the camera becomes a third-person orbit camera: the
/// mouse orbits around the followed character, and WASD steers the character
/// itself (camera-relative). The character must be a
/// [SkinnedMesh](skinned_mesh.md) with a `capsule`, so it has a kinematic
/// character capsule to move.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct FollowController {
    /// Name of the followed [SkinnedMesh](skinned_mesh.md). It must declare a
    /// `capsule`.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub target: Option<AssetId>,
    /// Orbit distance from the pivot to the camera, in world units.
    pub distance: f32,
    /// Pivot height above the character's feet, in world units.
    pub height: f32,
    /// How the character moves; see [FollowDrive](#followdrive).
    pub drive: FollowDrive,
    /// Character turn rate toward the input heading, in radians per second.
    pub turn_speed: f32,
    /// Name of the character's [AnimGraph](anim_graph.md) float parameter
    /// that receives the current travel speed in world units per second
    /// (drives a locomotion blendspace). Empty disables parameter writes,
    /// leaving the graph externally driven.
    pub speed_parameter: String,
    /// Jump apex height in world units when the jump key is pressed while
    /// grounded. `0` disables jumping.
    pub jump_height: f32,
}

impl Default for FollowController {
    fn default() -> Self {
        Self {
            target: None,
            distance: 4.0,
            height: 1.5,
            drive: FollowDrive::RootMotion,
            turn_speed: 10.0,
            speed_parameter: "speed".to_string(),
            jump_height: 0.0,
        }
    }
}

/// First-person / fly-through controller settings carried on a `Camera3D`.
///
/// A `Camera3D` whose `controller` is set (the default) is driven each frame by
/// the internal camera controller, which turns mouse/keyboard input into a
/// camera orientation and a movement intent. Set `controller` to `null` for a
/// camera driven by something else (a `CameraShot` / `SceneReel` cutscene).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CameraController {
    /// Direct 6-DoF flight mode. WASD moves along the camera's full forward
    /// vector (yaw + pitch) and jump rises along world +Y; the controller
    /// writes the new position straight onto Camera3D, bypassing the physics
    /// step and the bounds box. Used for inspector / fly-through cameras (the
    /// default, e.g. the `cn add foo.glb` scaffold). Set `false` for the
    /// FPS-style ground walker.
    pub free_fly: bool,
    /// Walk / fly speed in world units per second.
    pub move_speed: f32,
    /// Sprint multiplier applied when the sprint key is held.
    pub sprint_multiplier: f32,
    /// Mouse look sensitivity in radians per pixel.
    pub mouse_sensitivity: f32,
    /// Margin kept between the camera and the bounds box (world units).
    pub player_radius: f32,
    /// AABB minimum corner the camera centre must stay inside [x, y, z].
    pub bounds_min: [f32; 3],
    /// AABB maximum corner the camera centre must stay inside [x, y, z].
    pub bounds_max: [f32; 3],
    /// Third-person follow settings; see [FollowController](#followcontroller).
    /// When set, the camera orbits the followed character and WASD steers the
    /// character instead of the camera (`free_fly` and the bounds box are
    /// ignored). `null` (the default) keeps the first-person / fly modes.
    pub follow: Option<FollowController>,
}

impl Default for CameraController {
    fn default() -> Self {
        const BIG: f32 = 1.0e9;
        Self {
            // A bare `Camera3D` is navigable out of the box as a free-fly
            // inspector: the `cn add foo.glb` scaffold relies on this. Worlds
            // that want the FPS ground walker set `free_fly: false`.
            free_fly: true,
            move_speed: 1.0,
            sprint_multiplier: 3.0,
            mouse_sensitivity: 0.0015,
            player_radius: 0.3,
            bounds_min: [-BIG, -BIG, -BIG],
            bounds_max: [BIG, BIG, BIG],
            follow: None,
        }
    }
}

// A `Camera3D` with no explicit `controller` gets the default inspector
// controller, so an authored scene is navigable out of the box.
fn default_controller() -> Option<CameraController> {
    Some(CameraController::default())
}

/// Authored fields of a `Camera3D`; the runtime view matrix and per-frame input
/// intent are not declared.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Camera3DArgs {
    /// Vertical field-of-view in degrees.
    pub fov_y_degrees: f32,
    /// Near clip plane distance.
    pub near: f32,
    /// Far clip plane distance.
    pub far: f32,
    /// Initial eye position in world space [x, y, z].
    pub position: [f32; 3],
    /// Initial yaw in radians (0 = looking toward -Z).
    pub yaw: f32,
    /// Initial pitch in radians.
    pub pitch: f32,
    /// Input controller settings, or `null` to leave the camera uncontrolled
    /// (driven by a [CameraShot](#camerashot) / [SceneReel](#scenereel)
    /// cutscene). Omitted defaults to a free-fly inspector controller.
    #[serde(default = "default_controller")]
    pub controller: Option<CameraController>,
}

impl Default for Camera3DArgs {
    fn default() -> Self {
        Self {
            fov_y_degrees: 75.0,
            near: 0.05,
            far: 200.0,
            position: [0.0, 1.7, 0.0],
            yaw: 0.0,
            pitch: 0.0,
            controller: default_controller(),
        }
    }
}
