// Camera-shot preset schema.

use alloc::string::String;

/// A reusable [Camera3D](#camera3d) preset: reference it from a
/// [Scene](#scene)'s `camera_shot`, or use it standalone.
///
/// Used standalone, it expands into a [Camera3D](#camera3d) with the same
/// parameters.
///
/// **Examples**
///
/// With Scenes: camera switches per scene (declared on each Scene):
/// ```jsonl
/// {"name":"wide", "type":"CameraShot","args":{"fov_y_degrees":80,"position":[0,1.75,8],"yaw":3.14}}
/// {"name":"close","type":"CameraShot","args":{"fov_y_degrees":55,"position":[0,1.5,3],"yaw":3.14}}
/// {"name":"intro", "type":"Scene","args":{"camera_shot":"wide"}}
/// {"name":"detail","type":"Scene","args":{"camera_shot":"close"}}
/// ```
///
/// From library preset (standalone, replaces Camera3D):
/// ```jsonl
/// {"name":"cam","type":"CameraShot","args":{"preset":"shot_outdoor_wide"}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CameraShot {
    /// Name of a built-in or file-backed preset (e.g. "shot_eye_level").
    /// Preset values are used as defaults; any inline fields override them.
    pub preset: String,
    /// Vertical field of view in degrees.
    pub fov_y_degrees: f32,
    /// Near clip plane distance in world units.
    pub near: f32,
    /// Far clip plane distance in world units.
    pub far: f32,
    /// World-space camera position.
    pub position: [f32; 3],
    /// Yaw rotation in radians (Y-axis, applied first).
    pub yaw: f32,
    /// Pitch rotation in radians (X-axis, applied second).
    pub pitch: f32,
}

impl Default for CameraShot {
    fn default() -> Self {
        Self {
            preset: String::new(),
            fov_y_degrees: 75.0,
            near: 0.05,
            far: 200.0,
            position: [0.0, 0.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_camera_they_configure() {
        // A shot overwrites a Camera3D's framing, so an unset field has to mean
        // the same thing on both.
        let s = CameraShot::default();
        assert!(s.preset.is_empty());
        assert_eq!(s.fov_y_degrees, 75.0);
        assert_eq!((s.near, s.far), (0.05, 200.0));
        assert_eq!(s.position, [0.0, 0.0, 0.0]);
        assert_eq!((s.yaw, s.pitch), (0.0, 0.0));
    }

    #[test]
    fn a_named_preset_parses_and_round_trips_through_postcard() {
        let s: CameraShot = serde_json::from_str(
            r#"{"preset":"establishing","fov_y_degrees":40,"position":[0,3,8],"yaw":1.5}"#,
        )
        .unwrap();
        assert_eq!(s.preset, "establishing");
        assert_eq!(s.fov_y_degrees, 40.0);
        assert_eq!(s.position, [0.0, 3.0, 8.0]);
        assert_eq!(s.yaw, 1.5);

        let bytes = postcard::to_allocvec(&s).unwrap();
        let back: CameraShot = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.preset, "establishing");
        assert_eq!(back.far, 200.0);
    }
}
