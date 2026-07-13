// Scene marker schema.

use crate::{AssetId, de_opt_asset_ref};
use alloc::string::{String, ToString};

/// A named group marker with timing and transition settings for
/// [SceneReel](#scenereel).
///
/// [Prop](#prop)s belong to a Scene by naming convention: props whose `name`
/// begins with `<scene_name>_` are associated with that Scene.
///
/// ```jsonl
/// {"name":"day",  "type":"Scene","args":{"duration_secs":5.0,"transition":"FadeBlack"}}
/// {"name":"night","type":"Scene","args":{"duration_secs":5.0,"transition":"FadeBlack"}}
/// // Props named "day_*" belong to Scene "day"; "night_*" to Scene "night"
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Scene {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Seconds to hold this scene before advancing. None = hold indefinitely.
    pub duration_secs: Option<f32>,
    /// Transition into this scene: "Cut" (immediate) or "FadeBlack" (dip to black).
    pub transition: String,
    /// A [CameraShot](#camerashot) or [Camera3D](#camera3d) to activate when
    /// this scene becomes active. `None` keeps the current camera unchanged.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub camera_shot: Option<AssetId>,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            duration_secs: None,
            transition: "Cut".to_string(),
            camera_shot: None,
        }
    }
}
