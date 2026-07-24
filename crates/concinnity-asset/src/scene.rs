// Scene marker schema.

use crate::{AssetId, de_opt_asset_ref};

/// A named group of world content.
///
/// [Prop](#prop)s belong to a Scene by naming convention: props whose `name`
/// begins with `<scene_name>_` are associated with that Scene. Props not
/// prefixed by any scene name are visible in every scene.
///
/// The first declared Scene is active at world start. Scene changes are driven
/// by actions: a UI `scene:<name>` action ([HitRegion](#hitregion) /
/// [KeyBinding](#keybinding)) or a [Reaction](#reaction) scene action jumps to
/// the named scene, with the transition ("Cut" or "FadeBlack") declared on the
/// jump.
///
/// ```jsonl
/// {"name":"day",  "type":"Scene","args":{}}
/// {"name":"night","type":"Scene","args":{}}
/// // Props named "day_*" belong to Scene "day"; "night_*" to Scene "night"
/// {"name":"nightfall","type":"Reaction","args":{"on":{"timer":{"interval":5.0}},"actions":[{"scene":{"scene":"night"}}]}}
/// ```
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Scene {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// A [CameraShot](#camerashot) or [Camera3D](#camera3d) to activate when
    /// this scene becomes active. `None` keeps the current camera unchanged.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub camera_shot: Option<AssetId>,
}
