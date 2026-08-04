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
/// [KeyBinding](#keybinding)) or a [Behavior](#behavior) scene node jumps to
/// the named scene, with the transition ("Cut" or "FadeBlack") declared on the
/// jump.
///
/// ```jsonl
/// {"name":"day",  "type":"Scene","args":{}}
/// {"name":"night","type":"Scene","args":{}}
/// // Props named "day_*" belong to Scene "day"; "night_*" to Scene "night"
/// {"name":"nightfall","type":"Behavior","args":{"on":{"timer":{"interval":5.0}},"do":[{"scene":{"scene":"night"}}]}}
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scene_with_no_shot_leaves_the_camera_where_it_is() {
        let s = Scene::default();
        assert!(s.camera_shot.is_none());
        assert_eq!(s.asset_id, AssetId::default());
        assert!(
            serde_json::from_str::<Scene>("{}")
                .unwrap()
                .camera_shot
                .is_none()
        );
    }

    #[test]
    fn a_named_shot_parses_and_round_trips_through_postcard() {
        crate::test_support::install_resolvers();
        let s: Scene = serde_json::from_str(r#"{"camera_shot":"establishing"}"#).unwrap();
        assert_eq!(s.camera_shot, Some(AssetId(12)));

        let bytes = postcard::to_allocvec(&s).unwrap();
        let back: Scene = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.camera_shot, Some(AssetId(12)));
        assert_eq!(back.asset_id, AssetId::default());
    }
}
