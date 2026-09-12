// Scene marker schema.

use crate::components::{vocabulary, vocabulary_synonyms};
use crate::ecs::asset_id::AssetId;
use crate::ecs::asset_id::de_opt_asset_ref;

/// How a scene jump reaches the new scene. The single accepted vocabulary for
/// a [Behavior](#behavior) scene node's `transition` and a `scene:<name>` UI
/// action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SceneTransition {
    /// Fade the whole composited image to black, swap scenes at the bottom of
    /// the fade, then fade back in.
    #[default]
    FadeBlack,
    /// Swap scenes on the next frame with no fade.
    Cut,
}

vocabulary!(SceneTransition {
    FadeBlack => "FadeBlack",
    Cut => "Cut",
});
vocabulary_synonyms!(SceneTransition, "a scene transition index");

impl SceneTransition {
    /// The transition an authored name selects, case-insensitively. `None` for
    /// an unknown name.
    pub fn from_str_norm(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "fadeblack" => Some(Self::FadeBlack),
            "cut" => Some(Self::Cut),
            _ => None,
        }
    }
}

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

    // NAMES is what the editor's picker offers and what an authored world
    // writes, so every one has to resolve and every transition has to be named.
    #[test]
    fn every_transition_name_resolves_back_to_its_transition() {
        assert_eq!(SceneTransition::ALL.len(), SceneTransition::NAMES.len());
        for (transition, name) in SceneTransition::ALL.iter().zip(SceneTransition::NAMES) {
            assert_eq!(transition.as_str(), *name);
            assert_eq!(SceneTransition::from_str_norm(name), Some(*transition));
            assert_eq!(
                serde_json::to_string(transition).expect("serializes"),
                alloc::format!("\"{name}\"")
            );
        }
        assert_eq!(SceneTransition::from_str_norm("dissolve"), None);
        // Case-insensitively, so a world that authored the lowercase spelling
        // still loads.
        assert_eq!(
            serde_json::from_str::<SceneTransition>(r#""fadeblack""#).expect("loads"),
            SceneTransition::FadeBlack
        );
        serde_json::from_str::<SceneTransition>(r#""dissolve""#)
            .expect_err("an unknown transition does not deserialize");
    }

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
        let s: Scene = crate::test_support::from_json(r#"{"camera_shot":"establishing"}"#);
        assert_eq!(s.camera_shot, Some(AssetId(12)));

        let bytes = postcard::to_allocvec(&s).unwrap();
        let back: Scene = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.camera_shot, Some(AssetId(12)));
        assert_eq!(back.asset_id, AssetId::default());
    }
}
