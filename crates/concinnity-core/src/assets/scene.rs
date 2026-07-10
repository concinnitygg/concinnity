// src/assets/scene.rs

use crate::assets::Scene;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

impl Component for Scene {
    const NAME: &'static str = "Scene";

    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }

    fn from_args(args: Self) -> Self {
        args
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

impl crate::check::cross_reference::CrossReferenced for Scene {
    fn cross_refs(
        name: &str,
        args: &serde_json::Value,
    ) -> Vec<crate::check::cross_reference::CrossRef> {
        use crate::check::cross_reference::{CrossRef, RefKind};
        let mut refs = Vec::new();

        let shot_ref = args
            .get("camera_shot")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if !shot_ref.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::CameraShot,
                target: shot_ref.to_string(),
                error: format!(
                    "Scene '{}': camera_shot '{}' not found, add a CameraShot or Camera3D asset with that name",
                    name, shot_ref
                ),
            });
        }

        refs
    }
}
