// src/assets/scene_reel.rs

use crate::assets::SceneReel;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

impl Component for SceneReel {
    const NAME: &'static str = "SceneReel";
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

impl crate::check::cross_reference::CrossReferenced for SceneReel {
    fn cross_refs(
        name: &str,
        args: &serde_json::Value,
    ) -> Vec<crate::check::cross_reference::CrossRef> {
        use crate::check::cross_reference::{CrossRef, RefKind};
        let mut refs = Vec::new();

        if let Some(entries) = args.get("scenes").and_then(|v| v.as_array()) {
            if entries.is_empty() {
                refs.push(CrossRef::Issue(format!(
                    "SceneReel '{}': scenes list is empty",
                    name
                )));
            }
            for (i, entry) in entries.iter().enumerate() {
                let scene_ref = entry.as_str().unwrap_or("");
                if scene_ref.is_empty() {
                    refs.push(CrossRef::Issue(format!(
                        "SceneReel '{}': scenes[{}] is not a valid scene name string",
                        name, i
                    )));
                } else {
                    refs.push(CrossRef::Resolve {
                        kind: RefKind::Scene,
                        target: scene_ref.to_string(),
                        error: format!(
                            "SceneReel '{}': scenes[{}] references unknown scene '{}', add a Scene asset with that name",
                            name, i, scene_ref
                        ),
                    });
                }
            }
        }

        refs
    }
}
