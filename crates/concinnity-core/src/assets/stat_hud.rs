// src/assets/stat_hud.rs
//
// StatHud component (pure data). The runtime behavior that reads it lives in
// the client crate's `hud::stat_hud`.

use crate::assets::StatHud;
use crate::ecs::{AssetOrigin, CompanionSpec, Component};

impl Component for StatHud {
    const NAME: &'static str = "StatHud";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }

    fn companions(_args: &serde_json::Value, _world: &[serde_json::Value]) -> Vec<CompanionSpec> {
        vec![CompanionSpec {
            name: "GraphicsConfig",
            asset_type: "GraphicsConfig",
            args: serde_json::json!({}),
        }]
    }
}

impl crate::check::cross_reference::CrossReferenced for StatHud {
    fn cross_refs(
        name: &str,
        args: &serde_json::Value,
    ) -> Vec<crate::check::cross_reference::CrossRef> {
        crate::check::cross_reference::label_refs(
            "StatHud",
            name,
            args,
            &["fps_label", "vram_label", "ev_label", "edr_label"],
        )
    }
}
