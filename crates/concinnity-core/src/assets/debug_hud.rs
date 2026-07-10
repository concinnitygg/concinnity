// src/assets/debug_hud.rs
//
// DebugHud component (pure data). The runtime behavior that reads it lives in
// the client crate's `hud::debug_hud`.

use crate::assets::DebugHud;
use crate::ecs::{AssetOrigin, CompanionSpec, Component};

impl Component for DebugHud {
    const NAME: &'static str = "DebugHud";
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

impl crate::check::cross_reference::CrossReferenced for DebugHud {
    fn cross_refs(
        name: &str,
        args: &serde_json::Value,
    ) -> Vec<crate::check::cross_reference::CrossRef> {
        crate::check::cross_reference::label_refs(
            "DebugHud",
            name,
            args,
            &["passes_label", "mouse_label", "camera_label"],
        )
    }
}
