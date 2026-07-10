// src/assets/text_label.rs

use crate::assets::TextLabel;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, CompanionSpec, Component};

impl Component for TextLabel {
    const NAME: &'static str = "TextLabel";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn ref_fields() -> &'static [(&'static str, &'static str)] {
        &[("font", "Font"), ("view", "View")]
    }

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }

    fn companions(_args: &serde_json::Value, _world: &[serde_json::Value]) -> Vec<CompanionSpec> {
        vec![CompanionSpec {
            name: "GraphicsConfig",
            asset_type: "GraphicsConfig",
            args: serde_json::json!({}),
        }]
    }
}
