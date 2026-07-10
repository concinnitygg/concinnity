// src/assets/fps_counter.rs
//
// FpsCounter component (pure data). The runtime behavior that reads it lives in
// the client crate's `hud::fps_counter`.

use crate::assets::FpsCounter;
use crate::ecs::{AssetOrigin, Component};

impl Component for FpsCounter {
    const NAME: &'static str = "FpsCounter";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn ref_fields() -> &'static [(&'static str, &'static str)] {
        &[("label", "TextLabel")]
    }

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }
}

impl crate::check::cross_reference::CrossReferenced for FpsCounter {
    fn cross_refs(
        name: &str,
        args: &serde_json::Value,
    ) -> Vec<crate::check::cross_reference::CrossRef> {
        crate::check::cross_reference::label_refs("FpsCounter", name, args, &["label"])
    }
}
