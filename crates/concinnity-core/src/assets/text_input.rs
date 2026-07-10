// src/assets/text_input.rs

use crate::assets::TextInput;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

impl Component for TextInput {
    const NAME: &'static str = "TextInput";
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_with_defaults() {
        let t: TextInput = serde_json::from_str("{}").unwrap();
        assert_eq!(t.content, "");
        assert_eq!(t.width, 240.0);
        assert_eq!(t.max_len, 0);
        assert!(t.visible);
        assert!(!t.focused);
        assert_eq!(t.caret, 0);
    }

    #[test]
    fn deserializes_with_fields() {
        let json = r#"{
            "placeholder": "Name", "content": "hi",
            "x": 10, "y": 20, "width": 300, "height": 48, "max_len": 24
        }"#;
        let t: TextInput = serde_json::from_str(json).unwrap();
        assert_eq!(t.placeholder, "Name");
        assert_eq!(t.content, "hi");
        assert_eq!(t.width, 300.0);
        assert_eq!(t.max_len, 24);
    }

    #[test]
    fn runtime_state_is_not_serialized() {
        // `focused` / `caret` / `asset_id` are runtime-only, so `args` (the
        // public schema) never carries them.
        let t = TextInput {
            focused: true,
            caret: 3,
            ..Default::default()
        };
        let v = serde_json::to_value(&t).unwrap();
        assert!(v.get("focused").is_none());
        assert!(v.get("caret").is_none());
        assert!(v.get("asset_id").is_none());
        assert!(v.is_object());
    }
}
