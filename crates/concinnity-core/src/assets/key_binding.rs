// src/assets/key_binding.rs

use crate::assets::KeyBinding;
use crate::ecs::{AssetOrigin, Component};

impl Component for KeyBinding {
    const NAME: &'static str = "KeyBinding";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_escape_to_view_toggle() {
        let json = r#"{"key":"Escape","action":"view:toggle:pause_menu"}"#;
        let kb: KeyBinding = serde_json::from_str(json).unwrap();
        assert_eq!(kb.key, "Escape");
        assert_eq!(kb.action, "view:toggle:pause_menu");
    }

    #[test]
    fn deserializes_with_defaults_to_empty_strings() {
        let kb: KeyBinding = serde_json::from_str("{}").unwrap();
        assert!(kb.key.is_empty());
        assert!(kb.action.is_empty());
    }
}
