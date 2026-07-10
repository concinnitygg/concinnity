// src/assets/slider.rs

use crate::assets::Slider;
use crate::ecs::{AssetOrigin, Component};

impl Component for Slider {
    const NAME: &'static str = "Slider";
    const ORIGIN: AssetOrigin = AssetOrigin::BuildOnly;
    type Args = Self;

    fn from_args(args: Self) -> Self {
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_args_deserialize_with_defaults() {
        let s: Slider = serde_json::from_str("{}").unwrap();
        assert!(s.setting.is_empty());
        assert_eq!(s.width, 360.0);
        assert_eq!(s.text_scale, 1.0);
        assert_eq!(s.handle_color, [1.0, 0.85, 0.3, 1.0]);
    }

    #[test]
    fn explicit_setting_and_label_round_trip() {
        let json = r#"{"setting":"exposure","label":"Exposure"}"#;
        let s: Slider = serde_json::from_str(json).unwrap();
        assert_eq!(s.setting, "exposure");
        assert_eq!(s.label, "Exposure");
        let back = serde_json::to_value(&s).unwrap();
        assert_eq!(back["setting"], "exposure");
    }
}
