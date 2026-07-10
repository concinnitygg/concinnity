// src/assets/option_select.rs

use crate::assets::OptionSelect;
use crate::ecs::{AssetOrigin, Component};

impl Component for OptionSelect {
    const NAME: &'static str = "OptionSelect";
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
        let o: OptionSelect = serde_json::from_str("{}").unwrap();
        assert!(o.setting.is_empty());
        assert_eq!(o.width, 360.0);
        assert_eq!(o.text_scale, 1.0);
    }

    #[test]
    fn explicit_setting_and_label_round_trip() {
        let json = r#"{"setting":"vsync","label":"Vsync"}"#;
        let o: OptionSelect = serde_json::from_str(json).unwrap();
        assert_eq!(o.setting, "vsync");
        assert_eq!(o.label, "Vsync");
        let back = serde_json::to_value(&o).unwrap();
        assert_eq!(back["setting"], "vsync");
    }
}
