// src/assets/panel.rs

use crate::assets::Panel;
use crate::ecs::{AssetOrigin, Component};

impl Component for Panel {
    const NAME: &'static str = "Panel";
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
        let p: Panel = serde_json::from_str("{}").unwrap();
        assert!(p.title.is_empty());
        assert_eq!(p.width, 400.0);
        assert_eq!(p.corner_radius, 8.0);
        assert_eq!(p.title_scale, 1.0);
    }

    #[test]
    fn explicit_fields_round_trip() {
        let json = r#"{"title":"Paused","x":440,"y":220,"width":400,"height":280}"#;
        let p: Panel = serde_json::from_str(json).unwrap();
        assert_eq!(p.title, "Paused");
        assert_eq!(p.x, 440.0);
        let back = serde_json::to_value(&p).unwrap();
        assert_eq!(back["title"], "Paused");
        assert!(back.is_object());
    }
}
