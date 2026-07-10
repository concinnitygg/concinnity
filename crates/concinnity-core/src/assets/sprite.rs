// src/assets/sprite.rs

use crate::assets::Sprite;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

impl Component for Sprite {
    const NAME: &'static str = "Sprite";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn ref_fields() -> &'static [(&'static str, &'static str)] {
        &[("texture", "Texture"), ("view", "View")]
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
    use crate::assets::SpriteFit;

    #[test]
    fn deserializes_with_all_fields() {
        let json = r#"{
            "x": 10, "y": 20, "width": 300, "height": 200,
            "tint": [0.5, 0.5, 0.5, 0.8], "visible": true
        }"#;
        let s: Sprite = serde_json::from_str(json).unwrap();
        assert_eq!(s.x, 10.0);
        assert_eq!(s.width, 300.0);
        assert_eq!(s.tint, [0.5, 0.5, 0.5, 0.8]);
        assert!(s.visible);
        assert!(s.texture.is_none());
    }

    #[test]
    fn deserializes_with_defaults() {
        let s: Sprite = serde_json::from_str("{}").unwrap();
        assert_eq!(s.tint, [1.0, 1.0, 1.0, 1.0]);
        assert!(s.visible);
        assert_eq!(s.width, 100.0);
        assert!(!s.follow_cursor);
        assert_eq!(s.fit, SpriteFit::Fit);
        assert_eq!(s.corner_radius, 0.0);
    }

    #[test]
    fn corner_radius_round_trips() {
        let s: Sprite = serde_json::from_str(r#"{"corner_radius":12.5}"#).unwrap();
        assert_eq!(s.corner_radius, 12.5);
        let back = serde_json::to_value(&s).unwrap();
        assert_eq!(back["corner_radius"], 12.5);
    }

    #[test]
    fn fit_deserializes_lowercase_and_round_trips() {
        let s: Sprite = serde_json::from_str(r#"{"fit":"cover"}"#).unwrap();
        assert_eq!(s.fit, SpriteFit::Cover);
        let back = serde_json::to_value(&s).unwrap();
        assert_eq!(back["fit"], "cover");
    }

    #[test]
    fn follow_cursor_round_trips() {
        let json = r#"{"follow_cursor":true,"width":16,"height":16}"#;
        let s: Sprite = serde_json::from_str(json).unwrap();
        assert!(s.follow_cursor);
        let back = serde_json::to_value(&s).unwrap();
        assert_eq!(back["follow_cursor"], true);
    }

    #[test]
    fn deserializes_with_texture_reference() {
        crate::ecs::asset_id::reset_interner();
        // The interner is global and not reset here; we just check the field
        // is populated when a string name is supplied (it interns lazily).
        let json = r#"{"texture":"tex_intro","tint":[1,1,1,1]}"#;
        let s: Sprite = serde_json::from_str(json).unwrap();
        assert!(s.texture.is_some());
    }
}
