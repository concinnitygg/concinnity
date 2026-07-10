// src/assets/decal.rs

use crate::assets::Decal;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

impl Component for Decal {
    const NAME: &'static str = "Decal";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn ref_fields() -> &'static [(&'static str, &'static str)] {
        &[("texture", "Texture")]
    }

    fn from_args(mut args: Self) -> Self {
        // Clamp the alpha to [0, 1] so a stray > 1 doesn't blow out the
        // composite. The size components are left as-authored: a non-positive
        // value silently disables the decal in the gfx-side resolver below.
        args.tint[3] = args.tint[3].clamp(0.0, 1.0);
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_with_defaults() {
        let d: Decal = serde_json::from_str("{}").unwrap();
        assert_eq!(d.position, [0.0, 0.0, 0.0]);
        assert_eq!(d.size, [1.0, 1.0, 1.0]);
        assert_eq!(d.tint, [1.0, 1.0, 1.0, 1.0]);
        assert!(d.visible);
        assert!(d.texture.is_none());
    }

    #[test]
    fn deserialises_with_all_fields() {
        crate::ecs::asset_id::reset_interner();
        let json = r#"{
            "texture":"tex_bullet",
            "position":[1.0,2.0,3.0],
            "rotation_deg":[0,90,0],
            "size":[0.4,0.2,0.4],
            "tint":[0.9,0.2,0.1,0.8],
            "visible":false
        }"#;
        let d: Decal = serde_json::from_str(json).unwrap();
        assert_eq!(d.position, [1.0, 2.0, 3.0]);
        assert_eq!(d.rotation_deg, [0.0, 90.0, 0.0]);
        assert_eq!(d.size, [0.4, 0.2, 0.4]);
        assert_eq!(d.tint, [0.9, 0.2, 0.1, 0.8]);
        assert!(!d.visible);
        assert!(d.texture.is_some());
    }

    #[test]
    fn clamps_alpha_through_from_args() {
        let json = r#"{"tint":[1,1,1,5.0]}"#;
        let parsed: Decal = serde_json::from_str(json).unwrap();
        let normalised = Decal::from_args(parsed);
        assert_eq!(normalised.tint[3], 1.0);

        let json = r#"{"tint":[1,1,1,-0.5]}"#;
        let parsed: Decal = serde_json::from_str(json).unwrap();
        let normalised = Decal::from_args(parsed);
        assert_eq!(normalised.tint[3], 0.0);
    }
}
