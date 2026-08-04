// Sprite element builder.

use crate::spec::AssetSpec;

// A rectangular Sprite at `rect` ([x, y, w, h], window pixels) with `tint`
// (RGBA). Visibility, texture, and corner radius keep their defaults; chain
// `.set("visible", ...)` to override.
pub fn sprite(name: impl Into<String>, rect: [f32; 4], tint: [f32; 4]) -> AssetSpec {
    AssetSpec::new(name, "Sprite")
        .set("x", rect[0])
        .set("y", rect[1])
        .set("width", rect[2])
        .set("height", rect[3])
        .set("tint", tint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::ArgValue;

    #[test]
    fn sprite_sets_rect_and_tint() {
        let s = sprite("bg", [10.0, 20.0, 100.0, 40.0], [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(s.asset_type, "Sprite");
        let field = |k: &str| s.fields.iter().find(|(key, _)| key == k).map(|(_, v)| v);
        assert_eq!(field("x"), Some(&ArgValue::Float(10.0)));
        assert_eq!(field("height"), Some(&ArgValue::Float(40.0)));
        assert_eq!(
            field("tint"),
            Some(&ArgValue::floats(&[1.0, 0.0, 0.0, 1.0]))
        );
    }
}
