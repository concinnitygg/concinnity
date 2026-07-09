// Text element builders: a static label and an editable text input.

use crate::spec::AssetSpec;
use alloc::string::String;

// A TextLabel of `content` at `pos` ([x, y], window pixels), coloured `color`
// (RGB) and aligned `align` (`"left"` / `"center"` / `"right"`). The font is a
// reference the caller supplies (on the world line or after materializing).
pub fn text_label(
    name: impl Into<String>,
    content: impl Into<String>,
    pos: [f32; 2],
    color: [f32; 3],
    align: &'static str,
) -> AssetSpec {
    AssetSpec::new(name, "TextLabel")
        .set("content", content.into())
        .set("x", pos[0])
        .set("y", pos[1])
        .set("color", color)
        .set("align", align)
}

// An editable TextInput showing `placeholder` when empty. Geometry, colours, and
// length cap keep their defaults; chain `.set(...)` for the rest.
pub fn text_input(name: impl Into<String>, placeholder: impl Into<String>) -> AssetSpec {
    AssetSpec::new(name, "TextInput").set("placeholder", placeholder.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::ArgValue;
    use alloc::string::ToString;

    #[test]
    fn text_label_sets_content_position_and_align() {
        let l = text_label("save", "SAVE", [44.0, 34.0], [1.0, 1.0, 1.0], "center");
        assert_eq!(l.asset_type, "TextLabel");
        let field = |k: &str| l.fields.iter().find(|(key, _)| key == k).map(|(_, v)| v);
        assert_eq!(field("content"), Some(&ArgValue::Str("SAVE".to_string())));
        assert_eq!(field("align"), Some(&ArgValue::Str("center".to_string())));
        assert_eq!(field("x"), Some(&ArgValue::Float(44.0)));
    }

    #[test]
    fn text_input_sets_placeholder() {
        let t = text_input("name_field", "name");
        assert_eq!(t.asset_type, "TextInput");
        assert_eq!(
            t.fields
                .iter()
                .find(|(k, _)| k == "placeholder")
                .map(|(_, v)| v),
            Some(&ArgValue::Str("name".to_string()))
        );
    }
}
