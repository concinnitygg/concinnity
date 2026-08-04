// Text element builders: a Font, a static label, a settings-menu label, and an
// editable text input.

use crate::spec::AssetSpec;

// A Font compiled at `size_px` pixels. With no `path` set it compiles from the
// built-in font; chain `.set("path", ...)` for a user typeface.
pub fn font(name: impl Into<String>, size_px: u32) -> AssetSpec {
    AssetSpec::new(name, "Font").set("size_px", size_px)
}

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

// A settings/menu TextLabel: `content` in `font` at `pos` ([x, y], window
// pixels), coloured `color` (RGB) and scaled by `scale`. `centered` is pinned
// false so the engine's default-font pass never recenters a menu label onto the
// viewport centre (the menu lays labels out itself); chain
// `.set("align", "center")` for a label centred on `x` with real font metrics.
pub fn menu_label(
    name: impl Into<String>,
    content: impl Into<String>,
    font: impl Into<String>,
    pos: [f32; 2],
    color: [f32; 3],
    scale: f32,
) -> AssetSpec {
    AssetSpec::new(name, "TextLabel")
        .set("content", content.into())
        .set("font", font.into())
        .set("x", pos[0])
        .set("y", pos[1])
        .set("color", color)
        .set("scale", scale)
        .set("centered", false)
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

    #[test]
    fn font_sets_size() {
        let f = font("menu_font", 48);
        assert_eq!(f.asset_type, "Font");
        assert_eq!(
            f.fields
                .iter()
                .find(|(k, _)| k == "size_px")
                .map(|(_, v)| v),
            Some(&ArgValue::Int(48))
        );
    }

    #[test]
    fn menu_label_sets_font_scale_and_pins_centered_false() {
        let l = menu_label(
            "row",
            "Vsync",
            "m_font",
            [100.0, 200.0],
            [1.0, 1.0, 1.0],
            0.5,
        );
        assert_eq!(l.asset_type, "TextLabel");
        let field = |k: &str| l.fields.iter().find(|(key, _)| key == k).map(|(_, v)| v);
        assert_eq!(field("content"), Some(&ArgValue::Str("Vsync".to_string())));
        assert_eq!(field("font"), Some(&ArgValue::Str("m_font".to_string())));
        assert_eq!(field("scale"), Some(&ArgValue::Float(0.5)));
        assert_eq!(field("centered"), Some(&ArgValue::Bool(false)));
        // No align by default; a centred variant chains `.set("align", ...)`.
        assert!(field("align").is_none());
    }

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
