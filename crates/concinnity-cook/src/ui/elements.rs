// The overlay elements the UI shorthands expand into, built once here as typed
// specs so every expander emits the same element shapes.

use crate::authoring::spec::{AssetSpec, asset};

// A settings/menu TextLabel with `centered` pinned false: the default-font
// pass would otherwise recenter a font-carrying label onto the viewport center,
// stacking every menu label. Left-aligned; see `centered_label` for the
// real-metrics centered variant.
pub(super) fn left_label(
    name: &str,
    content: &str,
    font: &str,
    x: f32,
    y: f32,
    color: [f32; 3],
    scale: f32,
) -> AssetSpec {
    asset::menu_label(name, content, font, [x, y], color, scale)
}

// A menu label horizontally centered on `x` with real font metrics (`align`
// center), for the vertically-stacked menu items, headings, and Back button so
// differing label widths share one exact center.
pub(super) fn centered_label(
    name: &str,
    content: &str,
    font: &str,
    x: f32,
    y: f32,
    color: [f32; 3],
    scale: f32,
) -> AssetSpec {
    asset::menu_label(name, content, font, [x, y], color, scale).set("align", "center")
}

// A solid-colored rectangle Sprite: settings row cards, slider tracks/handles,
// scrollbar chrome, menu backdrops.
pub(super) fn sprite(
    name: &str,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    tint: [f32; 4],
) -> AssetSpec {
    asset::sprite(name, [x, y, width, height], tint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authoring::spec::spec_to_value;

    #[test]
    fn left_label_pins_font_scale_and_uncentered() {
        let v = spec_to_value(&left_label(
            "row",
            "Vsync",
            "m_font",
            100.0,
            200.0,
            [1.0, 1.0, 1.0],
            0.5,
        ));
        assert_eq!(v["args"]["$id"], "row");
        assert_eq!(v["type"], "TextLabel");
        assert_eq!(v["args"]["content"], "Vsync");
        assert_eq!(v["args"]["font"], "m_font");
        assert_eq!(v["args"]["x"], 100.0);
        assert_eq!(v["args"]["scale"], 0.5);
        assert_eq!(v["args"]["centered"], false);
        assert!(v["args"].get("align").is_none());
    }

    #[test]
    fn centered_label_adds_center_align() {
        let v = spec_to_value(&centered_label(
            "t",
            "Paused",
            "m_font",
            640.0,
            40.0,
            [1.0, 1.0, 1.0],
            1.0,
        ));
        assert_eq!(v["args"]["align"], "center");
        assert_eq!(v["args"]["centered"], false);
    }

    #[test]
    fn sprite_sets_rect_and_tint() {
        let v = spec_to_value(&sprite(
            "bg",
            10.0,
            20.0,
            300.0,
            40.0,
            [0.5, 0.25, 0.125, 0.75],
        ));
        assert_eq!(v["type"], "Sprite");
        assert_eq!(v["args"]["x"], 10.0);
        assert_eq!(v["args"]["width"], 300.0);
        assert_eq!(
            v["args"]["tint"],
            serde_json::json!([0.5, 0.25, 0.125, 0.75])
        );
    }
}
