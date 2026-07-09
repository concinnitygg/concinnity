// Shared UI-expansion helpers.
//
// The build-time UI shorthands (MainMenu, OptionSelect, Slider, Panel, Story)
// all expand into the same handful of overlay primitives: settings labels,
// solid sprites, and a font-size lookup for vertical centering. These once lived
// as byte-identical copies in each expander; they now build the typed
// `concinnity_templates::asset` specs once here and convert them to the world's
// `serde_json::Value` with the core bridge, so the element shapes are
// single-sourced and match the specs the editor and world templates use.

use std::collections::HashMap;

use super::expand::{asset_name, type_norm};
use crate::assets::Font;
use concinnity_core::template_spec::spec_to_value;
use concinnity_templates::asset;

// A settings/menu TextLabel value with `centered` pinned false: the default-font
// pass would otherwise recenter a font-carrying label onto the viewport centre,
// stacking every menu label. Left-aligned; see `centered_label` for the
// real-metrics centered variant.
pub(crate) fn label_value(
    name: &str,
    content: &str,
    font: &str,
    x: f32,
    y: f32,
    color: [f32; 3],
    scale: f32,
) -> serde_json::Value {
    spec_to_value(&asset::menu_label(
        name,
        content,
        font,
        [x, y],
        color,
        scale,
    ))
}

// A menu label horizontally centered on `x` with real font metrics (`align`
// center), for the vertically-stacked menu items, headings, and Back button so
// differing label widths share one exact center.
pub(crate) fn centered_label(
    name: &str,
    content: &str,
    font: &str,
    x: f32,
    y: f32,
    color: [f32; 3],
    scale: f32,
) -> serde_json::Value {
    spec_to_value(
        &asset::menu_label(name, content, font, [x, y], color, scale).set("align", "center"),
    )
}

// A solid-coloured rectangle Sprite: settings row cards, slider tracks/handles,
// scrollbar chrome, menu backdrops.
pub(crate) fn sprite(
    name: &str,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    tint: [f32; 4],
) -> serde_json::Value {
    spec_to_value(&asset::sprite(name, [x, y, width, height], tint))
}

// Map of declared Font name to its pixel size, so an expander can vertically
// center row text against the font the world actually declares (falling back to
// the type default for an undeclared reference).
pub(crate) fn font_sizes(assets: &[serde_json::Value]) -> HashMap<String, f32> {
    let mut out = HashMap::new();
    for v in assets {
        if type_norm(v) != "font" {
            continue;
        }
        let name = asset_name(v);
        if name.is_empty() {
            continue;
        }
        let px = v
            .get("args")
            .and_then(|a| a.get("size_px"))
            .and_then(|x| x.as_u64())
            .unwrap_or_else(|| Font::default().size_px as u64) as f32;
        out.insert(name, px);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_value_pins_font_scale_and_uncentered() {
        let v = label_value("row", "Vsync", "m_font", 100.0, 200.0, [1.0, 1.0, 1.0], 0.5);
        assert_eq!(v["name"], "row");
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
        let v = centered_label("t", "Paused", "m_font", 640.0, 40.0, [1.0, 1.0, 1.0], 1.0);
        assert_eq!(v["args"]["align"], "center");
        assert_eq!(v["args"]["centered"], false);
    }

    #[test]
    fn sprite_sets_rect_and_tint() {
        let v = sprite("bg", 10.0, 20.0, 300.0, 40.0, [0.5, 0.25, 0.125, 0.75]);
        assert_eq!(v["type"], "Sprite");
        assert_eq!(v["args"]["x"], 10.0);
        assert_eq!(v["args"]["width"], 300.0);
        assert_eq!(
            v["args"]["tint"],
            serde_json::json!([0.5, 0.25, 0.125, 0.75])
        );
    }

    #[test]
    fn font_sizes_reads_declared_sizes_and_defaults_unknown() {
        let assets = vec![
            serde_json::json!({"name":"a","type":"Font","args":{"size_px":32}}),
            serde_json::json!({"name":"b","type":"Font","args":{}}),
        ];
        let map = font_sizes(&assets);
        assert_eq!(map["a"], 32.0);
        assert_eq!(map["b"], Font::default().size_px as f32);
    }
}
