// src/world/panel.rs
// Build-time expansion: Panel -> a background Sprite + (when a title is set) a
// heading TextLabel inset from the top-left. Names are prefixed with the Panel's
// own name so generated elements stay scoped to its Screen via the build
// pipeline's `<screen>_*` rule and never collide with hand-authored assets.

use super::expand::{asset_name, type_norm};
use super::ui_spec::label_value;
use concinnity_asset::cook::Panel;
use concinnity_world::spec::{asset, spec_to_value};

// Replace every Panel asset with the concrete UI assets it expands to.
pub(crate) fn expand_panels(assets: &mut Vec<serde_json::Value>) -> Result<(), String> {
    if !assets.iter().any(|v| type_norm(v) == "panel") {
        return Ok(());
    }

    let mut result: Vec<serde_json::Value> = Vec::new();
    for value in assets.drain(..) {
        if type_norm(&value) != "panel" {
            result.push(value);
            continue;
        }

        let name = asset_name(&value);
        if name.is_empty() {
            return Err("Panel: missing `name`".to_string());
        }
        let args = value
            .get("args")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        let panel: Panel = serde_json::from_value(args)
            .map_err(|e| format!("Panel '{}': invalid args: {}", name, e))?;

        result.extend(expand_one(&name, &panel));
    }

    *assets = result;
    Ok(())
}

fn expand_one(name: &str, p: &Panel) -> Vec<serde_json::Value> {
    let mut out = vec![spec_to_value(
        &asset::sprite(
            format!("{}_bg", name),
            [p.x, p.y, p.width, p.height],
            p.color,
        )
        .set("corner_radius", p.corner_radius),
    )];
    if !p.title.is_empty() {
        out.push(label_value(
            &format!("{}_title", name),
            &p.title,
            &p.title_font,
            p.x + p.padding,
            p.y + p.padding,
            p.title_color,
            p.title_scale,
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn by_name<'a>(assets: &'a [serde_json::Value], name: &str) -> &'a serde_json::Value {
        assets
            .iter()
            .find(|v| asset_name(v) == name)
            .unwrap_or_else(|| panic!("no asset named {name}"))
    }

    #[test]
    fn passes_through_without_panels() {
        let mut assets = vec![serde_json::json!({"name":"x","type":"Window","args":{}})];
        expand_panels(&mut assets).unwrap();
        assert_eq!(assets.len(), 1);
    }

    #[test]
    fn titled_panel_expands_to_bg_and_title() {
        let mut assets = vec![serde_json::json!({
            "name": "pause_card",
            "type": "Panel",
            "args": { "title": "Paused", "x": 440.0, "y": 220.0, "width": 400.0, "height": 280.0 }
        })];
        expand_panels(&mut assets).unwrap();

        assert!(!assets.iter().any(|v| type_norm(v) == "panel"));

        let bg = by_name(&assets, "pause_card_bg");
        assert_eq!(bg["type"], "Sprite");
        assert_eq!(bg["args"]["x"], 440.0);
        assert_eq!(bg["args"]["width"], 400.0);

        let title = by_name(&assets, "pause_card_title");
        assert_eq!(title["type"], "TextLabel");
        assert_eq!(title["args"]["content"], "Paused");
        assert_eq!(title["args"]["centered"], false);
    }

    #[test]
    fn untitled_panel_emits_only_bg() {
        let mut assets = vec![serde_json::json!({
            "name": "card", "type": "Panel", "args": { "width": 200.0, "height": 120.0 }
        })];
        expand_panels(&mut assets).unwrap();
        assert_eq!(by_name(&assets, "card_bg")["type"], "Sprite");
        assert!(!assets.iter().any(|v| asset_name(v) == "card_title"));
    }

    #[test]
    fn missing_name_is_an_error() {
        let mut assets = vec![serde_json::json!({"type":"Panel","args":{}})];
        assert!(expand_panels(&mut assets).is_err());
    }

    // Other assets keep their place in the list while the Panels around them
    // expand.
    #[test]
    fn other_assets_survive_a_panel_expansion() {
        let mut assets = vec![
            serde_json::json!({"name":"win","type":"Window","args":{}}),
            serde_json::json!({"name":"card","type":"Panel","args":{"width":10.0}}),
        ];
        expand_panels(&mut assets).unwrap();
        assert_eq!(assets[0]["name"], "win");
        assert_eq!(assets[1]["name"], "card_bg");
    }

    // A Panel with no args at all is the fully defaulted panel, not an error.
    #[test]
    fn panel_without_args_uses_type_defaults() {
        let mut assets = vec![serde_json::json!({"name":"card","type":"Panel"})];
        expand_panels(&mut assets).unwrap();
        let defaults = Panel::default();
        assert_eq!(assets.len(), 1);
        assert_eq!(by_name(&assets, "card_bg")["args"]["width"], defaults.width);
    }

    #[test]
    fn invalid_args_name_the_panel_and_the_field() {
        let mut assets = vec![serde_json::json!({
            "name": "card", "type": "Panel", "args": {"width": "wide"}
        })];
        let err = expand_panels(&mut assets).unwrap_err();
        assert!(err.contains("Panel 'card'"), "{err}");
        assert!(err.contains("invalid args"), "{err}");
    }
}
