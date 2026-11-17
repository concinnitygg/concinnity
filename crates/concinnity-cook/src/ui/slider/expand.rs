// Slider expansion: a name TextLabel + a value TextLabel + a
// track Sprite + a handle Sprite + a HitRegion that fires a
// setting `drag` action (carrying the handle so the runtime can move it).
//
// The handle and value label show a placeholder position here; the runtime
// corrects them to the live value on the first frame and while dragging. Names
// are prefixed with the Slider's own name so generated elements never collide
// with hand-authored assets, and every element joins the Slider's `screen`.

use crate::world_builder::Diagnostic;
use concinnity_engine::components::{NamedAction, SettingVerb};
use concinnity_engine::settings::{SettingKey, SettingKind};

use crate::authoring::spec::{AssetSpec, asset};
use crate::ui::elements::{left_label, sprite};
use crate::ui::expand::{Expand, ExpandContext};
use crate::ui::row_setting::row_setting;
use crate::ui::screen_scope::scope_to_screen;
use crate::ui::slider::schema::Slider;

// Where the control group (track + value) starts, as a fraction of the row
// width. The name occupies the left part, the control the right. Matches
// `OptionSelect` so slider and cycle rows line up in a shared menu.
const CONTROL_FRAC: f32 = 0.42;
// The control group is capped to this fixed width, anchored to the right of the
// row, so on a wide row the control stays a compact column aligned with the
// cycle rows (a narrow row falls back to `CONTROL_FRAC`). Mirrors
// `OptionSelect` and the settings menu; keep in sync.
const MAX_CONTROL_WIDTH: f32 = 360.0;
// Average glyph advance as a fraction of the font pixel size (the built-in font
// is proportional, so this is approximate). Used to reserve room for the value.
const AVG_ADVANCE_RATIO: f32 = 0.5;
// Characters of value text to reserve at the right (e.g. "+1.0 EV").
const VALUE_CHARS: f32 = 7.0;
// Placeholder shown until the runtime sets the live value on the first frame.
const VALUE_PLACEHOLDER: &str = "--";

impl Expand for Slider {
    fn expand(&self, name: &str, context: &ExpandContext) -> Result<Vec<AssetSpec>, Diagnostic> {
        let setting = row_setting("Slider", name, &self.setting, SettingKind::Slider)?;
        let font_px = context.font_px(&self.font, self.font_px);
        let mut children = expand_one(name, self, setting, font_px);
        scope_to_screen(&mut children, &self.screen);
        Ok(children)
    }
}

// The Sprite/TextLabel child names a Slider named `base` expands to (the
// elements a scroll panel reflows + clips with its row). The drag HitRegion is
// excluded: it has no asset id and is reflowed by position. Locked to the
// expansion output by `element_names_match_expansion`.
pub(in crate::ui) fn element_names(base: &str) -> Vec<String> {
    vec![
        format!("{base}_label"),
        format!("{base}_value"),
        format!("{base}_track"),
        format!("{base}_handle"),
    ]
}

fn expand_one(name: &str, s: &Slider, setting: SettingKey, font_px: f32) -> Vec<AssetSpec> {
    let glyph_w = font_px * AVG_ADVANCE_RATIO * s.text_scale;
    let line_h = font_px * s.text_scale;
    let text_y = s.y + (s.height - line_h) / 2.0;

    // Layout, left to right: the name fills the left part; then a track bar, and
    // the value text at the far right. The handle rides on the track.
    let ctrl_x = (s.x + s.width * CONTROL_FRAC).max(s.x + s.width - MAX_CONTROL_WIDTH);
    let right = s.x + s.width;
    let value_w = glyph_w * VALUE_CHARS;
    let value_x = right - value_w;
    let gap = glyph_w;
    let track_x = ctrl_x;
    let track_w = (value_x - gap - track_x).max(20.0);
    let track_h = (s.height * 0.12).max(4.0);
    let track_y = s.y + (s.height - track_h) / 2.0;
    let handle_w = (s.height * 0.45).max(10.0);
    let handle_h = (s.height * 0.7).max(16.0);
    let handle_y = s.y + (s.height - handle_h) / 2.0;

    let value_name = format!("{name}_value");
    let handle_name = format!("{name}_handle");

    vec![
        // Name (left).
        left_label(
            &format!("{name}_label"),
            &s.label,
            &s.font,
            s.x,
            text_y,
            s.text_color,
            s.text_scale,
        ),
        // Track bar (static background).
        sprite(
            &format!("{name}_track"),
            track_x,
            track_y,
            track_w,
            track_h,
            s.track_color,
        ),
        // Handle (placed at the left here; the runtime moves it to the live
        // value's fraction on the first frame and while dragging).
        sprite(
            &handle_name,
            track_x,
            handle_y,
            handle_w,
            handle_h,
            s.handle_color,
        ),
        // Value text (display only) at the far right.
        left_label(
            &value_name,
            VALUE_PLACEHOLDER,
            &s.font,
            value_x,
            text_y,
            s.value_color,
            s.text_scale,
        ),
        // Drag region spanning the track over the full row height. `label`
        // points at the value text (the runtime updates it); `drag_handle`
        // points at the handle sprite (the runtime moves it).
        asset::hit_region(
            format!("{name}_drag"),
            [track_x, s.y, track_w, s.height],
            NamedAction::Setting {
                key: setting,
                verb: SettingVerb::Drag,
            },
        )
        .set("label", value_name)
        .set("drag_handle", handle_name),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authoring::registry::RegisteredType;
    use crate::build_only::expand::{asset_name, registered_type};
    use crate::ui::expand::expand_all;

    fn by_name<'a>(assets: &'a [serde_json::Value], name: &str) -> &'a serde_json::Value {
        assets
            .iter()
            .find(|v| asset_name(v) == name)
            .unwrap_or_else(|| panic!("no asset named {name}"))
    }

    #[test]
    fn passes_through_without_sliders() {
        let mut assets = vec![serde_json::json!({"type":"Window","args":{"$id":"x"}})];
        expand_all::<Slider>(&mut assets).unwrap();
        assert_eq!(assets.len(), 1);
    }

    #[test]
    fn expands_to_label_value_track_handle_and_drag_region() {
        let mut assets = vec![serde_json::json!({
            "type": "Slider",
            "args": {
                "$id": "sld_exposure",
                "setting": "exposure", "label": "Exposure",
                "x": 100.0, "y": 200.0, "width": 400.0, "height": 48.0
            }
        })];
        expand_all::<Slider>(&mut assets).unwrap();

        assert!(
            !assets
                .iter()
                .any(|v| registered_type(v) == Some(RegisteredType::Slider))
        );

        let lbl = by_name(&assets, "sld_exposure_label");
        assert_eq!(lbl["type"], "TextLabel");
        assert_eq!(lbl["args"]["content"], "Exposure");
        assert_eq!(lbl["args"]["centered"], false);
        assert_eq!(lbl["args"]["x"], 100.0);

        let val = by_name(&assets, "sld_exposure_value");
        assert_eq!(val["type"], "TextLabel");
        assert_eq!(val["args"]["content"], VALUE_PLACEHOLDER);

        assert_eq!(by_name(&assets, "sld_exposure_track")["type"], "Sprite");
        assert_eq!(by_name(&assets, "sld_exposure_handle")["type"], "Sprite");

        // The drag region carries the value label, the handle, and the action.
        let drag = by_name(&assets, "sld_exposure_drag");
        assert_eq!(drag["type"], "HitRegion");
        assert_eq!(
            drag["args"]["action"],
            serde_json::json!({"setting": {"key": "exposure", "verb": "drag"}})
        );
        assert_eq!(drag["args"]["label"], "sld_exposure_value");
        assert_eq!(drag["args"]["drag_handle"], "sld_exposure_handle");
        // The region spans the track, at the full row height.
        assert_eq!(drag["args"]["y"], 200.0);
        assert_eq!(drag["args"]["height"], 48.0);
    }

    #[test]
    fn missing_name_is_an_error() {
        let mut assets = vec![serde_json::json!({"type":"Slider","args":{}})];
        assert!(expand_all::<Slider>(&mut assets).is_err());
    }

    // A Slider with no setting, an unknown one, or a cycle row's is a build error
    // naming the asset rather than a dead row.
    #[test]
    fn a_blank_unknown_or_cycle_setting_is_an_error() {
        for (setting, expected) in [
            ("", "missing `setting`"),
            ("bloom", "unknown setting 'bloom'"),
            ("vsync", "'vsync' is a Cycle setting"),
        ] {
            let mut assets = vec![serde_json::json!({
                "type": "Slider", "args": {"$id": "sld", "setting": setting}
            })];
            let err = expand_all::<Slider>(&mut assets).unwrap_err().to_string();
            assert!(err.contains("Slider 'sld'"), "{err}");
            assert!(err.contains(expected), "{err}");
        }
    }

    #[test]
    fn invalid_args_name_the_slider() {
        let mut assets = vec![serde_json::json!({
            "type": "Slider", "args": {"$id": "sld", "width": "wide"}
        })];
        let err = expand_all::<Slider>(&mut assets).unwrap_err().to_string();
        assert!(err.contains("Slider 'sld'"), "{err}");
        assert!(err.contains("invalid args: `width`"), "{err}");
    }

    // A Slider naming only its setting takes every other field from the type
    // defaults.
    #[test]
    fn slider_without_args_uses_type_defaults() {
        let mut assets = vec![serde_json::json!({
            "type": "Slider", "args": {"$id": "sld", "setting": "exposure"}
        })];
        expand_all::<Slider>(&mut assets).unwrap();
        let defaults = Slider::default();
        assert_eq!(by_name(&assets, "sld_label")["args"]["x"], defaults.x);
        assert_eq!(
            by_name(&assets, "sld_label")["args"]["content"],
            defaults.label
        );
    }

    // `element_names` must list exactly the Sprite/TextLabel children the
    // expansion emits (a scroll panel relies on these to reflow + clip the row).
    #[test]
    fn element_names_match_expansion() {
        let mut assets = vec![serde_json::json!({
            "type": "Slider",
            "args": { "$id": "sld", "setting": "exposure", "label": "Exposure" }
        })];
        expand_all::<Slider>(&mut assets).unwrap();
        let emitted: std::collections::HashSet<String> = assets
            .iter()
            .filter(|v| {
                matches!(
                    registered_type(v),
                    Some(RegisteredType::TextLabel | RegisteredType::Sprite)
                )
            })
            .map(asset_name)
            .collect();
        let listed: std::collections::HashSet<String> = element_names("sld").into_iter().collect();
        assert_eq!(listed, emitted, "element_names drifted from the expansion");
    }
    // Every generated element joins the Slider's screen; a Slider that names
    // none leaves them screen-less.
    #[test]
    fn generated_elements_join_the_sliders_screen() {
        let row = |args: serde_json::Value| {
            let mut assets = vec![serde_json::json!({
                "type": "Slider", "args": crate::authoring::world::args_with_id(args, "sld")
            })];
            expand_all::<Slider>(&mut assets).unwrap();
            assets
        };

        let assets = row(serde_json::json!({"setting": "exposure", "screen": "video"}));
        for suffix in ["label", "value", "track", "handle", "drag"] {
            assert_eq!(
                by_name(&assets, &format!("sld_{suffix}"))["args"]["screen"],
                "video",
                "sld_{suffix}"
            );
        }

        let assets = row(serde_json::json!({"setting": "exposure"}));
        assert!(
            by_name(&assets, "sld_label")["args"]
                .get("screen")
                .is_none()
        );
    }
}
