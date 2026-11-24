// OptionSelect expansion: a settings row. A setting with more than
// two options expands to a dropdown (name + current value + a downward chevron
// under one click region firing the setting's `open`, which opens a floating
// option list at runtime); a setting with two options (an Off/On toggle) expands
// to a `<`/`>` stepper (name + `<` + value + `>` over two regions stepping the
// setting back and forward). The option count is read from the shared
// registry in `concinnity_engine::settings`, so the row form always matches
// the setting the engine will apply.
//
// The value label shows a placeholder here and is bound to the setting, so the
// runtime shows the live value from the first frame. Names are prefixed with the OptionSelect's own name
// so generated elements never collide with hand-authored assets, and every
// element joins the OptionSelect's `screen`.

use crate::world_builder::Diagnostic;
use concinnity_engine::components::{Binding, NamedAction, SettingVerb};
use concinnity_engine::settings::{SettingKey, SettingKind};

use crate::authoring::spec::{AssetSpec, asset};
use crate::ui::elements::left_label;
use crate::ui::expand::{Expand, ExpandContext};
use crate::ui::option_select::schema::OptionSelect;
use crate::ui::row_setting::row_setting;
use crate::ui::screen_scope::scope_to_screen;

// Whether a setting row expands to a dropdown (more than two options, or a
// runtime-enumerated option list like `resolution`) rather than a `<`/`>`
// stepper.
fn is_dropdown(setting: SettingKey) -> bool {
    concinnity_engine::settings::options(setting).is_some_and(|o| o.len() > 2)
        || concinnity_engine::settings::is_dynamic_dropdown(setting)
}

// Where the control group (the `<` button + value + `>`) starts, as a fraction
// of the row width. The name occupies the left part, the control the right.
const CONTROL_FRAC: f32 = 0.42;
// The control group is capped to this fixed width, anchored to the right of the
// row, so on a wide row the control stays a compact column that lines up across
// rows (a narrow row falls back to `CONTROL_FRAC`). Mirrors `world/slider.rs`
// and the settings menu; keep in sync so all rows align.
const MAX_CONTROL_WIDTH: f32 = 360.0;
// Average glyph advance as a fraction of the font pixel size, for placing the
// single-character `<` / `>` glyphs (the built-in font is proportional, so this
// is approximate).
const AVG_ADVANCE_RATIO: f32 = 0.5;
// Padding around the `<` / `>` glyphs and the value, in pixels.
const GLYPH_PAD: f32 = 8.0;
// Placeholder shown until the runtime publishes the live value.
const VALUE_PLACEHOLDER: &str = "--";

impl Expand for OptionSelect {
    fn expand(&self, name: &str, context: &ExpandContext) -> Result<Vec<AssetSpec>, Diagnostic> {
        let setting = row_setting("OptionSelect", name, &self.setting, SettingKind::Cycle)?;
        let font_px = context.font_px(&self.font, self.font_px);
        let mut children = expand_one(name, self, setting, font_px);
        scope_to_screen(&mut children, &self.screen);
        Ok(children)
    }
}

// The Sprite/TextLabel child names an OptionSelect named `base` (for `setting`)
// expands to (the elements a scroll panel reflows + clips with its row). The
// HitRegions are excluded: they have no asset id and are reflowed by position.
// The child set depends on the row form (dropdown vs stepper), so it takes the
// setting key too. Locked to the expansion output by
// `element_names_match_expansion`.
pub(in crate::ui) fn element_names(base: &str, setting: SettingKey) -> Vec<String> {
    if is_dropdown(setting) {
        vec![
            format!("{base}_label"),
            format!("{base}_value"),
            format!("{base}_chevron"),
        ]
    } else {
        vec![
            format!("{base}_label"),
            format!("{base}_prev_glyph"),
            format!("{base}_value"),
            format!("{base}_next_glyph"),
        ]
    }
}

fn expand_one(name: &str, s: &OptionSelect, setting: SettingKey, font_px: f32) -> Vec<AssetSpec> {
    let line_h = font_px * s.text_scale;
    let text_y = s.y + (s.height - line_h) / 2.0;
    let value_name = format!("{name}_value");
    let binding = Binding::Setting(setting);
    let value_label = |x: f32| {
        left_label(
            &value_name,
            VALUE_PLACEHOLDER,
            &s.font,
            x,
            text_y,
            s.value_color,
            s.text_scale,
        )
        .set("text_binding", binding.text())
    };

    let glyph_w = font_px * AVG_ADVANCE_RATIO * s.text_scale;
    let ctrl_x = (s.x + s.width * CONTROL_FRAC).max(s.x + s.width - MAX_CONTROL_WIDTH);
    let right = s.x + s.width;

    // A setting with more than two options expands to a dropdown: the name, the
    // current value left-aligned in the control column, and a downward chevron
    // at the far right, all under one region that opens the floating list. The
    // chevron is an ASCII `v` (the built-in font atlas is ASCII-only).
    if is_dropdown(setting) {
        return vec![
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
            // Current value, left-aligned at the start of the control column.
            value_label(ctrl_x + GLYPH_PAD),
            // Downward chevron flush to the right edge (mirrors the stepper `>`).
            left_label(
                &format!("{name}_chevron"),
                "v",
                &s.font,
                right - glyph_w,
                text_y,
                s.value_color,
                s.text_scale,
            ),
            // One click region over the whole control column opens the list.
            region(
                &format!("{name}_open"),
                Rect {
                    x: ctrl_x,
                    y: s.y,
                    width: right - ctrl_x,
                    height: s.height,
                },
                &value_name,
                s,
                NamedAction::Setting {
                    key: setting,
                    verb: SettingVerb::Open,
                },
            ),
        ];
    }

    // Two options: a `<`/`>` stepper. Layout, left to right: the name fills the
    // left part; then a `<` button, the value (left-aligned and display-only),
    // and a `>` at the far right. Two non-overlapping click regions only -- `<`
    // steps to the previous option, and everything to its right (value + `>`)
    // steps to the next. Overlapping regions must be avoided: UiInputSystem
    // keeps scanning after a setting action fires (it returns no StepResult), so
    // two regions hit by one click would both fire and cancel out.
    let sw = s.stepper_width;
    let next_x = ctrl_x + sw;

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
        // `<` glyph, centered in the prev button.
        left_label(
            &format!("{name}_prev_glyph"),
            "<",
            &s.font,
            ctrl_x + (sw - glyph_w) / 2.0,
            text_y,
            s.value_color,
            s.text_scale,
        ),
        // Value (display only), left-aligned just past the `<` button.
        value_label(next_x + GLYPH_PAD),
        // `>` glyph, flush to the right edge of the row, so it mirrors the
        // left-aligned name and the row's left/right padding stays symmetric.
        left_label(
            &format!("{name}_next_glyph"),
            ">",
            &s.font,
            right - glyph_w,
            text_y,
            s.value_color,
            s.text_scale,
        ),
        // Prev click region (the `<` button).
        region(
            &format!("{name}_prev"),
            Rect {
                x: ctrl_x,
                y: s.y,
                width: sw,
                height: s.height,
            },
            &value_name,
            s,
            NamedAction::Step {
                binding: binding.clone(),
                by: -1,
            },
        ),
        // Next click region (value + `>`).
        region(
            &format!("{name}_next"),
            Rect {
                x: next_x,
                y: s.y,
                width: right - next_x,
                height: s.height,
            },
            &value_name,
            s,
            NamedAction::Step { binding, by: 1 },
        ),
    ]
}

// A settings-row rectangle in overlay coordinates.
#[derive(Clone, Copy)]
struct Rect {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

// Build a HitRegion value scoped to a settings row.
fn region(
    name: &str,
    rect: Rect,
    value_label: &str,
    s: &OptionSelect,
    action: NamedAction,
) -> AssetSpec {
    asset::hit_region(name, [rect.x, rect.y, rect.width, rect.height], action)
        .set("label", value_label)
        .set("hover_color", s.hover_color)
        .set("hover_scale", s.hover_scale)
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
    fn passes_through_without_selects() {
        let mut assets = vec![serde_json::json!({"type":"Window","args":{"$id":"x"}})];
        expand_all::<OptionSelect>(&mut assets).unwrap();
        assert_eq!(assets.len(), 1);
    }

    #[test]
    fn expands_to_name_value_glyphs_and_two_stepper_regions() {
        let mut assets = vec![serde_json::json!({
            "type": "OptionSelect",
            "args": {
                "$id": "opt_vsync",
                "setting": "vsync", "label": "Vsync",
                "x": 100.0, "y": 200.0, "width": 300.0, "stepper_width": 40.0
            }
        })];
        expand_all::<OptionSelect>(&mut assets).unwrap();

        assert!(
            !assets
                .iter()
                .any(|v| registered_type(v) == Some(RegisteredType::OptionSelect))
        );

        let lbl = by_name(&assets, "opt_vsync_label");
        assert_eq!(lbl["type"], "TextLabel");
        assert_eq!(lbl["args"]["content"], "Vsync");
        assert_eq!(lbl["args"]["centered"], false);
        assert_eq!(lbl["args"]["x"], 100.0);

        let val = by_name(&assets, "opt_vsync_value");
        assert_eq!(val["args"]["content"], VALUE_PLACEHOLDER);
        assert_eq!(val["args"]["text_binding"], "setting:vsync");

        // ASCII glyphs (the built-in font atlas is ASCII-only).
        assert_eq!(
            by_name(&assets, "opt_vsync_prev_glyph")["args"]["content"],
            "<"
        );
        assert_eq!(
            by_name(&assets, "opt_vsync_next_glyph")["args"]["content"],
            ">"
        );

        // Two non-overlapping click regions. The row is narrow enough that the
        // fraction wins over the right-anchored cap: ctrl_x =
        // max(100 + 300*0.42, 100 + 300 - 360) = max(226, 40) = 226.
        let prev = by_name(&assets, "opt_vsync_prev");
        assert_eq!(prev["type"], "HitRegion");
        assert_eq!(
            prev["args"]["action"],
            serde_json::json!({"step": {"binding": "setting:vsync", "by": -1}})
        );
        assert_eq!(prev["args"]["label"], "opt_vsync_value");
        assert_eq!(prev["args"]["x"], 226.0);
        assert_eq!(prev["args"]["width"], 40.0);

        let next = by_name(&assets, "opt_vsync_next");
        assert_eq!(
            next["args"]["action"],
            serde_json::json!({"step": {"binding": "setting:vsync", "by": 1}})
        );
        assert_eq!(next["args"]["label"], "opt_vsync_value");
        // next starts where prev ends (ctrl_x + stepper_width = 266) -> no overlap.
        assert_eq!(next["args"]["x"], 266.0);
        assert_eq!(next["args"]["width"], 134.0);
    }

    #[test]
    fn missing_name_is_an_error() {
        let mut assets = vec![serde_json::json!({"type":"OptionSelect","args":{}})];
        assert!(expand_all::<OptionSelect>(&mut assets).is_err());
    }

    // A row with no setting, an unknown one, or a slider's is a build error
    // naming the asset rather than a dead row.
    #[test]
    fn a_blank_unknown_or_slider_setting_is_an_error() {
        for (setting, expected) in [
            ("", "missing `setting`"),
            ("taa", "unknown setting 'taa'"),
            ("exposure", "'exposure' is a Slider setting"),
        ] {
            let mut assets = vec![serde_json::json!({
                "type": "OptionSelect", "args": {"$id": "opt", "setting": setting}
            })];
            let err = expand_all::<OptionSelect>(&mut assets)
                .unwrap_err()
                .to_string();
            assert!(err.contains("OptionSelect 'opt'"), "{err}");
            assert!(err.contains(expected), "{err}");
        }
    }

    #[test]
    fn invalid_args_name_the_select() {
        let mut assets = vec![serde_json::json!({
            "type": "OptionSelect", "args": {"$id": "opt", "width": "wide"}
        })];
        let err = expand_all::<OptionSelect>(&mut assets)
            .unwrap_err()
            .to_string();
        assert!(err.contains("OptionSelect 'opt'"), "{err}");
        assert!(err.contains("invalid args: `width`"), "{err}");
    }

    // A row naming only its setting takes every other field from the type
    // defaults.
    #[test]
    fn select_without_args_uses_type_defaults() {
        let mut assets = vec![serde_json::json!({
            "type": "OptionSelect", "args": {"$id": "opt", "setting": "vsync"}
        })];
        expand_all::<OptionSelect>(&mut assets).unwrap();
        let defaults = OptionSelect::default();
        assert_eq!(by_name(&assets, "opt_label")["args"]["x"], defaults.x);
        assert_eq!(
            by_name(&assets, "opt_label")["args"]["content"],
            defaults.label
        );
    }

    // A setting with more than two options (window_mode has three) expands to a
    // dropdown: name + value + chevron under a single open region, no `<`/`>`.
    #[test]
    fn expands_to_dropdown_with_open_region() {
        let mut assets = vec![serde_json::json!({
            "type": "OptionSelect",
            "args": {
                "$id": "opt_wm",
                "setting": "window_mode", "label": "Window Mode",
                "x": 100.0, "y": 200.0, "width": 300.0
            }
        })];
        expand_all::<OptionSelect>(&mut assets).unwrap();

        assert_eq!(
            by_name(&assets, "opt_wm_label")["args"]["content"],
            "Window Mode"
        );
        assert_eq!(
            by_name(&assets, "opt_wm_value")["args"]["content"],
            VALUE_PLACEHOLDER
        );
        // ASCII chevron (the built-in atlas is ASCII-only), and no stepper glyphs.
        assert_eq!(by_name(&assets, "opt_wm_chevron")["args"]["content"], "v");
        assert!(!assets.iter().any(|v| asset_name(v) == "opt_wm_prev_glyph"));
        assert!(!assets.iter().any(|v| asset_name(v) == "opt_wm_next_glyph"));

        // A single click region opens the floating list; no prev/next regions.
        let open = by_name(&assets, "opt_wm_open");
        assert_eq!(open["type"], "HitRegion");
        assert_eq!(
            open["args"]["action"],
            serde_json::json!({"setting": {"key": "window_mode", "verb": "open"}})
        );
        assert_eq!(open["args"]["label"], "opt_wm_value");
        assert!(!assets.iter().any(|v| asset_name(v) == "opt_wm_prev"));
        assert!(!assets.iter().any(|v| asset_name(v) == "opt_wm_next"));
    }

    // A runtime-enumerated setting (resolution) has no static option table but
    // still expands to a dropdown; the runtime seeds the list from the display.
    #[test]
    fn dynamic_setting_expands_to_dropdown() {
        let mut assets = vec![serde_json::json!({
            "type": "OptionSelect",
            "args": {
                "$id": "opt_res",
                "setting": "resolution", "label": "Resolution",
                "x": 100.0, "y": 200.0, "width": 300.0
            }
        })];
        expand_all::<OptionSelect>(&mut assets).unwrap();
        let open = by_name(&assets, "opt_res_open");
        assert_eq!(
            open["args"]["action"],
            serde_json::json!({"setting": {"key": "resolution", "verb": "open"}})
        );
        assert_eq!(by_name(&assets, "opt_res_chevron")["args"]["content"], "v");
        assert!(!assets.iter().any(|v| asset_name(v) == "opt_res_prev"));
    }

    // `element_names` must list exactly the Sprite/TextLabel children the
    // expansion emits (a scroll panel relies on these to reflow + clip the row),
    // for both the stepper (vsync, two options) and dropdown (window_mode, three)
    // forms.
    #[test]
    fn element_names_match_expansion() {
        for (setting, label) in [("vsync", "Vsync"), ("window_mode", "Window Mode")] {
            let mut assets = vec![serde_json::json!({
                "type": "OptionSelect",
                "args": { "$id": "opt", "setting": setting, "label": label }
            })];
            expand_all::<OptionSelect>(&mut assets).unwrap();
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
            let listed: std::collections::HashSet<String> =
                element_names("opt", SettingKey::parse(setting).unwrap())
                    .into_iter()
                    .collect();
            assert_eq!(listed, emitted, "element_names drifted for '{setting}'");
        }
    }
    // Every generated element joins the OptionSelect's screen; one that names
    // none leaves them screen-less.
    #[test]
    fn generated_elements_join_the_selects_screen() {
        let row = |args: serde_json::Value| {
            let mut assets = vec![serde_json::json!({
                "type": "OptionSelect", "args": crate::authoring::world::args_with_id(args, "opt")
            })];
            expand_all::<OptionSelect>(&mut assets).unwrap();
            assets
        };

        let assets = row(serde_json::json!({"setting": "vsync", "screen": "video"}));
        let scoped: Vec<&serde_json::Value> = assets
            .iter()
            .filter(|v| asset_name(v).starts_with("opt_"))
            .collect();
        assert!(!scoped.is_empty(), "the select expands to elements");
        for element in scoped {
            assert_eq!(
                element["args"]["screen"],
                "video",
                "{}",
                asset_name(element)
            );
        }

        let assets = row(serde_json::json!({"setting": "vsync"}));
        assert!(
            assets
                .iter()
                .filter(|v| asset_name(v).starts_with("opt_"))
                .all(|v| v["args"].get("screen").is_none())
        );
    }
}
