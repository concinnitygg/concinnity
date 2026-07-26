// src/world/companion.rs
// Inject companion assets implied by the presence of other assets.
//
// Each renderable asset's companions are declared in `companion_specs` and
// dispatched there by normalized type name. This module applies the resulting
// specs to a world JSONL value list. Two passes:
//
//   1. Injection runs to a fixed point. Each round snapshots the current
//      world, asks every declared asset for its companion specs, then filters
//      out specs whose `asset_type` already appears in the world or whose
//      `name` was already collected this round (by-name dedup within a round
//      so a single asset can request multiple companions of the same type,
//      e.g. GraphicsSystem's default Shader).
//
//   2. The default-font pass (`apply_default_font`) runs once with the final
//      world: when there are TextLabels but no Font, it injects one from the
//      engine's bundled font and points empty-`font` labels at it.

use super::companion_specs::{CompanionSpec, companions_for};
use super::expand::ExpandReport;
use std::collections::HashSet;

// Same normalization the rest of the codebase uses for type-name dedup:
// lowercase + strip underscores. Keeps "Camera3DSystem" / "camera3d_system"
// from being treated as different types.
fn type_norm_str(s: &str) -> String {
    s.to_lowercase().replace('_', "")
}

fn asset_type_norm(v: &serde_json::Value) -> String {
    v.get("type")
        .and_then(|t| t.as_str())
        .map(type_norm_str)
        .unwrap_or_default()
}

// Record a skipped companion the world provides itself under the spec's own
// name: that asset is the user's copy of the companion, and a listing needs to
// say so rather than leave the companion unaccounted for. A spec skipped because
// the world has that type under some OTHER name is not an override of this
// asset, and an injection from an earlier round is our own, so neither counts.
fn record_if_overridden(
    snapshot: &[serde_json::Value],
    report: &mut ExpandReport,
    spec: &CompanionSpec,
) {
    let claimed = snapshot
        .iter()
        .any(|v| v.get("name").and_then(|n| n.as_str()) == Some(spec.name));
    let ours = report.injected.iter().any(|i| i.name == spec.name);
    if claimed && !ours {
        report.record_shadowed(spec.name, spec.asset_type, "companion");
    }
}

// Dispatch a companion lookup for one asset by its normalized type name.
fn companions_for_type(
    asset_type: &str,
    args: &serde_json::Value,
    world: &[serde_json::Value],
) -> Vec<CompanionSpec> {
    companions_for(&type_norm_str(asset_type), args, world)
}

// Pixel size of the auto-injected default font. Deliberately larger than a
// hand-declared Font's default `size_px`: this is the big, HUD-style fallback
// face that font-less labels render with.
const DEFAULT_FONT_SIZE: u32 = 48;

// Ensure a font-less world still has a usable Font. When any TextLabel exists
// but no Font is declared, inject one compiled from the engine's bundled default
// font (a Font with no `path`), named after that file exactly as `cn add` would
// name it. Then point every empty-`font` label at it (and default such labels to
// centered, HUD-style). A world that declares its own Font pre-empts all of this.
fn apply_default_font(assets: &mut Vec<serde_json::Value>, report: &mut ExpandReport) {
    let has_text = assets.iter().any(|v| asset_type_norm(v) == "textlabel");
    if !has_text {
        return;
    }
    let font_name = super::asset_name_from_path(crate::font::BUILTIN_FONT_FILE);
    let has_font = assets.iter().any(|v| asset_type_norm(v) == "font");
    if !has_font {
        let args = serde_json::json!({ "size_px": DEFAULT_FONT_SIZE });
        assets.push(serde_json::json!({
            "name": font_name,
            "type": "Font",
            "args": args.clone(),
        }));
        report.record(&font_name, "Font", args, "default_font");
    }
    // Patch only if the default actually landed (nothing pre-empted it).
    let default_present = assets.iter().any(|v| {
        asset_type_norm(v) == "font"
            && v.get("name").and_then(|n| n.as_str()) == Some(font_name.as_str())
    });
    if !default_present {
        return;
    }
    for v in assets.iter_mut() {
        if asset_type_norm(v) != "textlabel" {
            continue;
        }
        let Some(args) = v.get_mut("args").and_then(|a| a.as_object_mut()) else {
            continue;
        };
        let font_empty = args
            .get("font")
            .and_then(|x| x.as_str())
            .map(|s| s.is_empty())
            .unwrap_or(true);
        if font_empty {
            args.insert(
                "font".to_string(),
                serde_json::Value::String(font_name.clone()),
            );
            args.entry("centered")
                .or_insert_with(|| serde_json::json!(true));
        }
    }
}

pub(crate) fn inject_companions(assets: &mut Vec<serde_json::Value>, report: &mut ExpandReport) {
    loop {
        // Snapshot the world for this round. New companions added in this
        // round only enter the visible set on the next iteration; that keeps
        // multi-spec batches from
        // shadowing each other through the per-spec type-dedup.
        let snapshot = assets.clone();
        let present_types: HashSet<String> = snapshot.iter().map(asset_type_norm).collect();

        // Collect every spec implied by every declared asset.
        let mut candidates: Vec<CompanionSpec> = Vec::new();
        for value in &snapshot {
            let Some(t) = value.get("type").and_then(|s| s.as_str()) else {
                continue;
            };
            let args = value
                .get("args")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            candidates.extend(companions_for_type(t, &args, &snapshot));
        }

        // Apply: skip a spec whose asset_type already exists in the
        // pre-round world. Within the round, dedup by `name` so two specs
        // sharing a type (e.g. the default shader stages) both pass.
        let mut seen_names: HashSet<String> = HashSet::new();
        let mut to_inject = Vec::new();
        for spec in candidates {
            if present_types.contains(&type_norm_str(spec.asset_type)) {
                record_if_overridden(&snapshot, report, &spec);
                continue;
            }
            if !seen_names.insert(spec.name.to_string()) {
                continue;
            }
            to_inject.push(spec);
        }

        if to_inject.is_empty() {
            break;
        }
        for spec in to_inject {
            assets.push(serde_json::json!({
                "name": spec.name,
                "type": spec.asset_type,
                "args": spec.args.clone(),
            }));
            report.record(spec.name, spec.asset_type, spec.args, "companion");
        }
    }

    // Default-font pass: ensure font-less labels have a Font to reference.
    apply_default_font(assets, report);
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test shim: run injection with a throwaway report.
    fn inject(assets: &mut Vec<serde_json::Value>) {
        let mut report = ExpandReport::default();
        inject_companions(assets, &mut report);
    }

    fn type_norm(v: &serde_json::Value) -> String {
        asset_type_norm(v)
    }

    // A world that declares a companion under the companion's own name is
    // overriding it: the spec is skipped and the override recorded, so a
    // listing can show the companion rather than silently losing it. The
    // recorded name lets the editor's Expanded tab gray the row it copied.
    #[test]
    fn a_world_supplied_companion_is_recorded_as_an_override() {
        let mut assets = vec![
            serde_json::json!({"name":"gfx","type":"GraphicsConfig","args":{}}),
            // "Window" is the name the Window companion injects under.
            serde_json::json!({"name":"Window","type":"Window","args":{}}),
        ];
        let mut report = ExpandReport::default();
        inject_companions(&mut assets, &mut report);
        assert_eq!(
            assets.iter().filter(|v| type_norm(v) == "window").count(),
            1,
            "the world's own Window wins, and is not duplicated"
        );
        let shadow = report
            .shadowed
            .iter()
            .find(|s| s.name == "Window")
            .expect("the overridden companion is recorded");
        assert_eq!(shadow.asset_type, "Window");
        assert_eq!(shadow.generated_by, "companion");
    }

    // A companion we injected ourselves is not an override, even though later
    // rounds see it in the world and skip the spec.
    #[test]
    fn an_injected_companion_is_not_recorded_as_an_override() {
        let mut assets = vec![serde_json::json!({"name":"gfx","type":"GraphicsConfig","args":{}})];
        let mut report = ExpandReport::default();
        inject_companions(&mut assets, &mut report);
        assert!(
            report.injected.iter().any(|i| i.name == "Window"),
            "the Window companion was injected"
        );
        assert!(
            report.shadowed.is_empty(),
            "our own injection is not the user overriding it: {:?}",
            report.shadowed
        );
    }

    // A world whose Window has a different name still suppresses the companion
    // (the skip is by type), but that is not an override OF the companion asset,
    // so nothing is recorded against its name.
    #[test]
    fn a_differently_named_asset_of_the_same_type_is_not_an_override() {
        let mut assets = vec![
            serde_json::json!({"name":"gfx","type":"GraphicsConfig","args":{}}),
            serde_json::json!({"name":"main_window","type":"Window","args":{}}),
        ];
        let mut report = ExpandReport::default();
        inject_companions(&mut assets, &mut report);
        assert_eq!(
            assets.iter().filter(|v| type_norm(v) == "window").count(),
            1
        );
        assert!(
            report.shadowed.is_empty(),
            "no asset named Window exists to be the override: {:?}",
            report.shadowed
        );
    }

    #[test]
    fn no_injection_without_trigger() {
        let mut assets = vec![serde_json::json!({"name":"w","type":"Window","args":{}})];
        inject(&mut assets);
        assert!(!assets.iter().any(|v| type_norm(v) == "graphicsconfig"));
    }

    #[test]
    fn text_injects_graphics_config_and_font() {
        let mut assets =
            vec![serde_json::json!({"name":"t","type":"TextLabel","args":{"content":"hi"}})];
        inject(&mut assets);
        assert!(assets.iter().any(|v| type_norm(v) == "graphicsconfig"));
        assert!(assets.iter().any(|v| type_norm(v) == "font"));
    }

    #[test]
    fn text_does_not_inject_duplicate_graphics_config() {
        let mut assets = vec![
            serde_json::json!({"name":"t","type":"TextLabel","args":{"content":"hi"}}),
            serde_json::json!({"name":"gfx","type":"GraphicsConfig","args":{}}),
        ];
        inject(&mut assets);
        let gfx_count = assets
            .iter()
            .filter(|v| type_norm(v) == "graphicsconfig")
            .count();
        assert_eq!(gfx_count, 1);
    }

    #[test]
    fn text_does_not_inject_font_when_font_present() {
        let mut assets = vec![
            serde_json::json!({"name":"t","type":"TextLabel","args":{"content":"hi"}}),
            serde_json::json!({"name":"f","type":"Font","args":{"path":"my.ttf","size_px":20}}),
        ];
        inject(&mut assets);
        let font_count = assets.iter().filter(|v| type_norm(v) == "font").count();
        assert_eq!(font_count, 1);
    }

    #[test]
    fn text_patches_empty_font_field_to_default() {
        let mut assets =
            vec![serde_json::json!({"name":"t","type":"TextLabel","args":{"content":"hi"}})];
        inject(&mut assets);
        let label = assets.iter().find(|v| type_norm(v) == "textlabel").unwrap();
        let font = label["args"]["font"].as_str().unwrap_or("");
        // The default font is named after its bundled file, exactly as `cn add`
        // would name it (`Questrial-Regular.ttf` -> `Questrial-Regular`); the
        // patched reference must match, and a Font by that name must exist.
        let expected = crate::world::asset_name_from_path(crate::font::BUILTIN_FONT_FILE);
        assert_eq!(font, expected);
        assert!(
            assets
                .iter()
                .any(|v| type_norm(v) == "font" && v["name"].as_str() == Some(expected.as_str()))
        );
    }

    #[test]
    fn text_sets_centered_when_patching() {
        let mut assets =
            vec![serde_json::json!({"name":"t","type":"TextLabel","args":{"content":"hi"}})];
        inject(&mut assets);
        let label = assets.iter().find(|v| type_norm(v) == "textlabel").unwrap();
        assert_eq!(label["args"]["centered"], true);
    }

    #[test]
    fn text_does_not_override_explicit_font_on_label() {
        let mut assets = vec![serde_json::json!({
            "name": "t",
            "type": "TextLabel",
            "args": {"content": "hi", "font": "myfont"}
        })];
        inject(&mut assets);
        let label = assets.iter().find(|v| type_norm(v) == "textlabel").unwrap();
        assert_eq!(label["args"]["font"].as_str().unwrap(), "myfont");
    }

    #[test]
    fn graphics_config_injects_default_shader() {
        // TextLabel injects a GraphicsConfig, which in turn injects the
        // default Shader with its vertex + fragment stages.
        let mut assets =
            vec![serde_json::json!({"name":"t","type":"TextLabel","args":{"content":"hi"}})];
        inject(&mut assets);
        let shader = assets
            .iter()
            .find(|v| type_norm(v) == "shader")
            .expect("default Shader injected");
        assert!(shader["args"]["vertex"].is_object());
        assert!(shader["args"]["fragment"].is_object());
    }

    #[test]
    fn does_not_inject_a_shader_when_one_declared() {
        let mut assets = vec![
            serde_json::json!({"name":"gfx","type":"GraphicsConfig","args":{}}),
            serde_json::json!({
                "name":"custom","type":"Shader",
                "args":{"vertex":{"source":"custom.metal"},"fragment":{"source":"custom.metal"}}
            }),
        ];
        inject(&mut assets);
        let shader_count = assets.iter().filter(|v| type_norm(v) == "shader").count();
        assert_eq!(shader_count, 1);
    }

    #[test]
    fn graphics_config_injects_window() {
        let mut assets = vec![serde_json::json!({"name":"gfx","type":"GraphicsConfig","args":{}})];
        inject(&mut assets);
        assert!(assets.iter().any(|v| type_norm(v) == "window"));
    }

    // A label with no args object has nowhere to hold the font reference, so
    // the default-font pass leaves it alone rather than rewriting the entry.
    #[test]
    fn a_label_without_args_is_left_unpatched() {
        let mut assets = vec![serde_json::json!({"name":"t","type":"TextLabel"})];
        inject(&mut assets);
        let label = assets.iter().find(|v| type_norm(v) == "textlabel").unwrap();
        assert!(label.get("args").is_none());
        // The Font itself is still injected for the labels that can take it.
        assert!(assets.iter().any(|v| type_norm(v) == "font"));
    }

    // An entry with no `type` implies no companions and must not derail the
    // scan of the assets around it.
    #[test]
    fn a_typeless_entry_is_skipped() {
        let mut assets = vec![
            serde_json::json!({"name":"junk","args":{}}),
            serde_json::json!({"name":"gfx","type":"GraphicsConfig","args":{}}),
        ];
        inject(&mut assets);
        assert!(assets.iter().any(|v| type_norm(v) == "window"));
        assert!(assets.iter().any(|v| v["name"] == "junk"));
    }

    #[test]
    fn camera3d_injects_no_companions() {
        // The camera controller is now a field on Camera3D, not an injected
        // system, so a bare Camera3D pulls in nothing.
        let mut assets = vec![serde_json::json!({"name":"c","type":"Camera3D","args":{}})];
        inject(&mut assets);
        assert_eq!(assets.len(), 1);
        assert!(type_norm(&assets[0]) == "camera3d");
    }
}
