// src/world/companion.rs
// Inject companion assets implied by the presence of other assets.
//
// Each renderable asset's companions are declared in `companion_specs` and
// dispatched there by normalized type name. This module applies the resulting
// specs to a world JSONL value list.
//
// Injection runs to a fixed point. Each round snapshots the current world, asks
// every declared asset for its companion specs, then filters out specs whose
// `asset_type` already appears in the world or whose `name` was already
// collected this round (by-name dedup within a round so a single asset can
// request multiple companions of the same type, e.g. GraphicsSystem's default
// Shader).
//
// Text naming no Font gets no companion: the renderer draws it with the face
// baked into the binary, so a world that wants no particular face compiles no
// atlas for one.

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
// name: that asset is the user's patch of the companion, so the spec's args
// are merged under it and a listing can say so rather than leave the companion
// unaccounted for. A spec skipped because the world has that type under some
// OTHER name is not an override of this asset, and an injection from an
// earlier round is our own, so neither counts.
fn record_if_overridden(
    assets: &mut [serde_json::Value],
    claimed_names: &HashSet<String>,
    report: &mut ExpandReport,
    spec: &CompanionSpec,
) {
    let claimed = claimed_names.contains(spec.name);
    let ours = report.injected.iter().any(|i| i.name == spec.name);
    if claimed && !ours {
        report.record_shadowed(spec.name, spec.asset_type, "companion", spec.args.clone());
        super::shadow::merge_into_authored(assets, spec.name, &spec.args);
    }
}

// Dispatch a companion lookup for one asset by its normalized type name.
fn companions_for_type(asset_type: &str) -> Vec<CompanionSpec> {
    companions_for(&type_norm_str(asset_type))
}

pub(crate) fn inject_companions(assets: &mut Vec<serde_json::Value>, report: &mut ExpandReport) {
    loop {
        // Freeze what this round sees before injecting anything: companions
        // added below only enter the visible set on the next iteration, which
        // keeps multi-spec batches from shadowing each other through the
        // per-spec type-dedup.
        let present_types: HashSet<String> = assets.iter().map(asset_type_norm).collect();
        let claimed_names: HashSet<String> = assets
            .iter()
            .filter_map(|v| v.get("name").and_then(|n| n.as_str()))
            .map(str::to_string)
            .collect();

        // Collect every spec implied by every declared asset.
        let mut candidates: Vec<CompanionSpec> = Vec::new();
        for value in assets.iter() {
            let Some(t) = value.get("type").and_then(|s| s.as_str()) else {
                continue;
            };
            candidates.extend(companions_for_type(t));
        }

        // Apply: skip a spec whose asset_type already exists in the
        // pre-round world. Within the round, dedup by `name` so two specs
        // sharing a type (e.g. the default shader stages) both pass.
        let mut seen_names: HashSet<String> = HashSet::new();
        let mut to_inject = Vec::new();
        for spec in candidates {
            if present_types.contains(&type_norm_str(spec.asset_type)) {
                record_if_overridden(assets, &claimed_names, report, &spec);
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
    fn text_injects_graphics_config() {
        let mut assets =
            vec![serde_json::json!({"name":"t","type":"TextLabel","args":{"content":"hi"}})];
        inject(&mut assets);
        assert!(assets.iter().any(|v| type_norm(v) == "graphicsconfig"));
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

    // A label naming no Font compiles no atlas for one: the renderer draws it
    // with the face baked into the binary, so injecting a second rasterisation
    // of that same face would cost blob space for nothing.
    #[test]
    fn text_naming_no_font_injects_none() {
        let mut assets =
            vec![serde_json::json!({"name":"t","type":"TextLabel","args":{"content":"hi"}})];
        inject(&mut assets);
        assert!(!assets.iter().any(|v| type_norm(v) == "font"));
    }

    // And the label itself comes through untouched. Writing a `font` or a
    // `centered` into it would make a cooked world render differently from the
    // same assets built in code, and would silently discard the authored x/y.
    #[test]
    fn a_font_less_label_is_left_exactly_as_authored() {
        let authored = serde_json::json!({
            "name": "t",
            "type": "TextLabel",
            "args": {"content": "hi", "x": 40.0, "y": 80.0}
        });
        let mut assets = vec![authored.clone()];
        inject(&mut assets);
        let label = assets.iter().find(|v| type_norm(v) == "textlabel").unwrap();
        assert_eq!(label, &authored);
    }

    #[test]
    fn a_declared_font_is_not_duplicated() {
        let mut assets = vec![
            serde_json::json!({"name":"t","type":"TextLabel","args":{"content":"hi"}}),
            serde_json::json!({"name":"f","type":"Font","args":{"path":"my.ttf","size_px":20}}),
        ];
        inject(&mut assets);
        let font_count = assets.iter().filter(|v| type_norm(v) == "font").count();
        assert_eq!(font_count, 1);
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
    fn graphics_config_injects_window() {
        let mut assets = vec![serde_json::json!({"name":"gfx","type":"GraphicsConfig","args":{}})];
        inject(&mut assets);
        assert!(assets.iter().any(|v| type_norm(v) == "window"));
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
