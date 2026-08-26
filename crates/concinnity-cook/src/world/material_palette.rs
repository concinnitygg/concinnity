// src/world/material_palette.rs
// Build-time expansion: MaterialPalette → Material assets.

use std::path::Path;

use super::expand::{asset_name, type_norm};
use super::preset::load_preset_obj;

pub(crate) fn expand_material_palettes(
    asset_values: &mut Vec<serde_json::Value>,
    assets_dir: Option<&Path>,
) {
    let mut result: Vec<serde_json::Value> = Vec::new();
    for value in asset_values.drain(..) {
        if type_norm(&value) != "materialpalette" {
            result.push(value);
            continue;
        }
        let palette_name = asset_name(&value);
        let args = value.get("args").cloned().unwrap_or(serde_json::json!({}));
        for mat in resolve_palette_materials(&palette_name, &args, assets_dir) {
            result.push(mat);
        }
    }
    *asset_values = result;
}

fn resolve_palette_materials(
    palette_name: &str,
    args: &serde_json::Value,
    assets_dir: Option<&Path>,
) -> Vec<serde_json::Value> {
    let preset = args.get("preset").and_then(|v| v.as_str()).unwrap_or("");
    let entries: Vec<serde_json::Value> = if !preset.is_empty() {
        let hardcoded = palette_preset_entries(preset);
        if hardcoded.is_empty() {
            load_preset_obj(preset, "palettes", assets_dir)
                .get("args")
                .and_then(|a| a.get("entries"))
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default()
        } else {
            hardcoded
        }
    } else {
        args.get("entries")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default()
    };

    entries
        .iter()
        .map(|entry| {
            let alias = entry.get("alias").and_then(|v| v.as_str()).unwrap_or("surface");
            let expanded = format!("{}_{}", palette_name, alias);
            serde_json::json!({
                "name": expanded,
                "type": "Material",
                "args": {
                    "albedo":          entry.get("albedo").cloned().unwrap_or(serde_json::json!("")),
                    "normal_map":      entry.get("normal_map").cloned().unwrap_or(serde_json::json!("")),
                    "roughness":       entry.get("roughness").and_then(|v| v.as_f64()).unwrap_or(0.8),
                    "metallic":        entry.get("metallic").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    "tint":            entry.get("tint").cloned().unwrap_or(serde_json::json!([1.0, 1.0, 1.0])),
                    "emissive_factor": entry.get("emissive_factor").cloned().unwrap_or(serde_json::json!([0.0, 0.0, 0.0]))
                }
            })
        })
        .collect()
}

fn palette_preset_entries(preset: &str) -> Vec<serde_json::Value> {
    match preset {
        "pal_stone_dungeon" => vec![
            serde_json::json!({"alias":"floor",  "albedo":"tex_stone","roughness":0.9, "metallic":0.0}),
            serde_json::json!({"alias":"wall",   "albedo":"tex_stone","roughness":0.85,"metallic":0.0}),
            serde_json::json!({"alias":"ceiling","albedo":"tex_stone","roughness":0.9, "metallic":0.0}),
            serde_json::json!({"alias":"pillar", "albedo":"tex_stone","roughness":0.8, "metallic":0.0}),
        ],
        "pal_wood_cabin" => vec![
            serde_json::json!({"alias":"floor","albedo":"tex_wood",   "roughness":0.7, "metallic":0.0}),
            serde_json::json!({"alias":"wall", "albedo":"tex_plaster","roughness":0.85,"metallic":0.0}),
            serde_json::json!({"alias":"beam", "albedo":"tex_wood",   "roughness":0.65,"metallic":0.0}),
            serde_json::json!({"alias":"trim", "albedo":"tex_wood",   "roughness":0.6, "metallic":0.0}),
        ],
        "pal_metal_industrial" => vec![
            serde_json::json!({"alias":"floor","albedo":"tex_concrete","roughness":0.85,"metallic":0.0}),
            serde_json::json!({"alias":"wall", "albedo":"tex_concrete","roughness":0.8, "metallic":0.0}),
            serde_json::json!({"alias":"pipe", "albedo":"tex_metal",   "roughness":0.4, "metallic":1.0}),
            serde_json::json!({"alias":"grate","albedo":"tex_metal",   "roughness":0.5, "metallic":0.8}),
        ],
        "pal_plaster_cottage" => vec![
            serde_json::json!({"alias":"floor","albedo":"tex_wood",   "roughness":0.7, "metallic":0.0}),
            serde_json::json!({"alias":"wall", "albedo":"tex_plaster","roughness":0.9, "metallic":0.0}),
            serde_json::json!({"alias":"trim", "albedo":"tex_wood",   "roughness":0.6, "metallic":0.0}),
            serde_json::json!({"alias":"door", "albedo":"tex_wood",   "roughness":0.65,"metallic":0.0}),
        ],
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_entries_expand_to_materials() {
        let mut assets = vec![serde_json::json!({
            "name": "pal",
            "type": "MaterialPalette",
            "args": {"entries": [
                {"alias":"floor","albedo":"tex_stone","roughness":0.9,"metallic":0.0},
                {"alias":"wall","albedo":"tex_brick","roughness":0.85,"metallic":0.0}
            ]}
        })];
        expand_material_palettes(&mut assets, None);
        assert_eq!(assets.len(), 2);
        assert_eq!(assets[0]["name"], "pal_floor");
        assert_eq!(assets[0]["type"], "Material");
        assert_eq!(assets[1]["name"], "pal_wall");
        assert_eq!(assets[0]["args"]["albedo"], "tex_stone");
    }

    #[test]
    fn preset_stone_dungeon_expands_four_materials() {
        let mut assets = vec![serde_json::json!({
            "name": "pal",
            "type": "MaterialPalette",
            "args": {"preset": "pal_stone_dungeon"}
        })];
        expand_material_palettes(&mut assets, None);
        assert_eq!(assets.len(), 4);
        let names: Vec<&str> = assets.iter().filter_map(|v| v["name"].as_str()).collect();
        assert!(names.contains(&"pal_floor"));
        assert!(names.contains(&"pal_wall"));
        assert!(names.contains(&"pal_ceiling"));
        assert!(names.contains(&"pal_pillar"));
    }

    #[test]
    fn material_palette_consumed_from_list() {
        let mut assets = vec![
            serde_json::json!({"name":"pal","type":"MaterialPalette","args":{"entries":[
                {"alias":"x","roughness":0.5}
            ]}}),
            serde_json::json!({"name":"other","type":"Logger","args":{}}),
        ];
        expand_material_palettes(&mut assets, None);
        assert!(!assets.iter().any(|v| v["type"] == "MaterialPalette"));
        assert!(assets.iter().any(|v| v["type"] == "Logger"));
    }

    #[test]
    fn material_defaults_applied() {
        let mut assets = vec![serde_json::json!({
            "name": "pal",
            "type": "MaterialPalette",
            "args": {"entries": [{"alias":"base"}]}
        })];
        expand_material_palettes(&mut assets, None);
        assert_eq!(assets[0]["args"]["roughness"], 0.8);
        assert_eq!(assets[0]["args"]["metallic"], 0.0);
    }

    // Expand a single-preset palette and return the alias suffix of every
    // generated Material (the part after the "pal_" prefix).
    fn expand_preset(preset: &str) -> Vec<String> {
        let mut assets = vec![serde_json::json!({
            "name": "pal",
            "type": "MaterialPalette",
            "args": {"preset": preset}
        })];
        expand_material_palettes(&mut assets, None);
        assets
            .iter()
            .filter_map(|v| v["name"].as_str())
            .map(|n| n.trim_start_matches("pal_").to_string())
            .collect()
    }

    #[test]
    fn preset_wood_cabin_expands_its_surfaces() {
        let aliases = expand_preset("pal_wood_cabin");
        assert_eq!(aliases, ["floor", "wall", "beam", "trim"]);
    }

    #[test]
    fn preset_metal_industrial_expands_its_surfaces() {
        let mut assets = vec![serde_json::json!({
            "name": "pal",
            "type": "MaterialPalette",
            "args": {"preset": "pal_metal_industrial"}
        })];
        expand_material_palettes(&mut assets, None);
        let names: Vec<&str> = assets.iter().filter_map(|v| v["name"].as_str()).collect();
        assert_eq!(names, ["pal_floor", "pal_wall", "pal_pipe", "pal_grate"]);
        // The pipe surface is fully metallic per the preset table.
        let pipe = assets.iter().find(|v| v["name"] == "pal_pipe").unwrap();
        assert_eq!(pipe["args"]["metallic"], 1.0);
    }

    #[test]
    fn preset_plaster_cottage_expands_its_surfaces() {
        let aliases = expand_preset("pal_plaster_cottage");
        assert_eq!(aliases, ["floor", "wall", "trim", "door"]);
    }

    // An unknown preset is not a build error: the on-disk preset lookup misses
    // and the palette expands to nothing.
    #[test]
    fn unknown_preset_expands_to_no_materials() {
        assert!(expand_preset("cn_test_no_such_palette").is_empty());
    }

    // An entry with no alias still gets a material, under the generic name.
    #[test]
    fn entry_without_an_alias_falls_back_to_surface() {
        let mut assets = vec![serde_json::json!({
            "name": "pal",
            "type": "MaterialPalette",
            "args": {"entries": [{"albedo": "tex_x"}]}
        })];
        expand_material_palettes(&mut assets, None);
        assert_eq!(assets[0]["name"], "pal_surface");
    }

    // A palette with neither preset nor entries is consumed and adds nothing.
    #[test]
    fn palette_without_entries_expands_to_nothing() {
        let mut assets = vec![serde_json::json!({"name":"pal","type":"MaterialPalette"})];
        expand_material_palettes(&mut assets, None);
        assert!(assets.is_empty());
    }

    #[test]
    fn entry_fields_override_material_defaults() {
        let mut assets = vec![serde_json::json!({
            "name": "pal",
            "type": "MaterialPalette",
            "args": {"entries": [{
                "alias": "hero",
                "albedo": "tex_gold",
                "normal_map": "tex_gold_n",
                "roughness": 0.2,
                "metallic": 1.0,
                "tint": [0.9, 0.8, 0.1],
                "emissive_factor": [0.5, 0.4, 0.0]
            }]}
        })];
        expand_material_palettes(&mut assets, None);
        let args = &assets[0]["args"];
        assert_eq!(args["albedo"], "tex_gold");
        assert_eq!(args["normal_map"], "tex_gold_n");
        assert_eq!(args["roughness"], 0.2);
        assert_eq!(args["metallic"], 1.0);
        assert_eq!(args["tint"], serde_json::json!([0.9, 0.8, 0.1]));
        assert_eq!(args["emissive_factor"], serde_json::json!([0.5, 0.4, 0.0]));
    }
}
