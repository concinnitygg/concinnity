// Build-time expansion: MaterialPalette -> Material assets.

use std::path::Path;

use crate::authoring::registry::RegisteredType;
use crate::authoring::registry::build_only::{MaterialPalette, PaletteEntry};
use crate::build_only::expand::{asset_name, registered_type, schema_args};
use crate::build_only::preset::load_preset_obj;

pub(crate) fn expand_material_palettes(
    asset_values: &mut Vec<serde_json::Value>,
    assets_dir: Option<&Path>,
) -> Result<(), String> {
    let mut result: Vec<serde_json::Value> = Vec::new();
    for value in asset_values.drain(..) {
        if registered_type(&value) != Some(RegisteredType::MaterialPalette) {
            result.push(value);
            continue;
        }
        let palette_name = asset_name(&value);
        let entries = resolve_palette_entries(&palette_name, value.get("args"), assets_dir)?;
        result.extend(entries.iter().map(|e| material_value(&palette_name, e)));
    }
    *asset_values = result;
    Ok(())
}

// The palette's entries: its preset's when one is named, else its inline list.
fn resolve_palette_entries(
    name: &str,
    args: Option<&serde_json::Value>,
    assets_dir: Option<&Path>,
) -> Result<Vec<PaletteEntry>, String> {
    let ty = RegisteredType::MaterialPalette;
    let palette: MaterialPalette = schema_args(ty, name, args)?;
    if palette.preset.is_empty() {
        return Ok(palette.entries);
    }
    let hardcoded = palette_preset_entries(&palette.preset);
    if !hardcoded.is_empty() {
        return Ok(hardcoded);
    }
    let loaded = load_preset_obj(&palette.preset, "palettes", assets_dir);
    let preset: MaterialPalette = schema_args(ty, name, loaded.get("args"))
        .map_err(|e| format!("{e} (in preset '{}')", palette.preset))?;
    Ok(preset.entries)
}

fn material_value(palette_name: &str, entry: &PaletteEntry) -> serde_json::Value {
    serde_json::json!({
        "name": format!("{}_{}", palette_name, entry.alias),
        "type": "Material",
        "args": {
            "albedo":          entry.albedo,
            "normal_map":      entry.normal_map,
            "roughness":       entry.roughness,
            "metallic":        entry.metallic,
            "tint":            entry.tint,
            "emissive_factor": entry.emissive_factor
        }
    })
}

fn preset_entry(alias: &str, albedo: &str, roughness: f32, metallic: f32) -> PaletteEntry {
    PaletteEntry {
        alias: alias.to_string(),
        albedo: albedo.to_string(),
        roughness,
        metallic,
        ..Default::default()
    }
}

fn palette_preset_entries(preset: &str) -> Vec<PaletteEntry> {
    let e = preset_entry;
    match preset {
        "pal_stone_dungeon" => vec![
            e("floor", "tex_stone", 0.9, 0.0),
            e("wall", "tex_stone", 0.85, 0.0),
            e("ceiling", "tex_stone", 0.9, 0.0),
            e("pillar", "tex_stone", 0.8, 0.0),
        ],
        "pal_wood_cabin" => vec![
            e("floor", "tex_wood", 0.7, 0.0),
            e("wall", "tex_plaster", 0.85, 0.0),
            e("beam", "tex_wood", 0.65, 0.0),
            e("trim", "tex_wood", 0.6, 0.0),
        ],
        "pal_metal_industrial" => vec![
            e("floor", "tex_concrete", 0.85, 0.0),
            e("wall", "tex_concrete", 0.8, 0.0),
            e("pipe", "tex_metal", 0.4, 1.0),
            e("grate", "tex_metal", 0.5, 0.8),
        ],
        "pal_plaster_cottage" => vec![
            e("floor", "tex_wood", 0.7, 0.0),
            e("wall", "tex_plaster", 0.9, 0.0),
            e("trim", "tex_wood", 0.6, 0.0),
            e("door", "tex_wood", 0.65, 0.0),
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
        expand_material_palettes(&mut assets, None).unwrap();
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
        expand_material_palettes(&mut assets, None).unwrap();
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
        expand_material_palettes(&mut assets, None).unwrap();
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
        expand_material_palettes(&mut assets, None).unwrap();
        assert_eq!(assets[0]["args"]["roughness"], serde_json::json!(0.8f32));
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
        expand_material_palettes(&mut assets, None).unwrap();
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
        expand_material_palettes(&mut assets, None).unwrap();
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
        expand_material_palettes(&mut assets, None).unwrap();
        assert_eq!(assets[0]["name"], "pal_surface");
    }

    // A palette with neither preset nor entries is consumed and adds nothing.
    #[test]
    fn palette_without_entries_expands_to_nothing() {
        let mut assets = vec![serde_json::json!({"name":"pal","type":"MaterialPalette"})];
        expand_material_palettes(&mut assets, None).unwrap();
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
        expand_material_palettes(&mut assets, None).unwrap();
        let args = &assets[0]["args"];
        assert_eq!(args["albedo"], "tex_gold");
        assert_eq!(args["normal_map"], "tex_gold_n");
        assert_eq!(args["roughness"], serde_json::json!(0.2f32));
        assert_eq!(args["metallic"], 1.0);
        assert_eq!(args["tint"], serde_json::json!([0.9f32, 0.8f32, 0.1f32]));
        assert_eq!(
            args["emissive_factor"],
            serde_json::json!([0.5f32, 0.4f32, 0.0f32])
        );
    }

    // A preset takes over the palette: inline entries are ignored.
    #[test]
    fn a_preset_palette_ignores_its_inline_entries() {
        let mut assets = vec![serde_json::json!({
            "name": "pal", "type": "MaterialPalette",
            "args": {"preset": "pal_wood_cabin", "entries": [{"alias": "extra"}]}
        })];
        expand_material_palettes(&mut assets, None).unwrap();
        let names: Vec<String> = assets.iter().map(asset_name).collect();
        assert_eq!(names, ["pal_floor", "pal_wall", "pal_beam", "pal_trim"]);
        assert_eq!(assets[0]["args"]["albedo"], "tex_wood");
        assert_eq!(assets[0]["args"]["normal_map"], "");
        assert_eq!(
            assets[0]["args"]["tint"],
            serde_json::json!([1.0, 1.0, 1.0])
        );
    }

    #[test]
    fn malformed_fields_name_the_palette_and_the_field() {
        for (args, field) in [
            (
                serde_json::json!({"preset": ["pal_wood_cabin"]}),
                "`preset`",
            ),
            (
                serde_json::json!({"entries": {"alias": "floor"}}),
                "`entries`",
            ),
            (
                serde_json::json!({"entries": [{"alias": "floor", "roughness": "rough"}]}),
                "`entries[0].roughness`",
            ),
            (
                serde_json::json!({"entries": [{"tint": [1.0, 1.0]}]}),
                "`entries[0].tint`",
            ),
        ] {
            let mut assets =
                vec![serde_json::json!({"name": "pal", "type": "MaterialPalette", "args": args})];
            let err = expand_material_palettes(&mut assets, None).unwrap_err();
            assert!(
                err.starts_with("MaterialPalette 'pal': invalid args: "),
                "{err}"
            );
            assert!(err.contains(field), "{err}");
        }
    }
}
