// Build-time expansion: Sand and Dirt -> a Material and its two Textures.

use crate::authoring::registry::RegisteredType;
use crate::authoring::registry::build_only::{Dirt, Sand};
use crate::build_only::expand::{asset_name, registered_type, schema_args};
use crate::compile::ground_surface::{dirt_peak, sand_peak};
use concinnity_engine::components::GroundMap;

pub(crate) fn expand_ground_looks(asset_values: &mut Vec<serde_json::Value>) -> Result<(), String> {
    let mut result = Vec::with_capacity(asset_values.len());
    for value in asset_values.drain(..) {
        let ty = registered_type(&value);
        let name = asset_name(&value);
        let entries = match ty {
            Some(RegisteredType::Sand) => {
                let sand: Sand = schema_args(RegisteredType::Sand, &name, value.get("args"))?;
                let look = serde_json::to_value(sand.look).map_err(|e| e.to_string())?;
                surface_entries(
                    &name,
                    ("sand", look),
                    sand.resolution,
                    sand_peak(&sand.look),
                    sand.look.roughness,
                )
            }
            Some(RegisteredType::Dirt) => {
                let dirt: Dirt = schema_args(RegisteredType::Dirt, &name, value.get("args"))?;
                let look = serde_json::to_value(dirt.look).map_err(|e| e.to_string())?;
                surface_entries(
                    &name,
                    ("dirt", look),
                    dirt.resolution,
                    dirt_peak(&dirt.look),
                    dirt.look.roughness,
                )
            }
            _ => vec![value],
        };
        result.extend(entries);
    }
    *asset_values = result;
    Ok(())
}

// The material `name` and its albedo and normal textures, drawn from the look
// under the Texture field `field`; the albedo map holds albedo over `peak`,
// which the material's tint restores.
fn surface_entries(
    name: &str,
    (field, look): (&str, serde_json::Value),
    resolution: u32,
    peak: [f32; 3],
    roughness: f32,
) -> Vec<serde_json::Value> {
    let texture = |map: GroundMap| {
        serde_json::json!({
            "type": "Texture",
            "args": {
                "$id": format!("{name}_{}", map.as_str()),
                field: look.clone(),
                "ground_map": map.as_str(),
                "resolution": resolution,
            }
        })
    };
    vec![
        texture(GroundMap::Albedo),
        texture(GroundMap::Normal),
        serde_json::json!({
            "type": "Material",
            "args": {
                "$id": name,
                "albedo": format!("{name}_albedo"),
                "normal_map": format!("{name}_normal"),
                "tint": peak,
                "roughness": roughness,
            }
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::components::{DirtLook, SandLook, Texture};

    #[test]
    fn a_sand_becomes_its_material_and_two_textures() {
        let mut assets = vec![
            serde_json::json!({"type": "Sand", "args": {"$id": "beach", "shells": 0.4}}),
            serde_json::json!({"type": "Logger", "args": {"$id": "other"}}),
        ];
        expand_ground_looks(&mut assets).unwrap();
        let ids: Vec<&str> = assets
            .iter()
            .filter_map(|v| v["args"]["$id"].as_str())
            .collect();
        assert_eq!(ids, ["beach_albedo", "beach_normal", "beach", "other"]);

        let material = &assets[2]["args"];
        assert_eq!(assets[2]["type"], "Material");
        assert_eq!(material["albedo"], "beach_albedo");
        assert_eq!(material["normal_map"], "beach_normal");
        let look = SandLook {
            shells: 0.4,
            ..SandLook::default()
        };
        assert_eq!(material["tint"], serde_json::json!(sand_peak(&look)));

        let normal: Texture = serde_json::from_value(assets[1]["args"].clone()).unwrap();
        assert_eq!(normal.ground_map, GroundMap::Normal);
        assert_eq!(normal.sand, Some(look));
        assert!(normal.dirt.is_none());
        assert_eq!(normal.resolution, 1024);
    }

    #[test]
    fn a_dirt_keeps_its_look_and_resolution() {
        let mut assets = vec![serde_json::json!({
            "type": "Dirt",
            "args": {"$id": "soil", "cracks": 0.5, "resolution": 256, "roughness": 0.8}
        })];
        expand_ground_looks(&mut assets).unwrap();
        let albedo: Texture = serde_json::from_value(assets[0]["args"].clone()).unwrap();
        assert_eq!(albedo.ground_map, GroundMap::Albedo);
        assert_eq!(albedo.resolution, 256);
        assert_eq!(albedo.dirt.map(|d| d.cracks), Some(0.5));
        assert_eq!(assets[2]["args"]["roughness"], serde_json::json!(0.8_f32));
        assert_ne!(DirtLook::default().cracks, 0.5);
    }

    #[test]
    fn a_misspelled_field_names_the_asset() {
        let mut assets =
            vec![serde_json::json!({"type": "Sand", "args": {"$id": "beach", "shells": "lots"}})];
        let err = expand_ground_looks(&mut assets).unwrap_err();
        assert!(err.contains("beach"), "{err}");
    }
}
