// Textures a world reads only at cook time. A terrain's heightmap and its
// layers' density masks are decoded into the terrain's own payload, so a
// texture nothing else names never needs a GPU copy; its resource record says
// so, and the renderer keeps it out of the texture pool.

use std::collections::HashSet;

use crate::authoring::registry::RegisteredType;
use crate::authoring::world::WorldJsonlAsset;

// The ids of every Texture some Terrain names and no other asset does. Any
// string in another asset's args equal to a texture's id counts as a use, so
// a reference the scan cannot place keeps the texture on the GPU.
pub(in crate::pipeline) fn cook_only_textures(assets: &[WorldJsonlAsset]) -> HashSet<String> {
    let textures: HashSet<&str> = assets
        .iter()
        .filter(|a| a.asset_type == RegisteredType::Texture)
        .map(|a| a.id.as_str())
        .collect();
    let mut cooked = HashSet::new();
    let mut sampled = HashSet::new();
    for asset in assets {
        let into = if asset.asset_type == RegisteredType::Terrain {
            &mut cooked
        } else {
            &mut sampled
        };
        collect_names(&asset.args, &textures, into);
    }
    cooked
        .difference(&sampled)
        .map(|name| name.to_string())
        .collect()
}

// Every string leaf of `value` that names one of `textures`.
fn collect_names<'a>(
    value: &serde_json::Value,
    textures: &HashSet<&'a str>,
    out: &mut HashSet<&'a str>,
) {
    match value {
        serde_json::Value::String(s) => {
            if let Some(&name) = textures.get(s.as_str()) {
                out.insert(name);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_names(item, textures, out);
            }
        }
        serde_json::Value::Object(fields) => {
            for field in fields.values() {
                collect_names(field, textures, out);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::fixtures::wja;
    use serde_json::json;

    fn texture(name: &str) -> WorldJsonlAsset {
        wja(name, RegisteredType::Texture, json!({ "source": "x.png" }))
    }

    fn terrain(heightmap: &str, mask: &str) -> WorldJsonlAsset {
        wja(
            "ground",
            RegisteredType::Terrain,
            json!({
                "material": "soil",
                "heightmap": heightmap,
                "layers": [{ "grass": "meadow", "density_mask": mask }],
            }),
        )
    }

    #[test]
    fn a_terrains_heightmap_and_masks_are_cook_only() {
        let assets = vec![
            texture("hills"),
            texture("patches"),
            texture("dirt"),
            terrain("hills", "patches"),
            wja(
                "soil",
                RegisteredType::Material,
                json!({ "albedo": "dirt" }),
            ),
        ];
        let cook_only = cook_only_textures(&assets);
        assert_eq!(
            cook_only,
            HashSet::from(["hills".to_string(), "patches".to_string()])
        );
    }

    // A texture a terrain cooks that something else also samples stays on
    // the GPU, wherever the other reference sits.
    #[test]
    fn a_texture_another_asset_names_stays_on_the_gpu() {
        let assets = vec![
            texture("hills"),
            texture("patches"),
            terrain("hills", "patches"),
            wja(
                "rock",
                RegisteredType::Material,
                json!({ "albedo": "patches" }),
            ),
            wja(
                "splat",
                RegisteredType::Decal,
                json!({ "nested": [{ "deep": "hills" }] }),
            ),
        ];
        assert!(cook_only_textures(&assets).is_empty());
    }

    // Only a Texture can be cook-only: a terrain naming its Grass or Material
    // marks nothing, and a texture no terrain names keeps its record.
    #[test]
    fn only_textures_a_terrain_names_are_considered() {
        let assets = vec![
            texture("unused"),
            terrain("missing", "also_missing"),
            wja("meadow", RegisteredType::Grass, json!({})),
        ];
        assert!(cook_only_textures(&assets).is_empty());
    }
}
