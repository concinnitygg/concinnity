//! Cooking a `Terrain`: the height grid, from generated noise or a decoded
//! heightmap `Texture`, and each layer's density mask, from its decoded
//! `Texture`. The images are decoded here; the grid and mask math is
//! `concinnity_core::terrain`'s.

use std::path::Path;

use concinnity_core::components::{Terrain, validate};
use concinnity_core::terrain::payload::TerrainPayload;
use concinnity_core::terrain::{DensityMask, TerrainGrid, heightmap_heights, noise_heights};

use crate::authoring::registry::RegisteredType;
use crate::build_only::expand::schema_args;

/// The texture names a terrain's args reference: its heightmap, then every
/// layer's density mask.
pub(crate) fn texture_names(args: &serde_json::Value) -> Vec<&str> {
    let heightmap = args.get("heightmap").and_then(|v| v.as_str());
    let masks = args
        .get("layers")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|layer| layer.get("density_mask").and_then(|v| v.as_str()));
    heightmap.into_iter().chain(masks).collect()
}

// `args` with every reference taken out: the compile reads the textures it
// names straight from the args and no other reference at all, so parsing
// needs no name resolver.
fn without_references(args: &serde_json::Value) -> serde_json::Value {
    let mut args = args.clone();
    if let Some(obj) = args.as_object_mut() {
        obj.remove("material");
        obj.remove("heightmap");
        for layer in obj
            .get_mut("layers")
            .and_then(|v| v.as_array_mut())
            .into_iter()
            .flatten()
        {
            if let Some(layer) = layer.as_object_mut() {
                layer.remove("grass");
                layer.remove("density_mask");
            }
        }
    }
    args
}

/// Compile the `Terrain` named `name` from its `args`, looking each texture it
/// names up with `texture` (a Texture asset's args by name).
pub(crate) fn compile_terrain_payload<'a>(
    name: &str,
    args: &serde_json::Value,
    texture: impl Fn(&str) -> Option<&'a serde_json::Value>,
    assets_dir: Option<&Path>,
) -> Result<Vec<u8>, String> {
    let scalars = without_references(args);
    let terrain: Terrain = schema_args(RegisteredType::Terrain, name, Some(&scalars))?;
    let terrain = validate::terrain(terrain);
    let pixels = |field: &str, texture_name: &str| {
        let args = texture(texture_name)
            .ok_or_else(|| format!("{field} names no Texture '{texture_name}'"))?;
        crate::compile::texture::texture_rgba8(args, assets_dir)
            .map_err(|e| format!("{field} Texture '{texture_name}': {e}"))
    };

    let heights = match args.get("heightmap").and_then(|v| v.as_str()) {
        Some(map) => {
            let (w, h, rgba) = pixels("heightmap", map)?;
            heightmap_heights(
                terrain.resolution,
                w,
                h,
                &rgba,
                [terrain.elevation_min, terrain.elevation_max],
            )
            .map_err(|e| format!("heightmap Texture '{map}': {e}"))?
        }
        None => noise_heights(terrain.resolution, terrain.amplitude, terrain.seed),
    };
    let grid = TerrainGrid::new(terrain.resolution, terrain.extent, heights)?;

    let layers = args.get("layers").and_then(|v| v.as_array());
    let mut masks = Vec::with_capacity(terrain.layers.len());
    for (i, layer) in layers.into_iter().flatten().enumerate() {
        let mask = match layer.get("density_mask").and_then(|v| v.as_str()) {
            Some(mask) => {
                let field = format!("layers[{i}].density_mask");
                let (w, h, rgba) = pixels(&field, mask)?;
                Some(DensityMask::from_rgba(w, h, &rgba).map_err(|e| format!("{field}: {e}"))?)
            }
            None => None,
        };
        masks.push(mask);
    }
    Ok(TerrainPayload { grid, masks }.encode())
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::terrain::terrain_chunks;
    use serde_json::json;

    fn no_textures(_: &str) -> Option<&'static serde_json::Value> {
        None
    }

    fn decode(bytes: &[u8], extent: [f32; 2]) -> TerrainPayload {
        TerrainPayload::decode(bytes, extent).expect("a cooked terrain decodes")
    }

    #[test]
    fn generated_heights_follow_amplitude_and_seed() {
        let args = json!({"extent": [20.0, 10.0], "resolution": 16, "amplitude": 3.0, "seed": 5});
        let p = decode(
            &compile_terrain_payload("hills", &args, no_textures, None).unwrap(),
            [20.0, 10.0],
        );
        assert_eq!(p.grid.resolution(), 16);
        assert_eq!(p.grid.heights(), noise_heights(16, 3.0, 5).as_slice());
        assert!(p.masks.is_empty());
    }

    #[test]
    fn defaults_cook_a_default_terrain() {
        let p = decode(
            &compile_terrain_payload("t", &json!({}), no_textures, None).unwrap(),
            Terrain::default().extent,
        );
        assert_eq!(p.grid.resolution(), Terrain::default().resolution);
    }

    // The authored resolution is clamped the way the runtime component is, so
    // the payload's grid and the component always describe the same terrain.
    #[test]
    fn the_resolution_is_clamped_like_the_component() {
        let args = json!({"resolution": 2});
        let p = decode(
            &compile_terrain_payload("t", &args, no_textures, None).unwrap(),
            Terrain::default().extent,
        );
        assert_eq!(
            p.grid.resolution(),
            concinnity_core::terrain::MIN_TERRAIN_RESOLUTION
        );
    }

    #[test]
    fn a_heightmap_texture_sets_the_heights() {
        let checker = json!({"generator": "checker", "resolution": 8});
        let args = json!({
            "resolution": 8,
            "heightmap": "bumps",
            "elevation_min": 1.0,
            "elevation_max": 3.0,
        });
        let lookup = |name: &str| (name == "bumps").then_some(&checker);
        let p = decode(
            &compile_terrain_payload("t", &args, lookup, None).unwrap(),
            Terrain::default().extent,
        );
        let (w, h, rgba) = crate::compile::texture::texture_rgba8(&checker, None).unwrap();
        assert_eq!(
            p.grid.heights(),
            heightmap_heights(8, w, h, &rgba, [1.0, 3.0])
                .unwrap()
                .as_slice()
        );
    }

    #[test]
    fn each_layer_keeps_its_mask_in_order() {
        let checker = json!({"generator": "checker", "resolution": 4});
        let args = json!({
            "resolution": 4,
            "layers": [
                {"grass": "meadow"},
                {"grass": "meadow", "density_mask": "patches"},
            ],
        });
        let lookup = |name: &str| (name == "patches").then_some(&checker);
        let p = decode(
            &compile_terrain_payload("t", &args, lookup, None).unwrap(),
            Terrain::default().extent,
        );
        assert_eq!(p.masks.len(), 2);
        assert!(p.masks[0].is_none());
        let mask = p.masks[1].as_ref().unwrap();
        let (w, h, rgba) = crate::compile::texture::texture_rgba8(&checker, None).unwrap();
        assert_eq!(mask, &DensityMask::from_rgba(w, h, &rgba).unwrap());
        assert_eq!(texture_names(&args), ["patches"]);
    }

    #[test]
    fn a_missing_texture_is_named() {
        let args = json!({"heightmap": "nowhere"});
        let err = compile_terrain_payload("t", &args, no_textures, None).unwrap_err();
        assert!(
            err.contains("heightmap names no Texture 'nowhere'"),
            "{err}"
        );
        let args = json!({"layers": [{"grass": "g", "density_mask": "gone"}]});
        let err = compile_terrain_payload("t", &args, no_textures, None).unwrap_err();
        assert!(err.contains("layers[0].density_mask"), "{err}");
    }

    // References parse to nothing here, so a compile worker with no name
    // resolver installed cooks a terrain that names its grass and material.
    #[test]
    fn references_need_no_resolver() {
        let args = json!({
            "material": "soil",
            "layers": [{"grass": "meadow"}, {"grass": "clover"}],
        });
        let p = decode(
            &compile_terrain_payload("t", &args, no_textures, None).unwrap(),
            Terrain::default().extent,
        );
        assert_eq!(p.masks, [None, None]);
    }

    #[test]
    fn mistyped_args_are_refused() {
        let err = compile_terrain_payload("t", &json!({"resolution": "lots"}), no_textures, None)
            .unwrap_err();
        assert!(err.contains("resolution"), "{err}");
    }

    // The cooked grid is the one surface the mesh, the collider and the grass
    // all read: the mesh's every vertex is a grid corner.
    #[test]
    fn the_cooked_grid_meshes_onto_itself() {
        let args = json!({"extent": [8.0, 8.0], "resolution": 10, "amplitude": 2.0});
        let p = decode(
            &compile_terrain_payload("t", &args, no_textures, None).unwrap(),
            [8.0, 8.0],
        );
        for chunk in terrain_chunks(&p.grid) {
            for v in &chunk.vertices {
                let h = p.grid.height_at([v.pos[0], v.pos[2]]).unwrap();
                assert!((h - v.pos[1]).abs() < 1e-5);
            }
        }
    }
}
