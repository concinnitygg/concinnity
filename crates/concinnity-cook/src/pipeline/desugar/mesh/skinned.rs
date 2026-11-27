//! `SkinnedMesh` assets read from a source file or a character model.

use std::collections::HashMap;
use std::io;
use std::path::Path;

use super::super::skin_index_arg;
use super::{MeshFormat, cached, invalid, source_arg, write_args};
use crate::authoring::registry::RegisteredType;
use crate::authoring::world::WorldJsonlAsset;
use crate::import::glb::ImportedSkinnedMesh;
use crate::pipeline::pack::MeshCacheEntry;
use crate::pipeline::progress::StageProgress;

// Expand source-backed SkinnedMesh assets in place: the imported geometry and
// skeleton land in the asset's inline `vertices` / `indices` / `skeleton` args,
// with any morph targets beside them. A source that names no other format is
// read as glTF. A SkinnedMesh with neither a `source` nor a `character_model`
// is left untouched, so an inline-authored mesh is byte-for-byte unchanged.
pub(in crate::pipeline) fn desugar_skinned_meshes(
    assets: &mut [WorldJsonlAsset],
    mesh_cache: &HashMap<String, MeshCacheEntry>,
    root: Option<&Path>,
    progress: &StageProgress<'_>,
) -> io::Result<()> {
    for asset in assets.iter_mut() {
        if asset.asset_type != RegisteredType::SkinnedMesh {
            continue;
        }
        let source = source_arg(asset).to_owned();
        let character_model = asset.args.get("character_model").cloned();
        if (character_model.is_none() && source.is_empty()) || cached(mesh_cache, asset) {
            continue;
        }

        let item = if source.is_empty() {
            asset.id.clone()
        } else {
            source.clone()
        };
        progress.begin(&item);
        let (label, imported) = if let Some(arg) = character_model {
            ("character model", import_character(&asset.id, arg, root)?)
        } else {
            let format = MeshFormat::of(&source).unwrap_or(MeshFormat::Gltf);
            let imported = import(format, &source, skin_index_arg(asset), root).map_err(|e| {
                invalid(format!(
                    "Asset '{}': {} import failed: {e}",
                    asset.id,
                    format.label()
                ))
            })?;
            (format.label(), imported)
        };

        let mut fields = vec![
            ("vertices", serde_json::to_value(&imported.vertices)),
            ("indices", serde_json::to_value(&imported.indices)),
            ("skeleton", serde_json::to_value(&imported.skeleton)),
        ];
        if !imported.morph_target_names.is_empty() {
            fields.push((
                "morph_target_names",
                serde_json::to_value(&imported.morph_target_names),
            ));
            fields.push(("morph_deltas", serde_json::to_value(&imported.morph_deltas)));
        }
        write_args(asset, fields)?;
        if let Some(obj) = asset.args.as_object_mut() {
            obj.remove("character_model");
        }
        tracing::info!(
            "Asset '{}': imported {label} '{item}': {} vertices, {} indices, {} joints, {} morph target(s)",
            asset.id,
            imported.vertices.len(),
            imported.indices.len(),
            imported.skeleton.len(),
            imported.morph_target_names.len()
        );
        progress.end(&item, false);
    }
    Ok(())
}

// Build the skinned mesh a `character_model` arg describes.
fn import_character(
    name: &str,
    arg: serde_json::Value,
    root: Option<&Path>,
) -> io::Result<ImportedSkinnedMesh> {
    let arg: crate::compile::character::import::CharacterModelArg = serde_json::from_value(arg)
        .map_err(|e| invalid(format!("Asset '{name}': character_model: {e}")))?;
    crate::compile::character::import::import_model(name, &arg.schema, &arg.model, root)
        .map_err(invalid)
}

// Read skin `skin_index` of `source` under `root`.
fn import(
    format: MeshFormat,
    source: &str,
    skin_index: u32,
    root: Option<&Path>,
) -> Result<ImportedSkinnedMesh, String> {
    match format {
        MeshFormat::Gltf => crate::import::gltf::import_skinned_glb(source, skin_index, root),
        MeshFormat::Fbx => {
            let path = concinnity_host::store::source::resolve(source, root);
            crate::import::fbx::import_skinned_fbx(&path.to_string_lossy(), skin_index)
        }
    }
}
