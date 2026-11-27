//! Source-backed Mesh and `SkinnedMesh` assets: the referenced `.glb`, `.gltf`
//! or `.fbx` is read and its geometry, skeleton and morph targets land in the
//! asset's inline args. Each source format answers the same questions, so one
//! pass per asset kind serves every format.

mod skinned;
mod static_mesh;

#[cfg(test)]
mod tests;

pub(in crate::pipeline) use skinned::desugar_skinned_meshes;
pub(in crate::pipeline) use static_mesh::desugar_meshes;

use std::collections::HashMap;
use std::io;
use std::path::Path;

use super::super::pack::MeshCacheEntry;
use crate::authoring::world::WorldJsonlAsset;

/// The container a mesh source is read from, by its file extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::pipeline) enum MeshFormat {
    /// A binary `.glb` or text `.gltf` container.
    Gltf,
    /// An `.fbx` document.
    Fbx,
}

impl MeshFormat {
    /// The format `source` names, or `None` for an extension no importer reads.
    pub(in crate::pipeline) fn of(source: &str) -> Option<Self> {
        let ext = Path::new(source).extension()?;
        if ext.eq_ignore_ascii_case("glb") || ext.eq_ignore_ascii_case("gltf") {
            Some(Self::Gltf)
        } else if ext.eq_ignore_ascii_case("fbx") {
            Some(Self::Fbx)
        } else {
            None
        }
    }

    // The name an import error and the import log carry.
    fn label(self) -> &'static str {
        match self {
            Self::Gltf => "glTF",
            Self::Fbx => "FBX",
        }
    }
}

/// The asset's `source` arg, empty when it has none.
pub(in crate::pipeline) fn source_arg(asset: &WorldJsonlAsset) -> &str {
    asset
        .args
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or("")
}

/// Whether the payload cache already holds the compiled bytes for the asset, so
/// its source need not be read. Its args then stay pre-desugar, keeping the next
/// build's cache key.
pub(in crate::pipeline) fn cached(
    mesh_cache: &HashMap<String, MeshCacheEntry>,
    asset: &WorldJsonlAsset,
) -> bool {
    matches!(
        mesh_cache.get(&asset.id),
        Some(MeshCacheEntry { bytes: Some(_), .. })
    )
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

// Write each imported field into the asset's args, serialized.
fn write_args(
    asset: &mut WorldJsonlAsset,
    fields: impl IntoIterator<Item = (&'static str, serde_json::Result<serde_json::Value>)>,
) -> io::Result<()> {
    let name = asset.id.clone();
    let obj = asset
        .args
        .as_object_mut()
        .ok_or_else(|| invalid(format!("Asset '{name}': args is not a JSON object")))?;
    for (field, value) in fields {
        let value = value.map_err(|e| {
            invalid(format!(
                "Asset '{name}': failed to encode imported {field}: {e}"
            ))
        })?;
        obj.insert(field.to_string(), value);
    }
    Ok(())
}
