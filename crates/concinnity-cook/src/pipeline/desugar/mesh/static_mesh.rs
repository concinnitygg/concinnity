//! Static `Mesh` assets read from one primitive of a source file.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::io;
use std::path::Path;

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::components::VertexData;

use super::{MeshFormat, cached, invalid, source_arg, write_args};
use crate::authoring::registry::RegisteredType;
use crate::authoring::world::WorldJsonlAsset;
use crate::import::fbx::FbxScene;
use crate::import::gltf_source::GltfDoc;
use crate::pipeline::pack::MeshCacheEntry;
use crate::pipeline::progress::StageProgress;

// One u16 chunk of a split primitive: its vertices and index buffer.
type Chunk = (Vec<VertexData>, Vec<u16>);

// A parsed source file, read once however many Mesh assets fan out of it.
enum ParsedSource {
    Gltf(Box<GltfDoc>),
    Fbx(FbxScene),
}

impl ParsedSource {
    fn parse(format: MeshFormat, source: &str, root: Option<&Path>) -> Result<Self, String> {
        match format {
            MeshFormat::Gltf => {
                crate::import::glb::parse_glb(source, root).map(|doc| Self::Gltf(Box::new(doc)))
            }
            MeshFormat::Fbx => {
                let path = concinnity_host::store::source::resolve(source, root);
                crate::import::fbx::parse_fbx(&path.to_string_lossy()).map(Self::Fbx)
            }
        }
    }

    // Primitive `index` with 32-bit indices, before any chunk split.
    fn primitive(&self, source: &str, index: u32) -> Result<(Vec<VertexData>, Vec<u32>), String> {
        match self {
            Self::Gltf(doc) => crate::import::glb::read_primitive_geometry(doc, source, index),
            Self::Fbx(scene) => crate::import::fbx::read_primitive_geometry(scene, index),
        }
    }
}

// Expand source-backed static Mesh assets in place: primitive `primitive_index`
// of the source, or chunk `chunk_index` of its u16 split, lands in the asset's
// inline `vertices` / `indices` args. A Mesh whose source is no format an
// importer reads is left untouched. Each file is parsed once and each
// primitive split once, since a large scene fans thousands of Mesh assets out
// of one file.
pub(in crate::pipeline) fn desugar_meshes(
    assets: &mut [WorldJsonlAsset],
    mesh_cache: &HashMap<String, MeshCacheEntry>,
    root: Option<&Path>,
    progress: &StageProgress<'_>,
) -> io::Result<()> {
    let mut parsed: HashMap<String, ParsedSource> = HashMap::new();
    let mut chunks: HashMap<(String, u32), Vec<Chunk>> = HashMap::new();

    for asset in assets.iter_mut() {
        if asset.asset_type != RegisteredType::Mesh {
            continue;
        }
        let source = source_arg(asset).to_owned();
        let Some(format) = MeshFormat::of(&source) else {
            continue;
        };
        if cached(mesh_cache, asset) {
            continue;
        }
        let primitive_index = u32::saturating_from(
            asset
                .args
                .get("primitive_index")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
        );
        let chunk_index = asset
            .args
            .get("chunk_index")
            .and_then(serde_json::Value::as_u64)
            .map(usize::saturating_from);

        progress.begin(&source);
        let failed = |e: String| {
            invalid(format!(
                "Asset '{}': {} import failed: {e}",
                asset.id,
                format.label()
            ))
        };
        let doc = match parsed.entry(source.clone()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                entry.insert(ParsedSource::parse(format, &source, root).map_err(failed)?)
            }
        };
        // A glTF primitive with no `chunk_index` imports whole; every other
        // read goes through the chunk split, an FBX primitive at chunk 0.
        let whole = match (&*doc, chunk_index) {
            (ParsedSource::Gltf(gltf), None) => {
                Some(crate::import::glb::import_static_glb_primitive_from_doc(
                    gltf,
                    &source,
                    primitive_index,
                ))
            }
            _ => None,
        };
        let (vertices, indices) = if let Some(whole) = whole {
            whole.map_err(failed)?
        } else {
            let chunk_index = chunk_index.unwrap_or(0);
            let split = match chunks.entry((source.clone(), primitive_index)) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let (verts, indices32) =
                        doc.primitive(&source, primitive_index).map_err(failed)?;
                    entry.insert(crate::import::glb::split_into_u16_chunks(
                        &verts, &indices32,
                    ))
                }
            };
            split.get(chunk_index).cloned().ok_or_else(|| {
                invalid(format!(
                    "Asset '{}': chunk_index {chunk_index} out of range, '{source}' \
                     primitive {primitive_index} splits into {} chunk(s)",
                    asset.id,
                    split.len(),
                ))
            })?
        };

        let (vlen, ilen) = (vertices.len(), indices.len());
        write_args(
            asset,
            [
                ("vertices", serde_json::to_value(&vertices)),
                ("indices", serde_json::to_value(&indices)),
            ],
        )?;
        tracing::info!(
            "Asset '{}': imported {} '{source}' primitive {primitive_index}{}: {vlen} vertices, {ilen} indices",
            asset.id,
            format.label(),
            chunk_index.map_or_else(String::new, |c| format!(" chunk {c}")),
        );
        progress.end(&source, false);
    }
    Ok(())
}
