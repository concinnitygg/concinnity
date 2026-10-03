//! Rewrites source-backed assets into the inline data the compile pass expects:
//! a `.glb` or `.fbx` reference becomes vertices, a skeleton, or animation
//! tracks in the asset's own args. Runs before anything else reads them.

mod animation;
mod fbx;
mod gltf;

#[cfg(test)]
mod fixtures;

pub(in crate::pipeline) use animation::{desugar_animation_imports, desugar_root_motion};
pub(in crate::pipeline) use fbx::{desugar_fbx_meshes, desugar_fbx_skinned_meshes};
pub(in crate::pipeline) use gltf::{desugar_gltf_meshes, desugar_gltf_skinned_meshes};

use super::pack::MeshCacheEntry;
use crate::authoring::registry::RegisteredType;
use crate::authoring::world::WorldJsonlAsset;

// How many assets the import passes will read a source file for: every
// source-backed mesh the payload cache missed, and every source-backed
// animation.
pub(in crate::pipeline) fn pending_imports(
    assets: &[WorldJsonlAsset],
    mesh_cache: &std::collections::HashMap<String, MeshCacheEntry>,
) -> u32 {
    let cached = |asset: &WorldJsonlAsset| {
        matches!(
            mesh_cache.get(&asset.id),
            Some(MeshCacheEntry { bytes: Some(_), .. })
        )
    };
    let source = |asset: &WorldJsonlAsset| {
        asset
            .args
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_lowercase()
    };
    let pending = assets.iter().filter(|asset| match asset.asset_type {
        RegisteredType::Mesh => {
            let source = source(asset);
            [".glb", ".gltf", ".fbx"]
                .iter()
                .any(|ext| source.ends_with(ext))
                && !cached(asset)
        }
        RegisteredType::SkinnedMesh => {
            (asset.args.get("character_model").is_some() || !source(asset).is_empty())
                && !cached(asset)
        }
        RegisteredType::Animation => !source(asset).is_empty(),
        _ => false,
    });
    pending.count() as u32
}

// Which skinned mesh of the asset's source file it selects; absent means the
// file's first.
fn skin_index_arg(asset: &WorldJsonlAsset) -> u32 {
    asset
        .args
        .get("skin_index")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::desugar::fixtures::hit_cache;
    use crate::pipeline::fixtures::wja;

    #[test]
    fn pending_imports_counts_source_backed_assets_the_cache_missed() {
        let assets = [
            wja(
                "glb",
                RegisteredType::Mesh,
                serde_json::json!({"source": "a.glb"}),
            ),
            wja(
                "fbx",
                RegisteredType::Mesh,
                serde_json::json!({"source": "b.FBX"}),
            ),
            wja(
                "cached",
                RegisteredType::Mesh,
                serde_json::json!({"source": "c.glb"}),
            ),
            wja(
                "inline",
                RegisteredType::Mesh,
                serde_json::json!({"vertices": []}),
            ),
            wja(
                "skin",
                RegisteredType::SkinnedMesh,
                serde_json::json!({"source": "d.fbx"}),
            ),
            wja(
                "walk",
                RegisteredType::Animation,
                serde_json::json!({"source": "d.glb"}),
            ),
            wja("still", RegisteredType::Animation, serde_json::json!({})),
            wja(
                "box",
                RegisteredType::ProceduralMesh,
                serde_json::json!({"source": "e.glb"}),
            ),
        ];
        assert_eq!(pending_imports(&assets, &hit_cache("cached")), 4);
    }
}
