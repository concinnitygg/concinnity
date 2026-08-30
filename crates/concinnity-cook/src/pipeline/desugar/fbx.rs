//! FBX-sourced Mesh and SkinnedMesh assets: the counterpart to the glTF passes,
//! reading the same geometry and skeleton out of a `.fbx` document.

use crate::authoring::world::WorldJsonlAsset;

use super::super::pack::MeshCacheEntry;
use super::super::{MESH_TYPE, SKINNED_MESH_TYPE};
use super::skin_index_arg;

// Expand FBX-sourced SkinnedMesh assets in place, mirroring the glTF pass:
// the file's first skinned geometry lands in the asset's inline `vertices` /
// `indices` / `skeleton` args.
pub(in crate::pipeline) fn desugar_fbx_skinned_meshes(
    assets: &mut [WorldJsonlAsset],
    mesh_cache: &std::collections::HashMap<String, MeshCacheEntry>,
) -> std::io::Result<()> {
    for asset in assets.iter_mut() {
        if asset.asset_type != SKINNED_MESH_TYPE {
            continue;
        }
        let source = asset
            .args
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if !source.to_lowercase().ends_with(".fbx") {
            continue;
        }
        if matches!(
            mesh_cache.get(&asset.name),
            Some(MeshCacheEntry { bytes: Some(_), .. })
        ) {
            continue;
        }

        let imported = crate::import::fbx::import_skinned_fbx(&source, skin_index_arg(asset))
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Asset '{}': FBX import failed: {}", asset.name, e),
                )
            })?;

        let name = asset.name.clone();
        let obj = asset.args.as_object_mut().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset '{}': args is not a JSON object", name),
            )
        })?;
        let encode = |field: &str, value: serde_json::Result<serde_json::Value>| {
            value.map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Asset '{}': failed to encode imported {}: {}",
                        name, field, e
                    ),
                )
            })
        };
        obj.insert(
            "vertices".to_string(),
            encode("vertices", serde_json::to_value(&imported.vertices))?,
        );
        obj.insert(
            "indices".to_string(),
            encode("indices", serde_json::to_value(&imported.indices))?,
        );
        obj.insert(
            "skeleton".to_string(),
            encode("skeleton", serde_json::to_value(&imported.skeleton))?,
        );
        tracing::info!(
            "Asset '{}': imported FBX '{}': {} vertices, {} indices, {} joints",
            asset.name,
            source,
            imported.vertices.len(),
            imported.indices.len(),
            imported.skeleton.len()
        );
    }
    Ok(())
}

// Expand FBX-sourced Mesh assets in place: parse the `.fbx` into an FbxScene
// and write the imported geometry into each asset's inline `vertices` /
// `indices` args, keyed by `primitive_index` and optional `chunk_index`. A Mesh
// whose source is not a `.fbx` is left to `desugar_gltf_meshes`. The FBX is
// parsed once per unique path (Bistro fans thousands of Mesh assets out of one
// file) and each primitive's u16 chunk split is memoized.
pub(in crate::pipeline) fn desugar_fbx_meshes(
    assets: &mut [WorldJsonlAsset],
    mesh_cache: &std::collections::HashMap<String, MeshCacheEntry>,
) -> std::io::Result<()> {
    use crate::components::VertexData;
    use crate::import::fbx::FbxScene;
    use std::collections::HashMap;

    type Chunk = (Vec<VertexData>, Vec<u16>);

    let mut parsed_cache: HashMap<String, FbxScene> = HashMap::new();
    let mut chunk_cache: HashMap<(String, u32), Vec<Chunk>> = HashMap::new();

    for asset in assets.iter_mut() {
        if asset.asset_type != MESH_TYPE {
            continue;
        }
        let source = asset
            .args
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if !source.to_lowercase().ends_with(".fbx") {
            continue;
        }
        // Honour the same content-addressed cache the glTF pass uses: a probe
        // hit means the compiled payload is already in hand, so skip the parse.
        if matches!(
            mesh_cache.get(&asset.name),
            Some(MeshCacheEntry { bytes: Some(_), .. })
        ) {
            continue;
        }
        let primitive_index = asset
            .args
            .get("primitive_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let chunk_index = asset
            .args
            .get("chunk_index")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(0);

        if !parsed_cache.contains_key(&source) {
            let scene = crate::import::fbx::parse_fbx(&source).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Asset '{}': FBX import failed: {}", asset.name, e),
                )
            })?;
            parsed_cache.insert(source.clone(), scene);
        }
        let scene = parsed_cache.get(&source).expect("just inserted");

        let key = (source.clone(), primitive_index);
        if !chunk_cache.contains_key(&key) {
            let (verts, indices32) =
                crate::import::fbx::read_primitive_geometry(scene, primitive_index).map_err(
                    |e| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("Asset '{}': FBX import failed: {}", asset.name, e),
                        )
                    },
                )?;
            let chunks = crate::import::glb::split_into_u16_chunks(&verts, &indices32);
            chunk_cache.insert(key.clone(), chunks);
        }
        let chunks = chunk_cache.get(&key).expect("just inserted");
        let chunk = chunks.get(chunk_index).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Asset '{}': chunk_index {} out of range, '{}' primitive {} splits into {} chunk(s)",
                    asset.name,
                    chunk_index,
                    source,
                    primitive_index,
                    chunks.len(),
                ),
            )
        })?;
        let (vertices, indices) = chunk.clone();

        let name = asset.name.clone();
        let obj = asset.args.as_object_mut().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset '{}': args is not a JSON object", name),
            )
        })?;
        let vlen = vertices.len();
        let ilen = indices.len();
        obj.insert(
            "vertices".to_string(),
            serde_json::to_value(&vertices).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Asset '{}': failed to encode imported vertices: {}",
                        name, e
                    ),
                )
            })?,
        );
        obj.insert(
            "indices".to_string(),
            serde_json::to_value(&indices).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Asset '{}': failed to encode imported indices: {}", name, e),
                )
            })?,
        );
        tracing::info!(
            "Asset '{}': imported FBX '{}' primitive {} chunk {}: {} vertices, {} indices",
            asset.name,
            source,
            primitive_index,
            chunk_index,
            vlen,
            ilen,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::desugar::fixtures::{
        hit_cache, skinned_triangle_fbx, static_triangle_fbx,
    };
    use crate::pipeline::fixtures::{wja, write_fixture};

    #[test]
    fn fbx_fixture_parses_into_one_primitive() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_fixture(&dir, "tri.fbx", &static_triangle_fbx());
        let scene = crate::import::fbx::parse_fbx(&src).expect("fixture parses");
        let (vertices, indices) =
            crate::import::fbx::read_primitive_geometry(&scene, 0).expect("primitive 0");
        assert_eq!(vertices.len(), 3);
        assert_eq!(indices, vec![0, 1, 2]);
    }

    // A `.fbx`-sourced Mesh lands with inline geometry, and several assets fanned
    // out of one file share a single parse and a single chunk split.
    #[test]
    fn desugar_fbx_meshes_inlines_geometry_for_every_asset_sharing_a_source() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_fixture(&dir, "scene.fbx", &static_triangle_fbx());
        let mut assets = vec![
            wja(
                "part_a",
                MESH_TYPE,
                serde_json::json!({"source": src, "primitive_index": 0}),
            ),
            wja(
                "part_b",
                MESH_TYPE,
                serde_json::json!({"source": src, "primitive_index": 0, "chunk_index": 0}),
            ),
        ];
        desugar_fbx_meshes(&mut assets, &Default::default()).expect("desugar");
        for asset in &assets {
            assert_eq!(asset.args["vertices"].as_array().unwrap().len(), 3);
            assert_eq!(asset.args["indices"].as_array().unwrap(), &vec![0, 1, 2]);
        }
    }

    // The two failure modes past the parse: a primitive the file does not have,
    // and a chunk index past the split.
    #[test]
    fn desugar_fbx_meshes_rejects_an_unknown_primitive_and_chunk() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_fixture(&dir, "scene.fbx", &static_triangle_fbx());

        let mut ghost = vec![wja(
            "ghost",
            MESH_TYPE,
            serde_json::json!({"source": src, "primitive_index": 7}),
        )];
        let err =
            desugar_fbx_meshes(&mut ghost, &Default::default()).expect_err("primitive 7 is absent");
        let msg = err.to_string();
        assert!(msg.contains("Asset 'ghost'"), "got: {msg}");
        assert!(msg.contains("FBX import failed"), "got: {msg}");

        let mut past_end = vec![wja(
            "chunk9",
            MESH_TYPE,
            serde_json::json!({"source": src, "chunk_index": 9}),
        )];
        let err = desugar_fbx_meshes(&mut past_end, &Default::default())
            .expect_err("chunk 9 is past the split");
        let msg = err.to_string();
        assert!(msg.contains("chunk_index 9 out of range"), "got: {msg}");
        assert!(msg.contains("1 chunk(s)"), "got: {msg}");
    }

    // A `.fbx`-sourced SkinnedMesh lands with inline geometry and a skeleton,
    // mirroring the glTF pass; the `source` reference is what the compile step
    // no longer needs.
    #[test]
    fn desugar_fbx_skinned_meshes_inlines_geometry_and_skeleton() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_fixture(&dir, "hero.fbx", &skinned_triangle_fbx());
        let mut assets = vec![wja(
            "hero",
            SKINNED_MESH_TYPE,
            serde_json::json!({"source": src}),
        )];
        desugar_fbx_skinned_meshes(&mut assets, &Default::default()).expect("desugar");

        let args = &assets[0].args;
        assert_eq!(args["vertices"].as_array().unwrap().len(), 3);
        assert_eq!(args["indices"].as_array().unwrap(), &vec![0, 1, 2]);
        let skeleton = args["skeleton"].as_array().expect("skeleton inlined");
        assert_eq!(skeleton.len(), 1);
        assert_eq!(skeleton[0]["name"], "Root");
        assert_eq!(skeleton[0]["parent"], -1);
        // Every control point binds fully to the single cluster bone.
        assert_eq!(
            args["vertices"][0]["weights"],
            serde_json::json!([1.0, 0.0, 0.0, 0.0])
        );
    }

    // `.glb` sources belong to the glTF pass, and a probe hit means the
    // compiled payload is already in hand; neither is parsed here.
    #[test]
    fn desugar_fbx_skinned_meshes_skips_glb_sources_and_cache_hits() {
        let glb_args = serde_json::json!({"source": "/no/such/hero.glb"});
        let cached_args = serde_json::json!({"source": "/no/such/hero.fbx"});
        let mut assets = vec![
            wja("from_glb", SKINNED_MESH_TYPE, glb_args.clone()),
            wja("cached", SKINNED_MESH_TYPE, cached_args.clone()),
        ];
        desugar_fbx_skinned_meshes(&mut assets, &hit_cache("cached")).expect("desugar");
        assert_eq!(assets[0].args, glb_args);
        assert_eq!(assets[1].args, cached_args);
    }

    // A file with no skin deformer is a hard error, named against the asset.
    #[test]
    fn desugar_fbx_skinned_meshes_reports_a_file_without_a_skin() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_fixture(&dir, "static.fbx", &static_triangle_fbx());
        let mut assets = vec![wja(
            "hero",
            SKINNED_MESH_TYPE,
            serde_json::json!({"source": src}),
        )];
        let err = desugar_fbx_skinned_meshes(&mut assets, &Default::default())
            .expect_err("a static file has no skin");
        let msg = err.to_string();
        assert!(msg.contains("Asset 'hero'"), "got: {msg}");
        assert!(msg.contains("FBX import failed"), "got: {msg}");
    }

    #[test]
    fn desugar_fbx_meshes_missing_source_errors() {
        let mut assets = vec![wja(
            "bistro",
            MESH_TYPE,
            serde_json::json!({"source": "/no/such/scene.fbx"}),
        )];
        let err = desugar_fbx_meshes(&mut assets, &Default::default()).expect_err("missing .fbx");
        assert!(err.to_string().contains("Asset 'bistro'"), "got: {err}");
    }

    #[test]
    fn desugar_fbx_meshes_skips_cache_hits_and_non_fbx_sources() {
        let cached_args = serde_json::json!({"source": "/no/such/scene.fbx"});
        let glb_args = serde_json::json!({"source": "/no/such/scene.glb"});
        let mut assets = vec![
            wja("cached", MESH_TYPE, cached_args.clone()),
            wja("from_glb", MESH_TYPE, glb_args.clone()),
        ];
        desugar_fbx_meshes(&mut assets, &hit_cache("cached")).expect("desugar");
        assert_eq!(assets[0].args, cached_args);
        assert_eq!(assets[1].args, glb_args);
    }
}
