//! glTF-sourced Mesh and SkinnedMesh assets: the referenced `.glb`/`.gltf` is
//! parsed once per source and its geometry, skeleton, and morph targets land in
//! the asset's args.

use std::path::Path;

use crate::authoring::world::WorldJsonlAsset;

use super::super::pack::MeshCacheEntry;
use super::super::{MESH_TYPE, SKINNED_MESH_TYPE};
use super::skin_index_arg;

// Expand glTF-sourced SkinnedMesh assets in place: parse the referenced .glb
// and write the imported geometry + skeleton into the asset's inline
// `vertices` / `indices` / `skeleton` args. A SkinnedMesh with no `source` is
// left untouched, so an inline-authored mesh is byte-for-byte unchanged;
// `.fbx` sources belong to `desugar_fbx_skinned_meshes`.
// Skips an asset whose cache probe found a precompiled payload: there is no
// reason to parse the .glb when the bytes are already in hand.
pub(in crate::pipeline) fn desugar_gltf_skinned_meshes(
    assets: &mut [WorldJsonlAsset],
    mesh_cache: &std::collections::HashMap<String, MeshCacheEntry>,
    assets_dir: Option<&Path>,
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
        let character_model = asset.args.get("character_model").cloned();
        if character_model.is_none()
            && (source.is_empty() || source.to_lowercase().ends_with(".fbx"))
        {
            continue;
        }
        // Cache probe found a compiled payload for this asset, no need
        // to parse the .glb. compile_and_pack_payloads will use the bytes
        // directly. Leave the args un-desugared so they keep matching the
        // pre-desugar cache key on the next build.
        if matches!(
            mesh_cache.get(&asset.name),
            Some(MeshCacheEntry { bytes: Some(_), .. })
        ) {
            continue;
        }

        let invalid = |msg: String| std::io::Error::new(std::io::ErrorKind::InvalidData, msg);
        let imported = match character_model {
            Some(arg) => {
                let arg: crate::compile::character::import::CharacterModelArg =
                    serde_json::from_value(arg).map_err(|e| {
                        invalid(format!("Asset '{}': character_model: {e}", asset.name))
                    })?;
                crate::compile::character::import::import_model(
                    &asset.name,
                    &arg.schema,
                    &arg.model,
                    assets_dir,
                )
                .map_err(invalid)?
            }
            None => {
                crate::import::gltf::import_skinned_glb(&source, skin_index_arg(asset), assets_dir)
                    .map_err(|e| {
                        invalid(format!("Asset '{}': glTF import failed: {}", asset.name, e))
                    })?
            }
        };

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
        if !imported.morph_target_names.is_empty() {
            obj.insert(
                "morph_target_names".to_string(),
                encode(
                    "morph_target_names",
                    serde_json::to_value(&imported.morph_target_names),
                )?,
            );
            obj.insert(
                "morph_deltas".to_string(),
                encode("morph_deltas", serde_json::to_value(&imported.morph_deltas))?,
            );
        }
        obj.remove("character_model");
        tracing::info!(
            "Asset '{}': imported glTF '{}': {} vertices, {} indices, {} joints, {} morph target(s)",
            asset.name,
            if source.is_empty() {
                "character model"
            } else {
                &source
            },
            imported.vertices.len(),
            imported.indices.len(),
            imported.skeleton.len(),
            imported.morph_target_names.len()
        );
    }
    Ok(())
}

// Expand glTF-sourced static `Mesh` assets in place: parse the referenced
// `.glb` and write the imported primitive geometry into the asset's inline
// `vertices` / `indices` args. A Mesh with no `source` is left untouched. The
// GLB is parsed once per unique path; ABeautifulGame fans 35+ Mesh assets out
// of one file, so memoization keeps this O(files) rather than O(primitives).
pub(in crate::pipeline) fn desugar_gltf_meshes(
    assets: &mut [WorldJsonlAsset],
    mesh_cache: &std::collections::HashMap<String, MeshCacheEntry>,
    assets_dir: Option<&Path>,
) -> std::io::Result<()> {
    use crate::components::VertexData;
    use std::collections::HashMap;

    // One split chunk: its vertices and index buffer.
    type Chunk = (Vec<VertexData>, Vec<u16>);

    let mut parsed_cache: HashMap<String, crate::import::gltf_source::GltfDoc> = HashMap::new();
    // Memoize the chunk split per (source, primitive_index) so an oversized
    // primitive that fans into N chunked Mesh assets is split exactly once.
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
        if source.is_empty() {
            continue;
        }
        // `.fbx` sources are handled by `desugar_fbx_meshes`; this pass owns
        // only the glTF containers.
        let lower = source.to_lowercase();
        if !lower.ends_with(".glb") && !lower.ends_with(".gltf") {
            continue;
        }
        // Skip the .glb parse when the cache probe already produced bytes
        // for this asset (see `desugar_gltf_skinned_meshes` for the same
        // pattern). Args stay pre-desugar so the next build's probe hits.
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
            .map(|n| n as usize);

        if !parsed_cache.contains_key(&source) {
            let doc = crate::import::glb::parse_glb(&source, assets_dir).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Asset '{}': glTF import failed: {}", asset.name, e),
                )
            })?;
            parsed_cache.insert(source.clone(), doc);
        }
        let doc = parsed_cache.get(&source).expect("just inserted");

        let (vertices, indices) = if let Some(chunk_idx) = chunk_index {
            let key = (source.clone(), primitive_index);
            if !chunk_cache.contains_key(&key) {
                let (verts, indices32) =
                    crate::import::glb::read_primitive_geometry(doc, &source, primitive_index)
                        .map_err(|e| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                format!("Asset '{}': glTF import failed: {}", asset.name, e),
                            )
                        })?;
                let chunks = crate::import::glb::split_into_u16_chunks(&verts, &indices32);
                chunk_cache.insert(key.clone(), chunks);
            }
            let chunks = chunk_cache.get(&key).expect("just inserted");
            let chunk = chunks.get(chunk_idx).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Asset '{}': chunk_index {} out of range, '{}' primitive {} \
                         splits into {} chunk(s)",
                        asset.name,
                        chunk_idx,
                        source,
                        primitive_index,
                        chunks.len(),
                    ),
                )
            })?;
            chunk.clone()
        } else {
            crate::import::glb::import_static_glb_primitive_from_doc(doc, &source, primitive_index)
                .map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Asset '{}': glTF import failed: {}", asset.name, e),
                    )
                })?
        };

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
        let vlen = vertices.len();
        let ilen = indices.len();
        obj.insert(
            "vertices".to_string(),
            encode("vertices", serde_json::to_value(&vertices))?,
        );
        obj.insert(
            "indices".to_string(),
            encode("indices", serde_json::to_value(&indices))?,
        );
        match chunk_index {
            Some(c) => tracing::info!(
                "Asset '{}': imported glTF '{}' primitive {} chunk {}: {} vertices, {} indices",
                asset.name,
                source,
                primitive_index,
                c,
                vlen,
                ilen,
            ),
            None => tracing::info!(
                "Asset '{}': imported glTF '{}' primitive {}: {} vertices, {} indices",
                asset.name,
                source,
                primitive_index,
                vlen,
                ilen,
            ),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::desugar::fixtures::{hit_cache, morphing_skinned_glb};
    use crate::pipeline::fixtures::{wja, write_fixture};

    #[test]
    fn desugar_gltf_skinned_meshes_leaves_inline_and_cached_untouched() {
        let inline_args = serde_json::json!({"vertices": [], "indices": []});
        let cached_args = serde_json::json!({"source": "/no/such/hero.glb"});
        let mut assets = vec![
            wja("inline", SKINNED_MESH_TYPE, inline_args.clone()),
            wja("cached", SKINNED_MESH_TYPE, cached_args.clone()),
        ];
        desugar_gltf_skinned_meshes(&mut assets, &hit_cache("cached"), None).expect("desugar");
        // No source: untouched. Cache hit: the missing .glb is never parsed
        // and the args stay pre-desugar so the next probe key matches.
        assert_eq!(assets[0].args, inline_args);
        assert_eq!(assets[1].args, cached_args);
    }

    // A source-backed SkinnedMesh has its geometry and skeleton written into
    // the asset's inline args, replacing the `source` reference for the
    // compile step that follows.
    #[test]
    fn desugar_gltf_skinned_meshes_inlines_geometry_and_skeleton() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_fixture(
            &dir,
            "hero.glb",
            &crate::import::glb::test_fixtures::skinned_glb(),
        );
        let mut assets = vec![wja(
            "hero",
            SKINNED_MESH_TYPE,
            serde_json::json!({"source": src}),
        )];
        desugar_gltf_skinned_meshes(&mut assets, &Default::default(), None).expect("desugar");

        let args = &assets[0].args;
        assert_eq!(args["vertices"].as_array().unwrap().len(), 3);
        assert_eq!(args["indices"].as_array().unwrap(), &vec![0, 1, 2]);
        // The two-joint skin is reordered parents-before-children, so the
        // skeleton lands with the root first.
        assert_eq!(args["skeleton"].as_array().unwrap().len(), 2);
        // The fixture carries no morph targets, so no morph args appear.
        assert!(args.get("morph_target_names").is_none());
        assert!(args.get("morph_deltas").is_none());
    }

    // A source carrying morph targets writes the target names and the dense
    // delta block into the asset alongside the base geometry.
    #[test]
    fn desugar_gltf_skinned_meshes_inlines_morph_targets() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_fixture(&dir, "hero.glb", &morphing_skinned_glb());
        let mut assets = vec![wja(
            "hero",
            SKINNED_MESH_TYPE,
            serde_json::json!({"source": src}),
        )];
        desugar_gltf_skinned_meshes(&mut assets, &Default::default(), None).expect("desugar");

        let args = &assets[0].args;
        assert_eq!(args["morph_target_names"], serde_json::json!(["bulge"]));
        // One target over three vertices: a dense target-major delta block.
        assert_eq!(args["morph_deltas"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn desugar_gltf_skinned_meshes_missing_source_errors() {
        let mut assets = vec![wja(
            "hero",
            SKINNED_MESH_TYPE,
            serde_json::json!({"source": "/no/such/hero.glb"}),
        )];
        let err = desugar_gltf_skinned_meshes(&mut assets, &Default::default(), None)
            .expect_err("missing .glb");
        assert!(err.to_string().contains("Asset 'hero'"), "got: {err}");
    }

    #[test]
    fn desugar_gltf_meshes_skips_non_glb_sources_and_cache_hits() {
        let fbx_args = serde_json::json!({"source": "/no/such/scene.fbx"});
        let cached_args = serde_json::json!({"source": "/no/such/scene.glb"});
        let inline_args = serde_json::json!({"vertices": [], "indices": []});
        let mut assets = vec![
            wja("from_fbx", MESH_TYPE, fbx_args.clone()),
            wja("cached", MESH_TYPE, cached_args.clone()),
            wja("inline", MESH_TYPE, inline_args.clone()),
        ];
        desugar_gltf_meshes(&mut assets, &hit_cache("cached"), None).expect("desugar");
        assert_eq!(
            assets[0].args, fbx_args,
            ".fbx sources belong to the fbx pass"
        );
        assert_eq!(assets[1].args, cached_args, "cache hit skips the parse");
        assert_eq!(assets[2].args, inline_args, "no source: untouched");
    }

    #[test]
    fn desugar_gltf_meshes_missing_source_errors() {
        let mut assets = vec![wja(
            "crate_mesh",
            MESH_TYPE,
            serde_json::json!({"source": "/no/such/scene.glb"}),
        )];
        let err =
            desugar_gltf_meshes(&mut assets, &Default::default(), None).expect_err("missing .glb");
        assert!(err.to_string().contains("Asset 'crate_mesh'"), "got: {err}");
    }

    // The text `.gltf` container flows through the same desugar as `.glb`:
    // geometry lands inline from the external `.bin` beside the source.
    #[test]
    fn desugar_gltf_meshes_imports_a_text_gltf_with_an_external_buffer() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("geo.bin"),
            crate::import::glb::test_fixtures::static_triangle_bin(),
        )
        .unwrap();
        let mut json = crate::import::glb::test_fixtures::static_triangle_json();
        json["buffers"][0]["uri"] = "geo.bin".into();
        let gltf = dir.path().join("tri.gltf");
        std::fs::write(&gltf, serde_json::to_vec(&json).unwrap()).unwrap();

        let mut assets = vec![wja(
            "tri",
            MESH_TYPE,
            serde_json::json!({"source": gltf.to_str().unwrap(), "primitive_index": 0}),
        )];
        desugar_gltf_meshes(&mut assets, &Default::default(), None).expect("desugar");
        let vertices = assets[0].args.get("vertices").expect("inline vertices");
        assert_eq!(vertices.as_array().unwrap().len(), 3);
        assert_eq!(
            assets[0]
                .args
                .get("indices")
                .unwrap()
                .as_array()
                .unwrap()
                .len(),
            3
        );
    }

    // Two Mesh assets fanned out of one container parse the file once; both
    // still land with their own inline geometry.
    #[test]
    fn desugar_gltf_meshes_parses_a_shared_source_once_for_every_asset() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_fixture(
            &dir,
            "scene.glb",
            &crate::import::glb::test_fixtures::static_triangle_glb(),
        );
        let mut assets = vec![
            wja(
                "part_a",
                MESH_TYPE,
                serde_json::json!({"source": src, "primitive_index": 0}),
            ),
            wja(
                "part_b",
                MESH_TYPE,
                serde_json::json!({"source": src, "primitive_index": 0}),
            ),
        ];
        desugar_gltf_meshes(&mut assets, &Default::default(), None).expect("desugar");
        for asset in &assets {
            assert_eq!(asset.args["vertices"].as_array().unwrap().len(), 3);
            assert_eq!(asset.args["indices"].as_array().unwrap().len(), 3);
        }
    }

    // A primitive the container does not have fails on both the chunked and
    // the whole-primitive route, naming the asset either way.
    #[test]
    fn desugar_gltf_meshes_reports_a_primitive_the_file_does_not_have() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_fixture(
            &dir,
            "scene.glb",
            &crate::import::glb::test_fixtures::static_triangle_glb(),
        );
        for extra in [serde_json::json!({}), serde_json::json!({"chunk_index": 0})] {
            let mut args = serde_json::json!({"source": src, "primitive_index": 7});
            for (k, v) in extra.as_object().unwrap() {
                args[k] = v.clone();
            }
            let mut assets = vec![wja("ghost", MESH_TYPE, args)];
            let err = desugar_gltf_meshes(&mut assets, &Default::default(), None)
                .expect_err("primitive 7 does not exist");
            let msg = err.to_string();
            assert!(msg.contains("Asset 'ghost'"), "got: {msg}");
            assert!(msg.contains("glTF import failed"), "got: {msg}");
        }
    }

    // An oversized primitive fanned into several chunked Mesh assets is split
    // exactly once; every asset still gets its own inline geometry.
    #[test]
    fn desugar_gltf_meshes_splits_a_primitive_once_for_every_chunk_asset() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_fixture(
            &dir,
            "scene.glb",
            &crate::import::glb::test_fixtures::static_triangle_glb(),
        );
        let chunk = |name: &str| {
            wja(
                name,
                MESH_TYPE,
                serde_json::json!({"source": src, "primitive_index": 0, "chunk_index": 0}),
            )
        };
        let mut assets = vec![chunk("part_a"), chunk("part_b")];
        desugar_gltf_meshes(&mut assets, &Default::default(), None).expect("desugar");
        for asset in &assets {
            assert_eq!(asset.args["vertices"].as_array().unwrap().len(), 3);
        }
    }

    // An authored `chunk_index` routes through the u16 chunk split instead of
    // the whole-primitive import; an index past the last chunk names the file,
    // the primitive, and how many chunks it really produced.
    #[test]
    fn desugar_gltf_meshes_reads_a_chunk_and_rejects_one_out_of_range() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_fixture(
            &dir,
            "scene.glb",
            &crate::import::glb::test_fixtures::static_triangle_glb(),
        );
        let mut assets = vec![wja(
            "chunk0",
            MESH_TYPE,
            serde_json::json!({"source": src, "chunk_index": 0}),
        )];
        desugar_gltf_meshes(&mut assets, &Default::default(), None).expect("desugar");
        assert_eq!(assets[0].args["vertices"].as_array().unwrap().len(), 3);

        let mut past_end = vec![wja(
            "chunk9",
            MESH_TYPE,
            serde_json::json!({"source": src, "chunk_index": 9}),
        )];
        let err = desugar_gltf_meshes(&mut past_end, &Default::default(), None)
            .expect_err("chunk 9 does not exist");
        let msg = err.to_string();
        assert!(msg.contains("Asset 'chunk9'"), "got: {msg}");
        assert!(msg.contains("chunk_index 9 out of range"), "got: {msg}");
        assert!(msg.contains("1 chunk(s)"), "got: {msg}");
    }

    #[test]
    fn desugar_skinned_meshes_import_the_selected_skin() {
        use crate::import::glb::test_fixtures::two_skin_glb;

        let dir = tempfile::tempdir().expect("tempdir");
        let src = write_fixture(&dir, "hero.glb", &two_skin_glb());
        let mut assets = vec![
            wja(
                "body",
                SKINNED_MESH_TYPE,
                serde_json::json!({"source": src, "skin_index": 0}),
            ),
            wja(
                "hair",
                SKINNED_MESH_TYPE,
                serde_json::json!({"source": src, "skin_index": 1}),
            ),
        ];
        desugar_gltf_skinned_meshes(&mut assets, &Default::default(), None).expect("desugar");

        // Each asset inlines its own part's geometry.
        assert_eq!(
            assets[0].args["vertices"][0]["pos"],
            serde_json::json!([0.0, 0.0, 0.0])
        );
        assert_eq!(
            assets[1].args["vertices"][0]["pos"],
            serde_json::json!([5.0, 0.0, 0.0])
        );
    }
}
