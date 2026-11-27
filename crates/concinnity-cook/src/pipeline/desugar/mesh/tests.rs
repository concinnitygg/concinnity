use super::*;
use crate::authoring::registry::RegisteredType;
use crate::pipeline::desugar::fixtures::{
    hit_cache, morphing_skinned_glb, quiet, skinned_triangle_fbx, static_triangle_fbx,
};
use crate::pipeline::fixtures::{wja, write_fixture};
use std::collections::HashMap;

#[test]
fn a_source_format_is_read_from_its_extension() {
    assert_eq!(MeshFormat::of("a/b.GLB"), Some(MeshFormat::Gltf));
    assert_eq!(MeshFormat::of("scene.gltf"), Some(MeshFormat::Gltf));
    assert_eq!(MeshFormat::of("hero.Fbx"), Some(MeshFormat::Fbx));
    assert_eq!(MeshFormat::of("mesh.obj"), None);
    assert_eq!(MeshFormat::of("noext"), None);
    assert_eq!(MeshFormat::of(""), None);
}

#[test]
fn gltf_skinned_leaves_inline_and_cached_untouched() {
    let inline_args = serde_json::json!({"vertices": [], "indices": []});
    let cached_args = serde_json::json!({"source": "/no/such/hero.glb"});
    let mut assets = vec![
        wja("inline", RegisteredType::SkinnedMesh, inline_args.clone()),
        wja("cached", RegisteredType::SkinnedMesh, cached_args.clone()),
    ];
    desugar_skinned_meshes(&mut assets, &hit_cache("cached"), None, &quiet()).expect("desugar");
    // No source: untouched. Cache hit: the missing .glb is never parsed
    // and the args stay pre-desugar so the next probe key matches.
    assert_eq!(assets[0].args, inline_args);
    assert_eq!(assets[1].args, cached_args);
}

// A source-backed SkinnedMesh has its geometry and skeleton written into
// the asset's inline args, replacing the `source` reference for the
// compile step that follows.
#[test]
fn gltf_skinned_inlines_geometry_and_skeleton() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = write_fixture(
        &dir,
        "hero.glb",
        &crate::import::glb::test_fixtures::skinned_glb(),
    );
    let mut assets = vec![wja(
        "hero",
        RegisteredType::SkinnedMesh,
        serde_json::json!({"source": src}),
    )];
    desugar_skinned_meshes(&mut assets, &HashMap::default(), None, &quiet()).expect("desugar");

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
fn gltf_skinned_inlines_morph_targets() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = write_fixture(&dir, "hero.glb", &morphing_skinned_glb());
    let mut assets = vec![wja(
        "hero",
        RegisteredType::SkinnedMesh,
        serde_json::json!({"source": src}),
    )];
    desugar_skinned_meshes(&mut assets, &HashMap::default(), None, &quiet()).expect("desugar");

    let args = &assets[0].args;
    assert_eq!(args["morph_target_names"], serde_json::json!(["bulge"]));
    // One target over three vertices: a dense target-major delta block.
    assert_eq!(args["morph_deltas"].as_array().unwrap().len(), 3);
}

#[test]
fn gltf_skinned_missing_source_errors() {
    let mut assets = vec![wja(
        "hero",
        RegisteredType::SkinnedMesh,
        serde_json::json!({"source": "/no/such/hero.glb"}),
    )];
    let err = desugar_skinned_meshes(&mut assets, &HashMap::default(), None, &quiet())
        .expect_err("missing .glb");
    assert!(err.to_string().contains("Asset 'hero'"), "got: {err}");
}

#[test]
fn gltf_mesh_skips_unreadable_formats_and_cache_hits() {
    let obj_args = serde_json::json!({"source": "/no/such/scene.obj"});
    let cached_args = serde_json::json!({"source": "/no/such/scene.glb"});
    let inline_args = serde_json::json!({"vertices": [], "indices": []});
    let mut assets = vec![
        wja("from_obj", RegisteredType::Mesh, obj_args.clone()),
        wja("cached", RegisteredType::Mesh, cached_args.clone()),
        wja("inline", RegisteredType::Mesh, inline_args.clone()),
    ];
    desugar_meshes(&mut assets, &hit_cache("cached"), None, &quiet()).expect("desugar");
    assert_eq!(assets[0].args, obj_args, "no importer reads an .obj source");
    assert_eq!(assets[1].args, cached_args, "cache hit skips the parse");
    assert_eq!(assets[2].args, inline_args, "no source: untouched");
}

#[test]
fn gltf_mesh_missing_source_errors() {
    let mut assets = vec![wja(
        "crate_mesh",
        RegisteredType::Mesh,
        serde_json::json!({"source": "/no/such/scene.glb"}),
    )];
    let err =
        desugar_meshes(&mut assets, &HashMap::default(), None, &quiet()).expect_err("missing .glb");
    assert!(err.to_string().contains("Asset 'crate_mesh'"), "got: {err}");
}

// The text `.gltf` container flows through the same desugar as `.glb`:
// geometry lands inline from the external `.bin` beside the source.
#[test]
fn gltf_mesh_imports_a_text_gltf_with_an_external_buffer() {
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
        RegisteredType::Mesh,
        serde_json::json!({"source": gltf.to_str().unwrap(), "primitive_index": 0}),
    )];
    desugar_meshes(&mut assets, &HashMap::default(), None, &quiet()).expect("desugar");
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
fn gltf_mesh_parses_a_shared_source_once_for_every_asset() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = write_fixture(
        &dir,
        "scene.glb",
        &crate::import::glb::test_fixtures::static_triangle_glb(),
    );
    let mut assets = vec![
        wja(
            "part_a",
            RegisteredType::Mesh,
            serde_json::json!({"source": src, "primitive_index": 0}),
        ),
        wja(
            "part_b",
            RegisteredType::Mesh,
            serde_json::json!({"source": src, "primitive_index": 0}),
        ),
    ];
    desugar_meshes(&mut assets, &HashMap::default(), None, &quiet()).expect("desugar");
    for asset in &assets {
        assert_eq!(asset.args["vertices"].as_array().unwrap().len(), 3);
        assert_eq!(asset.args["indices"].as_array().unwrap().len(), 3);
    }
}

// A primitive the container does not have fails on both the chunked and
// the whole-primitive route, naming the asset either way.
#[test]
fn gltf_mesh_reports_a_primitive_the_file_does_not_have() {
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
        let mut assets = vec![wja("ghost", RegisteredType::Mesh, args)];
        let err = desugar_meshes(&mut assets, &HashMap::default(), None, &quiet())
            .expect_err("primitive 7 does not exist");
        let msg = err.to_string();
        assert!(msg.contains("Asset 'ghost'"), "got: {msg}");
        assert!(msg.contains("glTF import failed"), "got: {msg}");
    }
}

// An oversized primitive fanned into several chunked Mesh assets is split
// exactly once; every asset still gets its own inline geometry.
#[test]
fn gltf_mesh_splits_a_primitive_once_for_every_chunk_asset() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = write_fixture(
        &dir,
        "scene.glb",
        &crate::import::glb::test_fixtures::static_triangle_glb(),
    );
    let chunk = |name: &str| {
        wja(
            name,
            RegisteredType::Mesh,
            serde_json::json!({"source": src, "primitive_index": 0, "chunk_index": 0}),
        )
    };
    let mut assets = vec![chunk("part_a"), chunk("part_b")];
    desugar_meshes(&mut assets, &HashMap::default(), None, &quiet()).expect("desugar");
    for asset in &assets {
        assert_eq!(asset.args["vertices"].as_array().unwrap().len(), 3);
    }
}

// An authored `chunk_index` routes through the u16 chunk split instead of
// the whole-primitive import; an index past the last chunk names the file,
// the primitive, and how many chunks it really produced.
#[test]
fn gltf_mesh_reads_a_chunk_and_rejects_one_out_of_range() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = write_fixture(
        &dir,
        "scene.glb",
        &crate::import::glb::test_fixtures::static_triangle_glb(),
    );
    let mut assets = vec![wja(
        "chunk0",
        RegisteredType::Mesh,
        serde_json::json!({"source": src, "chunk_index": 0}),
    )];
    desugar_meshes(&mut assets, &HashMap::default(), None, &quiet()).expect("desugar");
    assert_eq!(assets[0].args["vertices"].as_array().unwrap().len(), 3);

    let mut past_end = vec![wja(
        "chunk9",
        RegisteredType::Mesh,
        serde_json::json!({"source": src, "chunk_index": 9}),
    )];
    let err = desugar_meshes(&mut past_end, &HashMap::default(), None, &quiet())
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
            RegisteredType::SkinnedMesh,
            serde_json::json!({"source": src, "skin_index": 0}),
        ),
        wja(
            "hair",
            RegisteredType::SkinnedMesh,
            serde_json::json!({"source": src, "skin_index": 1}),
        ),
    ];
    desugar_skinned_meshes(&mut assets, &HashMap::default(), None, &quiet()).expect("desugar");

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
fn fbx_mesh_inlines_geometry_for_every_asset_sharing_a_source() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = write_fixture(&dir, "scene.fbx", &static_triangle_fbx());
    let mut assets = vec![
        wja(
            "part_a",
            RegisteredType::Mesh,
            serde_json::json!({"source": src, "primitive_index": 0}),
        ),
        wja(
            "part_b",
            RegisteredType::Mesh,
            serde_json::json!({"source": src, "primitive_index": 0, "chunk_index": 0}),
        ),
    ];
    desugar_meshes(&mut assets, &HashMap::default(), None, &quiet()).expect("desugar");
    for asset in &assets {
        assert_eq!(asset.args["vertices"].as_array().unwrap().len(), 3);
        assert_eq!(asset.args["indices"].as_array().unwrap(), &vec![0, 1, 2]);
    }
}

// The two failure modes past the parse: a primitive the file does not have,
// and a chunk index past the split.
#[test]
fn fbx_mesh_rejects_an_unknown_primitive_and_chunk() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = write_fixture(&dir, "scene.fbx", &static_triangle_fbx());

    let mut ghost = vec![wja(
        "ghost",
        RegisteredType::Mesh,
        serde_json::json!({"source": src, "primitive_index": 7}),
    )];
    let err = desugar_meshes(&mut ghost, &HashMap::default(), None, &quiet())
        .expect_err("primitive 7 is absent");
    let msg = err.to_string();
    assert!(msg.contains("Asset 'ghost'"), "got: {msg}");
    assert!(msg.contains("FBX import failed"), "got: {msg}");

    let mut past_end = vec![wja(
        "chunk9",
        RegisteredType::Mesh,
        serde_json::json!({"source": src, "chunk_index": 9}),
    )];
    let err = desugar_meshes(&mut past_end, &HashMap::default(), None, &quiet())
        .expect_err("chunk 9 is past the split");
    let msg = err.to_string();
    assert!(msg.contains("chunk_index 9 out of range"), "got: {msg}");
    assert!(msg.contains("1 chunk(s)"), "got: {msg}");
}

// A `.fbx`-sourced SkinnedMesh lands with inline geometry and a skeleton,
// mirroring the glTF pass; the `source` reference is what the compile step
// no longer needs.
#[test]
fn fbx_skinned_inlines_geometry_and_skeleton() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = write_fixture(&dir, "hero.fbx", &skinned_triangle_fbx());
    let mut assets = vec![wja(
        "hero",
        RegisteredType::SkinnedMesh,
        serde_json::json!({"source": src}),
    )];
    desugar_skinned_meshes(&mut assets, &HashMap::default(), None, &quiet()).expect("desugar");

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

// A probe hit means the compiled payload is already in hand, so the missing
// `.fbx` is never parsed.
#[test]
fn fbx_skinned_skips_cache_hits() {
    let cached_args = serde_json::json!({"source": "/no/such/hero.fbx"});
    let mut assets = vec![wja(
        "cached",
        RegisteredType::SkinnedMesh,
        cached_args.clone(),
    )];
    desugar_skinned_meshes(&mut assets, &hit_cache("cached"), None, &quiet()).expect("desugar");
    assert_eq!(assets[0].args, cached_args);
}

// A file with no skin deformer is a hard error, named against the asset.
#[test]
fn fbx_skinned_reports_a_file_without_a_skin() {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = write_fixture(&dir, "static.fbx", &static_triangle_fbx());
    let mut assets = vec![wja(
        "hero",
        RegisteredType::SkinnedMesh,
        serde_json::json!({"source": src}),
    )];
    let err = desugar_skinned_meshes(&mut assets, &HashMap::default(), None, &quiet())
        .expect_err("a static file has no skin");
    let msg = err.to_string();
    assert!(msg.contains("Asset 'hero'"), "got: {msg}");
    assert!(msg.contains("FBX import failed"), "got: {msg}");
}

#[test]
fn fbx_mesh_missing_source_errors() {
    let mut assets = vec![wja(
        "bistro",
        RegisteredType::Mesh,
        serde_json::json!({"source": "/no/such/scene.fbx"}),
    )];
    let err =
        desugar_meshes(&mut assets, &HashMap::default(), None, &quiet()).expect_err("missing .fbx");
    assert!(err.to_string().contains("Asset 'bistro'"), "got: {err}");
}

#[test]
fn fbx_mesh_skips_cache_hits() {
    let cached_args = serde_json::json!({"source": "/no/such/scene.fbx"});
    let mut assets = vec![wja("cached", RegisteredType::Mesh, cached_args.clone())];
    desugar_meshes(&mut assets, &hit_cache("cached"), None, &quiet()).expect("desugar");
    assert_eq!(assets[0].args, cached_args);
}

// A relative source reads under the build's root, for both passes.
#[test]
fn a_relative_fbx_source_reads_under_the_root() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(dir.path().join("models")).expect("models dir");
    std::fs::write(dir.path().join("models/scene.fbx"), static_triangle_fbx())
        .expect("write scene");
    std::fs::write(dir.path().join("models/hero.fbx"), skinned_triangle_fbx()).expect("write hero");
    let mut meshes = vec![wja(
        "part",
        RegisteredType::Mesh,
        serde_json::json!({"source": "models/scene.fbx", "primitive_index": 0}),
    )];
    desugar_meshes(&mut meshes, &HashMap::default(), Some(dir.path()), &quiet())
        .expect("mesh desugar");
    assert_eq!(meshes[0].args["vertices"].as_array().unwrap().len(), 3);
    let mut skinned = vec![wja(
        "hero",
        RegisteredType::SkinnedMesh,
        serde_json::json!({"source": "models/hero.fbx"}),
    )];
    desugar_skinned_meshes(
        &mut skinned,
        &HashMap::default(),
        Some(dir.path()),
        &quiet(),
    )
    .expect("skinned desugar");
    assert!(skinned[0].args["skeleton"].is_array());
}

// One pass reads both formats: a `.glb` and a `.fbx` Mesh in the same slice
// each land with their own inline geometry.
#[test]
fn one_mesh_pass_fills_both_formats() {
    let dir = tempfile::tempdir().expect("tempdir");
    let glb = write_fixture(
        &dir,
        "scene.glb",
        &crate::import::glb::test_fixtures::static_triangle_glb(),
    );
    let fbx = write_fixture(&dir, "scene.fbx", &static_triangle_fbx());
    let mut assets = vec![
        wja(
            "from_glb",
            RegisteredType::Mesh,
            serde_json::json!({"source": glb, "primitive_index": 0}),
        ),
        wja(
            "from_fbx",
            RegisteredType::Mesh,
            serde_json::json!({"source": fbx, "primitive_index": 0}),
        ),
    ];
    desugar_meshes(&mut assets, &HashMap::default(), None, &quiet()).expect("desugar");
    for asset in &assets {
        assert_eq!(
            asset.args["vertices"].as_array().unwrap().len(),
            3,
            "{}",
            asset.id
        );
        assert_eq!(
            asset.args["indices"].as_array().unwrap().len(),
            3,
            "{}",
            asset.id
        );
    }
}

// A SkinnedMesh source with an extension no importer names is read as glTF.
#[test]
fn a_skinned_source_of_unknown_format_reads_as_gltf() {
    let mut assets = vec![wja(
        "hero",
        RegisteredType::SkinnedMesh,
        serde_json::json!({"source": "/no/such/hero.obj"}),
    )];
    let err = desugar_skinned_meshes(&mut assets, &HashMap::default(), None, &quiet())
        .expect_err("missing source");
    let msg = err.to_string();
    assert!(msg.contains("Asset 'hero'"), "got: {msg}");
    assert!(msg.contains("glTF import failed"), "got: {msg}");
}
