//! Per-type dispatch for the compile pass: which `BuildAsset` impl compiles a
//! payload, and which inputs that compile reads.

use crate::registry::RegisteredType;

// Dispatch payload compilation by RegisteredType. Every variant listed below
// has a `BuildAsset` impl in its asset file; the body of each call here is a
// one-liner that delegates to the trait. Adding a new compiled component
// means:
//   1. impl `Component` with `PAYLOAD = AssetPayload::Compiled` for the type
//   2. impl `BuildAsset` for the type in its asset file
//   3. Add one match arm here
pub(super) fn compile_by_type(
    ct: RegisteredType,
    args: &serde_json::Value,
    ctx: &crate::asset::BuildCtx<'_>,
) -> std::io::Result<Vec<u8>> {
    use crate::asset::BuildAsset;
    use crate::components::{File, ProceduralMesh, Room, SdfVolume, Shader, VoxelChunk};
    match ct {
        RegisteredType::ProceduralMesh => {
            <ProceduralMesh as BuildAsset>::compile_payload(args, ctx)
        }
        RegisteredType::VoxelChunk => <VoxelChunk as BuildAsset>::compile_payload(args, ctx),
        RegisteredType::File => <File as BuildAsset>::compile_payload(args, ctx),
        RegisteredType::Room => <Room as BuildAsset>::compile_payload(args, ctx),
        RegisteredType::Shader => <Shader as BuildAsset>::compile_payload(args, ctx),
        RegisteredType::SdfVolume => <SdfVolume as BuildAsset>::compile_payload(args, ctx),
        other => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Asset '{}' is marked Compiled but has no BuildAsset impl (RegisteredType {:?})",
                ctx.name, other
            ),
        )),
    }
}

// Dispatch each asset's payload-cache contribution by RegisteredType. Mirrors
// `compile_by_type` so the cache layer can fold a hash of every input the
// compile reads into its payload key. Types with no `BuildAsset` impl, or with
// the trait default, contribute nothing.
//
// `source_files` and `TARGET_DEPENDENT` are read together per arm: a new
// asset whose payload differs per backend cannot pick up one without the
// other.
pub(super) fn cache_inputs_by_type(
    ct: RegisteredType,
    args: &serde_json::Value,
    ctx: &crate::asset::BuildCtx<'_>,
) -> crate::asset::CacheInputs {
    use crate::asset::{BuildAsset, CacheInputs};
    use crate::components::{File, ProceduralMesh, Room, SdfVolume, Shader, VoxelChunk};
    macro_rules! inputs {
        ($t:ty) => {
            CacheInputs {
                sources: <$t as BuildAsset>::source_files(args, ctx),
                target_dependent: <$t as BuildAsset>::TARGET_DEPENDENT,
            }
        };
    }
    match ct {
        RegisteredType::ProceduralMesh => inputs!(ProceduralMesh),
        RegisteredType::VoxelChunk => inputs!(VoxelChunk),
        RegisteredType::File => inputs!(File),
        RegisteredType::Room => inputs!(Room),
        RegisteredType::Shader => inputs!(Shader),
        RegisteredType::SdfVolume => inputs!(SdfVolume),
        _ => CacheInputs::extra(Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::fixtures::wja;
    use crate::resource_handles::ResourceAssetCompile;

    #[test]
    fn voxel_chunk_payload_compiles_end_to_end() {
        let world = r#"{"name":"scene_shader","type":"Shader","args":{"vertex":{"source":"x.metal"},"fragment":{"source":"x.metal"}}}
{"name":"air","type":"BlockType","args":{"solid":false}}
{"name":"stone","type":"BlockType","args":{"uv_min":[0,0],"uv_max":[1,1]}}
{"name":"chunk","type":"VoxelChunk","args":{"palette":["air","stone"],"dim":[2,1,1],"blocks":[1,1]}}
"#;
        // We can't easily compile shaders here, so go through the geometry
        // entry point directly to verify the voxel chunk produces a non-empty
        // payload for two adjacent solid blocks (10 faces after interior cull).
        let chunk_args = serde_json::json!({
            "palette": ["air", "stone"],
            "dim": [2, 1, 1],
            "blocks": [1, 1],
            "block_size": 1.0,
        });
        let bt = |name: &str| -> Option<serde_json::Value> {
            match name {
                "air" => Some(serde_json::json!({"solid": false})),
                "stone" => Some(serde_json::json!({"uv_min":[0,0],"uv_max":[1,1]})),
                _ => None,
            }
        };
        let bytes = crate::compile::geometry::compile_voxel_chunk_payload(&chunk_args, bt).unwrap();
        assert!(!bytes.is_empty());
        let _ = world; // keeps the inline jsonl reference for documentation
    }

    fn ctx() -> crate::asset::BuildCtx<'static> {
        crate::asset::BuildCtx {
            platform: concinnity_core::platform::Platform::Metal,
            name: "test",
            assets_dir: None,
            artifacts_dir: None,
            all_assets: &[],
        }
    }

    #[test]
    fn compile_by_type_without_build_impl_errors() {
        let ct = RegisteredType::parse("Prop").expect("Prop is a registered component");
        let err = compile_by_type(ct, &serde_json::json!({}), &ctx())
            .expect_err("Prop has no BuildAsset impl");
        assert!(err.to_string().contains("no BuildAsset impl"), "got: {err}");
    }

    #[test]
    fn cache_inputs_by_type_defaults_to_empty_extras() {
        use crate::asset::SourceFiles;
        let ct = RegisteredType::parse("Prop").expect("Prop is a registered component");
        let inputs = cache_inputs_by_type(ct, &serde_json::json!({}), &ctx());
        assert_eq!(inputs.sources, SourceFiles::Extra(Vec::new()));
        assert!(!inputs.target_dependent);
    }

    // The arms that take the trait default report no inputs of their own: every
    // file they read is named by an args string, which the payload cache's
    // generic walk already hashes.
    #[test]
    fn cache_inputs_by_type_covers_the_args_walk_arms() {
        use crate::asset::SourceFiles;
        for name in ["ProceduralMesh", "VoxelChunk", "File", "Room"] {
            let inputs = cache_inputs_by_type(ct(name), &serde_json::json!({}), &ctx());
            assert_eq!(
                inputs.sources,
                SourceFiles::Extra(Vec::new()),
                "{name} must not narrow the generic args walk"
            );
            assert!(
                !inputs.target_dependent,
                "{name} compiles the same everywhere"
            );
        }
    }

    // AudioClip compiles through `RegisteredType` now, not `compile_by_type`
    // (it left the component registry). Its source-less error still surfaces, and
    // its source file is folded into the payload cache key.
    #[test]
    fn resource_asset_types_compile_audio_clip_texture_cubemap_env_lut_and_font() {
        use crate::registry::RegisteredType;
        let rt = RegisteredType::parse("AudioClip").expect("AudioClip is a resource asset");
        let err = rt
            .compile_payload(&serde_json::json!({}), None)
            .expect_err("a source-less AudioClip must fail to compile");
        assert!(err.to_string().contains("missing 'source'"), "got: {err}");
        assert_eq!(
            rt.source_files(&serde_json::json!({"source": "a.wav"}), None),
            vec!["a.wav".to_string()]
        );
        assert!(rt.source_files(&serde_json::json!({}), None).is_empty());

        // Texture is also a resource asset (it left the component registry). A
        // procedural texture compiles a non-empty payload, and a file-backed one
        // folds its source into the payload cache key.
        let tex = RegisteredType::parse("Texture").expect("Texture is a resource asset");
        let bytes = tex
            .compile_payload(
                &serde_json::json!({"generator": "checker", "resolution": 32}),
                None,
            )
            .expect("a procedural texture compiles");
        assert!(!bytes.is_empty());
        assert_eq!(
            tex.source_files(&serde_json::json!({"source": "a.png"}), None),
            vec!["a.png".to_string()]
        );

        // CubemapTexture is a resource asset too. Source-less args fail, and its
        // `.hdr` source folds into the payload cache key.
        let cube =
            RegisteredType::parse("CubemapTexture").expect("CubemapTexture is a resource asset");
        let err = cube
            .compile_payload(&serde_json::json!({}), None)
            .expect_err("a source-less CubemapTexture must fail to compile");
        assert!(
            err.to_string().contains("requires a `source` path"),
            "got: {err}"
        );
        assert_eq!(
            cube.source_files(&serde_json::json!({"source": "c.hdr"}), None),
            vec!["c.hdr".to_string()]
        );

        // EnvironmentMap and ColorLut are resource assets too. Both surface their
        // source-less error through `RegisteredType::compile_payload`, and fold
        // their `source` into the payload cache key.
        let env =
            RegisteredType::parse("EnvironmentMap").expect("EnvironmentMap is a resource asset");
        let err = env
            .compile_payload(&serde_json::json!({}), None)
            .expect_err("a source-less EnvironmentMap must fail to compile");
        assert!(
            err.to_string()
                .contains("requires either `source` or `generator`"),
            "got: {err}"
        );
        assert_eq!(
            env.source_files(&serde_json::json!({"source": "e.hdr"}), None),
            vec!["e.hdr".to_string()]
        );

        let lut = RegisteredType::parse("ColorLut").expect("ColorLut is a resource asset");
        let err = lut
            .compile_payload(&serde_json::json!({}), None)
            .expect_err("a source-less ColorLut must fail to compile");
        assert!(
            err.to_string().contains("requires a `source` path"),
            "got: {err}"
        );
        assert_eq!(
            lut.source_files(&serde_json::json!({"source": "l.cube"}), None),
            vec!["l.cube".to_string()]
        );

        // Font is a resource asset. The built-in font (empty `path`) compiles a
        // non-empty atlas, and a file-backed font folds its `path` (not `source`)
        // into the payload cache key.
        let font = RegisteredType::parse("Font").expect("Font is a resource asset");
        let bytes = font
            .compile_payload(&serde_json::json!({"size_px": 20}), None)
            .expect("the built-in font compiles");
        assert!(!bytes.is_empty());
        assert_eq!(
            font.source_files(&serde_json::json!({"path": "f.ttf"}), None),
            vec!["f.ttf".to_string()]
        );
        assert!(
            font.source_files(&serde_json::json!({"source": "x.ttf"}), None)
                .is_empty()
        );
    }

    // A mesh whose source is a text `.gltf` reads sibling files the args never
    // name; `source_files` must report them so an edited external buffer or
    // image busts the payload cache.
    #[test]
    fn gltf_sources_fold_referenced_sibling_files_into_source_files() {
        use crate::registry::RegisteredType;

        let dir = tempfile::tempdir().unwrap();
        let json = serde_json::json!({
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": 4, "uri": "geo.bin"}],
            "images": [{"uri": "albedo.png"}]
        });
        let gltf_path = dir.path().join("tri.gltf");
        std::fs::write(&gltf_path, serde_json::to_vec(&json).unwrap()).unwrap();
        let src = gltf_path.to_str().unwrap().to_string();

        for rt in [RegisteredType::Mesh, RegisteredType::SkinnedMesh] {
            let files = rt.source_files(&serde_json::json!({"source": src}), None);
            assert_eq!(files.len(), 3, "{rt:?}: {files:?}");
            assert_eq!(files[0], src);
            assert!(files.iter().any(|f| f.ends_with("geo.bin")), "{files:?}");
            assert!(files.iter().any(|f| f.ends_with("albedo.png")), "{files:?}");
        }

        // A `.glb` source reports only itself.
        let glb =
            RegisteredType::Mesh.source_files(&serde_json::json!({"source": "scene.glb"}), None);
        assert_eq!(glb, vec!["scene.glb".to_string()]);
    }

    // Dispatch coverage: compile_by_type / source_files_by_type route each
    // compiled RegisteredType to its asset_impls wrapper.

    fn ct(name: &str) -> RegisteredType {
        RegisteredType::parse(name).unwrap_or_else(|| panic!("{name} is a registered component"))
    }

    // Arms whose outcome is deterministic from inline args alone: a valid
    // minimal payload for the ones that need no source file, and the expected
    // error for the ones that require a source but got none.
    #[test]
    fn compile_by_type_dispatches_deterministic_arms() {
        // Mesh is a resource asset now: it compiles through
        // `RegisteredType::compile_payload`, not the RegisteredType dispatch.
        let mesh_bytes = crate::registry::RegisteredType::Mesh
            .compile_payload(
                &serde_json::json!({"generator": "box", "half_extents": [1, 1, 1]}),
                None,
            )
            .expect("Mesh compiles through the resource path");
        assert!(!mesh_bytes.is_empty());

        let ok_cases: &[(&str, serde_json::Value)] = &[
            (
                "ProceduralMesh",
                serde_json::json!({"generator": "sphere", "radius": 1.0}),
            ),
            ("Room", serde_json::json!({})),
        ];
        for case in ok_cases {
            let name = case.0;
            let args = &case.1;
            let bytes = compile_by_type(ct(name), args, &ctx())
                .unwrap_or_else(|e| panic!("{name} should compile: {e}"));
            assert!(!bytes.is_empty(), "{name} payload should be non-empty");
        }

        let err_cases: &[(&str, serde_json::Value, &str)] =
            &[("File", serde_json::json!({}), "unsupported File kind")];
        for case in err_cases {
            let name = case.0;
            let args = &case.1;
            let needle = case.2;
            let err = compile_by_type(ct(name), args, &ctx())
                .expect_err(&format!("{name} with empty args should error"));
            assert!(
                err.to_string().contains(needle),
                "{name} error should mention '{needle}', got: {err}"
            );
        }
    }

    // The File wrapper decodes an OBJ mesh source into a non-empty payload.
    #[test]
    fn compile_by_type_file_compiles_an_obj_source() {
        let dir = tempfile::tempdir().expect("tempdir");
        let obj = dir.path().join("tri.obj");
        std::fs::write(&obj, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").expect("write obj");
        let args = serde_json::json!({"path": obj.to_str().unwrap(), "kind": "obj"});
        let bytes = compile_by_type(ct("File"), &args, &ctx()).expect("obj compiles");
        assert!(!bytes.is_empty());
    }

    // The SkinnedMesh resource compiler deserialises args + an optional
    // skeleton, then bakes geometry: one vertex is enough for a payload, no
    // vertices and a malformed skeleton are the two error arms. Its baked
    // data form carries the interned name id and drops the geometry.
    #[test]
    fn skinned_mesh_resource_compile_paths() {
        use crate::registry::RegisteredType;
        let rt = RegisteredType::SkinnedMesh;

        let ok = serde_json::json!({"vertices": [{"pos": [0.0, 0.0, 0.0]}], "indices": []});
        let bytes = rt.compile_payload(&ok, None).expect("skinned compiles");
        assert!(!bytes.is_empty());

        let no_verts = rt
            .compile_payload(&serde_json::json!({}), None)
            .expect_err("no vertices");
        assert!(
            no_verts.to_string().contains("at least one vertex"),
            "got: {no_verts}"
        );

        let bad_skeleton = rt
            .compile_payload(
                &serde_json::json!({"vertices": [{"pos": [0.0, 0.0, 0.0]}], "skeleton": 5}),
                None,
            )
            .expect_err("malformed skeleton");
        assert!(
            bad_skeleton.to_string().contains("invalid skeleton args"),
            "got: {bad_skeleton}"
        );

        // The baked data tuple: name id first, then the clamped mesh with its
        // geometry cleared.
        crate::ecs::asset_id::reset_interner();
        let name_id = crate::ecs::asset_id::intern("hero");
        let data = rt
            .compile_data(
                "hero",
                &serde_json::json!({
                    "vertices": [{"pos": [0.0, 0.0, 0.0]}],
                    "scale": [0.0, 0.0, 0.0],
                    "max_instances": 999999,
                    "capsule": {"half_height": 0.6, "radius": 0.2},
                }),
            )
            .expect("data bakes")
            .expect("skinned mesh carries baked data");
        let (baked_name, sm): (u32, crate::components::SkinnedMesh) =
            postcard::from_bytes(&data).unwrap();
        assert_eq!(baked_name, name_id.0);
        assert_eq!(sm.scale, [1.0, 1.0, 1.0], "zero scale clamps to unit");
        assert_eq!(sm.max_instances, 4096, "reserve caps at 4096");
        assert!(sm.vertices.is_empty(), "geometry rides the payload");
        assert!(sm.capsule.is_some());
    }

    // The VoxelChunk wrapper resolves its palette from sibling BlockType assets
    // in the build context.
    #[test]
    fn compile_by_type_voxel_chunk_resolves_palette_from_ctx() {
        let blocks = vec![
            wja("air", "BlockType", serde_json::json!({"solid": false})),
            wja(
                "stone",
                "BlockType",
                serde_json::json!({"uv_min": [0, 0], "uv_max": [1, 1]}),
            ),
        ];
        let vctx = crate::asset::BuildCtx {
            platform: concinnity_core::platform::Platform::Metal,
            name: "chunk",
            assets_dir: None,
            artifacts_dir: None,
            all_assets: &blocks,
        };
        let args = serde_json::json!({
            "palette": ["air", "stone"],
            "dim": [2, 1, 1],
            "blocks": [1, 1],
            "block_size": 1.0,
        });
        let bytes = compile_by_type(ct("VoxelChunk"), &args, &vctx).expect("voxel compiles");
        assert!(!bytes.is_empty());
    }

    // The SdfVolume wrapper transports the current backend's fragment shader
    // bytes verbatim (no MSL/GLSL compilation); a missing source is a hard
    // error rather than a silent empty payload.
    #[test]
    fn compile_by_type_sdf_volume_transports_shader_bytes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let shader = dir.path().join("blob.metal");
        let source = b"// sdf fragment source\n";
        std::fs::write(&shader, source).expect("write shader");
        let path = shader.to_str().unwrap();
        // Set every backend's key to the same file so the test is
        // platform-independent: only the current backend's entry is read.
        let args = serde_json::json!({
            "fragment_shaders": {"metal": path, "hlsl": path, "glsl": path}
        });
        let bytes = compile_by_type(ct("SdfVolume"), &args, &ctx()).expect("sdf reads source");
        assert_eq!(bytes, source);

        let err = compile_by_type(ct("SdfVolume"), &serde_json::json!({}), &ctx())
            .expect_err("no fragment shader source");
        assert!(
            err.to_string().contains("no fragment shader source"),
            "got: {err}"
        );
    }

    // The Shader wrapper's non-compiling arms: a missing stage source is
    // either a hard error (Metal/HLSL) or the inline-GLSL stub (Vulkan).
    // Neither shells out to a shader toolchain, so the test stays
    // backend-agnostic.
    #[test]
    fn compile_by_type_shader_missing_source_does_not_shell_out() {
        let out = compile_by_type(
            ct("Shader"),
            &serde_json::json!({"vertex": {}, "fragment": {}}),
            &ctx(),
        );
        match out {
            Ok(bytes) => {
                let payload = concinnity_core::components::ShaderPayload::decode(&bytes)
                    .expect("empty container decodes");
                assert!(payload.stages.is_empty(), "glsl stub compiles no stages");
            }
            Err(e) => assert!(e.to_string().contains("no shader source"), "got: {e}"),
        }
    }

    // cache_inputs_by_type routes to the two overriding wrappers. Both report
    // `Only` -- the complete input set the current backend reads -- so an edit
    // to a sibling backend's shader leaves this backend's payload cached.
    #[test]
    fn cache_inputs_by_type_covers_the_overriding_wrappers() {
        use crate::asset::SourceFiles;
        let dir = tempfile::tempdir().expect("tempdir");
        let shader = dir.path().join("blob.metal");
        std::fs::write(&shader, b"x").expect("write shader");
        let path = shader.to_str().unwrap();

        // SdfVolume reports the resolved path for the current backend, and
        // transports it verbatim, so the compile target is not an input.
        let sdf_args = serde_json::json!({
            "fragment_shaders": {"metal": path, "hlsl": path, "glsl": path}
        });
        let sdf = cache_inputs_by_type(ct("SdfVolume"), &sdf_args, &ctx());
        assert_eq!(sdf.sources, SourceFiles::Only(vec![path.to_string()]));
        assert!(!sdf.target_dependent);
        assert_eq!(
            cache_inputs_by_type(ct("SdfVolume"), &serde_json::json!({}), &ctx()).sources,
            SourceFiles::Only(Vec::new())
        );

        // Shader compiles its stage sources, so the target is an input.
        let no_source = cache_inputs_by_type(ct("Shader"), &serde_json::json!({}), &ctx());
        assert_eq!(no_source.sources, SourceFiles::Only(Vec::new()));
        assert!(no_source.target_dependent);
    }
}
