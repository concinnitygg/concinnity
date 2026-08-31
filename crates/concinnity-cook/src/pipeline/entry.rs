//! The entry points of the compile stage, and the run they drive: probe the
//! payload cache, desugar source-backed assets, intern names, resolve scene
//! references, then compile and pack.

use std::path::Path;

use concinnity_core::platform::Platform;

use crate::asset_api::{self, AssetRequest};
use crate::authoring::world::WorldJsonlAsset;
use crate::ecs::{BlobAssetDef, asset_id};

use super::desugar::{
    desugar_animation_imports, desugar_fbx_meshes, desugar_fbx_skinned_meshes, desugar_gltf_meshes,
    desugar_gltf_skinned_meshes, desugar_root_motion,
};
use super::errors_to_io;
use super::pack::{PackContext, compile_and_pack_payloads, probe_mesh_payload_cache};
use super::result::{MeshSourceInfo, PipelineResult, TextureSourceInfo};
use super::scene_refs::resolve_scene_refs;

/// Build the world at `json_path` for `platform` into `tree` and write its
/// blobs to disk: sources resolve under the tree's `assets/` and the blobs land
/// in its `data/`. The whole-tree entry point; a host that builds against
/// unrelated directories calls
/// [`prepare_world`](crate::build_only::prepare_world) + [`build_compiled`].
pub fn build_from_path(
    tree: &crate::paths::StateTree,
    json_path: &str,
    platform: Platform,
) -> std::io::Result<()> {
    let content = std::fs::read_to_string(json_path)?;
    let assets_dir = tree.assets_dir();
    let loaded = crate::build_only::prepare_world(&content, Some(&assets_dir), platform)
        .map_err(|errs| crate::check::report_validation_errors(&errs))?;

    let result = build_compiled(loaded.assets, Some(&assets_dir), None, platform)?;

    let pack_result = write_build_outputs(tree, &result, &loaded.injected, &loaded.shadowed)?;
    for (blob_idx, path) in pack_result.blob_paths.iter().enumerate() {
        let payload_bytes = result.payloads.get(blob_idx).map(|b| b.len()).unwrap_or(0);
        println!("Wrote {} ({} payload bytes)", path, payload_bytes);
    }

    if result.cache_hits + result.cache_misses > 0 {
        println!(
            "Build cache: {} reused, {} compiled",
            result.cache_hits, result.cache_misses
        );
    }

    if !loaded.injected.is_empty() {
        println!(
            "Injected {} default asset(s) (see world-lock.json)",
            loaded.injected.len()
        );
    }
    println!("Wrote world-lock.json");

    Ok(())
}

/// Write a compiled world's blob files, naming the primary blob `primary`.
/// Every overflow payload blob is written as its sibling named by index, which
/// is the layout the runtime reads back. No lock file and no thumbnails: this
/// is the blob output alone.
pub fn write_blobs_to(
    result: &PipelineResult,
    primary: &std::path::Path,
) -> std::io::Result<crate::blob::PackResult> {
    crate::blob::write_blobs(
        crate::blob::BlobStreams {
            defs: &result.defs,
            resources: &result.resources,
            scene_groups: &result.scene_groups,
            mesh_bounds: &result.mesh_bounds,
            physics_budget: result.physics_budget,
        },
        &result.payloads,
        primary,
    )
}

/// Write the blobs and world-lock.json for a compiled world into `tree`: the
/// shared build tail used by the CLI and the FFI host. The lock records each
/// asset under its real name plus every injected default with its full args.
pub fn write_build_outputs(
    tree: &crate::paths::StateTree,
    result: &PipelineResult,
    injected: &[crate::build_only::InjectedAsset],
    shadowed: &[crate::build_only::ShadowedAsset],
) -> std::io::Result<crate::blob::PackResult> {
    let pack_result = write_blobs_to(
        result,
        &concinnity_host::store::blob::primary_in(&tree.data_dir()),
    )?;
    let named_refs: Vec<(&str, &BlobAssetDef)> = result
        .names
        .iter()
        .map(|n| n.as_str())
        .zip(result.defs.iter())
        .collect();
    crate::blob::write_lock(
        &named_refs,
        &result.resource_locks,
        injected,
        shadowed,
        &pack_result.blob_paths,
    )?;
    // Thumbnails are a best-effort side product: they are rendered after the
    // blobs the build exists to produce, and land in the build cache segment
    // beside the payloads they were rendered from.
    let thumbs = crate::compile::thumbnail::bake_thumbnails(result);
    if thumbs.baked > 0 {
        println!(
            "Baked {} thumbnail(s) ({} reused)",
            thumbs.baked, thumbs.reused
        );
    }
    Ok(pack_result)
}

/// Run the full build pipeline on an in-memory JSONL string without writing any
/// blobs. Loads, expands, and validates the world (crate::build_only::prepare_world),
/// then compiles it. `assets_dir` is the asset search root a bare `source`
/// filename is searched under; `artifacts_dir` is an optional directory
/// consulted when resolving bare shader filenames not found there, so pass the
/// account's artifact directory to compile user-written shaders.
pub fn build_pipeline_from_str(
    content: &str,
    assets_dir: Option<&Path>,
    artifacts_dir: Option<&str>,
    platform: Platform,
) -> std::io::Result<PipelineResult> {
    let loaded =
        crate::build_only::prepare_world(content, assets_dir, platform).map_err(errors_to_io)?;
    build_compiled(loaded.assets, assets_dir, artifacts_dir, platform)
}

/// A progress report from the compile pipeline: the stage's name and its
/// done / total counts. `total == 0` marks a stage that cannot count its work
/// (progress there is indeterminate).
#[derive(Debug, Clone, Copy)]
pub struct BuildProgress {
    /// The stage's name.
    pub stage: &'static str,
    /// Work completed in this stage.
    pub done: u32,
    /// Total work in this stage; 0 when the stage cannot count it.
    pub total: u32,
}

/// Compile an already-prepared world (expanded + structurally and semantically
/// validated) into in-memory blobs. This is the compile-only stage; it assumes
/// the assets have passed crate::build_only::prepare_world, which should have been
/// given the same `assets_dir`: an asset resolves its source the same way in
/// both halves.
pub fn build_compiled(
    assets: Vec<WorldJsonlAsset>,
    assets_dir: Option<&Path>,
    artifacts_dir: Option<&str>,
    platform: Platform,
) -> std::io::Result<PipelineResult> {
    build_compiled_with_progress(assets, assets_dir, artifacts_dir, platform, None)
}

/// [`build_compiled`] with a progress callback. The callback fires from the
/// desugar stage and, concurrently, from the parallel payload compile (hence
/// `Sync`); it must be cheap and non-blocking.
pub fn build_compiled_with_progress(
    mut assets: Vec<WorldJsonlAsset>,
    assets_dir: Option<&Path>,
    artifacts_dir: Option<&str>,
    platform: Platform,
    progress: Option<&(dyn Fn(BuildProgress) + Sync)>,
) -> std::io::Result<PipelineResult> {
    if let Some(p) = progress {
        p(BuildProgress {
            stage: "desugar",
            done: 0,
            total: 0,
        });
    }

    // Cache probe runs before desugar. For every glTF-sourced Mesh /
    // SkinnedMesh, hash the un-desugared args + referenced .glb and look up
    // the compiled payload by that key. On a hit, we hold the bytes and skip
    // the .glb parse entirely (the original goal: an unchanged source file
    // means no work). On a miss, the recorded key is used when the compile
    // step stores the freshly produced payload, so the next build's probe
    // can re-use it.
    let mesh_cache = probe_mesh_payload_cache(&assets, assets_dir, artifacts_dir, platform);

    // Expand any glTF-sourced SkinnedMesh and Mesh assets into inline geometry
    // before anything else looks at their args. Animations expand after the
    // skinned-mesh pass so an importer that wanted to share state could read
    // already-imported skeletons; today both passes parse the .glb fresh,
    // but the ordering keeps that option open without an API churn.
    desugar_gltf_skinned_meshes(&mut assets, &mesh_cache, assets_dir)?;
    desugar_fbx_skinned_meshes(&mut assets, &mesh_cache)?;
    desugar_gltf_meshes(&mut assets, &mesh_cache, assets_dir)?;
    desugar_fbx_meshes(&mut assets, &mesh_cache)?;
    desugar_animation_imports(&mut assets, assets_dir)?;
    desugar_root_motion(&mut assets)?;
    crate::compile::character_shape::warn_unresolved(&assets);
    crate::compile::character::bake::bake_shapes(&mut assets, |name| {
        mesh_cache
            .get(name)
            .map(|e| crate::compile::character::bake::TargetCache {
                key: e.key.clone(),
                replayed: e.capsule_scale,
            })
    })?;

    // Intern every asset name to a dense AssetId in declaration order, then
    // resolve the scene-by-naming-convention references that the runtime can
    asset_id::reset_interner();
    let names: Vec<&str> = assets.iter().map(|a| a.name.as_str()).collect();
    asset_id::intern_all(&names);
    resolve_scene_refs(&mut assets);

    // Assign each resource its dense per-kind handle in declaration order and
    // install the map so resource references resolve during the reserialize pass
    // below: texture references (Material.albedo, Room.*_texture,
    // Decal/ParticleEmitter.texture) to a `TextureHandle`, and audio-clip
    // references (AudioEmitter.clip, AudioCue.clip, Story music/sounds) to an
    // `AudioClipHandle`. The assignment walks this same `assets` list that the
    // blob is emitted from, so a resource's handle equals the position the
    // runtime encounters it (a texture's albedo pool slot, an audio clip's drain
    // index / resource-table slot).
    crate::resource_handles::reset_resource_handles();
    let resource_assets = assets.iter().filter_map(|a| {
        crate::resource_handles::asset_resource_kind(&a.asset_type)
            .map(|kind| (asset_id::intern(&a.name), kind))
    });
    let mut resource_handles =
        crate::resource_handles::ResourceHandles::from_assets(resource_assets);
    // The mesh-source handle space spans four kinds (Mesh, ProceduralMesh,
    // VoxelChunk, mesh-kind File) and File is polymorphic, so it is assigned in a
    // second pass in the fixed block order the runtime enumerates mesh sources
    // rather than through the per-type classifier above.
    crate::resource_handles::assign_mesh_source_handles(&mut resource_handles, &assets);
    // Shader handles walk the same list, so a Material's `shader` reference
    // bakes to the position the runtime encounters that Shader at drain time.
    crate::resource_handles::assign_shader_handles(&mut resource_handles, &assets);
    // Install a clone; the original is kept to look up each resource asset's
    // handle while partitioning below.
    crate::resource_handles::install_resource_handles(resource_handles.clone());

    // Partition the world into component assets (each becomes a `BlobAssetDef`)
    // and resource assets (each becomes a resource-stream record). A resource
    // asset (AudioClip) has left the component registry, so it never goes through
    // `create_asset_def`; it is compiled + packed as a resource below. `named` is
    // therefore no longer 1:1 with `assets`, so `named_src[i]` records the source
    // asset index of each component def.
    use crate::registry::RegisteredType;
    let mut named: Vec<(String, BlobAssetDef)> = Vec::new();
    let mut named_src: Vec<usize> = Vec::new();
    let mut resource_jobs: Vec<(usize, RegisteredType, u32)> = Vec::new();
    for (i, asset) in assets.iter().enumerate() {
        if let Some((rt, kind)) =
            RegisteredType::parse(&asset.asset_type).and_then(|t| t.resource_kind().map(|k| (t, k)))
        {
            let id = asset_id::intern(&asset.name);
            let handle = resource_handles
                .get(kind, id)
                .expect("resource asset was assigned a handle above");
            resource_jobs.push((i, rt, handle));
            continue;
        }
        let req = AssetRequest {
            asset_type: asset.asset_type.clone(),
            args: Some(asset.args.clone()),
        };
        let mut def = asset_api::create_asset_def(&req, platform).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset '{}': {}", asset.name, e),
            )
        })?;
        def.name = Some(asset_id::intern(&asset.name));
        named.push((asset.name.clone(), def));
        named_src.push(i);
    }

    // Dev-only: the file source behind each texture handle, so `cn debug`'s
    // hot-reload watcher can map a saved file back to its handle. Built in
    // handle order from the same resource jobs; a procedural texture (generator
    // set) leaves an empty source (nothing to watch).
    let texture_count = resource_jobs
        .iter()
        .filter(|(_, rt, _)| *rt == RegisteredType::Texture)
        .map(|(_, _, h)| *h as usize + 1)
        .max()
        .unwrap_or(0);
    let mut texture_sources = vec![TextureSourceInfo::default(); texture_count];
    for (asset_idx, rt, handle) in &resource_jobs {
        if *rt != RegisteredType::Texture {
            continue;
        }
        let asset = &assets[*asset_idx];
        let generator = asset
            .args
            .get("generator")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let (source, image_index) = if generator.is_empty() {
            (
                asset
                    .args
                    .get("source")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                asset
                    .args
                    .get("image_index")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32,
            )
        } else {
            (String::new(), 0)
        };
        texture_sources[*handle as usize] = TextureSourceInfo {
            name_id: asset_id::intern(&asset.name).0,
            source,
            image_index,
        };
    }

    // Dev-only: the file source behind each mesh handle, so `cn debug`'s
    // hot-reload watcher can re-import a saved `.glb`/`.fbx` into its draw
    // slots. Mesh handles are dense from 0 (the Mesh block leads the shared
    // mesh-source space); an inline-authored mesh leaves an empty source.
    let mesh_count = resource_jobs
        .iter()
        .filter(|(_, rt, _)| *rt == RegisteredType::Mesh)
        .map(|(_, _, h)| *h as usize + 1)
        .max()
        .unwrap_or(0);
    let mut mesh_sources = vec![MeshSourceInfo::default(); mesh_count];
    for (asset_idx, rt, handle) in &resource_jobs {
        if *rt != RegisteredType::Mesh {
            continue;
        }
        let args = &assets[*asset_idx].args;
        let str_arg = |key: &str| {
            args.get(key)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };
        let u32_arg = |key: &str, default: u32| {
            args.get(key)
                .and_then(|v| v.as_u64())
                .unwrap_or(default as u64) as u32
        };
        mesh_sources[*handle as usize] = MeshSourceInfo {
            source: str_arg("source"),
            primitive_index: u32_arg("primitive_index", 0),
            lod_levels: u32_arg("lod_levels", 1),
            lod_distances: args
                .get("lod_distances")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|d| d.as_f64())
                        .map(|d| d as f32)
                        .collect()
                })
                .unwrap_or_default(),
        };
    }

    // Scene payload ownership, derived from the resolved scene memberships and
    // the reference graph; drives the grouped packing below.
    let partition = crate::compile::scene_partition::partition_scenes(&assets);

    // The world's physics reservation, counted from the same fully expanded
    // asset list the blob is emitted from.
    let physics_budget = crate::compile::physics_budget::compute(&assets);
    crate::compile::physics_budget::report_spawn_reservation(&assets);

    let compiled = compile_and_pack_payloads(
        &mut named,
        &named_src,
        PackContext {
            assets: &assets,
            resource_jobs: &resource_jobs,
            partition: &partition,
            mesh_source_handles: &resource_handles,
            max_blob_bytes: crate::blob::DEFAULT_MAX_BLOB_BYTES,
            assets_dir,
            artifacts_dir,
            platform,
            mesh_cache: &mesh_cache,
            progress,
        },
    )?;

    // Lock-file provenance for the resource stream: `compiled.resources` is
    // emitted in `resource_jobs` order, so the two zip index-aligned. Texture
    // and Mesh records also carry their hot-reload source info so a blob boot
    // can reconstruct the catalogues without the authored args.
    let resource_locks: Vec<crate::blob::LockedResource> = resource_jobs
        .iter()
        .zip(compiled.resources.iter())
        .map(|((asset_idx, rt, handle), record)| {
            let asset = &assets[*asset_idx];
            crate::blob::LockedResource {
                name: asset.name.clone(),
                // Already interned by the declaration-order pass above, so
                // this is a lookup of the id the build assigned.
                id: Some(asset_id::intern(&asset.name).0),
                kind: rt.as_str().to_string(),
                handle: *handle,
                args_hash: crate::blob::checksum(asset.args.to_string().as_bytes()),
                payload_blob: record.payload.as_ref().map(|p| p.blob_index),
                texture_source: (*rt == RegisteredType::Texture).then(|| {
                    let t = &texture_sources[*handle as usize];
                    crate::blob::LockedTextureSource {
                        source: t.source.clone(),
                        image_index: t.image_index,
                    }
                }),
                mesh_source: (*rt == RegisteredType::Mesh).then(|| {
                    let m = &mesh_sources[*handle as usize];
                    crate::blob::LockedMeshSource {
                        source: m.source.clone(),
                        primitive_index: m.primitive_index,
                        lod_levels: m.lod_levels,
                        lod_distances: m.lod_distances.clone(),
                    }
                }),
            }
        })
        .collect();

    // The blob carries components (emitted in declaration order) plus the
    // resource stream. (System run order is no longer a build concern: every
    // system is internal client code ordered by the client's
    // `World::start` schedule.)
    let (names, defs): (Vec<String>, Vec<BlobAssetDef>) = named.into_iter().unzip();

    // The compile is done producing entries, so the segment holding them is
    // written once, here, rather than per payload while the compile runs.
    crate::cache::flush();

    Ok(PipelineResult {
        defs,
        names,
        resources: compiled.resources,
        scene_groups: compiled.scene_groups,
        mesh_bounds: compiled.mesh_bounds,
        physics_budget,
        mesh_component_names: compiled.mesh_component_names,
        payloads: compiled.blobs,
        cache_hits: compiled.cache_hits,
        cache_misses: compiled.cache_misses,
        texture_sources,
        mesh_sources,
        resource_locks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::MESH_TYPE;
    use crate::pipeline::fixtures::{SHADER_BUILD_LOCK, wja, write_fixture};

    #[test]
    fn build_pipeline_interns_names_and_resolves_refs() {
        let _guard = SHADER_BUILD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::compile::shader::install_stub_toolchain();
        // box=0, day=1, day_crate=2 in declaration order.
        let world = concat!(
            r#"{"name":"box","type":"ProceduralMesh","args":{"generator":"box","half_extents":[1,1,1]}}"#,
            "\n",
            r#"{"name":"day","type":"Scene","args":{}}"#,
            "\n",
            r#"{"name":"day_crate","type":"Prop","args":{"mesh":"box"}}"#,
            "\n",
        );
        let result =
            build_pipeline_from_str(world, None, None, Platform::Metal).expect("build pipeline");

        // The Prop def's identity is the interned id, not a name string.
        let prop = result
            .defs
            .iter()
            .find(|d| d.name == Some(crate::ecs::asset_id::AssetId(2)))
            .expect("day_crate def present with interned id 2");

        let baked: crate::components::Prop = postcard::from_bytes(&prop.args_bytes).unwrap();
        // The `mesh` reference resolved to box's handle (0).
        assert_eq!(baked.mesh, Some(crate::ecs::MeshHandle(0)));
        // The `day_` name prefix resolved to Scene `day`'s id (1).
        assert_eq!(baked.scene, Some(crate::ecs::asset_id::AssetId(1)));
    }

    // A world with physics content but no PhysicsConfig receives one at world
    // start rather than in the build, so the blob carries none and the shipped
    // budget is derived from the same defaults either way.
    #[test]
    fn a_physics_world_carries_no_config_into_the_blob() {
        let _guard = SHADER_BUILD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::compile::shader::install_stub_toolchain();
        let world = concat!(
            r#"{"name":"box","type":"ProceduralMesh","args":{"generator":"box","half_extents":[1,1,1]}}"#,
            "\n",
            r#"{"name":"crate_a","type":"Prop","args":{"mesh":"box","collider":{"shape":"cuboid"}}}"#,
            "\n",
            r#"{"name":"crate_body","type":"PropBody","args":{"prop_name":"crate_a"}}"#,
            "\n",
        );
        let result =
            build_pipeline_from_str(world, None, None, Platform::Metal).expect("build pipeline");

        assert!(
            !result.names.iter().any(|n| n == "physics_config"),
            "no config is compiled in: {:?}",
            result.names
        );
        // The reservation is the authored content plus the floor, on the same
        // strict spawn cap the injected config carries.
        let budget = result.physics_budget.expect("a physics budget");
        assert_eq!(budget.spawn_headroom, 0);
        assert_eq!(budget.dynamic, 1, "the crate");
    }

    // The opt-out directive is a stored component now, so it survives the build
    // and reaches the world-start pass that reads it.
    #[test]
    fn engine_defaults_reach_the_blob_as_a_component() {
        let _guard = SHADER_BUILD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::compile::shader::install_stub_toolchain();
        let world = concat!(
            r#"{"name":"gfx","type":"GraphicsConfig","args":{}}"#,
            "\n",
            r#"{"name":"defaults","type":"EngineDefaults","args":{"sky":false}}"#,
            "\n",
        );
        let result =
            build_pipeline_from_str(world, None, None, Platform::Metal).expect("build pipeline");

        let index = result
            .names
            .iter()
            .position(|n| n == "defaults")
            .expect("the directive compiled into the blob");
        let baked: concinnity_core::components::EngineDefaults =
            postcard::from_bytes(&result.defs[index].args_bytes).unwrap();
        assert!(!baked.sky);
        assert!(baked.debug_hud, "the flags it does not name stay on");
    }

    // A resource asset (here a Font) leaves no component def, so the lock
    // records it through `resource_locks` instead: name, kind, handle, args
    // hash, and the blob its payload landed in.
    #[test]
    fn build_pipeline_records_resource_lock_provenance() {
        let _guard = SHADER_BUILD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::compile::shader::install_stub_toolchain();
        let world = concat!(
            r#"{"name":"f","type":"Font","args":{"size_px":20}}"#,
            "\n",
            r#"{"name":"pause","type":"Screen","args":{}}"#,
            "\n",
        );
        let result = build_pipeline_from_str(world, None, None, Platform::Metal).expect("build");

        assert_eq!(result.resource_locks.len(), result.resources.len());
        let font = result
            .resource_locks
            .iter()
            .find(|r| r.name == "f")
            .expect("font provenance recorded");
        assert_eq!(font.kind, "Font");
        assert_eq!(font.handle, 0);
        assert_eq!(font.args_hash.len(), 64);
        // A payload resource records which blob holds its bytes.
        assert!(font.payload_blob.is_some());
        // The resource is not in the component asset list.
        assert!(!result.names.iter().any(|n| n == "f"));
    }

    #[test]
    fn build_from_path_missing_world_file_errors() {
        let tree = crate::paths::StateTree::at(concinnity_testing::TempTree::new().path());
        assert!(build_from_path(&tree, "/no/such/world.jsonl", Platform::Metal).is_err());
    }

    #[test]
    fn build_from_path_reports_a_malformed_world_file() {
        let dir = concinnity_testing::TempTree::new();
        let world = dir.path().join("world.jsonl");
        std::fs::write(&world, "{not json\n").expect("write world");
        let tree = crate::paths::StateTree::at(dir.path());
        let err = build_from_path(&tree, world.to_str().unwrap(), Platform::Metal)
            .expect_err("malformed world");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    // A lock that cannot be written fails the build: shipping blobs without the
    // record of what went into them would leave the output unexplainable.
    #[test]
    fn write_build_outputs_fails_when_the_lock_cannot_be_written() {
        let output = crate::blob::test_output::Output::new();
        // A directory where the lock file belongs makes the write fail.
        std::fs::create_dir_all(crate::blob::LOCK_PATH).expect("occupy the lock path");

        let result = PipelineResult {
            defs: Vec::new(),
            names: Vec::new(),
            resources: Vec::new(),
            scene_groups: Vec::new(),
            mesh_bounds: Vec::new(),
            physics_budget: None,
            mesh_component_names: Vec::new(),
            payloads: vec![vec![1, 2, 3]],
            cache_hits: 0,
            cache_misses: 0,
            texture_sources: Vec::new(),
            mesh_sources: Vec::new(),
            resource_locks: Vec::new(),
        };
        assert!(
            write_build_outputs(output.tree(), &result, &[], &[]).is_err(),
            "an unwritable lock must fail the build"
        );
    }

    // The full build tail: compile the world, ship the blobs under the state
    // root, and record every asset in the lock beside them.
    #[test]
    fn build_from_path_writes_the_blobs_and_the_lock_beside_them() {
        let _shaders = SHADER_BUILD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::compile::shader::install_stub_toolchain();
        let output = crate::blob::test_output::Output::new();

        let dir = concinnity_testing::TempTree::new();
        let world_path = dir.path().join("world.jsonl");
        std::fs::write(
            &world_path,
            concat!(
                r#"{"name":"gfx","type":"GraphicsConfig","args":{}}"#,
                "\n",
                r#"{"name":"f","type":"Font","args":{"size_px":20}}"#,
                "\n",
                r#"{"name":"pause","type":"Screen","args":{}}"#,
                "\n",
            ),
        )
        .expect("write world");

        build_from_path(output.tree(), world_path.to_str().unwrap(), Platform::Metal)
            .expect("build");

        let raw = std::fs::read_to_string(crate::blob::LOCK_PATH).expect("lock written");
        let lock: crate::blob::BlobLock = serde_json::from_str(&raw).expect("lock is valid json");
        assert_eq!(lock.blobs.len(), 1);

        let (meta, _) = crate::blob::read_cnb(&lock.blobs[0].path).expect("blob 0 parses");
        assert_eq!(
            meta.defs.len(),
            lock.assets.len(),
            "the lock names every def the blob ships"
        );
        assert_eq!(meta.resources.len(), lock.resources.len());
        assert!(lock.assets.iter().any(|a| a.name == "pause"));

        let font = lock
            .resources
            .iter()
            .find(|r| r.name == "f")
            .expect("the font is recorded in the resource stream");
        assert_eq!(font.kind, "Font");
        assert_eq!(font.payload_blob, Some(0));
        assert!(
            !lock.injected.is_empty(),
            "engine defaults are recorded so they can be overridden"
        );
    }

    #[test]
    fn build_pipeline_from_str_rejects_malformed_jsonl() {
        let Err(err) = build_pipeline_from_str("{not json\n", None, None, Platform::Metal) else {
            panic!("malformed line must not build");
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn build_pipeline_from_str_reports_unknown_asset_types() {
        let world = r#"{"name":"mystery","type":"NotAType","args":{}}"#;
        let Err(err) = build_pipeline_from_str(world, None, None, Platform::Metal) else {
            panic!("unknown type must not build");
        };
        assert!(
            err.to_string().contains("NotAType"),
            "error should name the unknown type: {err}"
        );
    }

    // `build_compiled` runs on an already-prepared world, so a type the
    // component registry cannot resolve surfaces here rather than upstream.
    #[test]
    fn build_compiled_names_the_asset_whose_type_will_not_resolve() {
        let assets = vec![wja("mystery", "NotAType", serde_json::json!({}))];
        let Err(err) = build_compiled(assets, None, None, Platform::Metal) else {
            panic!("unknown type must not compile");
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("Asset 'mystery'"), "got: {err}");
    }

    // A payload that will not compile fails the whole build; the error names
    // the asset so the author knows which line to fix.
    #[test]
    fn build_compiled_surfaces_a_payload_compile_failure() {
        let assets = vec![wja(
            "shape",
            "ProceduralMesh",
            serde_json::json!({"generator": "not_a_generator"}),
        )];
        let Err(err) = build_compiled(assets, None, None, Platform::Metal) else {
            panic!("an uncompilable payload must not build");
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("not_a_generator"), "got: {err}");
    }

    // An uncompressed 24-bit BGR Targa, the cheapest real image source to
    // author inline.
    fn tga_2x2() -> Vec<u8> {
        let mut v = vec![0u8; 18];
        v[2] = 2; // uncompressed true-color
        v[12..14].copy_from_slice(&2u16.to_le_bytes());
        v[14..16].copy_from_slice(&2u16.to_le_bytes());
        v[16] = 24;
        v[17] = 0x20; // top origin
        v.extend_from_slice(&[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]);
        v
    }

    // `cn debug`'s hot-reload watcher maps a saved file back to the handle it
    // feeds, so every file-backed texture and mesh records its source in handle
    // order. A generated asset has nothing to watch and records an empty source.
    #[test]
    fn build_compiled_records_hot_reload_sources_in_handle_order() {
        let dir = tempfile::tempdir().expect("tempdir");
        let tga = write_fixture(&dir, "wall.tga", &tga_2x2());
        let glb = write_fixture(
            &dir,
            "scene.glb",
            &crate::import::glb::test_fixtures::static_triangle_glb(),
        );
        let assets = vec![
            wja(
                "proc_tex",
                "Texture",
                serde_json::json!({"generator": "checker", "resolution": 8}),
            ),
            wja(
                "wall_tex",
                "Texture",
                serde_json::json!({"source": tga, "image_index": 3}),
            ),
            wja(
                "inline_mesh",
                MESH_TYPE,
                serde_json::json!({"generator": "box", "half_extents": [1, 1, 1]}),
            ),
            wja(
                "file_mesh",
                MESH_TYPE,
                serde_json::json!({
                    "source": glb,
                    "primitive_index": 0,
                    "lod_levels": 3,
                    "lod_distances": [10.0, 20.0],
                }),
            ),
        ];
        let result = build_compiled(assets, None, None, Platform::Metal).expect("build");

        assert_eq!(result.texture_sources.len(), 2);
        assert_eq!(
            result.texture_sources[0],
            TextureSourceInfo {
                name_id: 0,
                source: String::new(),
                image_index: 0,
            },
            "a generated texture has no file to watch"
        );
        assert_eq!(
            result.texture_sources[1],
            TextureSourceInfo {
                name_id: 1,
                source: tga.clone(),
                image_index: 3,
            }
        );

        assert_eq!(result.mesh_sources.len(), 2);
        assert_eq!(
            result.mesh_sources[0],
            MeshSourceInfo {
                source: String::new(),
                primitive_index: 0,
                lod_levels: 1,
                lod_distances: Vec::new(),
            },
            "a generated mesh has no file to watch"
        );
        assert_eq!(
            result.mesh_sources[1],
            MeshSourceInfo {
                source: glb.clone(),
                primitive_index: 0,
                lod_levels: 3,
                lod_distances: vec![10.0, 20.0],
            }
        );

        // The lock records mirror the catalogues so a blob boot can rebuild
        // them without the authored args.
        let lock_tex: Vec<_> = result
            .resource_locks
            .iter()
            .filter(|r| r.kind == "Texture")
            .collect();
        assert_eq!(lock_tex.len(), 2);
        assert_eq!(lock_tex[0].texture_source.as_ref().unwrap().source, "");
        let wall = lock_tex[1].texture_source.as_ref().unwrap();
        assert_eq!(wall.source, tga);
        assert_eq!(wall.image_index, 3);
        assert!(lock_tex[1].mesh_source.is_none());

        let lock_mesh: Vec<_> = result
            .resource_locks
            .iter()
            .filter(|r| r.kind == "Mesh")
            .collect();
        assert_eq!(lock_mesh.len(), 2);
        assert!(lock_mesh[0].texture_source.is_none());
        let file_mesh = lock_mesh[1].mesh_source.as_ref().unwrap();
        assert_eq!(file_mesh.source, glb);
        assert_eq!(file_mesh.lod_levels, 3);
        assert_eq!(file_mesh.lod_distances, vec![10.0, 20.0]);
    }

    // A data resource carries its bytes inline in the record rather than in a
    // blob payload section, so the lock records no payload blob for it.
    #[test]
    fn build_compiled_keeps_a_data_resource_out_of_the_payload_sections() {
        let assets = vec![
            wja("wood", "Material", serde_json::json!({})),
            wja(
                "shape",
                "ProceduralMesh",
                serde_json::json!({"generator": "box"}),
            ),
        ];
        let result = build_compiled(assets, None, None, Platform::Metal).expect("build");

        assert_eq!(result.resources.len(), 1);
        let material = &result.resources[0];
        assert!(material.payload.is_none(), "a Material rides inline");
        assert!(!material.data_bytes.is_empty());
        postcard::from_bytes::<crate::components::Material>(&material.data_bytes)
            .expect("the inline bytes decode as a Material");
        assert_eq!(result.resource_locks[0].name, "wood");
        assert_eq!(result.resource_locks[0].payload_blob, None);
        // The component's payload is what actually occupies the blob.
        assert_eq!(result.defs.len(), 1);
        assert!(result.defs[0].payload.is_some());
    }

    // Only a File whose kind maps to a mesh payload is compiled; every other
    // kind stays a plain reference with no blob bytes.
    #[test]
    fn build_compiled_compiles_only_mesh_kind_file_assets() {
        let dir = tempfile::tempdir().expect("tempdir");
        let obj = write_fixture(&dir, "tri.obj", b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
        let png = write_fixture(&dir, "icon.png", b"not read");
        let assets = vec![
            wja(
                "model",
                "File",
                serde_json::json!({"path": obj, "kind": "obj"}),
            ),
            wja(
                "icon",
                "File",
                serde_json::json!({"path": png, "kind": "png"}),
            ),
        ];
        let result = build_compiled(assets, None, None, Platform::Metal).expect("build");

        assert_eq!(result.names, vec!["model".to_string(), "icon".to_string()]);
        let mesh_payload = result.defs[0]
            .payload
            .as_ref()
            .expect("the obj File compiles");
        assert!(mesh_payload.len > 0);
        assert!(
            result.defs[1].payload.is_none(),
            "a png File produces no blob payload"
        );
    }
}
