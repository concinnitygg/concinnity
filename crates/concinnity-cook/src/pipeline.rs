// Compile stage of the build pipeline. The world is loaded, expanded, and
// validated upstream by crate::world::prepare_world; this module takes the
// resulting WorldJsonlAsset list and:
// - Resolves each asset to a BlobAssetDef via asset_api::create_asset_def()
// - Compiles payloads for assets that need compilation
// - Packs all payloads into blobs using PayloadPacker (fills locators)
// - Sorts: components first, then systems in declared order

use crate::assets::FileKind;
use crate::world::{WorldConfig, WorldJsonlAsset, normalize_single_shader_type};

use crate::asset_api::{self, AssetRequest};
use crate::blob::PayloadPacker;
use crate::ecs::asset_id;
use crate::ecs::{AssetKind, BlobAssetDef, ResourceRecord};
use crate::registry::ComponentType;
use crate::resource_handles::ResourceAssetCompile;

// The mesh kinds' declarable type names. Both are resource assets (no
// `Component` impl, so no `::NAME` const); the desugar passes and the cache
// probe match on these.
const MESH_TYPE: &str = "Mesh";
const SKINNED_MESH_TYPE: &str = "SkinnedMesh";

pub fn build_from_path(json_path: &str) -> std::io::Result<()> {
    let content = std::fs::read_to_string(json_path)?;
    let loaded = crate::world::prepare_world(&content)
        .map_err(|errs| crate::check::report_validation_errors(&errs))?;

    let result = build_compiled(loaded.assets, None)?;

    let pack_result = write_build_outputs(&result, &loaded.injected)?;
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

// Write the blobs and world-lock.json for a compiled world: the shared build
// tail used by the CLI and the FFI host. The lock records each asset under its
// real name plus every injected default with its full args.
pub fn write_build_outputs(
    result: &PipelineResult,
    injected: &[crate::world::InjectedAsset],
) -> std::io::Result<crate::blob::PackResult> {
    let pack_result = crate::blob::write_blobs(&result.defs, &result.resources, &result.payloads)?;
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
        &pack_result.blob_paths,
    )?;
    Ok(pack_result)
}

// Collapse a list of validation errors into a single io::Error. The messages
// are newline-joined so an upstream caller (e.g. the infra agentic loop) sees
// every problem from one call.
fn errors_to_io(errors: Vec<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, errors.join("\n"))
}

// A texture's identity + on-disk source, in `TextureHandle` order. Now that
// Texture is a resource (no `source`/`asset_id` on a component the renderer
// drains), this is how a dev build hands the `cn debug` tools what they need: the
// hot-reload watcher maps `source` -> handle, and the runtime spawn-by-name path
// maps `name_id` -> handle. `source` is empty for a procedural texture (nothing
// to watch). `name_id` is the interned asset name (same interner the runtime
// shares in-process under `cn debug`), so nothing is interned at runtime.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TextureSourceInfo {
    pub name_id: u32,
    pub source: String,
    pub image_index: u32,
}

// A file-backed Mesh's re-import inputs, in `MeshHandle` order (the Mesh block
// leads the shared mesh-source handle space, so Mesh handles are dense from 0).
// Now that Mesh is a resource (no `source` on a component the renderer drains),
// this is how a dev build hands the `cn debug` hot-reload watcher what it needs
// to re-import a saved `.glb`/`.fbx`. `source` is empty for an inline-authored
// mesh (nothing to watch).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeshSourceInfo {
    pub source: String,
    pub primitive_index: u32,
    pub lod_levels: u32,
    pub lod_distances: Vec<f32>,
}

// The in-memory result of a complete build pipeline run.
// Defs have payload locators filled in; payloads[i] is the raw bytes for
// blob i. This can be used directly without touching disk.
pub struct PipelineResult {
    pub defs: Vec<BlobAssetDef>,
    // Asset name of each def, index-aligned with `defs` (defs only carry the
    // interned id; the lock file records the readable name).
    pub names: Vec<String>,
    // The blob's resource stream: compiled resources addressed by their dense
    // per-kind handle, carried alongside the component defs. Empty until a
    // resource kind migrates off the component registry (AudioClip first).
    pub resources: Vec<ResourceRecord>,
    pub payloads: Vec<Vec<u8>>,
    // Compiled-asset payloads served from the build cache this run.
    pub cache_hits: usize,
    // Compiled-asset payloads compiled fresh this run.
    pub cache_misses: usize,
    // File-backed texture sources in `TextureHandle` order, for the `cn debug`
    // hot-reload watcher. Dev-only info; not written to the shipped blob.
    pub texture_sources: Vec<TextureSourceInfo>,
    // File-backed mesh sources in `MeshHandle` order (dense over the Mesh block
    // of the shared mesh-source space), for the `cn debug` hot-reload watcher.
    // Dev-only info; not written to the shipped blob.
    pub mesh_sources: Vec<MeshSourceInfo>,
    // Lock-file provenance for the resource stream, index-aligned with
    // `resources` (records only carry the kind tag + handle; the lock records
    // the readable name and args hash).
    pub resource_locks: Vec<crate::blob::LockedResource>,
}

// Validate a single asset's type and generator without running the full build
// pipeline. Called by the server on each world_add so the LLM gets per-asset
// feedback without waiting for a WebSocket round-trip.
//
// Checks:
//   - asset type is registered (via asset_api::create_asset_def)
//   - per-type structural checks via crate::check
// Shader assets are not compiled here; use the validate_shader tool for that.
#[allow(dead_code)]
pub fn validate_asset(
    asset_type: &str,
    name: &str,
    args: &serde_json::Value,
) -> Result<(), String> {
    // Single-asset validation has no surrounding world to intern against; the
    // resulting ids are throwaway. Reset so calls do not accumulate entries.
    // Clear the resource handle map too: with no world there are no handles, so
    // a resource reference falls back to the interner (parses without resolving
    // to a real slot, which single-asset validation never needs).
    asset_id::reset_interner();
    crate::resource_handles::reset_resource_handles();
    let (asset_type, args) = normalize_single_shader_type(asset_type, args);
    let asset_type = asset_type.as_str();
    let type_norm = asset_type.to_lowercase().replace('_', "");

    // Build-time types are valid in world.jsonl; they are consumed by expansion
    // functions before the runtime asset registry sees them.
    if matches!(
        type_norm.as_str(),
        "environment" | "lightrig" | "materialpalette" | "camerashot" | "prefab" | "sceneimport"
    ) {
        return Ok(());
    }

    // Resource-only asset types (AudioClip) have left the component registry, so
    // they never build a component def; validate them as known types with a
    // structural check instead of routing through `create_asset_def`.
    if crate::resource_handles::ResourceAssetType::parse(asset_type).is_some() {
        crate::check::check_asset(&type_norm, name, &args)?;
        return Ok(());
    }

    let req = AssetRequest {
        asset_type: asset_type.to_string(),
        args: Some(args.clone()),
    };
    asset_api::create_asset_def(&req).map_err(|e| format!("Asset '{}': {}", name, e))?;

    crate::check::check_asset(&type_norm, name, &args)?;

    Ok(())
}

// Run the full build pipeline on an in-memory JSONL string without touching
// disk. Loads, expands, and validates the world (crate::world::prepare_world),
// then compiles it. `artifacts_dir` is an optional directory consulted when
// resolving bare shader filenames not found under assets/; pass the account's
// artifact directory so test_world can compile user-written shaders.
pub fn build_pipeline_from_str(
    content: &str,
    artifacts_dir: Option<&str>,
) -> std::io::Result<PipelineResult> {
    let loaded = crate::world::prepare_world(content).map_err(errors_to_io)?;
    build_compiled(loaded.assets, artifacts_dir)
}

// Compile an already-prepared world (expanded + structurally and semantically
// validated) into in-memory blobs. This is the compile-only stage; it assumes
// the assets have passed crate::world::prepare_world.
pub fn build_compiled(
    mut assets: Vec<WorldJsonlAsset>,
    artifacts_dir: Option<&str>,
) -> std::io::Result<PipelineResult> {
    let config = WorldConfig::default();

    // Cache probe runs before desugar. For every glTF-sourced Mesh /
    // SkinnedMesh, hash the un-desugared args + referenced .glb and look up
    // the compiled payload by that key. On a hit, we hold the bytes and skip
    // the .glb parse entirely (the original goal: an unchanged source file
    // means no work). On a miss, the recorded key is used when the compile
    // step stores the freshly produced payload, so the next build's probe
    // can re-use it.
    let gltf_cache = probe_gltf_cache(&assets, artifacts_dir);

    // Expand any glTF-sourced SkinnedMesh and Mesh assets into inline geometry
    // before anything else looks at their args. Animations expand after the
    // skinned-mesh pass so an importer that wanted to share state could read
    // already-imported skeletons; today both passes parse the .glb fresh,
    // but the ordering keeps that option open without an API churn.
    desugar_gltf_skinned_meshes(&mut assets, &gltf_cache)?;
    desugar_gltf_meshes(&mut assets, &gltf_cache)?;
    desugar_fbx_meshes(&mut assets, &gltf_cache)?;
    desugar_gltf_animations(&mut assets)?;
    desugar_root_motion(&mut assets)?;

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
    // Install a clone; the original is kept to look up each resource asset's
    // handle while partitioning below.
    crate::resource_handles::install_resource_handles(resource_handles.clone());

    // Partition the world into component assets (each becomes a `BlobAssetDef`)
    // and resource assets (each becomes a resource-stream record). A resource
    // asset (AudioClip) has left the component registry, so it never goes through
    // `create_asset_def`; it is compiled + packed as a resource below. `named` is
    // therefore no longer 1:1 with `assets`, so `named_src[i]` records the source
    // asset index of each component def.
    use crate::resource_handles::ResourceAssetType;
    let mut named: Vec<(String, BlobAssetDef)> = Vec::new();
    let mut named_src: Vec<usize> = Vec::new();
    let mut resource_jobs: Vec<(usize, ResourceAssetType, u32)> = Vec::new();
    for (i, asset) in assets.iter().enumerate() {
        if let Some(rt) = ResourceAssetType::parse(&asset.asset_type) {
            let id = asset_id::intern(&asset.name);
            let handle = resource_handles
                .get(rt.resource_kind(), id)
                .expect("resource asset was assigned a handle above");
            resource_jobs.push((i, rt, handle));
            continue;
        }
        let req = AssetRequest {
            asset_type: asset.asset_type.clone(),
            args: Some(asset.args.clone()),
        };
        let mut def = asset_api::create_asset_def(&req).map_err(|e| {
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
        .filter(|(_, rt, _)| *rt == ResourceAssetType::Texture)
        .map(|(_, _, h)| *h as usize + 1)
        .max()
        .unwrap_or(0);
    let mut texture_sources = vec![TextureSourceInfo::default(); texture_count];
    for (asset_idx, rt, handle) in &resource_jobs {
        if *rt != ResourceAssetType::Texture {
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
        .filter(|(_, rt, _)| *rt == ResourceAssetType::Mesh)
        .map(|(_, _, h)| *h as usize + 1)
        .max()
        .unwrap_or(0);
    let mut mesh_sources = vec![MeshSourceInfo::default(); mesh_count];
    for (asset_idx, rt, handle) in &resource_jobs {
        if *rt != ResourceAssetType::Mesh {
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

    let compiled = compile_and_pack_payloads(
        &mut named,
        &named_src,
        &assets,
        &resource_jobs,
        config.max_blob_bytes,
        artifacts_dir,
        &gltf_cache,
    )?;

    // Lock-file provenance for the resource stream: `compiled.resources` is
    // emitted in `resource_jobs` order, so the two zip index-aligned.
    let resource_locks: Vec<crate::blob::LockedResource> = resource_jobs
        .iter()
        .zip(compiled.resources.iter())
        .map(|((asset_idx, rt, handle), record)| {
            let asset = &assets[*asset_idx];
            crate::blob::LockedResource {
                name: asset.name.clone(),
                kind: rt.as_str().to_string(),
                handle: *handle,
                args_hash: crate::blob::checksum(asset.args.to_string().as_bytes()),
                payload_blob: record.payload.as_ref().map(|p| p.blob_index),
            }
        })
        .collect();

    // The blob carries components (emitted in declaration order) plus the
    // resource stream. (System run order is no longer a build concern: every
    // system is internal client code ordered by the client's
    // `World::build_internal_systems` schedule.)
    let (names, defs): (Vec<String>, Vec<BlobAssetDef>) = named.into_iter().unzip();

    Ok(PipelineResult {
        defs,
        names,
        resources: compiled.resources,
        payloads: compiled.blobs,
        cache_hits: compiled.cache_hits,
        cache_misses: compiled.cache_misses,
        texture_sources,
        mesh_sources,
        resource_locks,
    })
}

// Per-asset state recorded by `probe_gltf_cache`. `key` is the cache key
// computed from the asset's pre-desugar args; `bytes` is `Some` when the
// cache already held a compiled payload for that key. On a hit, the desugar
// pass skips the .glb parse for this asset; on a miss, compile_and_pack
// stores the freshly compiled payload under the same `key` so the next
// build's probe can re-use it.
#[derive(Clone)]
struct GltfCacheEntry {
    key: String,
    bytes: Option<Vec<u8>>,
}

// Hash every glTF-sourced Mesh / SkinnedMesh asset's pre-desugar args and
// referenced .glb, then probe the content-addressed payload cache. Returns
// one entry per (source-backed) asset name. Assets without a `source` are
// not probed: their args don't depend on a file, so the regular per-asset
// cache path inside compile_and_pack_payloads is sufficient.
fn probe_gltf_cache(
    assets: &[WorldJsonlAsset],
    artifacts_dir: Option<&str>,
) -> std::collections::HashMap<String, GltfCacheEntry> {
    use crate::resource_handles::{ResourceAssetCompile, ResourceAssetType};

    let mut out = std::collections::HashMap::new();
    let empty: [WorldJsonlAsset; 0] = [];
    for asset in assets {
        let has_source = asset
            .args
            .get("source")
            .and_then(|v| v.as_str())
            .map(|s| !s.is_empty())
            .unwrap_or(false);
        if !has_source {
            continue;
        }

        // Both mesh kinds are resource assets: their caches key on the
        // resource discriminant and resource source list.
        let rt = if asset.asset_type == MESH_TYPE {
            ResourceAssetType::Mesh
        } else if asset.asset_type == SKINNED_MESH_TYPE {
            ResourceAssetType::SkinnedMesh
        } else {
            continue;
        };
        let ctx = crate::asset::BuildCtx {
            name: asset.name.as_str(),
            artifacts_dir,
            all_assets: &empty,
        };
        let discriminant = RESOURCE_CACHE_DISC_BASE + rt.resource_kind() as u8;
        let extra_sources = rt.source_files(&asset.args);
        let key = crate::cache::payload_key(discriminant, &asset.args, &ctx, &extra_sources);
        let bytes = crate::cache::load(&key);
        out.insert(asset.name.clone(), GltfCacheEntry { key, bytes });
    }
    out
}

// Expand glTF-sourced SkinnedMesh assets in place: parse the referenced .glb
// and write the imported geometry + skeleton into the asset's inline
// `vertices` / `indices` / `skeleton` args. A SkinnedMesh with no `source` is
// left untouched, so an inline-authored mesh is byte-for-byte unchanged.
// Skips an asset whose cache probe found a precompiled payload: there is no
// reason to parse the .glb when the bytes are already in hand.
fn desugar_gltf_skinned_meshes(
    assets: &mut [WorldJsonlAsset],
    gltf_cache: &std::collections::HashMap<String, GltfCacheEntry>,
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
        if source.is_empty() {
            continue;
        }
        // Cache probe found a compiled payload for this asset, no need
        // to parse the .glb. compile_and_pack_payloads will use the bytes
        // directly. Leave the args un-desugared so they keep matching the
        // pre-desugar cache key on the next build.
        if matches!(
            gltf_cache.get(&asset.name),
            Some(GltfCacheEntry { bytes: Some(_), .. })
        ) {
            continue;
        }

        let imported = crate::gltf::import_skinned_glb(&source).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset '{}': glTF import failed: {}", asset.name, e),
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
            "Asset '{}': imported glTF '{}': {} vertices, {} indices, {} joints",
            asset.name,
            source,
            imported.vertices.len(),
            imported.indices.len(),
            imported.skeleton.len()
        );
    }
    Ok(())
}

// Expand glTF-sourced static `Mesh` assets in place: parse the referenced
// `.glb` and write the imported primitive geometry into the asset's inline
// `vertices` / `indices` args. A Mesh with no `source` is left untouched. The
// GLB is parsed once per unique path; ABeautifulGame fans 35+ Mesh assets out
// of one file, so memoization keeps this O(files) rather than O(primitives).
fn desugar_gltf_meshes(
    assets: &mut [WorldJsonlAsset],
    gltf_cache: &std::collections::HashMap<String, GltfCacheEntry>,
) -> std::io::Result<()> {
    use crate::assets::VertexData;
    use std::collections::HashMap;

    // One split chunk: its vertices and index buffer.
    type Chunk = (Vec<VertexData>, Vec<u16>);

    let mut parsed_cache: HashMap<String, gltf::Gltf> = HashMap::new();
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
        // only the glTF container.
        if !source.to_lowercase().ends_with(".glb") {
            continue;
        }
        // Skip the .glb parse when the cache probe already produced bytes
        // for this asset (see `desugar_gltf_skinned_meshes` for the same
        // pattern). Args stay pre-desugar so the next build's probe hits.
        if matches!(
            gltf_cache.get(&asset.name),
            Some(GltfCacheEntry { bytes: Some(_), .. })
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
            let doc = crate::glb::parse_glb(&source).map_err(|e| {
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
                    crate::glb::read_primitive_geometry(doc, &source, primitive_index).map_err(
                        |e| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                format!("Asset '{}': glTF import failed: {}", asset.name, e),
                            )
                        },
                    )?;
                let chunks = crate::glb::split_into_u16_chunks(&verts, &indices32);
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
            crate::glb::import_static_glb_primitive_from_doc(doc, &source, primitive_index)
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

// Expand FBX-sourced Mesh assets in place: parse the `.fbx` into an FbxScene
// and write the imported geometry into each asset's inline `vertices` /
// `indices` args, keyed by `primitive_index` and optional `chunk_index`. A Mesh
// whose source is not a `.fbx` is left to `desugar_gltf_meshes`. The FBX is
// parsed once per unique path (Bistro fans thousands of Mesh assets out of one
// file) and each primitive's u16 chunk split is memoized.
fn desugar_fbx_meshes(
    assets: &mut [WorldJsonlAsset],
    gltf_cache: &std::collections::HashMap<String, GltfCacheEntry>,
) -> std::io::Result<()> {
    use crate::assets::VertexData;
    use crate::fbx::FbxScene;
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
            gltf_cache.get(&asset.name),
            Some(GltfCacheEntry { bytes: Some(_), .. })
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
            let scene = crate::fbx::parse_fbx(&source).map_err(|e| {
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
            let (verts, indices32) = crate::fbx::read_primitive_geometry(scene, primitive_index)
                .map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Asset '{}': FBX import failed: {}", asset.name, e),
                    )
                })?;
            let chunks = crate::glb::split_into_u16_chunks(&verts, &indices32);
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

// Expand glTF-sourced `Animation` assets in place: parse the `.glb`, pick the
// animation by `animation_name` (preferred) or `animation_index`, and replace
// the asset's `duration` + `tracks` with the imported data. An Animation with
// no `source` is left untouched, so inline-authored clips are byte-for-byte
// unchanged. Channels targeting non-joint nodes are dropped silently by the
// importer.
fn desugar_gltf_animations(assets: &mut [WorldJsonlAsset]) -> std::io::Result<()> {
    use crate::assets::Animation;
    use crate::ecs::Component;

    for asset in assets.iter_mut() {
        if asset.asset_type != Animation::NAME {
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
        let animation_name = asset
            .args
            .get("animation_name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let animation_index = asset
            .args
            .get("animation_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        // Look up by name when authored; fall back to the numeric index.
        let resolved_index = if !animation_name.is_empty() {
            let names = crate::gltf::glb_animation_names(&source).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Asset '{}': glTF import failed: {}", asset.name, e),
                )
            })?;
            names
                .iter()
                .position(|n| n == &animation_name)
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "Asset '{}': glTF '{}' has no animation named '{}' \
                             (file contains: {:?})",
                            asset.name, source, animation_name, names
                        ),
                    )
                })?
        } else {
            animation_index
        };

        let imported = crate::gltf::import_glb_animation(&source, resolved_index).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset '{}': glTF import failed: {}", asset.name, e),
            )
        })?;

        // Convert ImportedAnimation -> the asset's serialised track shape.
        let tracks_json: Vec<serde_json::Value> = imported
            .tracks
            .iter()
            .map(|track| {
                let keyframes: Vec<serde_json::Value> = track
                    .keys
                    .iter()
                    .map(|k| {
                        serde_json::json!({
                            "time": k.time,
                            "translation": k.pose.translation,
                            "rotation_deg": k.pose.rotation_deg,
                            "scale": k.pose.scale,
                        })
                    })
                    .collect();
                serde_json::json!({
                    "joint": track.joint,
                    "keyframes": keyframes,
                })
            })
            .collect();

        let name = asset.name.clone();
        let obj = asset.args.as_object_mut().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset '{}': args is not a JSON object", name),
            )
        })?;
        obj.insert("duration".to_string(), serde_json::json!(imported.duration));
        obj.insert("tracks".to_string(), serde_json::Value::Array(tracks_json));
        tracing::info!(
            "Asset '{}': imported glTF '{}' animation {} ('{}'): {:.3} s, {} track(s)",
            asset.name,
            source,
            resolved_index,
            imported.name,
            imported.duration,
            imported.tracks.len(),
        );
    }
    Ok(())
}

// Bake root motion on every Animation that opted in: strip the root joint's
// travel out of the pose tracks into the asset's `root_track` (see
// `root_motion::bake_root_motion`). Runs after the glTF pass so imported
// tracks are already inline; an Animation without `root_motion` is
// untouched. A root-motion clip whose root joint has no track produces an
// empty curve, which would silently never move a character, so it warns.
fn desugar_root_motion(assets: &mut [WorldJsonlAsset]) -> std::io::Result<()> {
    use crate::assets::Animation;
    use crate::ecs::Component;

    // This deserializes each flagged clip (whose `target` is a name reference),
    // so the name resolver must be installed. The full pipeline resets the
    // interner before reaching here; installing it again is a cheap no-op and
    // keeps this pass correct when called on its own.
    crate::ecs::asset_id::ensure_name_resolver();

    for asset in assets.iter_mut() {
        if asset.asset_type != Animation::NAME
            || asset.args.get("root_motion").and_then(|v| v.as_bool()) != Some(true)
        {
            continue;
        }
        let mut anim: Animation = serde_json::from_value(asset.args.clone()).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Asset '{}': root-motion bake failed to parse args: {}",
                    asset.name, e
                ),
            )
        })?;
        crate::root_motion::bake_root_motion(&mut anim);
        if anim.root_track.is_empty() {
            tracing::warn!(
                "Asset '{}': root_motion is set but the clip has no track on the root \
                 joint; the character will not move",
                asset.name
            );
        }
        let name = asset.name.clone();
        let obj = asset.args.as_object_mut().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset '{}': args is not a JSON object", name),
            )
        })?;
        obj.insert(
            "tracks".to_string(),
            serde_json::to_value(&anim.tracks).expect("serialize animation tracks"),
        );
        obj.insert(
            "root_track".to_string(),
            serde_json::to_value(&anim.root_track).expect("serialize root track"),
        );
        tracing::info!(
            "Asset '{}': baked root motion ({} key(s){})",
            asset.name,
            anim.root_track.len(),
            if anim.root_motion_y { ", incl. Y" } else { "" },
        );
    }
    Ok(())
}

// Validate world JSONL without running compilation. Runs the full front half
// of the pipeline (load, expand, semantic checks) plus a per-asset type/args
// resolution, but stops short of compiling payloads: intended for fast
// server-side pre-deploy checks where shader compilation is not needed.
// Every problem found is reported in a single newline-joined error.
#[allow(dead_code)]
pub fn validate_world_jsonl(content: &str) -> std::io::Result<()> {
    let loaded = crate::world::prepare_world(content).map_err(errors_to_io)?;

    let mut errors: Vec<String> = Vec::new();
    for asset in &loaded.assets {
        // Resource-only assets (AudioClip) do not build a component def; they are
        // a known type on their own registry, so skip the component resolution.
        if crate::resource_handles::ResourceAssetType::parse(&asset.asset_type).is_some() {
            continue;
        }
        let req = AssetRequest {
            asset_type: asset.asset_type.clone(),
            args: Some(asset.args.clone()),
        };
        if let Err(e) = asset_api::create_asset_def(&req) {
            errors.push(format!("Asset '{}': {}", asset.name, e));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors_to_io(errors))
    }
}

// Cache-key discriminant base for resource payloads. Resource kinds are keyed
// as `128 + ResourceKind as u8` so their cache keys never collide with a
// component discriminant (all < 128, the `ComponentMask` ceiling).
const RESOURCE_CACHE_DISC_BASE: u8 = 128;

// One compiled resource awaiting packing: its kind tag + handle, the compiled
// bytes, whether those bytes are the record's inline data (a data resource like
// Material) or a blob payload, and the baked runtime data a hybrid kind
// (SkinnedMesh) carries alongside its payload (empty for everything else; it
// bakes from the authored args, so it sits outside the payload cache).
struct PendingResource {
    kind: u8,
    handle: u32,
    bytes: Vec<u8>,
    is_data: bool,
    extra_data: Vec<u8>,
}

// The output of the compile + pack pass: the packed blob payload sections, the
// resource-stream records (each with its payload locator), and cache accounting.
struct CompiledOutput {
    blobs: Vec<Vec<u8>>,
    resources: Vec<ResourceRecord>,
    cache_hits: usize,
    cache_misses: usize,
}

fn compile_and_pack_payloads(
    named: &mut [(String, BlobAssetDef)],
    named_src: &[usize],
    assets: &[WorldJsonlAsset],
    resource_jobs: &[(usize, crate::resource_handles::ResourceAssetType, u32)],
    max_blob_bytes: u64,
    artifacts_dir: Option<&str>,
    gltf_cache: &std::collections::HashMap<String, GltfCacheEntry>,
) -> std::io::Result<CompiledOutput> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let compiled_indices: Vec<usize> = named
        .iter()
        .enumerate()
        .filter(|(i, (_, def))| {
            if def.kind != AssetKind::Component {
                return false;
            }
            let Some(ct) = ComponentType::from_discriminant(def.discriminant) else {
                return false;
            };
            if ct.as_str() == "File" {
                // only compile File assets whose kind maps to a supported payload
                // `named[i]` maps to `assets[named_src[i]]`.
                return assets[named_src[*i]]
                    .args
                    .get("kind")
                    .and_then(|k| k.as_str())
                    .and_then(FileKind::from_ext)
                    .map(|fk| fk.is_mesh())
                    .unwrap_or(false);
            }
            ct.registration().needs_compilation()
        })
        .map(|(i, _)| i)
        .collect();

    // Snapshot each job's inputs so the parallel compile borrows nothing from
    // `named`, which is mutated afterwards to record payload locators.
    let jobs: Vec<(usize, String, u8)> = compiled_indices
        .iter()
        .map(|&idx| {
            let (name, def) = &named[idx];
            (idx, name.clone(), def.discriminant)
        })
        .collect();

    // Compile assets in parallel. Each job is independent (it reads only its
    // own args and produces its own payload bytes) and the payload cache is
    // content-addressed, so concurrent hits and stores never collide. The
    // collected order follows `jobs`, so packing below stays deterministic.
    let cache_hits = AtomicUsize::new(0);
    let pending: Vec<(usize, Vec<u8>)> = jobs
        .par_iter()
        .map(
            |(idx, name, discriminant)| -> std::io::Result<(usize, Vec<u8>)> {
                let ct = ComponentType::from_discriminant(*discriminant).ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Invalid ComponentType discriminant for asset '{}'", name),
                    )
                })?;

                // The job carries the `named` index; map it to its source asset
                // via `named_src` (`named` is not 1:1 with `assets` once resource
                // assets are partitioned out).
                let asset_args = &assets[named_src[*idx]].args;

                let ctx = crate::asset::BuildCtx {
                    name: name.as_str(),
                    artifacts_dir,
                    all_assets: assets,
                };

                // GLB-sourced Mesh / SkinnedMesh assets are probed before
                // desugar; honor those results here so the .glb parse really
                // is skipped on cache hits. On a miss the precomputed key is
                // used at store time, keeping the next build's probe valid.
                if let Some(entry) = gltf_cache.get(name) {
                    if let Some(bytes) = &entry.bytes {
                        cache_hits.fetch_add(1, Ordering::Relaxed);
                        return Ok((*idx, bytes.clone()));
                    }
                    let compiled_bytes = compile_by_type(ct, asset_args, &ctx)?;
                    crate::cache::store(&entry.key, &compiled_bytes);
                    return Ok((*idx, compiled_bytes));
                }

                // Reuse a cached payload when the asset's inputs are unchanged;
                // otherwise compile and populate the cache for the next build.
                let extra_sources = source_files_by_type(ct, asset_args, &ctx);
                let key =
                    crate::cache::payload_key(*discriminant, asset_args, &ctx, &extra_sources);
                if let Some(bytes) = crate::cache::load(&key) {
                    cache_hits.fetch_add(1, Ordering::Relaxed);
                    return Ok((*idx, bytes));
                }
                let compiled_bytes = compile_by_type(ct, asset_args, &ctx)?;
                crate::cache::store(&key, &compiled_bytes);
                Ok((*idx, compiled_bytes))
            },
        )
        .collect::<std::io::Result<Vec<_>>>()?;

    let component_hits = cache_hits.into_inner();

    // Compile the resource-stream payloads (AudioClip today). Few and cheap, so
    // this stays serial; the content-addressed payload cache still short-circuits
    // an unchanged source. Bypasses the `BuildAsset`/`ComponentType` path a
    // component takes -- a resource is no longer a component.
    let mut resource_hits = 0usize;
    let mut resource_pending: Vec<PendingResource> = Vec::new();
    for (asset_idx, rt, handle) in resource_jobs {
        let asset = &assets[*asset_idx];
        let ctx = crate::asset::BuildCtx {
            name: asset.name.as_str(),
            artifacts_dir,
            all_assets: assets,
        };
        let extra_data = rt
            .compile_data(&asset.name, &asset.args)?
            .unwrap_or_default();
        // A glTF/FBX-sourced mesh was probed before desugar; honor that result so
        // the source parse really is skipped on a hit and the pre-desugar key is
        // reused at store time (same contract as the component gltf-cache path).
        if let Some(entry) = gltf_cache.get(&asset.name) {
            let bytes = if let Some(bytes) = &entry.bytes {
                resource_hits += 1;
                bytes.clone()
            } else {
                let compiled = rt.compile_payload(&asset.args)?;
                crate::cache::store(&entry.key, &compiled);
                compiled
            };
            resource_pending.push(PendingResource {
                kind: rt.resource_kind() as u8,
                handle: *handle,
                bytes,
                is_data: rt.is_data(),
                extra_data,
            });
            continue;
        }
        let extra_sources = rt.source_files(&asset.args);
        let key = crate::cache::payload_key(
            RESOURCE_CACHE_DISC_BASE + rt.resource_kind() as u8,
            &asset.args,
            &ctx,
            &extra_sources,
        );
        let bytes = if let Some(bytes) = crate::cache::load(&key) {
            resource_hits += 1;
            bytes
        } else {
            let compiled = rt.compile_payload(&asset.args)?;
            crate::cache::store(&key, &compiled);
            compiled
        };
        resource_pending.push(PendingResource {
            kind: rt.resource_kind() as u8,
            handle: *handle,
            bytes,
            is_data: rt.is_data(),
            extra_data,
        });
    }

    let cache_hits = component_hits + resource_hits;
    let cache_misses = (pending.len() - component_hits) + (resource_pending.len() - resource_hits);

    if pending.is_empty() && resource_pending.is_empty() {
        return Ok(CompiledOutput {
            blobs: vec![Vec::new()],
            resources: Vec::new(),
            cache_hits: 0,
            cache_misses: 0,
        });
    }

    // Pack component payloads first (recording each def's locator), then the
    // resource payloads (building each resource record's locator). One packer, so
    // both streams address the same blob(s).
    let mut packer = PayloadPacker::new(max_blob_bytes);

    for (idx, bytes) in &pending {
        let locator = packer.push(bytes);
        named[*idx].1.payload = Some(locator);
    }

    let mut resources: Vec<ResourceRecord> = Vec::with_capacity(resource_pending.len());
    for pending in &resource_pending {
        // A data resource (Material) carries its bytes inline; a payload
        // resource parks its bytes in a blob section and records the locator,
        // plus any hybrid baked data (SkinnedMesh) inline beside it.
        let (payload, data_bytes) = if pending.is_data {
            (None, pending.bytes.clone())
        } else {
            (
                Some(packer.push(&pending.bytes)),
                pending.extra_data.clone(),
            )
        };
        resources.push(ResourceRecord {
            resource_kind: pending.kind,
            handle: pending.handle,
            payload,
            data_bytes,
        });
    }

    Ok(CompiledOutput {
        blobs: packer.finish(),
        resources,
        cache_hits,
        cache_misses,
    })
}

// Dispatch payload compilation by ComponentType. Every variant listed below
// has a `BuildAsset` impl in its asset file; the body of each call here is a
// one-liner that delegates to the trait. Adding a new compiled component
// means:
//   1. impl `Component` with `PAYLOAD = AssetPayload::Compiled` for the type
//   2. impl `BuildAsset` for the type in its asset file
//   3. Add one match arm here
fn compile_by_type(
    ct: ComponentType,
    args: &serde_json::Value,
    ctx: &crate::asset::BuildCtx<'_>,
) -> std::io::Result<Vec<u8>> {
    use crate::asset::BuildAsset;
    use crate::assets::{File, ProceduralMesh, Room, SdfVolume, ShaderStage, VoxelChunk};
    match ct {
        ComponentType::ProceduralMesh => <ProceduralMesh as BuildAsset>::compile_payload(args, ctx),
        ComponentType::VoxelChunk => <VoxelChunk as BuildAsset>::compile_payload(args, ctx),
        ComponentType::File => <File as BuildAsset>::compile_payload(args, ctx),
        ComponentType::Room => <Room as BuildAsset>::compile_payload(args, ctx),
        ComponentType::ShaderStage => <ShaderStage as BuildAsset>::compile_payload(args, ctx),
        ComponentType::SdfVolume => <SdfVolume as BuildAsset>::compile_payload(args, ctx),
        other => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Asset '{}' is marked Compiled but has no BuildAsset impl (ComponentType {:?})",
                ctx.name, other
            ),
        )),
    }
}

// Dispatch `BuildAsset::source_files` by ComponentType. Mirrors
// `compile_by_type` so the cache layer can fold the contents-hash of each
// asset's referenced source files into its payload key. Types with no
// `BuildAsset` impl, or with the trait default, contribute nothing.
fn source_files_by_type(
    ct: ComponentType,
    args: &serde_json::Value,
    ctx: &crate::asset::BuildCtx<'_>,
) -> Vec<String> {
    use crate::asset::BuildAsset;
    use crate::assets::{File, ProceduralMesh, Room, SdfVolume, ShaderStage, VoxelChunk};
    match ct {
        ComponentType::ProceduralMesh => <ProceduralMesh as BuildAsset>::source_files(args, ctx),
        ComponentType::VoxelChunk => <VoxelChunk as BuildAsset>::source_files(args, ctx),
        ComponentType::File => <File as BuildAsset>::source_files(args, ctx),
        ComponentType::Room => <Room as BuildAsset>::source_files(args, ctx),
        ComponentType::ShaderStage => <ShaderStage as BuildAsset>::source_files(args, ctx),
        ComponentType::SdfVolume => <SdfVolume as BuildAsset>::source_files(args, ctx),
        _ => Vec::new(),
    }
}

// Resolve scene + view associations that the runtime can no longer derive
// from name strings, baking them into the asset args so they survive as
// AssetId ids.
//
// Naming-convention relationships handled:
//   - A Prop named `<scene>_*` belongs to Scene `<scene>`. The matched scene
//     name is written into the prop's `scene` arg.
//   - A Sprite, TextLabel, or HitRegion named `<view>_*` belongs to View
//     `<view>`. The matched view name is written into the asset's `view` arg.
//   - A HitRegion or KeyBinding `action` of the form `scene:<name>`,
//     `view:show:<name>`, or `view:toggle:<name>` has its `<name>` part
//     rewritten to the interned id, so `UiInputSystem` can parse an integer
//     at runtime instead of a name.
fn resolve_scene_refs(assets: &mut [WorldJsonlAsset]) {
    let norm = |s: &str| s.to_lowercase().replace('_', "");

    let scene_names: Vec<String> = assets
        .iter()
        .filter(|a| norm(&a.asset_type) == "scene")
        .map(|a| a.name.clone())
        .collect();

    let view_names: Vec<String> = assets
        .iter()
        .filter(|a| norm(&a.asset_type) == "view")
        .map(|a| a.name.clone())
        .collect();

    // Rewrite an action string, replacing the trailing `<name>` after the
    // given action prefix with its interned id. Returns Some(new_action) when
    // the action used the prefix with an unresolved name; None otherwise.
    let resolve_action = |action: &str| -> Option<String> {
        for prefix in ["scene:", "view:show:", "view:toggle:"] {
            if let Some(rest) = action.strip_prefix(prefix) {
                if !rest.is_empty() && rest.parse::<u32>().is_err() {
                    return Some(format!("{prefix}{}", asset_id::intern(rest).0));
                }
                return None;
            }
        }
        None
    };

    for asset in assets.iter_mut() {
        match norm(&asset.asset_type).as_str() {
            "prop" => {
                if asset.args.get("scene").is_some() {
                    continue;
                }
                // Longest matching prefix wins so a nested name (e.g.
                // `level_boss_*` under both `level` and `level_boss`) binds to
                // the most specific scene. Equivalent to first-match when no
                // scene name prefixes another.
                let matched = scene_names
                    .iter()
                    .filter(|sn| asset.name.starts_with(&format!("{sn}_")))
                    .max_by_key(|sn| sn.len())
                    .cloned();
                if let (Some(sn), serde_json::Value::Object(m)) = (matched, &mut asset.args) {
                    m.insert("scene".to_string(), serde_json::Value::String(sn));
                }
            }
            "sprite" | "imageoverlay" | "textlabel" | "text" | "hitregion" | "scrollpanel" => {
                // Resolve view prefix association. Longest matching prefix wins
                // so a nested view name (e.g. `main_menu_settings_*` under both
                // `main_menu` and `main_menu_settings`) binds to the most
                // specific view. Equivalent to first-match when no view name
                // prefixes another.
                if asset.args.get("view").is_none() {
                    let matched = view_names
                        .iter()
                        .filter(|vn| asset.name.starts_with(&format!("{vn}_")))
                        .max_by_key(|vn| vn.len())
                        .cloned();
                    if let (Some(vn), serde_json::Value::Object(m)) = (matched, &mut asset.args) {
                        m.insert("view".to_string(), serde_json::Value::String(vn));
                    }
                }
                // Resolve view:* / scene:* action targets to interned ids.
                if matches!(norm(&asset.asset_type).as_str(), "hitregion") {
                    let new_action = asset
                        .args
                        .get("action")
                        .and_then(|v| v.as_str())
                        .and_then(resolve_action);
                    if let (Some(action), serde_json::Value::Object(m)) =
                        (new_action, &mut asset.args)
                    {
                        m.insert("action".to_string(), serde_json::Value::String(action));
                    }
                }
            }
            "keybinding" => {
                let new_action = asset
                    .args
                    .get("action")
                    .and_then(|v| v.as_str())
                    .and_then(resolve_action);
                if let (Some(action), serde_json::Value::Object(m)) = (new_action, &mut asset.args)
                {
                    m.insert("action".to_string(), serde_json::Value::String(action));
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Default-shader compilation writes intermediates to a shared
    // .concinnity/data path keyed by asset name, so tests whose worlds pull in
    // the default ShaderStages (any rendering world) must not build
    // concurrently.
    static SHADER_BUILD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn build_pipeline_interns_names_and_resolves_refs() {
        let _guard = SHADER_BUILD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // box=0, day=1, day_crate=2 in declaration order.
        let world = concat!(
            r#"{"name":"box","type":"ProceduralMesh","args":{"generator":"box","half_extents":[1,1,1]}}"#,
            "\n",
            r#"{"name":"day","type":"Scene","args":{}}"#,
            "\n",
            r#"{"name":"day_crate","type":"Prop","args":{"mesh":"box"}}"#,
            "\n",
        );
        let result = build_pipeline_from_str(world, None).expect("build pipeline");

        // The Prop def's identity is the interned id, not a name string.
        let prop = result
            .defs
            .iter()
            .find(|d| d.name == Some(crate::ecs::asset_id::AssetId(2)))
            .expect("day_crate def present with interned id 2");

        let baked: crate::assets::Prop = postcard::from_bytes(&prop.args_bytes).unwrap();
        // The `mesh` reference resolved to box's handle (0).
        assert_eq!(baked.mesh, Some(crate::ecs::MeshHandle(0)));
        // The `day_` name prefix resolved to Scene `day`'s id (1).
        assert_eq!(baked.scene, Some(crate::ecs::asset_id::AssetId(1)));
    }

    // A resource asset (here a Font) leaves no component def, so the lock
    // records it through `resource_locks` instead: name, kind, handle, args
    // hash, and the blob its payload landed in.
    #[test]
    fn build_pipeline_records_resource_lock_provenance() {
        let _guard = SHADER_BUILD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let world = concat!(
            r#"{"name":"f","type":"Font","args":{"size_px":20}}"#,
            "\n",
            r#"{"name":"pause","type":"View","args":{}}"#,
            "\n",
        );
        let result = build_pipeline_from_str(world, None).expect("build");

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

    // The visual_novel demo world (in concinnity-infra/worlds) exercises
    // Sprite + View + KeyBinding together. Validating it here catches asset
    // registration / pipeline regressions before we ship the world.
    #[test]
    fn visual_novel_world_validates() {
        // Inline a representative subset of the world so the test stays
        // hermetic (no infra path lookup needed). Covers: an initial View,
        // a Sprite under that view's prefix, a TextLabel under it, a
        // HitRegion firing view:show on another View, and a KeyBinding to
        // toggle a third (modal) View.
        let world = r#"{"name":"gfx","type":"GraphicsConfig","args":{}}
{"name":"f","type":"Font","args":{"size_px":20}}
{"name":"title_menu","type":"View","args":{"initial":true}}
{"name":"title_menu_bg","type":"Sprite","args":{"x":0,"y":0,"width":640,"height":360,"tint":[0.1,0.1,0.1,1]}}
{"name":"title_menu_lbl","type":"TextLabel","args":{"font":"f","content":"Start","x":260,"y":160}}
{"name":"title_menu_btn","type":"HitRegion","args":{"x":260,"y":156,"width":120,"height":40,"label":"title_menu_lbl","action":"view:show:vn_page_1"}}
{"name":"vn_page_1","type":"View","args":{}}
{"name":"vn_page_1_text","type":"TextLabel","args":{"font":"f","content":"hello","x":40,"y":40}}
{"name":"vn_page_1_next","type":"HitRegion","args":{"x":0,"y":0,"width":640,"height":360,"action":"view:show:title_menu"}}
{"name":"pause_menu","type":"View","args":{}}
{"name":"pause_menu_dim","type":"Sprite","args":{"x":0,"y":0,"width":640,"height":360,"tint":[0,0,0,0.6]}}
{"name":"esc","type":"KeyBinding","args":{"key":"Escape","action":"view:toggle:pause_menu"}}
"#;
        validate_world_jsonl(world).expect("visual_novel-shaped world should validate");
    }

    // `view:show:<name>` / `view:toggle:<name>` action targets are
    // rewritten to interned ids at build time, like `scene:<name>`.
    #[test]
    fn build_pipeline_resolves_view_action_refs() {
        let world = concat!(
            r#"{"name":"pause_menu","type":"View","args":{}}"#,
            "\n",
            r#"{"name":"btn","type":"HitRegion","args":{"x":0,"y":0,"width":10,"height":10,"action":"view:toggle:pause_menu"}}"#,
            "\n",
            r#"{"name":"esc","type":"KeyBinding","args":{"key":"Escape","action":"view:toggle:pause_menu"}}"#,
            "\n",
        );
        let result = build_pipeline_from_str(world, None).expect("build");
        // pause_menu interned id = 0 (first declared name).
        let btn = result
            .defs
            .iter()
            .find(|d| d.name == Some(crate::ecs::asset_id::AssetId(1)))
            .expect("HitRegion def");
        let baked: crate::assets::HitRegion = postcard::from_bytes(&btn.args_bytes).unwrap();
        assert_eq!(baked.action, "view:toggle:0");

        let esc = result
            .defs
            .iter()
            .find(|d| d.name == Some(crate::ecs::asset_id::AssetId(2)))
            .expect("KeyBinding def");
        let baked: crate::assets::KeyBinding = postcard::from_bytes(&esc.args_bytes).unwrap();
        assert_eq!(baked.action, "view:toggle:0");
    }

    // A Sprite/TextLabel/HitRegion named `<view>_*` has its `view` arg
    // resolved from the prefix at build time, mirroring Prop scene refs.
    #[test]
    fn build_pipeline_resolves_view_prefix_on_ui_assets() {
        let _guard = SHADER_BUILD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let world = concat!(
            r#"{"name":"pause_menu","type":"View","args":{}}"#,
            "\n",
            r#"{"name":"pause_menu_dim","type":"Sprite","args":{"x":0,"y":0,"width":10,"height":10}}"#,
            "\n",
            r#"{"name":"pause_menu_title","type":"TextLabel","args":{"font":"f","content":"x","x":0,"y":0}}"#,
            "\n",
            r#"{"name":"pause_menu_btn","type":"HitRegion","args":{"x":0,"y":0,"width":10,"height":10,"action":"view:hide"}}"#,
            "\n",
            r#"{"name":"f","type":"Font","args":{"size_px":16}}"#,
            "\n",
        );
        let result = build_pipeline_from_str(world, None).expect("build");
        // pause_menu interned id = 0; the UI assets intern in declaration order.
        let baked_view = |id: u32, expect: &str| {
            let def = result
                .defs
                .iter()
                .find(|d| d.name == Some(crate::ecs::asset_id::AssetId(id)))
                .unwrap_or_else(|| panic!("expected a def for {expect}"));
            let ct = crate::registry::ComponentType::from_discriminant(def.discriminant)
                .unwrap_or_else(|| panic!("{expect}: unknown discriminant"));
            match ct {
                crate::registry::ComponentType::Sprite => {
                    postcard::from_bytes::<crate::assets::Sprite>(&def.args_bytes)
                        .unwrap()
                        .view
                }
                crate::registry::ComponentType::TextLabel => {
                    postcard::from_bytes::<crate::assets::TextLabel>(&def.args_bytes)
                        .unwrap()
                        .view
                }
                crate::registry::ComponentType::HitRegion => {
                    postcard::from_bytes::<crate::assets::HitRegion>(&def.args_bytes)
                        .unwrap()
                        .view
                }
                other => panic!("{expect}: unexpected type {other:?}"),
            }
        };
        for (id, name) in [
            (1, "pause_menu_dim"),
            (2, "pause_menu_title"),
            (3, "pause_menu_btn"),
        ] {
            assert_eq!(
                baked_view(id, name),
                Some(crate::ecs::asset_id::AssetId(0)),
                "expected {name} to have view=0"
            );
        }
    }

    // Nested view names resolve by longest prefix: `<menu>_settings_*` binds
    // to the `<menu>_settings` view, not the enclosing `<menu>` view that is
    // declared first. (Regression: first-match claimed the nested elements,
    // so a MainMenu's settings sub-view rendered on top of the main menu.)
    #[test]
    fn resolve_scene_refs_picks_longest_view_prefix() {
        let mk = |name: &str, ty: &str| crate::world::WorldJsonlAsset {
            name: name.to_string(),
            asset_type: ty.to_string(),
            args: serde_json::json!({}),
        };
        let mut assets = vec![
            mk("menu", "View"),
            mk("menu_settings", "View"),
            mk("menu_title", "TextLabel"),
            mk("menu_settings_title", "TextLabel"),
        ];
        super::resolve_scene_refs(&mut assets);
        let view_of = |n: &str| {
            assets
                .iter()
                .find(|a| a.name == n)
                .and_then(|a| a.args.get("view"))
                .and_then(|v| v.as_str())
                .map(str::to_string)
        };
        assert_eq!(view_of("menu_title").as_deref(), Some("menu"));
        assert_eq!(
            view_of("menu_settings_title").as_deref(),
            Some("menu_settings")
        );
    }

    // Animation with no `source` is left byte-for-byte unchanged: the
    // inline-authored path must not regress.
    #[test]
    fn desugar_gltf_animations_skips_inline_clips() {
        let original = serde_json::json!({
            "target": "flag",
            "duration": 2.0,
            "tracks": [{"joint": 0, "keyframes": [{"time": 0.0, "rotation_deg": [0,0,0]}]}],
        });
        let mut assets = vec![crate::world::WorldJsonlAsset {
            name: "wave".to_string(),
            asset_type: "Animation".to_string(),
            args: original.clone(),
        }];
        desugar_gltf_animations(&mut assets).expect("desugar succeeds");
        assert_eq!(assets[0].args, original);
    }

    // Opting into root motion strips the root joint's X/Z travel into
    // `root_track` and anchors the pose; a clip without the flag is
    // untouched, and a second pass over already-baked args is a no-op.
    #[test]
    fn desugar_root_motion_bakes_the_root_track() {
        let walk = serde_json::json!({
            "target": "hero",
            "duration": 1.0,
            "root_motion": true,
            "tracks": [{"joint": 0, "keyframes": [
                {"time": 0.0, "translation": [0.0, 1.0, 0.0]},
                {"time": 1.0, "translation": [2.0, 1.0, 0.0]}
            ]}],
        });
        let plain = serde_json::json!({
            "target": "hero",
            "duration": 1.0,
            "tracks": [{"joint": 0, "keyframes": [
                {"time": 1.0, "translation": [2.0, 1.0, 0.0]}
            ]}],
        });
        let mut assets = vec![
            crate::world::WorldJsonlAsset {
                name: "walk".to_string(),
                asset_type: "Animation".to_string(),
                args: walk,
            },
            crate::world::WorldJsonlAsset {
                name: "plain".to_string(),
                asset_type: "Animation".to_string(),
                args: plain.clone(),
            },
        ];
        desugar_root_motion(&mut assets).expect("desugar succeeds");

        let baked = &assets[0].args;
        assert_eq!(baked["root_track"][1]["translation"][0], 2.0);
        assert_eq!(baked["root_track"][1]["translation"][1], 0.0);
        // The pose keeps Y but stays anchored on X.
        assert_eq!(baked["tracks"][0]["keyframes"][1]["translation"][0], 0.0);
        assert_eq!(baked["tracks"][0]["keyframes"][1]["translation"][1], 1.0);
        assert_eq!(assets[1].args, plain, "flag-less clip untouched");

        let after_first = assets[0].args.clone();
        desugar_root_motion(&mut assets).expect("second pass succeeds");
        assert_eq!(assets[0].args, after_first, "re-bake is a no-op");
    }

    #[test]
    fn voxel_chunk_payload_compiles_end_to_end() {
        let world = r#"{"name":"vert","type":"ShaderStage","args":{"kind":"vertex","source":"x.metal"}}
{"name":"frag","type":"ShaderStage","args":{"kind":"fragment","source":"x.metal"}}
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
        let bytes = crate::geometry::compile_voxel_chunk_payload(&chunk_args, bt).unwrap();
        assert!(!bytes.is_empty());
        let _ = world; // keeps the inline jsonl reference for documentation
    }

    fn wja(name: &str, ty: &str, args: serde_json::Value) -> crate::world::WorldJsonlAsset {
        crate::world::WorldJsonlAsset {
            name: name.to_string(),
            asset_type: ty.to_string(),
            args,
        }
    }

    fn ctx() -> crate::asset::BuildCtx<'static> {
        crate::asset::BuildCtx {
            name: "test",
            artifacts_dir: None,
            all_assets: &[],
        }
    }

    // A cache map that claims a compiled payload is already in hand for the
    // named asset, so every desugar pass must skip its source parse.
    fn hit_cache(name: &str) -> std::collections::HashMap<String, GltfCacheEntry> {
        let mut m = std::collections::HashMap::new();
        m.insert(
            name.to_string(),
            GltfCacheEntry {
                key: "k".to_string(),
                bytes: Some(vec![1, 2, 3]),
            },
        );
        m
    }

    #[test]
    fn build_from_path_missing_world_file_errors() {
        assert!(build_from_path("/no/such/world.jsonl").is_err());
    }

    #[test]
    fn build_pipeline_from_str_rejects_malformed_jsonl() {
        let Err(err) = build_pipeline_from_str("{not json\n", None) else {
            panic!("malformed line must not build");
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn build_pipeline_from_str_reports_unknown_asset_types() {
        let world = r#"{"name":"mystery","type":"NotAType","args":{}}"#;
        let Err(err) = build_pipeline_from_str(world, None) else {
            panic!("unknown type must not build");
        };
        assert!(
            err.to_string().contains("NotAType"),
            "error should name the unknown type: {err}"
        );
    }

    #[test]
    fn validate_asset_accepts_build_time_expansion_types() {
        // Build-time types are expanded before the runtime registry sees
        // them, so they validate structurally regardless of args.
        for ty in ["SceneImport", "Environment", "LightRig", "Prefab"] {
            validate_asset(ty, "x", &serde_json::json!({}))
                .unwrap_or_else(|e| panic!("{ty} should validate: {e}"));
        }
    }

    #[test]
    fn validate_asset_unknown_type_mentions_the_asset_name() {
        let err =
            validate_asset("Bogus", "my_thing", &serde_json::json!({})).expect_err("unknown type");
        assert!(err.contains("my_thing"), "got: {err}");
    }

    #[test]
    fn validate_asset_bad_args_mention_the_asset_name() {
        // `generator` must be a string; a number fails args deserialization.
        let err = validate_asset(
            "ProceduralMesh",
            "bad_mesh",
            &serde_json::json!({"generator": 5}),
        )
        .expect_err("bad args");
        assert!(err.contains("bad_mesh"), "got: {err}");
    }

    #[test]
    fn desugar_gltf_skinned_meshes_leaves_inline_and_cached_untouched() {
        let inline_args = serde_json::json!({"vertices": [], "indices": []});
        let cached_args = serde_json::json!({"source": "/no/such/hero.glb"});
        let mut assets = vec![
            wja("inline", SKINNED_MESH_TYPE, inline_args.clone()),
            wja("cached", SKINNED_MESH_TYPE, cached_args.clone()),
        ];
        desugar_gltf_skinned_meshes(&mut assets, &hit_cache("cached")).expect("desugar");
        // No source: untouched. Cache hit: the missing .glb is never parsed
        // and the args stay pre-desugar so the next probe key matches.
        assert_eq!(assets[0].args, inline_args);
        assert_eq!(assets[1].args, cached_args);
    }

    #[test]
    fn desugar_gltf_skinned_meshes_missing_source_errors() {
        let mut assets = vec![wja(
            "hero",
            SKINNED_MESH_TYPE,
            serde_json::json!({"source": "/no/such/hero.glb"}),
        )];
        let err = desugar_gltf_skinned_meshes(&mut assets, &Default::default())
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
        desugar_gltf_meshes(&mut assets, &hit_cache("cached")).expect("desugar");
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
        let err = desugar_gltf_meshes(&mut assets, &Default::default()).expect_err("missing .glb");
        assert!(err.to_string().contains("Asset 'crate_mesh'"), "got: {err}");
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

    #[test]
    fn desugar_gltf_animations_missing_source_errors() {
        let mut assets = vec![wja(
            "walk",
            "Animation",
            serde_json::json!({"source": "/no/such/anim.glb"}),
        )];
        let err = desugar_gltf_animations(&mut assets).expect_err("missing .glb");
        assert!(err.to_string().contains("Asset 'walk'"), "got: {err}");
    }

    #[test]
    fn desugar_gltf_animations_missing_named_clip_errors() {
        // The by-name lookup also starts by reading the file, so a missing
        // source fails before the name search; the error still names the asset.
        let mut assets = vec![wja(
            "run",
            "Animation",
            serde_json::json!({"source": "/no/such/anim.glb", "animation_name": "Run"}),
        )];
        let err = desugar_gltf_animations(&mut assets).expect_err("missing .glb");
        assert!(err.to_string().contains("Asset 'run'"), "got: {err}");
    }

    #[test]
    fn desugar_root_motion_rejects_malformed_args() {
        let mut assets = vec![wja(
            "walk",
            "Animation",
            serde_json::json!({"root_motion": true, "duration": "long"}),
        )];
        let err = desugar_root_motion(&mut assets).expect_err("bad duration");
        assert!(
            err.to_string()
                .contains("root-motion bake failed to parse args"),
            "got: {err}"
        );
    }

    #[test]
    fn desugar_root_motion_tolerates_a_clip_with_no_root_track() {
        // Only joint 1 is animated: there is nothing to strip from the root,
        // so the bake warns and leaves an empty curve rather than failing.
        let mut assets = vec![wja(
            "wave",
            "Animation",
            serde_json::json!({
                "root_motion": true,
                "duration": 1.0,
                "tracks": [{"joint": 1, "keyframes": [
                    {"time": 0.0, "translation": [1.0, 0.0, 0.0]}
                ]}],
            }),
        )];
        desugar_root_motion(&mut assets).expect("bake succeeds");
        assert_eq!(assets[0].args["root_track"], serde_json::json!([]));
        // The non-root track is untouched.
        assert_eq!(
            assets[0].args["tracks"][0]["keyframes"][0]["translation"][0],
            1.0
        );
    }

    #[test]
    fn resolve_scene_refs_prop_scene_prefix_rules() {
        let mut assets = vec![
            wja("level", "Scene", serde_json::json!({})),
            wja("level_boss", "Scene", serde_json::json!({})),
            wja("level_boss_door", "Prop", serde_json::json!({})),
            wja("level_gate", "Prop", serde_json::json!({"scene": "other"})),
            wja("solo_thing", "Prop", serde_json::json!({})),
        ];
        super::resolve_scene_refs(&mut assets);

        // Longest scene prefix wins for the nested name.
        assert_eq!(assets[2].args["scene"], "level_boss");
        // An authored `scene` arg is never overwritten.
        assert_eq!(assets[3].args["scene"], "other");
        // No matching prefix: no `scene` arg appears.
        assert!(assets[4].args.get("scene").is_none());
    }

    #[test]
    fn resolve_scene_refs_rewrites_action_names_to_interned_ids() {
        crate::ecs::asset_id::reset_interner();
        let mut assets = vec![
            wja(
                "btn",
                "HitRegion",
                serde_json::json!({"action": "view:show:pause"}),
            ),
            wja(
                "key",
                "KeyBinding",
                serde_json::json!({"action": "scene:day"}),
            ),
        ];
        super::resolve_scene_refs(&mut assets);

        // Names intern in resolution order on this thread's fresh interner:
        // "pause" -> 0, "day" -> 1.
        assert_eq!(assets[0].args["action"], "view:show:0");
        assert_eq!(assets[1].args["action"], "scene:1");
    }

    #[test]
    fn resolve_scene_refs_leaves_numeric_and_foreign_actions_alone() {
        let mut assets = vec![
            wja(
                "a",
                "HitRegion",
                serde_json::json!({"action": "view:toggle:3"}),
            ),
            wja("b", "HitRegion", serde_json::json!({"action": "quit"})),
            wja("c", "KeyBinding", serde_json::json!({"action": "scene:"})),
        ];
        super::resolve_scene_refs(&mut assets);

        // Already an id, not a recognised prefix, and an empty target: all
        // pass through unchanged.
        assert_eq!(assets[0].args["action"], "view:toggle:3");
        assert_eq!(assets[1].args["action"], "quit");
        assert_eq!(assets[2].args["action"], "scene:");
    }

    #[test]
    fn probe_gltf_cache_probes_only_source_backed_mesh_assets() {
        let assets = vec![
            wja("m", MESH_TYPE, serde_json::json!({"source": "x.glb"})),
            wja("inline", MESH_TYPE, serde_json::json!({"vertices": []})),
            wja(
                "s",
                SKINNED_MESH_TYPE,
                serde_json::json!({"source": "y.glb"}),
            ),
            wja(
                "p",
                "ProceduralMesh",
                serde_json::json!({"generator": "box"}),
            ),
        ];
        let probed = probe_gltf_cache(&assets, None);

        let mut names: Vec<&str> = probed.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        assert_eq!(names, vec!["m", "s"]);
        for entry in probed.values() {
            assert!(!entry.key.is_empty());
            // The payload cache is disabled under cargo test, so a probe can
            // only ever record a miss here.
            assert!(entry.bytes.is_none());
        }
    }

    #[test]
    fn compile_by_type_without_build_impl_errors() {
        let ct = ComponentType::parse("Prop").expect("Prop is a registered component");
        let err = compile_by_type(ct, &serde_json::json!({}), &ctx())
            .expect_err("Prop has no BuildAsset impl");
        assert!(err.to_string().contains("no BuildAsset impl"), "got: {err}");
    }

    #[test]
    fn source_files_by_type_defaults_to_empty() {
        let ct = ComponentType::parse("Prop").expect("Prop is a registered component");
        assert!(source_files_by_type(ct, &serde_json::json!({}), &ctx()).is_empty());
    }

    // AudioClip compiles through `ResourceAssetType` now, not `compile_by_type`
    // (it left the component registry). Its source-less error still surfaces, and
    // its source file is folded into the payload cache key.
    #[test]
    fn resource_asset_types_compile_audio_clip_texture_cubemap_env_lut_and_font() {
        use crate::resource_handles::ResourceAssetType;
        let rt = ResourceAssetType::parse("AudioClip").expect("AudioClip is a resource asset");
        let err = rt
            .compile_payload(&serde_json::json!({}))
            .expect_err("a source-less AudioClip must fail to compile");
        assert!(err.to_string().contains("missing 'source'"), "got: {err}");
        assert_eq!(
            rt.source_files(&serde_json::json!({"source": "a.wav"})),
            vec!["a.wav".to_string()]
        );
        assert!(rt.source_files(&serde_json::json!({})).is_empty());

        // Texture is also a resource asset (it left the component registry). A
        // procedural texture compiles a non-empty payload, and a file-backed one
        // folds its source into the payload cache key.
        let tex = ResourceAssetType::parse("Texture").expect("Texture is a resource asset");
        let bytes = tex
            .compile_payload(&serde_json::json!({"generator": "checker", "resolution": 32}))
            .expect("a procedural texture compiles");
        assert!(!bytes.is_empty());
        assert_eq!(
            tex.source_files(&serde_json::json!({"source": "a.png"})),
            vec!["a.png".to_string()]
        );

        // CubemapTexture is a resource asset too. Source-less args fail, and its
        // `.hdr` source folds into the payload cache key.
        let cube =
            ResourceAssetType::parse("CubemapTexture").expect("CubemapTexture is a resource asset");
        let err = cube
            .compile_payload(&serde_json::json!({}))
            .expect_err("a source-less CubemapTexture must fail to compile");
        assert!(
            err.to_string().contains("requires a `source` path"),
            "got: {err}"
        );
        assert_eq!(
            cube.source_files(&serde_json::json!({"source": "c.hdr"})),
            vec!["c.hdr".to_string()]
        );

        // EnvironmentMap and ColorLut are resource assets too. Both surface their
        // source-less error through `ResourceAssetType::compile_payload`, and fold
        // their `source` into the payload cache key.
        let env =
            ResourceAssetType::parse("EnvironmentMap").expect("EnvironmentMap is a resource asset");
        let err = env
            .compile_payload(&serde_json::json!({}))
            .expect_err("a source-less EnvironmentMap must fail to compile");
        assert!(
            err.to_string()
                .contains("requires either `source` or `generator`"),
            "got: {err}"
        );
        assert_eq!(
            env.source_files(&serde_json::json!({"source": "e.hdr"})),
            vec!["e.hdr".to_string()]
        );

        let lut = ResourceAssetType::parse("ColorLut").expect("ColorLut is a resource asset");
        let err = lut
            .compile_payload(&serde_json::json!({}))
            .expect_err("a source-less ColorLut must fail to compile");
        assert!(
            err.to_string().contains("requires a `source` path"),
            "got: {err}"
        );
        assert_eq!(
            lut.source_files(&serde_json::json!({"source": "l.cube"})),
            vec!["l.cube".to_string()]
        );

        // Font is a resource asset. The built-in font (empty `path`) compiles a
        // non-empty atlas, and a file-backed font folds its `path` (not `source`)
        // into the payload cache key.
        let font = ResourceAssetType::parse("Font").expect("Font is a resource asset");
        let bytes = font
            .compile_payload(&serde_json::json!({"size_px": 20}))
            .expect("the built-in font compiles");
        assert!(!bytes.is_empty());
        assert_eq!(
            font.source_files(&serde_json::json!({"path": "f.ttf"})),
            vec!["f.ttf".to_string()]
        );
        assert!(
            font.source_files(&serde_json::json!({"source": "x.ttf"}))
                .is_empty()
        );
    }

    // Dispatch coverage: compile_by_type / source_files_by_type route each
    // compiled ComponentType to its asset_impls wrapper.

    fn ct(name: &str) -> ComponentType {
        ComponentType::parse(name).unwrap_or_else(|| panic!("{name} is a registered component"))
    }

    // Arms whose outcome is deterministic from inline args alone: a valid
    // minimal payload for the ones that need no source file, and the expected
    // error for the ones that require a source but got none.
    #[test]
    fn compile_by_type_dispatches_deterministic_arms() {
        // Mesh is a resource asset now: it compiles through
        // `ResourceAssetType::compile_payload`, not the ComponentType dispatch.
        let mesh_bytes = crate::resource_handles::ResourceAssetType::Mesh
            .compile_payload(&serde_json::json!({"generator": "box", "half_extents": [1, 1, 1]}))
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
        use crate::resource_handles::ResourceAssetType;
        let rt = ResourceAssetType::SkinnedMesh;

        let ok = serde_json::json!({"vertices": [{"pos": [0.0, 0.0, 0.0]}], "indices": []});
        let bytes = rt.compile_payload(&ok).expect("skinned compiles");
        assert!(!bytes.is_empty());

        let no_verts = rt
            .compile_payload(&serde_json::json!({}))
            .expect_err("no vertices");
        assert!(
            no_verts.to_string().contains("at least one vertex"),
            "got: {no_verts}"
        );

        let bad_skeleton = rt
            .compile_payload(
                &serde_json::json!({"vertices": [{"pos": [0.0, 0.0, 0.0]}], "skeleton": 5}),
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
        let (baked_name, sm): (u32, crate::assets::SkinnedMesh) =
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
            name: "chunk",
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

    // The ShaderStage wrapper's non-compiling arms: a missing source is either
    // a hard error (Metal/HLSL) or the inline-GLSL stub (Vulkan). Neither shells
    // out to a shader toolchain, so the test stays backend-agnostic.
    #[test]
    fn compile_by_type_shader_stage_missing_source_does_not_shell_out() {
        let out = compile_by_type(
            ct("ShaderStage"),
            &serde_json::json!({"kind": "vertex"}),
            &ctx(),
        );
        match out {
            Ok(bytes) => assert!(bytes.is_empty(), "glsl stub yields empty bytes"),
            Err(e) => assert!(e.to_string().contains("no shader source"), "got: {e}"),
        }
    }

    // source_files_by_type routes to the two overriding wrappers: SdfVolume
    // returns the resolved shader path, ShaderStage short-circuits built-ins
    // and no-source to empty.
    #[test]
    fn source_files_by_type_covers_the_overriding_wrappers() {
        let dir = tempfile::tempdir().expect("tempdir");
        let shader = dir.path().join("blob.metal");
        std::fs::write(&shader, b"x").expect("write shader");
        let path = shader.to_str().unwrap();
        let sdf_args = serde_json::json!({
            "fragment_shaders": {"metal": path, "hlsl": path, "glsl": path}
        });
        assert_eq!(
            source_files_by_type(ct("SdfVolume"), &sdf_args, &ctx()),
            vec![path.to_string()]
        );
        assert!(source_files_by_type(ct("SdfVolume"), &serde_json::json!({}), &ctx()).is_empty());

        // A built-in shader name short-circuits to empty; so does no source.
        assert!(
            source_files_by_type(
                ct("ShaderStage"),
                &serde_json::json!({"source": "default.metal"}),
                &ctx()
            )
            .is_empty()
        );
        assert!(source_files_by_type(ct("ShaderStage"), &serde_json::json!({}), &ctx()).is_empty());
    }
}
