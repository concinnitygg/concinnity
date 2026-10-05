//! The payload cache probe that runs ahead of desugar, and the compile + pack
//! pass that turns the resolved defs into blob sections.

use concinnity_core::blob::{MeshBoundsRecord, SceneGroup};
use concinnity_core::components::FileKind;
use concinnity_core::ecs::{BlobAssetDef, ResourceRecord};
use concinnity_host::thread::asset_id;
use rayon::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::dispatch::build_asset;
use super::partition::ResourceJob;
use super::progress::{BuildStage, Progress, StageProgress};
use crate::authoring::registry::RegisteredType;
use crate::authoring::world::WorldJsonlAsset;
use crate::blob::PayloadPacker;

// Every entry this stage stores is a compiled payload; the scene expansion that
// shares the segment stores its own kind upstream.
const PAYLOAD: crate::cache::CacheEntryKind = crate::cache::CacheEntryKind::Payload;

// The resource kind of a job selected by `collect_resource_jobs`. Every entry
// there was chosen by having one, so the lookup cannot fail.
fn job_resource_kind(
    rt: crate::authoring::registry::RegisteredType,
) -> concinnity_core::blob::ResourceKind {
    rt.resource_kind()
        .expect("a resource job carries a resource type")
}

// Per-asset state recorded by `probe_mesh_payload_cache`. `key` is the cache key
// computed from the asset's pre-desugar args; `bytes` is `Some` when the
// cache already held a compiled payload for that key. On a hit, the desugar
// pass skips the .glb parse for this asset; on a miss, compile_and_pack
// stores the freshly compiled payload under the same `key` so the next
// build's probe can re-use it.
#[derive(Clone)]
pub(in crate::pipeline) struct MeshCacheEntry {
    pub(in crate::pipeline) key: String,
    pub(in crate::pipeline) bytes: Option<Vec<u8>>,
    // The capsule scaling recorded when the payload was compiled, for a target
    // a `bake: true` shape claims. `Some` exactly when `bytes` is.
    pub(in crate::pipeline) capsule_scale: Option<crate::compile::character::bake::CapsuleScale>,
}

// Hash every source-backed Mesh / SkinnedMesh asset's pre-desugar args and
// referenced source file (`.glb` or `.fbx`), then probe the content-addressed
// payload cache. Returns one entry per source-backed asset name. Assets
// without a `source` are not probed: their args don't depend on a file, so the
// regular per-asset cache path inside compile_and_pack_payloads is sufficient.
pub(in crate::pipeline) fn probe_mesh_payload_cache(
    assets: &[WorldJsonlAsset],
    assets_dir: Option<&Path>,
    platform: concinnity_core::platform::Platform,
) -> std::collections::HashMap<String, MeshCacheEntry> {
    let mut out = std::collections::HashMap::new();
    let empty: [WorldJsonlAsset; 0] = [];
    for asset in assets {
        let has_source = asset
            .args
            .get("source")
            .and_then(|v| v.as_str())
            .map(|s| !s.is_empty())
            .unwrap_or(false);
        if !has_source && asset.args.get("character_model").is_none() {
            continue;
        }

        // Both mesh kinds are resource assets: their caches key on the
        // resource discriminant and resource source list.
        let rt = match asset.asset_type {
            RegisteredType::Mesh | RegisteredType::SkinnedMesh => asset.asset_type,
            _ => continue,
        };
        let ctx = crate::asset::BuildCtx {
            name: asset.id.as_str(),
            platform,
            assets_dir,
            all_assets: &empty,
        };
        let discriminant = RESOURCE_CACHE_DISC_BASE + job_resource_kind(rt) as u8;
        let inputs = crate::asset::CacheInputs::extra(rt.source_files(&asset.args, assets_dir));
        // A shape baked into the mesh changes its payload as much as the
        // source does, so its args join the key.
        let shape = crate::compile::character::bake::baking_shape_args(assets, &asset.id);
        let keyed = match &shape {
            Some(shape) => serde_json::json!({"mesh": asset.args, "baked_shape": shape}),
            None => asset.args.clone(),
        };
        let key = crate::cache::payload_key(discriminant, &keyed, &ctx, &inputs);
        let mut bytes = crate::cache::load(PAYLOAD, &key);
        // That bake also scaled the capsule, by a factor drawn from the
        // pre-bake bind pose the payload replaces. Without the companion entry
        // holding it the bake cannot be replayed, so the payload counts as a
        // miss and the import runs again rather than the capsule going wrong.
        let capsule_scale = shape
            .is_some()
            .then(|| crate::compile::character::bake::load_capsule_scale(&key))
            .flatten();
        if shape.is_some() && capsule_scale.is_none() {
            bytes = None;
        }
        out.insert(
            asset.id.clone(),
            MeshCacheEntry {
                key,
                bytes,
                capsule_scale,
            },
        );
    }
    out
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
    kind: concinnity_core::blob::ResourceKind,
    handle: u32,
    bytes: Vec<u8>,
    is_data: bool,
    extra_data: Vec<u8>,
}

// The output of the compile + pack pass: the packed blob payload sections, the
// resource-stream records (each with its payload locator), and cache accounting.
pub(in crate::pipeline) struct CompiledOutput {
    pub(in crate::pipeline) scene_groups: Vec<SceneGroup>,
    pub(in crate::pipeline) mesh_bounds: Vec<MeshBoundsRecord>,
    pub(in crate::pipeline) mesh_component_names: Vec<(u32, String)>,
    pub(in crate::pipeline) blobs: Vec<Vec<u8>>,
    pub(in crate::pipeline) resources: Vec<ResourceRecord>,
    pub(in crate::pipeline) cache_hits: usize,
    pub(in crate::pipeline) cache_misses: usize,
}

// Baked geometry summary of one compiled static-mesh payload, keyed by its
// unified mesh-source handle. None when the payload does not parse as a
// static mesh (VoxelChunk voxel data, a malformed payload); absence means the
// runtime decodes that payload eagerly.
fn mesh_bounds_record(handle: u32, bytes: &[u8]) -> Option<MeshBoundsRecord> {
    let (verts, idxs, _) = concinnity_core::gfx::mesh_payload::deserialize_with_lods(bytes).ok()?;
    let first = verts.first()?;
    let mut min = first.pos;
    let mut max = first.pos;
    for v in &verts {
        for axis in 0..3 {
            min[axis] = min[axis].min(v.pos[axis]);
            max[axis] = max[axis].max(v.pos[axis]);
        }
    }
    Some(MeshBoundsRecord {
        handle,
        min,
        max,
        vertex_count: verts.len() as u32,
        index_count: idxs.len() as u32,
    })
}

// Read-only inputs to the compile + pack pass: the world being packed and the
// build context, as opposed to the def stream the pass mutates.
#[derive(Clone, Copy)]
pub(in crate::pipeline) struct PackContext<'a> {
    pub(in crate::pipeline) assets: &'a [WorldJsonlAsset],
    pub(in crate::pipeline) resource_jobs: &'a [ResourceJob],
    pub(in crate::pipeline) partition: &'a crate::compile::scene_partition::ScenePartition,
    pub(in crate::pipeline) mesh_source_handles: &'a concinnity_core::resource::ResourceHandles,
    pub(in crate::pipeline) max_blob_bytes: u64,
    pub(in crate::pipeline) assets_dir: Option<&'a Path>,
    pub(in crate::pipeline) platform: concinnity_core::platform::Platform,
    pub(in crate::pipeline) mesh_cache: &'a std::collections::HashMap<String, MeshCacheEntry>,
    pub(in crate::pipeline) progress: Progress<'a>,
}

pub(in crate::pipeline) fn compile_and_pack_payloads(
    named: &mut [(String, BlobAssetDef)],
    named_src: &[usize],
    pack_ctx: PackContext<'_>,
) -> std::io::Result<CompiledOutput> {
    let PackContext {
        assets,
        resource_jobs,
        partition,
        mesh_source_handles,
        max_blob_bytes,
        assets_dir,
        platform,
        mesh_cache,
        progress,
    } = pack_ctx;

    let compiled: Vec<(usize, RegisteredType)> = named
        .iter()
        .enumerate()
        .filter_map(|(i, (_, def))| {
            let ct = RegisteredType::from_discriminant(def.discriminant)?;
            let compiles = if ct == RegisteredType::File {
                // only compile File assets whose kind maps to a supported payload
                // `named[i]` maps to `assets[named_src[i]]`.
                assets[named_src[i]]
                    .args
                    .get("kind")
                    .and_then(|k| k.as_str())
                    .and_then(FileKind::from_ext)
                    .map(|fk| fk.is_mesh())
                    .unwrap_or(false)
            } else {
                ct.registration().needs_compilation()
            };
            compiles.then_some((i, ct))
        })
        .collect();

    // Snapshot each job's inputs so the parallel compile borrows nothing from
    // `named`, which is mutated afterwards to record payload locators. The raw
    // discriminant stays alongside the type because the payload cache keys on it.
    let jobs: Vec<ComponentJob> = compiled
        .iter()
        .map(|&(idx, ct)| {
            let (name, def) = &named[idx];
            (idx, name.clone(), ct, def.discriminant)
        })
        .collect();

    // Compile component and resource payloads in parallel, both streams at
    // once, on a pool sharing this thread's name and handle tables. Each job is
    // independent (it reads only its own args and produces its own payload
    // bytes) and the payload cache is content-addressed, so concurrent hits and
    // stores never collide. Each collect follows its job order, so packing
    // below stays deterministic.
    let cache_hits = AtomicUsize::new(0);
    let stage = progress.stage(
        BuildStage::Compile,
        (jobs.len() + resource_jobs.len()) as u32,
    );
    let shared = CompileShared {
        assets,
        named_src,
        assets_dir,
        platform,
        mesh_cache,
        stage: &stage,
        cache_hits: &cache_hits,
    };
    let (pending, resource_pending) = super::build_pool::build_pool()?.install(|| {
        rayon::join(
            || {
                jobs.par_iter()
                    .map(|job| compile_component(job, &shared))
                    .collect::<std::io::Result<Vec<_>>>()
            },
            || {
                resource_jobs
                    .par_iter()
                    .map(|job| compile_resource(job, &shared))
                    .collect::<std::io::Result<Vec<_>>>()
            },
        )
    });
    let (pending, resource_pending) = (pending?, resource_pending?);

    let cache_hits = cache_hits.into_inner();
    let cache_misses = pending.len() + resource_pending.len() - cache_hits;

    // Ownership of each payload, precomputed so the packing loops below can
    // mutate `named` freely. Resource jobs and `resource_pending` are
    // index-aligned.
    use crate::compile::scene_partition::Owner;
    let comp_owners: Vec<Owner> = pending
        .iter()
        .map(|(idx, _)| partition.owner(&named[*idx].0))
        .collect();
    let res_owners: Vec<Owner> = resource_jobs
        .iter()
        .map(|(asset_idx, _, _)| partition.owner(&assets[*asset_idx].id))
        .collect();

    // Baked AABB + counts for every static mesh payload, resource-stream Mesh
    // entries first (their resource handle IS the mesh-source handle) then the
    // compiled mesh-source components, sorted by handle for determinism.
    let mut mesh_bounds: Vec<MeshBoundsRecord> = Vec::new();
    for ((_, rt, handle), res) in resource_jobs.iter().zip(&resource_pending) {
        if *rt == crate::authoring::registry::RegisteredType::Mesh
            && let Some(record) = mesh_bounds_record(*handle, &res.bytes)
        {
            mesh_bounds.push(record);
        }
    }
    let mut mesh_component_names: Vec<(u32, String)> = Vec::new();
    for (idx, bytes) in &pending {
        let asset = &assets[named_src[*idx]];
        if !crate::authoring::resource_type::is_mesh_source(asset.asset_type, &asset.args) {
            continue;
        }
        let id = asset_id::intern(&asset.id);
        if let Some(handle) = mesh_source_handles.get(concinnity_core::blob::ResourceKind::Mesh, id)
        {
            // Handle -> asset name for mesh payloads riding component defs,
            // so a consumer can find any sub-mesh payload by unified handle
            // (resource-stream Mesh handles lead the space and resolve
            // through the resource records instead).
            mesh_component_names.push((handle, asset.id.clone()));
            if let Some(record) = mesh_bounds_record(handle, bytes) {
                mesh_bounds.push(record);
            }
        }
    }
    mesh_bounds.sort_unstable_by_key(|r| r.handle);

    // One group per scene (declaration order, possibly empty), carrying the
    // resource-stream entries and payload defs that scene exclusively owns.
    let scene_groups: Vec<SceneGroup> = (0..partition.scenes.len())
        .map(|s| SceneGroup {
            scene: asset_id::intern(&partition.scenes[s]),
            resources: resource_jobs
                .iter()
                .zip(&res_owners)
                .filter(|(_, o)| **o == Owner::Scene(s))
                .map(|((_, rt, handle), _)| (job_resource_kind(*rt), *handle))
                .collect(),
            defs: pending
                .iter()
                .zip(&comp_owners)
                .filter(|(_, o)| **o == Owner::Scene(s))
                .filter_map(|((idx, _), _)| named[*idx].1.name)
                .collect(),
        })
        .collect();

    if pending.is_empty() && resource_pending.is_empty() {
        return Ok(CompiledOutput {
            scene_groups,
            mesh_bounds,
            mesh_component_names,
            blobs: vec![Vec::new()],
            resources: Vec::new(),
            cache_hits: 0,
            cache_misses: 0,
        });
    }

    // Pack payloads by ownership group: the global set first (blob 0 onward),
    // then each scene's exclusive set starting at a fresh blob, so a scene's
    // payloads are contiguous and stay unread until the scene loads. Within a
    // group, component payloads pack before resource payloads, both in their
    // stream order. One packer, so every group addresses the same blob space;
    // record order in the metadata streams is unchanged (locators are
    // per-record, so packing order is independent).
    let mut packer = PayloadPacker::new(max_blob_bytes);
    let mut resource_locators: Vec<Option<concinnity_core::ecs::PayloadLocator>> =
        vec![None; resource_pending.len()];

    for group in 0..=partition.scenes.len() {
        let owner = match group {
            0 => Owner::Global,
            s => Owner::Scene(s - 1),
        };
        if group > 0 {
            packer.start_group();
        }
        for ((idx, bytes), item_owner) in pending.iter().zip(&comp_owners) {
            if *item_owner == owner {
                named[*idx].1.payload = Some(packer.push(bytes));
            }
        }
        for (i, (res, item_owner)) in resource_pending.iter().zip(&res_owners).enumerate() {
            if !res.is_data && *item_owner == owner {
                resource_locators[i] = Some(packer.push(&res.bytes));
            }
        }
    }

    let mut resources: Vec<ResourceRecord> = Vec::with_capacity(resource_pending.len());
    for (pending, locator) in resource_pending.iter().zip(resource_locators) {
        // A data resource (Material) carries its bytes inline; a payload
        // resource parks its bytes in a blob section and records the locator,
        // plus any hybrid baked data (SkinnedMesh) inline beside it.
        let (payload, data_bytes) = if pending.is_data {
            (None, pending.bytes.clone())
        } else {
            (locator, pending.extra_data.clone())
        };
        resources.push(ResourceRecord {
            resource_kind: pending.kind,
            handle: pending.handle,
            payload,
            data_bytes,
        });
    }

    Ok(CompiledOutput {
        scene_groups,
        mesh_bounds,
        mesh_component_names,
        blobs: packer.finish(),
        resources,
        cache_hits,
        cache_misses,
    })
}

// A component def queued for compile: its index in `named`, its name, its type,
// and the raw discriminant the payload cache keys on.
type ComponentJob = (usize, String, RegisteredType, u8);

// What every compile job reads besides its own args, plus the progress counter
// and cache-hit tally they all share.
struct CompileShared<'a> {
    assets: &'a [WorldJsonlAsset],
    named_src: &'a [usize],
    assets_dir: Option<&'a Path>,
    platform: concinnity_core::platform::Platform,
    mesh_cache: &'a std::collections::HashMap<String, MeshCacheEntry>,
    stage: &'a StageProgress<'a>,
    cache_hits: &'a AtomicUsize,
}

// Compile one component payload, or serve it from the probe or the payload
// cache when its inputs are unchanged.
fn compile_component(
    (idx, name, ct, discriminant): &ComponentJob,
    shared: &CompileShared<'_>,
) -> std::io::Result<(usize, Vec<u8>)> {
    let ct = *ct;
    shared.stage.begin(name);

    // The job carries the `named` index; map it to its source asset via
    // `named_src` (`named` is not 1:1 with `assets` once resource assets are
    // partitioned out).
    let asset_args = &shared.assets[shared.named_src[*idx]].args;

    let ctx = crate::asset::BuildCtx {
        name: name.as_str(),
        platform: shared.platform,
        assets_dir: shared.assets_dir,
        all_assets: shared.assets,
    };
    let build = build_asset(ct).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Asset '{name}' is marked Compiled but has no BuildAsset impl (RegisteredType {ct:?})"
            ),
        )
    })?;

    // GLB-sourced Mesh / SkinnedMesh assets are probed before desugar; honor
    // those results here so the .glb parse really is skipped on cache hits. On
    // a miss the precomputed key is used at store time, keeping the next
    // build's probe valid.
    let reused = |bytes: Vec<u8>| {
        shared.cache_hits.fetch_add(1, Ordering::Relaxed);
        shared.stage.end(name, true);
        Ok((*idx, bytes))
    };
    if let Some(entry) = shared.mesh_cache.get(name) {
        if let Some(bytes) = &entry.bytes {
            return reused(bytes.clone());
        }
        let compiled_bytes = (build.compile)(asset_args, &ctx)?;
        crate::cache::store(PAYLOAD, &entry.key, &compiled_bytes);
        shared.stage.end(name, false);
        return Ok((*idx, compiled_bytes));
    }

    // Reuse a cached payload when the asset's inputs are unchanged; otherwise
    // compile and populate the cache for the next build.
    let inputs = (build.cache_inputs)(asset_args, &ctx);
    let key = crate::cache::payload_key(*discriminant, asset_args, &ctx, &inputs);
    if let Some(bytes) = crate::cache::load(PAYLOAD, &key) {
        return reused(bytes);
    }
    let compiled_bytes = (build.compile)(asset_args, &ctx)?;
    crate::cache::store(PAYLOAD, &key, &compiled_bytes);
    shared.stage.end(name, false);
    Ok((*idx, compiled_bytes))
}

// Compile one resource-stream payload, or serve it from the probe or the
// payload cache when its inputs are unchanged. Bypasses the
// `BuildAsset`/`RegisteredType` path a component takes, since a resource is
// not a component.
fn compile_resource(
    (asset_idx, rt, handle): &ResourceJob,
    shared: &CompileShared<'_>,
) -> std::io::Result<PendingResource> {
    let asset = &shared.assets[*asset_idx];
    shared.stage.begin(&asset.id);
    let ctx = crate::asset::BuildCtx {
        name: asset.id.as_str(),
        platform: shared.platform,
        assets_dir: shared.assets_dir,
        all_assets: shared.assets,
    };
    let extra_data = rt.compile_data(&asset.id, &asset.args)?.unwrap_or_default();
    // A glTF/FBX-sourced mesh was probed before desugar; honor that result so
    // the source parse really is skipped on a hit and the pre-desugar key is
    // reused at store time (same contract as the component gltf-cache path).
    let (bytes, reused) = if let Some(entry) = shared.mesh_cache.get(&asset.id) {
        match &entry.bytes {
            Some(bytes) => (bytes.clone(), true),
            None => {
                let compiled = rt.compile_payload(&asset.args, shared.assets_dir)?;
                crate::cache::store(PAYLOAD, &entry.key, &compiled);
                (compiled, false)
            }
        }
    } else {
        // Every resource asset compiles identically on every backend, so its
        // entry is shared across a DirectX and a Vulkan cook.
        let inputs =
            crate::asset::CacheInputs::extra(rt.source_files(&asset.args, shared.assets_dir));
        let key = crate::cache::payload_key(
            RESOURCE_CACHE_DISC_BASE + job_resource_kind(*rt) as u8,
            &rt.payload_key_args(&asset.args),
            &ctx,
            &inputs,
        );
        match crate::cache::load(PAYLOAD, &key) {
            Some(bytes) => (bytes, true),
            None => {
                let compiled = rt.compile_payload(&asset.args, shared.assets_dir)?;
                crate::cache::store(PAYLOAD, &key, &compiled);
                (compiled, false)
            }
        }
    };
    if reused {
        shared.cache_hits.fetch_add(1, Ordering::Relaxed);
    }
    shared.stage.end(&asset.id, reused);
    Ok(PendingResource {
        kind: job_resource_kind(*rt),
        handle: *handle,
        bytes,
        is_data: rt.is_data(),
        extra_data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset_api::{self, AssetRequest};
    use crate::pipeline::fixtures::wja;

    #[test]
    fn probe_mesh_payload_cache_probes_only_source_backed_mesh_assets() {
        let assets = vec![
            wja(
                "m",
                RegisteredType::Mesh,
                serde_json::json!({"source": "x.glb"}),
            ),
            wja(
                "inline",
                RegisteredType::Mesh,
                serde_json::json!({"vertices": []}),
            ),
            wja(
                "s",
                RegisteredType::SkinnedMesh,
                serde_json::json!({"source": "y.glb"}),
            ),
            wja(
                "p",
                RegisteredType::ProceduralMesh,
                serde_json::json!({"generator": "box"}),
            ),
        ];
        let probed =
            probe_mesh_payload_cache(&assets, None, concinnity_core::platform::Platform::Metal);

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

    // A source-backed asset that is neither mesh kind is skipped: its payload
    // cache is handled by the per-asset path inside the compile pass.
    #[test]
    fn probe_mesh_payload_cache_skips_a_source_backed_non_mesh_asset() {
        let assets = vec![
            wja(
                "tex",
                RegisteredType::Texture,
                serde_json::json!({"source": "wall.png"}),
            ),
            wja(
                "m",
                RegisteredType::Mesh,
                serde_json::json!({"source": "x.glb"}),
            ),
        ];
        let probed =
            probe_mesh_payload_cache(&assets, None, concinnity_core::platform::Platform::Metal);
        assert_eq!(probed.len(), 1);
        assert!(probed.contains_key("m"));
    }

    use concinnity_core::blob::ResourceKind;

    fn procedural_mesh_def() -> BlobAssetDef {
        asset_api::create_asset_def(&AssetRequest {
            asset_type: "ProceduralMesh".to_string(),
            args: Some(serde_json::json!({"generator": "box"})),
        })
        .expect("ProceduralMesh def")
    }

    // The pre-desugar probe is the whole point of the glTF cache: when it holds
    // bytes for an asset, neither the component nor the resource path may touch
    // the source again. Both assets here name inputs that would fail to
    // compile, so a recompile would be loud.
    #[test]
    fn compile_and_pack_payloads_serves_probed_bytes_without_recompiling() {
        let assets = vec![
            wja(
                "shape",
                RegisteredType::ProceduralMesh,
                serde_json::json!({"generator": "not_a_generator"}),
            ),
            wja(
                "body",
                RegisteredType::Mesh,
                serde_json::json!({"source": "/no/such/body.glb"}),
            ),
        ];
        let mut named = vec![("shape".to_string(), procedural_mesh_def())];
        let resource_jobs = vec![(1usize, RegisteredType::Mesh, 0u32)];
        let mut cache = std::collections::HashMap::new();
        cache.insert(
            "shape".to_string(),
            MeshCacheEntry {
                key: "shape-key".to_string(),
                bytes: Some(vec![1, 2, 3]),
                capsule_scale: None,
            },
        );
        cache.insert(
            "body".to_string(),
            MeshCacheEntry {
                key: "body-key".to_string(),
                bytes: Some(vec![4, 5, 6, 7]),
                capsule_scale: None,
            },
        );

        let out = compile_and_pack_payloads(
            &mut named,
            &[0],
            PackContext {
                platform: concinnity_core::platform::Platform::Metal,
                assets: &assets,
                resource_jobs: &resource_jobs,
                partition: &crate::compile::scene_partition::partition_scenes(&assets),
                mesh_source_handles: &Default::default(),
                max_blob_bytes: 1024,
                assets_dir: None,
                mesh_cache: &cache,
                progress: Progress::none(),
            },
        )
        .expect("probed payloads need no compiler");

        assert_eq!(out.cache_hits, 2);
        assert_eq!(out.cache_misses, 0);
        // Components pack first, then the resource stream, into one blob.
        assert_eq!(out.blobs, vec![vec![1, 2, 3, 4, 5, 6, 7]]);
        let component = named[0].1.payload.as_ref().expect("component locator");
        assert_eq!(
            (component.blob_index, component.offset, component.len),
            (0, 0, 3)
        );
        let resource = out.resources[0].payload.as_ref().expect("resource locator");
        assert_eq!(
            (resource.blob_index, resource.offset, resource.len),
            (0, 3, 4)
        );
        assert_eq!(out.resources[0].resource_kind, ResourceKind::Mesh);
        assert_eq!(out.resources[0].handle, 0);
    }

    // A scene-exclusive resource packs into its own blob after the global set,
    // and the scene group records it; record order stays resource_jobs order.
    #[test]
    fn scene_owned_payloads_pack_into_their_own_blob() {
        let assets = vec![
            wja("day", RegisteredType::Scene, serde_json::json!({})),
            wja(
                "day_prop",
                RegisteredType::Prop,
                serde_json::json!({"mesh":"day_mesh","scene":"day"}),
            ),
            wja(
                "bg_prop",
                RegisteredType::Prop,
                serde_json::json!({"mesh":"bg_mesh"}),
            ),
            wja(
                "day_mesh",
                RegisteredType::Mesh,
                serde_json::json!({"source": "/no/such/day.glb"}),
            ),
            wja(
                "bg_mesh",
                RegisteredType::Mesh,
                serde_json::json!({"source": "/no/such/bg.glb"}),
            ),
        ];
        let mut named: Vec<(String, BlobAssetDef)> = Vec::new();
        let resource_jobs = vec![
            (3usize, RegisteredType::Mesh, 0u32),
            (4usize, RegisteredType::Mesh, 1u32),
        ];
        let cache = std::collections::HashMap::from([
            (
                "day_mesh".to_string(),
                MeshCacheEntry {
                    key: "day-key".to_string(),
                    bytes: Some(vec![0xDD; 4]),
                    capsule_scale: None,
                },
            ),
            (
                "bg_mesh".to_string(),
                MeshCacheEntry {
                    key: "bg-key".to_string(),
                    bytes: Some(vec![0xBB; 2]),
                    capsule_scale: None,
                },
            ),
        ]);

        let out = compile_and_pack_payloads(
            &mut named,
            &[],
            PackContext {
                platform: concinnity_core::platform::Platform::Metal,
                assets: &assets,
                resource_jobs: &resource_jobs,
                partition: &crate::compile::scene_partition::partition_scenes(&assets),
                mesh_source_handles: &Default::default(),
                max_blob_bytes: 1 << 20,
                assets_dir: None,
                mesh_cache: &cache,
                progress: Progress::none(),
            },
        )
        .expect("probed payloads need no compiler");

        // Global set (bg) fills blob 0; day's exclusive mesh starts blob 1.
        assert_eq!(out.blobs, vec![vec![0xBB; 2], vec![0xDD; 4]]);
        let day = out.resources[0].payload.as_ref().expect("day locator");
        assert_eq!((day.blob_index, day.offset, day.len), (1, 0, 4));
        let bg = out.resources[1].payload.as_ref().expect("bg locator");
        assert_eq!((bg.blob_index, bg.offset, bg.len), (0, 0, 2));

        assert_eq!(out.scene_groups.len(), 1);
        assert_eq!(out.scene_groups[0].resources, vec![(ResourceKind::Mesh, 0)]);
        assert!(out.scene_groups[0].defs.is_empty());
    }

    // Every compiled static-mesh payload gets a baked AABB + counts record
    // keyed by its mesh-source handle.
    #[test]
    fn mesh_bounds_are_baked_for_compiled_mesh_sources() {
        let assets = vec![wja(
            "shape",
            RegisteredType::ProceduralMesh,
            serde_json::json!({"generator": "box"}),
        )];
        let mut named = vec![("shape".to_string(), procedural_mesh_def())];
        let mut handles = concinnity_core::resource::ResourceHandles::default();
        crate::resource_handles::assign_mesh_source_handles(&mut handles, &assets);
        let out = compile_and_pack_payloads(
            &mut named,
            &[0],
            PackContext {
                platform: concinnity_core::platform::Platform::Metal,
                assets: &assets,
                resource_jobs: &[],
                partition: &crate::compile::scene_partition::partition_scenes(&assets),
                mesh_source_handles: &handles,
                max_blob_bytes: 1 << 20,
                assets_dir: None,
                mesh_cache: &Default::default(),
                progress: Progress::none(),
            },
        )
        .expect("box compiles");
        assert_eq!(out.mesh_bounds.len(), 1);
        let record = out.mesh_bounds[0];
        assert_eq!(record.handle, 0);
        assert!(record.vertex_count > 0 && record.index_count > 0);
        for axis in 0..3 {
            assert!(record.min[axis] < record.max[axis]);
        }
    }

    // A probe that recorded a miss compiles for real, on both the component and
    // the resource path, and both payloads land in the packed blob.
    #[test]
    fn compile_and_pack_payloads_compiles_a_probe_miss() {
        let assets = vec![
            wja(
                "shape",
                RegisteredType::ProceduralMesh,
                serde_json::json!({"generator": "box"}),
            ),
            wja(
                "body",
                RegisteredType::Mesh,
                serde_json::json!({"generator": "sphere", "radius": 1.0}),
            ),
        ];
        let mut named = vec![("shape".to_string(), procedural_mesh_def())];
        let resource_jobs = vec![(1usize, RegisteredType::Mesh, 0u32)];
        let miss = |key: &str| MeshCacheEntry {
            key: key.to_string(),
            bytes: None,
            capsule_scale: None,
        };
        let cache = std::collections::HashMap::from([
            ("shape".to_string(), miss("shape-key")),
            ("body".to_string(), miss("body-key")),
        ]);

        let out = compile_and_pack_payloads(
            &mut named,
            &[0],
            PackContext {
                platform: concinnity_core::platform::Platform::Metal,
                assets: &assets,
                resource_jobs: &resource_jobs,
                partition: &crate::compile::scene_partition::partition_scenes(&assets),
                mesh_source_handles: &Default::default(),
                max_blob_bytes: 1 << 20,
                assets_dir: None,
                mesh_cache: &cache,
                progress: Progress::none(),
            },
        )
        .expect("a probe miss compiles");

        assert_eq!(out.cache_hits, 0);
        assert_eq!(out.cache_misses, 2);
        let component = named[0].1.payload.as_ref().expect("component locator");
        let resource = out.resources[0].payload.as_ref().expect("resource locator");
        assert!(component.len > 0);
        assert!(resource.len > 0);
        assert_eq!(
            out.blobs[0].len() as u64,
            component.len + resource.len,
            "both compiled payloads land in the blob"
        );
    }

    // Resource payloads compile in parallel, but the records come back in job
    // order with the bytes a serial compile on the calling thread produces.
    // Materials resolve texture names through the build's handle map and a
    // SkinnedMesh bakes its interned name, so a worker that cannot see this
    // thread's tables would bake different bytes.
    #[test]
    fn parallel_resource_compile_matches_a_serial_compile_in_job_order() {
        use concinnity_core::ecs::asset_id::AssetId;
        use concinnity_host::thread::asset_id;

        let tri = serde_json::json!({
            "vertices": [{"pos": [0.0, 0.0, 0.0]}, {"pos": [1.0, 0.0, 0.0]}, {"pos": [0.0, 1.0, 0.0]}],
            "indices": [0, 1, 2],
        });
        let assets = vec![
            wja("floor_tex", RegisteredType::Texture, serde_json::json!({})),
            wja("moss_tex", RegisteredType::Texture, serde_json::json!({})),
            wja(
                "stone",
                RegisteredType::Material,
                serde_json::json!({"albedo": "moss_tex", "roughness": 0.3}),
            ),
            wja(
                "ball",
                RegisteredType::Mesh,
                serde_json::json!({"generator": "sphere", "radius": 1.0}),
            ),
            wja("hero", RegisteredType::SkinnedMesh, tri),
            wja(
                "moss",
                RegisteredType::Material,
                serde_json::json!({"albedo": "floor_tex"}),
            ),
            wja(
                "pebble",
                RegisteredType::Mesh,
                serde_json::json!({"generator": "sphere", "radius": 0.25}),
            ),
            wja(
                "wood",
                RegisteredType::Material,
                serde_json::json!({"albedo": "moss_tex", "metallic": 0.5}),
            ),
        ];

        let names: Vec<&str> = assets.iter().map(|a| a.id.as_str()).collect();
        asset_id::reset_interner();
        asset_id::intern_all(&names);
        let handles = concinnity_core::resource::ResourceHandles::from_assets(
            assets.iter().enumerate().filter_map(|(i, a)| {
                a.asset_type
                    .resource_kind()
                    .map(|kind| (AssetId(i as u32), kind))
            }),
        );
        crate::resource_handles::reset_resource_handles();
        crate::resource_handles::install_resource_handles(handles.clone());
        let resource_jobs: Vec<ResourceJob> = (2..assets.len())
            .map(|i| {
                let rt = assets[i].asset_type;
                let kind = rt.resource_kind().unwrap();
                (i, rt, handles.get(kind, AssetId(i as u32)).unwrap())
            })
            .collect();

        let out = compile_and_pack_payloads(
            &mut [],
            &[],
            PackContext {
                platform: concinnity_core::platform::Platform::Metal,
                assets: &assets,
                resource_jobs: &resource_jobs,
                partition: &crate::compile::scene_partition::partition_scenes(&assets),
                mesh_source_handles: &handles,
                max_blob_bytes: 1 << 20,
                assets_dir: None,
                mesh_cache: &Default::default(),
                progress: Progress::none(),
            },
        )
        .expect("inline resources compile");
        assert_eq!((out.cache_hits, out.cache_misses), (0, resource_jobs.len()));
        assert_eq!(out.resources.len(), resource_jobs.len());

        for (&(i, rt, handle), record) in resource_jobs.iter().zip(&out.resources) {
            let asset = &assets[i];
            let payload = rt.compile_payload(&asset.args, None).unwrap();
            let extra = rt.compile_data(&asset.id, &asset.args).unwrap();
            assert_eq!(
                record.resource_kind,
                rt.resource_kind().unwrap(),
                "{}",
                asset.id
            );
            assert_eq!(record.handle, handle, "{}", asset.id);
            if rt.is_data() {
                assert!(record.payload.is_none(), "{}", asset.id);
                assert_eq!(record.data_bytes, payload, "{}", asset.id);
            } else {
                let loc = record.payload.as_ref().expect("payload locator");
                let start = loc.offset as usize;
                let packed = &out.blobs[loc.blob_index as usize][start..start + loc.len as usize];
                assert_eq!(packed, payload.as_slice(), "{}", asset.id);
                assert_eq!(record.data_bytes, extra.unwrap_or_default(), "{}", asset.id);
            }
        }

        // The reference itself resolved through the build's tables rather than
        // a fallback: stone's albedo is moss_tex's handle, hero bakes its id.
        let stone: concinnity_core::components::Material =
            postcard::from_bytes(&out.resources[0].data_bytes).unwrap();
        assert_eq!(
            stone.albedo,
            Some(concinnity_core::ecs::TextureHandle::new(1))
        );
        let (hero_id, _): (u32, concinnity_core::components::SkinnedMesh) =
            postcard::from_bytes(&out.resources[2].data_bytes).unwrap();
        assert_eq!(hero_id, 4);
    }

    // A world whose assets all carry inline args produces no payload sections
    // at all, and still reports one (empty) blob for the metadata to ride in.
    #[test]
    fn compile_and_pack_payloads_returns_one_empty_blob_for_a_payload_less_world() {
        let assets = vec![wja("day", RegisteredType::Scene, serde_json::json!({}))];
        let mut named = vec![(
            "day".to_string(),
            asset_api::create_asset_def(&AssetRequest {
                asset_type: "Scene".to_string(),
                args: Some(serde_json::json!({})),
            })
            .expect("Scene def"),
        )];
        let out = compile_and_pack_payloads(
            &mut named,
            &[0],
            PackContext {
                platform: concinnity_core::platform::Platform::Metal,
                assets: &assets,
                resource_jobs: &[],
                partition: &crate::compile::scene_partition::partition_scenes(&assets),
                mesh_source_handles: &Default::default(),
                max_blob_bytes: 1024,
                assets_dir: None,
                mesh_cache: &Default::default(),
                progress: Progress::none(),
            },
        )
        .expect("pack");

        assert_eq!(out.blobs, vec![Vec::<u8>::new()]);
        assert!(out.resources.is_empty());
        assert_eq!((out.cache_hits, out.cache_misses), (0, 0));
        assert!(named[0].1.payload.is_none());
    }

    // The compile pass selects its work by discriminant. A def carrying one the
    // component registry does not know is skipped, so an unrecognized record
    // cannot abort a build.
    #[test]
    fn compile_and_pack_payloads_skips_a_def_with_an_unknown_discriminant() {
        let assets = vec![wja(
            "mystery",
            RegisteredType::ProceduralMesh,
            serde_json::json!({}),
        )];
        let mut named = vec![(
            "mystery".to_string(),
            BlobAssetDef {
                name: None,
                discriminant: 200,
                args_bytes: Vec::new(),
                payload: None,
            },
        )];
        assert!(
            RegisteredType::from_discriminant(200).is_none(),
            "200 must stay outside the registered discriminant range"
        );

        let out = compile_and_pack_payloads(
            &mut named,
            &[0],
            PackContext {
                platform: concinnity_core::platform::Platform::Metal,
                assets: &assets,
                resource_jobs: &[],
                partition: &crate::compile::scene_partition::partition_scenes(&assets),
                mesh_source_handles: &Default::default(),
                max_blob_bytes: 1024,
                assets_dir: None,
                mesh_cache: &Default::default(),
                progress: Progress::none(),
            },
        )
        .expect("pack");

        assert!(named[0].1.payload.is_none());
        assert_eq!(out.blobs, vec![Vec::<u8>::new()]);
    }

    // The blob size ceiling is packing policy, not a format limit: payloads
    // that overflow it roll into the next blob and their locators follow.
    #[test]
    fn compile_and_pack_payloads_rolls_payloads_into_overflow_blobs() {
        let assets = vec![
            wja("a", RegisteredType::ProceduralMesh, serde_json::json!({})),
            wja("b", RegisteredType::ProceduralMesh, serde_json::json!({})),
        ];
        let mut named = vec![
            ("a".to_string(), procedural_mesh_def()),
            ("b".to_string(), procedural_mesh_def()),
        ];
        let cache = std::collections::HashMap::from([
            (
                "a".to_string(),
                MeshCacheEntry {
                    key: "a".to_string(),
                    bytes: Some(vec![0xAA; 6]),
                    capsule_scale: None,
                },
            ),
            (
                "b".to_string(),
                MeshCacheEntry {
                    key: "b".to_string(),
                    bytes: Some(vec![0xBB; 6]),
                    capsule_scale: None,
                },
            ),
        ]);

        let out = compile_and_pack_payloads(
            &mut named,
            &[0, 1],
            PackContext {
                platform: concinnity_core::platform::Platform::Metal,
                assets: &assets,
                resource_jobs: &[],
                partition: &crate::compile::scene_partition::partition_scenes(&assets),
                mesh_source_handles: &Default::default(),
                max_blob_bytes: 8,
                assets_dir: None,
                mesh_cache: &cache,
                progress: Progress::none(),
            },
        )
        .expect("pack");

        assert_eq!(out.blobs, vec![vec![0xAA; 6], vec![0xBB; 6]]);
        assert_eq!(named[0].1.payload.as_ref().unwrap().blob_index, 0);
        let second = named[1].1.payload.as_ref().unwrap();
        assert_eq!((second.blob_index, second.offset), (1, 0));
    }
}
