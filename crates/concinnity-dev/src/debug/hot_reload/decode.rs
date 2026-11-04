//! Off-thread asset decode + apply: spawns the decode + envmap-convolution
//! worker threads, then `poll_pending_assets` / `poll_pending_envmap` drain the
//! completed work on a later frame and push it through the backend `update_*`
//! calls. Keeps the (seconds-long) decode off the render thread.

use concinnity_core::fanout::FanOut;
use concinnity_engine::bake::texture::TextureImage;
use concinnity_engine::gfx::render_types;
use concinnity_engine::render::backend;
use concinnity_engine::render::world_load::source_maps::{
    ColorLutSource, MeshSourceEntry, SkinnedMeshSourceEntry, TextureSourceEntry,
};
use concinnity_host::thread::jobs::pool;
use concinnity_host::thread::{ThreadRole, set_current_thread_role};

use super::apply;
use super::state::{
    AssetHotReloadState, DecodedAssetBatch, DecodedColorLut, DecodedMesh, DecodedSkinnedMesh,
    DecodedTexture,
};

// Trigger a hot-reload pass. Spawns at most two worker threads: one for the
// IBL convolution (when an EnvironmentMap is declared) and one for the rest
// of the captured catalog (textures, ColorLut, file-backed Mesh /
// SkinnedMesh decode). The decode work is CPU-bound (PNG/JPEG inflate, glTF
// parse, vertex normalization) and used to stall the render thread for
// several seconds on a 43-texture / 14-mesh world; moving it off-thread lets
// the render loop keep drawing while the worker runs. [`poll_pending_assets`]
// + [`poll_pending_envmap`] drain the workers' results on a later frame and
// dispatch them through the matching backend `update_*` calls (those are
// fast, single MTL buffer / texture swaps).
pub(crate) fn reload_assets(state: &AssetHotReloadState) {
    spawn_asset_decode_worker(state);
    spawn_envmap_worker(state);
}

// Spawn the texture + ColorLut + Mesh + SkinnedMesh decode worker if any
// source catalog carries entries. The previous batch's receiver is
// consulted first: if a decode is already in flight, this pass logs and
// skips so the user re-triggers after the result lands. Sources are cloned
// into the worker closure so the worker has no reference back to
// AssetHotReloadState's storage and the render thread keeps mutating its
// own copies freely.
#[expect(
    clippy::let_underscore_must_use,
    reason = "a dropped receiver means the asset state went away and no longer wants the batch"
)]
fn spawn_asset_decode_worker(state: &AssetHotReloadState) {
    let has_work = !state.map.entries.is_empty()
        || state.color_lut.is_some()
        || !state.meshes.entries.is_empty()
        || !state.skinned_meshes.entries.is_empty();
    if !has_work {
        return;
    }
    let mut slot = match state.asset_batch_inflight.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if slot.is_some() {
        tracing::info!("asset hot-reload: decode batch skipped; previous worker still running");
        return;
    }
    // Snapshot every source the worker needs. Cheap: the maps are a few KB
    // total even on the showcase world.
    let textures = state.map.entries.clone();
    let color_lut = state.color_lut.clone();
    let meshes = state.meshes.entries.clone();
    let skinned = state.skinned_meshes.entries.clone();
    let totals = (
        textures.len(),
        meshes.len(),
        skinned.len(),
        color_lut.is_some(),
    );
    let (tx, rx) = std::sync::mpsc::channel();
    match std::thread::Builder::new()
        .name("cn-asset-reload".into())
        .spawn(move || {
            set_current_thread_role(ThreadRole::Background);
            // Run inside JobPool::install so any nested rayon `par_iter`
            // dispatches to the bounded `available_parallelism() - 1` pool
            // (mirroring the envmap worker). decode_asset_batch itself uses
            // par_iter to fan textures across cores.
            let batch =
                pool().scope(|| decode_asset_batch(&textures, color_lut, &meshes, &skinned));
            let _ = tx.send(batch);
        }) {
        Ok(_) => {
            *slot = Some(rx);
            tracing::info!(
                "asset hot-reload: decode batch spawned on worker thread \
                 ({} texture(s), {} Mesh(es), {} SkinnedMesh(es), {} ColorLut)",
                totals.0,
                totals.1,
                totals.2,
                i32::from(totals.3)
            );
        }
        Err(e) => {
            tracing::error!(
                "asset hot-reload: failed to spawn decode worker: {} \
                 (all assets kept their old payloads)",
                e
            );
        }
    }
}

// Spawn the EnvironmentMap convolution worker if an envmap is declared.
// Same pattern as the asset decode worker but kept on its own slot because
// the convolution often takes seconds (much longer than texture / mesh
// decode) and the user benefits from re-triggering envmap and asset
// reloads independently.
#[expect(
    clippy::let_underscore_must_use,
    reason = "a dropped receiver means the asset state went away and no longer wants the map"
)]
fn spawn_envmap_worker(state: &AssetHotReloadState) {
    let Some(env_map) = state.environment_map.as_ref() else {
        return;
    };
    let mut slot = match state.env_map_inflight.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if slot.is_some() {
        tracing::info!(
            "asset hot-reload: EnvironmentMap reload skipped for '{}'; previous \
             convolution still in flight on a worker thread",
            env_map.resolved_path
        );
        return;
    }
    let (tx, rx) = std::sync::mpsc::channel();
    let env_map_copy = env_map.clone();
    match std::thread::Builder::new()
        .name("cn-envmap-reload".into())
        .spawn(move || {
            set_current_thread_role(ThreadRole::Background);
            let result = pool().scope(|| {
                concinnity_cook::compile::environment_map::decode_source(
                    &env_map_copy.resolved_path,
                    env_map_copy.prefilter_face_size,
                    env_map_copy.irradiance_face_size,
                    env_map_copy.prefilter_samples,
                    env_map_copy.prefilter_clamp,
                )
                .map_err(|e| e.to_string())
            });
            let _ = tx.send(result);
        }) {
        Ok(_) => {
            *slot = Some(rx);
            tracing::info!(
                "asset hot-reload: EnvironmentMap convolution for '{}' spawned \
                 on worker thread (render loop keeps drawing; result will land \
                 on a later frame via poll_pending_envmap)",
                env_map.resolved_path
            );
        }
        Err(e) => {
            tracing::error!(
                "asset hot-reload: failed to spawn EnvironmentMap worker thread \
                 for '{}': {} (IBL kept its old payload)",
                env_map.resolved_path,
                e
            );
        }
    }
}

// Worker body: re-decode every captured source from disk and pack the
// results into a `DecodedAssetBatch`. Per-entry failures are logged at
// `error` and counted in `decode_failures` but never abort the rest of the
// batch: a half-written file is picked up on the next reload when the
// writer finishes. `parsed_glb_cache` amortizes `parse_glb` across every
// texture / mesh / skinned mesh that shares the same `.glb` (the showcase
// world fans 43 textures + 35 meshes out of a single 43 MB ABeautifulGame
// chess set, so without the cache the parse cost dominates the worker).
pub(super) fn decode_asset_batch(
    textures: &[TextureSourceEntry],
    color_lut: Option<ColorLutSource>,
    meshes: &[MeshSourceEntry],
    skinned: &[SkinnedMeshSourceEntry],
) -> DecodedAssetBatch {
    use rayon::prelude::*;
    use std::sync::Mutex;

    let mut batch = DecodedAssetBatch::default();
    // Shared parsed-glb cache across textures / meshes / skinned meshes.
    // Wrapped in a Mutex so the par_iter texture loop can populate it from
    // worker threads; contention only fires the first time each unique
    // source is seen.
    let parsed_glb_cache: Mutex<
        std::collections::HashMap<String, concinnity_cook::import::gltf_source::GltfDoc>,
    > = Mutex::new(std::collections::HashMap::new());

    // Helper: fetch (or parse + cache) the glTF doc for `source`. Returns a
    // cloned `GltfDoc`: the parsed document is reference-counted internally
    // and the resolved buffers are `Arc`-backed, so the clone is cheap.
    let assets_dir = crate::project::assets_dir();
    let load_glb = |source: &str| -> Result<concinnity_cook::import::gltf_source::GltfDoc, String> {
        if let Some(doc) = parsed_glb_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(source)
        {
            return Ok(doc.clone());
        }
        let doc = concinnity_cook::import::glb::parse_glb(source, assets_dir.as_deref())?;
        parsed_glb_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(source.to_string(), doc.clone());
        Ok(doc)
    };

    // Decode textures in parallel. Each entry is independent; the worker
    // pool's bounded thread count keeps the render thread breathing room.
    let texture_results: Vec<_> = textures
        .par_iter()
        .map(|entry| {
            let lower = entry.source.to_lowercase();
            let decoded = if std::path::Path::new(&lower)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("glb"))
                || std::path::Path::new(&lower)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("gltf"))
            {
                match load_glb(&entry.source) {
                    Ok(doc) => concinnity_cook::compile::texture::decode_glb_image_from_doc(
                        &doc,
                        &entry.source,
                        entry.image_index,
                    )
                    .map_err(|e| e.to_string()),
                    Err(e) => Err(e),
                }
            } else {
                concinnity_cook::compile::texture::decode_source(
                    &entry.source,
                    entry.image_index,
                    assets_dir.as_deref(),
                )
                .map_err(|e| e.to_string())
            };
            let image = decoded.map(|(w, h, px)| TextureImage::rgba8(w, h, px).with_mip_chain());
            (entry.clone(), image)
        })
        .collect();
    for (entry, decoded) in texture_results {
        match decoded {
            Ok(image) => batch.textures.push(DecodedTexture {
                slot: entry.slot,
                image,
                source: entry.source,
            }),
            Err(e) => {
                tracing::error!(
                    "asset hot-reload: failed to decode '{}': {} (slot {} kept its old pixels)",
                    entry.source,
                    e,
                    entry.slot
                );
                batch.decode_failures += 1;
            }
        }
    }

    // ColorLut: single source, decoded serially.
    if let Some(lut) = color_lut {
        match concinnity_cook::compile::color_lut::decode_source(&lut.resolved_path) {
            Ok((size, data)) => {
                batch.color_lut = Some(DecodedColorLut {
                    size,
                    data,
                    source: lut.resolved_path,
                });
            }
            Err(e) => {
                tracing::error!(
                    "asset hot-reload: failed to decode ColorLut '{}': {} (LUT kept its \
                     old payload)",
                    lut.resolved_path,
                    e
                );
                batch.decode_failures += 1;
            }
        }
    }

    // Static Meshes: re-import each via the shared glb cache. Serial: the
    // per-mesh decode is fast enough that the parallel-decode-then-collect
    // shape isn't worth the bookkeeping (most meshes share their parsed
    // doc, so the parse is amortized regardless).
    for (entry_idx, entry) in meshes.iter().enumerate() {
        let doc = match load_glb(&entry.source) {
            Ok(d) => d,
            Err(e) => {
                tracing::error!(
                    "asset hot-reload: failed to parse Mesh source '{}': {} \
                     ({} draw slot(s) kept their old geometry)",
                    entry.source,
                    e,
                    entry.draw_indices.len()
                );
                batch.decode_failures += 1;
                continue;
            }
        };
        match concinnity_cook::import::mesh_reimport::decode_mesh_from_parsed_glb(
            &doc,
            &entry.source,
            entry.primitive_index,
            entry.lod_levels,
            &entry.lod_distances,
        ) {
            Ok((verts, idxs, lods)) => batch.meshes.push(DecodedMesh {
                entry_idx,
                vertices: verts,
                indices: idxs,
                lod_alternates: lods,
            }),
            Err(e) => {
                tracing::error!(
                    "asset hot-reload: failed to decode Mesh primitive {} from \
                     '{}': {} ({} draw slot(s) kept their old geometry)",
                    entry.primitive_index,
                    entry.source,
                    e,
                    entry.draw_indices.len()
                );
                batch.decode_failures += 1;
            }
        }
    }

    // Skinned meshes: same pattern as static.
    for (entry_idx, entry) in skinned.iter().enumerate() {
        let doc = match load_glb(&entry.source) {
            Ok(d) => d,
            Err(e) => {
                tracing::error!(
                    "asset hot-reload: failed to parse SkinnedMesh source '{}': {} \
                     (skinned slot {} kept its old geometry)",
                    entry.source,
                    e,
                    entry.skinned_index
                );
                batch.decode_failures += 1;
                continue;
            }
        };
        match concinnity_cook::import::mesh_reimport::decode_skinned_from_parsed_glb(
            &doc,
            &entry.source,
            entry.skin_index,
        ) {
            Ok((verts, idxs, skeleton)) => batch.skinned_meshes.push(DecodedSkinnedMesh {
                entry_idx,
                vertices: verts,
                indices: idxs,
                skeleton,
            }),
            Err(e) => {
                tracing::error!(
                    "asset hot-reload: failed to decode SkinnedMesh '{}': {} \
                     (skinned slot {} kept its old geometry)",
                    entry.source,
                    e,
                    entry.skinned_index
                );
                batch.decode_failures += 1;
            }
        }
    }

    batch
}

// Refresh `SkinnedMeshSourceEntry`s' captured `vertex_base` / `vertex_count`
// / `index_count` from the post-rebuild layouts returned by
// [`concinnity_engine::render::backend::RenderBackend::rebuild_skinned_geometry`]. Entries
// whose `skinned_index` is not in `layouts` are left untouched (the backend
// did not include them in the rebuild), but a typical Metal rebuild
// returns one layout per slot since the whole shared buffer is re-laid out,
// so every entry gets refreshed. Extracted from the apply path so the
// (testable) mutation is independent of the backend call.
pub(super) fn apply_skinned_layouts_to_entries(
    entries: &mut [SkinnedMeshSourceEntry],
    layouts: &[backend::SkinnedSlotLayout],
) {
    let layout_by_skinned: std::collections::HashMap<
        render_types::SkinnedIndex,
        &backend::SkinnedSlotLayout,
    > = layouts.iter().map(|l| (l.skinned_index, l)).collect();
    for entry in entries.iter_mut() {
        if let Some(layout) = layout_by_skinned.get(&entry.skinned_index) {
            entry.vertex_base = layout.vertex_base;
            entry.vertex_count = layout.vertex_count;
            entry.index_count = layout.index_count;
        }
    }
}

// Try to receive a completed off-thread asset decode batch and push every
// item through the matching backend `update_*` call. Called every frame from
// `GraphicsSystem::step`; cheap when nothing is in flight (a Mutex lock +
// `None` check). Per-item dispatches:
//
// - Textures → `update_texture_slot` / `update_normal_map_slot`.
// - ColorLut → `update_color_lut`.
// - Static Meshes → `update_mesh_geometry` per draw slot when the slot's
//   init-time vertex / index counts (and per-LOD counts) still match the
//   re-imported geometry; otherwise the whole entry's
//   `DrawGeometryUpdate`s are accumulated and applied in a single
//   `rebuild_static_geometry` call after the in-place loop.
// - SkinnedMeshes → `update_skinned_mesh_geometry` per slot when the
//   slot's init-time vertex / index counts still match the re-imported
//   geometry; otherwise the whole entry's
//   `SkinnedDrawGeometryUpdate`s are accumulated and applied in a single
//   `rebuild_skinned_geometry` call after the in-place loop. Joint-count
//   changes resize the backend's per-slot joint-matrix buffers via
//   `update_skinned_skeleton` and queue a `PendingSkeletonUpdate` for the
//   next `GraphicsSystem::step` to refresh the ECS-owned `SkeletonPose`.
//
// Returns `true` when a batch was applied (success or failure), `false`
// when still waiting or nothing scheduled.
pub(crate) fn poll_pending_assets(
    state: &mut AssetHotReloadState,
    backend: &mut dyn backend::RenderBackend,
) -> bool {
    let mut slot = match state.asset_batch_inflight.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let Some(rx) = slot.as_ref() else {
        return false;
    };
    let batch = match rx.try_recv() {
        Ok(b) => b,
        Err(std::sync::mpsc::TryRecvError::Empty) => return false,
        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
            *slot = None;
            tracing::error!(
                "asset hot-reload: decode worker disconnected without sending a \
                 batch (assets kept their old payloads)"
            );
            return true;
        }
    };
    *slot = None;
    drop(slot);

    let mut tally = apply::Tally::new(batch.decode_failures);
    apply::textures(backend, &batch.textures, &mut tally);
    // Everything past the textures rewrites built content in place.
    let Some(backend) = backend.live_edit() else {
        tracing::error!("asset hot-reload: the backend cannot rewrite a built world");
        return true;
    };
    if let Some(lut) = &batch.color_lut {
        apply::color_lut(backend, lut, &mut tally);
    }
    apply::static_meshes(backend, &state.meshes.entries, &batch.meshes, &mut tally);
    apply::skinned_meshes(state, backend, &batch.skinned_meshes, &mut tally);

    tracing::info!(
        "asset hot-reload: applied off-thread batch, reloaded {} asset(s) ({} failed)",
        tally.reloaded,
        tally.failed
    );
    true
}

// Try to receive a completed off-thread EnvironmentMap convolution and push
// the payload to the backend. Called every frame from `GraphicsSystem::step`;
// cheap when nothing is in flight (a Mutex lock + `None` check). Returns
// `true` when something was consumed (success or failure) so the caller can
// log a single line per result; `false` when still waiting or nothing
// scheduled.
pub(crate) fn poll_pending_envmap(
    state: &AssetHotReloadState,
    backend: &mut dyn backend::LiveEdit,
) -> bool {
    let mut slot = match state.env_map_inflight.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let Some(rx) = slot.as_ref() else {
        return false;
    };
    match rx.try_recv() {
        Ok(Ok(payload)) => {
            *slot = None;
            // Drop the lock before the (potentially long) backend call so a
            // re-entrant `reload_assets` invocation from the same step
            // doesn't deadlock. Backend update is fast (it's a single
            // cubemap-pair swap) so dropping is just defensive.
            drop(slot);
            match backend.update_environment_map(&payload) {
                Ok(()) => {
                    tracing::info!(
                        "asset hot-reload: EnvironmentMap convolution completed \
                         off-thread and applied to IBL cubes"
                    );
                }
                Err(e) => {
                    tracing::error!(
                        "asset hot-reload: backend rejected off-thread EnvironmentMap \
                         payload: {} (IBL kept its old payload)",
                        e
                    );
                }
            }
            true
        }
        Ok(Err(msg)) => {
            *slot = None;
            tracing::error!(
                "asset hot-reload: off-thread EnvironmentMap convolution failed: {} \
                 (IBL kept its old payload)",
                msg
            );
            true
        }
        Err(std::sync::mpsc::TryRecvError::Empty) => false,
        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
            *slot = None;
            tracing::error!(
                "asset hot-reload: EnvironmentMap worker thread disconnected without \
                 sending a result (IBL kept its old payload)"
            );
            true
        }
    }
}
