//! Pushing a decoded asset batch through the backend, one asset kind at a
//! time. Each applier counts what reloaded and what the backend refused; a
//! refusal is logged and the asset keeps its old payload.

use std::collections::HashSet;

use concinnity_engine::components::build_skeleton_from_joint_defs;
use concinnity_engine::gfx::render_types::{self, TextureSlot};
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::backend::{
    DrawGeometryUpdate, GeometryRebuild, RenderBackend, SkinnedDrawGeometryUpdate, TextureUploads,
};
use concinnity_engine::render::world_load::source_maps::MeshSourceEntry;

use super::decode::apply_skinned_layouts_to_entries;
use super::state::{
    AssetHotReloadState, DecodedColorLut, DecodedMesh, DecodedSkinnedMesh, DecodedTexture,
    PendingSkeletonUpdate,
};

/// What one batch reloaded and what failed.
pub(super) struct Tally {
    pub(super) reloaded: usize,
    pub(super) failed: usize,
}

impl Tally {
    pub(super) fn new(decode_failures: usize) -> Self {
        Self {
            reloaded: 0,
            failed: decode_failures,
        }
    }

    // Count `count` assets as reloaded on success; on failure log `what` with
    // the error and count them as failed.
    fn record(&mut self, count: usize, result: RenderResult<()>, what: impl FnOnce() -> String) {
        match result {
            Ok(()) => self.reloaded += count,
            Err(e) => {
                tracing::error!("asset hot-reload: {}: {e}", what());
                self.failed += count;
            }
        }
    }
}

pub(super) fn textures(
    backend: &mut dyn TextureUploads,
    textures: &[DecodedTexture],
    tally: &mut Tally,
) {
    for tex in textures {
        let result = backend.update_texture_slot(TextureSlot::from_usize(tex.slot), &tex.image);
        tally.record(1, result, || {
            format!(
                "backend rejected slot {} update for '{}'",
                tex.slot, tex.source
            )
        });
    }
}

pub(super) fn color_lut(
    backend: &mut dyn TextureUploads,
    lut: &DecodedColorLut,
    tally: &mut Tally,
) {
    let result = backend.update_color_lut(lut.size, &lut.data);
    tally.record(1, result, || {
        format!("backend rejected ColorLut update for '{}'", lut.source)
    });
}

// Whether `mesh` no longer fits the slots `entry` draws into, base geometry
// or any LOD, so its draws need the shared buffers rebuilt.
fn mesh_size_changed(
    backend: &dyn GeometryRebuild,
    entry: &MeshSourceEntry,
    mesh: &DecodedMesh,
) -> bool {
    entry.draw_indices.iter().any(|&draw_idx| {
        let base_changed = backend
            .draw_geometry_size(draw_idx)
            .is_some_and(|(v, i)| v != mesh.vertices.len() || i != mesh.indices.len());
        let lod_changed = backend
            .draw_lod_index_counts(draw_idx)
            .is_some_and(|counts| {
                counts.len() != mesh.lod_alternates.len()
                    || counts
                        .iter()
                        .zip(mesh.lod_alternates.iter())
                        .any(|(c, (_, idx))| *c != idx.len())
            });
        base_changed || lod_changed
    })
}

// Static meshes: rewritten in place per draw slot while their sizes still
// fit; the entries that no longer fit are rebuilt together in one pass.
pub(super) fn static_meshes(
    backend: &mut dyn GeometryRebuild,
    entries: &[MeshSourceEntry],
    meshes: &[DecodedMesh],
    tally: &mut Tally,
) {
    let mut rebuild: Vec<DrawGeometryUpdate> = Vec::new();
    let mut rebuild_entries = 0usize;
    for mesh in meshes {
        let Some(entry) = entries.get(mesh.entry_idx) else {
            tracing::warn!(
                "asset hot-reload: decoded Mesh entry_idx {} out of range (source \
                 map mutated mid-pass?); skipping",
                mesh.entry_idx
            );
            tally.failed += 1;
            continue;
        };
        if mesh_size_changed(backend, entry, mesh) {
            rebuild_entries += 1;
            rebuild.extend(
                entry
                    .draw_indices
                    .iter()
                    .map(|&draw_idx| DrawGeometryUpdate {
                        draw_idx,
                        vertices: mesh.vertices.clone(),
                        indices: mesh.indices.clone(),
                        lod_alternates: mesh.lod_alternates.clone(),
                    }),
            );
            continue;
        }
        let mut slot_failures = 0usize;
        for &draw_idx in &entry.draw_indices {
            if let Err(e) = backend.update_mesh_geometry(
                draw_idx,
                &mesh.vertices,
                &mesh.indices,
                &mesh.lod_alternates,
            ) {
                tracing::error!(
                    "asset hot-reload: backend rejected mesh update for draw {} \
                     (source '{}', primitive {}): {}",
                    draw_idx,
                    entry.source,
                    entry.primitive_index,
                    e
                );
                slot_failures += 1;
            }
        }
        if slot_failures == 0 {
            tally.reloaded += 1;
        } else {
            tally.failed += 1;
        }
    }
    if rebuild.is_empty() {
        return;
    }
    tracing::info!(
        "asset hot-reload: {} Mesh entry/entries had size changes; rebuilding \
         shared static-geometry buffers ({} draw slot(s) affected)",
        rebuild_entries,
        rebuild.len()
    );
    let result = backend.rebuild_static_geometry(rebuild);
    tally.record(rebuild_entries, result, || {
        format!(
            "rebuild_static_geometry failed ({rebuild_entries} Mesh entry/entries kept \
             their old geometry)"
        )
    });
}

// What the skinned pass found to do beyond in-place rewrites.
#[derive(Default)]
struct SkinnedPlan {
    rebuild: Vec<SkinnedDrawGeometryUpdate>,
    rebuild_entries: Vec<usize>,
    // (entry index, new joint count), applied once the rebuild has settled.
    joint_counts: Vec<(usize, usize)>,
}

// Skinned meshes: the same in-place or rebuild split as static meshes, plus
// joint-count changes, which resize the slot's joint buffers and queue a
// `SkeletonPose` refresh. Vertex counts and joint counts change
// independently, so either or both may fire for one entry.
pub(super) fn skinned_meshes(
    state: &mut AssetHotReloadState,
    backend: &mut dyn RenderBackend,
    meshes: &[DecodedSkinnedMesh],
    tally: &mut Tally,
) {
    let mut plan = SkinnedPlan::default();
    for mesh in meshes {
        rewrite_skinned(state, backend, mesh, &mut plan, tally);
    }
    rebuild_skinned(state, backend, &mut plan, tally);
    resize_joints(state, backend, &plan.joint_counts, tally);
}

// Rewrite one decoded skinned mesh in place, or plan its rebuild and its
// joint-count change.
fn rewrite_skinned(
    state: &mut AssetHotReloadState,
    backend: &mut dyn RenderBackend,
    mesh: &DecodedSkinnedMesh,
    plan: &mut SkinnedPlan,
    tally: &mut Tally,
) {
    let Some(entry) = state.skinned_meshes.entries.get(mesh.entry_idx) else {
        tracing::warn!(
            "asset hot-reload: decoded SkinnedMesh entry_idx {} out of range; skipping",
            mesh.entry_idx
        );
        tally.failed += 1;
        return;
    };
    let new_joint_count = mesh.skeleton.len().min(render_types::MAX_JOINTS);
    if new_joint_count != entry.joint_count {
        plan.joint_counts.push((mesh.entry_idx, new_joint_count));
        state.pending_skeleton_updates.push(PendingSkeletonUpdate {
            skinned_index: entry.skinned_index,
            new_skeleton: build_skeleton_from_joint_defs(&mesh.skeleton),
        });
        tracing::info!(
            "asset hot-reload: SkinnedMesh '{}' joint count changed ({} → {}), \
             resizing joint buffers and refreshing SkeletonPose",
            entry.source,
            entry.joint_count,
            new_joint_count
        );
    }
    if mesh.vertices.len() != entry.vertex_count || mesh.indices.len() != entry.index_count {
        plan.rebuild_entries.push(mesh.entry_idx);
        plan.rebuild.push(SkinnedDrawGeometryUpdate {
            skinned_index: entry.skinned_index,
            vertices: mesh.vertices.clone(),
            indices: mesh.indices.clone(),
        });
        return;
    }
    let result = backend.update_skinned_mesh_geometry(
        entry.skinned_index,
        entry.vertex_base,
        &mesh.vertices,
        &mesh.indices,
    );
    tally.record(1, result, || {
        format!(
            "backend rejected SkinnedMesh update for skinned slot {} (source '{}')",
            entry.skinned_index, entry.source
        )
    });
}

// Rebuild the planned skinned entries in one pass. On success every entry's
// layout is refreshed, since the rebuild repacks from scratch; on failure the
// skeleton refreshes and joint-count changes for those slots are dropped, so
// nothing runs ahead of the geometry they kept.
fn rebuild_skinned(
    state: &mut AssetHotReloadState,
    backend: &mut dyn RenderBackend,
    plan: &mut SkinnedPlan,
    tally: &mut Tally,
) {
    if plan.rebuild.is_empty() {
        return;
    }
    tracing::info!(
        "asset hot-reload: {} SkinnedMesh entry/entries had size changes; \
         rebuilding shared skinned vertex / index buffers",
        plan.rebuild.len()
    );
    let count = plan.rebuild_entries.len();
    match backend.rebuild_skinned_geometry(std::mem::take(&mut plan.rebuild)) {
        Ok(layouts) => {
            apply_skinned_layouts_to_entries(&mut state.skinned_meshes.entries, &layouts);
            tally.reloaded += count;
        }
        Err(e) => {
            tracing::error!(
                "asset hot-reload: rebuild_skinned_geometry failed: {} \
                 ({} SkinnedMesh entry/entries kept their old geometry)",
                e,
                count
            );
            tally.failed += count;
            let entries = &state.skinned_meshes.entries;
            let failed: HashSet<render_types::SkinnedIndex> = plan
                .rebuild_entries
                .iter()
                .filter_map(|&i| entries.get(i))
                .map(|e| e.skinned_index)
                .collect();
            state
                .pending_skeleton_updates
                .retain(|u| !failed.contains(&u.skinned_index));
            plan.joint_counts.retain(|(entry_idx, _)| {
                entries
                    .get(*entry_idx)
                    .is_none_or(|e| !failed.contains(&e.skinned_index))
            });
        }
    }
}

// Resize each changed slot's joint buffers and record its new count. A slot
// the backend refuses keeps its old count, and its queued skeleton refresh is
// dropped so the pose stays in step with the buffers.
fn resize_joints(
    state: &mut AssetHotReloadState,
    backend: &mut dyn RenderBackend,
    joint_counts: &[(usize, usize)],
    tally: &mut Tally,
) {
    for &(entry_idx, new_count) in joint_counts {
        let Some(entry) = state.skinned_meshes.entries.get_mut(entry_idx) else {
            continue;
        };
        match backend.update_skinned_skeleton(entry.skinned_index, new_count) {
            Ok(()) => entry.joint_count = new_count,
            Err(e) => {
                tracing::error!(
                    "asset hot-reload: backend rejected joint-count update for skinned slot {} \
                     (source '{}'): {} (joint_count stays at {})",
                    entry.skinned_index,
                    entry.source,
                    e,
                    entry.joint_count
                );
                let stale = entry.skinned_index;
                state
                    .pending_skeleton_updates
                    .retain(|u| u.skinned_index != stale);
                tally.failed += 1;
            }
        }
    }
}
