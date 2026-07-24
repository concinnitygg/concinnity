// src/gfx/draw_list.rs
//
// Render-prep helpers that consume asset components and produce GPU-ready data.
// None of these functions hold or borrow a backend handle.

use crate::assets::{
    File, FileKind, InstancedProp, InstancedPropGeometry, ProceduralMesh, Room, SubMeshRef,
    VoxelChunk,
};
use crate::ecs::PipelineContext;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{MaterialHandle, MeshHandle, TextureHandle};
use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::{
    DrawObject, InstancedCluster, LodSlice, MaterialUniforms, NO_NORMAL_MAP_SLOT,
};

pub(crate) const IDENTITY4: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

// (albedo_slot, normal_map_slot, gpu material uniforms), passed through build_draw_list.
pub(crate) type MaterialEntry = (usize, usize, MaterialUniforms);

// Geometry decoded for one Room: the asset, its vertices, LOD0 indices, and
// LOD alternates (switch_distance, indices).
pub(crate) type RoomGeometry = (Room, Vec<Vertex>, Vec<u16>, Vec<(f32, Vec<u16>)>);

// Mesh-geometry lookup tables from `load_mesh_geometry`: the loaded geometry
// (dense, indexed by the unified mesh-source handle -- a `.mesh` reference's
// `MeshHandle` indexes it directly), file-backed Mesh source metadata keyed by
// handle (dev-only), the always-resident handle set, and the asset id ->
// handle map for the geometry producers that are still components
// (ProceduralMesh / VoxelChunk / File).
pub(crate) type MeshGeometryMaps = (
    Vec<LoadedMesh>,
    std::collections::HashMap<usize, MeshSourceMeta>,
    std::collections::HashSet<usize>,
    std::collections::HashMap<AssetId, usize>,
    std::collections::HashMap<usize, DeferredMeshSeed>,
);

// A deferred mesh's payload reference, captured while its decode was skipped:
// the locator, plus the raw bytes when the blob is RAM-backed (an in-memory
// world may release its payload sections after init, so the bytes are copied
// out now; a disk-backed world re-reads the blob file range instead).
pub(crate) struct DeferredMeshSeed {
    pub locator: crate::ecs::PayloadLocator,
    pub bytes: Option<Vec<u8>>,
}

// Mesh sources whose payload decode init defers: exclusively owned by a scene
// other than the start scene, with baked bounds from the blob. `bounds` and
// `counts` are keyed by mesh-source handle; a member with no baked record
// decodes eagerly.
#[derive(Default)]
pub(crate) struct DeferredMeshSources {
    pub by_handle: std::collections::HashSet<u32>,
    pub by_def: std::collections::HashSet<AssetId>,
    pub bounds: std::collections::HashMap<u32, ([f32; 3], [f32; 3])>,
    pub counts: std::collections::HashMap<u32, (u32, u32)>,
}

impl DeferredMeshSources {
    // Baked bounds for a deferred resource-stream Mesh, or None to decode.
    fn resource_bounds(&self, handle: usize) -> Option<([f32; 3], [f32; 3])> {
        if !self.by_handle.contains(&(handle as u32)) {
            return None;
        }
        self.bounds.get(&(handle as u32)).copied()
    }

    // Baked bounds for a deferred mesh-source component at its push position,
    // or None to decode.
    fn def_bounds(&self, id: AssetId, handle: usize) -> Option<([f32; 3], [f32; 3])> {
        if !self.by_def.contains(&id) {
            return None;
        }
        self.bounds.get(&(handle as u32)).copied()
    }
}

// Output of `build_draw_list`. `prop_draw_indices` and `prop_local_bounds`
// are column-aligned with the input `items`; `mesh_handle_to_draws` backs
// hot-reload. A prop whose meshes carry no vertices gets the non-finite
// UNCULLED_BB as its local bounds (unpickable, uncullable).
pub(crate) struct DrawListData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub draw_objects: Vec<DrawObject>,
    pub instanced_clusters: Vec<InstancedCluster>,
    pub prop_draw_indices: Vec<Vec<usize>>,
    pub mesh_handle_to_draws: std::collections::HashMap<usize, Vec<usize>>,
    pub prop_local_bounds: Vec<([f32; 3], [f32; 3])>,
}

// One appended mesh's placement in the shared buffers: vertex_offset,
// vertex_count, index_offset, index_count, LOD slices, and local AABB min/max.
type AppendedMesh = (
    usize,
    usize,
    usize,
    usize,
    Vec<LodSlice>,
    [f32; 3],
    [f32; 3],
);

// Sentinel AABB used when a draw object opts out of culling (e.g. unbounded
// skybox geometry). Both metal and vulkan/directx backends should treat any
// non-finite component as "always draw".
const UNCULLED_BB: ([f32; 3], [f32; 3]) = (
    [f32::NAN, f32::NAN, f32::NAN],
    [f32::NAN, f32::NAN, f32::NAN],
);

fn local_bounds(verts: &[Vertex]) -> ([f32; 3], [f32; 3]) {
    if verts.is_empty() {
        return UNCULLED_BB;
    }
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for v in verts {
        for i in 0..3 {
            mn[i] = mn[i].min(v.pos[i]);
            mx[i] = mx[i].max(v.pos[i]);
        }
    }
    (mn, mx)
}

// The renderer-relevant view of one placement that build_draw_list consumes:
// the mesh/model/material/texture refs, the cull distance, whether it is dynamic
// (skips frustum culling), and the asset id (error logging only). Built from an
// entity's MeshRenderer/ModelRenderer + tag components by
// `decomposed_renderable_item`.
//
// An entity is dynamic (pulled out of the BVH and always drawn after a per-object
// frustum test) when it carries a Pickup, Interactable, Parent, or Collider tag.
// The BVH is built once at init and does not refit, so a moving entity would
// otherwise risk being culled against its stale init-time AABB.
#[derive(Debug, PartialEq)]
pub(crate) struct RenderableItem {
    pub asset_id: AssetId,
    pub model: Option<AssetId>,
    pub mesh: Option<MeshHandle>,
    pub material: Option<MaterialHandle>,
    pub texture: Option<TextureHandle>,
    pub cull_distance: f32,
    pub is_dynamic: bool,
}

// Build one entity's RenderableItem: read its renderer fields from its
// MeshRenderer xor ModelRenderer and its dynamic flag from the Pickup /
// Interactable / Parent / Collider tags. asset_id is for error logging only
// (resolved from the name index by the caller).
pub(crate) fn decomposed_renderable_item(
    ctx: &crate::ecs::PipelineContext,
    entity: crate::ecs::Entity,
    asset_id: AssetId,
) -> RenderableItem {
    use crate::assets::{Collider, Interactable, MeshRenderer, ModelRenderer, Parent, Pickup};

    let (model, mesh, material, texture, cull_distance) =
        if let Some(m) = ctx.get::<ModelRenderer>(entity) {
            (Some(m.model), None, None, None, m.cull_distance)
        } else if let Some(m) = ctx.get::<MeshRenderer>(entity) {
            (None, m.mesh, m.material, m.texture, m.cull_distance)
        } else {
            (None, None, None, None, 0.0)
        };
    let is_dynamic = ctx.get::<Pickup>(entity).is_some()
        || ctx.get::<Interactable>(entity).is_some()
        || ctx.get::<Parent>(entity).is_some()
        || ctx.get::<Collider>(entity).is_some();
    RenderableItem {
        asset_id,
        model,
        mesh,
        material,
        texture,
        cull_distance,
        is_dynamic,
    }
}

// Column-major 4×4 matrix multiply: result = a * b; layout m[col][row].
fn mat_mul4(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            for k in 0..4 {
                out[col][row] += a[k][row] * b[col][k];
            }
        }
    }
    out
}

// Resolve each entity's world matrix from its Transform and Parent chain: roots
// use their local matrix, children compose parent-world * local, and cyclic
// parents fall back to their local matrix. Returns an entity -> world matrix map.
// Shared by the per-frame propagate_transforms and the render-init draw-list
// build.
pub(crate) fn resolve_world_matrices(
    ctx: &crate::ecs::PipelineContext,
) -> std::collections::HashMap<crate::ecs::Entity, [[f32; 4]; 4]> {
    use crate::assets::{Parent, Transform};
    use crate::ecs::Entity;
    use std::collections::HashMap;

    let parents: HashMap<Entity, Entity> = ctx
        .query_with_entity::<Parent>()
        .map(|(entity, parent)| (entity, parent.0))
        .collect();
    let locals: Vec<(Entity, [[f32; 4]; 4])> = ctx
        .query_with_entity::<Transform>()
        .map(|(entity, transform)| (entity, transform.model_matrix()))
        .collect();

    // Fixed-point resolution: keep a pass running while any entity newly
    // resolves; stop on a pass with no progress (a cycle) or once all are done.
    let mut world: HashMap<Entity, [[f32; 4]; 4]> = HashMap::with_capacity(locals.len());
    loop {
        let mut progressed = false;
        for (entity, local) in &locals {
            if world.contains_key(entity) {
                continue;
            }
            let resolved = match parents.get(entity) {
                None => Some(*local),
                Some(parent) => world.get(parent).map(|pw| mat_mul4(*pw, *local)),
            };
            if let Some(matrix) = resolved {
                world.insert(*entity, matrix);
                progressed = true;
            }
        }
        if !progressed || world.len() == locals.len() {
            break;
        }
    }
    // Cyclic entities fall back to their local matrix.
    for (entity, local) in &locals {
        world.entry(*entity).or_insert(*local);
    }
    world
}

pub(crate) fn propagate_transforms(ctx: &mut crate::ecs::PipelineContext) {
    use crate::assets::GlobalTransform;

    let world = resolve_world_matrices(ctx);
    for (entity, matrix) in world {
        if let Some(global) = ctx.get_mut::<GlobalTransform>(entity) {
            global.0 = matrix;
        }
    }
}

// Reused scratch plus change-tracking for the per-frame transform propagation.
// GraphicsSystem owns one and passes it to `propagate_transforms_cached` each
// frame: the three working buffers are cleared and refilled in place (no
// per-frame allocation once they reach steady-state capacity), and propagation
// is skipped entirely on frames where neither the Transform nor the Parent
// column changed since the last recompute -- so a static scene recomputes and
// re-uploads nothing.
#[derive(Default)]
pub(crate) struct TransformCache {
    parents: std::collections::HashMap<crate::ecs::Entity, crate::ecs::Entity>,
    locals: Vec<(crate::ecs::Entity, [[f32; 4]; 4])>,
    world: std::collections::HashMap<crate::ecs::Entity, [[f32; 4]; 4]>,
    // (Transform column tick, Parent column tick) observed at the last recompute.
    // `None` until the first propagation, which always runs.
    last_ticks: Option<(crate::ecs::Tick, crate::ecs::Tick)>,
}

impl TransformCache {
    // Recompute world matrices into the reused buffers from the live Transform +
    // Parent columns. Same fixed-point resolution as `resolve_world_matrices`,
    // but writing into `self`'s retained-capacity buffers instead of fresh ones.
    fn resolve(&mut self, ctx: &crate::ecs::PipelineContext) {
        use crate::assets::{Parent, Transform};

        self.parents.clear();
        self.locals.clear();
        self.world.clear();
        for (entity, parent) in ctx.query_with_entity::<Parent>() {
            self.parents.insert(entity, parent.0);
        }
        for (entity, transform) in ctx.query_with_entity::<Transform>() {
            self.locals.push((entity, transform.model_matrix()));
        }
        loop {
            let mut progressed = false;
            for (entity, local) in &self.locals {
                if self.world.contains_key(entity) {
                    continue;
                }
                let resolved = match self.parents.get(entity) {
                    None => Some(*local),
                    Some(parent) => self.world.get(parent).map(|pw| mat_mul4(*pw, *local)),
                };
                if let Some(matrix) = resolved {
                    self.world.insert(*entity, matrix);
                    progressed = true;
                }
            }
            if !progressed || self.world.len() == self.locals.len() {
                break;
            }
        }
        for (entity, local) in &self.locals {
            self.world.entry(*entity).or_insert(*local);
        }
    }
}

// Per-frame transform propagation with the reused scratch + change-tracking in
// `cache`. Writes each entity's GlobalTransform from its Transform + Parent
// chain, exactly as `propagate_transforms`, but skips the whole pass when
// neither source column changed since the last recompute (the GlobalTransforms
// written then still stand). Used by GraphicsSystem's per-frame step; the
// uncached `propagate_transforms` remains for the one-shot reparent recompose.
pub(crate) fn propagate_transforms_cached(
    ctx: &mut crate::ecs::PipelineContext,
    cache: &mut TransformCache,
) {
    use crate::assets::{Parent, Transform};

    let ticks = (
        ctx.changed_tick::<Transform>(),
        ctx.changed_tick::<Parent>(),
    );
    if cache.last_ticks == Some(ticks) {
        return;
    }
    cache.resolve(ctx);
    for (entity, matrix) in &cache.world {
        if let Some(global) = ctx.get_mut::<crate::assets::GlobalTransform>(*entity) {
            global.0 = *matrix;
        }
    }
    cache.last_ticks = Some(ticks);
}

// Re-parent an entity at runtime: detach it from its current parent (if any),
// attach it under `new_parent` (or leave it a root when `None`), keep both
// parents' Children lists in sync, and recompose world matrices so the new
// chain shows up immediately. Entity-keyed throughout, so it is invariant to
// component-column order. Driven by ReparentRequest events the GraphicsSystem
// drains each step.
pub(crate) fn reparent(
    ctx: &mut crate::ecs::PipelineContext,
    child: crate::ecs::Entity,
    new_parent: Option<crate::ecs::Entity>,
) {
    use crate::assets::{Children, Parent};

    // Drop the old parent edge and unlist the child from that parent.
    if let Some(old) = ctx.remove::<Parent>(child)
        && let Some(siblings) = ctx.get_mut::<Children>(old.0)
    {
        siblings.0.retain(|&e| e != child);
    }

    // Attach under the new parent (None leaves it a root). The Parent column is
    // free of `child` here (just removed), so the insert never duplicates.
    if let Some(parent) = new_parent {
        ctx.insert(child, Parent(parent));
        match ctx.get_mut::<Children>(parent) {
            Some(kids) => {
                if !kids.0.contains(&child) {
                    kids.0.push(child);
                }
            }
            None => ctx.insert(parent, Children(vec![child])),
        }
    }

    propagate_transforms(ctx);
}

// Decoded mesh geometry plus its optional LOD trailer. Returned by
// `load_mesh_geometry` and consumed by `build_draw_list`. The `vertices`
// slice is shared across LOD0 and every alternate; vertex-clustering
// decimation reuses the original vertex set and only generates new index
// lists. Empty `lod_alternates` means the mesh declared `lod_levels <= 1`
// (or the build dropped degenerate decimations); the runtime then keeps
// the single LOD0 slice.
pub(crate) struct LoadedMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
    pub lod_alternates: Vec<(f32, Vec<u16>)>,
    // Baked local AABB for a deferred mesh whose vertices were not decoded;
    // None computes bounds from the vertices.
    pub bounds: Option<([f32; 3], [f32; 3])>,
    // Baked (vertex, index) counts for a deferred mesh, so its draw record
    // matches the geometry the streamer uploads later; None uses the decoded
    // lengths.
    pub counts: Option<(u32, u32)>,
}

// Hot-reload source metadata for a file-backed `Mesh`. Captured by
// `load_mesh_geometry` before the Mesh is drained and consumed; the
// `(asset_id, source, primitive_index, lod_levels, lod_distances)` tuple is
// later cross-referenced against `build_draw_list`'s mesh_id → draw_indices
// map to build the runtime
// [`MeshSourceMap`](crate::gfx::graphics_system::hot_reload_sources::MeshSourceMap).
pub(crate) struct MeshSourceMeta {
    pub source: String,
    pub primitive_index: u32,
    pub lod_levels: u32,
    pub lod_distances: Vec<f32>,
}

// Decode all mesh-source payloads into a dense, handle-indexed geometry table.
// The Mesh block comes from the blob's resource stream (`MeshTable`); the
// remaining geometry producers (ProceduralMesh, VoxelChunk, mesh-kind File) are
// still components and are decoded after it, in the same fixed block order cook
// assigned handles, so `geometry[h]` is the source cook gave handle `h`.
// Returns None if any payload is missing or malformed. Also returns a
// handle-keyed source-meta map for file-backed Mesh declarations under
// `cn debug` (from the dev `MeshSources` catalogue), the set of handles whose
// props must always stay resident (skybox-class geometry that encloses the
// camera), and the asset id -> handle map for the still-component producers.
pub(crate) fn load_mesh_geometry(
    ctx: &mut PipelineContext,
    deferred: &DeferredMeshSources,
    blob_disk_backed: bool,
) -> Option<MeshGeometryMaps> {
    let mut deferred_payloads: std::collections::HashMap<usize, DeferredMeshSeed> =
        std::collections::HashMap::new();
    let mesh_table = ctx
        .resource::<crate::resource::MeshTable>()
        .cloned()
        .unwrap_or_default();
    // Dev-only source catalogue (present under `cn debug`) so the hot-reload
    // watcher can map a mesh handle back to the file that backs it. Mesh is a
    // resource now, so there is no drained component `source` to capture.
    let capture_sources = crate::app::dev_flags::enabled();
    let mut mesh_sources: std::collections::HashMap<usize, MeshSourceMeta> =
        std::collections::HashMap::new();
    if capture_sources && let Some(sources) = ctx.resource::<crate::resource::MeshSources>() {
        for (handle, info) in sources.0.iter().enumerate() {
            if !info.source.is_empty() {
                mesh_sources.insert(
                    handle,
                    MeshSourceMeta {
                        source: info.source.clone(),
                        primitive_index: info.primitive_index,
                        lod_levels: info.lod_levels,
                        lod_distances: info.lod_distances.clone(),
                    },
                );
            }
        }
    }
    // ProceduralMesh components are cloned rather than drained: PhysicsSystem
    // inits after GraphicsSystem and resolves its `terrain_mesh` reference by
    // querying ProceduralMesh for the live heightfield args. Same precedent as
    // the audio-clip residency the graphics init leaves resident for AudioSystem:
    // leave the component in place so a later init step can still read it.
    let proc_meshes: Vec<ProceduralMesh> = ctx.query::<ProceduralMesh>().cloned().collect();
    let voxel_chunks = ctx.drain::<VoxelChunk>();
    let file_assets = ctx.drain::<File>();
    let file_meshes: Vec<&File> = file_assets
        .iter()
        .filter(|f| f.kind.as_ref().map(FileKind::is_mesh).unwrap_or(false))
        .collect();

    if mesh_table.is_empty()
        && proc_meshes.is_empty()
        && voxel_chunks.is_empty()
        && file_meshes.is_empty()
    {
        // Room path can carry the scene without any explicit Mesh/ProceduralMesh
        tracing::info!(
            "GraphicsSystem: no Mesh, ProceduralMesh, VoxelChunk, or mesh-kind File sources found"
        );
    }

    let mut geometry: Vec<LoadedMesh> = Vec::new();
    // Asset id -> handle for the geometry producers that are still components,
    // so init can cross-reference their id-keyed metadata (e.g. the
    // ProceduralMesh args snapshot) with the handle-keyed draw map.
    let mut component_mesh_handles: std::collections::HashMap<AssetId, usize> =
        std::collections::HashMap::new();

    // Mesh block first: decode each resource-table entry at its handle position.
    for (handle, entry) in mesh_table.0.iter().enumerate() {
        let locator = match &entry.payload {
            Some(l) => l,
            None => {
                tracing::error!(
                    "GraphicsSystem: Mesh handle {} has no compiled payload -- did the build succeed?",
                    handle
                );
                return None;
            }
        };
        if let Some(bounds) = deferred.resource_bounds(handle) {
            let bytes = if blob_disk_backed {
                None
            } else {
                match ctx.read_payload(locator) {
                    Ok(b) => Some(b.to_vec()),
                    Err(e) => {
                        tracing::error!("GraphicsSystem: failed to read Mesh payload: {:?}", e);
                        return None;
                    }
                }
            };
            deferred_payloads.insert(
                handle,
                DeferredMeshSeed {
                    locator: locator.clone(),
                    bytes,
                },
            );
            geometry.push(LoadedMesh {
                vertices: Vec::new(),
                indices: Vec::new(),
                lod_alternates: Vec::new(),
                bounds: Some(bounds),
                counts: deferred.counts.get(&(handle as u32)).copied(),
            });
            continue;
        }
        let bytes = match ctx.read_payload(locator) {
            Ok(b) => b.to_vec(),
            Err(e) => {
                tracing::error!("GraphicsSystem: failed to read Mesh payload: {:?}", e);
                return None;
            }
        };
        // `deserialise_with_lods` parses the optional LOD trailer when the
        // build emitted one and falls back to an empty alternates vec for
        // legacy single-LOD payloads.
        match crate::gfx::mesh_payload::deserialise_with_lods(&bytes) {
            Ok((verts, idxs, alternates)) => geometry.push(LoadedMesh {
                vertices: verts,
                indices: idxs,
                lod_alternates: alternates,
                bounds: None,
                counts: None,
            }),
            Err(e) => {
                tracing::error!("GraphicsSystem: malformed Mesh payload: {}", e);
                return None;
            }
        }
    }

    macro_rules! load_meshes {
        ($label:expr_2021, $items:expr_2021) => {
            for (i, mesh) in $items.iter().enumerate() {
                let locator = match &mesh.locator {
                    Some(l) => l,
                    None => {
                        tracing::error!(
                            "GraphicsSystem: {}[{}] {} has no compiled payload",
                            $label,
                            i,
                            mesh.asset_id
                        );
                        return None;
                    }
                };
                if let Some(bounds) = deferred.def_bounds(mesh.asset_id, geometry.len()) {
                    let bytes = if blob_disk_backed {
                        None
                    } else {
                        match ctx.read_payload(locator) {
                            Ok(b) => Some(b.to_vec()),
                            Err(e) => {
                                tracing::error!(
                                    "GraphicsSystem: failed to read {} payload: {:?}",
                                    $label,
                                    e
                                );
                                return None;
                            }
                        }
                    };
                    deferred_payloads.insert(
                        geometry.len(),
                        DeferredMeshSeed {
                            locator: locator.clone(),
                            bytes,
                        },
                    );
                    component_mesh_handles.insert(mesh.asset_id, geometry.len());
                    let counts = deferred.counts.get(&(geometry.len() as u32)).copied();
                    geometry.push(LoadedMesh {
                        vertices: Vec::new(),
                        indices: Vec::new(),
                        lod_alternates: Vec::new(),
                        bounds: Some(bounds),
                        counts,
                    });
                    continue;
                }
                let bytes = match ctx.read_payload(locator) {
                    Ok(b) => b.to_vec(),
                    Err(e) => {
                        tracing::error!(
                            "GraphicsSystem: failed to read {} payload: {:?}",
                            $label,
                            e
                        );
                        return None;
                    }
                };
                match crate::gfx::mesh_payload::deserialise_with_lods(&bytes) {
                    Ok((verts, idxs, alternates)) => {
                        // This source's handle is its push position: the blocks
                        // are loaded in cook's block order and each iterates in
                        // declaration order.
                        component_mesh_handles.insert(mesh.asset_id, geometry.len());
                        geometry.push(LoadedMesh {
                            vertices: verts,
                            indices: idxs,
                            lod_alternates: alternates,
                            bounds: None,
                            counts: None,
                        });
                    }
                    Err(e) => {
                        tracing::error!("GraphicsSystem: malformed {} payload: {}", $label, e);
                        return None;
                    }
                }
            }
        };
    }
    load_meshes!("ProceduralMesh", proc_meshes);
    load_meshes!("VoxelChunk", voxel_chunks);
    load_meshes!("File", file_meshes);

    // Skybox-generated meshes enclose the camera, so any prop using one must
    // opt out of frustum culling AND streaming residency (per the
    // StreamingConfig docstring's "skybox always stays resident" promise).
    let always_resident_meshes: std::collections::HashSet<usize> = proc_meshes
        .iter()
        .filter(|pm| pm.generator == "skybox")
        .filter_map(|pm| component_mesh_handles.get(&pm.asset_id).copied())
        .collect();

    Some((
        geometry,
        mesh_sources,
        always_resident_meshes,
        component_mesh_handles,
        deferred_payloads,
    ))
}

// Decode all Room mesh payloads and collect blob indices for the release step.
// Returns None if any payload is missing or malformed (error already logged).
pub(crate) fn load_room_geometry(
    ctx: &mut PipelineContext,
) -> Option<(Vec<RoomGeometry>, Vec<u32>)> {
    let rooms = ctx.drain::<Room>();
    let mut room_geometry: Vec<RoomGeometry> = Vec::new();
    let mut blob_indices: Vec<u32> = Vec::new();

    for (i, room) in rooms.into_iter().enumerate() {
        let locator = match &room.locator {
            Some(l) => l.clone(),
            None => {
                tracing::error!(
                    "GraphicsSystem: Room[{}] {} has no compiled payload -- did the build succeed?",
                    i,
                    room.asset_id
                );
                return None;
            }
        };
        blob_indices.push(locator.blob_index);
        let bytes = match ctx.read_payload(&locator) {
            Ok(b) => b.to_vec(),
            Err(e) => {
                tracing::error!(
                    "GraphicsSystem: failed to read Room {} payload: {:?}",
                    room.asset_id,
                    e
                );
                return None;
            }
        };
        match crate::gfx::mesh_payload::deserialise_with_lods(&bytes) {
            Ok((verts, idxs, alternates)) => room_geometry.push((room, verts, idxs, alternates)),
            Err(e) => {
                tracing::error!("GraphicsSystem: malformed Room payload: {}", e);
                return None;
            }
        }
    }

    Some((room_geometry, blob_indices))
}

// Assemble the shared vertex/index buffers and per-object draw records from all
// scene geometry (props, unreferenced meshes, rooms). Also returns the per-prop
// draw-index table for runtime model-matrix updates and the GPU-instanced
// cluster list (one entry per InstancedProp).
// Returns None if any referenced asset is missing (error already logged).
// The read-only scene lookup tables consumed by [`build_draw_list`]: the
// renderable items and instanced props plus every catalogue needed to resolve
// their geometry, textures, and materials.
pub(crate) struct DrawListInputs<'a> {
    pub items: &'a [RenderableItem],
    pub instanced_props: &'a [InstancedProp],
    pub world_mats: &'a [[[f32; 4]; 4]],
    pub model_map: &'a std::collections::HashMap<AssetId, Vec<SubMeshRef>>,
    // Dense mesh-source geometry from `load_mesh_geometry`: a `.mesh`
    // reference's `MeshHandle` indexes it directly.
    pub mesh_geometry: &'a [LoadedMesh],
    pub room_geometry: &'a [RoomGeometry],
    // Size of the shared texture pool; a texture handle is in range when its
    // index is below this. A legacy texture-on-mesh reference past it falls back
    // to slot 0.
    pub texture_count: usize,
    pub material_map: &'a std::collections::HashMap<MaterialHandle, MaterialEntry>,
    pub always_resident_meshes: &'a std::collections::HashSet<usize>,
}

// Resolve the (albedo_slot, normal_map_slot, material) a draw object binds. A
// material handle wins and must resolve in `material_map`; an unresolved one
// comes back as `Err(handle)` so the caller can log its own context. With no
// material, a texture handle contributes its albedo slot (clamped to slot 0 when
// past `texture_count`) over the default material; with neither, slot 0 and the
// default material.
pub(crate) fn resolve_material_slots(
    material: Option<MaterialHandle>,
    texture: Option<TextureHandle>,
    material_map: &std::collections::HashMap<MaterialHandle, MaterialEntry>,
    texture_count: usize,
) -> Result<MaterialEntry, MaterialHandle> {
    if let Some(mat_id) = material {
        return material_map.get(&mat_id).copied().ok_or(mat_id);
    }
    if let Some(tex_id) = texture {
        let slot = tex_id.index();
        let slot = if slot < texture_count { slot } else { 0 };
        return Ok((slot, NO_NORMAL_MAP_SLOT, MaterialUniforms::DEFAULT));
    }
    Ok((0, NO_NORMAL_MAP_SLOT, MaterialUniforms::DEFAULT))
}

pub(crate) fn build_draw_list(inputs: DrawListInputs) -> Option<DrawListData> {
    let DrawListInputs {
        items,
        instanced_props,
        world_mats,
        model_map,
        mesh_geometry,
        room_geometry,
        texture_count,
        material_map,
        always_resident_meshes,
    } = inputs;
    let mut all_vertices: Vec<Vertex> = Vec::new();
    let mut all_indices: Vec<u32> = Vec::new();
    let mut draw_objects: Vec<DrawObject> = Vec::new();
    let mut instanced_clusters: Vec<InstancedCluster> = Vec::new();
    let mut prop_draw_indices: Vec<Vec<usize>> = Vec::new();
    let mut prop_local_bounds: Vec<([f32; 3], [f32; 3])> = Vec::new();
    // Map every mesh-source handle to the draw slots that received a copy of
    // its geometry. Hot-reload (`cn debug` only) walks this to know which slots
    // to overwrite when the source `.glb` changes. The `Vec<usize>` accumulates
    // every push since a mesh shared by N `Prop`s yields N independent draw
    // objects.
    let mut mesh_handle_to_draws: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();

    // track explicitly referenced mesh handles so unreferenced ones get auto-rendered
    let mut referenced: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for item in items {
        if let Some(mesh) = item.mesh {
            referenced.insert(mesh.index());
        }
        if let Some(model_id) = item.model
            && let Some(submeshes) = model_map.get(&model_id)
        {
            for sub in submeshes {
                if let Some(sub_mesh) = sub.mesh {
                    referenced.insert(sub_mesh.index());
                }
            }
        }
    }
    for inst in instanced_props {
        if let Some(mesh) = inst.mesh {
            referenced.insert(mesh.index());
        }
    }

    // append_mesh: add a mesh into the shared buffers by handle, return
    // (vertex_offset, vertex_count, index_offset, index_count, lod_slices,
    // local_bb_min, local_bb_max). `lod_slices` is empty for legacy
    // single-LOD meshes; otherwise each entry is a `LodSlice` pointing at the
    // alternate's rebased indices in `all_indices`, paired with its switch
    // distance. Every LOD alternate reuses the same `vertex_offset` /
    // `vertex_count` since clustering decimation does not modify the vertex
    // set.
    let mut append_mesh = |handle: usize| -> Option<AppendedMesh> {
        let loaded = mesh_geometry.get(handle)?;
        let vertex_byte_offset = all_vertices.len() * std::mem::size_of::<Vertex>();
        let index_elem_offset = all_indices.len();
        let base = all_vertices.len() as u32;
        let (bb_min, bb_max) = loaded
            .bounds
            .unwrap_or_else(|| local_bounds(&loaded.vertices));
        all_vertices.extend_from_slice(&loaded.vertices);
        all_indices.extend(loaded.indices.iter().map(|i| u32::from(*i) + base));
        let mut lod_slices: Vec<LodSlice> = Vec::with_capacity(loaded.lod_alternates.len());
        for (switch_distance, alt_idx) in &loaded.lod_alternates {
            let alt_offset = all_indices.len();
            all_indices.extend(alt_idx.iter().map(|i| u32::from(*i) + base));
            lod_slices.push(LodSlice {
                index_offset: alt_offset,
                index_count: alt_idx.len(),
                switch_distance: *switch_distance,
            });
        }
        // A deferred mesh appended no bytes; its draw record carries the baked
        // counts so the streamed upload's size check matches the real geometry.
        let (vertex_count, index_count) = loaded
            .counts
            .map(|(v, i)| (v as usize, i as usize))
            .unwrap_or((loaded.vertices.len(), loaded.indices.len()));
        Some((
            vertex_byte_offset,
            vertex_count,
            index_elem_offset,
            index_count,
            lod_slices,
            bb_min,
            bb_max,
        ))
    };

    for (item_idx, item) in items.iter().enumerate() {
        let model_mat = world_mats[item_idx];
        let mut prop_idxs: Vec<usize> = Vec::new();
        // Union of this prop's sub-mesh local bounds (all in the same model
        // space). NaN sentinels from empty meshes fall out of min/max.
        let mut prop_min = [f32::INFINITY; 3];
        let mut prop_max = [f32::NEG_INFINITY; 3];
        let mut union_local = |mn: [f32; 3], mx: [f32; 3]| {
            for i in 0..3 {
                prop_min[i] = prop_min[i].min(mn[i]);
                prop_max[i] = prop_max[i].max(mx[i]);
            }
        };

        if let Some(model_id) = item.model {
            // multi-mesh model path: one draw object per sub-mesh
            let submeshes = match model_map.get(&model_id) {
                Some(s) => s,
                None => {
                    tracing::error!(
                        "GraphicsSystem: Prop {} references unknown model {} -- add a Model asset with that id",
                        item.asset_id,
                        model_id
                    );
                    return None;
                }
            };
            for sub in submeshes {
                let sub_mesh = match sub.mesh {
                    Some(m) => m.index(),
                    None => {
                        tracing::error!(
                            "GraphicsSystem: Model {} has a sub-mesh with no mesh",
                            model_id
                        );
                        return None;
                    }
                };
                let (
                    vertex_offset,
                    vertex_count,
                    index_offset,
                    index_count,
                    lod_alternates,
                    local_min,
                    local_max,
                ) = match append_mesh(sub_mesh) {
                    Some(t) => t,
                    None => {
                        tracing::error!(
                            "GraphicsSystem: Model {} sub-mesh handle {} out of range -- add a Mesh or ProceduralMesh asset with that name",
                            model_id,
                            sub_mesh
                        );
                        return None;
                    }
                };
                let (texture_slot, normal_map_slot, material) =
                    match resolve_material_slots(sub.material, None, material_map, texture_count) {
                        Ok(entry) => entry,
                        Err(mat_id) => {
                            tracing::error!(
                                "GraphicsSystem: Model {} sub-mesh material {} not found",
                                model_id,
                                mat_id.index()
                            );
                            return None;
                        }
                    };
                let (bb_min, bb_max) =
                    if item.is_dynamic || always_resident_meshes.contains(&sub_mesh) {
                        UNCULLED_BB
                    } else {
                        crate::gfx::frustum::transform_aabb(local_min, local_max, model_mat)
                    };
                union_local(local_min, local_max);
                prop_idxs.push(draw_objects.len());
                mesh_handle_to_draws
                    .entry(sub_mesh)
                    .or_default()
                    .push(draw_objects.len());
                draw_objects.push(DrawObject {
                    vertex_offset,
                    vertex_count,
                    index_offset,
                    index_count,
                    // Static geometry: indices are absolute into the shared
                    // vertex buffer, so no per-draw base.
                    base_vertex: 0,
                    model: model_mat,
                    texture_slot,
                    normal_map_slot,
                    material,
                    visible: true,
                    resident: true,
                    bb_min,
                    bb_max,
                    cull_distance: item.cull_distance,
                    lod_alternates,
                });
            }
        } else {
            // single-mesh path
            let mesh_handle = match item.mesh {
                Some(m) => m.index(),
                None => {
                    tracing::error!(
                        "GraphicsSystem: Prop {} has neither a model nor a mesh",
                        item.asset_id
                    );
                    return None;
                }
            };
            let (
                vertex_offset,
                vertex_count,
                index_offset,
                index_count,
                lod_alternates,
                local_min,
                local_max,
            ) = match append_mesh(mesh_handle) {
                Some(t) => t,
                None => {
                    tracing::error!(
                        "GraphicsSystem: Prop {} references out-of-range mesh handle {} -- add a Mesh or ProceduralMesh asset with that name",
                        item.asset_id,
                        mesh_handle
                    );
                    return None;
                }
            };
            // The texture handle is the texture's declaration-order pool slot;
            // an out-of-range handle falls back to slot 0.
            let (texture_slot, normal_map_slot, material) = match resolve_material_slots(
                item.material,
                item.texture,
                material_map,
                texture_count,
            ) {
                Ok(entry) => entry,
                Err(mat_id) => {
                    tracing::error!(
                        "GraphicsSystem: Prop {} references unknown material {} -- add a Material asset with that id",
                        item.asset_id,
                        mat_id.index()
                    );
                    return None;
                }
            };
            let (bb_min, bb_max) =
                if item.is_dynamic || always_resident_meshes.contains(&mesh_handle) {
                    UNCULLED_BB
                } else {
                    crate::gfx::frustum::transform_aabb(local_min, local_max, model_mat)
                };
            union_local(local_min, local_max);
            prop_idxs.push(draw_objects.len());
            mesh_handle_to_draws
                .entry(mesh_handle)
                .or_default()
                .push(draw_objects.len());
            draw_objects.push(DrawObject {
                vertex_offset,
                vertex_count,
                index_offset,
                index_count,
                base_vertex: 0,
                model: model_mat,
                texture_slot,
                normal_map_slot,
                material,
                visible: true,
                resident: true,
                bb_min,
                bb_max,
                cull_distance: item.cull_distance,
                lod_alternates,
            });
        }

        prop_draw_indices.push(prop_idxs);
        let finite = prop_min
            .iter()
            .chain(prop_max.iter())
            .all(|v| v.is_finite());
        prop_local_bounds.push(if finite {
            (prop_min, prop_max)
        } else {
            UNCULLED_BB
        });
    }

    // InstancedProp -> one GPU-instanced cluster per InstancedProp.
    // The cluster mesh is appended once; per-instance model matrices are
    // resolved up front and uploaded to the GPU each frame. The cluster's
    // union AABB is used as a single frustum-cull test for the whole batch.
    for inst in instanced_props {
        let mesh_handle = match inst.mesh {
            Some(m) if !inst.instances.is_empty() => m.index(),
            _ => continue,
        };
        // Instanced clusters carry the mesh's LOD alternates and bucket
        // their per-instance matrices by camera distance at draw time;
        // see [`InstancedCluster::lod_buckets`].
        let (
            vertex_offset,
            vertex_count,
            index_offset,
            index_count,
            lod_alternates,
            local_min,
            local_max,
        ) = match append_mesh(mesh_handle) {
            Some(t) => t,
            None => {
                tracing::error!(
                    "GraphicsSystem: InstancedProp {} references out-of-range mesh handle {}",
                    inst.asset_id,
                    mesh_handle
                );
                return None;
            }
        };
        let (texture_slot, normal_map_slot, material) = match resolve_material_slots(
            inst.material,
            inst.texture,
            material_map,
            texture_count,
        ) {
            Ok(entry) => entry,
            Err(mat_id) => {
                tracing::error!(
                    "GraphicsSystem: InstancedProp {} references unknown material {}",
                    inst.asset_id,
                    mat_id.index()
                );
                return None;
            }
        };

        let mut instance_mats: Vec<[[f32; 4]; 4]> = Vec::with_capacity(inst.instances.len());
        let mut cluster_min = [f32::INFINITY; 3];
        let mut cluster_max = [f32::NEG_INFINITY; 3];
        for i in 0..inst.instances.len() {
            let Some(model_mat) = inst.instance_model_matrix(i) else {
                continue;
            };
            let (bb_min, bb_max) =
                crate::gfx::frustum::transform_aabb(local_min, local_max, model_mat);
            for k in 0..3 {
                cluster_min[k] = cluster_min[k].min(bb_min[k]);
                cluster_max[k] = cluster_max[k].max(bb_max[k]);
            }
            instance_mats.push(model_mat);
        }
        if instance_mats.is_empty() {
            continue;
        }

        instanced_clusters.push(InstancedCluster {
            vertex_offset,
            vertex_count,
            index_offset,
            index_count,
            texture_slot,
            normal_map_slot,
            material,
            cluster_bb_min: cluster_min,
            cluster_bb_max: cluster_max,
            local_bb_min: local_min,
            local_bb_max: local_max,
            cull_distance: inst.cull_distance,
            instances: instance_mats,
            lod_alternates,
        });
    }

    // unreferenced meshes (e.g. a standalone room): identity model matrix, slot 0.
    // These are drawn unconditionally; culling is disabled via the sentinel AABB.
    for mesh_handle in 0..mesh_geometry.len() {
        if referenced.contains(&mesh_handle) {
            continue;
        }
        if let Some((
            vertex_offset,
            vertex_count,
            index_offset,
            index_count,
            lod_alternates,
            _,
            _,
        )) = append_mesh(mesh_handle)
        {
            // Auto-rendered unreferenced meshes (e.g. a standalone room mesh)
            // are non-cullable, so distance-keyed LOD swaps make no sense
            // here. Drop any alternates the build emitted; the LOD0 draw is
            // the only one that will fire.
            let _ = lod_alternates;
            mesh_handle_to_draws
                .entry(mesh_handle)
                .or_default()
                .push(draw_objects.len());
            draw_objects.push(DrawObject {
                vertex_offset,
                vertex_count,
                index_offset,
                index_count,
                base_vertex: 0,
                model: IDENTITY4,
                texture_slot: 0,
                normal_map_slot: NO_NORMAL_MAP_SLOT,
                material: MaterialUniforms::DEFAULT,
                visible: true,
                resident: true,
                bb_min: UNCULLED_BB.0,
                bb_max: UNCULLED_BB.1,
                cull_distance: 0.0,
                lod_alternates: Vec::new(),
            });
        }
    }

    // Room components placed at the world origin with optional texture.
    // Rooms also opt out of culling (they enclose the camera). LOD picks
    // come from camera-to-origin distance per [`crate::gfx::lod::camera_distance`]'s
    // sentinel-AABB fallback, so practical swaps only fire if the camera
    // wanders far from the world origin.
    for (room, verts, idxs, room_lods) in room_geometry {
        let vertex_byte_offset = all_vertices.len() * std::mem::size_of::<Vertex>();
        let index_elem_offset = all_indices.len();
        let base = all_vertices.len() as u32;
        all_vertices.extend_from_slice(verts);
        all_indices.extend(idxs.iter().map(|i| u32::from(*i) + base));
        let mut lod_slices: Vec<LodSlice> = Vec::with_capacity(room_lods.len());
        for (switch_distance, alt_idx) in room_lods {
            let alt_offset = all_indices.len();
            all_indices.extend(alt_idx.iter().map(|i| u32::from(*i) + base));
            lod_slices.push(LodSlice {
                index_offset: alt_offset,
                index_count: alt_idx.len(),
                switch_distance: *switch_distance,
            });
        }
        // A room's texture carries its cook-assigned `TextureHandle`, whose
        // value is the texture's slot in the albedo pool. An out-of-range handle
        // (an unresolved generator name) falls back to slot 0, as before.
        let texture_slot = match room.effective_texture() {
            None => 0,
            Some(handle) => {
                let slot = handle.index();
                if slot < texture_count { slot } else { 0 }
            }
        };
        draw_objects.push(DrawObject {
            vertex_offset: vertex_byte_offset,
            vertex_count: verts.len(),
            index_offset: index_elem_offset,
            index_count: idxs.len(),
            base_vertex: 0,
            model: IDENTITY4,
            texture_slot,
            normal_map_slot: NO_NORMAL_MAP_SLOT,
            material: MaterialUniforms::DEFAULT,
            visible: true,
            resident: true,
            bb_min: UNCULLED_BB.0,
            bb_max: UNCULLED_BB.1,
            cull_distance: 0.0,
            lod_alternates: lod_slices,
        });
    }

    Some(DrawListData {
        vertices: all_vertices,
        indices: all_indices,
        draw_objects,
        instanced_clusters,
        prop_draw_indices,
        mesh_handle_to_draws,
        prop_local_bounds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::Prop;
    use crate::ecs::TextureHandle;

    fn material_map_with(
        handle: MaterialHandle,
        entry: MaterialEntry,
    ) -> std::collections::HashMap<MaterialHandle, MaterialEntry> {
        let mut m = std::collections::HashMap::new();
        m.insert(handle, entry);
        m
    }

    #[test]
    fn resolve_material_slots_prefers_a_resolved_material() {
        let entry: MaterialEntry = (7, 3, MaterialUniforms::DEFAULT);
        let map = material_map_with(MaterialHandle(2), entry);
        // A material handle wins even when a texture is also present; its albedo
        // and normal-map slots come straight from the map entry.
        let got = resolve_material_slots(Some(MaterialHandle(2)), Some(TextureHandle(9)), &map, 16)
            .expect("material resolves");
        assert_eq!((got.0, got.1), (7, 3));
    }

    #[test]
    fn resolve_material_slots_reports_a_missing_material_by_handle() {
        let map = std::collections::HashMap::new();
        let got = resolve_material_slots(Some(MaterialHandle(5)), None, &map, 16);
        assert_eq!(got.err(), Some(MaterialHandle(5)));
    }

    #[test]
    fn resolve_material_slots_falls_back_to_the_texture_slot() {
        let map = std::collections::HashMap::new();
        let got = resolve_material_slots(None, Some(TextureHandle(4)), &map, 16).expect("ok");
        assert_eq!((got.0, got.1), (4, NO_NORMAL_MAP_SLOT));
    }

    #[test]
    fn resolve_material_slots_clamps_an_out_of_range_texture_to_slot_zero() {
        let map = std::collections::HashMap::new();
        // Handle 20 is past the pool of 16, so it clamps to slot 0.
        let got = resolve_material_slots(None, Some(TextureHandle(20)), &map, 16).expect("ok");
        assert_eq!((got.0, got.1), (0, NO_NORMAL_MAP_SLOT));
    }

    #[test]
    fn resolve_material_slots_defaults_with_no_material_or_texture() {
        let map = std::collections::HashMap::new();
        let got = resolve_material_slots(None, None, &map, 16).expect("ok");
        assert_eq!((got.0, got.1), (0, NO_NORMAL_MAP_SLOT));
    }

    fn make_prop(position: [f32; 3]) -> Prop {
        Prop {
            asset_id: AssetId::default(),
            model: None,
            mesh: None,
            material: None,
            texture: None,
            position,
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            collider: None,
            interactable: false,
            pickup: false,
            parent: None,
            scene: None,
            prefab: String::new(),
            cull_distance: 0.0,
            is_held: false,
        }
    }

    // propagate_transforms composes each entity's GlobalTransform from its parent
    // chain: a root's world matrix is its local, a child's is parent_world * local.
    #[test]
    fn propagate_transforms_composes_parent_then_child() {
        use crate::assets::{GlobalTransform, Parent, Transform};
        use crate::blob::BlobData;
        use crate::ecs::{ComponentStorage, PipelineContext, Resources};
        use crate::gfx::profile::FrameProfile;

        let parent_t = Transform {
            position: [1.0, 2.0, 3.0],
            rotation_deg: [0.0, 30.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        };
        let child_t = Transform {
            position: [0.0, 0.0, 1.0],
            rotation_deg: [10.0, 0.0, 5.0],
            scale: [2.0, 2.0, 2.0],
        };

        let mut components = ComponentStorage::default();
        let mut blob = BlobData::empty();
        let mut profile = FrameProfile::default();
        let mut resources = Resources::new();
        let mut ctx = PipelineContext {
            components: &mut components,
            blob: &mut blob,
            profile: &mut profile,
            resources: &mut resources,
        };

        // A child parented to a root, each with its own GlobalTransform to write.
        let parent_e = ctx.components.spawn();
        ctx.insert(parent_e, parent_t);
        ctx.insert(parent_e, GlobalTransform::default());
        let child_e = ctx.components.spawn();
        ctx.insert(child_e, child_t);
        ctx.insert(child_e, Parent(parent_e));
        ctx.insert(child_e, GlobalTransform::default());

        propagate_transforms(&mut ctx);

        let parent_g = ctx.components.get::<GlobalTransform>(parent_e).unwrap().0;
        let child_g = ctx.components.get::<GlobalTransform>(child_e).unwrap().0;
        assert_eq!(parent_g, parent_t.model_matrix(), "root world = local");
        assert_eq!(
            child_g,
            mat_mul4(parent_t.model_matrix(), child_t.model_matrix()),
            "child world = parent_world * local"
        );
    }

    // The cached per-frame path resolves the same parent-then-child composition
    // as the uncached `propagate_transforms`.
    #[test]
    fn cached_propagation_matches_the_uncached_path() {
        use crate::assets::{GlobalTransform, Parent, Transform};
        use crate::blob::BlobData;
        use crate::ecs::{ComponentStorage, PipelineContext, Resources};
        use crate::gfx::profile::FrameProfile;

        let parent_t = Transform {
            position: [1.0, 2.0, 3.0],
            rotation_deg: [0.0, 30.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        };
        let child_t = Transform {
            position: [0.0, 0.0, 1.0],
            rotation_deg: [10.0, 0.0, 5.0],
            scale: [2.0, 2.0, 2.0],
        };

        let mut components = ComponentStorage::default();
        let mut blob = BlobData::empty();
        let mut profile = FrameProfile::default();
        let mut resources = Resources::new();
        let mut ctx = PipelineContext {
            components: &mut components,
            blob: &mut blob,
            profile: &mut profile,
            resources: &mut resources,
        };

        let parent_e = ctx.components.spawn();
        ctx.insert(parent_e, parent_t);
        ctx.insert(parent_e, GlobalTransform::default());
        let child_e = ctx.components.spawn();
        ctx.insert(child_e, child_t);
        ctx.insert(child_e, Parent(parent_e));
        ctx.insert(child_e, GlobalTransform::default());

        let mut cache = TransformCache::default();
        propagate_transforms_cached(&mut ctx, &mut cache);

        assert_eq!(
            ctx.components.get::<GlobalTransform>(parent_e).unwrap().0,
            parent_t.model_matrix()
        );
        assert_eq!(
            ctx.components.get::<GlobalTransform>(child_e).unwrap().0,
            mat_mul4(parent_t.model_matrix(), child_t.model_matrix())
        );
    }

    // The cached path skips the resolve (and the GlobalTransform writes) on
    // frames where no Transform / Parent changed, and recomputes once one does.
    #[test]
    fn cached_propagation_skips_until_a_transform_changes() {
        use crate::assets::{GlobalTransform, Transform};
        use crate::blob::BlobData;
        use crate::ecs::{ComponentStorage, PipelineContext, Resources};
        use crate::gfx::profile::FrameProfile;

        let mut components = ComponentStorage::default();
        let mut blob = BlobData::empty();
        let mut profile = FrameProfile::default();
        let mut resources = Resources::new();
        let mut ctx = PipelineContext {
            components: &mut components,
            blob: &mut blob,
            profile: &mut profile,
            resources: &mut resources,
        };

        let e = ctx.components.spawn();
        let t0 = Transform {
            position: [1.0, 0.0, 0.0],
            rotation_deg: [0.0; 3],
            scale: [1.0; 3],
        };
        ctx.insert(e, t0);
        ctx.insert(e, GlobalTransform::default());

        let mut cache = TransformCache::default();
        propagate_transforms_cached(&mut ctx, &mut cache);
        assert_eq!(
            ctx.components.get::<GlobalTransform>(e).unwrap().0,
            t0.model_matrix()
        );

        // A GlobalTransform write does not dirty the Transform column, so the
        // next pass must skip and leave the (deliberately corrupted) value.
        ctx.get_mut::<GlobalTransform>(e).unwrap().0 = IDENTITY4;
        propagate_transforms_cached(&mut ctx, &mut cache);
        assert_eq!(
            ctx.components.get::<GlobalTransform>(e).unwrap().0,
            IDENTITY4,
            "unchanged Transform => propagation skipped"
        );

        // Mutating the Transform dirties its column; the next pass recomputes.
        let t1 = Transform {
            position: [0.0, 5.0, 0.0],
            rotation_deg: [0.0; 3],
            scale: [1.0; 3],
        };
        *ctx.get_mut::<Transform>(e).unwrap() = t1;
        propagate_transforms_cached(&mut ctx, &mut cache);
        assert_eq!(
            ctx.components.get::<GlobalTransform>(e).unwrap().0,
            t1.model_matrix(),
            "changed Transform => propagation recomputed"
        );
    }

    #[test]
    fn reparent_recomposes_child_world_matrix_and_relists() {
        use crate::assets::{Children, GlobalTransform, Parent, Transform};
        use crate::blob::BlobData;
        use crate::ecs::{ComponentStorage, PipelineContext, Resources};
        use crate::gfx::profile::FrameProfile;

        let translate = |x: f32| Transform {
            position: [x, 0.0, 0.0],
            rotation_deg: [0.0; 3],
            scale: [1.0; 3],
        };
        let (a_t, b_t, child_t) = (translate(10.0), translate(-5.0), translate(1.0));

        let mut components = ComponentStorage::default();
        let mut blob = BlobData::empty();
        let mut profile = FrameProfile::default();
        let mut resources = Resources::new();
        let mut ctx = PipelineContext {
            components: &mut components,
            blob: &mut blob,
            profile: &mut profile,
            resources: &mut resources,
        };

        // Two candidate parents and a child, each with a GlobalTransform slot.
        let a = ctx.components.spawn();
        ctx.insert(a, a_t);
        ctx.insert(a, GlobalTransform::default());
        let b = ctx.components.spawn();
        ctx.insert(b, b_t);
        ctx.insert(b, GlobalTransform::default());
        let child = ctx.components.spawn();
        ctx.insert(child, child_t);
        ctx.insert(child, GlobalTransform::default());

        // Attach under A: the child's world matrix composes A x local, and A
        // lists it.
        reparent(&mut ctx, child, Some(a));
        let under_a = ctx.components.get::<GlobalTransform>(child).unwrap().0;
        assert_eq!(
            under_a,
            mat_mul4(a_t.model_matrix(), child_t.model_matrix())
        );
        assert_eq!(ctx.components.get::<Children>(a).unwrap().0, vec![child]);

        // Move under B: world matrix recomposes against B, A unlists it.
        reparent(&mut ctx, child, Some(b));
        let under_b = ctx.components.get::<GlobalTransform>(child).unwrap().0;
        assert_eq!(
            under_b,
            mat_mul4(b_t.model_matrix(), child_t.model_matrix())
        );
        assert_ne!(under_a, under_b, "the child actually moved");
        assert!(
            ctx.components.get::<Children>(a).unwrap().0.is_empty(),
            "A unlisted the child"
        );
        assert_eq!(ctx.components.get::<Children>(b).unwrap().0, vec![child]);
        assert_eq!(ctx.components.get::<Parent>(child).unwrap().0, b);

        // Detach to a root: no Parent, world matrix is its own local.
        reparent(&mut ctx, child, None);
        assert_eq!(
            ctx.components.get::<GlobalTransform>(child).unwrap().0,
            child_t.model_matrix()
        );
        assert!(
            ctx.components.get::<Parent>(child).is_none(),
            "child is now a root"
        );
        assert!(
            ctx.components.get::<Children>(b).unwrap().0.is_empty(),
            "B unlisted the child"
        );
    }

    #[test]
    fn mat_mul4_identity_is_no_op() {
        let m = [
            [1.0, 2.0, 3.0, 0.0],
            [4.0, 5.0, 6.0, 0.0],
            [7.0, 8.0, 9.0, 0.0],
            [10.0, 11.0, 12.0, 1.0],
        ];
        assert_eq!(mat_mul4(m, IDENTITY4), m);
        assert_eq!(mat_mul4(IDENTITY4, m), m);
    }

    #[test]
    fn mat_mul4_translations_compose() {
        // T(1,0,0) * T(0,1,0) should give combined translation (1,1,0).
        // Column-major: translation is in col 3.
        let tx = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ];
        let ty = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ];
        let result = mat_mul4(tx, ty);
        assert_eq!(result[3], [1.0, 1.0, 0.0, 1.0]);
        assert_eq!(result[0], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(result[1], [0.0, 1.0, 0.0, 0.0]);
    }

    fn unit_quad_mesh() -> LoadedMesh {
        // Axis-aligned unit cube centred at origin; bounds = [-0.5, 0.5]^3.
        let mk = |x, y, z| Vertex {
            pos: [x, y, z],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0],
            uv: [0.0, 0.0],
        };
        let v = vec![
            mk(-0.5, -0.5, -0.5),
            mk(0.5, -0.5, -0.5),
            mk(0.5, 0.5, -0.5),
            mk(-0.5, 0.5, -0.5),
        ];
        let i = vec![0u16, 1, 2, 0, 2, 3];
        LoadedMesh {
            vertices: v,
            indices: i,
            lod_alternates: Vec::new(),
            bounds: None,
            counts: None,
        }
    }

    #[test]
    fn build_draw_list_emits_one_cluster_for_instanced_prop() {
        let mesh_geometry = vec![unit_quad_mesh()];

        let inst = crate::assets::InstancedProp {
            asset_id: AssetId::default(),
            mesh: Some(MeshHandle(0)),
            material: None,
            texture: None,
            cull_distance: 0.0,
            instances: vec![
                crate::assets::InstanceTransform {
                    position: [0.0, 0.0, 0.0],
                    rotation_deg: [0.0; 3],
                    scale: [1.0; 3],
                },
                crate::assets::InstanceTransform {
                    position: [5.0, 0.0, 0.0],
                    rotation_deg: [0.0; 3],
                    scale: [1.0; 3],
                },
                crate::assets::InstanceTransform {
                    position: [-3.0, 0.0, 2.0],
                    rotation_deg: [0.0; 3],
                    scale: [1.0; 3],
                },
            ],
        };

        let data = build_draw_list(DrawListInputs {
            items: &[],
            instanced_props: &[inst],
            world_mats: &[],
            model_map: &std::collections::HashMap::new(),
            mesh_geometry: &mesh_geometry,
            room_geometry: &[],
            texture_count: 0,
            material_map: &std::collections::HashMap::new(),
            always_resident_meshes: &std::collections::HashSet::new(),
        })
        .expect("build_draw_list");
        let DrawListData {
            vertices: verts,
            indices: idxs,
            draw_objects,
            instanced_clusters: clusters,
            mesh_handle_to_draws,
            ..
        } = data;

        // Cluster mesh appended exactly once into the shared buffers.
        assert_eq!(verts.len(), 4);
        assert_eq!(idxs.len(), 6);
        // InstancedProp meshes go into clusters, not draw_objects; the
        // hot-reload map (which only tracks draw_objects-backed pushes) stays
        // empty for this scene.
        assert!(mesh_handle_to_draws.is_empty());

        // Each instance no longer emits its own DrawObject; the cluster carries
        // every transform.
        assert!(draw_objects.is_empty());
        assert_eq!(clusters.len(), 1);
        let c = &clusters[0];
        assert_eq!(c.instances.len(), 3);
        assert_eq!(c.index_count, 6);

        // Union AABB over all per-instance world AABBs. The unit_quad_mesh
        // is planar at z=-0.5, so each instance contributes a flat slab in z;
        // x and y span the quad extent [-0.5, 0.5].
        assert!((c.cluster_bb_min[0] - (-3.5)).abs() < 1e-5);
        assert!((c.cluster_bb_max[0] - 5.5).abs() < 1e-5);
        assert!((c.cluster_bb_min[1] - (-0.5)).abs() < 1e-5);
        assert!((c.cluster_bb_max[1] - 0.5).abs() < 1e-5);
        // z: instances at z=0 give [-0.5,-0.5]; instance at z=2 gives [1.5,1.5];
        // union is [-0.5, 1.5].
        assert!((c.cluster_bb_min[2] - (-0.5)).abs() < 1e-5);
        assert!((c.cluster_bb_max[2] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn build_draw_list_skips_empty_instanced_prop() {
        let mesh_geometry = vec![unit_quad_mesh()];

        let inst = crate::assets::InstancedProp {
            asset_id: AssetId::default(),
            mesh: Some(MeshHandle(0)),
            material: None,
            texture: None,
            cull_distance: 0.0,
            instances: Vec::new(),
        };

        let data = build_draw_list(DrawListInputs {
            items: &[],
            instanced_props: &[inst],
            world_mats: &[],
            model_map: &std::collections::HashMap::new(),
            mesh_geometry: &mesh_geometry,
            room_geometry: &[],
            texture_count: 0,
            material_map: &std::collections::HashMap::new(),
            always_resident_meshes: &std::collections::HashSet::new(),
        })
        .expect("build_draw_list");
        let DrawListData {
            draw_objects,
            instanced_clusters: clusters,
            ..
        } = data;

        assert!(draw_objects.is_empty());
        assert!(clusters.is_empty());
    }

    #[test]
    fn always_resident_mesh_forces_uncullable_bb_on_static_prop() {
        // A static prop with no dynamic flags would normally get a finite AABB
        // and be picked up by the streamer's `obj.cullable()` selection. When
        // its mesh is in the always_resident_meshes set (e.g. the auto-generated
        // skybox), the bb is forced to NaN so the prop opts out of frustum
        // culling and of mesh streaming. This is what the StreamingConfig
        // docstring promises for the skybox.
        let mesh_geometry = vec![unit_quad_mesh()];

        // A single static mesh-backed item referencing the always-resident mesh.
        let items = vec![RenderableItem {
            asset_id: AssetId(0),
            model: None,
            mesh: Some(MeshHandle(0)),
            material: None,
            texture: None,
            cull_distance: 0.0,
            is_dynamic: false,
        }];
        let world_mats = vec![IDENTITY4];

        let mut always_resident = std::collections::HashSet::new();
        always_resident.insert(0usize);

        let data = build_draw_list(DrawListInputs {
            items: &items,
            instanced_props: &[],
            world_mats: &world_mats,
            model_map: &std::collections::HashMap::new(),
            mesh_geometry: &mesh_geometry,
            room_geometry: &[],
            texture_count: 0,
            material_map: &std::collections::HashMap::new(),
            always_resident_meshes: &always_resident,
        })
        .expect("build_draw_list");
        let DrawListData { draw_objects, .. } = data;

        assert_eq!(draw_objects.len(), 1);
        // UNCULLED_BB is all-NaN; `cullable()` returns false in that case.
        assert!(draw_objects[0].bb_min[0].is_nan());
        assert!(draw_objects[0].bb_max[0].is_nan());
        assert!(!draw_objects[0].cullable());
    }

    // The item built from a mesh entity's components reads the renderer fields
    // from MeshRenderer and marks the entity dynamic from its Pickup / Collider
    // tags.
    #[test]
    fn decomposed_renderable_item_matches_a_mesh_prop() {
        use crate::assets::{Collider, MeshRenderer, Pickup, PropCollider};
        use crate::blob::BlobData;
        use crate::ecs::{ComponentStorage, PipelineContext, Resources};
        use crate::gfx::profile::FrameProfile;

        let mut prop = make_prop([0.0; 3]);
        prop.asset_id = AssetId(7);
        prop.mesh = Some(MeshHandle(10));
        prop.material = Some(MaterialHandle(20));
        prop.cull_distance = 50.0;
        prop.pickup = true;
        prop.collider = Some(PropCollider::default());

        let mut components = ComponentStorage::default();
        let mut blob = BlobData::empty();
        let mut profile = FrameProfile::default();
        let mut resources = Resources::new();
        let mut ctx = PipelineContext {
            components: &mut components,
            blob: &mut blob,
            profile: &mut profile,
            resources: &mut resources,
        };

        let e = ctx.components.spawn();
        ctx.insert(
            e,
            MeshRenderer {
                mesh: prop.mesh,
                material: prop.material,
                texture: prop.texture,
                cull_distance: prop.cull_distance,
            },
        );
        ctx.insert(e, Pickup);
        ctx.insert(e, Collider(prop.collider.clone().unwrap()));

        let item = decomposed_renderable_item(&ctx, e, prop.asset_id);
        assert_eq!(
            item,
            RenderableItem {
                asset_id: AssetId(7),
                model: None,
                mesh: Some(MeshHandle(10)),
                material: Some(MaterialHandle(20)),
                texture: None,
                cull_distance: 50.0,
                is_dynamic: true,
            }
        );
    }

    fn mesh_item(mesh: AssetId) -> RenderableItem {
        RenderableItem {
            asset_id: mesh,
            model: None,
            // A `.mesh` handle indexes the dense geometry slice directly, so a
            // test item's handle is the geometry index it draws.
            mesh: Some(MeshHandle(mesh.0)),
            material: None,
            texture: None,
            cull_distance: 0.0,
            is_dynamic: false,
        }
    }

    fn model_item(model: AssetId) -> RenderableItem {
        RenderableItem {
            asset_id: model,
            model: Some(model),
            mesh: None,
            material: None,
            texture: None,
            cull_distance: 0.0,
            is_dynamic: false,
        }
    }

    // The model path emits one draw object per sub-mesh, each over its own
    // geometry region, and records both draws under the shared prop index.
    #[test]
    fn build_draw_list_model_emits_one_draw_per_submesh() {
        let mesh_geometry = vec![unit_quad_mesh(), unit_quad_mesh()];

        let mut model_map = std::collections::HashMap::new();
        model_map.insert(
            AssetId(1),
            vec![
                SubMeshRef {
                    mesh: Some(MeshHandle(0)),
                    material: Some(MaterialHandle(20)),
                },
                SubMeshRef {
                    mesh: Some(MeshHandle(1)),
                    material: None,
                },
            ],
        );

        let mut material_map = std::collections::HashMap::new();
        material_map.insert(
            MaterialHandle(20),
            (3usize, 4usize, MaterialUniforms::DEFAULT),
        );

        let data = build_draw_list(DrawListInputs {
            items: &[model_item(AssetId(1))],
            instanced_props: &[],
            world_mats: &[IDENTITY4],
            model_map: &model_map,
            mesh_geometry: &mesh_geometry,
            room_geometry: &[],
            texture_count: 0,
            material_map: &material_map,
            always_resident_meshes: &std::collections::HashSet::new(),
        })
        .expect("build_draw_list");
        let DrawListData {
            vertices: verts,
            indices: idxs,
            draw_objects,
            instanced_clusters: clusters,
            prop_draw_indices: prop_idxs,
            mesh_handle_to_draws,
            ..
        } = data;

        assert!(clusters.is_empty());
        // Two sub-meshes -> two draws, both belonging to the one prop.
        assert_eq!(draw_objects.len(), 2);
        assert_eq!(prop_idxs, vec![vec![0, 1]]);
        assert_eq!(verts.len(), 8, "each quad's 4 verts appended once");
        assert_eq!(idxs.len(), 12);
        // First sub-mesh took its material's albedo/normal slots; the second
        // used the default (slot 0, flat normal).
        assert_eq!(draw_objects[0].texture_slot, 3);
        assert_eq!(draw_objects[0].normal_map_slot, 4);
        assert_eq!(draw_objects[1].texture_slot, 0);
        assert_eq!(draw_objects[1].normal_map_slot, NO_NORMAL_MAP_SLOT);
        // Hot-reload map tracks each sub-mesh id -> its draw slot.
        assert_eq!(mesh_handle_to_draws.get(&0), Some(&vec![0]));
        assert_eq!(mesh_handle_to_draws.get(&1), Some(&vec![1]));
    }

    // A mesh present in the geometry table but referenced by no item, model, or
    // instanced prop is auto-rendered at the origin with culling disabled.
    #[test]
    fn build_draw_list_auto_renders_unreferenced_mesh() {
        let mesh_geometry = vec![unit_quad_mesh()];

        let data = build_draw_list(DrawListInputs {
            items: &[],
            instanced_props: &[],
            world_mats: &[],
            model_map: &std::collections::HashMap::new(),
            mesh_geometry: &mesh_geometry,
            room_geometry: &[],
            texture_count: 0,
            material_map: &std::collections::HashMap::new(),
            always_resident_meshes: &std::collections::HashSet::new(),
        })
        .expect("build_draw_list");
        let DrawListData {
            draw_objects,
            prop_draw_indices: prop_idxs,
            mesh_handle_to_draws,
            ..
        } = data;

        assert!(prop_idxs.is_empty(), "no props drove this draw");
        assert_eq!(draw_objects.len(), 1);
        let d = &draw_objects[0];
        assert_eq!(d.model, IDENTITY4);
        assert_eq!(d.texture_slot, 0);
        assert!(!d.cullable(), "unreferenced mesh draws unconditionally");
        assert!(d.lod_alternates.is_empty());
        assert_eq!(mesh_handle_to_draws.get(&0), Some(&vec![0]));
    }

    // A Room is placed at the origin with culling disabled; its cook-assigned
    // texture handle is used directly as its albedo pool slot and its LOD
    // alternates carry through.
    #[test]
    fn build_draw_list_places_room_at_origin_with_texture_and_lods() {
        let room = Room {
            asset_id: AssetId(50),
            half_width: 8.0,
            half_depth: 10.0,
            ceiling_height: 3.5,
            texture: Some(TextureHandle(6)),
            wall_texture: None,
            floor_texture: None,
            ceiling_texture: None,
            locator: None,
        };
        let verts = unit_quad_mesh().vertices;
        let idxs = vec![0u16, 1, 2, 0, 2, 3];
        let room_lods = vec![(12.0_f32, vec![0u16, 1, 2])];
        let room_geometry = vec![(room, verts, idxs, room_lods)];

        // Handle 6 must land inside the pool; a 7-texture pool (slots 0..=6)
        // makes it the last valid slot.
        let data = build_draw_list(DrawListInputs {
            items: &[],
            instanced_props: &[],
            world_mats: &[],
            model_map: &std::collections::HashMap::new(),
            mesh_geometry: &[],
            room_geometry: &room_geometry,
            texture_count: 7,
            material_map: &std::collections::HashMap::new(),
            always_resident_meshes: &std::collections::HashSet::new(),
        })
        .expect("build_draw_list");
        let DrawListData {
            vertices: rv,
            indices: ri,
            draw_objects,
            ..
        } = data;

        assert_eq!(draw_objects.len(), 1);
        let d = &draw_objects[0];
        assert_eq!(d.model, IDENTITY4);
        assert_eq!(d.texture_slot, 6, "room texture resolved to its slot");
        assert!(!d.cullable(), "rooms enclose the camera and skip culling");
        assert_eq!(d.lod_alternates.len(), 1);
        assert_eq!(d.lod_alternates[0].switch_distance, 12.0);
        // LOD0 (6) + one alternate (3) indices appended after the 4 verts.
        assert_eq!(rv.len(), 4);
        assert_eq!(ri.len(), 9);
    }

    // A single-mesh item with a texture (and no material) resolves the texture
    // slot and keeps the default material.
    #[test]
    fn build_draw_list_single_mesh_resolves_texture_slot() {
        let mesh_geometry = vec![unit_quad_mesh()];
        // The texture handle is the pool slot directly; the pool size (3) makes
        // slot 2 in range.
        let item = RenderableItem {
            asset_id: AssetId(0),
            model: None,
            mesh: Some(MeshHandle(0)),
            material: None,
            texture: Some(TextureHandle(2)),
            cull_distance: 0.0,
            is_dynamic: false,
        };

        let data = build_draw_list(DrawListInputs {
            items: &[item],
            instanced_props: &[],
            world_mats: &[IDENTITY4],
            model_map: &std::collections::HashMap::new(),
            mesh_geometry: &mesh_geometry,
            room_geometry: &[],
            texture_count: 3,
            material_map: &std::collections::HashMap::new(),
            always_resident_meshes: &std::collections::HashSet::new(),
        })
        .expect("build_draw_list");
        let DrawListData { draw_objects, .. } = data;

        assert_eq!(draw_objects.len(), 1);
        assert_eq!(draw_objects[0].texture_slot, 2);
        assert_eq!(draw_objects[0].normal_map_slot, NO_NORMAL_MAP_SLOT);
    }

    // Every missing-reference branch returns None (the error is logged and the
    // build aborts) rather than emitting a partial draw list.
    #[test]
    fn build_draw_list_returns_none_on_missing_references() {
        let mesh = || vec![unit_quad_mesh()];
        let none = |inputs: DrawListInputs| build_draw_list(inputs).is_none();

        // Model referenced by an item but absent from the model_map.
        assert!(none(DrawListInputs {
            items: &[model_item(AssetId(1))],
            instanced_props: &[],
            world_mats: &[IDENTITY4],
            model_map: &std::collections::HashMap::new(),
            mesh_geometry: &mesh(),
            room_geometry: &[],
            texture_count: 0,
            material_map: &std::collections::HashMap::new(),
            always_resident_meshes: &std::collections::HashSet::new(),
        }));

        // Model sub-mesh with no mesh field.
        let mut model_no_mesh = std::collections::HashMap::new();
        model_no_mesh.insert(
            AssetId(1),
            vec![SubMeshRef {
                mesh: None,
                material: None,
            }],
        );
        assert!(none(DrawListInputs {
            items: &[model_item(AssetId(1))],
            instanced_props: &[],
            world_mats: &[IDENTITY4],
            model_map: &model_no_mesh,
            mesh_geometry: &mesh(),
            room_geometry: &[],
            texture_count: 0,
            material_map: &std::collections::HashMap::new(),
            always_resident_meshes: &std::collections::HashSet::new(),
        }));

        // Model sub-mesh whose mesh id has no geometry.
        let mut model_bad_geo = std::collections::HashMap::new();
        model_bad_geo.insert(
            AssetId(1),
            vec![SubMeshRef {
                mesh: Some(MeshHandle(999)),
                material: None,
            }],
        );
        assert!(none(DrawListInputs {
            items: &[model_item(AssetId(1))],
            instanced_props: &[],
            world_mats: &[IDENTITY4],
            model_map: &model_bad_geo,
            mesh_geometry: &mesh(),
            room_geometry: &[],
            texture_count: 0,
            material_map: &std::collections::HashMap::new(),
            always_resident_meshes: &std::collections::HashSet::new(),
        }));

        // Model sub-mesh referencing a material absent from the material_map.
        let mut model_bad_mat = std::collections::HashMap::new();
        model_bad_mat.insert(
            AssetId(1),
            vec![SubMeshRef {
                mesh: Some(MeshHandle(0)),
                material: Some(MaterialHandle(404)),
            }],
        );
        assert!(none(DrawListInputs {
            items: &[model_item(AssetId(1))],
            instanced_props: &[],
            world_mats: &[IDENTITY4],
            model_map: &model_bad_mat,
            mesh_geometry: &mesh(),
            room_geometry: &[],
            texture_count: 0,
            material_map: &std::collections::HashMap::new(),
            always_resident_meshes: &std::collections::HashSet::new(),
        }));

        // Single-mesh item whose mesh id has no geometry.
        assert!(none(DrawListInputs {
            items: &[mesh_item(AssetId(999))],
            instanced_props: &[],
            world_mats: &[IDENTITY4],
            model_map: &std::collections::HashMap::new(),
            mesh_geometry: &mesh(),
            room_geometry: &[],
            texture_count: 0,
            material_map: &std::collections::HashMap::new(),
            always_resident_meshes: &std::collections::HashSet::new(),
        }));

        // Single-mesh item referencing a material absent from the material_map.
        let mut item_bad_mat = mesh_item(AssetId(0));
        item_bad_mat.material = Some(MaterialHandle(404));
        assert!(none(DrawListInputs {
            items: &[item_bad_mat],
            instanced_props: &[],
            world_mats: &[IDENTITY4],
            model_map: &std::collections::HashMap::new(),
            mesh_geometry: &mesh(),
            room_geometry: &[],
            texture_count: 0,
            material_map: &std::collections::HashMap::new(),
            always_resident_meshes: &std::collections::HashSet::new(),
        }));

        // Item carrying neither a model nor a mesh.
        assert!(none(DrawListInputs {
            items: &[RenderableItem {
                asset_id: AssetId(0),
                model: None,
                mesh: None,
                material: None,
                texture: None,
                cull_distance: 0.0,
                is_dynamic: false,
            }],
            instanced_props: &[],
            world_mats: &[IDENTITY4],
            model_map: &std::collections::HashMap::new(),
            mesh_geometry: &mesh(),
            room_geometry: &[],
            texture_count: 0,
            material_map: &std::collections::HashMap::new(),
            always_resident_meshes: &std::collections::HashSet::new(),
        }));

        // InstancedProp mesh id has no geometry.
        let inst_bad_mesh = InstancedProp {
            asset_id: AssetId::default(),
            mesh: Some(MeshHandle(999)),
            material: None,
            texture: None,
            cull_distance: 0.0,
            instances: vec![crate::assets::InstanceTransform::default()],
        };
        assert!(none(DrawListInputs {
            items: &[],
            instanced_props: &[inst_bad_mesh],
            world_mats: &[],
            model_map: &std::collections::HashMap::new(),
            mesh_geometry: &mesh(),
            room_geometry: &[],
            texture_count: 0,
            material_map: &std::collections::HashMap::new(),
            always_resident_meshes: &std::collections::HashSet::new(),
        }));

        // InstancedProp material absent from the material_map.
        let inst_bad_mat = InstancedProp {
            asset_id: AssetId::default(),
            mesh: Some(MeshHandle(0)),
            material: Some(MaterialHandle(404)),
            texture: None,
            cull_distance: 0.0,
            instances: vec![crate::assets::InstanceTransform::default()],
        };
        assert!(none(DrawListInputs {
            items: &[],
            instanced_props: &[inst_bad_mat],
            world_mats: &[],
            model_map: &std::collections::HashMap::new(),
            mesh_geometry: &mesh(),
            room_geometry: &[],
            texture_count: 0,
            material_map: &std::collections::HashMap::new(),
            always_resident_meshes: &std::collections::HashSet::new(),
        }));
    }

    // Accumulates components + a single blob section so load_mesh_geometry /
    // load_room_geometry can decode in-memory payloads, mirroring the
    // GraphicsSystem WorldBuilder precedent.
    struct BlobWorld {
        components: crate::ecs::ComponentStorage,
        section: Vec<u8>,
    }

    struct SealedWorld {
        components: crate::ecs::ComponentStorage,
        blob: crate::blob::BlobData,
        profile: crate::gfx::profile::FrameProfile,
        resources: crate::ecs::Resources,
    }

    impl BlobWorld {
        fn new() -> Self {
            Self {
                components: crate::ecs::ComponentStorage::default(),
                section: Vec::new(),
            }
        }

        fn payload(&mut self, bytes: &[u8]) -> crate::ecs::PayloadLocator {
            let offset = self.section.len() as u64;
            self.section.extend_from_slice(bytes);
            crate::ecs::PayloadLocator {
                blob_index: 0,
                offset,
                len: bytes.len() as u64,
            }
        }

        fn push<C: crate::ecs::ComponentSlot>(&mut self, c: C) {
            self.components.push_typed(c);
        }

        fn seal(self) -> SealedWorld {
            SealedWorld {
                components: self.components,
                blob: crate::blob::BlobData::new(vec![Some(self.section)]),
                profile: crate::gfx::profile::FrameProfile::default(),
                resources: crate::ecs::Resources::new(),
            }
        }
    }

    impl SealedWorld {
        fn ctx(&mut self) -> crate::ecs::PipelineContext<'_> {
            crate::ecs::PipelineContext {
                components: &mut self.components,
                blob: &mut self.blob,
                profile: &mut self.profile,
                resources: &mut self.resources,
            }
        }

        // Install a `MeshTable` with one entry per locator (handle == index),
        // standing in for the blob resource stream a real build provides.
        fn with_mesh_table(
            mut self,
            locators: Vec<Option<crate::ecs::PayloadLocator>>,
        ) -> SealedWorld {
            let entries = locators
                .into_iter()
                .map(|payload| crate::resource::ResourceEntry {
                    payload,
                    data_bytes: Vec::new(),
                })
                .collect();
            self.resources.insert(crate::resource::MeshTable(entries));
            self
        }
    }

    // A single-triangle static mesh payload in the compiled format.
    fn tri_payload() -> Vec<u8> {
        let v = |x: f32, z: f32| {
            (
                [x, 0.0, z],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0],
            )
        };
        crate::gfx::mesh_payload::serialise(&[v(0.0, 0.0), v(1.0, 0.0), v(0.0, 1.0)], &[0u16, 1, 2])
    }

    // load_mesh_geometry decodes a MeshTable entry's in-memory payload into the
    // dense geometry table at its handle.
    #[test]
    fn load_mesh_geometry_decodes_in_memory_mesh() {
        let mut b = BlobWorld::new();
        let loc = b.payload(&tri_payload());
        let mut world = b.seal().with_mesh_table(vec![Some(loc)]);
        let mut ctx = world.ctx();

        let (geometry, sources, resident, component_handles, _deferred) =
            load_mesh_geometry(&mut ctx, &DeferredMeshSources::default(), false).expect("decoded");
        assert_eq!(geometry.len(), 1);
        let m = &geometry[0];
        assert_eq!(m.vertices.len(), 3);
        assert_eq!(m.indices, vec![0, 1, 2]);
        assert!(m.lod_alternates.is_empty());
        // No component-backed producers in this world.
        assert!(component_handles.is_empty());
        // The source-capture map only fills under the dev-flag global, which the
        // tests never set, so it stays empty here.
        assert!(sources.is_empty());
        assert!(
            resident.is_empty(),
            "no skybox mesh -> nothing always-resident"
        );
    }

    // A deferred Mesh skips its decode: empty geometry with the baked bounds,
    // and its payload seed (locator + RAM bytes) is captured for the streamer.
    #[test]
    fn load_mesh_geometry_defers_scene_owned_mesh_with_baked_bounds() {
        let mut b = BlobWorld::new();
        let loc = b.payload(&tri_payload());
        let mut world = b.seal().with_mesh_table(vec![Some(loc)]);
        let mut ctx = world.ctx();

        let mut deferred = DeferredMeshSources::default();
        deferred.by_handle.insert(0);
        deferred
            .bounds
            .insert(0, ([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]));

        let (geometry, _sources, _resident, _handles, seeds) =
            load_mesh_geometry(&mut ctx, &deferred, false).expect("ok");
        assert_eq!(geometry.len(), 1);
        assert!(geometry[0].vertices.is_empty(), "decode skipped");
        assert_eq!(
            geometry[0].bounds,
            Some(([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]))
        );
        let seed = seeds.get(&0).expect("payload seed captured");
        assert!(
            seed.bytes.as_deref().is_some_and(|b| !b.is_empty()),
            "RAM-backed seed carries the payload bytes"
        );
    }

    // A deferred member with no baked bounds record decodes eagerly.
    #[test]
    fn load_mesh_geometry_decodes_eagerly_without_baked_bounds() {
        let mut b = BlobWorld::new();
        let loc = b.payload(&tri_payload());
        let mut world = b.seal().with_mesh_table(vec![Some(loc)]);
        let mut ctx = world.ctx();

        let mut deferred = DeferredMeshSources::default();
        deferred.by_handle.insert(0);

        let (geometry, _sources, _resident, _handles, seeds) =
            load_mesh_geometry(&mut ctx, &deferred, false).expect("ok");
        assert_eq!(geometry[0].vertices.len(), 3, "no bounds record -> decode");
        assert!(seeds.is_empty());
    }

    // A skybox ProceduralMesh decodes and is marked always-resident so its props
    // opt out of culling and streaming.
    #[test]
    fn load_mesh_geometry_marks_skybox_always_resident() {
        let mut b = BlobWorld::new();
        let loc = b.payload(&tri_payload());
        b.push(ProceduralMesh {
            asset_id: AssetId(2),
            generator: "skybox".to_string(),
            locator: Some(loc),
            ..Default::default()
        });
        let mut world = b.seal();
        let mut ctx = world.ctx();

        let (geometry, _sources, resident, component_handles, _deferred) =
            load_mesh_geometry(&mut ctx, &DeferredMeshSources::default(), false).expect("decoded");
        // The lone component-backed producer got the first handle.
        assert_eq!(component_handles.get(&AssetId(2)), Some(&0));
        assert_eq!(geometry.len(), 1);
        assert!(resident.contains(&0), "skybox generator stays resident");
    }

    // A Mesh with no compiled payload aborts the whole load.
    #[test]
    fn load_mesh_geometry_missing_locator_returns_none() {
        let mut world = BlobWorld::new().seal().with_mesh_table(vec![None]);
        let mut ctx = world.ctx();
        assert!(load_mesh_geometry(&mut ctx, &DeferredMeshSources::default(), false).is_none());
    }

    // A malformed payload (too short to hold its declared vertices) aborts.
    #[test]
    fn load_mesh_geometry_malformed_payload_returns_none() {
        let mut b = BlobWorld::new();
        // Claims one vertex but carries no vertex bytes.
        let loc = b.payload(&1u32.to_le_bytes());
        let mut world = b.seal().with_mesh_table(vec![Some(loc)]);
        let mut ctx = world.ctx();
        assert!(load_mesh_geometry(&mut ctx, &DeferredMeshSources::default(), false).is_none());
    }

    // An empty world (no mesh sources at all) still succeeds with empty maps.
    #[test]
    fn load_mesh_geometry_empty_world_is_ok_and_empty() {
        let mut world = BlobWorld::new().seal();
        let mut ctx = world.ctx();
        let (geometry, sources, resident, component_handles, _deferred) =
            load_mesh_geometry(&mut ctx, &DeferredMeshSources::default(), false).expect("ok");
        assert!(geometry.is_empty() && sources.is_empty() && resident.is_empty());
        assert!(component_handles.is_empty());
    }

    fn test_room(locator: Option<crate::ecs::PayloadLocator>) -> Room {
        Room {
            asset_id: AssetId(50),
            half_width: 8.0,
            half_depth: 10.0,
            ceiling_height: 3.5,
            texture: None,
            wall_texture: None,
            floor_texture: None,
            ceiling_texture: None,
            locator,
        }
    }

    // load_room_geometry decodes each Room payload and reports its blob index.
    #[test]
    fn load_room_geometry_decodes_in_memory_room() {
        let mut b = BlobWorld::new();
        let loc = b.payload(&tri_payload());
        b.push(test_room(Some(loc)));
        let mut world = b.seal();
        let mut ctx = world.ctx();

        let (room_geometry, blob_indices) = load_room_geometry(&mut ctx).expect("decoded");
        assert_eq!(room_geometry.len(), 1);
        let (_room, verts, idxs, _lods) = &room_geometry[0];
        assert_eq!(verts.len(), 3);
        assert_eq!(*idxs, vec![0, 1, 2]);
        assert_eq!(blob_indices, vec![0]);
    }

    // A Room with no compiled payload aborts the load.
    #[test]
    fn load_room_geometry_missing_locator_returns_none() {
        let mut b = BlobWorld::new();
        b.push(test_room(None));
        let mut world = b.seal();
        let mut ctx = world.ctx();
        assert!(load_room_geometry(&mut ctx).is_none());
    }

    // resolve_world_matrices breaks a parent cycle: mutually-parented entities
    // fall back to their own local matrix rather than looping forever.
    #[test]
    fn resolve_world_matrices_breaks_parent_cycle() {
        use crate::assets::{Parent, Transform};

        let mut components = crate::ecs::ComponentStorage::default();
        let mut blob = crate::blob::BlobData::empty();
        let mut profile = crate::gfx::profile::FrameProfile::default();
        let mut resources = crate::ecs::Resources::new();
        let mut ctx = crate::ecs::PipelineContext {
            components: &mut components,
            blob: &mut blob,
            profile: &mut profile,
            resources: &mut resources,
        };

        let a_t = Transform {
            position: [1.0, 0.0, 0.0],
            rotation_deg: [0.0; 3],
            scale: [1.0; 3],
        };
        let b_t = Transform {
            position: [0.0, 2.0, 0.0],
            rotation_deg: [0.0; 3],
            scale: [1.0; 3],
        };

        // a parents b and b parents a: a cycle with no root.
        let a = ctx.components.spawn();
        ctx.insert(a, a_t);
        let b = ctx.components.spawn();
        ctx.insert(b, b_t);
        ctx.insert(a, Parent(b));
        ctx.insert(b, Parent(a));

        let world = resolve_world_matrices(&ctx);
        assert_eq!(world.len(), 2);
        // Neither resolved through the chain, so each keeps its own local matrix.
        assert_eq!(world.get(&a).copied(), Some(a_t.model_matrix()));
        assert_eq!(world.get(&b).copied(), Some(b_t.model_matrix()));
    }

    // Same for a model entity: ModelRenderer fields, and with no dynamic tags the
    // item is static.
    #[test]
    fn decomposed_renderable_item_matches_a_model_prop() {
        use crate::assets::ModelRenderer;
        use crate::blob::BlobData;
        use crate::ecs::{ComponentStorage, PipelineContext, Resources};
        use crate::gfx::profile::FrameProfile;

        let mut prop = make_prop([0.0; 3]);
        prop.asset_id = AssetId(8);
        prop.model = Some(AssetId(100));
        prop.cull_distance = 30.0;

        let mut components = ComponentStorage::default();
        let mut blob = BlobData::empty();
        let mut profile = FrameProfile::default();
        let mut resources = Resources::new();
        let mut ctx = PipelineContext {
            components: &mut components,
            blob: &mut blob,
            profile: &mut profile,
            resources: &mut resources,
        };

        let e = ctx.components.spawn();
        ctx.insert(
            e,
            ModelRenderer {
                model: prop.model.unwrap(),
                cull_distance: prop.cull_distance,
            },
        );

        let item = decomposed_renderable_item(&ctx, e, prop.asset_id);
        assert_eq!(
            item,
            RenderableItem {
                asset_id: AssetId(8),
                model: Some(AssetId(100)),
                mesh: None,
                material: None,
                texture: None,
                cull_distance: 30.0,
                is_dynamic: false,
            }
        );
    }
}
