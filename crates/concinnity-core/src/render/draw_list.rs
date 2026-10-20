//! Render-prep helpers that consume asset components and produce GPU-ready data.
//! None of these functions hold or borrow a backend handle.

use crate::components::{
    File, FileKind, Identity, InstancedProp, ProceduralMesh, Room, SubMeshRef, VoxelChunk,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{Entity, MaterialHandle, MeshHandle, PayloadLocator, PipelineContext};
use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::{
    DrawIndex, DrawObject, InstancedCluster, LodSlice, MaterialUniforms, NO_NORMAL_MAP_SLOT,
};
use crate::gfx::{frustum, mesh_payload};
use crate::math::SaturatingFrom;
use crate::render::material_entry::{MaterialEntry, resolve_material_slots};
use crate::resource::MeshTable;
use crate::transform::IDENTITY as IDENTITY4;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// Geometry decoded for one Room: the asset, its vertices, LOD0 indices, and
/// LOD alternates (`switch_distance`, indices).
pub type RoomGeometry = (Room, Vec<Vertex>, Vec<u16>, Vec<(f32, Vec<u16>)>);

/// Mesh-geometry lookup tables from `load_mesh_geometry`: the loaded geometry
/// (dense, indexed by the unified mesh-source handle, which a `.mesh`
/// reference's `MeshHandle` indexes directly), file-backed Mesh source metadata
/// keyed by handle (dev-only), and the asset id -> handle map for the geometry
/// producers that are still components (`ProceduralMesh` / `VoxelChunk` / File).
pub struct MeshGeometry {
    /// Decoded geometry, indexed by the unified mesh-source handle.
    pub meshes: Vec<LoadedMesh>,
    /// File-backed `Mesh` source metadata keyed by handle (dev-only).
    pub sources: hashbrown::HashMap<usize, MeshSourceMeta>,
    /// Asset id to handle, for the geometry producers that are still
    /// components (`ProceduralMesh` / `VoxelChunk` / `File`).
    pub component_handles: hashbrown::HashMap<AssetId, usize>,
    /// Payload references captured for meshes whose decode was deferred.
    pub deferred_seeds: hashbrown::HashMap<usize, DeferredMeshSeed>,
}

/// A deferred mesh's payload reference, captured while its decode was skipped:
/// the locator, plus the raw bytes when the blob is RAM-backed (an in-memory
/// world may release its payload sections after init, so the bytes are copied
/// out now; a disk-backed world re-reads the blob file range instead).
pub struct DeferredMeshSeed {
    /// Where the payload lives in its blob.
    pub locator: PayloadLocator,
    /// The payload bytes, copied out of a RAM-backed blob.
    pub bytes: Option<Vec<u8>>,
}

/// Mesh sources whose payload decode init defers: exclusively owned by a scene
/// other than the start scene, with baked bounds from the blob. `bounds` and
/// `counts` are keyed by mesh-source handle; a member with no baked record
/// decodes eagerly.
#[derive(Default)]
pub struct DeferredMeshSources {
    /// Deferred resource-stream Mesh handles.
    pub by_handle: hashbrown::HashSet<u32>,
    /// Deferred mesh-source components, by asset id.
    pub by_def: hashbrown::HashSet<AssetId>,
    /// Baked local bounds by mesh-source handle.
    pub bounds: hashbrown::HashMap<u32, ([f32; 3], [f32; 3])>,
    /// Baked (vertex, index) counts by mesh-source handle.
    pub counts: hashbrown::HashMap<u32, (u32, u32)>,
}

impl DeferredMeshSources {
    // Baked bounds for a deferred resource-stream Mesh, or None to decode.
    fn resource_bounds(&self, handle: usize) -> Option<([f32; 3], [f32; 3])> {
        if !self.by_handle.contains(&u32::saturating_from(handle)) {
            return None;
        }
        self.bounds.get(&u32::saturating_from(handle)).copied()
    }

    // Baked bounds for a deferred mesh-source component at its push position,
    // or None to decode.
    fn def_bounds(&self, id: AssetId, handle: usize) -> Option<([f32; 3], [f32; 3])> {
        if !self.by_def.contains(&id) {
            return None;
        }
        self.bounds.get(&u32::saturating_from(handle)).copied()
    }
}

/// Output of `build_draw_list`. `prop_draw_indices` and `prop_local_bounds`
/// are column-aligned with the input `items`; `mesh_handle_to_draws` backs
/// hot-reload. A prop whose meshes carry no vertices gets the non-finite
/// `UNCULLED_BB` as its local bounds (unpickable, uncullable).
pub struct DrawListData {
    /// The shared vertex buffer.
    pub vertices: Vec<Vertex>,
    /// The shared index buffer.
    pub indices: Vec<u32>,
    /// One record per draw.
    pub draw_objects: Vec<DrawObject>,
    /// One cluster per instanced prop.
    pub instanced_clusters: Vec<InstancedCluster>,
    /// Each item's draws, column-aligned with the input items.
    pub prop_draw_indices: Vec<Vec<DrawIndex>>,
    /// Each mesh-source handle's draws, for hot reload.
    pub mesh_handle_to_draws: hashbrown::HashMap<usize, Vec<DrawIndex>>,
    /// Each item's model-space bounds, column-aligned with the input items.
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

// Sentinel AABB used when a draw object opts out of culling (a dynamic prop,
// whose bounds move, or a room that encloses the camera). Every backend treats
// a non-finite component as "always draw".
const UNCULLED_BB: ([f32; 3], [f32; 3]) = (
    [f32::NAN, f32::NAN, f32::NAN],
    [f32::NAN, f32::NAN, f32::NAN],
);

// A box as its (min, max) corners.
type Aabb = ([f32; 3], [f32; 3]);

// A prop draw's world culling box and the model-space box it follows: the
// sentinel for a dynamic prop, else its mesh's box placed at `model`.
fn culled_bounds(
    item: &RenderableItem,
    local_min: [f32; 3],
    local_max: [f32; 3],
    model: [[f32; 4]; 4],
) -> ([f32; 3], [f32; 3], Option<Aabb>) {
    if item.is_dynamic || !crate::gfx::lod::bounds_finite(local_min, local_max) {
        return (UNCULLED_BB.0, UNCULLED_BB.1, None);
    }
    let (bb_min, bb_max) = frustum::transform_aabb(local_min, local_max, model);
    (bb_min, bb_max, Some((local_min, local_max)))
}

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

/// The renderer-relevant view of one placement that `build_draw_list` consumes:
/// the mesh/model/material/texture refs, the cull distance, whether it is dynamic
/// (skips frustum culling), and the asset id. Built from an
/// entity's MeshRenderer/ModelRenderer + tag components by
/// `decomposed_renderable_item`.
///
/// An entity is dynamic (never culled) when it carries a Pickup, Interactable,
/// Parent, or Collider tag. A static draw keeps its mesh's model-space box, so
/// a rigid move of the whole world (a planet's frame moving) re-places its
/// culling box with its model and it stays static.
#[derive(Debug, PartialEq)]
pub struct RenderableItem {
    /// The placement's asset id.
    pub asset_id: Option<AssetId>,
    /// The Model it draws, for a `ModelRenderer`.
    pub model: Option<AssetId>,
    /// The mesh it draws, for a `MeshRenderer`.
    pub mesh: Option<MeshHandle>,
    /// The material a `MeshRenderer` binds.
    pub material: Option<MaterialHandle>,
    /// Camera distance past which it is not drawn; 0 never culls by distance.
    pub cull_distance: f32,
    /// Whether it skips frustum culling.
    pub is_dynamic: bool,
}

/// Build one entity's `RenderableItem`: read its renderer fields from its
/// `MeshRenderer` xor `ModelRenderer`, its dynamic flag from the Pickup /
/// Interactable / Parent / Collider tags, and its asset id from its Identity.
pub fn decomposed_renderable_item(ctx: &PipelineContext<'_>, entity: Entity) -> RenderableItem {
    use crate::components::{Collider, Interactable, MeshRenderer, ModelRenderer, Parent, Pickup};

    let (model, mesh, material, cull_distance) = if let Some(m) = ctx.get::<ModelRenderer>(entity) {
        (Some(m.model), None, None, m.cull_distance)
    } else if let Some(m) = ctx.get::<MeshRenderer>(entity) {
        (None, m.mesh, m.material, m.cull_distance)
    } else {
        (None, None, None, 0.0)
    };
    let is_dynamic = ctx.get::<Pickup>(entity).is_some()
        || ctx.get::<Interactable>(entity).is_some()
        || ctx.get::<Parent>(entity).is_some()
        || ctx.get::<Collider>(entity).is_some();
    RenderableItem {
        asset_id: ctx.get::<Identity>(entity).map(|i| i.id()),
        model,
        mesh,
        material,
        cull_distance,
        is_dynamic,
    }
}

/// Append each LOD alternate's indices to `indices`, rebased onto `base`, and
/// return the slice table addressing them. The alternates share the base mesh's
/// vertex range, so only the index run differs per level.
pub fn append_lod_slices<T>(
    indices: &mut Vec<T>,
    alternates: &[(f32, Vec<u16>)],
    base: T,
) -> Vec<LodSlice>
where
    T: Copy + From<u16> + core::ops::Add<Output = T>,
{
    let mut slices = Vec::with_capacity(alternates.len());
    for (switch_distance, alt) in alternates {
        let index_offset = indices.len();
        indices.extend(alt.iter().map(|&i| T::from(i) + base));
        slices.push(LodSlice {
            index_offset,
            index_count: alt.len(),
            switch_distance: *switch_distance,
        });
    }
    slices
}

/// Decoded mesh geometry plus its optional LOD trailer. Returned by
/// `load_mesh_geometry` and consumed by `build_draw_list`. The `vertices`
/// slice is shared across LOD0 and every alternate; vertex-clustering
/// decimation reuses the original vertex set and only generates new index
/// lists. Empty `lod_alternates` means the mesh declared `lod_levels <= 1`
/// (or the build dropped degenerate decimations); the runtime then keeps
/// the single LOD0 slice.
pub struct LoadedMesh {
    /// The vertices every LOD shares.
    pub vertices: Vec<Vertex>,
    /// LOD0's indices.
    pub indices: Vec<u16>,
    /// Each LOD alternate's switch distance and indices.
    pub lod_alternates: Vec<(f32, Vec<u16>)>,
    /// Baked local AABB for a deferred mesh whose vertices were not decoded;
    /// None computes bounds from the vertices.
    pub bounds: Option<([f32; 3], [f32; 3])>,
    /// Baked (vertex, index) counts for a deferred mesh, so its draw record
    /// matches the geometry the streamer uploads later; None uses the decoded
    /// lengths.
    pub counts: Option<(u32, u32)>,
}

/// Hot-reload source metadata for a file-backed `Mesh`. Captured by
/// `load_mesh_geometry` before the Mesh is drained and consumed; the
/// `(asset_id, source, primitive_index, lod_levels, lod_distances)` tuple is
/// later cross-referenced against `build_draw_list`'s `mesh_id` to `draw_indices`
/// map to build the hot-reload source map.
pub struct MeshSourceMeta {
    /// Authored source path.
    pub source: String,
    /// Index of the primitive within the source document.
    pub primitive_index: u32,
    /// How many LODs the mesh declares, including LOD0.
    pub lod_levels: u32,
    /// Camera distance at which each LOD past 0 takes over.
    pub lod_distances: Vec<f32>,
}

/// Decode all mesh-source payloads into a dense, handle-indexed geometry table.
/// The Mesh block comes from the blob's resource stream (`MeshTable`); the
/// remaining geometry producers (`ProceduralMesh`, `VoxelChunk`, mesh-kind File) are
/// still components and are decoded after it, in the same fixed block order cook
/// assigned handles, so `geometry[h]` is the source cook gave handle `h`.
/// Returns None if any payload is missing or malformed. Also returns a
/// handle-keyed source-meta map for file-backed Mesh declarations, read from
/// the `MeshSources` catalog when `capture_sources` is set (a development
/// loop), and the asset id -> handle map for the still-component producers.
pub fn load_mesh_geometry(
    ctx: &mut PipelineContext<'_>,
    deferred: &DeferredMeshSources,
    blob_disk_backed: bool,
    capture_sources: bool,
) -> Option<MeshGeometry> {
    let mesh_table = ctx.resource::<MeshTable>().cloned().unwrap_or_default();
    let mut mesh_sources: hashbrown::HashMap<usize, MeshSourceMeta> = hashbrown::HashMap::new();
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
    // ProceduralMesh components are cloned rather than drained, so the column
    // stays the world's record of what its generators built. A generator the world baked for itself at start carries no locator: its
    // payload is in the runtime store with the rest of the trailing
    // `MeshBlock::Runtime` block, loaded after the compiled ones so a handle
    // the build assigned keeps its index.
    type Identified<C> = (Option<AssetId>, C);
    let (proc_meshes, baked_meshes): (
        Vec<Identified<ProceduralMesh>>,
        Vec<Identified<ProceduralMesh>>,
    ) = ctx
        .query_with_entity::<ProceduralMesh>()
        .map(|(entity, m)| (ctx.get::<Identity>(entity).map(|i| i.id()), m.clone()))
        .partition(|(_, m)| m.locator.is_some());
    let baked_payloads = ctx
        .resource::<crate::resource::RuntimeMeshPayloads>()
        .cloned()
        .unwrap_or_default();
    for (id, _) in &baked_meshes {
        if id.is_none_or(|id| baked_payloads.get(id).is_none()) {
            tracing::error!(
                "graphics: ProceduralMesh {id:?} was baked at start but left no payload"
            );
            return None;
        }
    }
    let voxel_chunks = ctx.drain_with_ids::<VoxelChunk>();
    let file_assets = ctx.drain_with_ids::<File>();
    let file_meshes: Vec<&Identified<File>> = file_assets
        .iter()
        .filter(|(_, f)| f.kind.as_ref().is_some_and(FileKind::is_mesh))
        .collect();

    if mesh_table.is_empty()
        && proc_meshes.is_empty()
        && baked_payloads.is_empty()
        && voxel_chunks.is_empty()
        && file_meshes.is_empty()
    {
        // Room path can carry the scene without any explicit Mesh/ProceduralMesh
        tracing::info!(
            "graphics: no Mesh, ProceduralMesh, VoxelChunk, or mesh-kind File sources found"
        );
    }

    let mut sink = MeshSink::new(deferred, blob_disk_backed);

    // Mesh block first: each resource-table entry lands at its own handle.
    for (handle, entry) in mesh_table.0.iter().enumerate() {
        let Some(locator) = &entry.payload else {
            tracing::error!(
                "graphics: Mesh handle {} has no compiled payload; did the build succeed?",
                handle
            );
            return None;
        };
        sink.push(ctx, locator, deferred.resource_bounds(handle), "Mesh")?;
    }

    sink.push_components(
        ctx,
        "ProceduralMesh",
        proc_meshes.iter().map(|(id, m)| (*id, m.locator.as_ref())),
    )?;
    sink.push_components(
        ctx,
        "VoxelChunk",
        voxel_chunks.iter().map(|(id, c)| (*id, c.locator.as_ref())),
    )?;
    sink.push_components(
        ctx,
        "File",
        file_meshes.iter().map(|(id, f)| (*id, f.locator.as_ref())),
    )?;

    // The world's own block: geometry baked at start, whose payload bytes are
    // already in memory rather than behind a locator, in install order.
    for (id, bytes) in baked_payloads.iter() {
        let handle = sink.geometry.len();
        sink.geometry
            .push(decode_loaded(bytes, &format!("baked mesh {id}"))?);
        sink.component_handles.insert(id, handle);
    }

    let MeshSink {
        geometry,
        deferred_payloads,
        component_handles,
        ..
    } = sink;

    Some(MeshGeometry {
        meshes: geometry,
        sources: mesh_sources,
        component_handles,
        deferred_seeds: deferred_payloads,
    })
}

// Decode one compiled mesh payload, logging a malformed one under `label`.
fn decode_loaded(bytes: &[u8], label: &str) -> Option<LoadedMesh> {
    match mesh_payload::deserialize_with_lods(bytes) {
        Ok((vertices, indices, lod_alternates)) => Some(LoadedMesh {
            vertices,
            indices,
            lod_alternates,
            bounds: None,
            counts: None,
        }),
        Err(e) => {
            tracing::error!("graphics: malformed {label} payload: {e}");
            None
        }
    }
}

// Read the payload behind `locator`, logging a failed read under `label`.
fn read_payload(
    ctx: &mut PipelineContext<'_>,
    locator: &PayloadLocator,
    label: &str,
) -> Option<Vec<u8>> {
    match ctx.read_payload(locator) {
        Ok(bytes) => Some(bytes.to_vec()),
        Err(e) => {
            tracing::error!("graphics: failed to read {label} payload: {e}");
            None
        }
    }
}

// The dense geometry table `load_mesh_geometry` fills in handle order, plus the
// payload seeds of the meshes it deferred and the handles of the mesh sources
// that are still components.
struct MeshSink<'a> {
    geometry: Vec<LoadedMesh>,
    deferred_payloads: hashbrown::HashMap<usize, DeferredMeshSeed>,
    component_handles: hashbrown::HashMap<AssetId, usize>,
    deferred: &'a DeferredMeshSources,
    blob_disk_backed: bool,
}

impl<'a> MeshSink<'a> {
    fn new(deferred: &'a DeferredMeshSources, blob_disk_backed: bool) -> Self {
        Self {
            geometry: Vec::new(),
            deferred_payloads: hashbrown::HashMap::new(),
            component_handles: hashbrown::HashMap::new(),
            deferred,
            blob_disk_backed,
        }
    }

    // Push the mesh behind `locator` at the next handle and return that handle.
    // Baked `bounds` defer its decode to the streamer, keeping the payload bytes
    // only when the blob is not disk-backed; without them it decodes now.
    fn push(
        &mut self,
        ctx: &mut PipelineContext<'_>,
        locator: &PayloadLocator,
        bounds: Option<([f32; 3], [f32; 3])>,
        label: &str,
    ) -> Option<usize> {
        let handle = self.geometry.len();
        let mesh = match bounds {
            None => decode_loaded(&read_payload(ctx, locator, label)?, label)?,
            Some(bounds) => {
                let bytes = if self.blob_disk_backed {
                    None
                } else {
                    Some(read_payload(ctx, locator, label)?)
                };
                self.deferred_payloads.insert(
                    handle,
                    DeferredMeshSeed {
                        locator: locator.clone(),
                        bytes,
                    },
                );
                LoadedMesh {
                    vertices: Vec::new(),
                    indices: Vec::new(),
                    lod_alternates: Vec::new(),
                    bounds: Some(bounds),
                    counts: self
                        .deferred
                        .counts
                        .get(&u32::saturating_from(handle))
                        .copied(),
                }
            }
        };
        self.geometry.push(mesh);
        Some(handle)
    }

    // Push one component-backed mesh block, recording each asset id against its
    // handle: its push position, since blocks load in cook's block order and each
    // iterates in declaration order.
    fn push_components<'m>(
        &mut self,
        ctx: &mut PipelineContext<'_>,
        label: &str,
        items: impl Iterator<Item = (Option<AssetId>, Option<&'m PayloadLocator>)>,
    ) -> Option<()> {
        for (i, (id, locator)) in items.enumerate() {
            let (Some(id), Some(locator)) = (id, locator) else {
                tracing::error!("graphics: {label}[{i}] {id:?} has no compiled payload");
                return None;
            };
            let bounds = self.deferred.def_bounds(id, self.geometry.len());
            let handle = self.push(ctx, locator, bounds, label)?;
            self.component_handles.insert(id, handle);
        }
        Some(())
    }
}

/// Decode all Room mesh payloads and collect blob indices for the release step.
/// Returns None if any payload is missing or malformed (error already logged).
pub fn load_room_geometry(ctx: &mut PipelineContext<'_>) -> Option<(Vec<RoomGeometry>, Vec<u32>)> {
    let rooms = ctx.drain_with_ids::<Room>();
    let mut room_geometry: Vec<RoomGeometry> = Vec::new();
    let mut blob_indices: Vec<u32> = Vec::new();

    for (i, (id, room)) in rooms.into_iter().enumerate() {
        let locator = if let Some(l) = &room.locator {
            l.clone()
        } else {
            tracing::error!(
                "graphics: Room[{}] {:?} has no compiled payload; did the build succeed?",
                i,
                id
            );
            return None;
        };
        blob_indices.push(locator.blob_index);
        let bytes = match ctx.read_payload(&locator) {
            Ok(b) => b.to_vec(),
            Err(e) => {
                tracing::error!("graphics: failed to read Room {:?} payload: {}", id, e);
                return None;
            }
        };
        match mesh_payload::deserialize_with_lods(&bytes) {
            Ok((verts, idxs, alternates)) => room_geometry.push((room, verts, idxs, alternates)),
            Err(e) => {
                tracing::error!("graphics: malformed Room payload: {}", e);
                return None;
            }
        }
    }

    Some((room_geometry, blob_indices))
}

/// The read-only scene lookup tables consumed by [`build_draw_list`]: the
/// renderable items and instanced props plus every catalog needed to resolve
/// their geometry, textures, and materials.
#[derive(Clone, Copy)]
pub struct DrawListInputs<'a> {
    /// The placements to draw.
    pub items: &'a [RenderableItem],
    /// The GPU-instanced props, one cluster each.
    pub instanced_props: &'a [InstancedProp],
    /// Each item's world matrix, column-aligned with `items`.
    pub world_mats: &'a [[[f32; 4]; 4]],
    /// Each Model's sub-meshes by asset id.
    pub model_map: &'a hashbrown::HashMap<AssetId, Vec<SubMeshRef>>,
    /// Dense mesh-source geometry from `load_mesh_geometry`: a `.mesh`
    /// reference's `MeshHandle` indexes it directly.
    pub mesh_geometry: &'a [LoadedMesh],
    /// Every room's decoded geometry.
    pub room_geometry: &'a [RoomGeometry],
    /// Every terrain, appended as static culled chunks.
    pub terrains: &'a [crate::render::environment::terrain::LoadedTerrain],
    /// Size of the shared texture pool; a texture handle is in range when its
    /// index is below this. A legacy texture-on-mesh reference past it falls back
    /// to slot 0.
    pub texture_count: usize,
    /// Every baked material by handle.
    pub material_map: &'a hashbrown::HashMap<MaterialHandle, MaterialEntry>,
}

/// Assemble the shared vertex/index buffers and per-object draw records from all
/// scene geometry (props, rooms, terrains). A mesh nothing places draws
/// nothing. Also returns the per-prop draw-index table for runtime model-matrix
/// updates and the GPU-instanced cluster list (one entry per `InstancedProp`).
/// Returns None if any referenced asset is missing (error already logged).
pub fn build_draw_list(inputs: DrawListInputs<'_>) -> Option<DrawListData> {
    let DrawListInputs {
        items,
        instanced_props,
        world_mats,
        model_map,
        mesh_geometry,
        room_geometry,
        terrains,
        texture_count,
        material_map,
    } = inputs;
    let mut all_vertices: Vec<Vertex> = Vec::new();
    let mut all_indices: Vec<u32> = Vec::new();
    let mut draw_objects: Vec<DrawObject> = Vec::new();
    let mut instanced_clusters: Vec<InstancedCluster> = Vec::new();
    let mut prop_draw_indices: Vec<Vec<DrawIndex>> = Vec::new();
    let mut prop_local_bounds: Vec<([f32; 3], [f32; 3])> = Vec::new();
    // Map every mesh-source handle to the draw slots that received a copy of
    // its geometry. Hot-reload (`cn debug` only) walks this to know which slots
    // to overwrite when the source `.glb` changes. The slot list accumulates
    // every push since a mesh shared by N `Prop`s yields N independent draw
    // objects.
    let mut mesh_handle_to_draws: hashbrown::HashMap<usize, Vec<DrawIndex>> =
        hashbrown::HashMap::new();

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
        let vertex_byte_offset = all_vertices.len() * size_of::<Vertex>();
        let index_elem_offset = all_indices.len();
        let base = u32::saturating_from(all_vertices.len());
        let (bb_min, bb_max) = loaded
            .bounds
            .unwrap_or_else(|| local_bounds(&loaded.vertices));
        all_vertices.extend_from_slice(&loaded.vertices);
        all_indices.extend(loaded.indices.iter().map(|i| u32::from(*i) + base));
        let lod_slices = append_lod_slices(&mut all_indices, &loaded.lod_alternates, base);
        // A deferred mesh appended no bytes; its draw record carries the baked
        // counts so the streamed upload's size check matches the real geometry.
        let (vertex_count, index_count) = loaded
            .counts
            .map_or((loaded.vertices.len(), loaded.indices.len()), |(v, i)| {
                (v as usize, i as usize)
            });
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
        let mut prop_idxs: Vec<DrawIndex> = Vec::new();
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
            let Some(submeshes) = model_map.get(&model_id) else {
                tracing::error!(
                    "graphics: Prop {:?} references unknown model {}; add a Model asset with that id",
                    item.asset_id,
                    model_id
                );
                return None;
            };
            for sub in submeshes {
                let sub_mesh = if let Some(m) = sub.mesh {
                    m.index()
                } else {
                    tracing::error!("graphics: Model {} has a sub-mesh with no mesh", model_id);
                    return None;
                };
                let Some((
                    vertex_offset,
                    vertex_count,
                    index_offset,
                    index_count,
                    lod_alternates,
                    local_min,
                    local_max,
                )) = append_mesh(sub_mesh)
                else {
                    tracing::error!(
                        "graphics: Model {} sub-mesh handle {} out of range; add a Mesh or ProceduralMesh asset with that name",
                        model_id,
                        sub_mesh
                    );
                    return None;
                };
                let mat_entry = match resolve_material_slots(sub.material, material_map) {
                    Ok(entry) => entry,
                    Err(mat_id) => {
                        tracing::error!(
                            "graphics: Model {} sub-mesh material {} not found",
                            model_id,
                            mat_id.index()
                        );
                        return None;
                    }
                };
                let (bb_min, bb_max, local_bounds) =
                    culled_bounds(item, local_min, local_max, model_mat);
                union_local(local_min, local_max);
                let slot = DrawIndex::from_usize(draw_objects.len());
                prop_idxs.push(slot);
                mesh_handle_to_draws.entry(sub_mesh).or_default().push(slot);
                draw_objects.push(DrawObject {
                    vertex_offset,
                    vertex_count,
                    index_offset,
                    index_count,
                    planet_tile: None,
                    // Static geometry: indices are absolute into the shared
                    // vertex buffer, so no per-draw base.
                    base_vertex: 0,
                    geometry_generation: 0,
                    model: model_mat,
                    texture_slot: mat_entry.albedo_slot,
                    normal_map_slot: mat_entry.normal_map_slot,
                    material: mat_entry.uniforms,
                    shader_bucket: mat_entry.shader_bucket,
                    visible: true,
                    resident: true,
                    bb_min,
                    bb_max,
                    local_bounds,
                    cull_distance: item.cull_distance,
                    lod_alternates,
                });
            }
        } else {
            // single-mesh path
            let mesh_handle = if let Some(m) = item.mesh {
                m.index()
            } else {
                tracing::error!(
                    "graphics: Prop {:?} has neither a model nor a mesh",
                    item.asset_id
                );
                return None;
            };
            let Some((
                vertex_offset,
                vertex_count,
                index_offset,
                index_count,
                lod_alternates,
                local_min,
                local_max,
            )) = append_mesh(mesh_handle)
            else {
                tracing::error!(
                    "graphics: Prop {:?} references out-of-range mesh handle {}; add a Mesh or ProceduralMesh asset with that name",
                    item.asset_id,
                    mesh_handle
                );
                return None;
            };
            let mat_entry = match resolve_material_slots(item.material, material_map) {
                Ok(entry) => entry,
                Err(mat_id) => {
                    tracing::error!(
                        "graphics: Prop {:?} references unknown material {}; add a Material asset with that id",
                        item.asset_id,
                        mat_id.index()
                    );
                    return None;
                }
            };
            let (bb_min, bb_max, local_bounds) =
                culled_bounds(item, local_min, local_max, model_mat);
            union_local(local_min, local_max);
            let slot = DrawIndex::from_usize(draw_objects.len());
            prop_idxs.push(slot);
            mesh_handle_to_draws
                .entry(mesh_handle)
                .or_default()
                .push(slot);
            draw_objects.push(DrawObject {
                vertex_offset,
                vertex_count,
                index_offset,
                index_count,
                planet_tile: None,
                base_vertex: 0,
                geometry_generation: 0,
                model: model_mat,
                texture_slot: mat_entry.albedo_slot,
                normal_map_slot: mat_entry.normal_map_slot,
                material: mat_entry.uniforms,
                shader_bucket: mat_entry.shader_bucket,
                visible: true,
                resident: true,
                bb_min,
                bb_max,
                local_bounds,
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
    for (cluster, inst) in instanced_props.iter().enumerate() {
        let mesh_handle = match inst.mesh {
            Some(m) if !inst.instances.is_empty() => m.index(),
            _ => continue,
        };
        // Instanced clusters carry the mesh's LOD alternates; each instance
        // picks its slice by camera distance in its cull record (see
        // `lod::for_each_instance_lod`).
        let Some((
            vertex_offset,
            vertex_count,
            index_offset,
            index_count,
            lod_alternates,
            local_min,
            local_max,
        )) = append_mesh(mesh_handle)
        else {
            tracing::error!(
                "graphics: InstancedProp[{}] references out-of-range mesh handle {}",
                cluster,
                mesh_handle
            );
            return None;
        };
        let mat_entry = match resolve_material_slots(inst.material, material_map) {
            Ok(entry) => entry,
            Err(mat_id) => {
                tracing::error!(
                    "graphics: InstancedProp[{}] references unknown material {}",
                    cluster,
                    mat_id.index()
                );
                return None;
            }
        };
        let (texture_slot, normal_map_slot, material) = (
            mat_entry.albedo_slot,
            mat_entry.normal_map_slot,
            mat_entry.uniforms,
        );

        let mut instance_mats: Vec<[[f32; 4]; 4]> = Vec::with_capacity(inst.instances.len());
        let mut cluster_min = [f32::INFINITY; 3];
        let mut cluster_max = [f32::NEG_INFINITY; 3];
        for i in 0..inst.instances.len() {
            let Some(model_mat) = inst.instance_model_matrix(i) else {
                continue;
            };
            let (bb_min, bb_max) = frustum::transform_aabb(local_min, local_max, model_mat);
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

    // Room components placed at the world origin with optional texture.
    // Rooms also opt out of culling (they enclose the camera). LOD picks
    // come from camera-to-origin distance per [`crate::gfx::lod::camera_distance`]'s
    // sentinel-AABB fallback, so practical swaps only fire if the camera
    // wanders far from the world origin.
    for (room, verts, idxs, room_lods) in room_geometry {
        let vertex_byte_offset = all_vertices.len() * size_of::<Vertex>();
        let index_elem_offset = all_indices.len();
        let base = u32::saturating_from(all_vertices.len());
        all_vertices.extend_from_slice(verts);
        all_indices.extend(idxs.iter().map(|i| u32::from(*i) + base));
        let lod_slices = append_lod_slices(&mut all_indices, room_lods, base);
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
            planet_tile: None,
            base_vertex: 0,
            geometry_generation: 0,
            model: IDENTITY4,
            texture_slot,
            normal_map_slot: NO_NORMAL_MAP_SLOT,
            material: MaterialUniforms::DEFAULT,
            shader_bucket: 0,
            visible: true,
            resident: true,
            bb_min: UNCULLED_BB.0,
            bb_max: UNCULLED_BB.1,
            local_bounds: None,
            cull_distance: 0.0,
            lod_alternates: lod_slices,
        });
    }

    for terrain in terrains {
        let material = match resolve_material_slots(terrain.terrain.material, material_map) {
            Ok(entry) => entry,
            Err(mat_id) => {
                tracing::error!(
                    "graphics: a Terrain references unknown material {}; add a Material asset with that id",
                    mat_id.index()
                );
                return None;
            }
        };
        crate::render::environment::terrain::append_terrain_draws(
            terrain,
            material,
            &mut all_vertices,
            &mut all_indices,
            &mut draw_objects,
        );
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
    use crate::components::{InstanceTransform, Prop};
    use crate::ecs::NameRef;
    use crate::ecs::{ComponentSlot, PayloadStore, Ref, TextureHandle, World};
    use crate::error::PayloadError;
    use crate::gfx::render_types::NO_ALBEDO_SLOT;
    use crate::resource::ResourceEntry;
    use alloc::boxed::Box;
    use alloc::string::ToString;
    use alloc::vec;

    // One resident payload section, read by locator range.
    struct OneSection(Vec<u8>);

    impl PayloadStore for OneSection {
        fn read(&mut self, locator: &PayloadLocator) -> Result<&[u8], PayloadError> {
            let start = usize::saturating_from(locator.offset);
            let end = start + usize::saturating_from(locator.len);
            self.0.get(start..end).ok_or(PayloadError::OutOfBounds {
                index: locator.blob_index,
                offset: locator.offset,
                len: locator.len,
                section_len: self.0.len(),
            })
        }

        fn release(&mut self, _blob_index: u32) {}

        fn disk_backed(&self) -> bool {
            false
        }
    }

    fn make_prop(position: [f32; 3]) -> Prop {
        Prop {
            model: None,
            mesh: None,
            material: None,
            position,
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            collider: None,
            interactable: false,
            pickup: false,
            parent: None,
            scene: None,
            prefab: NameRef::default(),
            cull_distance: 0.0,
            is_held: false,
        }
    }

    fn unit_quad_mesh() -> LoadedMesh {
        // Axis-aligned unit cube centered at origin; bounds = [-0.5, 0.5]^3.
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

        let inst = InstancedProp {
            mesh: Some(MeshHandle::new(0)),
            material: None,
            cull_distance: 0.0,
            instances: vec![
                InstanceTransform {
                    position: [0.0, 0.0, 0.0],
                    rotation_deg: [0.0; 3],
                    scale: [1.0; 3],
                },
                InstanceTransform {
                    position: [5.0, 0.0, 0.0],
                    rotation_deg: [0.0; 3],
                    scale: [1.0; 3],
                },
                InstanceTransform {
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
            model_map: &hashbrown::HashMap::new(),
            mesh_geometry: &mesh_geometry,
            room_geometry: &[],
            terrains: &[],
            texture_count: 0,
            material_map: &hashbrown::HashMap::new(),
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

        let inst = InstancedProp {
            mesh: Some(MeshHandle::new(0)),
            material: None,
            cull_distance: 0.0,
            instances: Vec::new(),
        };

        let data = build_draw_list(DrawListInputs {
            items: &[],
            instanced_props: &[inst],
            world_mats: &[],
            model_map: &hashbrown::HashMap::new(),
            mesh_geometry: &mesh_geometry,
            room_geometry: &[],
            terrains: &[],
            texture_count: 0,
            material_map: &hashbrown::HashMap::new(),
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

    // The item built from a mesh entity's components reads the renderer fields
    // from MeshRenderer and marks the entity dynamic from its Pickup / Collider
    // tags.
    #[test]
    fn decomposed_renderable_item_matches_a_mesh_prop() {
        use crate::components::{Collider, MeshRenderer, Pickup, PropCollider};

        let mut prop = make_prop([0.0; 3]);
        prop.mesh = Some(MeshHandle::new(10));
        prop.material = Some(MaterialHandle::new(20));
        prop.cull_distance = 50.0;
        prop.pickup = true;
        prop.collider = Some(PropCollider::default());

        let mut world = World::new();
        let mut ctx = world.context();

        let e = ctx.components.spawn();
        ctx.insert(
            e,
            MeshRenderer {
                mesh: prop.mesh,
                material: prop.material,
                cull_distance: prop.cull_distance,
            },
        );
        ctx.insert(e, Pickup);
        ctx.insert(e, Collider(prop.collider.unwrap()));

        ctx.identify(e, AssetId(7));
        let item = decomposed_renderable_item(&ctx, e);
        assert_eq!(
            item,
            RenderableItem {
                asset_id: Some(AssetId(7)),
                model: None,
                mesh: Some(MeshHandle::new(10)),
                material: Some(MaterialHandle::new(20)),
                cull_distance: 50.0,
                is_dynamic: true,
            }
        );
    }

    // A planet world's frame moves are rigid, so its placements stay static
    // and keep the model-space box their culling box follows.
    #[test]
    fn a_planet_world_keeps_placements_static() {
        use crate::components::{MeshRenderer, Planet};
        let mut world = World::new();
        let e = world.push(MeshRenderer::default());
        world.push(Planet::default());
        let item = decomposed_renderable_item(&world.context(), e);
        assert!(!item.is_dynamic);
        let model = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [5.0, 0.0, 0.0, 1.0],
        ];
        let (lo, hi, local) = culled_bounds(&item, [-1.0; 3], [1.0; 3], model);
        assert_eq!((lo, hi), ([4.0, -1.0, -1.0], [6.0, 1.0, 1.0]));
        assert_eq!(local, Some(([-1.0; 3], [1.0; 3])));
        let dynamic = RenderableItem {
            is_dynamic: true,
            ..item
        };
        let (lo, _, local) = culled_bounds(&dynamic, [-1.0; 3], [1.0; 3], model);
        assert!(lo[0].is_nan() && local.is_none());
    }

    fn mesh_item(mesh: AssetId) -> RenderableItem {
        RenderableItem {
            asset_id: Some(mesh),
            model: None,
            // A `.mesh` handle indexes the dense geometry slice directly, so a
            // test item's handle is the geometry index it draws.
            mesh: Some(MeshHandle::new(mesh.0)),
            material: None,
            cull_distance: 0.0,
            is_dynamic: false,
        }
    }

    fn model_item(model: AssetId) -> RenderableItem {
        RenderableItem {
            asset_id: Some(model),
            model: Some(model),
            mesh: None,
            material: None,
            cull_distance: 0.0,
            is_dynamic: false,
        }
    }

    // The model path emits one draw object per sub-mesh, each over its own
    // geometry region, and records both draws under the shared prop index.
    #[test]
    fn build_draw_list_model_emits_one_draw_per_submesh() {
        let mesh_geometry = vec![unit_quad_mesh(), unit_quad_mesh()];

        let mut model_map = hashbrown::HashMap::new();
        model_map.insert(
            AssetId(1),
            vec![
                SubMeshRef {
                    mesh: Some(MeshHandle::new(0)),
                    material: Some(MaterialHandle::new(20)),
                },
                SubMeshRef {
                    mesh: Some(MeshHandle::new(1)),
                    material: None,
                },
            ],
        );

        let mut material_map = hashbrown::HashMap::new();
        material_map.insert(
            MaterialHandle::new(20),
            MaterialEntry {
                albedo_slot: 3,
                normal_map_slot: 4,
                uniforms: MaterialUniforms::DEFAULT,
                shader_bucket: 0,
            },
        );

        let data = build_draw_list(DrawListInputs {
            items: &[model_item(AssetId(1))],
            instanced_props: &[],
            world_mats: &[IDENTITY4],
            model_map: &model_map,
            mesh_geometry: &mesh_geometry,
            room_geometry: &[],
            terrains: &[],
            texture_count: 0,
            material_map: &material_map,
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
        assert_eq!(prop_idxs, vec![vec![DrawIndex(0), DrawIndex(1)]]);
        assert_eq!(verts.len(), 8, "each quad's 4 verts appended once");
        assert_eq!(idxs.len(), 12);
        // First sub-mesh took its material's albedo/normal slots; the second
        // used the default (white, flat normal).
        assert_eq!(draw_objects[0].texture_slot, 3);
        assert_eq!(draw_objects[0].normal_map_slot, 4);
        assert_eq!(draw_objects[1].texture_slot, NO_ALBEDO_SLOT);
        assert_eq!(draw_objects[1].normal_map_slot, NO_NORMAL_MAP_SLOT);
        // Hot-reload map tracks each sub-mesh id -> its draw slot.
        assert_eq!(mesh_handle_to_draws.get(&0), Some(&vec![DrawIndex(0)]));
        assert_eq!(mesh_handle_to_draws.get(&1), Some(&vec![DrawIndex(1)]));
    }

    // A mesh present in the geometry table but placed by no item, model, or
    // instanced prop draws nothing and takes no room in the shared buffers.
    #[test]
    fn build_draw_list_skips_an_unplaced_mesh() {
        let mesh_geometry = vec![unit_quad_mesh()];

        let data = build_draw_list(DrawListInputs {
            items: &[],
            instanced_props: &[],
            world_mats: &[],
            model_map: &hashbrown::HashMap::new(),
            mesh_geometry: &mesh_geometry,
            room_geometry: &[],
            terrains: &[],
            texture_count: 0,
            material_map: &hashbrown::HashMap::new(),
        })
        .expect("build_draw_list");

        assert!(data.draw_objects.is_empty());
        assert!(data.vertices.is_empty() && data.indices.is_empty());
        assert!(data.mesh_handle_to_draws.is_empty());
    }

    // A Room is placed at the origin with culling disabled; its cook-assigned
    // texture handle is used directly as its albedo pool slot and its LOD
    // alternates carry through.
    #[test]
    fn build_draw_list_places_room_at_origin_with_texture_and_lods() {
        let room = Room {
            half_width: 8.0,
            half_depth: 10.0,
            ceiling_height: 3.5,
            texture: Some(TextureHandle::new(6)),
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
            model_map: &hashbrown::HashMap::new(),
            mesh_geometry: &[],
            room_geometry: &room_geometry,
            terrains: &[],
            texture_count: 7,
            material_map: &hashbrown::HashMap::new(),
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

    // A single-mesh item naming no material binds the reserved albedo and
    // normal-map fallbacks under the default material.
    #[test]
    fn build_draw_list_single_mesh_without_a_material_is_untextured() {
        let mesh_geometry = vec![unit_quad_mesh()];
        // The texture handle is the pool slot directly; the pool size (3) makes
        // slot 2 in range.
        let item = RenderableItem {
            asset_id: None,
            model: None,
            mesh: Some(MeshHandle::new(0)),
            material: None,
            cull_distance: 0.0,
            is_dynamic: false,
        };

        let data = build_draw_list(DrawListInputs {
            items: &[item],
            instanced_props: &[],
            world_mats: &[IDENTITY4],
            model_map: &hashbrown::HashMap::new(),
            mesh_geometry: &mesh_geometry,
            room_geometry: &[],
            terrains: &[],
            texture_count: 3,
            material_map: &hashbrown::HashMap::new(),
        })
        .expect("build_draw_list");
        let DrawListData { draw_objects, .. } = data;

        assert_eq!(draw_objects.len(), 1);
        assert_eq!(draw_objects[0].texture_slot, NO_ALBEDO_SLOT);
        assert_eq!(draw_objects[0].normal_map_slot, NO_NORMAL_MAP_SLOT);
    }

    // Every missing-reference branch returns None (the error is logged and the
    // build aborts) rather than emitting a partial draw list.
    #[test]
    fn build_draw_list_returns_none_on_missing_references() {
        let mesh = || vec![unit_quad_mesh()];
        let none = |inputs: DrawListInputs<'_>| build_draw_list(inputs).is_none();

        // Model referenced by an item but absent from the model_map.
        assert!(none(DrawListInputs {
            items: &[model_item(AssetId(1))],
            instanced_props: &[],
            world_mats: &[IDENTITY4],
            model_map: &hashbrown::HashMap::new(),
            mesh_geometry: &mesh(),
            room_geometry: &[],
            terrains: &[],
            texture_count: 0,
            material_map: &hashbrown::HashMap::new(),
        }));

        // Model sub-mesh with no mesh field.
        let mut model_no_mesh = hashbrown::HashMap::new();
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
            terrains: &[],
            texture_count: 0,
            material_map: &hashbrown::HashMap::new(),
        }));

        // Model sub-mesh whose mesh id has no geometry.
        let mut model_bad_geo = hashbrown::HashMap::new();
        model_bad_geo.insert(
            AssetId(1),
            vec![SubMeshRef {
                mesh: Some(MeshHandle::new(999)),
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
            terrains: &[],
            texture_count: 0,
            material_map: &hashbrown::HashMap::new(),
        }));

        // Model sub-mesh referencing a material absent from the material_map.
        let mut model_bad_mat = hashbrown::HashMap::new();
        model_bad_mat.insert(
            AssetId(1),
            vec![SubMeshRef {
                mesh: Some(MeshHandle::new(0)),
                material: Some(MaterialHandle::new(404)),
            }],
        );
        assert!(none(DrawListInputs {
            items: &[model_item(AssetId(1))],
            instanced_props: &[],
            world_mats: &[IDENTITY4],
            model_map: &model_bad_mat,
            mesh_geometry: &mesh(),
            room_geometry: &[],
            terrains: &[],
            texture_count: 0,
            material_map: &hashbrown::HashMap::new(),
        }));

        // Single-mesh item whose mesh id has no geometry.
        assert!(none(DrawListInputs {
            items: &[mesh_item(AssetId(999))],
            instanced_props: &[],
            world_mats: &[IDENTITY4],
            model_map: &hashbrown::HashMap::new(),
            mesh_geometry: &mesh(),
            room_geometry: &[],
            terrains: &[],
            texture_count: 0,
            material_map: &hashbrown::HashMap::new(),
        }));

        // Single-mesh item referencing a material absent from the material_map.
        let mut item_bad_mat = mesh_item(AssetId(0));
        item_bad_mat.material = Some(MaterialHandle::new(404));
        assert!(none(DrawListInputs {
            items: &[item_bad_mat],
            instanced_props: &[],
            world_mats: &[IDENTITY4],
            model_map: &hashbrown::HashMap::new(),
            mesh_geometry: &mesh(),
            room_geometry: &[],
            terrains: &[],
            texture_count: 0,
            material_map: &hashbrown::HashMap::new(),
        }));

        // Item carrying neither a model nor a mesh.
        assert!(none(DrawListInputs {
            items: &[RenderableItem {
                asset_id: None,
                model: None,
                mesh: None,
                material: None,
                cull_distance: 0.0,
                is_dynamic: false,
            }],
            instanced_props: &[],
            world_mats: &[IDENTITY4],
            model_map: &hashbrown::HashMap::new(),
            mesh_geometry: &mesh(),
            room_geometry: &[],
            terrains: &[],
            texture_count: 0,
            material_map: &hashbrown::HashMap::new(),
        }));

        // InstancedProp mesh id has no geometry.
        let inst_bad_mesh = InstancedProp {
            mesh: Some(MeshHandle::new(999)),
            material: None,
            cull_distance: 0.0,
            instances: vec![InstanceTransform::default()],
        };
        assert!(none(DrawListInputs {
            items: &[],
            instanced_props: &[inst_bad_mesh],
            world_mats: &[],
            model_map: &hashbrown::HashMap::new(),
            mesh_geometry: &mesh(),
            room_geometry: &[],
            terrains: &[],
            texture_count: 0,
            material_map: &hashbrown::HashMap::new(),
        }));

        // InstancedProp material absent from the material_map.
        let inst_bad_mat = InstancedProp {
            mesh: Some(MeshHandle::new(0)),
            material: Some(MaterialHandle::new(404)),
            cull_distance: 0.0,
            instances: vec![InstanceTransform::default()],
        };
        assert!(none(DrawListInputs {
            items: &[],
            instanced_props: &[inst_bad_mat],
            world_mats: &[],
            model_map: &hashbrown::HashMap::new(),
            mesh_geometry: &mesh(),
            room_geometry: &[],
            terrains: &[],
            texture_count: 0,
            material_map: &hashbrown::HashMap::new(),
        }));
    }

    // Accumulates components + a single blob section so load_mesh_geometry /
    // load_room_geometry can decode in-memory payloads. The components are
    // added once the section is sealed into the world's payload store.
    struct BlobWorld {
        section: Vec<u8>,
        pending: Vec<PendingAdd>,
    }

    // A component add held until `seal` has built the world it goes into.
    type PendingAdd = Box<dyn FnOnce(&mut World)>;

    impl BlobWorld {
        fn new() -> Self {
            Self {
                section: Vec::new(),
                pending: Vec::new(),
            }
        }

        fn payload(&mut self, bytes: &[u8]) -> PayloadLocator {
            let offset = self.section.len() as u64;
            self.section.extend_from_slice(bytes);
            PayloadLocator {
                blob_index: 0,
                offset,
                len: bytes.len() as u64,
            }
        }

        fn push<C: ComponentSlot>(&mut self, c: C) {
            self.pending.push(Box::new(move |world| {
                world.push(c);
            }));
        }

        fn push_identified<C: ComponentSlot>(&mut self, id: AssetId, c: C) {
            self.pending.push(Box::new(move |world| {
                world.push_identified(id, c);
            }));
        }

        fn seal(self) -> World {
            let mut world = World::from_payloads(Box::new(OneSection(self.section)));
            for add in self.pending {
                add(&mut world);
            }
            world
        }
    }

    // Install a `MeshTable` with one entry per locator (handle == index),
    // standing in for the blob resource stream a real build provides.
    fn with_mesh_table(mut world: World, locators: Vec<Option<PayloadLocator>>) -> World {
        let entries = locators
            .into_iter()
            .map(|payload| ResourceEntry {
                payload,
                ..Default::default()
            })
            .collect();
        world.insert_resource(MeshTable(entries));
        world
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
        mesh_payload::serialize(&[v(0.0, 0.0), v(1.0, 0.0), v(0.0, 1.0)], &[0u16, 1, 2])
    }

    // load_mesh_geometry decodes a MeshTable entry's in-memory payload into the
    // dense geometry table at its handle.
    #[test]
    fn load_mesh_geometry_decodes_in_memory_mesh() {
        let mut b = BlobWorld::new();
        let loc = b.payload(&tri_payload());
        let mut world = with_mesh_table(b.seal(), vec![Some(loc)]);
        let mut ctx = world.context();

        let MeshGeometry {
            meshes: geometry,
            sources,
            component_handles,
            ..
        } = load_mesh_geometry(&mut ctx, &DeferredMeshSources::default(), false, false)
            .expect("decoded");
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
    }

    // A deferred Mesh skips its decode: empty geometry with the baked bounds,
    // and its payload seed (locator + RAM bytes) is captured for the streamer.
    #[test]
    fn load_mesh_geometry_defers_scene_owned_mesh_with_baked_bounds() {
        let mut b = BlobWorld::new();
        let loc = b.payload(&tri_payload());
        let mut world = with_mesh_table(b.seal(), vec![Some(loc)]);
        let mut ctx = world.context();

        let mut deferred = DeferredMeshSources::default();
        deferred.by_handle.insert(0);
        deferred
            .bounds
            .insert(0, ([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]));

        let MeshGeometry {
            meshes: geometry,
            deferred_seeds: seeds,
            ..
        } = load_mesh_geometry(&mut ctx, &deferred, false, false).expect("ok");
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
        let mut world = with_mesh_table(b.seal(), vec![Some(loc)]);
        let mut ctx = world.context();

        let mut deferred = DeferredMeshSources::default();
        deferred.by_handle.insert(0);

        let MeshGeometry {
            meshes: geometry,
            deferred_seeds: seeds,
            ..
        } = load_mesh_geometry(&mut ctx, &deferred, false, false).expect("ok");
        assert_eq!(geometry[0].vertices.len(), 3, "no bounds record -> decode");
        assert!(seeds.is_empty());
    }

    // Geometry the world baked for itself at start loads from the payload map
    // rather than a locator, and lands past every handle the build assigned.
    #[test]
    fn load_mesh_geometry_loads_the_baked_block_last() {
        let mut b = BlobWorld::new();
        let compiled = b.payload(&tri_payload());
        // One compiled ProceduralMesh (handle 1, after the Mesh block) and one
        // baked at start.
        b.push_identified(
            AssetId(2),
            ProceduralMesh {
                generator: "box".to_string(),
                locator: Some(compiled.clone()),
                ..Default::default()
            },
        );
        b.push_identified(
            AssetId(3),
            ProceduralMesh {
                generator: "sphere".to_string(),
                ..Default::default()
            },
        );
        let mut world = with_mesh_table(b.seal(), vec![Some(compiled)]);
        let mut payloads = crate::resource::RuntimeMeshPayloads::default();
        payloads.push(AssetId(3), tri_payload());
        world.insert_resource(payloads);
        let mut ctx = world.context();

        let MeshGeometry {
            meshes: geometry,
            component_handles,
            ..
        } = load_mesh_geometry(&mut ctx, &DeferredMeshSources::default(), false, false)
            .expect("decoded");
        assert_eq!(geometry.len(), 3);
        assert_eq!(component_handles.get(&AssetId(2)), Some(&1));
        assert_eq!(
            component_handles.get(&AssetId(3)),
            Some(&2),
            "the baked mesh trails the compiled blocks"
        );
        assert_eq!(geometry[2].vertices.len(), 3, "the baked payload decoded");
    }

    // A mesh baked at start whose payload never reached the map aborts the
    // load, the way a missing compiled payload does.
    #[test]
    fn load_mesh_geometry_baked_mesh_without_a_payload_returns_none() {
        let mut b = BlobWorld::new();
        b.push_identified(
            AssetId(3),
            ProceduralMesh {
                generator: "sphere".to_string(),
                ..Default::default()
            },
        );
        let mut world = b.seal();
        let mut ctx = world.context();
        assert!(
            load_mesh_geometry(&mut ctx, &DeferredMeshSources::default(), false, false).is_none()
        );
    }

    // A Mesh with no compiled payload aborts the whole load.
    #[test]
    fn load_mesh_geometry_missing_locator_returns_none() {
        let mut world = with_mesh_table(BlobWorld::new().seal(), vec![None]);
        let mut ctx = world.context();
        assert!(
            load_mesh_geometry(&mut ctx, &DeferredMeshSources::default(), false, false).is_none()
        );
    }

    // A malformed payload (too short to hold its declared vertices) aborts.
    #[test]
    fn load_mesh_geometry_malformed_payload_returns_none() {
        let mut b = BlobWorld::new();
        // Claims one vertex but carries no vertex bytes.
        let loc = b.payload(&1u32.to_le_bytes());
        let mut world = with_mesh_table(b.seal(), vec![Some(loc)]);
        let mut ctx = world.context();
        assert!(
            load_mesh_geometry(&mut ctx, &DeferredMeshSources::default(), false, false).is_none()
        );
    }

    // An empty world (no mesh sources at all) still succeeds with empty maps.
    #[test]
    fn load_mesh_geometry_empty_world_is_ok_and_empty() {
        let mut world = BlobWorld::new().seal();
        let mut ctx = world.context();
        let MeshGeometry {
            meshes: geometry,
            sources,
            component_handles,
            ..
        } = load_mesh_geometry(&mut ctx, &DeferredMeshSources::default(), false, false)
            .expect("ok");
        assert!(geometry.is_empty() && sources.is_empty());
        assert!(component_handles.is_empty());
    }

    // A valid payload decodes with no baked bounds or counts attached.
    #[test]
    fn decode_loaded_decodes_a_valid_payload() {
        let mesh = decode_loaded(&tri_payload(), "Mesh").expect("decoded");
        assert_eq!(mesh.vertices.len(), 3);
        assert_eq!(mesh.indices, vec![0, 1, 2]);
        assert!(mesh.bounds.is_none() && mesh.counts.is_none());
    }

    // Bytes too short to hold their declared vertices do not decode.
    #[test]
    fn decode_loaded_rejects_a_truncated_payload() {
        assert!(decode_loaded(&1u32.to_le_bytes(), "Mesh").is_none());
    }

    // A deferred push against a disk-backed blob keeps only the locator, since
    // the streamer re-reads the file, and pushes an empty mesh carrying the
    // baked bounds and counts.
    #[test]
    fn a_disk_backed_deferred_push_keeps_no_payload_bytes() {
        let mut b = BlobWorld::new();
        let loc = b.payload(&tri_payload());
        let mut world = b.seal();
        let mut ctx = world.context();
        let mut deferred = DeferredMeshSources::default();
        deferred.counts.insert(0, (3, 3));
        let bounds = ([-1.0; 3], [1.0; 3]);

        let mut sink = MeshSink::new(&deferred, true);
        assert_eq!(sink.push(&mut ctx, &loc, Some(bounds), "Mesh"), Some(0));

        let seed = sink.deferred_payloads.get(&0).expect("seed captured");
        assert!(seed.bytes.is_none(), "a disk-backed seed re-reads the blob");
        assert_eq!(
            (seed.locator.offset, seed.locator.len),
            (loc.offset, loc.len)
        );
        let mesh = &sink.geometry[0];
        assert!(mesh.vertices.is_empty(), "decode deferred");
        assert_eq!(mesh.bounds, Some(bounds));
        assert_eq!(mesh.counts, Some((3, 3)));
    }

    fn test_room(locator: Option<PayloadLocator>) -> Room {
        Room {
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
        let mut ctx = world.context();

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
        let mut ctx = world.context();
        assert!(load_room_geometry(&mut ctx).is_none());
    }

    // Same for a model entity: ModelRenderer fields, and with no dynamic tags the
    // item is static.
    #[test]
    fn decomposed_renderable_item_matches_a_model_prop() {
        use crate::components::ModelRenderer;

        let mut prop = make_prop([0.0; 3]);
        prop.model = Some(Ref::new(AssetId(100)));
        prop.cull_distance = 30.0;

        let mut world = World::new();
        let mut ctx = world.context();

        let e = ctx.components.spawn();
        ctx.insert(
            e,
            ModelRenderer {
                model: prop.model.unwrap().id(),
                cull_distance: prop.cull_distance,
            },
        );

        ctx.identify(e, AssetId(8));
        let item = decomposed_renderable_item(&ctx, e);
        assert_eq!(
            item,
            RenderableItem {
                asset_id: Some(AssetId(8)),
                model: Some(AssetId(100)),
                mesh: None,
                material: None,
                cull_distance: 30.0,
                is_dynamic: false,
            }
        );
    }
}
