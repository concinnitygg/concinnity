//! What the streamer moves in and out of a live backend's slots.
//!
//! Mesh residency and the chunk pool a streamed world places geometry into.
//! Every entry point here is addressed by a draw slot or a pool index the
//! engine's allocators own, so the backend never decides where something
//! lands.

use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::{DrawIndex, MaterialUniforms};
use crate::render::backend::SceneHost;
use crate::render::backend::error::{RenderError, RenderResult};

/// One streamed chunk's geometry plus placement, supplied to
/// [`DrawStreaming::add_chunk_mesh`]. `frame` reclaims retired deferred frees
/// before the chunk is placed in the streaming headroom.
#[derive(Clone, Copy)]
pub struct ChunkMesh<'a> {
    /// Chunk vertices.
    pub verts: &'a [Vertex],
    /// Chunk indices, mesh-relative.
    pub idxs: &'a [u16],
    /// Column-major placement matrix.
    pub model: [[f32; 4]; 4],
    /// Index into the shared texture pool for the albedo map.
    pub texture_slot: usize,
    /// Index into the shared texture pool for the normal map.
    pub normal_map_slot: usize,
    /// Per-chunk material scalars.
    pub material: MaterialUniforms,
    /// Current frame number, used to reclaim retired deferred frees.
    pub frame: u64,
    /// The planet tile the chunk draws, when it is one.
    pub planet_tile: Option<PlanetChunk>,
    /// The box around `verts` in model space, for a chunk the cull tests:
    /// its culling box follows `model` from then on. `None` draws the chunk
    /// unculled.
    pub bounds: Option<([f32; 3], [f32; 3])>,
}

/// The planet tile a streamed chunk draws: which square of the planet, and
/// how far its surface lies along the up at its origin, so the passes that
/// read a planet's ground can find it in the shared vertex buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanetChunk {
    /// The tile.
    pub tile: crate::environment::planet::TileId,
    /// The lowest and highest its surface lies along the up at its origin, in
    /// meters from the origin.
    pub radial: [f32; 2],
    /// The farthest any of its corners moves when it morphs all the way into
    /// its parent's mesh, in meters.
    pub morph_extent: f32,
    /// How high its origin lies above the planet's radius, in meters.
    pub origin_altitude: f32,
}

/// Draw-slot residency: the mesh and chunk uploads and evictions a streaming
/// world drives.
///
/// The chunk headroom is required: a backend that cannot grow its geometry
/// buffers cannot draw a streamed world at all. Placing, evicting, moving and
/// cloning draws only edit the scene and write geometry bytes, so they are
/// defaulted over [`SceneHost`] and report [`RenderError::Unsupported`]
/// without a scene.
pub trait DrawStreaming: SceneHost {
    /// Return a streamed mesh's vertex / index regions to the sub-allocators
    /// and mark the draw slot non-resident. The regions are held against
    /// `retire_frame`, so a frame still in flight cannot have them reused
    /// underneath it.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] without a scene, or an error if the
    /// slot is out of range.
    fn evict_mesh(&mut self, draw_idx: DrawIndex, retire_frame: u64) -> RenderResult<()> {
        self.edit_geometry(&mut |scene, writer| scene.evict_mesh(draw_idx, retire_frame, writer))
            .unwrap_or(Err(RenderError::Unsupported { op: "evict_mesh" }))
    }
    /// Upload a streamed mesh's geometry into a draw slot.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] without a scene, or an error if the
    /// slot is out of range or the geometry does not fit the shared buffers.
    fn upload_mesh(
        &mut self,
        draw_idx: DrawIndex,
        verts: &[Vertex],
        idxs: &[u16],
        frame: u64,
    ) -> RenderResult<()> {
        self.edit_geometry(&mut |scene, writer| {
            scene.upload_mesh(draw_idx, verts, idxs, frame, writer)
        })
        .unwrap_or(Err(RenderError::Unsupported { op: "upload_mesh" }))
    }

    /// Seed the streamed-mesh sub-allocators with one reserved headroom block
    /// (byte ranges in the shared vertex / index buffers) instead of the
    /// per-mesh build-time regions. Used by the shrinkable-seed path: the
    /// streamed geometry is no longer baked into the buffers at build time, so
    /// the renderer hands the allocators one contiguous block sized to the
    /// cap-many resident meshes rather than the whole streamed set. A setup
    /// feed: without a scene there are no allocators to seed, and this
    /// succeeds doing nothing.
    ///
    /// # Errors
    ///
    /// Returns an error if the geometry edit that seeds the allocators fails.
    fn seed_mesh_streaming(
        &mut self,
        vtx_offset: u64,
        vtx_bytes: u64,
        idx_offset: u64,
        idx_bytes: u64,
    ) -> RenderResult<()> {
        self.edit_geometry(&mut |scene, writer| {
            scene.seed_mesh_streaming(vtx_offset, vtx_bytes, idx_offset, idx_bytes, writer);
            Ok(())
        })
        .unwrap_or(Ok(()))
    }

    /// Voxel-world chunk streaming: grow the shared geometry buffers by the
    /// chunk headroom. A chunk's material slots ride its own draw record.
    ///
    /// # Errors
    ///
    /// Returns an error if the geometry buffers cannot be grown.
    fn setup_chunk_streaming(
        &mut self,
        chunk_vtx_bytes: usize,
        chunk_idx_bytes: usize,
    ) -> RenderResult<()>;
    /// The destination draw slot comes from the engine's allocator, like
    /// `clone_static_draw_object`; the freed slot is likewise returned to it by
    /// the caller of `remove_chunk_mesh`.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] without a scene, or an error if the
    /// geometry does not fit the shared buffers.
    fn add_chunk_mesh(
        &mut self,
        mesh: ChunkMesh<'_>,
        dst: crate::render::slots::draw_slot::SlotAlloc,
    ) -> RenderResult<()> {
        self.edit_geometry(&mut |scene, writer| scene.add_chunk_mesh(&mesh, dst, writer))
            .unwrap_or(Err(RenderError::Unsupported {
                op: "add_chunk_mesh",
            }))
    }
    /// Free a streamed chunk's geometry, retiring it after `retire_frame`.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] without a scene, or an error if
    /// `draw_idx` names no streamed chunk.
    fn remove_chunk_mesh(&mut self, draw_idx: DrawIndex, retire_frame: u64) -> RenderResult<()> {
        self.edit_geometry(&mut |scene, writer| {
            scene.remove_chunk_mesh(draw_idx, retire_frame, writer)
        })
        .unwrap_or(Err(RenderError::Unsupported {
            op: "remove_chunk_mesh",
        }))
    }
    /// Move a streamed chunk by replacing its placement matrix.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] without a scene, or an error if
    /// `draw_idx` is out of range.
    fn set_chunk_model(&mut self, draw_idx: DrawIndex, model: [[f32; 4]; 4]) -> RenderResult<()> {
        match self.scene_mut() {
            Some(scene) => scene.set_chunk_model(draw_idx, model),
            None => Err(RenderError::Unsupported {
                op: "set_chunk_model",
            }),
        }
    }

    /// Instantiate a runtime copy of an existing draw object at a new transform:
    /// re-use the source slot's geometry region (`vertex_offset` / `vertex_count`
    /// / `index_offset` / `index_count` / `base_vertex` / `lod_alternates`) and
    /// copy its texture slots, material, and cull distance, swapping only the
    /// model matrix. The new slot reuses one freed by `retire_draw_object` before
    /// growing the draw-object vec. The destination slot comes from the
    /// engine's draw-slot allocator: `Reuse` overwrites a vacated entry,
    /// `Append` grows the vec (the index always equals the current length,
    /// which implementations debug-assert). Driven by runtime entity spawn
    /// (`SpawnRequest`). The copy is non-cullable (sentinel AABB) and drawn
    /// every frame, since the init-time BVH cannot refit to admit a slot added
    /// at runtime; moving copies (the common case) opt out of the static BVH
    /// exactly like streamed chunks and held items. Without a scene this is
    /// [`RenderError::Unsupported`], which the spawn path logs and skips.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] without a scene, or an error if
    /// `src_draw_idx` names no draw object.
    fn clone_static_draw_object(
        &mut self,
        src_draw_idx: DrawIndex,
        model: [[f32; 4]; 4],
        dst: crate::render::slots::draw_slot::SlotAlloc,
    ) -> RenderResult<()> {
        match self.scene_mut() {
            Some(scene) => scene.clone_static_draw_object(src_draw_idx, model, dst),
            None => Err(RenderError::Unsupported {
                op: "clone_static_draw_object",
            }),
        }
    }
}
