//! What the streamer moves in and out of a live backend's slots.
//!
//! Mesh and texture residency, the chunk pool a streamed world places
//! geometry into, and the world-shader buckets a pinned scene installs. Every
//! entry point here is addressed by a draw slot or a pool index the engine's
//! allocators own, so the backend never decides where something lands.
//!
//! Albedo and normal maps share one handle-indexed texture pool, so every
//! streamed texture flows through the same slot pair whatever its role. The
//! image carries its GPU format and mip chain: RGBA8 regenerates mips on
//! upload, block-compressed formats upload their chain verbatim.

use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::MaterialUniforms;
use crate::render::backend_init::WorldShader;
use crate::render::error::RenderResult;
use alloc::string::String;
use alloc::string::ToString;

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
}

/// Slot residency: the mesh, texture and chunk uploads and evictions a
/// streaming world drives, plus the world-shader buckets a scene pins.
///
/// The upload and eviction paths are required: a backend that cannot fill a
/// slot cannot draw a streamed world at all. The chunk-pool and world-shader
/// paths default to a no-op or an `Err`, so a backend supports them only if it
/// has them.
pub trait DrawStreaming {
    /// Release a texture slot's image, leaving a 1x1 placeholder in the slot so
    /// a draw still holding the handle has something to sample. `Err` when the
    /// slot is out of range.
    fn evict_texture_slot(&mut self, slot: usize) -> Result<(), String>;
    /// Replace a texture slot's image after a streaming upload.
    fn update_texture_slot(
        &mut self,
        slot: usize,
        image: &crate::bake::texture::TextureImage,
    ) -> RenderResult<()>;

    /// Return a streamed mesh's vertex / index regions to the sub-allocators
    /// and mark the draw slot non-resident. The regions are held against
    /// `retire_frame`, so a frame still in flight cannot have them reused
    /// underneath it. `Err` when the slot is out of range.
    fn evict_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> Result<(), String>;
    /// Upload a streamed mesh's geometry into a draw slot.
    fn upload_mesh(
        &mut self,
        draw_idx: usize,
        verts: &[Vertex],
        idxs: &[u16],
        frame: u64,
    ) -> RenderResult<()>;

    /// Seed the streamed-mesh sub-allocators with one reserved headroom block
    /// (byte ranges in the shared vertex / index buffers) instead of the
    /// per-mesh build-time regions. Used by the shrinkable-seed path: the
    /// streamed geometry is no longer baked into the buffers at build time, so
    /// the renderer hands the allocators one contiguous block sized to the
    /// cap-many resident meshes rather than the whole streamed set. Implemented
    /// on Metal + DirectX + Vulkan. Default no-op: a backend without the
    /// shrinkable seed keeps freeing each mesh's build-time region in
    /// `setup_mesh_streaming`.
    fn seed_mesh_streaming(
        &mut self,
        vtx_offset: u64,
        vtx_bytes: u64,
        idx_offset: u64,
        idx_bytes: u64,
    ) {
        let _ = (vtx_offset, vtx_bytes, idx_offset, idx_bytes);
    }

    /// Voxel-world chunk streaming: grow the shared geometry buffers by the
    /// chunk headroom. A chunk's material slots ride its own draw record.
    fn setup_chunk_streaming(
        &mut self,
        chunk_vtx_bytes: usize,
        chunk_idx_bytes: usize,
    ) -> RenderResult<()>;
    /// The destination draw slot comes from the engine's allocator, like
    /// `clone_static_draw_object`; the freed slot is likewise returned to it by
    /// the caller of `remove_chunk_mesh`.
    fn add_chunk_mesh(
        &mut self,
        mesh: ChunkMesh<'_>,
        dst: crate::render::draw_slot::SlotAlloc,
    ) -> RenderResult<()>;
    /// Free a streamed chunk's geometry, retiring it after `retire_frame`.
    fn remove_chunk_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> Result<(), String>;
    /// Move a streamed chunk by replacing its placement matrix.
    fn set_chunk_model(&mut self, draw_idx: usize, model: [[f32; 4]; 4]) -> Result<(), String>;

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
    /// exactly like streamed chunks and held items. Default no-op (returns
    /// `Err`): backends without an implementation leave the spawn path
    /// logged + skipped at the caller.
    fn clone_static_draw_object(
        &mut self,
        src_draw_idx: usize,
        model: [[f32; 4]; 4],
        dst: crate::render::draw_slot::SlotAlloc,
    ) -> Result<(), String> {
        let _ = (src_draw_idx, model, dst);
        Err("clone_static_draw_object: not implemented on this backend".to_string())
    }

    /// Build the render pipeline for one shader bucket from its compiled stage
    /// bytes, making draws that carry that bucket renderable. Called by the
    /// streaming pump when a scene that exclusively owns the bucket's `Shader`
    /// pins: init skipped the build, so this is where the cost lands (behind
    /// the loading screen, since the bucket counts as scene-resident content).
    /// Bucket 0 is the world default program and is never installed this way.
    ///
    /// Default no-op-with-Ok: a backend that renders every draw with the world
    /// default program has no per-bucket pipeline to build, and the bucket is
    /// resident as far as scene loading is concerned.
    fn install_world_shader(&mut self, bucket: u32, shader: WorldShader<'_>) -> RenderResult<()> {
        let _ = (bucket, shader);
        Ok(())
    }

    /// Release one shader bucket's render pipeline, undoing
    /// [`Self::install_world_shader`]. Called when the owning scene unpins;
    /// draws carrying the bucket stop rendering until it is installed again.
    /// Default no-op, for the same reason as above.
    fn evict_world_shader(&mut self, bucket: u32) {
        let _ = bucket;
    }
}
