// src/metal/streaming.rs
//
// VoxelWorld chunk streaming for MtlContext: sub-allocator setup and the
// add / remove / move-chunk-mesh operations driven after init.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::gfx::mesh_payload::Vertex;
use concinnity_core::gfx::render_types::DrawObject;
use concinnity_core::render::backend::ChunkMesh;
use concinnity_core::render::draw_slot;
use objc2_metal::{MTLBuffer, MTLResourceOptions};

use super::context::*;

impl MtlContext {
    // `VoxelWorld` chunks and seed the chunk sub-allocators with it.
    //
    // Called once at init by `GraphicsSystem` when a `VoxelWorld` is present.
    // The build-time geometry is copied verbatim into the start of the new
    // (larger) buffers; chunks are placed in the appended headroom by
    // `add_chunk_mesh`. This runs before the first frame, so no in-flight
    // command buffer references the replaced buffers.
    pub(crate) fn setup_chunk_streaming(
        &mut self,
        chunk_vtx_bytes: usize,
        chunk_idx_bytes: usize,
    ) -> Result<(), String> {
        let old_v_len = self.vertex_buffer.length();
        let old_i_len = self.index_buffer.length();

        let new_vbuf = self
            .allocator
            .alloc_buffer(
                old_v_len + chunk_vtx_bytes,
                MTLResourceOptions::StorageModeShared,
            )
            .map_err(|e| format!("setup_chunk_streaming: chunk vertex buffer: {e}"))?;
        let new_ibuf = self
            .allocator
            .alloc_buffer(
                old_i_len + chunk_idx_bytes,
                MTLResourceOptions::StorageModeShared,
            )
            .map_err(|e| format!("setup_chunk_streaming: chunk index buffer: {e}"))?;

        // Copy the build-time geometry into the start of the grown buffers so
        // every existing draw's offsets stay valid.
        copy_buffer_prefix(&self.vertex_buffer, &new_vbuf, old_v_len);
        copy_buffer_prefix(&self.index_buffer, &new_ibuf, old_i_len);
        self.vertex_buffer = new_vbuf;
        self.index_buffer = new_ibuf;

        // Seed the chunk allocators with the appended headroom. retire_frame 0:
        // nothing has been drawn, so the space is reusable immediately.
        self.geometry_alloc
            .chunk_vtx
            .free(old_v_len as u64, chunk_vtx_bytes as u64, 0);
        self.geometry_alloc
            .chunk_idx
            .free(old_i_len as u64, chunk_idx_bytes as u64, 0);
        Ok(())
    }

    // Place one streamed chunk's geometry in the chunk headroom region and
    // write its `DrawObject` at the engine-allocated destination slot.
    //
    // The chunk is non-cullable (sentinel AABB): the streaming window already
    // bounds the resident chunk count, so the renderer draws every resident
    // chunk. `frame` reclaims retired deferred frees first.
    pub(crate) fn add_chunk_mesh(
        &mut self,
        mesh: ChunkMesh<'_>,
        dst: draw_slot::SlotAlloc,
    ) -> Result<(), String> {
        let ChunkMesh {
            verts: vertices,
            idxs: indices,
            model,
            texture_slot,
            normal_map_slot,
            material,
            frame,
        } = mesh;
        if vertices.is_empty() || indices.is_empty() {
            return Err("add_chunk_mesh: empty chunk geometry".to_string());
        }
        self.geometry_alloc.chunk_vtx.reclaim(frame);
        self.geometry_alloc.chunk_idx.reclaim(frame);

        let v_len = std::mem::size_of_val(vertices);
        // The shared index buffer is u32-typed; the input `indices` are u16 and
        // get widened on write below, so size the allocation against the u32
        // stride. Sizing against the u16 source would alloc half the bytes the
        // write needs and corrupt the next chunk's indices.
        let i_len = indices.len() * std::mem::size_of::<u32>();
        let v_off = self
            .geometry_alloc
            .chunk_vtx
            .alloc(v_len as u64)
            .ok_or_else(|| {
                format!(
                    "add_chunk_mesh: no free chunk vertex space for {} bytes",
                    v_len
                )
            })? as usize;
        let i_off = match self.geometry_alloc.chunk_idx.alloc(i_len as u64) {
            Some(o) => o as usize,
            None => {
                self.geometry_alloc
                    .chunk_vtx
                    .free(v_off as u64, v_len as u64, 0);
                return Err(format!(
                    "add_chunk_mesh: no free chunk index space for {} bytes",
                    i_len
                ));
            }
        };

        // Vertices copy verbatim. Indices stay mesh-relative (0-based): a chunk
        // can land far past the 65 535-vertex u16 index range, so rather than
        // rebasing the indices the draw passes the vertex region's base as
        // `baseVertex`. v_off is a multiple of size_of::<Vertex>() (the
        // headroom start and every alloc are), so the base is an exact index.
        // The shared index_buffer is u32-typed, so widen the per-mesh u16
        // indices before writing.
        write_buffer_region(&self.vertex_buffer, v_off, bytes_of_slice(vertices))?;
        let indices_u32: Vec<u32> = indices.iter().map(|&i| u32::from(i)).collect();
        write_buffer_region(&self.index_buffer, i_off, bytes_of_slice(&indices_u32))?;
        let base_vertex = (v_off / std::mem::size_of::<Vertex>()) as i32;

        let obj = DrawObject {
            vertex_offset: v_off,
            vertex_count: vertices.len(),
            index_offset: i_off / std::mem::size_of::<u32>(),
            index_count: indices.len(),
            base_vertex,
            geometry_generation: 0,
            model,
            texture_slot,
            normal_map_slot,
            material,
            // Streamed chunks always render under the world default shader.
            shader_bucket: 0,
            visible: true,
            resident: true,
            // Non-cullable: degenerate AABB disables frustum/distance culling.
            bb_min: [f32::NAN; 3],
            bb_max: [f32::NAN; 3],
            cull_distance: 0.0,
            // Streamed `VoxelWorld` chunks do not run through the build-time
            // per-draw LOD decimator: distance LOD is handled by the streaming
            // window instead, which meshes a near chunk at full voxel detail
            // and a distant one as a coarse impostor (`ChunkDetail`), each a
            // single resolution. So no per-draw `lod_alternates` here.
            lod_alternates: Vec::new(),
        };

        self.place_draw_object(obj, dst);
        // A new resident chunk changes the RT-relevant draw set; the next RT
        // update folds it into the BVH (building just this chunk's BLAS).
        self.rt.topology_dirty = true;
        Ok(())
    }

    // Free a streamed chunk's geometry region and retire its `DrawObject`
    // slot for reuse.
    //
    // `retire_frame` is `current_frame + frames_in_flight` so an in-flight
    // command buffer never has the freed region overwritten by a later
    // `add_chunk_mesh`.
    pub(crate) fn remove_chunk_mesh(
        &mut self,
        draw_idx: usize,
        retire_frame: u64,
    ) -> Result<(), String> {
        let region = draw_slot::retire_chunk_slot(&mut self.draw.objects, draw_idx)?;
        zero_buffer_region(
            &self.vertex_buffer,
            region.vertex_offset as usize,
            region.vertex_bytes as usize,
        )?;
        zero_buffer_region(
            &self.index_buffer,
            region.index_offset as usize,
            region.index_bytes as usize,
        )?;
        self.geometry_alloc
            .chunk_vtx
            .free(region.vertex_offset, region.vertex_bytes, retire_frame);
        self.geometry_alloc
            .chunk_idx
            .free(region.index_offset, region.index_bytes, retire_frame);
        // The removed chunk leaves the RT-relevant draw set; the next RT update
        // drops its BLAS (deferred-freed once in-flight traces retire).
        self.rt.topology_dirty = true;
        Ok(())
    }

    pub(crate) fn set_chunk_model(
        &mut self,
        draw_idx: usize,
        model: [[f32; 4]; 4],
    ) -> Result<(), String> {
        draw_slot::set_chunk_model(&mut self.draw.objects, draw_idx, model)
    }
}
