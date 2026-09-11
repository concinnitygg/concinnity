// src/vulkan/resources/streaming.rs
//
// VoxelWorld chunk streaming for VkContext: appends a headroom region to the
// shared vertex/index buffers, builds the chunk descriptor set from the
// world's chunk material, then allocates / frees per-chunk geometry from that
// headroom on demand.

use ash::vk;
use concinnity_core::gfx::mesh_payload::Vertex;
use concinnity_core::gfx::render_types::*;
use concinnity_core::render::backend::ChunkMesh;
use concinnity_core::render::draw_slot;
use concinnity_core::render::error;

use super::super::context::*;
use super::super::texture;

impl VkContext {
    // Grow the shared vertex/index buffers by a headroom region for streamed
    // `VoxelWorld` chunks and seed the chunk sub-allocators with it. The chunk
    // material's texture slots ride each chunk's cull record, so no descriptor
    // is baked here.
    pub(crate) fn setup_chunk_streaming(
        &mut self,
        chunk_vtx_bytes: usize,
        chunk_idx_bytes: usize,
    ) -> error::RenderResult<()> {
        self.wait_idle();
        let old_v = self.geometry.vertex_buffer_bytes;
        let old_i = self.geometry.index_buffer_bytes;
        let new_v = old_v + chunk_vtx_bytes as u64;
        let new_i = old_i + chunk_idx_bytes as u64;

        let shared = super::shared_geometry_usage(self.rt_capable);
        let new_vbuf = self.alloc.create_buffer(
            new_v,
            vk::BufferUsageFlags::VERTEX_BUFFER | shared,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let new_ibuf = self.alloc.create_buffer(
            new_i,
            vk::BufferUsageFlags::INDEX_BUFFER | shared,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Copy the build-time geometry into the start of the grown buffers so
        // every existing draw's offsets stay valid.
        texture::one_shot_submit(
            &self.device,
            self.commands.command_pool,
            self.graphics_queue,
            |cmd| {
                let vcopy = vk::BufferCopy::default().size(old_v);
                let icopy = vk::BufferCopy::default().size(old_i);
                // SAFETY: `cmd` is a command buffer in the recording state, and every handle and
                // slice these commands name is live for the call.
                unsafe {
                    self.device.cmd_copy_buffer(
                        cmd,
                        self.geometry.vertex_buffer.buffer(),
                        new_vbuf.buffer(),
                        std::slice::from_ref(&vcopy),
                    );
                    self.device.cmd_copy_buffer(
                        cmd,
                        self.geometry.index_buffer.buffer(),
                        new_ibuf.buffer(),
                        std::slice::from_ref(&icopy),
                    );
                }
            },
        )?;

        self.geometry.vertex_buffer = new_vbuf;
        self.geometry.index_buffer = new_ibuf;
        self.geometry.vertex_buffer_bytes = new_v;
        self.geometry.index_buffer_bytes = new_i;

        self.chunk_stream
            .vtx_alloc
            .free(old_v, chunk_vtx_bytes as u64, 0);
        self.chunk_stream
            .idx_alloc
            .free(old_i, chunk_idx_bytes as u64, 0);

        Ok(())
    }

    // Place one streamed chunk's geometry in the chunk headroom region and
    // write its `DrawObject` at the engine-allocated destination slot.
    pub(crate) fn add_chunk_mesh(
        &mut self,
        mesh: ChunkMesh<'_>,
        dst: draw_slot::SlotAlloc,
    ) -> error::RenderResult<()> {
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
            return Err("add_chunk_mesh: empty chunk geometry".into());
        }
        self.chunk_stream.vtx_alloc.reclaim(frame);
        self.chunk_stream.idx_alloc.reclaim(frame);

        let v_len = std::mem::size_of_val(vertices);
        let i_len = indices.len() * std::mem::size_of::<u32>();
        let v_off = self
            .chunk_stream
            .vtx_alloc
            .alloc(v_len as u64)
            .ok_or_else(|| {
                error::RenderError::OutOfDeviceMemory(format!(
                    "add_chunk_mesh: no free chunk vertex space for {} bytes",
                    v_len
                ))
            })? as usize;
        let i_off = match self.chunk_stream.idx_alloc.alloc(i_len as u64) {
            Some(o) => o as usize,
            None => {
                self.chunk_stream
                    .vtx_alloc
                    .free(v_off as u64, v_len as u64, 0);
                return Err(error::RenderError::OutOfDeviceMemory(format!(
                    "add_chunk_mesh: no free chunk index space for {} bytes",
                    i_len
                )));
            }
        };

        self.wait_idle();

        let vert_bytes = bytemuck::cast_slice(vertices);
        self.write_geometry_region(
            self.geometry.vertex_buffer.buffer(),
            v_off as u64,
            vert_bytes,
        )?;
        let widened: Vec<u32> = indices.iter().map(|&i| u32::from(i)).collect();
        let idx_bytes = bytemuck::cast_slice(&widened);
        self.write_geometry_region(self.geometry.index_buffer.buffer(), i_off as u64, idx_bytes)?;

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
            visible: true,
            resident: true,
            bb_min: [f32::NAN; 3],
            bb_max: [f32::NAN; 3],
            cull_distance: 0.0,
            lod_alternates: Vec::new(),
            // Streamed chunks render through the world default program.
            shader_bucket: 0,
        };

        // Write at the engine-allocated destination slot.
        let draw_idx = match dst {
            draw_slot::SlotAlloc::Reuse(slot) => {
                self.draw.objects[slot] = obj;
                slot
            }
            draw_slot::SlotAlloc::Append(slot) => {
                debug_assert_eq!(
                    slot,
                    self.draw.objects.len(),
                    "appended draw slot must match the draw-object count"
                );
                self.draw.objects.push(obj);
                slot
            }
        };
        // The slot's model-history entry belongs to whatever held it before, so
        // a chunk that streams in reprojects through its own transform for one
        // frame rather than ghosting from the previous occupant's.
        self.model_history.borrow_mut().reoccupy_draw(draw_idx);
        // A new resident chunk changes the RT-relevant draw set; the next RT
        // update folds it into the BVH (building just this chunk's BLAS).
        self.rt_topology_dirty = true;
        Ok(())
    }

    // Free a streamed chunk's geometry region and retire its `DrawObject`
    // slot for reuse.
    //
    // `retire_frame` is `current_frame + frames_in_flight` so an in-flight
    // submission never has the freed region overwritten by a later
    // `add_chunk_mesh`. The region is not zeroed: a non-resident draw is
    // skipped everywhere and an `alloc` hands back exactly `size` bytes that
    // `add_chunk_mesh` fully overwrites.
    pub(crate) fn remove_chunk_mesh(
        &mut self,
        draw_idx: usize,
        retire_frame: u64,
    ) -> Result<(), String> {
        let region = draw_slot::retire_chunk_slot(&mut self.draw.objects, draw_idx)?;
        self.chunk_stream
            .vtx_alloc
            .free(region.vertex_offset, region.vertex_bytes, retire_frame);
        self.chunk_stream
            .idx_alloc
            .free(region.index_offset, region.index_bytes, retire_frame);
        // The removed chunk leaves the RT-relevant draw set; the next RT update
        // drops its BLAS (deferred-freed once in-flight traces retire).
        self.rt_topology_dirty = true;
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
