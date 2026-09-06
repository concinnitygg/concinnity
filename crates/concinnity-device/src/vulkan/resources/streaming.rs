// src/vulkan/resources/streaming.rs
//
// VoxelWorld chunk streaming for VkContext: appends a headroom region to the
// shared vertex/index buffers, builds the chunk descriptor set from the
// world's chunk material, then allocates / frees per-chunk geometry from that
// headroom on demand.

use ash::vk;

use crate::gfx::backend::ChunkMesh;
use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::*;

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
    ) -> crate::gfx::error::RenderResult<()> {
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
        dst: crate::gfx::draw_slot::SlotAlloc,
    ) -> crate::gfx::error::RenderResult<()> {
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
                crate::gfx::error::RenderError::OutOfDeviceMemory(format!(
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
                return Err(crate::gfx::error::RenderError::OutOfDeviceMemory(format!(
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
            crate::gfx::draw_slot::SlotAlloc::Reuse(slot) => {
                self.draw.objects[slot] = obj;
                slot
            }
            crate::gfx::draw_slot::SlotAlloc::Append(slot) => {
                debug_assert_eq!(
                    slot,
                    self.draw.objects.len(),
                    "appended draw slot must match the draw-object count"
                );
                self.draw.objects.push(obj);
                slot
            }
        };
        // Seed the streamed-chunk previous transform onto the unified G-buffer's
        // velocity bookkeeping so a chunk that streams in does not ghost from
        // IDENTITY on its first frame.
        if let Some(gb) = &mut self.gbuffer
            && draw_idx < gb.prev_models.len()
        {
            gb.prev_models[draw_idx] = model;
        }
        // A new resident chunk changes the RT-relevant draw set; the next RT
        // update folds it into the BVH (building just this chunk's BLAS).
        self.rt_topology_dirty = true;
        Ok(())
    }

    // Free a streamed chunk's geometry region and retire its `DrawObject`
    // slot for reuse.
    pub(crate) fn remove_chunk_mesh(
        &mut self,
        draw_idx: usize,
        retire_frame: u64,
    ) -> Result<(), String> {
        let obj =
            self.draw.objects.get(draw_idx).ok_or_else(|| {
                format!("remove_chunk_mesh: draw object {} out of range", draw_idx)
            })?;
        let v_off = obj.vertex_offset as u64;
        let v_len = (obj.vertex_count * std::mem::size_of::<Vertex>()) as u64;
        let i_off = (obj.index_offset * std::mem::size_of::<u32>()) as u64;
        let i_len = (obj.index_count * std::mem::size_of::<u32>()) as u64;
        self.chunk_stream.vtx_alloc.free(v_off, v_len, retire_frame);
        self.chunk_stream.idx_alloc.free(i_off, i_len, retire_frame);
        let obj = &mut self.draw.objects[draw_idx];
        obj.visible = false;
        obj.resident = false;
        // The removed chunk leaves the RT-relevant draw set; the next RT update
        // drops its BLAS (deferred-freed once in-flight traces retire).
        self.rt_topology_dirty = true;
        Ok(())
    }

    // Rewrite a resident chunk's model matrix.
    pub(crate) fn set_chunk_model(
        &mut self,
        draw_idx: usize,
        model: [[f32; 4]; 4],
    ) -> Result<(), String> {
        let obj = self
            .draw
            .objects
            .get_mut(draw_idx)
            .ok_or_else(|| format!("set_chunk_model: draw object {} out of range", draw_idx))?;
        obj.model = model;
        Ok(())
    }
}
