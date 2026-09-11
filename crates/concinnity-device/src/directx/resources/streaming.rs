// src/directx/resources/streaming.rs
//
// `VoxelWorld` chunk streaming for DxContext: the init-time headroom growth
// that seeds the chunk sub-allocators, then per-chunk add / remove / move
// within that headroom.

use concinnity_core::gfx::mesh_payload::Vertex;
use concinnity_core::gfx::render_types::*;
use concinnity_core::render::backend::ChunkMesh;
use concinnity_core::render::draw_slot;
use concinnity_core::render::error;
use windows::Win32::Graphics::Direct3D12::*;

use super::super::com;
use super::super::context::*;
use super::super::texture::*;

impl DxContext {
    // Grow the shared vertex/index buffers by a headroom region for streamed
    // `VoxelWorld` chunks and seed the chunk sub-allocators with it. The chunk
    // material's texture slots ride each chunk's cull record.
    //
    // Called once at init by `GraphicsSystem` when a `VoxelWorld` is present.
    // The build-time geometry is copied verbatim into the start of the new
    // (larger) DEFAULT-heap buffers; chunks are placed in the appended
    // headroom by `add_chunk_mesh`. This runs before the first frame, so no
    // in-flight command list references the replaced buffers.
    pub(crate) fn setup_chunk_streaming(
        &mut self,
        chunk_vtx_bytes: usize,
        chunk_idx_bytes: usize,
    ) -> Result<(), String> {
        self.wait_idle();
        let old_v_len = self.geometry.vertex_buffer_view.SizeInBytes as u64;
        let old_i_len = self.geometry.index_buffer_view.SizeInBytes as u64;
        let new_v_len = old_v_len + chunk_vtx_bytes as u64;
        let new_i_len = old_i_len + chunk_idx_bytes as u64;

        // Buffers are created in COMMON; the CopyBufferRegion below implicitly
        // promotes the destination COMMON -> COPY_DEST.
        let new_vbuf = create_buffer(
            &self.alloc,
            new_v_len,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
        )?;
        let new_ibuf = create_buffer(
            &self.alloc,
            new_i_len,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
        )?;

        // Copy the build-time geometry into the start of the grown buffers so
        // every existing draw's offsets stay valid.
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        one_shot_submit(&self.device, &self.command_queue, |cmd| unsafe {
            let v_src = transition_barrier(
                &self.geometry.vertex_buffer,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            let i_src = transition_barrier(
                &self.geometry.index_buffer,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            cmd.ResourceBarrier(&[v_src, i_src]);
            cmd.CopyBufferRegion(&*new_vbuf, 0, &*self.geometry.vertex_buffer, 0, old_v_len);
            cmd.CopyBufferRegion(&*new_ibuf, 0, &*self.geometry.index_buffer, 0, old_i_len);
            let v_dst = transition_barrier(
                &new_vbuf,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            );
            let i_dst = transition_barrier(
                &new_ibuf,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
            );
            cmd.ResourceBarrier(&[v_dst, i_dst]);
        })?;

        self.geometry.vertex_buffer_view = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: com::gpu_va(&new_vbuf),
            SizeInBytes: new_v_len as u32,
            StrideInBytes: std::mem::size_of::<Vertex>() as u32,
        };
        self.geometry.index_buffer_view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: com::gpu_va(&new_ibuf),
            SizeInBytes: new_i_len as u32,
            // Static IB is u32 (matches the `Format` chosen in init/mod.rs).
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT,
        };
        self.geometry.vertex_buffer = new_vbuf;
        self.geometry.index_buffer = new_ibuf;

        // Seed the chunk allocators with the appended headroom. retire_frame 0:
        // nothing has been drawn, so the space is reusable immediately.
        self.chunk_stream
            .vtx_alloc
            .free(old_v_len, chunk_vtx_bytes as u64, 0);
        self.chunk_stream
            .idx_alloc
            .free(old_i_len, chunk_idx_bytes as u64, 0);
        Ok(())
    }

    // Place one streamed chunk's geometry in the chunk headroom region and
    // write its `DrawObject` at the engine-allocated destination slot.
    //
    // The chunk is non-cullable and joins the `draw.always` set: the streaming
    // window already bounds the resident chunk count. Indices stay
    // mesh-relative (0-based) and the draw passes the vertex region's base as
    // `base_vertex`, so a chunk placed past the 65 535-vertex `u16` index
    // range still renders. `frame` reclaims retired deferred frees first.
    // `wait_idle` runs before the geometry copy so the whole-resource
    // COPY_DEST transition races no in-flight command list.
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
        // Static IB is u32; chunk indices come in as u16 and get widened on
        // write. Size the allocation against the u32 stride.
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

        // Vertices and indices both copy verbatim: the indices stay 0-based and
        // the draw fixes them up with `base_vertex`.
        let vert_bytes = bytemuck::cast_slice(vertices);
        self.write_geometry_region(
            &self.geometry.vertex_buffer,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            v_off as u64,
            vert_bytes,
        )?;
        // Chunk indices stay mesh-relative; the draw fixes them up with
        // `base_vertex`. Widen u16 → u32 to match the static IB's stride.
        let widened: Vec<u32> = indices.iter().map(|&i| u32::from(i)).collect();
        let idx_bytes = bytemuck::cast_slice(&widened);
        self.write_geometry_region(
            &self.geometry.index_buffer,
            D3D12_RESOURCE_STATE_INDEX_BUFFER,
            i_off as u64,
            idx_bytes,
        )?;

        // v_off is a multiple of size_of::<Vertex>() (the headroom start and
        // every alloc are), so the base is an exact vertex index.
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
            // Non-cullable: degenerate AABB disables frustum/distance culling.
            bb_min: [f32::NAN; 3],
            bb_max: [f32::NAN; 3],
            cull_distance: 0.0,
            // Streamed chunks always render at the build-time mesh; no LOD.
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
                self.model_history.borrow_mut().reoccupy_draw(slot);
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
