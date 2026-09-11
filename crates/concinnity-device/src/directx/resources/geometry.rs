// src/directx/resources/geometry.rs
//
// Streamed-mesh upload + eviction for DxContext: placement in the shared
// vertex / index buffers through the mesh sub-allocators, in-place per-slot
// geometry replacement for hot-reload, and the init-time seeding of the
// streaming headroom.

use windows::Win32::Graphics::Direct3D12::*;

use crate::gfx::mesh_payload::Vertex;

use super::super::texture::{create_buffer, one_shot_submit, transition_barrier};

use super::super::context::*;

impl DxContext {
    // Copy `data` into a sub-region of a DEFAULT-heap geometry buffer.
    //
    // `dest` is a buffer currently in `usage_state` (the vertex or index
    // buffer). The copy goes through a temporary UPLOAD-heap staging buffer
    // and a one-shot command list that transitions the resource
    // `usage_state -> COPY_DEST -> usage_state` around a `CopyBufferRegion`.
    // The caller must `wait_idle` first: the COPY_DEST transition covers the
    // whole resource, so no in-flight command list may still reference it.
    pub(in crate::directx) fn write_geometry_region(
        &self,
        dest: &ID3D12Resource,
        usage_state: D3D12_RESOURCE_STATES,
        offset: u64,
        data: &[u8],
    ) -> Result<(), String> {
        if data.is_empty() {
            return Ok(());
        }
        let upload = create_buffer(
            &self.alloc,
            data.len() as u64,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
        // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local
        // that receives the mapping.
        unsafe { upload.Map(0, None, Some(&mut ptr)) }
            .map_err(|e| format!("mesh region map: {e}"))?;
        // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and the
        // source is a separate allocation, so the ranges cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            upload.Unmap(0, None);
        }
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        one_shot_submit(&self.device, &self.command_queue, |cmd| unsafe {
            let to_dst = transition_barrier(dest, usage_state, D3D12_RESOURCE_STATE_COPY_DEST);
            cmd.ResourceBarrier(&[to_dst]);
            cmd.CopyBufferRegion(dest, offset, &*upload, 0, data.len() as u64);
            let back = transition_barrier(dest, D3D12_RESOURCE_STATE_COPY_DEST, usage_state);
            cmd.ResourceBarrier(&[back]);
        })
    }

    // Upload a streamed mesh's geometry into the shared vertex and index
    // buffers, place it via the sub-allocators, and mark the draw resident.
    //
    // The mesh-streaming subsystem calls this to bring a mesh resident after
    // init. The geometry is placed wherever the allocators find free space
    // (not the build-time region), so `DrawObject::vertex_offset` /
    // `index_offset` are rewritten here. `vertices` / `indices` must match the
    // fixed `vertex_count` / `index_count` recorded by `build_draw_list`.
    //
    // `indices` are mesh-relative (0-based); they are rebased onto the chosen
    // vertex region before upload, so the D3D12 draw can keep a 0 base-vertex.
    // `frame` reclaims deferred frees that have retired by then. `wait_idle`
    // runs first so the whole-resource COPY_DEST transition races no in-flight
    // command list (see `write_geometry_region`).
    pub(crate) fn upload_mesh(
        &mut self,
        draw_idx: usize,
        vertices: &[Vertex],
        indices: &[u16],
        frame: u64,
    ) -> Result<(), String> {
        let obj = self
            .draw
            .objects
            .get(draw_idx)
            .ok_or_else(|| format!("upload_mesh: draw object {} out of range", draw_idx))?;
        let (vertex_count, index_count) = (obj.vertex_count, obj.index_count);
        if vertices.len() != vertex_count {
            return Err(format!(
                "upload_mesh: draw {} expects {} vertices, got {}",
                draw_idx,
                vertex_count,
                vertices.len()
            ));
        }
        if indices.len() != index_count {
            return Err(format!(
                "upload_mesh: draw {} expects {} indices, got {}",
                draw_idx,
                index_count,
                indices.len()
            ));
        }

        // Reclaim frees whose in-flight frames have retired, then place the
        // geometry. build_draw_list never emits a zero-length mesh, so an
        // empty allocation request is treated as a hard error.
        self.mesh_stream.vtx_alloc.reclaim(frame);
        self.mesh_stream.idx_alloc.reclaim(frame);
        let v_len = std::mem::size_of_val(vertices);
        // Static IB is u32 (the per-scene total can exceed u16); per-mesh
        // indices come in as u16 (each mesh fits in u16, enforced by the
        // build-time splitter) and get widened on write below. Size the
        // allocation against the u32 stride. Mirrors metal's upload_mesh.
        let i_len = indices.len() * std::mem::size_of::<u32>();
        let v_off = self
            .mesh_stream
            .vtx_alloc
            .alloc(v_len as u64)
            .ok_or_else(|| {
                format!(
                    "upload_mesh: draw {}: no free vertex space for {} bytes",
                    draw_idx, v_len
                )
            })? as usize;
        let i_off = match self.mesh_stream.idx_alloc.alloc(i_len as u64) {
            Some(o) => o as usize,
            None => {
                // hand the vertex region back so a half-failed upload leaks no
                // space (frame 0: it was never written or drawn)
                self.mesh_stream
                    .vtx_alloc
                    .free(v_off as u64, v_len as u64, 0);
                return Err(format!(
                    "upload_mesh: draw {}: no free index space for {} bytes",
                    draw_idx, i_len
                ));
            }
        };

        self.wait_idle();

        // Vertices copy verbatim. Indices are mesh-relative, so rebase them to
        // the vertex region the allocator chose: v_off is always a multiple of
        // size_of::<Vertex>() (every seed region and allocation is), so the
        // base is an exact vertex index.
        let vert_bytes = bytemuck::cast_slice(vertices);
        self.write_geometry_region(
            &self.geometry.vertex_buffer,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            v_off as u64,
            vert_bytes,
        )?;
        let base = (v_off / std::mem::size_of::<Vertex>()) as u32;
        // Widen u16 → u32 while rebasing onto the chosen vertex region.
        let rebased: Vec<u32> = indices.iter().map(|&i| u32::from(i) + base).collect();
        let idx_bytes = bytemuck::cast_slice(&rebased);
        self.write_geometry_region(
            &self.geometry.index_buffer,
            D3D12_RESOURCE_STATE_INDEX_BUFFER,
            i_off as u64,
            idx_bytes,
        )?;

        let obj = &mut self.draw.objects[draw_idx];
        obj.vertex_offset = v_off;
        obj.index_offset = i_off / std::mem::size_of::<u32>();
        obj.resident = true;
        // The mesh joins the RT-relevant draw set at a freshly allocated region;
        // the next RT update builds its BLAS over the new slice.
        self.rt_topology_dirty = true;
        Ok(())
    }

    // Overwrite a `Mesh` draw slot's vertex / index data in place. Driven by
    // asset hot-reload (`cn debug` only). New `vertices` / `indices` are
    // written at the draw object's existing offsets in the shared vertex /
    // index buffers, so the slot's count must match init-time; size-changing
    // reloads need `rebuild_static_geometry`, not this call. Each entry in
    // `lod_alternates` is written to the matching slot's pre-allocated LOD
    // region; LOD counts and per-LOD index counts must match init-time too.
    // Per-LOD `switch_distance`s are re-stored so JSON-side tweaks to
    // `lod_distances` propagate without a process restart. `wait_idle` is
    // folded into each `write_geometry_region` call (the whole-resource
    // COPY_DEST transition needs no in-flight command list referencing the
    // buffer). Mirrors `MtlContext::update_mesh_geometry`.
    pub(crate) fn update_mesh_geometry(
        &mut self,
        draw_idx: usize,
        vertices: &[Vertex],
        indices: &[u16],
        lod_alternates: &[(f32, Vec<u16>)],
    ) -> Result<(), String> {
        let obj = self.draw.objects.get(draw_idx).ok_or_else(|| {
            format!(
                "update_mesh_geometry: draw object {} out of range",
                draw_idx
            )
        })?;
        if vertices.len() != obj.vertex_count {
            return Err(format!(
                "update_mesh_geometry: draw {} expects {} vertices, got {} \
                 (in-place path is size-matched only; size changes route through \
                 rebuild_static_geometry)",
                draw_idx,
                obj.vertex_count,
                vertices.len()
            ));
        }
        if indices.len() != obj.index_count {
            return Err(format!(
                "update_mesh_geometry: draw {} expects {} indices, got {} \
                 (in-place path is size-matched only; size changes route through \
                 rebuild_static_geometry)",
                draw_idx,
                obj.index_count,
                indices.len()
            ));
        }
        if lod_alternates.len() != obj.lod_alternates.len() {
            return Err(format!(
                "update_mesh_geometry: draw {} expects {} LOD alternate(s), got {} \
                 (LOD-count changes need rebuild_static_geometry)",
                draw_idx,
                obj.lod_alternates.len(),
                lod_alternates.len()
            ));
        }
        for (lod_idx, ((_, alt_idx), slice)) in lod_alternates
            .iter()
            .zip(obj.lod_alternates.iter())
            .enumerate()
        {
            if alt_idx.len() != slice.index_count {
                return Err(format!(
                    "update_mesh_geometry: draw {} LOD{} expects {} indices, got {} \
                     (LOD size changes need rebuild_static_geometry)",
                    draw_idx,
                    lod_idx + 1,
                    slice.index_count,
                    alt_idx.len()
                ));
            }
        }
        let v_off = obj.vertex_offset as u64;
        let i_off_bytes = (obj.index_offset * std::mem::size_of::<u32>()) as u64;
        // Static draws keep indices absolute (base_vertex == 0), so rebase
        // mesh-relative u16 indices onto the slot's vertex_offset and widen to
        // u32 before writing, matching the shared u32 index buffer and the
        // streaming upload_mesh path. `v_off` is always a multiple of
        // size_of::<Vertex>() (every region build_draw_list emits starts on a
        // vertex boundary).
        let base = (obj.vertex_offset / std::mem::size_of::<Vertex>()) as u32;
        let lod_byte_offsets: Vec<u64> = obj
            .lod_alternates
            .iter()
            .map(|s| (s.index_offset * std::mem::size_of::<u32>()) as u64)
            .collect();

        self.wait_idle();

        let vert_bytes = bytemuck::cast_slice(vertices);
        self.write_geometry_region(
            &self.geometry.vertex_buffer,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            v_off,
            vert_bytes,
        )?;
        let rebased: Vec<u32> = indices.iter().map(|&i| u32::from(i) + base).collect();
        let idx_bytes = bytemuck::cast_slice(&rebased);
        self.write_geometry_region(
            &self.geometry.index_buffer,
            D3D12_RESOURCE_STATE_INDEX_BUFFER,
            i_off_bytes,
            idx_bytes,
        )?;
        // LOD alternate slots were laid out at init alongside LOD0 in the
        // same shared index buffer. Each alternate shares LOD0's vertex
        // region (LOD decimation never touches vertices), so rebase onto the
        // same `base`.
        for ((_, alt_idx), &alt_off_bytes) in lod_alternates.iter().zip(lod_byte_offsets.iter()) {
            let alt_rebased: Vec<u32> = alt_idx.iter().map(|&i| u32::from(i) + base).collect();
            let alt_bytes = bytemuck::cast_slice(&alt_rebased);
            self.write_geometry_region(
                &self.geometry.index_buffer,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
                alt_off_bytes,
                alt_bytes,
            )?;
        }
        // Refresh the per-LOD switch distances so JSON-side tweaks to
        // `lod_distances` propagate without a process restart.
        let slot = &mut self.draw.objects[draw_idx];
        for ((switch_distance, _), slice) in
            lod_alternates.iter().zip(slot.lod_alternates.iter_mut())
        {
            slice.switch_distance = *switch_distance;
        }
        // The slot now holds different triangles at the same offsets, so its RT
        // BLAS traces the pre-reload positions. Nothing else in the geometry
        // signature moved, so bump the generation (which the signature carries)
        // and flag the topology: the next RT update rebuilds this slot's BLAS
        // rather than reusing the stale one.
        slot.geometry_generation = slot.geometry_generation.wrapping_add(1);
        self.rt_topology_dirty = true;
        Ok(())
    }

    // Return a streamed mesh's geometry region to the sub-allocators and mark
    // the draw non-resident so it is skipped in every pass.
    //
    // `retire_frame` is the frame from which the freed region may be reused:
    // pass `current_frame + frames_in_flight` for a runtime eviction so a
    // still-in-flight command list never has its region overwritten by a
    // later `upload_mesh`, and `0` at init, where nothing has been drawn.
    // The region is not zeroed: the draw leaves the RT-relevant set here, so
    // the next RT update retires its BLAS rather than tracing the vacated
    // bytes, and every raster pass skips a non-resident draw.
    pub(crate) fn evict_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> Result<(), String> {
        let obj = self
            .draw
            .objects
            .get(draw_idx)
            .ok_or_else(|| format!("evict_mesh: draw object {} out of range", draw_idx))?;
        let v_off = obj.vertex_offset as u64;
        let v_len = (obj.vertex_count * std::mem::size_of::<Vertex>()) as u64;
        let i_off = (obj.index_offset * std::mem::size_of::<u32>()) as u64;
        let i_len = (obj.index_count * std::mem::size_of::<u32>()) as u64;
        self.mesh_stream.vtx_alloc.free(v_off, v_len, retire_frame);
        self.mesh_stream.idx_alloc.free(i_off, i_len, retire_frame);
        self.draw.objects[draw_idx].resident = false;
        // The mesh leaves the RT-relevant draw set; the next RT update drops its
        // BLAS (deferred-freed once in-flight traces retire).
        self.rt_topology_dirty = true;
        Ok(())
    }

    // Seed the streamed-mesh sub-allocators with one reserved headroom block
    // (byte ranges in the shared vertex / index buffers), for the
    // shrinkable-seed path.
    //
    // The streamed geometry is not baked into the buffers at build time;
    // instead the buffers carry one zeroed headroom region (sized to the
    // cap-many resident meshes) at these offsets, which `compact_for_streaming`
    // appended before init. `retire_frame 0`: nothing has been drawn yet, so
    // the space is allocatable immediately -- mirrors `setup_chunk_streaming`'s
    // seeding. From then on `upload_mesh` / `evict_mesh` place and free streamed
    // meshes within it. Mirrors `MtlContext::seed_mesh_streaming`.
    pub(crate) fn seed_mesh_streaming(
        &mut self,
        vtx_offset: u64,
        vtx_bytes: u64,
        idx_offset: u64,
        idx_bytes: u64,
    ) {
        self.mesh_stream.vtx_alloc.free(vtx_offset, vtx_bytes, 0);
        self.mesh_stream.vtx_alloc.reclaim(0);
        self.mesh_stream.idx_alloc.free(idx_offset, idx_bytes, 0);
        self.mesh_stream.idx_alloc.reclaim(0);
    }
}
