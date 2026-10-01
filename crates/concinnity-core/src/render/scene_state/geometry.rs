//! Streamed geometry in the shared vertex and index buffers: where each mesh and
//! chunk is placed, and the bytes written there.
//!
//! The shared index buffer is `u32`-typed while mesh and chunk indices arrive
//! as `u16`, so every write widens them and every allocation is sized by the
//! `u32` stride. A streamed mesh keeps absolute indices (rebased onto its
//! vertex region); a chunk keeps mesh-relative ones and draws through
//! `base_vertex`, since it can land past the `u16` index range.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::mem::size_of;

use super::SceneState;
use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::{DrawIndex, DrawObject};
use crate::render::backend::ChunkMesh;
use crate::render::draw_slot::{self, SlotAlloc};
use crate::render::error::{RenderError, RenderResult};
use crate::render::range_alloc::RangeAllocator;

/// One of the two shared geometry buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryBuffer {
    /// The shared vertex buffer.
    Vertex,
    /// The shared `u32` index buffer.
    Index,
}

/// Lands bytes in the shared geometry buffers. Each backend supplies one: a
/// write may go straight into CPU-visible memory or be staged for a copy that
/// runs ahead of the next GPU read of the buffer.
pub trait GeometryWriter {
    /// Write `bytes` into `buffer` at byte `offset`. The region was placed
    /// where no in-flight frame reads it.
    fn write(&mut self, buffer: GeometryBuffer, offset: usize, bytes: &[u8]) -> RenderResult<()>;

    /// A region of `buffer` the scene no longer draws from. Default: nothing,
    /// since every pass skips a non-resident draw and a later placement
    /// overwrites the bytes in full.
    fn release(&mut self, buffer: GeometryBuffer, offset: usize, len: usize) -> RenderResult<()> {
        let _ = (buffer, offset, len);
        Ok(())
    }

    /// Room for `bytes` of streamed writes is about to be needed. Default:
    /// nothing to prepare.
    fn reserve(&mut self, bytes: u64) {
        let _ = bytes;
    }
}

/// The byte-range allocators that place streamed meshes and chunks in the
/// shared buffers, one pair each. Both start empty and are seeded with
/// headroom: the meshes by [`SceneState::seed_mesh_streaming`] or by evicting
/// each build-time region, the chunks by the backend that grows the buffers.
#[derive(Default)]
pub struct GeometryPlacement {
    /// Streamed-mesh vertex ranges.
    pub mesh_vtx: RangeAllocator,
    /// Streamed-mesh index ranges.
    pub mesh_idx: RangeAllocator,
    /// Chunk vertex ranges.
    pub chunk_vtx: RangeAllocator,
    /// Chunk index ranges.
    pub chunk_idx: RangeAllocator,
}

// Place `v_len` vertex bytes and `i_len` index bytes, all or nothing, each on
// its own element boundary: a pool's headroom can start anywhere in its buffer.
// A full pool is `OutOfDeviceMemory`; when only the index pool is full the
// vertex range goes back at retire frame 0, since nothing wrote or drew it.
fn place_mesh(
    vtx: &mut RangeAllocator,
    idx: &mut RangeAllocator,
    v_len: usize,
    i_len: usize,
    what: impl Fn() -> String,
) -> RenderResult<(usize, usize)> {
    let v_off = vtx
        .alloc_aligned(v_len as u64, VERTEX_STRIDE)
        .ok_or_else(|| {
            RenderError::OutOfDeviceMemory(format!(
                "{}: no free vertex space for {v_len} bytes",
                what()
            ))
        })?;
    let Some(i_off) = idx.alloc_aligned(i_len as u64, INDEX_STRIDE) else {
        vtx.free(v_off, v_len as u64, 0);
        return Err(RenderError::OutOfDeviceMemory(format!(
            "{}: no free index space for {i_len} bytes",
            what()
        )));
    };
    Ok((v_off as usize, i_off as usize))
}

// Write a freshly placed mesh's vertex and index bytes. A failed write hands
// both ranges back at `frame`, since no draw will ever read them.
fn write_placed(
    writer: &mut dyn GeometryWriter,
    vtx: &mut RangeAllocator,
    idx: &mut RangeAllocator,
    vertex: (usize, &[u8]),
    index: (usize, &[u8]),
    frame: u64,
) -> RenderResult<()> {
    let written = writer
        .write(GeometryBuffer::Vertex, vertex.0, vertex.1)
        .and_then(|()| writer.write(GeometryBuffer::Index, index.0, index.1));
    if written.is_err() {
        vtx.free(vertex.0 as u64, vertex.1.len() as u64, frame);
        idx.free(index.0 as u64, index.1.len() as u64, frame);
    }
    written
}

// `indices` widened to the shared buffer's `u32` stride, offset by `base`.
fn widen(indices: &[u16], base: u32) -> Vec<u32> {
    indices.iter().map(|&i| u32::from(i) + base).collect()
}

// The vertex index a byte offset into the shared vertex buffer starts at. Every
// region is placed on a vertex boundary, so the division is exact.
fn vertex_base(vertex_offset: usize) -> usize {
    vertex_offset / size_of::<Vertex>()
}

const VERTEX_STRIDE: u64 = size_of::<Vertex>() as u64;
const INDEX_STRIDE: u64 = size_of::<u32>() as u64;

fn out_of_range(op: &str, draw_idx: DrawIndex) -> RenderError {
    RenderError::Other(format!("{op}: draw object {draw_idx} out of range"))
}

impl SceneState {
    /// Bring a streamed mesh resident: place its geometry wherever the mesh
    /// allocators find room, write it, and point the slot at it. `frame`
    /// reclaims the frees that have retired by then. The counts must match the
    /// slot's build-time `vertex_count` / `index_count`.
    pub fn upload_mesh(
        &mut self,
        draw_idx: DrawIndex,
        vertices: &[Vertex],
        indices: &[u16],
        frame: u64,
        writer: &mut dyn GeometryWriter,
    ) -> RenderResult<()> {
        let obj = self
            .draw
            .objects
            .get(draw_idx.index())
            .ok_or_else(|| out_of_range("upload_mesh", draw_idx))?;
        if vertices.len() != obj.vertex_count {
            return Err(RenderError::Other(format!(
                "upload_mesh: draw {} expects {} vertices, got {}",
                draw_idx,
                obj.vertex_count,
                vertices.len()
            )));
        }
        if indices.len() != obj.index_count {
            return Err(RenderError::Other(format!(
                "upload_mesh: draw {} expects {} indices, got {}",
                draw_idx,
                obj.index_count,
                indices.len()
            )));
        }

        let placement = &mut self.placement;
        placement.mesh_vtx.reclaim(frame);
        placement.mesh_idx.reclaim(frame);
        let (v_off, i_off) = place_mesh(
            &mut placement.mesh_vtx,
            &mut placement.mesh_idx,
            size_of_val(vertices),
            indices.len() * size_of::<u32>(),
            || format!("upload_mesh: draw {draw_idx}"),
        )?;
        let rebased = widen(indices, vertex_base(v_off) as u32);
        write_placed(
            writer,
            &mut placement.mesh_vtx,
            &mut placement.mesh_idx,
            (v_off, bytemuck::cast_slice(vertices)),
            (i_off, bytemuck::cast_slice(&rebased)),
            frame,
        )?;

        let obj = &mut self.draw.objects[draw_idx.index()];
        obj.vertex_offset = v_off;
        obj.index_offset = i_off / size_of::<u32>();
        obj.resident = true;
        self.gpu_dirty.rt_topology = true;
        Ok(())
    }

    /// Make a streamed mesh non-resident and return its regions to the mesh
    /// allocators, held until `retire_frame` so no frame still in flight has
    /// them reused underneath it.
    pub fn evict_mesh(
        &mut self,
        draw_idx: DrawIndex,
        retire_frame: u64,
        writer: &mut dyn GeometryWriter,
    ) -> RenderResult<()> {
        let obj = self
            .draw
            .objects
            .get(draw_idx.index())
            .ok_or_else(|| out_of_range("evict_mesh", draw_idx))?;
        let v_off = obj.vertex_offset;
        let v_len = obj.vertex_count * size_of::<Vertex>();
        let i_off = obj.index_offset * size_of::<u32>();
        let i_len = obj.index_count * size_of::<u32>();
        writer.release(GeometryBuffer::Vertex, v_off, v_len)?;
        writer.release(GeometryBuffer::Index, i_off, i_len)?;
        self.placement
            .mesh_vtx
            .free(v_off as u64, v_len as u64, retire_frame);
        self.placement
            .mesh_idx
            .free(i_off as u64, i_len as u64, retire_frame);
        self.draw.objects[draw_idx.index()].resident = false;
        self.gpu_dirty.rt_topology = true;
        Ok(())
    }

    /// Seed the mesh allocators with one headroom block per buffer, free from
    /// the start since nothing has been drawn, and let the writer prepare for
    /// that much streamed geometry.
    pub fn seed_mesh_streaming(
        &mut self,
        vtx_offset: u64,
        vtx_bytes: u64,
        idx_offset: u64,
        idx_bytes: u64,
        writer: &mut dyn GeometryWriter,
    ) {
        self.placement.mesh_vtx.free(vtx_offset, vtx_bytes, 0);
        self.placement.mesh_vtx.reclaim(0);
        self.placement.mesh_idx.free(idx_offset, idx_bytes, 0);
        self.placement.mesh_idx.reclaim(0);
        writer.reserve(vtx_bytes + idx_bytes);
    }

    /// Rewrite a slot's vertices, indices and LOD slices in place, at its
    /// existing offsets. Every count must match the slot's; a size change
    /// needs a full geometry rebuild instead. The per-LOD switch distances are
    /// refreshed, and the slot's geometry generation is bumped so the
    /// ray-tracing BLAS over the old triangles is rebuilt.
    ///
    /// No fence orders this against frames in flight, which suits a
    /// human-paced live edit and nothing on a steady-state path.
    pub fn update_mesh_geometry(
        &mut self,
        draw_idx: DrawIndex,
        vertices: &[Vertex],
        indices: &[u16],
        lod_alternates: &[(f32, Vec<u16>)],
        writer: &mut dyn GeometryWriter,
    ) -> RenderResult<()> {
        let obj = self
            .draw
            .objects
            .get(draw_idx.index())
            .ok_or_else(|| out_of_range("update_mesh_geometry", draw_idx))?;
        check_in_place_counts(obj, draw_idx, vertices, indices, lod_alternates)?;

        let base = vertex_base(obj.vertex_offset) as u32;
        let v_off = obj.vertex_offset;
        let i_off = obj.index_offset * size_of::<u32>();
        let lod_offsets: Vec<usize> = obj
            .lod_alternates
            .iter()
            .map(|s| s.index_offset * size_of::<u32>())
            .collect();
        writer.write(
            GeometryBuffer::Vertex,
            v_off,
            bytemuck::cast_slice(vertices),
        )?;
        let rebased = widen(indices, base);
        writer.write(GeometryBuffer::Index, i_off, bytemuck::cast_slice(&rebased))?;
        for ((_, alt), &offset) in lod_alternates.iter().zip(&lod_offsets) {
            let rebased = widen(alt, base);
            writer.write(
                GeometryBuffer::Index,
                offset,
                bytemuck::cast_slice(&rebased),
            )?;
        }

        let slot = &mut self.draw.objects[draw_idx.index()];
        for ((switch_distance, _), slice) in
            lod_alternates.iter().zip(slot.lod_alternates.iter_mut())
        {
            slice.switch_distance = *switch_distance;
        }
        slot.geometry_generation = slot.geometry_generation.wrapping_add(1);
        self.gpu_dirty.rt_topology = true;
        Ok(())
    }

    /// Place one streamed chunk in the chunk headroom and write its draw at
    /// `dst`. The chunk is never culled: the streaming window already bounds
    /// how many are resident.
    pub fn add_chunk_mesh(
        &mut self,
        mesh: ChunkMesh<'_>,
        dst: SlotAlloc,
        writer: &mut dyn GeometryWriter,
    ) -> RenderResult<()> {
        let ChunkMesh {
            verts,
            idxs,
            model,
            texture_slot,
            normal_map_slot,
            material,
            frame,
        } = mesh;
        if verts.is_empty() || idxs.is_empty() {
            return Err(RenderError::Other(
                "add_chunk_mesh: empty chunk geometry".into(),
            ));
        }
        let placement = &mut self.placement;
        placement.chunk_vtx.reclaim(frame);
        placement.chunk_idx.reclaim(frame);
        let (v_off, i_off) = place_mesh(
            &mut placement.chunk_vtx,
            &mut placement.chunk_idx,
            size_of_val(verts),
            idxs.len() * size_of::<u32>(),
            || "add_chunk_mesh".into(),
        )?;
        let widened = widen(idxs, 0);
        write_placed(
            writer,
            &mut placement.chunk_vtx,
            &mut placement.chunk_idx,
            (v_off, bytemuck::cast_slice(verts)),
            (i_off, bytemuck::cast_slice(&widened)),
            frame,
        )?;

        let obj = DrawObject {
            vertex_offset: v_off,
            vertex_count: verts.len(),
            index_offset: i_off / size_of::<u32>(),
            index_count: idxs.len(),
            base_vertex: vertex_base(v_off) as i32,
            geometry_generation: 0,
            model,
            texture_slot,
            normal_map_slot,
            material,
            shader_bucket: 0,
            visible: true,
            resident: true,
            bb_min: [f32::NAN; 3],
            bb_max: [f32::NAN; 3],
            cull_distance: 0.0,
            // A chunk's distance LOD is the streaming window's choice of mesh
            // detail, so it carries no per-draw alternates.
            lod_alternates: Vec::new(),
        };
        self.place_draw_object(obj, dst);
        self.gpu_dirty.rt_topology = true;
        Ok(())
    }

    /// Hide a streamed chunk and return its regions to the chunk allocators,
    /// held until `retire_frame`.
    pub fn remove_chunk_mesh(
        &mut self,
        draw_idx: DrawIndex,
        retire_frame: u64,
        writer: &mut dyn GeometryWriter,
    ) -> RenderResult<()> {
        let region = draw_slot::retire_chunk_slot(&mut self.draw.objects, draw_idx)
            .map_err(RenderError::Other)?;
        // The slot is already hidden, so its ranges go back first: a failed
        // release below must not strand them.
        self.placement
            .chunk_vtx
            .free(region.vertex_offset, region.vertex_bytes, retire_frame);
        self.placement
            .chunk_idx
            .free(region.index_offset, region.index_bytes, retire_frame);
        self.gpu_dirty.rt_topology = true;
        writer.release(
            GeometryBuffer::Vertex,
            region.vertex_offset as usize,
            region.vertex_bytes as usize,
        )?;
        writer.release(
            GeometryBuffer::Index,
            region.index_offset as usize,
            region.index_bytes as usize,
        )
    }
}

// The in-place rewrite fits only a slot whose every count is unchanged.
fn check_in_place_counts(
    obj: &DrawObject,
    draw_idx: DrawIndex,
    vertices: &[Vertex],
    indices: &[u16],
    lod_alternates: &[(f32, Vec<u16>)],
) -> RenderResult<()> {
    if vertices.len() != obj.vertex_count {
        return Err(RenderError::Other(format!(
            "update_mesh_geometry: draw {} expects {} vertices, got {} \
             (in-place path is size-matched only; size changes route through \
             rebuild_static_geometry)",
            draw_idx,
            obj.vertex_count,
            vertices.len()
        )));
    }
    if indices.len() != obj.index_count {
        return Err(RenderError::Other(format!(
            "update_mesh_geometry: draw {} expects {} indices, got {} \
             (in-place path is size-matched only; size changes route through \
             rebuild_static_geometry)",
            draw_idx,
            obj.index_count,
            indices.len()
        )));
    }
    if lod_alternates.len() != obj.lod_alternates.len() {
        return Err(RenderError::Other(format!(
            "update_mesh_geometry: draw {} expects {} LOD alternate(s), got {} \
             (LOD-count changes need rebuild_static_geometry)",
            draw_idx,
            obj.lod_alternates.len(),
            lod_alternates.len()
        )));
    }
    for (lod_idx, ((_, alt), slice)) in lod_alternates.iter().zip(&obj.lod_alternates).enumerate() {
        if alt.len() != slice.index_count {
            return Err(RenderError::Other(format!(
                "update_mesh_geometry: draw {} LOD{} expects {} indices, got {} \
                 (LOD size changes need rebuild_static_geometry)",
                draw_idx,
                lod_idx + 1,
                slice.index_count,
                alt.len()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::render_types::{LodSlice, MaterialUniforms};
    use crate::render::scene_state::DrawList;
    use crate::test_support::draw_object;
    use alloc::string::ToString;
    use alloc::vec;

    const VERTEX: usize = size_of::<Vertex>();

    // Records every call; `release` and `reserve` are recorded too, so a test
    // can tell a default hook from an overridden one. `writes_before_failure`
    // refuses every write once that many have landed; `fail_releases` refuses
    // every release.
    #[derive(Default)]
    struct Recorder {
        writes: Vec<(GeometryBuffer, usize, Vec<u8>)>,
        releases: Vec<(GeometryBuffer, usize, usize)>,
        reserved: Vec<u64>,
        writes_before_failure: Option<usize>,
        fail_releases: bool,
    }

    impl Recorder {
        fn failing_after(writes: usize) -> Self {
            Self {
                writes_before_failure: Some(writes),
                ..Self::default()
            }
        }
    }

    impl GeometryWriter for Recorder {
        fn write(
            &mut self,
            buffer: GeometryBuffer,
            offset: usize,
            bytes: &[u8],
        ) -> RenderResult<()> {
            if self.writes_before_failure == Some(self.writes.len()) {
                return Err(RenderError::Other("write refused".into()));
            }
            self.writes.push((buffer, offset, bytes.to_vec()));
            Ok(())
        }
        fn release(
            &mut self,
            buffer: GeometryBuffer,
            offset: usize,
            len: usize,
        ) -> RenderResult<()> {
            if self.fail_releases {
                return Err(RenderError::Other("release refused".into()));
            }
            self.releases.push((buffer, offset, len));
            Ok(())
        }
        fn reserve(&mut self, bytes: u64) {
            self.reserved.push(bytes);
        }
    }

    // A writer that keeps every default hook.
    struct WriteOnly;

    impl GeometryWriter for WriteOnly {
        fn write(&mut self, _: GeometryBuffer, _: usize, _: &[u8]) -> RenderResult<()> {
            Ok(())
        }
    }

    fn vertices(n: usize) -> Vec<Vertex> {
        vec![crate::test_support::vertex(); n]
    }

    fn indices(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    // One streamed slot expecting 3 vertices and 3 indices, with the mesh
    // allocators seeded with room for `meshes` such meshes.
    fn streamed(meshes: usize) -> SceneState {
        let mut obj = draw_object();
        obj.vertex_count = 3;
        obj.index_count = 3;
        obj.resident = false;
        let mut scene = SceneState::new(DrawList::unreserved(vec![obj], 0), [0.0; 4]);
        scene.seed_mesh_streaming(
            0,
            (meshes * 3 * VERTEX) as u64,
            0,
            (meshes * 3 * 4) as u64,
            &mut WriteOnly,
        );
        scene
    }

    #[test]
    fn an_upload_places_rebases_and_marks_the_slot_resident() {
        let mut scene = streamed(2);
        // Occupy the front of the pools so the mesh lands past them.
        scene.placement.mesh_vtx.alloc(3 * VERTEX as u64);
        scene.placement.mesh_idx.alloc(12);
        let mut w = Recorder::default();
        scene
            .upload_mesh(DrawIndex(0), &vertices(3), &[0, 1, 2], 0, &mut w)
            .expect("room for the mesh");
        assert_eq!(w.writes.len(), 2);
        assert_eq!(
            (w.writes[0].0, w.writes[0].1),
            (GeometryBuffer::Vertex, 3 * VERTEX)
        );
        assert_eq!(w.writes[0].2.len(), 3 * VERTEX);
        assert_eq!((w.writes[1].0, w.writes[1].1), (GeometryBuffer::Index, 12));
        assert_eq!(indices(&w.writes[1].2), vec![3, 4, 5]);
        let obj = &scene.draw.objects[0];
        assert_eq!((obj.vertex_offset, obj.index_offset), (3 * VERTEX, 3));
        assert!(obj.resident);
        assert!(scene.gpu_dirty.rt_topology);
    }

    #[test]
    fn an_upload_with_the_wrong_counts_is_refused() {
        let mut scene = streamed(1);
        let err = scene
            .upload_mesh(DrawIndex(0), &vertices(2), &[0, 1, 2], 0, &mut WriteOnly)
            .unwrap_err();
        assert!(
            err.to_string().contains("expects 3 vertices, got 2"),
            "{err}"
        );
        let err = scene
            .upload_mesh(DrawIndex(0), &vertices(3), &[0, 1], 0, &mut WriteOnly)
            .unwrap_err();
        assert!(
            err.to_string().contains("expects 3 indices, got 2"),
            "{err}"
        );
        let err = scene
            .upload_mesh(DrawIndex(7), &vertices(3), &[0, 1, 2], 0, &mut WriteOnly)
            .unwrap_err();
        assert!(
            err.to_string().contains("draw object 7 out of range"),
            "{err}"
        );
        assert!(!scene.gpu_dirty.rt_topology);
    }

    #[test]
    fn an_upload_into_full_pools_is_out_of_device_memory() {
        let mut scene = streamed(0);
        let err = scene
            .upload_mesh(DrawIndex(0), &vertices(3), &[0, 1, 2], 0, &mut WriteOnly)
            .unwrap_err();
        assert!(matches!(err, RenderError::OutOfDeviceMemory(_)), "{err}");
        assert!(err.to_string().contains("upload_mesh: draw 0"), "{err}");
    }

    #[test]
    fn a_full_index_pool_hands_the_vertex_range_back() {
        let mut scene = streamed(1);
        scene.placement.mesh_idx.alloc(12);
        assert!(
            scene
                .upload_mesh(DrawIndex(0), &vertices(3), &[0, 1, 2], 0, &mut WriteOnly)
                .is_err()
        );
        scene.placement.mesh_vtx.reclaim(0);
        assert_eq!(scene.placement.mesh_vtx.free_bytes(), 3 * VERTEX as u64);
    }

    #[test]
    fn a_failed_write_leaves_the_slot_non_resident_and_the_pools_whole() {
        // Fail the vertex write, then the index write after the vertex landed.
        for landed in [0, 1] {
            let mut scene = streamed(1);
            let mut w = Recorder::failing_after(landed);
            assert!(
                scene
                    .upload_mesh(DrawIndex(0), &vertices(3), &[0, 1, 2], 7, &mut w)
                    .is_err()
            );
            assert!(!scene.draw.objects[0].resident);
            scene.placement.mesh_vtx.reclaim(7);
            scene.placement.mesh_idx.reclaim(7);
            assert_eq!(scene.placement.mesh_vtx.free_bytes(), 3 * VERTEX as u64);
            assert_eq!(scene.placement.mesh_idx.free_bytes(), 12);
        }
    }

    #[test]
    fn an_eviction_releases_and_retires_the_regions() {
        let mut scene = streamed(1);
        scene
            .upload_mesh(DrawIndex(0), &vertices(3), &[0, 1, 2], 0, &mut WriteOnly)
            .expect("room for the mesh");
        scene.gpu_dirty.rt_topology = false;
        let mut w = Recorder::default();
        scene
            .evict_mesh(DrawIndex(0), 5, &mut w)
            .expect("slot 0 exists");
        assert_eq!(
            w.releases,
            vec![
                (GeometryBuffer::Vertex, 0, 3 * VERTEX),
                (GeometryBuffer::Index, 0, 12)
            ]
        );
        assert!(!scene.draw.objects[0].resident);
        assert!(scene.gpu_dirty.rt_topology);
        // Held until the retire frame.
        scene.placement.mesh_vtx.reclaim(4);
        assert_eq!(scene.placement.mesh_vtx.free_bytes(), 0);
        scene.placement.mesh_vtx.reclaim(5);
        assert_eq!(scene.placement.mesh_vtx.free_bytes(), 3 * VERTEX as u64);
    }

    #[test]
    fn evicting_an_out_of_range_slot_is_an_error() {
        let mut scene = streamed(1);
        let err = scene
            .evict_mesh(DrawIndex(3), 0, &mut WriteOnly)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("evict_mesh: draw object 3 out of range"),
            "{err}"
        );
    }

    #[test]
    fn seeding_frees_the_headroom_and_reserves_staging() {
        let mut scene = streamed(0);
        let mut w = Recorder::default();
        scene.seed_mesh_streaming(64, 128, 32, 48, &mut w);
        assert_eq!(scene.placement.mesh_vtx.free_bytes(), 128);
        assert_eq!(scene.placement.mesh_idx.free_bytes(), 48);
        assert_eq!(w.reserved, vec![176]);
    }

    // A build-time slot at vertex 2 with one LOD slice, for the in-place path.
    fn in_place() -> SceneState {
        let mut obj = draw_object();
        obj.vertex_offset = 2 * VERTEX;
        obj.vertex_count = 3;
        obj.index_offset = 4;
        obj.index_count = 3;
        obj.lod_alternates = vec![LodSlice {
            index_offset: 10,
            index_count: 2,
            switch_distance: 5.0,
        }];
        SceneState::new(DrawList::unreserved(vec![obj], 0), [0.0; 4])
    }

    #[test]
    fn an_in_place_update_rewrites_every_region_and_bumps_the_generation() {
        let mut scene = in_place();
        let mut w = Recorder::default();
        scene
            .update_mesh_geometry(
                DrawIndex(0),
                &vertices(3),
                &[0, 1, 2],
                &[(9.0, vec![0, 2])],
                &mut w,
            )
            .expect("counts match");
        assert_eq!(w.writes.len(), 3);
        assert_eq!(
            (w.writes[0].0, w.writes[0].1),
            (GeometryBuffer::Vertex, 2 * VERTEX)
        );
        assert_eq!((w.writes[1].0, w.writes[1].1), (GeometryBuffer::Index, 16));
        assert_eq!(indices(&w.writes[1].2), vec![2, 3, 4]);
        assert_eq!((w.writes[2].0, w.writes[2].1), (GeometryBuffer::Index, 40));
        assert_eq!(indices(&w.writes[2].2), vec![2, 4]);
        let obj = &scene.draw.objects[0];
        assert_eq!(obj.lod_alternates[0].switch_distance, 9.0);
        assert_eq!(obj.geometry_generation, 1);
        assert!(scene.gpu_dirty.rt_topology);
    }

    #[test]
    fn an_in_place_update_refuses_any_size_change() {
        let mut scene = in_place();
        let mut refusal = |verts: &[Vertex], idxs: &[u16], lods: &[(f32, Vec<u16>)]| {
            scene
                .update_mesh_geometry(DrawIndex(0), verts, idxs, lods, &mut WriteOnly)
                .unwrap_err()
                .to_string()
        };
        let lods = [(9.0, vec![0, 2])];
        let err = refusal(&vertices(4), &[0, 1, 2], &lods);
        assert!(err.contains("expects 3 vertices, got 4"), "{err}");
        let err = refusal(&vertices(3), &[0, 1], &lods);
        assert!(err.contains("expects 3 indices, got 2"), "{err}");
        let err = refusal(&vertices(3), &[0, 1, 2], &[]);
        assert!(err.contains("expects 1 LOD alternate(s), got 0"), "{err}");
        let err = refusal(&vertices(3), &[0, 1, 2], &[(9.0, vec![0])]);
        assert!(err.contains("LOD1 expects 2 indices, got 1"), "{err}");
        assert_eq!(scene.draw.objects[0].geometry_generation, 0);
        assert!(!scene.gpu_dirty.rt_topology);
    }

    fn chunk<'a>(verts: &'a [Vertex], idxs: &'a [u16]) -> ChunkMesh<'a> {
        ChunkMesh {
            verts,
            idxs,
            model: crate::transform::IDENTITY,
            texture_slot: 2,
            normal_map_slot: 3,
            material: MaterialUniforms::DEFAULT,
            frame: 0,
        }
    }

    // A scene with chunk headroom for `vertices` vertices and `indices` indices.
    fn chunked(vertices: usize, indices: usize) -> SceneState {
        let mut scene = SceneState::new(DrawList::with_runtime_reserve(vec![], 0, 4), [0.0; 4]);
        scene
            .placement
            .chunk_vtx
            .free(0, (vertices * VERTEX) as u64, 0);
        scene.placement.chunk_idx.free(0, (indices * 4) as u64, 0);
        scene
    }

    #[test]
    fn a_chunk_keeps_relative_indices_and_draws_through_base_vertex() {
        let mut scene = chunked(8, 8);
        scene.placement.chunk_vtx.reclaim(0);
        scene.placement.chunk_vtx.alloc(2 * VERTEX as u64);
        let mut w = Recorder::default();
        scene
            .add_chunk_mesh(
                chunk(&vertices(3), &[0, 1, 2]),
                SlotAlloc::Append(DrawIndex(0)),
                &mut w,
            )
            .expect("room for the chunk");
        assert_eq!(indices(&w.writes[1].2), vec![0, 1, 2]);
        let obj = &scene.draw.objects[0];
        assert_eq!(obj.base_vertex, 2);
        assert_eq!((obj.texture_slot, obj.normal_map_slot), (2, 3));
        assert!(obj.visible && obj.resident && !obj.cullable());
        assert!(obj.lod_alternates.is_empty());
        assert!(scene.gpu_dirty.rt_topology);
    }

    #[test]
    fn a_chunk_in_headroom_off_a_vertex_boundary_still_draws_its_own_vertices() {
        // An empty world's shared vertex buffer is a 4-byte placeholder, so the
        // chunk headroom appended after it starts mid-vertex.
        let mut scene = SceneState::new(DrawList::with_runtime_reserve(vec![], 0, 4), [0.0; 4]);
        scene.placement.chunk_vtx.free(4, (4 * VERTEX) as u64, 0);
        scene.placement.chunk_idx.free(2, 16, 0);
        let mut w = Recorder::default();
        scene
            .add_chunk_mesh(
                chunk(&vertices(3), &[0, 1, 2]),
                SlotAlloc::Append(DrawIndex(0)),
                &mut w,
            )
            .expect("room for the chunk");
        let obj = &scene.draw.objects[0];
        assert_eq!(obj.vertex_offset, VERTEX);
        assert_eq!(obj.base_vertex as usize * VERTEX, w.writes[0].1);
        assert_eq!(obj.index_offset * 4, w.writes[1].1);
    }

    #[test]
    fn a_failed_chunk_write_hands_its_ranges_back() {
        let mut scene = chunked(3, 3);
        let mut w = Recorder::failing_after(1);
        assert!(
            scene
                .add_chunk_mesh(
                    chunk(&vertices(3), &[0, 1, 2]),
                    SlotAlloc::Append(DrawIndex(0)),
                    &mut w
                )
                .is_err()
        );
        assert!(scene.draw.objects.is_empty());
        scene.placement.chunk_vtx.reclaim(0);
        scene.placement.chunk_idx.reclaim(0);
        assert_eq!(scene.placement.chunk_vtx.free_bytes(), 3 * VERTEX as u64);
        assert_eq!(scene.placement.chunk_idx.free_bytes(), 12);
    }

    #[test]
    fn a_failed_chunk_release_still_retires_its_ranges() {
        let mut scene = chunked(3, 3);
        scene
            .add_chunk_mesh(
                chunk(&vertices(3), &[0, 1, 2]),
                SlotAlloc::Append(DrawIndex(0)),
                &mut WriteOnly,
            )
            .expect("room for the chunk");
        scene.gpu_dirty.rt_topology = false;
        let mut w = Recorder {
            fail_releases: true,
            ..Recorder::default()
        };
        assert!(scene.remove_chunk_mesh(DrawIndex(0), 2, &mut w).is_err());
        assert!(scene.gpu_dirty.rt_topology);
        scene.placement.chunk_vtx.reclaim(2);
        scene.placement.chunk_idx.reclaim(2);
        assert_eq!(scene.placement.chunk_vtx.free_bytes(), 3 * VERTEX as u64);
        assert_eq!(scene.placement.chunk_idx.free_bytes(), 12);
    }

    #[test]
    fn an_empty_chunk_is_refused() {
        let mut scene = chunked(8, 8);
        let err = scene
            .add_chunk_mesh(
                chunk(&[], &[0]),
                SlotAlloc::Append(DrawIndex(0)),
                &mut WriteOnly,
            )
            .unwrap_err();
        assert!(err.to_string().contains("empty chunk geometry"), "{err}");
    }

    #[test]
    fn removing_a_chunk_releases_and_retires_its_regions() {
        let mut scene = chunked(3, 3);
        scene
            .add_chunk_mesh(
                chunk(&vertices(3), &[0, 1, 2]),
                SlotAlloc::Append(DrawIndex(0)),
                &mut WriteOnly,
            )
            .expect("room for the chunk");
        let mut w = Recorder::default();
        scene
            .remove_chunk_mesh(DrawIndex(0), 3, &mut w)
            .expect("slot 0 exists");
        assert_eq!(
            w.releases,
            vec![
                (GeometryBuffer::Vertex, 0, 3 * VERTEX),
                (GeometryBuffer::Index, 0, 12)
            ]
        );
        assert!(!scene.draw.objects[0].visible);
        scene.placement.chunk_idx.reclaim(3);
        assert_eq!(scene.placement.chunk_idx.free_bytes(), 12);
        assert!(scene.remove_chunk_mesh(DrawIndex(4), 3, &mut w).is_err());
    }

    #[test]
    fn the_default_hooks_do_nothing() {
        let mut w = WriteOnly;
        assert!(w.release(GeometryBuffer::Index, 0, 8).is_ok());
        w.reserve(64);
    }
}
