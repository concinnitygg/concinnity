//! The shared static and skinned vertex / index buffers a size-changing asset
//! hot-reload reads back and replaces whole; the repack itself is the
//! engine's. The DEFAULT-heap buffers force a CPU round-trip (READBACK staging
//! for the old contents, UPLOAD staging for the new ones) where Metal can just
//! read `StorageModeShared` `contents()`.

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::mesh_payload::{SkinnedVertex, Vertex};
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::backend::{GeometryBuffers, GeometryRebuild, RenderBackend};
use concinnity_engine::render::rt_accel::geom;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT;

use super::super::allocator::PooledBuffer;
use super::super::com;
use super::super::context::DxContext;
use super::super::texture::{one_shot_submit, transition_barrier};
use crate::directx::error::map_hresult;

// Byte capacity of a buffer resource. `Width` is the buffer's size in bytes.
fn buffer_bytes(res: &ID3D12Resource) -> u64 {
    // SAFETY: a property query on a live resource; it only reads.
    unsafe { res.GetDesc() }.Width
}

// Map a READBACK-heap buffer and copy `count` `T`s out into a CPU Vec. The
// caller has already gated the GPU writes (via `one_shot_submit`'s internal
// fence wait), so the memcpy sees fully committed bytes. `T` must match the
// buffer's stride exactly.
fn read_typed_vec<T: Copy>(src: &ID3D12Resource, count: usize) -> RenderResult<Vec<T>> {
    let want = (count * size_of::<T>()) as u64;
    let have = buffer_bytes(src);
    assert!(
        want <= have,
        "read_typed_vec: {want} bytes exceeds readback buffer {have}"
    );
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local that
    // receives the mapping.
    unsafe { src.Map(0, None, Some(&raw mut ptr)) }
        .map_err(|e| map_hresult(e.code(), "rebuild readback map"))?;
    let mut out: Vec<T> = Vec::with_capacity(count);
    // SAFETY: `Map` returned the live mapping of a READBACK buffer the assert above proved holds at
    // least `count` `T`s, `out` reserved that many, and the two are separate allocations.
    unsafe { std::ptr::copy_nonoverlapping(ptr as *const T, out.as_mut_ptr(), count) };
    // SAFETY: the copy above initialized all `count` elements, within the reserved capacity.
    unsafe { out.set_len(count) };
    // SAFETY: unmaps subresource 0 of `src`, mapped above; the copy is done.
    unsafe { src.Unmap(0, None) };
    Ok(out)
}

// Map an UPLOAD-heap buffer and copy `bytes` into it. Standard UPLOAD-heap
// idiom: Map (CPU writes), copy_nonoverlapping, Unmap (driver flushes).
fn write_upload_buffer(dest: &ID3D12Resource, bytes: &[u8]) -> RenderResult<()> {
    let have = buffer_bytes(dest);
    assert!(
        bytes.len() as u64 <= have,
        "write_upload_buffer: {} bytes exceeds upload buffer {have}",
        bytes.len()
    );
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local that
    // receives the mapping.
    unsafe { dest.Map(0, None, Some(&raw mut ptr)) }
        .map_err(|e| map_hresult(e.code(), "rebuild upload map"))?;
    // SAFETY: `Map` returned the live mapping of an UPLOAD buffer the assert above proved is at
    // least `bytes.len()` long, and the source is a separate allocation, so the ranges cannot
    // overlap.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len()) };
    // SAFETY: unmaps subresource 0 of `dest`, mapped above; the write is done.
    unsafe { dest.Unmap(0, None) };
    Ok(())
}

impl DxContext {
    // Read a vertex + index buffer pair back to CPU memory through READBACK
    // staging (DEFAULT-heap resources are not CPU-mappable). One one-shot
    // submit moves both to COPY_SOURCE, copies each into its staging buffer and
    // moves them back; the caller drained every reader, and the one-shot's
    // fence wait gates the staging side.
    fn read_back_pair<V: Copy>(
        &self,
        vertex_buffer: &ID3D12Resource,
        index_buffer: &ID3D12Resource,
        v_bytes: u64,
        i_bytes: u64,
    ) -> RenderResult<(Vec<V>, Vec<u32>)> {
        let v_readback = self.hw.alloc.alloc_buffer(
            v_bytes,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        let i_readback = self.hw.alloc.alloc_buffer(
            i_bytes,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        one_shot_submit(&self.hw.device, &self.hw.command_queue, |cmd| {
            let v_src = transition_barrier(
                vertex_buffer,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            let i_src = transition_barrier(
                index_buffer,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            // SAFETY: the one-shot list is recording, and the caller drained every reader of both
            // buffers.
            unsafe { cmd.ResourceBarrier(&[v_src, i_src]) };
            // SAFETY: the vertex buffer is in COPY_SOURCE from the barrier above, and its staging
            // buffer was sized `v_bytes` for it.
            unsafe { cmd.CopyBufferRegion(&*v_readback, 0, vertex_buffer, 0, v_bytes) };
            // SAFETY: the index buffer is in COPY_SOURCE from the barrier above, and its staging
            // buffer was sized `i_bytes` for it.
            unsafe { cmd.CopyBufferRegion(&*i_readback, 0, index_buffer, 0, i_bytes) };
            let v_back = transition_barrier(
                vertex_buffer,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            );
            let i_back = transition_barrier(
                index_buffer,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
            );
            // SAFETY: the list is recording, and both buffers are live through the fence wait.
            unsafe { cmd.ResourceBarrier(&[v_back, i_back]) };
        })?;
        let v_count = usize::saturating_from(v_bytes) / size_of::<V>();
        let i_count = usize::saturating_from(i_bytes) / size_of::<u32>();
        Ok((
            read_typed_vec(&v_readback, v_count)?,
            read_typed_vec(&i_readback, i_count)?,
        ))
    }

    // Fresh DEFAULT-heap buffers holding `vertices` and `indices` (the index
    // buffer `i_bytes` long), filled through UPLOAD staging in one one-shot
    // submit and left in their read states. The new resources start in COMMON,
    // which CopyBufferRegion promotes to COPY_DEST.
    fn upload_pair(
        &self,
        vertices: &[u8],
        indices: &[u8],
        i_bytes: u64,
    ) -> RenderResult<(PooledBuffer, PooledBuffer)> {
        let v_bytes = vertices.len() as u64;
        let alloc = |bytes, heap, state| self.hw.alloc.alloc_buffer(bytes, heap, state);
        let vbuf = alloc(
            v_bytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
        )?;
        let ibuf = alloc(
            i_bytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
        )?;
        let v_upload = alloc(
            v_bytes,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let i_upload = alloc(
            i_bytes,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        write_upload_buffer(&v_upload, vertices)?;
        write_upload_buffer(&i_upload, indices)?;
        let index_len = indices.len() as u64;
        one_shot_submit(&self.hw.device, &self.hw.command_queue, |cmd| {
            // SAFETY: the one-shot list is recording, the new buffer promotes from COMMON to
            // COPY_DEST, and `v_upload` holds the vertices and outlives the fence wait.
            unsafe { cmd.CopyBufferRegion(&*vbuf, 0, &*v_upload, 0, v_bytes) };
            // SAFETY: as for the vertices, with `i_upload` holding the indices.
            unsafe { cmd.CopyBufferRegion(&*ibuf, 0, &*i_upload, 0, index_len) };
            let v_dst = transition_barrier(
                &vbuf,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            );
            let i_dst = transition_barrier(
                &ibuf,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
            );
            // SAFETY: the list is recording, and both new buffers are owned by this function.
            unsafe { cmd.ResourceBarrier(&[v_dst, i_dst]) };
        })?;
        Ok((vbuf, ibuf))
    }
}

impl GeometryBuffers for DxContext {
    fn read_static_geometry(&mut self) -> RenderResult<(Vec<Vertex>, Vec<u32>)> {
        self.wait_idle();
        let geometry = &self.scene.geometry;
        self.read_back_pair(
            &geometry.vertex_buffer,
            &geometry.index_buffer,
            u64::from(geometry.vertex_buffer_view.SizeInBytes),
            u64::from(geometry.index_buffer_view.SizeInBytes),
        )
    }

    // The old buffers drop once the views point at the new pair; the read's
    // drain already gated every frame that referenced them.
    fn replace_static_geometry(
        &mut self,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> RenderResult<()> {
        let v_bytes = size_of_val(vertices) as u64;
        let i_bytes = size_of_val(indices) as u64;
        let (vbuf, ibuf) = self.upload_pair(
            bytemuck::cast_slice(vertices),
            bytemuck::cast_slice(indices),
            i_bytes,
        )?;
        let geometry = &mut self.scene.geometry;
        geometry.vertex_buffer_view = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: com::gpu_va(&vbuf),
            SizeInBytes: u32::saturating_from(v_bytes),
            StrideInBytes: u32::saturating_from(size_of::<Vertex>()),
        };
        geometry.index_buffer_view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: com::gpu_va(&ibuf),
            SizeInBytes: u32::saturating_from(i_bytes),
            Format: DXGI_FORMAT_R32_UINT,
        };
        geometry.vertex_buffer = vbuf;
        geometry.index_buffer = ibuf;
        Ok(())
    }

    // The RT acceleration structure was built against the buffers just
    // dropped and the per-draw offsets just rewritten, so its BLAS hold the old
    // geometry. Rebuild it over the fresh layout; the GPU is still quiet.
    fn static_geometry_moved(&mut self) {
        if self.rt.accel.is_some() {
            self.rebuild_rt_accel();
        }
    }

    fn read_skinned_geometry(&mut self) -> RenderResult<(Vec<SkinnedVertex>, Vec<u32>)> {
        let (Some(vertex_buffer), Some(index_buffer)) = (
            self.skinned.vertex_buffer.clone(),
            self.skinned.index_buffer.clone(),
        ) else {
            return Err(RenderError::Other(
                "rebuild_skinned_geometry: no skinned buffers (was upload_skinned called?)".into(),
            ));
        };
        self.wait_idle();
        self.read_back_pair(
            &vertex_buffer,
            &index_buffer,
            u64::from(self.skinned.vertex_buffer_view.SizeInBytes),
            u64::from(self.skinned.index_buffer_view.SizeInBytes),
        )
    }

    // The skinned IB stays `DXGI_FORMAT_R32_UINT`, sized in whole u32 words
    // (see `upload_skinned`).
    fn replace_skinned_geometry(
        &mut self,
        vertices: &[SkinnedVertex],
        indices: &[u32],
    ) -> RenderResult<()> {
        let v_bytes = size_of_val(vertices) as u64;
        let i_bytes = size_of_val(indices) as u64;
        let (vbuf, ibuf) = self.upload_pair(
            bytemuck::cast_slice(vertices),
            bytemuck::cast_slice(indices),
            geom::skinned_index_buffer_bytes(indices.len()) as u64,
        )?;
        self.skinned.vertex_buffer_view = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: com::gpu_va(&vbuf),
            SizeInBytes: u32::saturating_from(v_bytes),
            StrideInBytes: u32::saturating_from(size_of::<SkinnedVertex>()),
        };
        self.skinned.index_buffer_view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: com::gpu_va(&ibuf),
            SizeInBytes: u32::saturating_from(i_bytes),
            Format: DXGI_FORMAT_R32_UINT,
        };
        self.skinned.vertex_buffer = Some(vbuf);
        self.skinned.index_buffer = Some(ibuf);
        Ok(())
    }
}

impl GeometryRebuild for DxContext {}
