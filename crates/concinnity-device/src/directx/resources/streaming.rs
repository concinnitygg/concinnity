//! `VoxelWorld` chunk streaming for DxContext: the init-time headroom growth
//! that seeds the chunk allocators. Placing, moving and removing chunks is
//! scene bookkeeping, done in `SceneState`.

use concinnity_core::gfx::mesh_payload::Vertex;
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
    ) -> error::RenderResult<()> {
        self.wait_idle();
        let old_v_len = self.scene.geometry.vertex_buffer_view.SizeInBytes as u64;
        let old_i_len = self.scene.geometry.index_buffer_view.SizeInBytes as u64;
        let new_v_len = old_v_len + chunk_vtx_bytes as u64;
        let new_i_len = old_i_len + chunk_idx_bytes as u64;

        // Buffers are created in COMMON; the CopyBufferRegion below implicitly
        // promotes the destination COMMON -> COPY_DEST.
        let new_vbuf = self.hw.alloc.alloc_buffer(
            new_v_len,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
        )?;
        let new_ibuf = self.hw.alloc.alloc_buffer(
            new_i_len,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
        )?;

        // Copy the build-time geometry into the start of the grown buffers so
        // every existing draw's offsets stay valid.
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        one_shot_submit(&self.hw.device, &self.hw.command_queue, |cmd| unsafe {
            let v_src = transition_barrier(
                &self.scene.geometry.vertex_buffer,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            let i_src = transition_barrier(
                &self.scene.geometry.index_buffer,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            cmd.ResourceBarrier(&[v_src, i_src]);
            cmd.CopyBufferRegion(
                &*new_vbuf,
                0,
                &*self.scene.geometry.vertex_buffer,
                0,
                old_v_len,
            );
            cmd.CopyBufferRegion(
                &*new_ibuf,
                0,
                &*self.scene.geometry.index_buffer,
                0,
                old_i_len,
            );
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

        self.scene.geometry.vertex_buffer_view = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: com::gpu_va(&new_vbuf),
            SizeInBytes: new_v_len as u32,
            StrideInBytes: std::mem::size_of::<Vertex>() as u32,
        };
        self.scene.geometry.index_buffer_view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: com::gpu_va(&new_ibuf),
            SizeInBytes: new_i_len as u32,
            // Static IB is u32 (matches the `Format` chosen in init/mod.rs).
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT,
        };
        self.scene.geometry.vertex_buffer = new_vbuf;
        self.scene.geometry.index_buffer = new_ibuf;

        self.geometry_uploads
            .get_mut()
            .reserve(&self.hw.alloc, (chunk_vtx_bytes + chunk_idx_bytes) as u64)?;

        // Seed the chunk allocators with the appended headroom. retire_frame 0:
        // nothing has been drawn, so the space is reusable immediately.
        self.state
            .placement
            .chunk_vtx
            .free(old_v_len, chunk_vtx_bytes as u64, 0);
        self.state
            .placement
            .chunk_idx
            .free(old_i_len, chunk_idx_bytes as u64, 0);
        Ok(())
    }
}
