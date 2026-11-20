//! `VoxelWorld` chunk streaming for `MtlContext`: growing the shared buffers by the
//! chunk headroom and seeding the chunk allocators with it. Placing, moving and
//! removing chunks is scene bookkeeping, done in `SceneState`.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_engine::render::backend::DrawStreaming;
use concinnity_engine::render::backend::error::RenderResult;
use objc2_metal::{MTLBuffer, MTLResourceOptions};

use crate::metal::context::{MtlContext, copy_buffer_prefix};

impl DrawStreaming for MtlContext {
    // Grow the shared vertex and index buffers by a headroom region for
    // streamed `VoxelWorld` chunks and seed the chunk allocators with it. Runs
    // once at init, before any command buffer references the buffers.
    fn setup_chunk_streaming(
        &mut self,
        chunk_vtx_bytes: usize,
        chunk_idx_bytes: usize,
    ) -> RenderResult<()> {
        let old_v_len = self.scene.vertex_buffer.length();
        let old_i_len = self.scene.index_buffer.length();

        let new_vbuf = self
            .hw
            .allocator
            .alloc_buffer(
                old_v_len + chunk_vtx_bytes,
                MTLResourceOptions::StorageModeShared,
            )
            .map_err(|e| e.context("setup_chunk_streaming: chunk vertex buffer"))?;
        let new_ibuf = self
            .hw
            .allocator
            .alloc_buffer(
                old_i_len + chunk_idx_bytes,
                MTLResourceOptions::StorageModeShared,
            )
            .map_err(|e| e.context("setup_chunk_streaming: chunk index buffer"))?;

        // Copy the build-time geometry into the start of the grown buffers so
        // every existing draw's offsets stay valid.
        copy_buffer_prefix(&self.scene.vertex_buffer, &new_vbuf, old_v_len);
        copy_buffer_prefix(&self.scene.index_buffer, &new_ibuf, old_i_len);
        self.scene.vertex_buffer = new_vbuf;
        self.scene.index_buffer = new_ibuf;

        // Seed the chunk allocators with the appended headroom. retire_frame 0:
        // nothing has been drawn, so the space is reusable immediately.
        self.state
            .placement
            .chunk_vtx
            .free(old_v_len as u64, chunk_vtx_bytes as u64, 0);
        self.state
            .placement
            .chunk_idx
            .free(old_i_len as u64, chunk_idx_bytes as u64, 0);
        Ok(())
    }
}
