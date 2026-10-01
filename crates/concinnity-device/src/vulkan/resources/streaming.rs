//! VoxelWorld chunk streaming for VkContext: appends a headroom region to the
//! shared vertex/index buffers and seeds the chunk allocators with it. Placing,
//! moving and removing chunks is scene bookkeeping, done in `SceneState`.

use ash::vk;
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

        let shared = super::shared_geometry_usage(self.hw.rt_capable);
        let new_vbuf = self.hw.alloc.create_buffer(
            new_v,
            vk::BufferUsageFlags::VERTEX_BUFFER | shared,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let new_ibuf = self.hw.alloc.create_buffer(
            new_i,
            vk::BufferUsageFlags::INDEX_BUFFER | shared,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Copy the build-time geometry into the start of the grown buffers so
        // every existing draw's offsets stay valid.
        texture::one_shot_submit(
            &self.hw.device,
            self.commands.command_pool,
            self.hw.graphics_queue,
            |cmd| {
                let vcopy = vk::BufferCopy::default().size(old_v);
                let icopy = vk::BufferCopy::default().size(old_i);
                // SAFETY: `cmd` is a command buffer in the recording state, and every handle and
                // slice these commands name is live for the call.
                unsafe {
                    self.hw.device.cmd_copy_buffer(
                        cmd,
                        self.geometry.vertex_buffer.buffer(),
                        new_vbuf.buffer(),
                        std::slice::from_ref(&vcopy),
                    );
                    self.hw.device.cmd_copy_buffer(
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
        self.rewire_shared_geometry_readers();

        self.geometry_uploads
            .get_mut()
            .reserve(&self.hw.alloc, (chunk_vtx_bytes + chunk_idx_bytes) as u64)?;

        self.state
            .placement
            .chunk_vtx
            .free(old_v, chunk_vtx_bytes as u64, 0);
        self.state
            .placement
            .chunk_idx
            .free(old_i, chunk_idx_bytes as u64, 0);

        Ok(())
    }
}
