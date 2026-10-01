//! The scene VkContext lends to the `SceneHost` defaults, and the writer that
//! stages its streamed geometry for the shared vertex / index buffers: each
//! write is copied by the next geometry submit (see `geometry_upload`), so none
//! waits on the GPU. Also the blocking `write_geometry_region` helper, which
//! copies into a buffer through its own one-shot submit, for the init and
//! rebuild paths that write fresh buffers.

use ash::vk;
use concinnity_core::render::backend::{GeometryEdit, SceneHost};
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::scene_state::{GeometryBuffer, GeometryWriter, SceneState};

use super::super::allocator::DeviceAllocator;
use super::super::context::*;
use super::super::geometry_upload::{GeometryTarget, GeometryUploads};
use super::super::texture;

// Stages each write for the next geometry copy submit, which lands it after
// every earlier GPU read of the buffer and before any later one.
struct StagingWriter<'a> {
    alloc: &'a DeviceAllocator,
    uploads: &'a mut GeometryUploads,
}

impl GeometryWriter for StagingWriter<'_> {
    fn write(&mut self, buffer: GeometryBuffer, offset: usize, bytes: &[u8]) -> RenderResult<()> {
        let target = match buffer {
            GeometryBuffer::Vertex => GeometryTarget::Vertex,
            GeometryBuffer::Index => GeometryTarget::Index,
        };
        self.uploads.stage(self.alloc, target, offset as u64, bytes)
    }

    fn reserve(&mut self, bytes: u64) {
        if let Err(e) = self.uploads.reserve(self.alloc, bytes) {
            tracing::warn!("mesh streaming: geometry staging reserve failed: {e}");
        }
    }
}

impl SceneHost for VkContext {
    fn scene(&self) -> Option<&SceneState> {
        Some(&self.state)
    }

    fn scene_mut(&mut self) -> Option<&mut SceneState> {
        debug_assert_main_thread("scene_mut");
        Some(&mut self.state)
    }

    fn edit_geometry(&mut self, edit: GeometryEdit<'_>) -> Option<RenderResult<()>> {
        debug_assert_main_thread("edit_geometry");
        let mut writer = StagingWriter {
            alloc: &self.hw.alloc,
            uploads: self.geometry_uploads.get_mut(),
        };
        Some(edit(&mut self.state, &mut writer))
    }
}

impl VkContext {
    // Copy `data` into a sub-region of a DEVICE_LOCAL geometry buffer.
    //
    // `dest` is the vertex or index buffer (both created with `TRANSFER_DST`).
    // The copy goes through a host-visible staging buffer and a one-shot
    // command buffer, mirroring `upload_geometry_buffer`'s init path. The
    // caller must `wait_idle` first so no in-flight command buffer still reads
    // `dest` while the transfer writes it.
    pub(in crate::vulkan) fn write_geometry_region(
        &self,
        dest: vk::Buffer,
        offset: u64,
        data: &[u8],
    ) -> RenderResult<()> {
        if data.is_empty() {
            return Ok(());
        }
        let size = data.len() as u64;
        let staging = self.hw.alloc.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        staging.write_bytes(0, data);
        texture::one_shot_submit(
            &self.hw.device,
            self.commands.command_pool,
            self.hw.graphics_queue,
            |cmd| {
                let copy = vk::BufferCopy::default().dst_offset(offset).size(size);
                // SAFETY: `cmd` is a command buffer in the recording state, and every handle and
                // slice these commands name is live for the call.
                unsafe {
                    self.hw.device.cmd_copy_buffer(
                        cmd,
                        staging.buffer(),
                        dest,
                        std::slice::from_ref(&copy),
                    )
                };
            },
        )?;
        // The one-shot idled the queue; drop the staging buffer and retire it
        // immediately so a per-region upload loop reuses one staging range.
        drop(staging);
        self.hw.alloc.reclaim_idle();
        Ok(())
    }
}
