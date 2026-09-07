// src/vulkan/cull_readback.rs
//
// Per-object cull-status readback for the Vulkan backend. The `cn debug` WS
// server's `cull-status` command routes here (via
// `RenderBackend::read_cull_status`) to copy the GPU-driven cull's status
// buffer for the most recently submitted frame into a host-visible buffer and
// hand back one `CullStatus` value per live cull record.
//
// This is the only observable record of what the cull decided: the submitted
// draw-call count is CPU-side and does not move when the GPU rejects an object,
// and an object the Hi-Z test correctly occluded leaves no trace in the
// presented pixels. Readback is synchronous (it idles the device), so it is a
// probe-only path, never a per-frame one. Mirrors src/directx/cull_readback.rs.

use ash::vk;

use super::context::VkContext;
use super::texture::one_shot_submit;
use concinnity_core::gfx::cull_status;

impl VkContext {
    // Read the last submitted frame's cull-status buffer back to the host, one
    // u32 per live cull record. Distinct name from the
    // `RenderBackend::read_cull_status` trait method so the backend forwarder
    // is unambiguous.
    pub(in crate::vulkan) fn read_cull_status_buffer(&mut self) -> Result<Vec<u32>, String> {
        if self.cull.cull_status_buffers.is_empty() {
            return Err("cull-status: this world does not run the GPU-driven cull".into());
        }
        if self.swapchain.last_present_index.is_none() {
            return Err("cull-status: no frame has been submitted yet".into());
        }
        let count = self.cull_count();
        if count == 0 {
            return Ok(Vec::new());
        }

        // The ring cursor advances past the frame it just recorded, so the
        // buffer holding the newest cull results is the slot behind it.
        let frames = self.frames_in_flight.max(1);
        let slot = (self.current_frame + frames - 1) % frames;
        let src = self.cull.cull_status_buffers[slot].buffer();
        let byte_size = (count * std::mem::size_of::<u32>()) as u64;

        // The GPU must be idle: the status buffer is then settled and no
        // in-flight cull dispatch is still writing the slot being copied.
        // SAFETY: a wait on this device's own queues; it takes no borrowed state.
        unsafe { self.device.device_wait_idle() }
            .map_err(|e| format!("cull-status: wait idle: {e}"))?;

        let readback = self.alloc.create_buffer(
            byte_size,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // No barrier either side of the copy: buffers have no layout to
        // transition, and the idle wait above already retired the cull
        // dispatch that wrote this slot. Emitting one here would also hand a
        // second owner to a resource the graph's barrier registry resolves.
        let device = self.device.clone();
        one_shot_submit(
            &device,
            self.commands.command_pool,
            self.graphics_queue,
            |cmd| {
                let region = vk::BufferCopy::default().size(byte_size);
                // SAFETY: `cmd` is a command buffer in the recording state, and every handle and
                // slice these commands name is live for the call.
                unsafe {
                    device.cmd_copy_buffer(cmd, src, readback.buffer(), &[region]);
                }
            },
        )?;

        // SAFETY: the buffer is HOST_COHERENT and at least `byte_size` bytes long, and the copy
        // above completed (one_shot_submit waits its fence).
        let raw = unsafe { std::slice::from_raw_parts(readback.mapped_ptr(), byte_size as usize) };
        cull_status::decode(raw, count).map_err(|e| format!("cull-status: {e}"))
    }
}
