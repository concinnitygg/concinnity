// src/directx/cull_readback.rs
//
// Per-object cull-status readback for the D3D12 backend. The `cn debug` WS
// server's `cull-status` command routes here (via
// `RenderBackend::read_cull_status`) to copy the GPU-driven cull's status
// buffer for the most recently submitted frame into a READBACK-heap buffer and
// hand back one `CullStatus` value per live cull record.
//
// This is the only observable record of what the cull decided: the submitted
// draw-call count is CPU-side and does not move when the GPU rejects an object,
// and an object the Hi-Z test correctly occluded leaves no trace in the
// presented pixels. Readback is synchronous (it idles the GPU), so it is a
// probe-only path, never a per-frame one. Mirrors src/vulkan/cull_readback.rs.

use concinnity_core::gfx::cull_status;
use windows::Win32::Graphics::Direct3D12::*;

use super::context::{DxContext, FRAMES};
use super::texture::{create_buffer, one_shot_submit};

impl DxContext {
    // Read the last submitted frame's cull-status buffer back to the host, one
    // u32 per live cull record. Distinct name from the
    // `RenderBackend::read_cull_status` trait method so the backend forwarder
    // is unambiguous.
    pub(crate) fn read_cull_status_buffer(&mut self) -> Result<Vec<u32>, String> {
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
        let slot = (self.current_frame + FRAMES - 1) % FRAMES;
        let src = self.cull.cull_status_buffers[slot].clone();
        let byte_size = (count * std::mem::size_of::<u32>()) as u64;

        // The GPU must be idle: the status buffer is then settled and no
        // in-flight cull dispatch is still writing the slot being copied.
        self.wait_idle();

        // READBACK-heap resources start in COPY_DEST and never need a barrier.
        let readback = create_buffer(
            &self.alloc,
            byte_size,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;

        // No barrier either side of the copy. Buffers decay to COMMON when the
        // submission that promoted them retires, so by this one-shot list the
        // status buffer is COMMON however the frame's cull left it, and
        // `CopyBufferRegion` promotes it to COPY_SOURCE implicitly. Declaring a
        // transition out of the cull's UNORDERED_ACCESS would name a state the
        // resource is no longer in.
        // SAFETY: the command list is in the recording state, and every resource these commands
        // name is live for the call.
        one_shot_submit(&self.device, &self.command_queue, |cmd| unsafe {
            cmd.CopyBufferRegion(&*readback, 0, &src, 0, byte_size);
        })?;

        let mut map_ptr = std::ptr::null_mut::<std::ffi::c_void>();
        // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local
        // that receives the mapping.
        unsafe { readback.Map(0, None, Some(&mut map_ptr)) }
            .map_err(|e| format!("cull-status: map readback: {e}"))?;
        // SAFETY: the mapping covers `byte_size` bytes (the size the buffer was created at), and
        // the copy completed (one_shot_submit waits its fence).
        let raw = unsafe { std::slice::from_raw_parts(map_ptr as *const u8, byte_size as usize) };
        let decoded = cull_status::decode(raw, count).map_err(|e| format!("cull-status: {e}"));
        // SAFETY: the resource is live and this code mapped it, and nothing keeps the mapping past
        // this call.
        unsafe { readback.Unmap(0, None) };
        decoded
    }
}
