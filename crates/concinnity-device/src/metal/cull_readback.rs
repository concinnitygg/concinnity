// src/metal/cull_readback.rs
//
// Per-object cull-status readback for the Metal backend. The `cn debug` WS
// server's `cull-status` command routes here (via
// `RenderBackend::read_cull_status`) to copy the GPU-driven cull's status
// buffer into a host-readable buffer and hand back one `CullStatus` value per
// live cull record.
//
// This is the only observable record of what the cull decided: the submitted
// draw-call count is CPU-side and does not move when the GPU rejects an object,
// and an object the Hi-Z test correctly occluded leaves no trace in the
// presented pixels. Readback is synchronous, so it is a probe-only path, never
// a per-frame one. Mirrors src/vulkan/cull_readback.rs.
//
// Metal's status buffer is a single `StorageModePrivate` allocation rather than
// a per-frame ring (the decision and encode kernels of one frame are the only
// readers), so this blits it into a `StorageModeShared` staging buffer on the
// shared queue. Same-queue FIFO order puts that blit behind the frame command
// buffer that wrote the statuses, and `waitUntilCompleted` puts the host read
// behind the blit.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2_metal::{
    MTLBlitCommandEncoder as _, MTLBuffer as _, MTLCommandBuffer as _, MTLCommandEncoder as _,
    MTLCommandQueue as _, MTLDevice as _, MTLResourceOptions,
};

use concinnity_core::gfx::cull_status;

use super::context::MtlContext;

impl MtlContext {
    // Read the cull-status buffer back to the host, one u32 per live cull
    // record. Distinct name from the `RenderBackend::read_cull_status` trait
    // method so the backend forwarder is unambiguous.
    pub(in crate::metal) fn read_cull_status_buffer(&mut self) -> Result<Vec<u32>, String> {
        // `None` both on a non-bindless world (no cull pipeline, so
        // `ensure_icb_capacity` allocates nothing) and before the first frame.
        let src = self
            .cull
            .status_buffer
            .clone()
            .ok_or("cull-status: this world does not run the GPU-driven cull")?;
        let count = self.cull_count();
        if count == 0 {
            return Ok(Vec::new());
        }
        let byte_size = count * std::mem::size_of::<u32>();
        if byte_size > src.length() {
            return Err("cull-status: status buffer is smaller than the live object count".into());
        }

        let staging = self
            .device
            .newBufferWithLength_options(byte_size, MTLResourceOptions::StorageModeShared)
            .ok_or("cull-status: failed to create staging buffer")?;

        let cmd_buf = self
            .command_queue
            .commandBuffer()
            .ok_or("cull-status: failed to get command buffer")?;
        let blit = cmd_buf
            .blitCommandEncoder()
            .ok_or("cull-status: failed to get blit encoder")?;
        // SAFETY: both buffers are at least `byte_size` bytes long (checked above for `src`,
        // requested for `staging`), so the copied range is in bounds on each.
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &src, 0, &staging, 0, byte_size,
            );
        }
        blit.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        // SAFETY: the staging buffer is `StorageModeShared` and `byte_size` bytes long, and the
        // blit completed (`waitUntilCompleted` above), so its contents are readable and settled.
        let raw = unsafe {
            std::slice::from_raw_parts(staging.contents().as_ptr().cast::<u8>(), byte_size)
        };
        cull_status::decode(raw, count).map_err(|e| format!("cull-status: {e}"))
    }
}
