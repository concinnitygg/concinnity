//! Pulling a rendered result back to the CPU: the presented frame as a PNG,
//! and the GPU-driven cull's per-object decisions.

use crate::render::backend::error::{RenderError, RenderResult};
use alloc::string::String;
use alloc::vec::Vec;

/// Saving presented frames to disk.
///
/// Defaulted: a backend without capture reports
/// [`RenderError::Unsupported`] for a screenshot and has no capture to wait
/// for.
pub trait FrameCapture {
    /// Capture the last presented frame to a PNG at `path` and return the saved
    /// path. Driven by the `screenshot` debug tool call for headless
    /// on-GPU render verification.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] on a backend without capture, or an
    /// error if no frame was presented or the PNG cannot be written.
    fn screenshot(&mut self, path: &str) -> RenderResult<String> {
        let _ = path;
        Err(RenderError::Unsupported { op: "screenshot" })
    }

    /// Block until every frame capture asked of
    /// [`RenderBackend::draw_frame`](super::RenderBackend::draw_frame) so far
    /// is written to disk or has failed. Waits for the GPU first.
    fn finish_captures(&mut self) {}
}

/// Reading the GPU-driven cull's decisions back.
pub trait CullReadback {
    /// Read the GPU-driven cull's per-object status buffer back to the host,
    /// one [`crate::gfx::cull_status::CullStatus`] value per live cull record,
    /// for the most recently submitted frame.
    ///
    /// The submitted draw-call count is a CPU-side number that does not move
    /// when the GPU rejects an object, and an object the Hi-Z test correctly
    /// occluded leaves no trace in the presented pixels, so this buffer is the
    /// only observable record of what the cull decided. Driven by the
    /// `cull-status` debug tool call; synchronous (it idles the device).
    ///
    /// Default [`RenderError::Unsupported`]: a backend with no GPU-driven cull.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] on a backend without a GPU-driven
    /// cull, or an error if the readback fails.
    fn read_cull_status(&mut self) -> RenderResult<Vec<u32>> {
        Err(RenderError::Unsupported {
            op: "read_cull_status",
        })
    }
}
