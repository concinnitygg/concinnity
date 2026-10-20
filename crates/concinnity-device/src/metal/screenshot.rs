//! Frame capture for the Metal backend, two ways.
//!
//! A frame asked to capture (`FrameParams::capture`) blits its drawable into a
//! reused `StorageModeShared` buffer inside the presenting command buffer, and
//! that buffer's completion handler hands the bytes to the shared capture
//! writer, so the presenting thread neither waits nor encodes.
//!
//! The `screenshot` debug call instead reads back the last presented drawable
//! on demand and blocks: a one-shot blit into a staging texture, a wait, then
//! decode and encode on the calling thread. Metal has no persistent swapchain
//! image array, so `draw_frame` retains the last presented drawable's texture
//! in `last_present_texture`. The blit's command buffer commits after every
//! frame command buffer on the same queue, so FIFO order has the drawable fully
//! rendered before the copy reads it.
//!
//! Both read a drawable, which is only blit-readable with `framebufferOnly`
//! off; init does that only when capture is enabled.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::gfx::image_decode::PixelLayout;
use concinnity_core::math::SaturatingFrom;
use concinnity_core::render::backend::error::{RenderError, RenderResult};
use concinnity_core::render::output::hdr::HdrEncoding;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder as _, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandEncoder as _, MTLCommandQueue as _, MTLDevice as _, MTLOrigin, MTLPixelFormat,
    MTLRegion, MTLResourceOptions, MTLSize, MTLStorageMode, MTLTexture,
};

use super::context::MtlContext;
use super::descriptors::TextureDesc;
use super::error::{allocation_failed, completed_command_buffer};
use crate::frame_capture::{self, CaptureImage, FrameCaptures, Mapped, RetiredCapture};

type MtlBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;

/// The Metal backend's frame captures. A completion handler reports
/// retirement, so nothing is parked by token.
pub(in crate::metal) type MtlFrameCaptures = FrameCaptures<MtlBuffer, ()>;

impl MtlContext {
    // Capture the last presented frame to a PNG at `path`. Returns the path on
    // success. Distinct name from the `RenderBackend::screenshot` trait method
    // so the backend forwarder is unambiguous. Reached through the
    // `RenderBackend` vtable (bin-only `cn debug`).
    pub(in crate::metal) fn capture_screenshot(&self, path: &str) -> RenderResult<String> {
        // `None` both before the first present and in production (capture is a
        // `cn debug`-only feature; see `last_present_texture`). The retained
        // texture keeps the drawable's color surface alive for the read-back.
        let src = self.last_present_texture.clone().ok_or_else(|| {
            RenderError::Other(
                "screenshot: no frame has been presented yet (capture is cn debug only)"
                    .to_string(),
            )
        })?;
        let width = src.width();
        let height = src.height();
        if width == 0 || height == 0 {
            return Err(RenderError::Other(
                "screenshot: zero-sized drawable".to_string(),
            ));
        }

        // Host-readable staging texture matching the drawable's format. The
        // drawable's own storage mode is driver-chosen (often not host-visible),
        // so blit into a `StorageModeShared` texture we control and `getBytes`
        // from that. `ShaderRead` is the default usage and is enough for a blit
        // destination.
        let desc = TextureDesc {
            format: self.hw.swap_pixel_format,
            width,
            height,
            storage: MTLStorageMode::Shared,
            ..Default::default()
        }
        .build();
        let staging = self
            .hw
            .device
            .newTextureWithDescriptor(&desc)
            .ok_or_else(|| allocation_failed("screenshot staging texture"))?;

        // One-shot blit: drawable color -> staging. Committed after every
        // frame command buffer on the shared queue, so FIFO order has the
        // composite pass (which wrote the drawable) complete first; the
        // `waitUntilCompleted` then guarantees the copy is done before the read.
        let cmd_buf = self.hw.command_queue.commandBuffer().ok_or_else(|| {
            RenderError::Other("screenshot: failed to get command buffer".to_string())
        })?;
        let blit = cmd_buf.blitCommandEncoder().ok_or_else(|| {
            RenderError::Other("screenshot: failed to get blit encoder".to_string())
        })?;
        // SAFETY: `staging` was created with the same format and at least `width` x `height` texels
        // as `src`, and the origin/size cover exactly that region of slice 0, mip 0 of both.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &src,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize { width, height, depth: 1 },
                &staging,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
            );
        }
        blit.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        completed_command_buffer(&cmd_buf, "screenshot blit")?;

        // Read the staging texture back tightly (no row padding) and decode.
        let image = CaptureImage::tight(
            u32::saturating_from(width),
            u32::saturating_from(height),
            classify(self.hw.swap_pixel_format, self.hw.hdr_encoding()),
        );
        let bytes_per_row = image.row_pitch;
        let mut raw = vec![0u8; bytes_per_row * height];
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width,
                height,
                depth: 1,
            },
        };
        // SAFETY: `raw` is `bytes_per_row * height` bytes long (exactly the tight
        // footprint requested), the staging texture is `StorageModeShared` and
        // the blit completed (`waitUntilCompleted` above), so the copy is valid.
        unsafe {
            staging.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                std::ptr::NonNull::new(raw.as_mut_ptr().cast::<std::ffi::c_void>()).ok_or_else(
                    || RenderError::Other("screenshot: null readback pointer".to_string()),
                )?,
                bytes_per_row,
                region,
                0,
            );
        }

        frame_capture::save_png(path, &image, &raw)?;
        Ok(path.to_string())
    }

    // Keep a read-back buffer the size of `drawable` ready, so the frame that
    // captures allocates nothing.
    pub(in crate::metal) fn reserve_frame_capture(
        &mut self,
        drawable: &ProtocolObject<dyn MTLTexture>,
    ) {
        let image = self.capture_image(drawable);
        let device = &self.hw.device;
        self.frame_capture
            .reserve(image.byte_len(), |len| shared_buffer(device, len));
    }

    // Record this frame's capture, if one is armed: blit `drawable` into a
    // read-back buffer on `cmd_buf`, ahead of its present, and hand the bytes
    // to the writer from the buffer's completion handler.
    pub(in crate::metal) fn record_frame_capture(
        &mut self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        drawable: &ProtocolObject<dyn MTLTexture>,
    ) {
        let Some(path) = self.frame_capture.take_armed() else {
            return;
        };
        if !self.frame_capture.enabled() {
            tracing::warn!("frame capture {path} skipped: frame capture is not enabled");
            return;
        }
        let image = self.capture_image(drawable);
        let device = &self.hw.device;
        let (buffer, pending) = match self
            .frame_capture
            .begin(path.clone(), image, |len| shared_buffer(device, len))
        {
            Ok(lent) => lent,
            Err(e) => {
                tracing::warn!("frame capture {path} failed: {e}");
                return;
            }
        };
        let Some(blit) = cmd_buf.blitCommandEncoder() else {
            pending.abandon("no blit encoder");
            return;
        };
        // SAFETY: `buffer` holds `image.byte_len()` bytes, exactly `height` rows
        // of `row_pitch` texels of the drawable's format, and the region is slice
        // 0, mip 0 of `drawable`, whose size `image` was taken from.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                drawable,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize {
                    width: image.width as usize,
                    height: image.height as usize,
                    depth: 1,
                },
                &buffer,
                0,
                image.row_pitch,
                image.byte_len(),
            );
        }
        blit.endEncoding();

        let sender = self.frame_capture.sender();
        let pending = std::sync::Mutex::new(Some(pending));
        let handler = block2::RcBlock::new(
            move |cb: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                let Some(pending) = pending.lock().ok().and_then(|mut p| p.take()) else {
                    return;
                };
                // SAFETY: Metal hands the completion handler a live command
                // buffer, and the borrow does not escape the block.
                let cb = unsafe { cb.as_ref() };
                if cb.status() == MTLCommandBufferStatus::Error {
                    pending.abandon("its frame faulted on the GPU");
                } else {
                    // SAFETY: the handler runs once the command buffer holding
                    // the blit has completed.
                    sender.send(unsafe { RetiredCapture::new(pending) });
                }
            },
        );
        // SAFETY: addCompletedHandler copies the block (Block_copy), so the
        // RcBlock is free to drop when this scope ends.
        unsafe { cmd_buf.addCompletedHandler(block2::RcBlock::as_ptr(&handler)) };
    }

    // Write every capture recorded so far, once the GPU has finished them.
    pub(in crate::metal) fn finish_frame_captures(&mut self) {
        self.wait_idle();
        // SAFETY: the queue is idle, and nothing is parked by token on Metal.
        unsafe { self.frame_capture.finish() };
    }

    fn capture_image(&self, drawable: &ProtocolObject<dyn MTLTexture>) -> CaptureImage {
        CaptureImage::tight(
            u32::saturating_from(drawable.width()),
            u32::saturating_from(drawable.height()),
            classify(self.hw.swap_pixel_format, self.hw.hdr_encoding()),
        )
    }
}

// A host-visible buffer of `len` bytes and where its contents are mapped.
fn shared_buffer(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    len: usize,
) -> RenderResult<Mapped<MtlBuffer>> {
    let buffer = device
        .newBufferWithLength_options(len.max(1), MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| allocation_failed("frame capture read-back buffer"))?;
    let mapped = buffer.contents().as_ptr() as *const u8;
    Ok(Mapped { buffer, mapped })
}

// Classify the swapchain color format (+ resolved HDR encoding) into the
// backend-free `PixelLayout` the shared decoder understands. The MTKView only
// presents `BGRA8Unorm`/`_sRGB` for SDR or `RGBA16Float` for the HDR EDR path;
// `encoding` (None on SDR) only matters for the float swapchain.
fn classify(format: MTLPixelFormat, encoding: Option<HdrEncoding>) -> PixelLayout {
    match format {
        MTLPixelFormat::RGBA16Float => PixelLayout::Rgba16F {
            scrgb: !matches!(encoding, Some(HdrEncoding::Pq)),
        },
        MTLPixelFormat::BGRA8Unorm | MTLPixelFormat::BGRA8Unorm_sRGB => PixelLayout::Bgra8,
        _ => PixelLayout::Rgba8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_maps_swapchain_formats_to_pixel_layouts() {
        // SDR BGRA (both linear + sRGB) swizzles; RGBA passes through.
        assert_eq!(
            classify(MTLPixelFormat::BGRA8Unorm, None),
            PixelLayout::Bgra8
        );
        assert_eq!(
            classify(MTLPixelFormat::BGRA8Unorm_sRGB, None),
            PixelLayout::Bgra8
        );
        assert_eq!(
            classify(MTLPixelFormat::RGBA8Unorm, None),
            PixelLayout::Rgba8
        );
        // The float HDR swapchain applies the sRGB OETF on the scRGB path and
        // passes PQ code values through; unset encoding is treated as scRGB.
        assert_eq!(
            classify(
                MTLPixelFormat::RGBA16Float,
                Some(HdrEncoding::ExtendedLinear)
            ),
            PixelLayout::Rgba16F { scrgb: true }
        );
        assert_eq!(
            classify(MTLPixelFormat::RGBA16Float, Some(HdrEncoding::Pq)),
            PixelLayout::Rgba16F { scrgb: false }
        );
        assert_eq!(
            classify(MTLPixelFormat::RGBA16Float, None),
            PixelLayout::Rgba16F { scrgb: true }
        );
    }
}
