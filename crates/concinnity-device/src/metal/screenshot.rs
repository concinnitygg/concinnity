// src/metal/screenshot.rs
//
// Headless frame capture for the Metal backend. The `cn debug` WS server's
// `screenshot` command routes here (via `RenderBackend::screenshot`) to copy
// the most recently presented drawable's colour texture into a host-readable
// texture and encode it to a PNG on disk. This is the on-GPU verification path
// the renderer otherwise leaves to a human eyeballing the live window: a
// headless smoke can now assert on actual pixels. Mirrors
// src/directx/screenshot.rs / src/vulkan/screenshot.rs.
//
// Metal has no persistent swapchain image array to read back the way D3D12 /
// Vulkan do; `CAMetalDrawable`s are transient. So `draw_frame` retains the last
// presented drawable's texture in `last_present_texture` (only under
// `hot_reload`, the path that also switches the MTKView's `framebufferOnly`
// off so the drawable can be a blit source). Capture is synchronous: a one-shot
// blit copies that texture into a `StorageModeShared` staging texture, waits,
// then `getBytes` + decode + PNG-encode on the CPU. The blit's own command
// buffer commits after every frame command buffer on the same queue, so
// same-queue FIFO order guarantees the drawable is fully rendered before the
// copy reads it. The decode follows the swapchain format (4-byte SDR `BGRA8` or
// 8-byte HDR `RGBA16Float`), not a fixed texel size.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2_metal::{
    MTLBlitCommandEncoder as _, MTLCommandBuffer as _, MTLCommandEncoder as _,
    MTLCommandQueue as _, MTLDevice as _, MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize,
    MTLStorageMode, MTLTexture as _,
};

use crate::gfx::hdr_output::HdrEncoding;
use crate::gfx::image_decode::{self, PixelLayout};

use super::context::MtlContext;
use super::descriptors::TextureDesc;

impl MtlContext {
    // Capture the last presented frame to a PNG at `path`. Returns the path on
    // success. Distinct name from the `RenderBackend::screenshot` trait method
    // so the backend forwarder is unambiguous. Reached through the
    // `RenderBackend` vtable (bin-only `cn debug`).
    pub(in crate::metal) fn capture_screenshot(&mut self, path: &str) -> Result<String, String> {
        // `None` both before the first present and in production (capture is a
        // `cn debug`-only feature; see `last_present_texture`). The retained
        // texture keeps the drawable's colour surface alive for the read-back.
        let src = self
            .last_present_texture
            .clone()
            .ok_or("screenshot: no frame has been presented yet (capture is cn debug only)")?;
        let width = src.width();
        let height = src.height();
        if width == 0 || height == 0 {
            return Err("screenshot: zero-sized drawable".into());
        }

        // Host-readable staging texture matching the drawable's format. The
        // drawable's own storage mode is driver-chosen (often not host-visible),
        // so blit into a `StorageModeShared` texture we control and `getBytes`
        // from that. `ShaderRead` is the default usage and is enough for a blit
        // destination.
        let desc = TextureDesc {
            format: self.swap_pixel_format,
            width,
            height,
            storage: MTLStorageMode::Shared,
            ..Default::default()
        }
        .build();
        let staging = self
            .device
            .newTextureWithDescriptor(&desc)
            .ok_or("screenshot: failed to create staging texture")?;

        // One-shot blit: drawable colour -> staging. Committed after every
        // frame command buffer on the shared queue, so FIFO order has the
        // composite pass (which wrote the drawable) complete first; the
        // `waitUntilCompleted` then guarantees the copy is done before the read.
        let cmd_buf = self
            .command_queue
            .commandBuffer()
            .ok_or("screenshot: failed to get command buffer")?;
        let blit = cmd_buf
            .blitCommandEncoder()
            .ok_or("screenshot: failed to get blit encoder")?;
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

        // Read the staging texture back tightly (no row padding) and decode.
        let bytes_per_pixel = swapchain_bytes_per_pixel(self.swap_pixel_format) as usize;
        let bytes_per_row = width * bytes_per_pixel;
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
                std::ptr::NonNull::new(raw.as_mut_ptr() as *mut std::ffi::c_void)
                    .ok_or("screenshot: null readback pointer")?,
                bytes_per_row,
                region,
                0,
            );
        }

        let rgba = image_decode::decode_to_rgba8(
            &raw,
            classify(self.swap_pixel_format, self.hdr.encoding),
        );
        encode_png(path, width as u32, height as u32, &rgba)?;
        Ok(path.to_string())
    }
}

// Bytes per texel for the swapchain colour formats this backend can present.
// The MTKView only ever presents `BGRA8Unorm` for SDR or `RGBA16Float` for the
// HDR EDR path (see `metal/init/window.rs::swap_pixel_format`). Unknown formats
// default to 4, the common 32-bit-texel case.
fn swapchain_bytes_per_pixel(format: MTLPixelFormat) -> u32 {
    match format {
        MTLPixelFormat::RGBA16Float => 8,
        _ => 4,
    }
}

// Classify the swapchain colour format (+ resolved HDR encoding) into the
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

// Write RGBA8 pixel data to a PNG file.
fn encode_png(path: &str, width: u32, height: u32, rgba: &[u8]) -> Result<(), String> {
    let file =
        std::fs::File::create(path).map_err(|e| format!("screenshot: create {path}: {e}"))?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|e| format!("screenshot: png header: {e}"))?;
    writer
        .write_image_data(rgba)
        .map_err(|e| format!("screenshot: png data: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_per_pixel_matches_swapchain_formats() {
        // The SDR swapchain is 4 B/px; the float HDR swapchain is 8.
        assert_eq!(swapchain_bytes_per_pixel(MTLPixelFormat::BGRA8Unorm), 4);
        assert_eq!(swapchain_bytes_per_pixel(MTLPixelFormat::RGBA8Unorm), 4);
        assert_eq!(swapchain_bytes_per_pixel(MTLPixelFormat::RGBA16Float), 8);
    }

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
