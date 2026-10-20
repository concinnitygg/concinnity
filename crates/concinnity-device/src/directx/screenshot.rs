//! Frame capture for the D3D12 backend, two ways.
//!
//! A frame asked to capture (`FrameParams::capture`) copies its back buffer
//! into a reused `READBACK` buffer at the tail of its own end command list,
//! between the composite's `PRESENT` transition and the present. The capture
//! is parked under the fence value the frame signals and handed to the shared
//! capture writer once that value completes, so the presenting thread neither
//! waits nor encodes.
//!
//! The `screenshot` debug call instead reads back the last presented back
//! buffer on demand and blocks: it idles the GPU, copies on a one-shot list,
//! then decodes and encodes on the calling thread. A swapchain rebuild clears
//! `swapchain.last_present_index`, so a capture in the brief window before the
//! next present returns a clean error rather than reading an unrendered buffer.
//!
//! Both size the read-back from the copyable footprint: D3D12 aligns each row
//! to `D3D12_TEXTURE_DATA_PITCH_ALIGNMENT` (256 bytes), which the decode strips.

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::image_decode::PixelLayout;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::output::hdr::HdrEncoding;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::com;
use super::context::DxContext;
use super::texture::{one_shot_submit, transition_barrier};
use crate::directx::error::map_hresult;
use crate::frame_capture::{self, CaptureImage, FrameCaptures, Mapped, PendingCapture};

/// The D3D12 backend's frame captures, parked under the fence value their
/// frame signals.
pub(crate) type DxFrameCaptures = FrameCaptures<PooledBuffer, u64>;

impl DxContext {
    // Capture the last presented frame to a PNG at `path`. Returns the path on
    // success. Distinct name from the `RenderBackend::screenshot` trait method
    // so the backend forwarder is unambiguous. Reached through the
    // `RenderBackend` vtable (bin-only `cn debug`).
    pub(crate) fn capture_screenshot(&self, path: &str) -> RenderResult<String> {
        let Some(back_idx) = self.swapchain.last_present_index else {
            return Err(RenderError::Other(
                "screenshot: no frame has been presented yet".into(),
            ));
        };
        let back_buffer = self
            .swapchain
            .back_buffers
            .get(back_idx)
            .ok_or_else(|| RenderError::Other("screenshot: stale back-buffer index".into()))?
            .clone();
        let width = self.targets.extent.output_width;
        let height = self.targets.extent.output_height;
        if width == 0 || height == 0 {
            return Err(RenderError::Other(
                "screenshot: zero-sized swapchain".into(),
            ));
        }

        // The GPU must be idle: the last-presented buffer is then stable and no
        // in-flight command list still references the resources we touch.
        self.wait_idle();

        // Host-readable buffer sized for the padded footprint, mapped for its
        // life.
        let (layout, image) = self.capture_footprint();
        let Mapped {
            buffer: readback,
            mapped,
        } = readback_buffer(&self.hw.alloc, image.byte_len())?;

        // Copy the presented back-buffer into the readback buffer, leaving it in
        // PRESENT for the next present. `one_shot_submit` fence-waits internally.
        // Leaking a back-buffer reference here would later block `ResizeBuffers`;
        // `readback` / `back_buffer` outlive the synchronous `one_shot_submit`.
        one_shot_submit(&self.hw.device, &self.hw.command_queue, |cmd| {
            encode_copy(cmd, &back_buffer, &readback, layout);
        })?;

        // SAFETY: the mapping spans the `image.byte_len()` bytes the buffer was sized from, and
        // the copy into it completed (one_shot_submit waits its fence).
        let bytes = unsafe { std::slice::from_raw_parts(mapped, image.byte_len()) };
        frame_capture::save_png(path, &image, bytes)?;
        Ok(path.to_string())
    }

    // Keep a read-back buffer for the swapchain's footprint ready, so the frame
    // that captures allocates nothing.
    pub(in crate::directx) fn reserve_frame_capture(&mut self) {
        let (_, image) = self.capture_footprint();
        let alloc = &self.hw.alloc;
        self.frame_capture
            .reserve(image.byte_len(), |len| readback_buffer(alloc, len));
    }

    // Record this frame's capture, if one is armed: copy `back_buffer` (in
    // `PRESENT` after the composite) into a read-back buffer on `cmd`, leaving
    // it in `PRESENT` for the present.
    pub(in crate::directx) fn record_frame_capture(
        &mut self,
        cmd: &ID3D12GraphicsCommandList,
        back_buffer: &ID3D12Resource,
    ) -> Option<PendingCapture> {
        let path = self.frame_capture.take_armed()?;
        let (footprint, image) = self.capture_footprint();
        let alloc = &self.hw.alloc;
        let (readback, pending) = match self
            .frame_capture
            .begin(path.clone(), image, |len| readback_buffer(alloc, len))
        {
            Ok(lent) => lent,
            Err(e) => {
                tracing::warn!("frame capture {path} failed: {e}");
                return None;
            }
        };
        encode_copy(cmd, back_buffer, &readback, footprint);
        Some(pending)
    }

    // Hand the writer every capture whose frame's fence value has completed.
    pub(in crate::directx) fn retire_frame_captures(&mut self) {
        // SAFETY: a property query on a live fence; it only reads.
        let completed = unsafe { self.frame_sync.fence.GetCompletedValue() };
        // SAFETY: a capture is parked under the fence value its frame signals
        // after the copy, so a completed value means the copy is done.
        unsafe { self.frame_capture.retire(|&value| value <= completed) };
    }

    // Write every capture recorded so far, once the GPU has finished them.
    pub(in crate::directx) fn finish_frame_captures(&mut self) {
        self.wait_idle();
        // SAFETY: the queue is idle.
        unsafe { self.frame_capture.finish() };
    }

    // The swapchain back buffer's copy footprint, and the image it reads back
    // as.
    fn capture_footprint(&self) -> (D3D12_PLACED_SUBRESOURCE_FOOTPRINT, CaptureImage) {
        let width = self.targets.extent.output_width.max(1);
        let height = self.targets.extent.output_height.max(1);
        let layout = classify(self.swapchain.format, self.hw.hdr_encoding());
        let tight = CaptureImage::tight(width, height, layout);
        let image = CaptureImage {
            row_pitch: padded_row_pitch(tight.row_bytes()),
            ..tight
        };
        let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Format: self.swapchain.format,
                Width: width,
                Height: height,
                Depth: 1,
                RowPitch: u32::saturating_from(image.row_pitch),
            },
        };
        (footprint, image)
    }
}

// Copy `back_buffer`, in `PRESENT`, into `readback` laid out as `footprint`,
// and return it to `PRESENT`.
fn encode_copy(
    cmd: &ID3D12GraphicsCommandList,
    back_buffer: &ID3D12Resource,
    readback: &ID3D12Resource,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) {
    let dst = D3D12_TEXTURE_COPY_LOCATION {
        pResource: com::borrowed(readback),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: footprint,
        },
    };
    let src = D3D12_TEXTURE_COPY_LOCATION {
        pResource: com::borrowed(back_buffer),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    };
    let to_copy = transition_barrier(
        back_buffer,
        D3D12_RESOURCE_STATE_PRESENT,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    // SAFETY: the command list is in the recording state, and the caller keeps the back buffer
    // alive until the list has executed.
    unsafe { cmd.ResourceBarrier(&[to_copy]) };
    // SAFETY: the back buffer is in COPY_SOURCE from the barrier above, and both locations borrow
    // resources the caller keeps alive until the list has executed.
    unsafe { cmd.CopyTextureRegion(&raw const dst, 0, 0, 0, &raw const src, None) };
    let to_present = transition_barrier(
        back_buffer,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_PRESENT,
    );
    // SAFETY: the list is recording, and the back buffer is still live.
    unsafe { cmd.ResourceBarrier(&[to_present]) };
}

// A row of `row_bytes` padded to the pitch alignment a buffer copy needs.
fn padded_row_pitch(row_bytes: usize) -> usize {
    let align = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT as usize;
    row_bytes.div_ceil(align) * align
}

// A `len`-byte READBACK buffer, mapped for its whole life. READBACK heap
// resources start in COPY_DEST and never need a barrier.
fn readback_buffer(alloc: &DeviceAllocator, len: usize) -> RenderResult<Mapped<PooledBuffer>> {
    let buffer = alloc.alloc_buffer(
        len as u64,
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;
    let mut mapped = std::ptr::null_mut::<std::ffi::c_void>();
    // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local
    // that receives the mapping. A READBACK buffer may stay mapped while the GPU writes it.
    unsafe { buffer.Map(0, None, Some(&raw mut mapped)) }
        .map_err(|e| map_hresult(e.code(), "frame capture: map readback"))?;
    Ok(Mapped {
        buffer,
        mapped: mapped as *const u8,
    })
}

// Classify the swapchain color format (+ resolved HDR encoding) into the
// backend-free `PixelLayout` the shared decoder understands. The DX SDR
// swapchain is BGRA8 on Windows and the HDR EDR path is `R16G16B16A16_FLOAT`;
// `encoding` (None on SDR) only matters for the float swapchain.
fn classify(format: DXGI_FORMAT, encoding: Option<HdrEncoding>) -> PixelLayout {
    match format {
        DXGI_FORMAT_R16G16B16A16_FLOAT => PixelLayout::Rgba16F {
            scrgb: !matches!(encoding, Some(HdrEncoding::Pq)),
        },
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => PixelLayout::Bgra8,
        _ => PixelLayout::Rgba8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_rows_pad_to_the_pitch_alignment() {
        assert_eq!(padded_row_pitch(256), 256);
        assert_eq!(padded_row_pitch(257), 512);
        // 1920 BGRA8 texels.
        assert_eq!(padded_row_pitch(7680), 7680);
        // 1366 BGRA8 texels.
        assert_eq!(padded_row_pitch(5464), 5632);
    }

    #[test]
    fn classify_maps_swapchain_formats_to_pixel_layouts() {
        // SDR BGRA (both linear + sRGB) swizzles; RGBA passes through.
        assert_eq!(
            classify(DXGI_FORMAT_B8G8R8A8_UNORM, None),
            PixelLayout::Bgra8
        );
        assert_eq!(
            classify(DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, None),
            PixelLayout::Bgra8
        );
        assert_eq!(
            classify(DXGI_FORMAT_R8G8B8A8_UNORM, None),
            PixelLayout::Rgba8
        );
        // The float HDR swapchain applies the sRGB OETF on the scRGB path and
        // passes PQ code values through; unset encoding is treated as scRGB.
        assert_eq!(
            classify(
                DXGI_FORMAT_R16G16B16A16_FLOAT,
                Some(HdrEncoding::ExtendedLinear)
            ),
            PixelLayout::Rgba16F { scrgb: true }
        );
        assert_eq!(
            classify(DXGI_FORMAT_R16G16B16A16_FLOAT, Some(HdrEncoding::Pq)),
            PixelLayout::Rgba16F { scrgb: false }
        );
        assert_eq!(
            classify(DXGI_FORMAT_R16G16B16A16_FLOAT, None),
            PixelLayout::Rgba16F { scrgb: true }
        );
    }
}
