// src/directx/screenshot.rs
//
// Headless frame capture for the D3D12 backend. The `cn debug` WS server's
// `screenshot` command routes here (via `RenderBackend::screenshot`) to copy
// the most recently presented swapchain back-buffer into a readback buffer and
// encode it to a PNG on disk. This is the on-GPU verification path the renderer
// otherwise leaves to a human eyeballing the live window: a headless smoke can
// now assert on actual pixels. Mirrors src/vulkan/screenshot.rs.
//
// Capture is synchronous: it idles the GPU (so the last-presented buffer is
// stable and no in-flight command list still references it), copies the
// last-presented back-buffer (still in `PRESENT`) into a `READBACK` buffer on a
// one-shot DIRECT command list, restores the buffer to `PRESENT`, then maps +
// de-pads + decodes + PNG-encodes on the CPU. The readback buffer is sized from
// `GetCopyableFootprints` (D3D12 aligns each row to
// `D3D12_TEXTURE_DATA_PITCH_ALIGNMENT` = 256), so the per-row de-pad below
// strips that padding back to a tight RGBA8 image. A swapchain rebuild clears
// `last_present_index`, so a capture in the brief window before the next
// present returns a clean error rather than reading an unrendered buffer.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::gfx::hdr_output::HdrEncoding;
use crate::gfx::image_decode::{self, PixelLayout};

use super::com;
use super::context::DxContext;
use super::texture::{create_buffer, one_shot_submit, transition_barrier};

impl DxContext {
    // Capture the last presented frame to a PNG at `path`. Returns the path on
    // success. Distinct name from the `RenderBackend::screenshot` trait method
    // so the backend forwarder is unambiguous; `#[allow(dead_code)]` because it
    // is reached only through the `RenderBackend` vtable (bin-only `cn debug`).
    #[allow(dead_code)]
    pub fn capture_screenshot(&mut self, path: &str) -> Result<String, String> {
        let Some(back_idx) = self.last_present_index else {
            return Err("screenshot: no frame has been presented yet".into());
        };
        let back_buffer = self
            .back_buffers
            .get(back_idx)
            .ok_or("screenshot: stale back-buffer index")?
            .clone();
        let width = self.output_width;
        let height = self.output_height;
        if width == 0 || height == 0 {
            return Err("screenshot: zero-sized swapchain".into());
        }

        // The GPU must be idle: the last-presented buffer is then stable and no
        // in-flight command list still references the resources we touch.
        self.wait_idle();

        // Describe the back-buffer so `GetCopyableFootprints` can hand back the
        // placed-footprint layout (row pitch, total padded size) the copy needs.
        let tex_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: width as u64,
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: self.swap_format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let mut layout = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut row_count: u32 = 0;
        let mut row_size: u64 = 0;
        let mut total_size: u64 = 0;
        // SAFETY: a query on a live COM object; the descriptor it reads and the out-parameters it
        // fills are live locals that outlive the call.
        unsafe {
            self.device.GetCopyableFootprints(
                &tex_desc,
                0,
                1,
                0,
                Some(&mut layout),
                Some(&mut row_count),
                Some(&mut row_size),
                Some(&mut total_size),
            );
        }

        // Host-readable buffer sized for the padded footprint. READBACK heap
        // resources start in COPY_DEST and never need a barrier.
        let readback = create_buffer(
            &self.alloc,
            total_size,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;

        // Copy the presented back-buffer into the readback buffer, bracketing
        // with PRESENT <-> COPY_SOURCE barriers so the buffer is left exactly as
        // the next present expects it. `one_shot_submit` fence-waits internally.
        // Leaking a back-buffer reference here would later block `ResizeBuffers`;
        // `readback` / `back_buffer` outlive the synchronous `one_shot_submit`.
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: com::borrowed(&readback),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: layout,
            },
        };
        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: com::borrowed(&back_buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        one_shot_submit(&self.device, &self.command_queue, |cmd| unsafe {
            let to_src = transition_barrier(
                &back_buffer,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            cmd.ResourceBarrier(&[to_src]);
            cmd.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
            let to_present = transition_barrier(
                &back_buffer,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_PRESENT,
            );
            cmd.ResourceBarrier(&[to_present]);
        })?;

        // Map + de-pad + decode, then always unmap. The readback rows are padded
        // to `layout.Footprint.RowPitch`; `row_size` is the tight byte width.
        let row_pitch = layout.Footprint.RowPitch as usize;
        let tight_row = row_size as usize;
        let mut map_ptr = std::ptr::null_mut::<std::ffi::c_void>();
        // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local
        // that receives the mapping.
        unsafe { readback.Map(0, None, Some(&mut map_ptr)) }
            .map_err(|e| format!("screenshot: map readback: {e}"))?;
        // The copy completed (one_shot_submit waits its fence). Read each row's
        // tight span out of the padded footprint into a contiguous source image.
        let mut packed = vec![0u8; tight_row * height as usize];
        for row in 0..height as usize {
            // SAFETY: the readback buffer holds `row_pitch * height` bytes (the padded footprint
            // the copy was sized from), so `row * row_pitch` addresses the start of a row inside
            // it.
            let src = unsafe { (map_ptr as *const u8).add(row * row_pitch) };
            // SAFETY: `tight_row` bytes are valid within each padded row.
            let src_slice = unsafe { std::slice::from_raw_parts(src, tight_row) };
            packed[row * tight_row..(row + 1) * tight_row].copy_from_slice(src_slice);
        }
        // SAFETY: the resource is live and this code mapped it, and nothing keeps the mapping past
        // this call.
        unsafe { readback.Unmap(0, None) };

        let rgba =
            image_decode::decode_to_rgba8(&packed, classify(self.swap_format, self.hdr_encoding));
        encode_png(path, width, height, &rgba)?;
        Ok(path.to_string())
    }
}

// Bytes per texel for the swapchain colour formats this backend can present.
// The DX swapchain only ever resolves to `B8G8R8A8_UNORM` for SDR or
// `R16G16B16A16_FLOAT` for the HDR (scRGB-linear / PQ-float) path; see
// `init/window.rs`. Unknown formats default to 4, the common 32-bit-texel case.
// The capture path sizes its readback from `GetCopyableFootprints` (which also
// folds in the 256-byte row alignment), so this helper only documents +
// asserts the format-to-texel-size mapping under test.
#[allow(dead_code)]
fn swapchain_bytes_per_pixel(format: DXGI_FORMAT) -> u32 {
    match format {
        DXGI_FORMAT_R16G16B16A16_FLOAT => 8,
        _ => 4,
    }
}

// Classify the swapchain colour format (+ resolved HDR encoding) into the
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
        assert_eq!(swapchain_bytes_per_pixel(DXGI_FORMAT_B8G8R8A8_UNORM), 4);
        assert_eq!(swapchain_bytes_per_pixel(DXGI_FORMAT_R8G8B8A8_UNORM), 4);
        assert_eq!(swapchain_bytes_per_pixel(DXGI_FORMAT_R16G16B16A16_FLOAT), 8);
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
