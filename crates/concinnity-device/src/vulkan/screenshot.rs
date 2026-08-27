// src/vulkan/screenshot.rs
//
// Headless frame capture for the Vulkan backend. The `cn debug` WS server's
// `screenshot` command routes here (via `RenderBackend::screenshot`) to copy
// the most recently presented swapchain image into a host-visible buffer and
// encode it to a PNG on disk. This is the on-GPU verification path the renderer
// otherwise leaves to a human eyeballing the live window: a headless probe can
// now assert on actual pixels.
//
// The swapchain images are created with `TRANSFER_SRC` usage (see
// `swapchain.rs`) so the presented image can be copied. Capture is synchronous:
// it idles the device, copies the last-presented image (still in
// `PRESENT_SRC_KHR`) into the buffer, restores the image to `PRESENT_SRC_KHR`,
// then maps + decodes + PNG-encodes on the CPU. The read-back buffer and the
// per-pixel decode both follow the swapchain format (4-byte SDR `BGRA8` or
// 8-byte HDR `RGBA16F`), not a fixed texel size. A swapchain rebuild clears
// `swapchain.last_present_index`, so a capture in the brief window before the next present
// returns a clean error rather than reading an unrendered image.

use ash::vk;

use super::context::VkContext;
use super::texture::one_shot_submit;
use crate::gfx::hdr_output::{HdrEncoding, HdrOutputMode};
use crate::gfx::image_decode::{self, PixelLayout};

impl VkContext {
    // Capture the last presented frame to a PNG at `path`. Returns the path on
    // success. Distinct name from the `RenderBackend::screenshot` trait method
    // so the backend forwarder is unambiguous. Reached through the
    // `RenderBackend` vtable (bin-only `cn debug`).
    pub(in crate::vulkan) fn capture_screenshot(&mut self, path: &str) -> Result<String, String> {
        let Some(image_index) = self.swapchain.last_present_index else {
            return Err("screenshot: no frame has been presented yet".into());
        };
        let src_image = *self
            .swapchain
            .images
            .get(image_index as usize)
            .ok_or("screenshot: stale swapchain image index")?;
        let width = self.swapchain.extent.width;
        let height = self.swapchain.extent.height;
        if width == 0 || height == 0 {
            return Err("screenshot: zero-sized swapchain".into());
        }

        // The GPU must be idle: the last-presented image is then stable and no
        // in-flight command buffer still references the resources we touch.
        // SAFETY: a wait on this device's own queues; it takes no borrowed state.
        unsafe { self.device.device_wait_idle() }
            .map_err(|e| format!("screenshot: wait idle: {e}"))?;

        // Host-visible readback buffer, tightly packed at the swapchain
        // format's texel size. The SDR swapchain is `BGRA8_UNORM` (4 B/px), but
        // the HDR swapchain is `R16G16B16A16_SFLOAT` (8 B/px); sizing this for a
        // fixed 4 B/px overflows the `vkCmdCopyImageToBuffer` on the HDR path and
        // loses the device, so derive it from the actual format.
        let bytes_per_pixel = swapchain_bytes_per_pixel(self.swapchain.format) as u64;
        let byte_size = (width as u64) * (height as u64) * bytes_per_pixel;
        let readback = self.alloc.create_buffer(
            byte_size,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // Copy the presented image into the buffer, bracketing with
        // PRESENT_SRC <-> TRANSFER_SRC barriers so the image is left exactly as
        // present expects it for the next acquire.
        let device = self.device.clone();
        let copied = one_shot_submit(
            &device,
            self.commands.command_pool,
            self.graphics_queue,
            |cmd| {
                let to_src = image_barrier(
                    src_image,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_READ,
                );
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    });
                let to_present = image_barrier(
                    src_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                    vk::AccessFlags::TRANSFER_READ,
                    vk::AccessFlags::empty(),
                );
                // SAFETY: `cmd` is a command buffer in the recording state, and every handle and
                // slice these commands name is live for the call.
                unsafe {
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_src],
                    );
                    device.cmd_copy_image_to_buffer(
                        cmd,
                        src_image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        readback.buffer(),
                        std::slice::from_ref(&region),
                    );
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_present],
                    );
                }
            },
        );

        // Map + swizzle + encode, then always free the buffer.
        let result = copied.and_then(|()| {
            // SAFETY: the buffer is HOST_COHERENT and `byte_size` bytes long; the
            // copy above completed (one_shot_submit waits its fence).
            let raw =
                unsafe { std::slice::from_raw_parts(readback.mapped_ptr(), byte_size as usize) };
            // The HDR float swapchain needs the encoding to decode for display:
            // scRGB-linear gets the sRGB OETF, PQ-encoded code values pass
            // through (not display-correct, but a valid PNG rather than a crash).
            let encoding = match self.hdr_mode {
                HdrOutputMode::Hdr { encoding, .. } => Some(encoding),
                HdrOutputMode::Sdr => None,
            };
            let rgba =
                image_decode::decode_to_rgba8(raw, classify(self.swapchain.format, encoding));
            encode_png(path, width, height, &rgba)
        });
        result.map(|()| path.to_string())
    }
}

// A whole-image colour barrier on a swapchain image, used to flip between
// PRESENT_SRC and TRANSFER_SRC for the readback copy.
fn image_barrier(
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    src: vk::AccessFlags,
    dst: vk::AccessFlags,
) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier::default()
        .src_access_mask(src)
        .dst_access_mask(dst)
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
}

// Bytes per texel for the swapchain colour formats this backend can present.
// The swapchain only ever resolves to one of these (see `create_swapchain_inner`
// in swapchain.rs): `BGRA8_UNORM` for SDR, `R16G16B16A16_SFLOAT` for the scRGB /
// PQ-float HDR path, or `A2B10G10R10_UNORM_PACK32` for the packed PQ fallback.
// Unknown formats default to 4, the common 32-bit-texel case.
fn swapchain_bytes_per_pixel(format: vk::Format) -> u32 {
    match format {
        vk::Format::R16G16B16A16_SFLOAT => 8,
        _ => 4,
    }
}

// Classify the swapchain colour format (+ resolved HDR encoding) into the
// backend-free `PixelLayout` the shared decoder understands. Almost always BGRA8
// on Windows; the float HDR swapchain and the packed 2-10-10-10 PQ fallback are
// handled too. `encoding` (None on SDR) only matters for the float swapchain.
fn classify(format: vk::Format, encoding: Option<HdrEncoding>) -> PixelLayout {
    match format {
        vk::Format::R16G16B16A16_SFLOAT => PixelLayout::Rgba16F {
            scrgb: !matches!(encoding, Some(HdrEncoding::Pq)),
        },
        vk::Format::A2B10G10R10_UNORM_PACK32 => PixelLayout::A2b10g10r10,
        vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB | vk::Format::B8G8R8A8_SNORM => {
            PixelLayout::Bgra8
        }
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
        // SDR + the packed PQ fallback are 4 B/px; the float HDR swapchain is 8.
        assert_eq!(swapchain_bytes_per_pixel(vk::Format::B8G8R8A8_UNORM), 4);
        assert_eq!(swapchain_bytes_per_pixel(vk::Format::R8G8B8A8_UNORM), 4);
        assert_eq!(
            swapchain_bytes_per_pixel(vk::Format::A2B10G10R10_UNORM_PACK32),
            4
        );
        assert_eq!(
            swapchain_bytes_per_pixel(vk::Format::R16G16B16A16_SFLOAT),
            8
        );
    }

    #[test]
    fn classify_maps_swapchain_formats_to_pixel_layouts() {
        // The three BGRA8 variants (unorm / sRGB / snorm) all swizzle; RGBA8
        // passes through; the packed 2-10-10-10 PQ fallback has its own layout.
        assert_eq!(
            classify(vk::Format::B8G8R8A8_UNORM, None),
            PixelLayout::Bgra8
        );
        assert_eq!(
            classify(vk::Format::B8G8R8A8_SRGB, None),
            PixelLayout::Bgra8
        );
        assert_eq!(
            classify(vk::Format::B8G8R8A8_SNORM, None),
            PixelLayout::Bgra8
        );
        assert_eq!(
            classify(vk::Format::R8G8B8A8_UNORM, None),
            PixelLayout::Rgba8
        );
        assert_eq!(
            classify(vk::Format::A2B10G10R10_UNORM_PACK32, None),
            PixelLayout::A2b10g10r10
        );
        // The float HDR swapchain applies the sRGB OETF on the scRGB path and
        // passes PQ code values through; unset encoding is treated as scRGB.
        assert_eq!(
            classify(
                vk::Format::R16G16B16A16_SFLOAT,
                Some(HdrEncoding::ExtendedLinear)
            ),
            PixelLayout::Rgba16F { scrgb: true }
        );
        assert_eq!(
            classify(vk::Format::R16G16B16A16_SFLOAT, Some(HdrEncoding::Pq)),
            PixelLayout::Rgba16F { scrgb: false }
        );
        assert_eq!(
            classify(vk::Format::R16G16B16A16_SFLOAT, None),
            PixelLayout::Rgba16F { scrgb: true }
        );
    }
}
