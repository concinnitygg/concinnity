//! Frame capture for the Vulkan backend, two ways.
//!
//! A frame asked to capture (`FrameParams::capture`) copies its swapchain image
//! into a reused host-visible buffer at the tail of its own end command buffer,
//! after the composite render pass left the image in `PRESENT_SRC_KHR`. The
//! capture is parked under its frame slot and handed to the shared capture
//! writer once that slot's fence has been waited, so the presenting thread
//! neither waits nor encodes.
//!
//! The `screenshot` debug call instead reads back the last presented image on
//! demand and blocks: it idles the device, copies on a one-shot buffer, then
//! decodes and encodes on the calling thread. A swapchain rebuild clears
//! `swapchain.last_present_index`, so a capture in the brief window before the
//! next present returns a clean error rather than reading an unrendered image.
//!
//! The swapchain images are created with `TRANSFER_SRC` usage (see
//! `swapchain.rs`) so the presented image can be copied. Both read-backs follow
//! the swapchain format's texel size (4-byte SDR `BGRA8` or 8-byte HDR
//! `RGBA16F`); a fixed 4 B/px would overflow the copy on the HDR path.

use ash::vk;
use concinnity_core::gfx::image_decode::PixelLayout;
use concinnity_core::render::backend::error::{RenderError, RenderResult};
use concinnity_core::render::output::hdr::{HdrEncoding, HdrOutputMode};

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::context::VkContext;
use super::owned::VkDevice;
use super::texture::one_shot_submit;
use crate::frame_capture::{self, CaptureImage, FrameCaptures, Mapped, PendingCapture};

/// The Vulkan backend's frame captures, parked under the frame slot whose
/// fence retires them.
pub(in crate::vulkan) type VkFrameCaptures = FrameCaptures<PooledBuffer, usize>;

impl VkContext {
    // Capture the last presented frame to a PNG at `path`. Returns the path on
    // success. Distinct name from the `RenderBackend::screenshot` trait method
    // so the backend forwarder is unambiguous. Reached through the
    // `RenderBackend` vtable (bin-only `cn debug`).
    pub(in crate::vulkan) fn capture_screenshot(&mut self, path: &str) -> RenderResult<String> {
        let Some(image_index) = self.swapchain.last_present_index else {
            return Err(RenderError::Other(
                "screenshot: no frame has been presented yet".into(),
            ));
        };
        let src_image = *self
            .swapchain
            .images
            .get(image_index as usize)
            .ok_or_else(|| RenderError::Other("screenshot: stale swapchain image index".into()))?;
        let width = self.swapchain.extent.width;
        let height = self.swapchain.extent.height;
        if width == 0 || height == 0 {
            return Err(RenderError::Other(
                "screenshot: zero-sized swapchain".into(),
            ));
        }

        // The GPU must be idle: the last-presented image is then stable and no
        // in-flight command buffer still references the resources we touch.
        // SAFETY: a wait on this device's own queues; it takes no borrowed state.
        unsafe { self.hw.device.device_wait_idle() }
            .map_err(|e| super::error::map_vk_result(e, "screenshot: wait idle"))?;

        let image = self.capture_image();
        let Mapped {
            buffer: readback,
            mapped,
        } = readback_buffer(&self.hw.alloc, image.byte_len())?;

        let device = self.hw.device.clone();
        one_shot_submit(
            &device,
            self.commands.command_pool,
            self.hw.graphics_queue,
            |cmd| encode_copy(&device, cmd, src_image, &readback, &image),
        )?;
        // SAFETY: the buffer is `image.byte_len()` mapped bytes, and the copy into it completed
        // (one_shot_submit waits its fence).
        let bytes = unsafe { std::slice::from_raw_parts(mapped, image.byte_len()) };
        frame_capture::save_png(path, &image, bytes)?;
        Ok(path.to_string())
    }

    // Keep a read-back buffer the size of the swapchain ready, so the frame
    // that captures allocates nothing.
    pub(in crate::vulkan) fn reserve_frame_capture(&mut self) {
        let image = self.capture_image();
        let alloc = &self.hw.alloc;
        self.frame_capture
            .reserve(image.byte_len(), |len| readback_buffer(alloc, len));
    }

    // Record this frame's capture, if one is armed: copy swapchain image
    // `image_index` (in `PRESENT_SRC_KHR` after the composite) into a read-back
    // buffer on `cmd`, leaving it in `PRESENT_SRC_KHR` for the present.
    pub(in crate::vulkan) fn record_frame_capture(
        &mut self,
        cmd: vk::CommandBuffer,
        image_index: u32,
    ) -> Option<PendingCapture> {
        let path = self.frame_capture.take_armed()?;
        let src_image = *self.swapchain.images.get(image_index as usize)?;
        let image = self.capture_image();
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
        encode_copy(&self.hw.device, cmd, src_image, &readback, &image);
        Some(pending)
    }

    // Hand the writer the captures frame slot `frame` carried, once its fence
    // has been waited.
    pub(in crate::vulkan) fn retire_frame_captures(&mut self, frame: usize) {
        // SAFETY: called after the wait on `frame`'s fence, which the submission
        // carrying the copy signals.
        unsafe { self.frame_capture.retire(|&slot| slot == frame) };
    }

    // Write every capture recorded so far, once the GPU has finished them.
    pub(in crate::vulkan) fn finish_frame_captures(&mut self) {
        self.wait_idle();
        // SAFETY: the device is idle.
        unsafe { self.frame_capture.finish() };
    }

    // The swapchain image as it reads back: tightly packed at its format's
    // texel size.
    fn capture_image(&self) -> CaptureImage {
        // The HDR float swapchain needs the encoding to decode for display:
        // scRGB-linear gets the sRGB OETF, PQ-encoded code values pass through
        // (not display-correct, but a valid PNG rather than a crash).
        let encoding = match self.hw.hdr_mode {
            HdrOutputMode::Hdr { encoding, .. } => Some(encoding),
            HdrOutputMode::Sdr => None,
        };
        CaptureImage::tight(
            self.swapchain.extent.width.max(1),
            self.swapchain.extent.height.max(1),
            classify(self.swapchain.format, encoding),
        )
    }
}

// A `len`-byte host-visible buffer a copy can write, preferring cached memory
// so the writer reads it back quickly. Mapped for its whole life.
fn readback_buffer(alloc: &DeviceAllocator, len: usize) -> RenderResult<Mapped<PooledBuffer>> {
    let visible = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let usage = vk::BufferUsageFlags::TRANSFER_DST;
    let buffer = alloc
        .create_buffer(
            len as u64,
            usage,
            visible | vk::MemoryPropertyFlags::HOST_CACHED,
        )
        .or_else(|_| alloc.create_buffer(len as u64, usage, visible))?;
    let mapped = buffer.mapped_ptr().cast_const();
    if mapped.is_null() {
        return Err(RenderError::Other(
            "frame capture: read-back buffer is not host-mapped".into(),
        ));
    }
    Ok(Mapped { buffer, mapped })
}

// Copy `src_image`, in `PRESENT_SRC_KHR` once the color attachment writes
// before it are done, into `readback` laid out as `image`, return the image to
// `PRESENT_SRC_KHR`, and make the copy visible to the host.
fn encode_copy(
    device: &VkDevice,
    cmd: vk::CommandBuffer,
    src_image: vk::Image,
    readback: &PooledBuffer,
    image: &CaptureImage,
) {
    let to_src = image_barrier(
        src_image,
        vk::ImageLayout::PRESENT_SRC_KHR,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
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
            width: image.width,
            height: image.height,
            depth: 1,
        });
    let to_present = image_barrier(
        src_image,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::ImageLayout::PRESENT_SRC_KHR,
        vk::AccessFlags::TRANSFER_READ,
        vk::AccessFlags::empty(),
    );
    let to_host = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(readback.buffer())
        .offset(0)
        .size(vk::WHOLE_SIZE);
    // SAFETY: `cmd` is a command buffer in the recording state, and the caller keeps the
    // swapchain image alive until the buffer has executed.
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_src],
        );
    }
    // SAFETY: the barrier above moved `src_image` to TRANSFER_SRC_OPTIMAL, and the caller keeps
    // `readback` alive until the buffer has executed.
    unsafe {
        device.cmd_copy_image_to_buffer(
            cmd,
            src_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            readback.buffer(),
            std::slice::from_ref(&region),
        );
    }
    // SAFETY: same recording `cmd`; this returns the image to present and the copy to the host.
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE | vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[],
            std::slice::from_ref(&to_host),
            &[to_present],
        );
    }
}

// A whole-image color barrier on a swapchain image, used to flip between
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

// Classify the swapchain color format (+ resolved HDR encoding) into the
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

#[cfg(test)]
mod tests {
    use super::*;

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
