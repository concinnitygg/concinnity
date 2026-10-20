//! Persistent per-frame-slot upload buffers for the composite pass's transient
//! HUD text geometry. Creating a vertex and an index buffer per label per frame
//! put two suballocator allocations (and two `VkBuffer` creations) on the hot
//! path for every line of HUD text; a bistro-sized overlay pays that dozens of
//! times a frame.
//!
//! Instead each frame-in-flight slot keeps one host-visible, persistently mapped
//! buffer. Every frame the slot's cursor resets to zero and each block of
//! geometry is appended at a rolling, aligned offset; the draw binds a sub-range
//! of the shared buffer. A slot is reallocated only when a frame's geometry
//! exceeds its capacity, which after warm-up never happens. The frame fence
//! (waited before a slot is reused) guarantees the GPU has finished reading a
//! slot's buffer before the CPU overwrites or replaces it.
//!
//! The ring itself is `concinnity_core::render::buffers::upload_ring`; this binds it to
//! pooled host-visible buffers. `metal/text_upload.rs` does the same job against
//! one `StorageModeShared` buffer per slot.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_core::render::backend::error::{RenderError, RenderResult};
use concinnity_core::render::buffers::upload_ring::{self, UploadBuffer};

use super::allocator::{DeviceAllocator, PooledBuffer};

// Sub-range alignment. 256 bytes satisfies every offset rule a bound
// vertex / index sub-range can face, MoltenVK's translation to Metal's
// stricter buffer-offset alignment included, and costs a HUD's worth of
// labels a few kilobytes.
pub(in crate::vulkan) const UPLOAD_ALIGN: u64 = 256;

// First-allocation capacity for a slot's buffer. A HUD's worth of text is a few
// kilobytes, so this avoids any growth in practice while staying tiny.
const UPLOAD_MIN_CAPACITY: u64 = 64 * 1024;

impl UploadBuffer for PooledBuffer {
    fn write(&self, offset: u64, bytes: &[u8]) {
        self.write_bytes(usize::saturating_from(offset), bytes);
    }
}

// One persistently mapped vertex + index upload buffer per frame slot.
pub(in crate::vulkan) struct UploadRing {
    ring: upload_ring::UploadRing<PooledBuffer>,
}

impl UploadRing {
    pub(in crate::vulkan) fn new(frames: usize) -> Self {
        Self {
            ring: upload_ring::UploadRing::new(frames, UPLOAD_ALIGN, UPLOAD_MIN_CAPACITY),
        }
    }

    // Start `frame`'s slot over with room for `needed` bytes, growing its buffer
    // when it is too small. The frame fence for `frame` has been waited.
    pub(in crate::vulkan) fn reserve(
        &self,
        alloc: &DeviceAllocator,
        frame: usize,
        needed: u64,
    ) -> RenderResult<()> {
        self.ring.reserve(frame, needed, |capacity| {
            let buffer = alloc.create_buffer(
                capacity,
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            if buffer.mapped_ptr().is_null() {
                return Err(RenderError::Other(
                    "text upload buffer is not host-mapped".to_string(),
                ));
            }
            Ok(buffer)
        })
    }

    // Copy `bytes` into `frame`'s slot, returning the buffer and offset to bind.
    pub(in crate::vulkan) fn push(
        &self,
        frame: usize,
        bytes: &[u8],
    ) -> RenderResult<(vk::Buffer, vk::DeviceSize)> {
        self.ring
            .push(frame, bytes, |buffer, offset| (buffer.buffer(), offset))
            .map_err(|o| {
                RenderError::Other(format!(
                    "text upload ring overflow: need {} bytes, reserved {}",
                    o.need, o.reserved
                ))
            })
    }

    pub(in crate::vulkan) fn destroy(&self) {
        self.ring.clear();
    }
}
