// src/vulkan/upload_ring.rs
//
// Persistent per-frame-slot upload buffers for the composite pass's transient
// HUD text geometry. Creating a vertex and an index buffer per label per frame
// put two suballocator allocations (and two `VkBuffer` creations) on the hot
// path for every line of HUD text; a bistro-sized overlay pays that dozens of
// times a frame.
//
// Instead each frame-in-flight slot keeps one host-visible, persistently mapped
// buffer. Every frame the slot's cursor resets to zero and each block of
// geometry is appended at a rolling, aligned offset; the draw binds a sub-range
// of the shared buffer. A slot is reallocated only when a frame's geometry
// exceeds its capacity, which after warm-up never happens. The frame fence
// (waited before a slot is reused) guarantees the GPU has finished reading a
// slot's buffer before the CPU overwrites or replaces it.
//
// Mirrors `directx/upload_ring.rs`; `metal/text_upload.rs` does the same job
// against one `StorageModeShared` buffer per slot.

use std::cell::RefCell;

use ash::vk;

use super::allocator::{DeviceAllocator, PooledBuffer};
use crate::gfx::fullscreen::align_up;

// Sub-range alignment. 256 bytes satisfies every offset rule a bound
// vertex / index sub-range can face, MoltenVK's translation to Metal's
// stricter buffer-offset alignment included, and costs a HUD's worth of
// labels a few kilobytes.
pub(in crate::vulkan) const UPLOAD_ALIGN: u64 = 256;

// First-allocation capacity for a slot's buffer. A HUD's worth of text is a few
// kilobytes, so this avoids any growth in practice while staying tiny.
const UPLOAD_MIN_CAPACITY: u64 = 64 * 1024;

// New capacity for a slot that must hold at least `needed` bytes, given its
// current `capacity`. Grows geometrically (doubling from the minimum) so a burst
// of small growths amortizes, but never returns less than `needed`.
fn grow_capacity(capacity: u64, needed: u64) -> u64 {
    let mut cap = capacity.max(UPLOAD_MIN_CAPACITY);
    while cap < needed {
        cap *= 2;
    }
    cap
}

// One frame slot's persistently mapped upload buffer. `buffer` is null until the
// slot's first reservation.
struct Slot {
    buffer: PooledBuffer,
    capacity: u64,
    cursor: u64,
}

impl Slot {
    fn empty() -> Self {
        Slot {
            buffer: PooledBuffer::null(),
            capacity: 0,
            cursor: 0,
        }
    }
}

// A persistently mapped upload buffer per frame-in-flight slot for transient
// geometry. Interior-mutable because the pass encoders run through `&self`; only
// the composite pass touches it, and that pass stays on the main thread (see
// `vulkan/parallel_encoder.rs`).
pub(in crate::vulkan) struct UploadRing {
    slots: Vec<RefCell<Slot>>,
}

impl UploadRing {
    pub(in crate::vulkan) fn new(frames: usize) -> Self {
        UploadRing {
            slots: (0..frames.max(1))
                .map(|_| RefCell::new(Slot::empty()))
                .collect(),
        }
    }

    // Begin a frame for `frame`'s slot: reset the write cursor and ensure the
    // buffer holds at least `needed` bytes, reallocating it if not. The caller
    // must invoke this once per frame before any `push`, after the frame fence
    // has confirmed the GPU is done with this slot.
    pub(in crate::vulkan) fn reserve(
        &self,
        alloc: &DeviceAllocator,
        frame: usize,
        needed: u64,
    ) -> Result<(), String> {
        let mut slot = self.slots[frame % self.slots.len()].borrow_mut();
        slot.cursor = 0;
        if needed <= slot.capacity {
            return Ok(());
        }
        let new_cap = grow_capacity(slot.capacity, needed);
        let buffer = alloc.create_buffer(
            new_cap,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        if buffer.mapped_ptr().is_null() {
            return Err("text upload buffer is not host-mapped".to_string());
        }
        // Replacing `buffer` retires the old one through the allocator, which
        // withholds its range until every in-flight frame has passed.
        slot.buffer = buffer;
        slot.capacity = new_cap;
        Ok(())
    }

    // Append `bytes` at the slot's next aligned offset and return the buffer to
    // bind plus the offset of the copy. Errors if the running total would exceed
    // the reserved capacity, which cannot happen when `reserve` was called with
    // the aligned-block sum of the same blocks.
    pub(in crate::vulkan) fn push(
        &self,
        frame: usize,
        bytes: &[u8],
    ) -> Result<(vk::Buffer, vk::DeviceSize), String> {
        let mut slot = self.slots[frame % self.slots.len()].borrow_mut();
        let offset = align_up(slot.cursor, UPLOAD_ALIGN);
        let end = offset + bytes.len() as u64;
        if end > slot.capacity {
            return Err(format!(
                "text upload ring overflow: need {end} bytes, reserved {}",
                slot.capacity
            ));
        }
        // SAFETY: the slot's buffer was created HOST_VISIBLE | HOST_COHERENT with `capacity` bytes
        // so `mapped_ptr()` is a live mapping of that length, and the bounds check above proved
        // `offset + bytes.len()` is within it. `bytes` is a separate borrow, so the ranges cannot
        // overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                slot.buffer.mapped_ptr().add(offset as usize),
                bytes.len(),
            );
        }
        slot.cursor = end;
        Ok((slot.buffer.buffer(), offset))
    }

    // Release every slot's buffer. Called from the context teardown while the
    // allocator is still alive, so each buffer's range returns to its block.
    pub(in crate::vulkan) fn destroy(&self) {
        for slot in &self.slots {
            *slot.borrow_mut() = Slot::empty();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grow_capacity_starts_at_minimum() {
        assert_eq!(grow_capacity(0, 1), UPLOAD_MIN_CAPACITY);
        assert_eq!(grow_capacity(0, 0), UPLOAD_MIN_CAPACITY);
    }

    #[test]
    fn grow_capacity_doubles_until_it_fits() {
        let need = UPLOAD_MIN_CAPACITY * 3 + 1;
        let cap = grow_capacity(0, need);
        assert!(cap >= need);
        assert_eq!(cap, UPLOAD_MIN_CAPACITY * 4);
    }

    #[test]
    fn grow_capacity_never_shrinks_below_existing() {
        let cap = grow_capacity(UPLOAD_MIN_CAPACITY * 8, 10);
        assert_eq!(cap, UPLOAD_MIN_CAPACITY * 8);
    }

    // The composite pass reserves `text_upload_bytes` for the frame and then
    // appends each block at the ring's alignment: the reservation has to bound
    // the cursor for every block sequence, or a push could overflow mid-frame.
    #[test]
    fn reserved_bytes_bound_the_ring_cursor() {
        let blocks: [u64; 5] = [128, 12, 4096, 1, 255];
        let total: u64 = blocks.iter().map(|&n| align_up(n, UPLOAD_ALIGN)).sum();
        let mut cursor = 0u64;
        for &n in &blocks {
            cursor = align_up(cursor, UPLOAD_ALIGN) + n;
            assert!(cursor <= total, "cursor {cursor} exceeded reserved {total}");
        }
    }
}
