// src/directx/upload_ring.rs
//
// Persistent per-frame-slot upload buffers for transient per-frame geometry
// (HUD text labels, expanded line ribbons). Allocating those with
// `CreateCommittedResource` per draw per frame is the classic D3D12 hot-path
// anti-pattern (each commit is hundreds of micros); for the bistro HUD that was
// the single largest slice of per-frame CPU.
//
// Instead each frame-in-flight slot keeps one persistently-mapped upload buffer.
// Every frame the slot's cursor is reset to zero and each block of geometry is
// appended at a rolling, aligned offset; the draw binds a sub-view into the
// shared buffer. The buffer is grown (reallocated larger) only when a frame
// exceeds the current capacity, which after warm-up never happens. The frame
// fence (waited before a slot is reused) guarantees the GPU has finished
// reading a slot's buffer before the CPU overwrites or grows it.

use std::cell::RefCell;

use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::allocator::{DeviceAllocator, PooledBuffer};
use crate::directx::texture::create_buffer;
// Sub-range offset rounding, shared with the other backends' text uploads.
pub(in crate::directx) use crate::gfx::fullscreen::align_up;

// Sub-allocation alignment. 16 bytes satisfies the index-buffer address
// requirement (a multiple of the R16 element size) and keeps each vertex
// sub-view comfortably aligned.
pub(in crate::directx) const UPLOAD_ALIGN: u64 = 16;

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

// One frame slot's persistently-mapped upload buffer. `base` is the CPU map
// pointer (null until the first allocation) and `gpu_va` its GPU virtual
// address; both are re-read whenever the buffer is grown.
struct Slot {
    buffer: Option<PooledBuffer>,
    base: *mut u8,
    gpu_va: u64,
    capacity: u64,
    cursor: u64,
}

impl Slot {
    fn empty() -> Self {
        Slot {
            buffer: None,
            base: std::ptr::null_mut(),
            gpu_va: 0,
            capacity: 0,
            cursor: 0,
        }
    }
}

// A persistently-mapped upload buffer per frame-in-flight slot for transient
// geometry. Interior-mutable (the pass encoders run through `&self`), matching
// the rest of the per-frame DX state.
pub(in crate::directx) struct UploadRing {
    slots: Vec<RefCell<Slot>>,
}

impl UploadRing {
    pub(in crate::directx) fn new(frames: usize) -> Self {
        UploadRing {
            slots: (0..frames).map(|_| RefCell::new(Slot::empty())).collect(),
        }
    }

    // Begin a frame for `frame`'s slot: reset the write cursor and ensure the
    // buffer holds at least `needed` bytes, growing (and remapping) it if not.
    // The caller must invoke this once per frame before any `push`, after the
    // frame fence has confirmed the GPU is done with this slot.
    pub(in crate::directx) fn reserve(
        &self,
        alloc: &DeviceAllocator,
        frame: usize,
        needed: u64,
    ) -> Result<(), String> {
        let mut slot = self.slots[frame].borrow_mut();
        slot.cursor = 0;
        if needed <= slot.capacity {
            return Ok(());
        }
        let new_cap = grow_capacity(slot.capacity, needed);
        let buffer = create_buffer(
            alloc,
            new_cap,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut base = std::ptr::null_mut::<std::ffi::c_void>();
        // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local
        // that receives the mapping.
        unsafe { buffer.Map(0, None, Some(&mut base)) }.map_err(|e| format!("upload map: {e}"))?;
        // SAFETY: a property query on a live resource; it only reads.
        let gpu_va = unsafe { buffer.GetGPUVirtualAddress() };
        // Replacing `buffer` drops the old resource (and unmaps it); the frame
        // fence already proved the GPU finished reading it.
        slot.buffer = Some(buffer);
        slot.base = base as *mut u8;
        slot.gpu_va = gpu_va;
        slot.capacity = new_cap;
        Ok(())
    }

    // Append `bytes` at the slot's next aligned offset and return the GPU
    // virtual address of the copy. Errors if the running total would exceed the
    // reserved capacity, which cannot happen when `reserve` was called with the
    // aligned-block sum of the same blocks.
    pub(in crate::directx) fn push(&self, frame: usize, bytes: &[u8]) -> Result<u64, String> {
        let mut slot = self.slots[frame].borrow_mut();
        let offset = align_up(slot.cursor, UPLOAD_ALIGN);
        let end = offset + bytes.len() as u64;
        if end > slot.capacity {
            return Err(format!(
                "upload ring overflow: need {end} bytes, reserved {}",
                slot.capacity
            ));
        }
        // SAFETY: `base` is the persistent map of a buffer of `capacity` bytes;
        // `offset + bytes.len() <= capacity` checked above; the slot is only
        // touched on the main render thread.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                slot.base.add(offset as usize),
                bytes.len(),
            );
        }
        slot.cursor = end;
        Ok(slot.gpu_va + offset)
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
}
