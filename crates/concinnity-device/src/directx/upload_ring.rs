//! Persistent per-frame-slot upload buffers for transient per-frame geometry
//! (HUD text labels, expanded line ribbons). Allocating those with
//! `CreateCommittedResource` per draw per frame is the classic D3D12 hot-path
//! anti-pattern (each commit is hundreds of micros); for the bistro HUD that was
//! the single largest slice of per-frame CPU.
//!
//! Instead each frame-in-flight slot keeps one persistently-mapped upload buffer.
//! Every frame the slot's cursor is reset to zero and each block of geometry is
//! appended at a rolling, aligned offset; the draw binds a sub-view into the
//! shared buffer. The buffer is grown (reallocated larger) only when a frame
//! exceeds the current capacity, which after warm-up never happens. The frame
//! fence (waited before a slot is reused) guarantees the GPU has finished
//! reading a slot's buffer before the CPU overwrites or grows it.

//!
//! The ring itself is `concinnity_core::render::upload_ring`; this binds it to
//! pooled UPLOAD-heap buffers.

use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::upload_ring::{self, UploadBuffer};
use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::allocator::{DeviceAllocator, PooledBuffer};
use crate::directx::com;

// Sub-allocation alignment. 16 bytes satisfies the index-buffer address
// requirement (a multiple of the R16 element size) and keeps each vertex
// sub-view comfortably aligned.
pub(in crate::directx) const UPLOAD_ALIGN: u64 = 16;

// First-allocation capacity for a slot's buffer. A HUD's worth of text is a few
// kilobytes, so this avoids any growth in practice while staying tiny.
const UPLOAD_MIN_CAPACITY: u64 = 64 * 1024;

// One slot's buffer, its persistent map and its GPU address.
struct MappedUpload {
    _buffer: PooledBuffer,
    base: *mut u8,
    gpu_va: u64,
}

impl UploadBuffer for MappedUpload {
    fn write(&self, offset: u64, bytes: &[u8]) {
        // SAFETY: `base` is the persistent map of the buffer the ring sized, and
        // the ring writes only inside that capacity. Every ring is reserved and
        // pushed on the main thread only, and the frame fence retired the GPU's
        // last read of this slot.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.base.add(offset as usize),
                bytes.len(),
            );
        }
    }
}

// One persistently mapped upload buffer per frame slot.
pub(in crate::directx) struct UploadRing {
    ring: upload_ring::UploadRing<MappedUpload>,
}

impl UploadRing {
    pub(in crate::directx) fn new(frames: usize) -> Self {
        Self {
            ring: upload_ring::UploadRing::new(frames, UPLOAD_ALIGN, UPLOAD_MIN_CAPACITY),
        }
    }

    // Start `frame`'s slot over with room for `needed` bytes, growing its buffer
    // when it is too small. The frame fence for `frame` has been waited.
    pub(in crate::directx) fn reserve(
        &self,
        alloc: &DeviceAllocator,
        frame: usize,
        needed: u64,
    ) -> RenderResult<()> {
        self.ring.reserve(frame, needed, |capacity| {
            let buffer = alloc.alloc_buffer(
                capacity,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut base = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { buffer.Map(0, None, Some(&raw mut base)) }
                .map_err(|e| super::error::map_hresult(e.code(), "upload map"))?;
            Ok(MappedUpload {
                gpu_va: com::gpu_va(&buffer),
                base: base.cast::<u8>(),
                _buffer: buffer,
            })
        })
    }

    // Copy `bytes` into `frame`'s slot, returning the GPU address to bind.
    pub(in crate::directx) fn push(&self, frame: usize, bytes: &[u8]) -> RenderResult<u64> {
        self.ring
            .push(frame, bytes, |mapped, offset| mapped.gpu_va + offset)
            .map_err(|o| {
                RenderError::Other(format!(
                    "upload ring overflow: need {} bytes, reserved {}",
                    o.need, o.reserved
                ))
            })
    }
}
