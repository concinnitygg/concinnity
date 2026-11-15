//! A ring of READBACK buffers, one per frame in flight, that a frame copies a
//! compute pass's output into and the CPU reads back once the fence has
//! retired that frame, so a reading is never waited on. Zeroed at creation,
//! so a slot no frame wrote reads back as zeroes.

use core::marker::PhantomData;

use concinnity_engine::render::backend::error::RenderResult;
use windows::Win32::Graphics::Direct3D12::*;

use super::allocator::{DeviceAllocator, PooledBuffer};
use crate::directx::error::map_hresult;
use crate::directx::texture::transition_barrier;

/// `len` values of `T` per frame in flight, persistently mapped.
pub(in crate::directx) struct ReadbackRing<T> {
    buffers: Vec<PooledBuffer>,
    mapped: Vec<*const T>,
    len: usize,
    _values: PhantomData<T>,
}

impl<T: bytemuck::Pod> ReadbackRing<T> {
    /// A slot of `len` values for each frame `alloc` keeps in flight.
    pub(in crate::directx) fn new(
        alloc: &DeviceAllocator,
        len: usize,
        label: &str,
    ) -> RenderResult<Self> {
        let bytes = len * size_of::<T>();
        let frames = alloc.frames_in_flight();
        let mut buffers = Vec::with_capacity(frames);
        let mut mapped = Vec::with_capacity(frames);
        for _ in 0..frames {
            let buf = alloc.alloc_buffer(
                bytes as u64,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // READBACK heaps allow leaving Map active across submissions; the
            // pointer stays valid for the resource's lifetime.
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { buf.Map(0, None, Some(&raw mut ptr)) }
                .map_err(|e| map_hresult(e.code(), label))?;
            // SAFETY: the mapping covers `bytes` bytes, and no GPU work references the buffer
            // yet.
            unsafe { std::ptr::write_bytes(ptr.cast::<u8>(), 0, bytes) };
            mapped.push(ptr.cast_const().cast::<T>());
            buffers.push(buf);
        }
        Ok(Self {
            buffers,
            mapped,
            len,
            _values: PhantomData,
        })
    }

    /// What frame slot `slot` holds, once the fence has retired its writer.
    pub(in crate::directx) fn read(&self, slot: usize) -> Option<&[T]> {
        let &ptr = self.mapped.get(slot)?;
        // SAFETY: `ptr` is this slot's persistent mapping of a READBACK buffer of `len` values,
        // live while `self` holds the buffer; the fence wait ahead of this call retired the copy
        // that wrote it; `Map` on a buffer returns at least 16-byte alignment, past
        // `align_of::<T>()` for these Pod values; and every bit pattern is a valid `T`.
        Some(unsafe { std::slice::from_raw_parts(ptr, self.len) })
    }

    /// Record the copy of the first `count` values of `src`, a buffer a
    /// compute pass just wrote and that rests in `UNORDERED_ACCESS`, into
    /// frame slot `slot`. The transition out of `UNORDERED_ACCESS` makes the
    /// pass's writes visible to the copy, and `src` returns to it after.
    pub(in crate::directx) fn copy_from(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        src: &ID3D12Resource,
        count: usize,
        slot: usize,
    ) {
        let Some(dst) = self.buffers.get(slot) else {
            return;
        };
        let to_copy = transition_barrier(
            src,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        // SAFETY: the command list is recording, and `src` is live.
        unsafe { cmd.ResourceBarrier(&[to_copy]) };
        let bytes = (count.min(self.len) * size_of::<T>()) as u64;
        // SAFETY: the command list is recording, and both buffers are live and hold at least
        // `bytes` bytes.
        unsafe { cmd.CopyBufferRegion(&**dst, 0, src, 0, bytes) };
        let to_uav = transition_barrier(
            src,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        // SAFETY: the command list is recording, and `src` is live.
        unsafe { cmd.ResourceBarrier(&[to_uav]) };
    }
}
