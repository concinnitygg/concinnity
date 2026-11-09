//! One persistently mapped upload-heap block per frame in flight, which a
//! pass writes its frame's block into and binds by address.

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::render::backend::error::RenderResult;
use windows::Win32::Graphics::Direct3D12::*;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::com;
use super::context::align256;
use super::error::map_hresult;

pub(in crate::directx) struct MappedBlocks {
    buffers: Vec<PooledBuffer>,
    ptrs: Vec<*mut u8>,
    size: usize,
}

impl MappedBlocks {
    pub(in crate::directx) fn new(
        alloc: &DeviceAllocator,
        size: usize,
        label: &str,
    ) -> RenderResult<Self> {
        let size = align256(size as u64);
        let frames = alloc.frames_in_flight();
        let mut buffers = Vec::with_capacity(frames);
        let mut ptrs = Vec::with_capacity(frames);
        for _ in 0..frames {
            let buf = alloc.alloc_buffer(
                size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { buf.Map(0, None, Some(&raw mut ptr)) }
                .map_err(|e| map_hresult(e.code(), label))?;
            ptrs.push(ptr.cast::<u8>());
            buffers.push(buf);
        }
        Ok(Self {
            buffers,
            ptrs,
            size: usize::saturating_from(size),
        })
    }

    // Frame `frame_idx`'s block's address; `None` past the frames in flight.
    pub(in crate::directx) fn address(&self, frame_idx: usize) -> Option<u64> {
        self.buffers.get(frame_idx).map(|b| com::gpu_va(b))
    }

    // Write `value` into frame `frame_idx`'s block and return its address.
    pub(in crate::directx) fn write<T: bytemuck::NoUninit>(
        &self,
        frame_idx: usize,
        value: &T,
    ) -> Option<u64> {
        self.write_bytes(frame_idx, bytemuck::bytes_of(value))
    }

    // Write `bytes`, at most the block's size, into frame `frame_idx`'s block
    // and return its address; `None` past the frames in flight.
    pub(in crate::directx) fn write_bytes(&self, frame_idx: usize, bytes: &[u8]) -> Option<u64> {
        let (Some(buffer), Some(&dst)) = (self.buffers.get(frame_idx), self.ptrs.get(frame_idx))
        else {
            return None;
        };
        let bytes = bytes.get(..bytes.len().min(self.size)).unwrap_or_default();
        // SAFETY: the destination is the persistent mapping of this frame's UPLOAD-heap buffer,
        // `size` bytes long and `bytes` no longer, which the GPU no longer reads once this
        // frame's slot has been waited on; the source is a separate live value, so the ranges
        // cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        }
        Some(com::gpu_va(buffer))
    }
}
