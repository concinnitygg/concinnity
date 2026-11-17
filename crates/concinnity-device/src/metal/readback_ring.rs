//! A ring of shared buffers, one per frame in flight, that a frame's GPU work
//! writes and the CPU reads back once the frames-in-flight fence has retired
//! that frame, so a reading is never waited on. Created zeroed, so a slot no
//! frame wrote reads back as zeroes.
#![deny(unsafe_op_in_unsafe_fn)]

use core::marker::PhantomData;

use concinnity_engine::render::backend::error::RenderResult;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::error::allocation_failed;

/// `len` values of `T` per frame in flight.
pub(super) struct ReadbackRing<T> {
    buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    len: usize,
    _values: PhantomData<T>,
}

impl<T: bytemuck::Pod> ReadbackRing<T> {
    /// A slot of `len` values for each of `frames` frames in flight.
    pub(super) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        frames: usize,
        len: usize,
        label: &'static str,
    ) -> RenderResult<Self> {
        let bytes = (len * size_of::<T>()).max(16);
        let buffers = (0..frames.max(1))
            .map(|_| {
                device
                    .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
                    .ok_or_else(|| allocation_failed(label))
            })
            .collect::<RenderResult<Vec<_>>>()?;
        Ok(Self {
            buffers,
            len,
            _values: PhantomData,
        })
    }

    /// The buffer slot `slot`'s frame writes.
    pub(super) fn buffer(&self, slot: usize) -> Option<&ProtocolObject<dyn MTLBuffer>> {
        self.buffers.get(slot).map(|b| &**b)
    }

    /// What slot `slot` holds, once the fence has retired its writer.
    pub(super) fn read(&self, slot: usize) -> Option<&[T]> {
        let buffer = self.buffers.get(slot)?;
        // SAFETY: `buffer` is shared storage of at least `len` values of `T`, so `contents()` is
        // a live CPU mapping of them for as long as `self` holds it; `begin` waited on the fence
        // that retired the frame writing this slot, so the GPU no longer writes it; `contents()`
        // is page aligned, past `align_of::<T>()` for these Pod values; and every bit pattern is
        // a valid `T`.
        Some(unsafe {
            core::slice::from_raw_parts(buffer.contents().as_ptr().cast::<T>(), self.len)
        })
    }
}
