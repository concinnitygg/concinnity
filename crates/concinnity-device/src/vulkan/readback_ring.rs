//! A ring of host-visible buffers, one per frame in flight, that a frame
//! copies a compute pass's output into and the CPU reads back once the fence
//! has retired that frame, so a reading is never waited on. Zeroed at
//! creation, so a slot no frame wrote reads back as zeroes.

use core::marker::PhantomData;

use ash::vk;
use concinnity_engine::render::backend::error::RenderResult;

use super::allocator::{DeviceAllocator, PooledBuffer};
use crate::vulkan::record::Recorder;

/// `len` values of `T` per frame in flight, persistently mapped.
pub(in crate::vulkan) struct ReadbackRing<T> {
    buffers: Vec<PooledBuffer>,
    len: usize,
    _values: PhantomData<T>,
}

impl<T: bytemuck::Pod> ReadbackRing<T> {
    /// A slot of `len` values for each of `frames` frames in flight,
    /// preferring cached memory so the host reads it quickly.
    pub(in crate::vulkan) fn new(
        alloc: &DeviceAllocator,
        frames: usize,
        len: usize,
    ) -> RenderResult<Self> {
        let bytes = len * size_of::<T>();
        let visible =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let usage = vk::BufferUsageFlags::TRANSFER_DST;
        let buffers = (0..frames.max(1))
            .map(|_| {
                let size = bytes as vk::DeviceSize;
                let buffer = alloc
                    .create_buffer(size, usage, visible | vk::MemoryPropertyFlags::HOST_CACHED)
                    .or_else(|_| alloc.create_buffer(size, usage, visible))?;
                // SAFETY: the mapping is a live HOST_COHERENT range of at least `bytes` bytes,
                // and no GPU work references the buffer yet.
                unsafe { std::ptr::write_bytes(buffer.mapped_ptr(), 0, bytes) };
                Ok(buffer)
            })
            .collect::<RenderResult<Vec<_>>>()?;
        Ok(Self {
            buffers,
            len,
            _values: PhantomData,
        })
    }

    /// What frame slot `slot` holds, once the fence has retired its writer.
    pub(in crate::vulkan) fn read(&self, slot: usize) -> Option<&[T]> {
        let ptr = self
            .buffers
            .get(slot)?
            .mapped_ptr()
            .cast_const()
            .cast::<T>();
        if ptr.is_null() || !ptr.is_aligned() {
            return None;
        }
        // SAFETY: `ptr` is non-null and aligned for `T` (checked above), and maps this slot's
        // host-visible buffer of `len` values, live while `self` holds it; the fence wait gated
        // the copy that wrote it, its host barrier made the copy visible, and every bit pattern
        // is a valid `T`.
        Some(unsafe { std::slice::from_raw_parts(ptr, self.len) })
    }

    /// Record the copy of the first `count` values of `src`, which a compute
    /// pass just wrote, into frame slot `slot`, made visible to the host. The
    /// compute stage the copy releases to takes no access: it orders this
    /// frame's copy ahead of the next frame's dispatch writing `src` again.
    pub(in crate::vulkan) fn copy_from(
        &self,
        rec: &Recorder<'_>,
        src: vk::Buffer,
        count: usize,
        slot: usize,
    ) {
        let Some(dst) = self.buffers.get(slot) else {
            return;
        };
        memory_barrier(
            rec,
            (
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_WRITE,
            ),
            (
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_READ,
            ),
        );
        let copy = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: (count.min(self.len) * size_of::<T>()) as vk::DeviceSize,
        };
        rec.copy_buffer(src, dst.buffer(), std::slice::from_ref(&copy));
        memory_barrier(
            rec,
            (
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_WRITE,
            ),
            (
                vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::HOST_READ,
            ),
        );
    }

    /// Free the buffers. Called from `Drop for VkContext` after the device
    /// idles.
    pub(in crate::vulkan) fn destroy(&mut self) {
        self.buffers.clear();
    }
}

/// One global memory barrier from `src` to `dst`, each a stage and its
/// access.
pub(in crate::vulkan) fn memory_barrier(
    rec: &Recorder<'_>,
    (src_stage, src_access): (vk::PipelineStageFlags, vk::AccessFlags),
    (dst_stage, dst_access): (vk::PipelineStageFlags, vk::AccessFlags),
) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);
    rec.pipeline_barrier(
        src_stage,
        dst_stage,
        std::slice::from_ref(&barrier),
        &[],
        &[],
    );
}
