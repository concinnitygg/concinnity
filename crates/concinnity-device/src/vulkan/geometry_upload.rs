//! Geometry writes into the shared vertex / index buffers without a CPU wait.
//!
//! `stage` copies the bytes into a persistently-mapped host-visible ring and
//! queues a region copy. `flush` records every queued copy into one command
//! buffer and submits it on the graphics queue, where submission order puts it
//! ahead of anything submitted later: the frame's own buffers, the probe
//! bakes, a one-shot RT build. An opening barrier holds the copies until every
//! earlier read of the buffers is done, and a closing one makes the writes
//! visible to every later read on the queue, the pair D3D12's COPY_DEST
//! transitions give its backend. `draw_frame` flushes right after its
//! frame-slot wait, and `wait_idle` flushes before it waits, so an idle device
//! has applied every staged write.
//!
//! The ring's bytes and the command buffer are held `frames_in_flight + 1`
//! ticks past the submit that read them, the window the device allocator
//! retires on; a ring buffer replaced by a larger one retires through the
//! allocator itself.

use ash::vk;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::retire_pool::RetirePool;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::error::map_vk_result;
use crate::suballoc::staging::{STAGING_ALIGN, StagingRing, grown_capacity, reserved_capacity};

// Which shared buffer a staged write lands in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::vulkan) enum GeometryTarget {
    Vertex,
    Index,
}

impl GeometryTarget {
    fn pick(self, dest: GeometryDest) -> vk::Buffer {
        match self {
            GeometryTarget::Vertex => dest.vertex,
            GeometryTarget::Index => dest.index,
        }
    }
}

// The shared buffers a flush copies into.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GeometryDest {
    pub vertex: vk::Buffer,
    pub index: vk::Buffer,
}

// The command pool and queue a flush records and submits on.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct CopySubmit {
    pub command_pool: vk::CommandPool,
    pub queue: vk::Queue,
}

// One queued region copy out of the ring. `src` is the ring buffer the bytes
// were placed in, which a later growth may have replaced.
struct StagedCopy {
    target: GeometryTarget,
    src: PooledBuffer,
    region: vk::BufferCopy,
}

struct Staging {
    buffer: PooledBuffer,
    ring: StagingRing,
}

pub(in crate::vulkan) struct GeometryUploads {
    staging: Option<Staging>,
    queued: Vec<StagedCopy>,
    command_buffers: RetirePool<vk::CommandBuffer>,
    tick: u64,
    depth: u64,
}

impl GeometryUploads {
    pub(in crate::vulkan) fn new(frames_in_flight: usize) -> Self {
        Self {
            staging: None,
            queued: Vec::new(),
            command_buffers: RetirePool::new(),
            tick: 0,
            depth: frames_in_flight as u64 + 1,
        }
    }

    // Drop the ring and every queued write, and forget the command buffers
    // their pool is about to free. The caller has idled the device.
    pub(in crate::vulkan) fn destroy(&mut self) {
        self.staging = None;
        self.queued.clear();
        self.command_buffers = RetirePool::new();
    }

    // Advance the frame tick. Called after the frame-slot fence wait, which is
    // what proves a submit `depth` ticks old has retired.
    pub(in crate::vulkan) fn begin_frame(&mut self) {
        self.tick += 1;
        if let Some(staging) = self.staging.as_mut() {
            staging.ring.reclaim(self.tick);
        }
    }

    // Create the ring at load, sized for a stream that may stage up to
    // `expected` bytes in one frame, so its first burst does not pay for an
    // allocation mid-frame. Never shrinks a ring.
    pub(in crate::vulkan) fn reserve(
        &mut self,
        alloc: &DeviceAllocator,
        expected: u64,
    ) -> RenderResult<()> {
        let capacity = reserved_capacity(expected);
        if self
            .staging
            .as_ref()
            .is_none_or(|s| s.ring.capacity() < capacity)
        {
            self.replace_ring(alloc, capacity)?;
        }
        Ok(())
    }

    // Queue a write of `bytes` at `dst_offset` in `target`'s buffer.
    pub(in crate::vulkan) fn stage(
        &mut self,
        alloc: &DeviceAllocator,
        target: GeometryTarget,
        dst_offset: u64,
        bytes: &[u8],
    ) -> RenderResult<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let size = bytes.len() as u64;
        let src_offset = match self.place(size) {
            Some(offset) => offset,
            None => {
                self.grow(alloc, size)?;
                self.place(size).ok_or_else(|| {
                    RenderError::Other("geometry staging: grown ring refused a write".into())
                })?
            }
        };
        let staging = self
            .staging
            .as_ref()
            .ok_or_else(|| RenderError::Other("geometry staging: no ring".into()))?;
        staging.buffer.write_bytes(src_offset as usize, bytes);
        self.queued.push(StagedCopy {
            target,
            src: staging.buffer.clone(),
            region: vk::BufferCopy::default()
                .src_offset(src_offset)
                .dst_offset(dst_offset)
                .size(size),
        });
        Ok(())
    }

    fn place(&mut self, size: u64) -> Option<u64> {
        self.staging
            .as_mut()
            .and_then(|s| s.ring.alloc(size, STAGING_ALIGN))
    }

    // Replace the ring with one that holds `size` more bytes. The old buffer
    // stays alive through the queued copies that name it, and its destruction
    // is deferred past the submits already in flight.
    fn grow(&mut self, alloc: &DeviceAllocator, size: u64) -> RenderResult<()> {
        let current = self.staging.as_ref().map_or(0, |s| s.ring.capacity());
        let capacity = grown_capacity(current, size);
        if current > 0 {
            tracing::info!(
                "geometry staging ring grew {} -> {} KiB for a {} KiB write",
                current / 1024,
                capacity / 1024,
                size / 1024
            );
        }
        self.replace_ring(alloc, capacity)
    }

    fn replace_ring(&mut self, alloc: &DeviceAllocator, capacity: u64) -> RenderResult<()> {
        let buffer = alloc.create_buffer(
            capacity,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        self.staging = Some(Staging {
            buffer,
            ring: StagingRing::new(capacity),
        });
        Ok(())
    }

    // Record and submit every queued copy. A no-op when nothing is queued.
    pub(in crate::vulkan) fn flush(
        &mut self,
        device: &ash::Device,
        submit_on: CopySubmit,
        dest: GeometryDest,
    ) -> RenderResult<()> {
        if self.queued.is_empty() {
            return Ok(());
        }
        let cmd = match self.command_buffers.pop_due(self.tick, self.depth) {
            Some(cmd) => cmd,
            None => {
                let info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(submit_on.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                // SAFETY: the allocate-info names this device's own pool and is live for the call.
                unsafe { device.allocate_command_buffers(&info) }
                    .map_err(|e| map_vk_result(e, "geometry copy allocate"))?[0]
            }
        };
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `cmd` came from a pool created with RESET_COMMAND_BUFFER, and its last submit
        // retired `depth` ticks ago, so it is not in flight; reset then begin puts it in the
        // recording state.
        unsafe {
            device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| map_vk_result(e, "geometry copy reset"))?;
            device
                .begin_command_buffer(cmd, &begin)
                .map_err(|e| map_vk_result(e, "geometry copy begin"))?;
        }
        record_copies(device, cmd, &self.queued, dest);
        // SAFETY: `cmd` is in the recording state, which is what `end_command_buffer` requires.
        unsafe { device.end_command_buffer(cmd) }
            .map_err(|e| map_vk_result(e, "geometry copy end"))?;
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        // SAFETY: `cmd` was ended and belongs to this device, and `submit` borrows it for the call.
        unsafe {
            device.queue_submit(
                submit_on.queue,
                std::slice::from_ref(&submit),
                vk::Fence::null(),
            )
        }
        .map_err(|e| map_vk_result(e, "geometry copy submit"))?;

        let retire_at = self.tick + self.depth;
        if let Some(staging) = self.staging.as_mut() {
            staging.ring.seal(retire_at);
        }
        self.queued.clear();
        self.command_buffers.push(self.tick, cmd);
        Ok(())
    }
}

fn record_copies(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    queued: &[StagedCopy],
    dest: GeometryDest,
) {
    // Write-after-read: the copies wait for every earlier read of the buffers,
    // so an in-place overwrite, or a region reused before the reads of its
    // last occupant retired, is safe.
    // SAFETY: `cmd` is in the recording state, and the barrier names no resource.
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[],
        );
    }
    for copy in queued {
        // SAFETY: `cmd` is in the recording state, and both buffers are live for the call: the
        // shared one through the caller's geometry, the source ring through its queued copy.
        unsafe {
            device.cmd_copy_buffer(
                cmd,
                copy.src.buffer(),
                copy.target.pick(dest),
                std::slice::from_ref(&copy.region),
            );
        }
    }
    let visible = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(
            vk::AccessFlags::VERTEX_ATTRIBUTE_READ
                | vk::AccessFlags::INDEX_READ
                | vk::AccessFlags::SHADER_READ
                | vk::AccessFlags::TRANSFER_READ,
        );
    // SAFETY: `cmd` is in the recording state, and the barrier names no resource.
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&visible),
            &[],
            &[],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suballoc::staging::STAGING_MIN_CAPACITY;
    use crate::vulkan::test_gpu::{TestGpu, test_gpu};

    const HOST: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
        vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
            | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
    );

    // Host-visible stand-ins for the shared buffers, so a test reads the
    // copies back through their mappings.
    struct Harness {
        alloc: DeviceAllocator,
        pool: vk::CommandPool,
        queue: vk::Queue,
        vertex: PooledBuffer,
        index: PooledBuffer,
    }

    impl Harness {
        fn new(gpu: &TestGpu, bytes: u64) -> Self {
            let alloc = DeviceAllocator::new(&gpu.instance, gpu.physical_device, &gpu.device, 2);
            let info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(0)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            // SAFETY: the create-info is live for the call and names this device's queue family.
            let pool = unsafe { gpu.device.create_command_pool(&info, None) }.expect("pool");
            // SAFETY: family 0, index 0 is the one queue `test_gpu` created.
            let queue = unsafe { gpu.device.get_device_queue(0, 0) };
            let dest = || {
                let buffer = alloc
                    .create_buffer(bytes, vk::BufferUsageFlags::TRANSFER_DST, HOST)
                    .expect("destination");
                buffer.zero_bytes(0, bytes as usize);
                buffer
            };
            let (vertex, index) = (dest(), dest());
            Self {
                alloc,
                pool,
                queue,
                vertex,
                index,
            }
        }

        fn flush(&self, gpu: &TestGpu, uploads: &mut GeometryUploads) {
            let submit_on = CopySubmit {
                command_pool: self.pool,
                queue: self.queue,
            };
            let dest = GeometryDest {
                vertex: self.vertex.buffer(),
                index: self.index.buffer(),
            };
            uploads.flush(&gpu.device, submit_on, dest).expect("flush");
            // SAFETY: the queue belongs to this device; the wait takes no borrowed state.
            unsafe { gpu.device.queue_wait_idle(self.queue) }.expect("wait");
        }

        fn read(buffer: &PooledBuffer, offset: usize, len: usize) -> Vec<u8> {
            // SAFETY: the buffer is host-visible and coherent, the queue idled after the copies,
            // and `offset + len` stays inside the leased range every caller sized.
            unsafe { std::slice::from_raw_parts(buffer.mapped_ptr().add(offset), len) }.to_vec()
        }

        // Drop every pooled buffer, `uploads`' ring included, then free the
        // allocator's blocks and the pool.
        fn destroy(self, gpu: &TestGpu, uploads: GeometryUploads) {
            drop((uploads, self.vertex, self.index));
            self.alloc.destroy();
            // SAFETY: the queue idled after the last flush, so no command buffer from the pool is
            // in flight.
            unsafe { gpu.device.destroy_command_pool(self.pool, None) };
        }
    }

    fn pattern(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    #[test]
    fn staged_writes_land_at_their_offsets_in_each_target() {
        let Some(gpu) = test_gpu() else {
            return;
        };
        let harness = Harness::new(&gpu, 4096);
        let mut uploads = GeometryUploads::new(2);
        let (verts, idxs) = (pattern(300, 7), pattern(120, 99));
        uploads
            .stage(&harness.alloc, GeometryTarget::Vertex, 1024, &verts)
            .expect("stage vertices");
        uploads
            .stage(&harness.alloc, GeometryTarget::Index, 64, &idxs)
            .expect("stage indices");
        harness.flush(&gpu, &mut uploads);
        assert_eq!(Harness::read(&harness.vertex, 1024, 300), verts);
        assert_eq!(Harness::read(&harness.index, 64, 120), idxs);
        assert!(
            Harness::read(&harness.vertex, 0, 1024)
                .iter()
                .all(|&b| b == 0)
        );
        assert!(uploads.queued.is_empty());
        harness.destroy(&gpu, uploads);
    }

    #[test]
    fn reserving_creates_the_ring_once() {
        let Some(gpu) = test_gpu() else {
            return;
        };
        let harness = Harness::new(&gpu, 4096);
        let mut uploads = GeometryUploads::new(2);
        uploads.reserve(&harness.alloc, 0).expect("reserve");
        let ring = uploads.staging.as_ref().expect("ring").buffer.buffer();
        assert_eq!(
            uploads.staging.as_ref().expect("ring").ring.capacity(),
            STAGING_MIN_CAPACITY
        );
        uploads.reserve(&harness.alloc, 0).expect("reserve again");
        uploads
            .stage(&harness.alloc, GeometryTarget::Vertex, 0, &pattern(64, 1))
            .expect("stage");
        assert_eq!(
            uploads.staging.as_ref().expect("ring").buffer.buffer(),
            ring
        );
        harness.flush(&gpu, &mut uploads);
        harness.destroy(&gpu, uploads);
    }

    #[test]
    fn a_write_queued_before_the_ring_grows_still_lands() {
        let Some(gpu) = test_gpu() else {
            return;
        };
        let big = STAGING_MIN_CAPACITY as usize + 1;
        let harness = Harness::new(&gpu, big as u64 + 4096);
        let mut uploads = GeometryUploads::new(2);
        let small = pattern(256, 3);
        uploads
            .stage(&harness.alloc, GeometryTarget::Vertex, 0, &small)
            .expect("stage small");
        let first_ring = uploads.staging.as_ref().expect("ring").buffer.buffer();
        let large = pattern(big, 11);
        uploads
            .stage(&harness.alloc, GeometryTarget::Vertex, 4096, &large)
            .expect("stage large");
        let grown = uploads.staging.as_ref().expect("ring");
        assert_ne!(
            grown.buffer.buffer(),
            first_ring,
            "the large write grew the ring"
        );
        assert!(grown.ring.capacity() >= big as u64);
        harness.flush(&gpu, &mut uploads);
        assert_eq!(Harness::read(&harness.vertex, 0, 256), small);
        assert_eq!(Harness::read(&harness.vertex, 4096, big), large);
        harness.destroy(&gpu, uploads);
    }

    #[test]
    fn ring_space_and_command_buffers_come_back_after_the_retire_window() {
        let Some(gpu) = test_gpu() else {
            return;
        };
        let harness = Harness::new(&gpu, STAGING_MIN_CAPACITY);
        let mut uploads = GeometryUploads::new(2);
        let chunk = pattern(STAGING_MIN_CAPACITY as usize / 2, 5);
        uploads
            .stage(&harness.alloc, GeometryTarget::Vertex, 0, &chunk)
            .expect("stage");
        harness.flush(&gpu, &mut uploads);
        let ring = uploads.staging.as_ref().expect("ring").buffer.buffer();
        let first_cmd = uploads.command_buffers.pop_due(u64::MAX, 0).expect("held");
        uploads.command_buffers.push(uploads.tick, first_cmd);
        for _ in 0..uploads.depth {
            uploads.begin_frame();
        }
        // Two more halves fit only because the first one's batch retired.
        let half = chunk.len();
        for (offset, seed) in [(0, 6), (half, 7)] {
            uploads
                .stage(
                    &harness.alloc,
                    GeometryTarget::Index,
                    offset as u64,
                    &pattern(half, seed),
                )
                .expect("stage");
        }
        assert_eq!(
            uploads.staging.as_ref().expect("ring").buffer.buffer(),
            ring
        );
        harness.flush(&gpu, &mut uploads);
        assert_eq!(
            uploads.command_buffers.pop_due(u64::MAX, 0),
            Some(first_cmd),
            "the retired command buffer was reused"
        );
        assert_eq!(Harness::read(&harness.index, 0, half), pattern(half, 6));
        assert_eq!(Harness::read(&harness.index, half, half), pattern(half, 7));
        harness.destroy(&gpu, uploads);
    }
}
