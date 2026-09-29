//! Geometry writes into the shared vertex / index buffers without a CPU wait.
//!
//! `stage` copies the bytes into a persistently-mapped UPLOAD ring and queues a
//! region copy. `flush` records every queued copy into one command list and
//! submits it on the direct queue, where queue order puts it ahead of anything
//! submitted later: the frame's own lists, the probe bakes, a one-shot RT
//! build. `draw_frame` flushes right after its frame-slot wait, and `wait_idle`
//! flushes before it waits, so an idle GPU has applied every staged write.
//!
//! The ring's bytes and the command list are held `FRAMES + 1` ticks past the
//! submit that read them, the window the device allocator retires on. A list
//! also holds the ring buffers its copies read until it is reused, so one a
//! larger ring replaced outlives the submits that still name it.

use concinnity_core::render::error::{RenderError, RenderResult};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::error::map_hresult;
use super::texture::transition_barrier;
use crate::suballoc::staging::{
    Recycler, STAGING_ALIGN, StagingRing, grown_capacity, reserved_capacity,
};

// Which shared buffer a staged write lands in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::directx) enum GeometryTarget {
    Vertex,
    Index,
}

impl GeometryTarget {
    // The state the buffer rests in between command lists.
    fn resting_state(self) -> D3D12_RESOURCE_STATES {
        match self {
            GeometryTarget::Vertex => D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            GeometryTarget::Index => D3D12_RESOURCE_STATE_INDEX_BUFFER,
        }
    }

    fn pick<'a>(self, dest: &GeometryDest<'a>) -> &'a ID3D12Resource {
        match self {
            GeometryTarget::Vertex => dest.vertex,
            GeometryTarget::Index => dest.index,
        }
    }
}

// The shared buffers a flush copies into.
pub(in crate::directx) struct GeometryDest<'a> {
    pub vertex: &'a ID3D12Resource,
    pub index: &'a ID3D12Resource,
}

// One queued region copy out of the ring. `src` is the ring buffer the bytes
// were placed in, which a later growth may have replaced.
struct StagedCopy {
    target: GeometryTarget,
    src: PooledBuffer,
    src_offset: u64,
    dst_offset: u64,
    size: u64,
}

struct Staging {
    buffer: PooledBuffer,
    base: *mut u8,
    ring: StagingRing,
}

// A reusable copy list plus the ring buffers its last submit read: a released
// `ID3D12Resource` is freed at once, in flight or not.
struct CopyList {
    allocator: ID3D12CommandAllocator,
    cmd: ID3D12GraphicsCommandList,
    sources: Vec<PooledBuffer>,
}

pub(in crate::directx) struct GeometryUploads {
    staging: Option<Staging>,
    queued: Vec<StagedCopy>,
    lists: Recycler<CopyList>,
    tick: u64,
    depth: u64,
}

impl GeometryUploads {
    pub(in crate::directx) fn new(frames_in_flight: usize) -> Self {
        Self {
            staging: None,
            queued: Vec::new(),
            lists: Recycler::new(),
            tick: 0,
            depth: frames_in_flight as u64 + 1,
        }
    }

    // Advance the frame tick. Called after the frame-slot fence wait, which is
    // what proves a submit `depth` ticks old has retired.
    pub(in crate::directx) fn begin_frame(&mut self) {
        self.tick += 1;
        if let Some(staging) = self.staging.as_mut() {
            staging.ring.reclaim(self.tick);
        }
    }

    // Create the ring at load, sized for a stream that may stage up to
    // `expected` bytes in one frame, so its first burst does not pay for a
    // heap mid-frame. Never shrinks a ring.
    pub(in crate::directx) fn reserve(
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
    pub(in crate::directx) fn stage(
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
        let placed = self
            .staging
            .as_mut()
            .and_then(|s| s.ring.alloc(size, STAGING_ALIGN));
        let src_offset = match placed {
            Some(offset) => offset,
            None => {
                self.grow(alloc, size)?;
                self.staging
                    .as_mut()
                    .and_then(|s| s.ring.alloc(size, STAGING_ALIGN))
                    .ok_or_else(|| {
                        RenderError::Other("geometry staging: grown ring refused a write".into())
                    })?
            }
        };
        let staging = self
            .staging
            .as_ref()
            .ok_or_else(|| RenderError::Other("geometry staging: no ring".into()))?;
        // SAFETY: `base` maps the whole ring buffer, and the ring placed `size`
        // bytes at `src_offset` inside its capacity. The source is a separate
        // borrow, so the ranges cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                staging.base.add(src_offset as usize),
                bytes.len(),
            );
        }
        self.queued.push(StagedCopy {
            target,
            src: staging.buffer.clone(),
            src_offset,
            dst_offset,
            size,
        });
        Ok(())
    }

    // Replace the ring with one that holds `size` more bytes. The old buffer
    // stays alive through the queued copies and the held submits that name it.
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
        let buffer = alloc.alloc_buffer(
            capacity,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut base = std::ptr::null_mut::<std::ffi::c_void>();
        // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local
        // that receives the mapping.
        unsafe { buffer.Map(0, None, Some(&mut base)) }
            .map_err(|e| map_hresult(e.code(), "geometry staging map"))?;
        self.staging = Some(Staging {
            buffer,
            base: base.cast(),
            ring: StagingRing::new(capacity),
        });
        Ok(())
    }

    // Record and submit every queued copy. A no-op when nothing is queued.
    pub(in crate::directx) fn flush(
        &mut self,
        device: &ID3D12Device,
        queue: &ID3D12CommandQueue,
        dest: &GeometryDest<'_>,
    ) -> RenderResult<()> {
        if self.queued.is_empty() {
            return Ok(());
        }
        let mut list = match self.lists.acquire(self.tick) {
            Some(list) => list,
            None => new_copy_list(device)?,
        };
        list.sources.clear();
        let CopyList { allocator, cmd, .. } = &list;
        // SAFETY: the list's last submit retired `depth` ticks ago, so nothing in flight still
        // references what is being reset.
        unsafe {
            allocator
                .Reset()
                .map_err(|e| map_hresult(e.code(), "geometry copy allocator reset"))?;
            cmd.Reset(allocator, None)
                .map_err(|e| map_hresult(e.code(), "geometry copy list reset"))?;
        }
        let targets = touched_targets(&self.queued);
        let to_copy: Vec<_> = targets
            .iter()
            .map(|t| {
                transition_barrier(
                    t.pick(dest),
                    t.resting_state(),
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )
            })
            .collect();
        let to_rest: Vec<_> = targets
            .iter()
            .map(|t| {
                transition_barrier(
                    t.pick(dest),
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    t.resting_state(),
                )
            })
            .collect();
        // SAFETY: the command list is in the recording state, and every resource these commands
        // name is live for the call: the shared buffers through `dest`, each source ring through
        // its queued copy.
        unsafe {
            cmd.ResourceBarrier(&to_copy);
            for copy in &self.queued {
                cmd.CopyBufferRegion(
                    copy.target.pick(dest),
                    copy.dst_offset,
                    &*copy.src,
                    copy.src_offset,
                    copy.size,
                );
            }
            cmd.ResourceBarrier(&to_rest);
            cmd.Close()
                .map_err(|e| map_hresult(e.code(), "geometry copy list close"))?;
        }
        let submit: ID3D12CommandList = cmd
            .cast()
            .map_err(|e| map_hresult(e.code(), "geometry copy list cast"))?;
        // SAFETY: the list is live and closed, and the slice outlives the call.
        unsafe { queue.ExecuteCommandLists(&[Some(submit)]) };

        let retire_at = self.tick + self.depth;
        if let Some(staging) = self.staging.as_mut() {
            staging.ring.seal(retire_at);
        }
        list.sources.extend(self.queued.drain(..).map(|c| c.src));
        list.sources.dedup_by(|a, b| **a == **b);
        self.lists.release(list, retire_at);
        Ok(())
    }
}

// The distinct targets `queued` writes, each once, in first-seen order.
fn touched_targets(queued: &[StagedCopy]) -> Vec<GeometryTarget> {
    let mut targets = Vec::with_capacity(2);
    for copy in queued {
        if !targets.contains(&copy.target) {
            targets.push(copy.target);
        }
    }
    targets
}

fn new_copy_list(device: &ID3D12Device) -> RenderResult<CopyList> {
    let allocator: ID3D12CommandAllocator =
        // SAFETY: the device is live for the call and the new COM object lands in a binding that
        // owns it.
        unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
            .map_err(|e| map_hresult(e.code(), "geometry copy allocator"))?;
    let cmd: ID3D12GraphicsCommandList =
        // SAFETY: the device and allocator are live for the call and the new COM object lands in a
        // binding that owns it.
        unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None) }
            .map_err(|e| map_hresult(e.code(), "geometry copy list"))?;
    // A new list opens recording; close it so every acquire resets alike.
    // SAFETY: the list is live and in the recording state, which is what `Close` requires.
    unsafe { cmd.Close() }.map_err(|e| map_hresult(e.code(), "geometry copy list close"))?;
    Ok(CopyList {
        allocator,
        cmd,
        sources: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directx::texture::one_shot_submit;
    use crate::suballoc::staging::STAGING_MIN_CAPACITY;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;

    // A device, its direct queue and an allocator over them, or None where no
    // D3D12 device exists (CI).
    fn gpu() -> Option<(ID3D12Device, ID3D12CommandQueue, DeviceAllocator)> {
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: the out-parameter is a live local that receives the new COM object.
        unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut device) }.ok()?;
        let device = device?;
        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        };
        // SAFETY: the create descriptor is live for the call, and the new COM object lands in a
        // binding that owns it.
        let queue: ID3D12CommandQueue = unsafe { device.CreateCommandQueue(&queue_desc) }.ok()?;
        let alloc = DeviceAllocator::new(&device, &queue, 3);
        Some((device, queue, alloc))
    }

    // DEFAULT-heap stand-ins for the shared buffers, resting where the real
    // ones rest between command lists.
    fn destinations(alloc: &DeviceAllocator, bytes: u64) -> (PooledBuffer, PooledBuffer) {
        let make = |state| {
            alloc
                .alloc_buffer(bytes, D3D12_HEAP_TYPE_DEFAULT, state)
                .expect("destination places")
        };
        (
            make(GeometryTarget::Vertex.resting_state()),
            make(GeometryTarget::Index.resting_state()),
        )
    }

    fn flush(
        device: &ID3D12Device,
        queue: &ID3D12CommandQueue,
        uploads: &mut GeometryUploads,
        (vertex, index): &(PooledBuffer, PooledBuffer),
    ) {
        let dest = GeometryDest { vertex, index };
        uploads.flush(device, queue, &dest).expect("flush");
    }

    // `len` bytes of `target`'s buffer from `offset`, copied out through a
    // READBACK buffer after everything queued so far has run.
    fn read(
        (device, queue, alloc): &(ID3D12Device, ID3D12CommandQueue, DeviceAllocator),
        target: GeometryTarget,
        buffer: &PooledBuffer,
        offset: u64,
        len: usize,
    ) -> Vec<u8> {
        let readback = alloc
            .alloc_buffer(
                len as u64,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )
            .expect("readback places");
        // SAFETY: the command list is in the recording state, and both resources are live for the
        // call.
        one_shot_submit(device, queue, |cmd| unsafe {
            let rest = target.resting_state();
            cmd.ResourceBarrier(&[transition_barrier(
                buffer,
                rest,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            cmd.CopyBufferRegion(&*readback, 0, &**buffer, offset, len as u64);
            cmd.ResourceBarrier(&[transition_barrier(
                buffer,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                rest,
            )]);
        })
        .expect("readback copy");
        let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
        // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local
        // that receives the mapping.
        unsafe { readback.Map(0, None, Some(&mut ptr)) }.expect("readback maps");
        // SAFETY: the mapping covers the `len`-byte buffer, and the copy into it has completed.
        unsafe { std::slice::from_raw_parts(ptr as *const u8, len).to_vec() }
    }

    fn pattern(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    #[test]
    fn staged_writes_land_at_their_offsets_in_each_target() {
        let Some(gpu) = gpu() else {
            return;
        };
        let (device, queue, alloc) = &gpu;
        let dest = destinations(alloc, 4096);
        let mut uploads = GeometryUploads::new(3);
        let (verts, idxs) = (pattern(300, 7), pattern(120, 99));
        uploads
            .stage(alloc, GeometryTarget::Vertex, 1024, &verts)
            .expect("stage vertices");
        uploads
            .stage(alloc, GeometryTarget::Index, 64, &idxs)
            .expect("stage indices");
        flush(device, queue, &mut uploads, &dest);
        assert!(uploads.queued.is_empty());
        assert_eq!(
            read(&gpu, GeometryTarget::Vertex, &dest.0, 1024, 300),
            verts
        );
        assert_eq!(read(&gpu, GeometryTarget::Index, &dest.1, 64, 120), idxs);
    }

    #[test]
    fn reserving_creates_the_ring_once() {
        let Some(gpu) = gpu() else {
            return;
        };
        let (device, queue, alloc) = &gpu;
        let dest = destinations(alloc, 4096);
        let mut uploads = GeometryUploads::new(3);
        uploads.reserve(alloc, 0).expect("reserve");
        let ring = uploads.staging.as_ref().expect("ring").buffer.clone();
        assert_eq!(
            uploads.staging.as_ref().expect("ring").ring.capacity(),
            STAGING_MIN_CAPACITY
        );
        uploads.reserve(alloc, 0).expect("reserve again");
        uploads
            .stage(alloc, GeometryTarget::Vertex, 0, &pattern(64, 1))
            .expect("stage");
        assert!(*uploads.staging.as_ref().expect("ring").buffer == *ring);
        flush(device, queue, &mut uploads, &dest);
        assert_eq!(
            read(&gpu, GeometryTarget::Vertex, &dest.0, 0, 64),
            pattern(64, 1)
        );
    }

    #[test]
    fn a_write_queued_before_the_ring_grows_still_lands() {
        let Some(gpu) = gpu() else {
            return;
        };
        let (device, queue, alloc) = &gpu;
        let big = STAGING_MIN_CAPACITY as usize + 1;
        let dest = destinations(alloc, big as u64 + 4096);
        let mut uploads = GeometryUploads::new(3);
        let small = pattern(256, 3);
        uploads
            .stage(alloc, GeometryTarget::Vertex, 0, &small)
            .expect("stage small");
        let first_ring = uploads.staging.as_ref().expect("ring").buffer.clone();
        let large = pattern(big, 11);
        uploads
            .stage(alloc, GeometryTarget::Vertex, 4096, &large)
            .expect("stage large");
        let grown = uploads.staging.as_ref().expect("ring");
        assert!(
            *grown.buffer != *first_ring,
            "the large write grew the ring"
        );
        assert!(grown.ring.capacity() >= big as u64);
        drop(first_ring);
        flush(device, queue, &mut uploads, &dest);
        assert_eq!(read(&gpu, GeometryTarget::Vertex, &dest.0, 0, 256), small);
        assert_eq!(
            read(&gpu, GeometryTarget::Vertex, &dest.0, 4096, big),
            large
        );
    }

    #[test]
    fn ring_space_and_copy_lists_come_back_after_the_retire_window() {
        let Some(gpu) = gpu() else {
            return;
        };
        let (device, queue, alloc) = &gpu;
        let dest = destinations(alloc, STAGING_MIN_CAPACITY);
        let mut uploads = GeometryUploads::new(3);
        let half = STAGING_MIN_CAPACITY as usize / 2;
        uploads
            .stage(alloc, GeometryTarget::Vertex, 0, &pattern(half, 5))
            .expect("stage");
        flush(device, queue, &mut uploads, &dest);
        let ring = uploads.staging.as_ref().expect("ring").buffer.clone();
        let first = uploads.lists.acquire(u64::MAX).expect("held");
        let first_cmd = first.cmd.clone();
        assert_eq!(first.sources.len(), 1, "one ring read, held once");
        uploads.lists.release(first, uploads.tick + uploads.depth);
        for _ in 0..uploads.depth {
            uploads.begin_frame();
        }
        // Two more halves fit only because the first one's batch retired.
        for (offset, seed) in [(0, 6), (half, 7)] {
            uploads
                .stage(
                    alloc,
                    GeometryTarget::Index,
                    offset as u64,
                    &pattern(half, seed),
                )
                .expect("stage");
        }
        assert!(*uploads.staging.as_ref().expect("ring").buffer == *ring);
        flush(device, queue, &mut uploads, &dest);
        let reused = uploads.lists.acquire(u64::MAX).expect("held");
        assert!(reused.cmd == first_cmd, "the retired copy list was reused");
        assert_eq!(
            read(&gpu, GeometryTarget::Index, &dest.1, 0, half),
            pattern(half, 6)
        );
        assert_eq!(
            read(&gpu, GeometryTarget::Index, &dest.1, half as u64, half),
            pattern(half, 7)
        );
    }
}
