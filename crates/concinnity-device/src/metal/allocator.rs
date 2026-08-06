// src/metal/allocator.rs
//
// The device-memory allocator persistent Metal resources are placed through.
// Buffers and textures are suballocated out of a few large placement `MTLHeap`s
// instead of each owning its own `newBufferWithLength` / `newTextureWithDescriptor`
// allocation.
//
// Metal has no `maxMemoryAllocationCount`, so unlike Vulkan this is not a
// scalability cliff. What it buys is footprint and overhead: every discrete
// allocation is rounded up to a page and carries driver-side bookkeeping, and a
// world whose texture pool is thousands of small entries pays that per entry.
// Placing them inside a handful of heaps makes the cost track bytes instead.
//
// `block_alloc::BlockAllocator` decides which block and what offset; this file
// is what makes those decisions Metal. The split is the one `transient_pool.rs`
// and the Vulkan `allocator.rs` already use: shared placement policy,
// backend-specific binding.
//
// Blocks are separated into pools by storage mode and CPU cache mode, because
// `newBufferWithLength:options:offset:` requires both to match the heap's, and
// `MTLHeapDescriptor` fixes both for the heap's lifetime. `Managed` and
// `Memoryless` cannot back a heap at all, so they are rejected rather than
// silently mispooled.
//
// Heaps are `Tracked`. Metal tracks hazards at heap granularity: a GPU write to
// any resource on a tracked heap delays reads and writes of every other
// resource on it. That is why only CPU-written, GPU-read-only resources are
// placed here -- with no GPU writes there is never a modification to serialise
// against, so the coarse granularity costs nothing while keeping the automatic
// tracking the rest of the backend assumes. Render targets, cull scratch and
// acceleration structures are GPU-written and stay on their own allocations;
// pooling them would trade a handful of allocations for false dependencies
// between unrelated passes.
//
// Frees are deferred and leases are RAII. Dropping a `PooledBuffer` /
// `PooledTexture` returns its range to the pool tagged with a retire frame
// `frames_in_flight + 1` ticks out, so the bytes are not handed to another
// resource until no in-flight command buffer can still reference them. This
// matters more than it did before pooling: a discrete allocation released early
// is merely undefined, whereas a range released early is reused almost
// immediately, and a placement heap explicitly aliases resources whose ranges
// overlap.
//
// The `Rc` behind the leases is main-thread state. `MtlContext` is `Send` and
// the parallel encoder hands workers a `&MtlContext`, but a worker only ever
// dereferences a pooled resource to bind it, which does not touch the lease;
// every allocation and drop happens on the main thread.
#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Deref;
use std::rc::{Rc, Weak};

use objc2::Message as _;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCPUCacheMode, MTLDevice as _, MTLHazardTrackingMode, MTLHeap, MTLHeapDescriptor,
    MTLHeapType, MTLResourceOptions, MTLStorageMode, MTLTexture, MTLTextureDescriptor,
};

use crate::gfx::block_alloc::{BlockAllocator, Placement};

// Largest block the pool asks for. Big enough that a heavy world holds its
// persistent set in a handful of heaps, small enough that one block is not an
// absurd commitment. A resource too large for one gets a dedicated block sized
// to itself.
const MAX_BLOCK_BYTES: u64 = 64 * 1024 * 1024;

// Size of a pool's first block. Blocks double from here to `MAX_BLOCK_BYTES` as
// a pool fills, so a small world commits megabytes rather than a full-size heap
// for a handful of textures.
const FIRST_BLOCK_BYTES: u64 = 8 * 1024 * 1024;

// The properties that must match for two resources to share a heap. Both are
// fixed at heap creation and `newBufferWithLength:options:offset:` /
// `newTextureWithDescriptor:offset:` reject a mismatch.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct PoolKey {
    storage: MTLStorageMode,
    cache: MTLCPUCacheMode,
}

impl PoolKey {
    // The resource options a placement call must be given for this pool: the
    // storage and cache mode only. Hazard tracking is the heap's, so passing a
    // resource-level mode here would contradict it.
    fn options(self) -> MTLResourceOptions {
        MTLResourceOptions(
            (self.storage.0 << MTL_RESOURCE_STORAGE_MODE_SHIFT)
                | (self.cache.0 << MTL_RESOURCE_CPU_CACHE_MODE_SHIFT),
        )
    }

    // Reject the storage modes `MTLHeapDescriptor` refuses, so a mispooled
    // resource fails at its allocation rather than at heap creation.
    fn check_heap_backed(self) -> Result<Self, String> {
        if self.storage == MTLStorageMode::Shared || self.storage == MTLStorageMode::Private {
            Ok(self)
        } else {
            Err(format!(
                "allocator: storage mode {} cannot back a heap",
                self.storage.0
            ))
        }
    }
}

// Bit positions `MTLResourceOptions` packs the two modes at. Not re-exported by
// objc2-metal, which only generates the combined `MTLResourceOptions` flags.
const MTL_RESOURCE_CPU_CACHE_MODE_SHIFT: usize = 0;
const MTL_RESOURCE_STORAGE_MODE_SHIFT: usize = 4;

// The pool an allocation belongs to, recovered from the options a caller asked
// for.
fn pool_key(options: MTLResourceOptions) -> Result<PoolKey, String> {
    PoolKey {
        storage: MTLStorageMode((options.0 >> MTL_RESOURCE_STORAGE_MODE_SHIFT) & 0xf),
        cache: MTLCPUCacheMode((options.0 >> MTL_RESOURCE_CPU_CACHE_MODE_SHIFT) & 0xf),
    }
    .check_heap_backed()
}

// The blocks of one pool. `placement` names them by index; `heaps` holds the
// matching device allocation, with `None` for a slot whose heap was released.
struct Pool {
    placement: BlockAllocator,
    heaps: Vec<Option<Retained<ProtocolObject<dyn MTLHeap>>>>,
}

impl Pool {
    fn new() -> Self {
        Self {
            placement: BlockAllocator::new(MAX_BLOCK_BYTES),
            heaps: Vec::new(),
        }
    }

    // How large the next block should be to host `size` bytes at `align`. The
    // standard size doubles with the pool's block count up to `MAX_BLOCK_BYTES`,
    // and any request too large for that gets a block of its own size (which
    // `BlockAllocator::add_block` then marks dedicated).
    fn next_block_bytes(&self, size: u64, align: u64) -> u64 {
        let grown = FIRST_BLOCK_BYTES
            .saturating_mul(1 << self.placement.block_count().min(3))
            .min(MAX_BLOCK_BYTES);
        let needed = size.saturating_add(align.max(1).saturating_sub(1));
        needed.max(grown)
    }
}

struct Inner {
    pools: HashMap<PoolKey, Pool>,
    // Monotonic frame tick driving the deferred frees. Not the frame-in-flight
    // index, which wraps.
    frame: u64,
    retire_depth: u64,
}

impl Inner {
    // Return a lease's range to its pool, withheld from reuse until enough
    // frames have ticked that no in-flight command buffer can reference it.
    fn free(&mut self, key: PoolKey, placement: Placement, size: u64) {
        let retire = self.frame + self.retire_depth;
        if let Some(pool) = self.pools.get_mut(&key) {
            pool.placement.free(placement, size, retire);
        }
    }
}

// What the allocator is holding, for diagnostics and the memory ledger.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::metal) struct AllocatorStats {
    // Bytes the device has committed across every heap.
    pub(in crate::metal) reserved_bytes: u64,
    // Bytes live resources occupy. The gap to `reserved_bytes` is alignment
    // padding, fragmentation, and unfilled block tails.
    pub(in crate::metal) in_use_bytes: u64,
    // Live heaps, i.e. how many device allocations back the pooled set.
    pub(in crate::metal) block_count: usize,
}

// A reserved range plus the heap to place into, handed from `reserve` to the
// buffer / texture placement calls.
struct Reservation {
    heap: Retained<ProtocolObject<dyn MTLHeap>>,
    key: PoolKey,
    placement: Placement,
    size: u64,
}

// A pooled range's claim on its block. Returns the range when dropped, which is
// what lets a pooled resource be replaced by plain assignment.
struct Lease {
    owner: Weak<RefCell<Inner>>,
    key: PoolKey,
    placement: Placement,
    size: u64,
}

impl Drop for Lease {
    fn drop(&mut self) {
        if let Some(inner) = self.owner.upgrade() {
            inner.borrow_mut().free(self.key, self.placement, self.size);
        }
    }
}

// A buffer placed inside a pooled heap. Derefs to the `MTLBuffer` so binding
// and `contents()` read exactly as they did on a discrete allocation.
pub(in crate::metal) struct PooledBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    _lease: Lease,
}

impl PooledBuffer {
    // An owned handle to the placed buffer, for a call that needs to hold one
    // past the borrow (a draw record, a geometry descriptor). It keeps the
    // buffer and its heap alive but does not extend the lease, so a holder that
    // outlives this `PooledBuffer` would be reading bytes the pool has since
    // handed to another resource: keep such handles within the frame.
    pub(in crate::metal) fn retained(&self) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        self.buffer.clone()
    }
}

impl Deref for PooledBuffer {
    type Target = ProtocolObject<dyn MTLBuffer>;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl AsRef<ProtocolObject<dyn MTLBuffer>> for PooledBuffer {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }
}

// A texture placed inside a pooled heap. Derefs to the `MTLTexture` so binding
// and `replaceRegion` read exactly as they did on a discrete allocation.
pub(in crate::metal) struct PooledTexture {
    texture: Retained<ProtocolObject<dyn MTLTexture>>,
    _lease: Lease,
}

impl Deref for PooledTexture {
    type Target = ProtocolObject<dyn MTLTexture>;

    fn deref(&self) -> &Self::Target {
        &self.texture
    }
}

impl AsRef<ProtocolObject<dyn MTLTexture>> for PooledTexture {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLTexture> {
        &self.texture
    }
}

// The allocator behind every pooled buffer and texture. See the module comment.
pub(in crate::metal) struct DeviceAllocator {
    device: Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
    inner: Rc<RefCell<Inner>>,
}

impl DeviceAllocator {
    pub(in crate::metal) fn new(
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        frames_in_flight: usize,
    ) -> Self {
        Self {
            device: device.retain(),
            inner: Rc::new(RefCell::new(Inner {
                pools: HashMap::new(),
                frame: 0,
                // One tick beyond the frames in flight, matching the streamed
                // upload retire discipline: a resource replaced between frames
                // must outlive the submission that was already in flight.
                retire_depth: frames_in_flight as u64 + 1,
            })),
        }
    }

    // The device the pooled heaps are created on. Lets a caller that already
    // holds an allocator build the resources that stay on their own allocations
    // (render targets, pipelines) without carrying a second handle.
    pub(in crate::metal) fn device(&self) -> &ProtocolObject<dyn objc2_metal::MTLDevice> {
        &self.device
    }

    // Place a `len`-byte buffer. `options` selects the pool; its hazard-tracking
    // bits are ignored, since a heap's resources all inherit the heap's mode.
    pub(in crate::metal) fn alloc_buffer(
        &self,
        len: usize,
        options: MTLResourceOptions,
    ) -> Result<PooledBuffer, String> {
        let key = pool_key(options)?;
        let len = len.max(1);
        let sizing = self
            .device
            .heapBufferSizeAndAlignWithLength_options(len, key.options());
        let reservation = self.reserve(key, sizing.size as u64, sizing.align as u64)?;
        // SAFETY: the offset came from the block allocator, which places within
        // the block's bounds at a multiple of the alignment Metal reported for
        // exactly these buffer parameters, and never overlaps a live resource.
        let placed = unsafe {
            reservation.heap.newBufferWithLength_options_offset(
                len,
                key.options(),
                reservation.placement.offset as usize,
            )
        };
        match placed {
            Some(buffer) => Ok(PooledBuffer {
                buffer,
                _lease: self.lease(reservation),
            }),
            None => {
                self.release(reservation);
                Err(format!("allocator: failed to place {len}-byte buffer"))
            }
        }
    }

    // Place a buffer holding `src`. The heap replacement for
    // `newBufferWithBytes`, which has no placement form; the copy needs
    // CPU-visible storage, so the pool must be a shared one.
    pub(in crate::metal) fn alloc_buffer_with_bytes(
        &self,
        src: &[u8],
        options: MTLResourceOptions,
    ) -> Result<PooledBuffer, String> {
        if pool_key(options)?.storage != MTLStorageMode::Shared {
            return Err("allocator: initialised buffers need shared storage".to_string());
        }
        let buffer = self.alloc_buffer(src.len(), options)?;
        super::context::write_buffer_region(&buffer, 0, src)?;
        Ok(buffer)
    }

    // Place a texture built from `desc`. The descriptor's storage and cache mode
    // select the pool; leave its hazard-tracking mode at the default so it
    // inherits the heap's.
    pub(in crate::metal) fn alloc_texture(
        &self,
        desc: &MTLTextureDescriptor,
    ) -> Result<PooledTexture, String> {
        let key = PoolKey {
            storage: desc.storageMode(),
            cache: desc.cpuCacheMode(),
        }
        .check_heap_backed()?;
        let sizing = self.device.heapTextureSizeAndAlignWithDescriptor(desc);
        let reservation = self.reserve(key, sizing.size as u64, sizing.align as u64)?;
        // SAFETY: as `alloc_buffer`, with the size and alignment Metal reported
        // for this descriptor.
        let placed = unsafe {
            reservation
                .heap
                .newTextureWithDescriptor_offset(desc, reservation.placement.offset as usize)
        };
        match placed {
            Some(texture) => Ok(PooledTexture {
                texture,
                _lease: self.lease(reservation),
            }),
            None => {
                self.release(reservation);
                Err(format!(
                    "allocator: failed to place {}-byte texture",
                    sizing.size
                ))
            }
        }
    }

    // Advance the frame tick, make retired frees placeable again, and drop any
    // heap that now holds nothing. A heap a still-live resource was placed on
    // outlives this: Metal reference counts, and a placed resource holds its
    // heap.
    pub(in crate::metal) fn begin_frame(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.frame += 1;
        let frame = inner.frame;
        for pool in inner.pools.values_mut() {
            pool.placement.reclaim(frame);
            for index in pool.placement.take_empty_blocks() {
                if let Some(slot) = pool.heaps.get_mut(index) {
                    *slot = None;
                }
            }
        }
    }

    pub(in crate::metal) fn stats(&self) -> AllocatorStats {
        let inner = self.inner.borrow();
        let mut stats = AllocatorStats::default();
        for pool in inner.pools.values() {
            stats.reserved_bytes += pool.placement.reserved_bytes();
            stats.in_use_bytes += pool.placement.in_use_bytes();
            stats.block_count += pool.placement.block_count();
        }
        stats
    }

    // Reserve `size` bytes at `align` in `key`'s pool, opening a heap when no
    // existing block can host them.
    fn reserve(&self, key: PoolKey, size: u64, align: u64) -> Result<Reservation, String> {
        let mut inner = self.inner.borrow_mut();
        let pool = inner.pools.entry(key).or_insert_with(Pool::new);

        if let Some(placement) = pool.placement.alloc(size, align) {
            let heap = pool.heaps[placement.block]
                .clone()
                .ok_or("allocator: placement named a released heap")?;
            return Ok(Reservation {
                heap,
                key,
                placement,
                size,
            });
        }

        let block_bytes = pool.next_block_bytes(size, align);
        let heap = new_heap(&self.device, key, block_bytes)?;
        let index = pool.placement.add_block(block_bytes);
        if index == pool.heaps.len() {
            pool.heaps.push(Some(heap.clone()));
        } else {
            pool.heaps[index] = Some(heap.clone());
        }
        let placement = pool
            .placement
            .alloc_in(index, size, align)
            .ok_or("allocator: a block sized for a request failed to host it")?;
        Ok(Reservation {
            heap,
            key,
            placement,
            size,
        })
    }

    fn lease(&self, reservation: Reservation) -> Lease {
        Lease {
            owner: Rc::downgrade(&self.inner),
            key: reservation.key,
            placement: reservation.placement,
            size: reservation.size,
        }
    }

    // Hand a reservation back when the placement call it was made for failed.
    // Retired immediately: no command buffer ever saw a resource there.
    fn release(&self, reservation: Reservation) {
        let mut inner = self.inner.borrow_mut();
        if let Some(pool) = inner.pools.get_mut(&reservation.key) {
            pool.placement
                .free(reservation.placement, reservation.size, 0);
        }
    }
}

// One placement heap of `size` bytes for `key`'s pool. Hazard tracking is set
// explicitly because heaps treat `Default` as `Untracked`, which would drop the
// automatic tracking the rest of the backend relies on.
fn new_heap(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    key: PoolKey,
    size: u64,
) -> Result<Retained<ProtocolObject<dyn MTLHeap>>, String> {
    let desc = MTLHeapDescriptor::new();
    desc.setType(MTLHeapType::Placement);
    desc.setStorageMode(key.storage);
    desc.setCpuCacheMode(key.cache);
    desc.setHazardTrackingMode(MTLHazardTrackingMode::Tracked);
    desc.setSize(size.max(1) as usize);
    device
        .newHeapWithDescriptor(&desc)
        .ok_or_else(|| format!("allocator: failed to create a {size}-byte heap"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::{MTLPixelFormat, MTLTextureUsage};

    fn device() -> Option<Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>> {
        objc2_metal::MTLCreateSystemDefaultDevice()
    }

    fn shared_texture_desc(width: usize) -> Retained<MTLTextureDescriptor> {
        let desc = MTLTextureDescriptor::new();
        // SAFETY: plain descriptor property setters, all values in range.
        unsafe {
            desc.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
            desc.setWidth(width);
            desc.setHeight(width);
            desc.setUsage(MTLTextureUsage::ShaderRead);
            desc.setStorageMode(MTLStorageMode::Shared);
        }
        desc
    }

    #[test]
    fn pool_key_round_trips_through_resource_options() {
        // The pool a caller lands in is decoded from the options it passed, so
        // the shift positions have to match Metal's packing exactly.
        let key = pool_key(MTLResourceOptions::StorageModeShared).expect("shared is heap-backed");
        assert_eq!(key.storage, MTLStorageMode::Shared);
        assert_eq!(key.cache, MTLCPUCacheMode::DefaultCache);
        assert_eq!(key.options(), MTLResourceOptions::StorageModeShared);

        let combined =
            MTLResourceOptions::StorageModePrivate | MTLResourceOptions::CPUCacheModeWriteCombined;
        let key = pool_key(combined).expect("private is heap-backed");
        assert_eq!(key.storage, MTLStorageMode::Private);
        assert_eq!(key.cache, MTLCPUCacheMode::WriteCombined);
        assert_eq!(key.options(), combined);
    }

    #[test]
    fn hazard_tracking_bits_do_not_split_a_pool() {
        // Hazard tracking is the heap's, so two requests differing only there
        // must share one pool rather than opening a second heap.
        let tracked = pool_key(
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked,
        )
        .expect("shared is heap-backed");
        let untracked = pool_key(
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked,
        )
        .expect("shared is heap-backed");
        assert_eq!(tracked, untracked);
        assert_eq!(tracked.options(), MTLResourceOptions::StorageModeShared);
    }

    #[test]
    fn storage_modes_a_heap_cannot_back_are_rejected() {
        // `MTLHeapDescriptor` refuses both, so they have to fail at the
        // allocation rather than at heap creation.
        assert!(pool_key(MTLResourceOptions::StorageModeManaged).is_err());
        assert!(pool_key(MTLResourceOptions::StorageModeMemoryless).is_err());
    }

    #[test]
    fn blocks_grow_from_the_first_size_up_to_the_cap() {
        // A small world should not commit a full-size heap for a few textures,
        // so the first blocks are small and the size doubles as the pool fills.
        let mut pool = Pool::new();
        let mut sizes = Vec::new();
        for _ in 0..5 {
            let bytes = pool.next_block_bytes(1024, 256);
            sizes.push(bytes);
            pool.placement.add_block(bytes);
        }
        assert_eq!(
            sizes,
            vec![
                FIRST_BLOCK_BYTES,
                FIRST_BLOCK_BYTES * 2,
                FIRST_BLOCK_BYTES * 4,
                MAX_BLOCK_BYTES,
                MAX_BLOCK_BYTES,
            ]
        );
    }

    #[test]
    fn an_oversized_request_sizes_its_own_block() {
        // Larger than the cap, so the block is sized to the request (and the
        // block allocator marks it dedicated rather than sharing the remainder).
        let pool = Pool::new();
        let huge = MAX_BLOCK_BYTES * 3;
        assert_eq!(pool.next_block_bytes(huge, 256), huge + 255);
    }

    #[test]
    fn many_buffers_share_few_heaps() {
        let Some(device) = device() else {
            return;
        };
        let alloc = DeviceAllocator::new(&device, 2);
        let buffers: Vec<PooledBuffer> = (0..512)
            .map(|_| {
                alloc
                    .alloc_buffer(4096, MTLResourceOptions::StorageModeShared)
                    .expect("shared buffer places")
            })
            .collect();
        let stats = alloc.stats();
        // 512 x 4 KiB is well under one block, so the whole set costs a single
        // device allocation rather than 512.
        assert_eq!(stats.block_count, 1, "{stats:?}");
        assert!(stats.in_use_bytes >= 512 * 4096, "{stats:?}");
        assert!(stats.reserved_bytes >= stats.in_use_bytes, "{stats:?}");
        drop(buffers);
    }

    #[test]
    fn placed_buffers_get_distinct_non_overlapping_storage() {
        let Some(device) = device() else {
            return;
        };
        let alloc = DeviceAllocator::new(&device, 2);
        let a = alloc
            .alloc_buffer_with_bytes(&[0xAAu8; 256], MTLResourceOptions::StorageModeShared)
            .expect("shared buffer places");
        let b = alloc
            .alloc_buffer_with_bytes(&[0x55u8; 256], MTLResourceOptions::StorageModeShared)
            .expect("shared buffer places");
        // Aliased ranges are the failure mode a placement heap makes possible,
        // so check the contents rather than just the offsets.
        // SAFETY: both are shared storage and at least 256 bytes long.
        let (a_bytes, b_bytes) = unsafe {
            (
                std::slice::from_raw_parts(a.contents().as_ptr() as *const u8, 256),
                std::slice::from_raw_parts(b.contents().as_ptr() as *const u8, 256),
            )
        };
        assert!(a_bytes.iter().all(|&x| x == 0xAA));
        assert!(b_bytes.iter().all(|&x| x == 0x55));
    }

    #[test]
    fn a_dropped_lease_is_withheld_until_its_retire_frame() {
        let Some(device) = device() else {
            return;
        };
        let alloc = DeviceAllocator::new(&device, 2);
        let first = alloc
            .alloc_buffer(4096, MTLResourceOptions::StorageModeShared)
            .expect("shared buffer places");
        let stats = alloc.stats();
        assert_eq!(stats.block_count, 1);
        drop(first);
        // The bytes stop counting as in use at once, but a frame that may still
        // reference them has to retire before they are placed again.
        assert_eq!(alloc.stats().in_use_bytes, 0);
        for _ in 0..4 {
            alloc.begin_frame();
        }
        assert_eq!(alloc.stats().block_count, 0, "emptied heap is released");
        assert_eq!(alloc.stats().reserved_bytes, 0);
    }

    #[test]
    fn textures_and_buffers_share_a_pool_when_their_modes_match() {
        let Some(device) = device() else {
            return;
        };
        let alloc = DeviceAllocator::new(&device, 2);
        let _buffer = alloc
            .alloc_buffer(4096, MTLResourceOptions::StorageModeShared)
            .expect("shared buffer places");
        let _texture = alloc
            .alloc_texture(&shared_texture_desc(64))
            .expect("shared texture places");
        // Metal has no buffer/image granularity rule to separate them, so both
        // land in the one shared pool.
        assert_eq!(alloc.stats().block_count, 1);
    }

    #[test]
    fn a_texture_larger_than_the_cap_gets_its_own_heap() {
        let Some(device) = device() else {
            return;
        };
        let alloc = DeviceAllocator::new(&device, 2);
        // 4096 x 4096 RGBA8 is 64 MiB, past the standard block size.
        let big = alloc
            .alloc_texture(&shared_texture_desc(4096))
            .expect("oversized texture places");
        let small = alloc
            .alloc_texture(&shared_texture_desc(64))
            .expect("small texture places");
        // The dedicated block is never shared, so the small texture opens a
        // standard block instead of stranding the remainder of the huge one.
        let stats = alloc.stats();
        assert_eq!(stats.block_count, 2, "{stats:?}");
        drop((big, small));
    }
}
