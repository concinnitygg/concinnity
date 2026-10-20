//! The device-memory allocator every persistent Vulkan resource is created
//! through. Buffers and images are suballocated out of a few large
//! `VkDeviceMemory` blocks instead of each owning one.
//!
//! `vkAllocateMemory` is a scarce call. `maxMemoryAllocationCount` is commonly
//! 4096 on desktop drivers and lower elsewhere, and it is a hard cap: past it
//! allocation fails outright, however much memory is free. One allocation per
//! resource spends that budget on resource count, so a world within the byte
//! budget can still fail to load. Blocks make the cap a function of bytes
//! rather than of how many things the world holds.
//!
//! `block_alloc::BlockAllocator` decides which block and what offset; this file
//! is what makes those decisions Vulkan. The split is the same one the transient
//! image pool uses: shared placement policy, backend-specific binding.
//!
//! Blocks are separated into pools by three properties, because each is fixed
//! for the lifetime of an allocation and cannot be mixed within one:
//!
//!   memory type      what `vkAllocateMemory` was given; a resource can only
//!                    bind to memory of a type its requirements permit
//!   tiling class     Vulkan requires linear and optimal-tiling resources
//!                    sharing an allocation to be separated by
//!                    `bufferImageGranularity`. Separate pools remove the
//!                    constraint rather than paying to honor it.
//!   device address   `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT` is a property of
//!                    the allocation, not the buffer, so ray-tracing buffers
//!                    that need `vkGetBufferDeviceAddress` need their own blocks
//!
//! Host-visible blocks are mapped once, at block creation, and stay mapped.
//! Vulkan forbids mapping one allocation twice, so a per-resource map is not
//! even available once resources share a block; each resource carries a pointer
//! to its own bytes at its own offset into the block's mapping. Every
//! host-visible memory type this backend allocates is coherent, so the pointers
//! need no flush discipline.
//!
//! Lifetime is RAII, like the Metal and D3D12 pools. Dropping a `PooledBuffer` /
//! `PooledImage` returns its range to the pool and queues its Vulkan handles for
//! destruction, both withheld until `frames_in_flight + 1` frame ticks have
//! passed, which covers any command buffer that could still reference them. A
//! caller therefore never destroys, frees, or reasons about GPU progress on
//! teardown; assignment and drop are the whole discipline. The leases are
//! cloneable so a resource can be held by a ring slot and the live field that
//! reads it at once.
//!
//! A caller that has idled the device can open an `IdleScope`, inside which a
//! dropped resource's range is placeable at once. A rebuild that replaces its
//! targets then refills the ranges the old ones held instead of opening blocks
//! beside them and stranding the old ranges as holes.
//!
//! The `Rc` behind the leases is main-thread state, on the same invariant as
//! `unsafe impl Send for VkContext`: the context migrates between threads but is
//! only ever used from one at a time, and workers given `&VkContext` only read
//! handles, never drop or allocate.

use ash::{Device, vk};
use concinnity_core::math::SaturatingFrom;
use concinnity_core::render::backend::error;
use concinnity_core::render::buffers::retire_pool::RetirePool;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

use concinnity_core::render::buffers::block_alloc::Placement;
use concinnity_core::render::buffers::block_pool::{BlockGrowth, BlockPool, ReserveError};

const GROWTH: BlockGrowth = BlockGrowth::STANDARD;

// Whether a resource is laid out linearly (buffers, LINEAR-tiled images) or in
// an implementation-defined tiling (OPTIMAL images). Kept apart so
// `bufferImageGranularity` never applies.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) enum ResourceKind {
    Linear,
    Optimal,
}

// The properties that must match for two resources to share a block.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct PoolKey {
    memory_type: u32,
    kind: ResourceKind,
    device_address: bool,
}

// One block: the device allocation plus its persistent mapping, if any.
#[derive(Clone, Copy)]
struct Block {
    memory: vk::DeviceMemory,
    size: u64,
    // Base of the block's mapping, or null when the memory type is not
    // host-visible.
    mapped: *mut u8,
}

// The blocks of one pool, by the index its placement names them.
type Pool = BlockPool<Block>;

// The Vulkan object a lease owns, destroyed when the lease retires.
enum PooledHandle {
    Buffer(vk::Buffer),
    Image(vk::Image),
}

// A dropped resource's handles, awaiting destruction once no in-flight command
// buffer can reference them.
struct Retired {
    handle: PooledHandle,
    views: Vec<vk::ImageView>,
}

struct Inner {
    pools: HashMap<PoolKey, Pool>,
    retired: RetirePool<Retired>,
    // Monotonic frame tick driving the deferred frees. Not the frame-in-flight
    // index, which wraps.
    frame: u64,
    retire_depth: u64,
    // Open `IdleScope`s. Nonzero means no command buffer is in flight.
    idle_scopes: u32,
}

impl Inner {
    // Return a lease's range to its pool and queue its handles, both withheld
    // until enough frames have ticked that no in-flight command buffer can
    // reference them. Inside an `IdleScope` the range is placeable at once; the
    // handles still wait, for the scope's close.
    fn release(&mut self, lease: &mut Lease) {
        let idle = self.idle_scopes > 0;
        let retire = if idle {
            self.frame
        } else {
            self.frame + self.retire_depth
        };
        if let Some(pool) = self.pools.get_mut(&lease.key) {
            pool.free(lease.placement, lease.size, retire);
            if idle {
                pool.reclaim(self.frame);
            }
        }
        self.retired.push(
            self.frame,
            Retired {
                handle: std::mem::replace(
                    &mut lease.handle,
                    PooledHandle::Buffer(vk::Buffer::null()),
                ),
                views: std::mem::take(&mut *lease.views.borrow_mut()),
            },
        );
    }
}

// A pooled resource's claim on its range and its Vulkan handles. The last
// holder dropping is what releases both, so a pooled resource is replaced by
// plain assignment. Shared, because a resource can be held by a ring slot and
// the live field that reads it at once; a cloned handle keeps the resource
// alive rather than outliving it.
struct Lease {
    owner: Weak<RefCell<Inner>>,
    key: PoolKey,
    placement: Placement,
    size: u64,
    handle: PooledHandle,
    // Views onto a pooled image, destroyed with it (views first). Interior
    // mutability because views are created after the image exists.
    views: RefCell<Vec<vk::ImageView>>,
}

impl Drop for Lease {
    fn drop(&mut self) {
        // If the allocator is already gone the device is being torn down and
        // `destroy` has run; there is nothing left to return the range to.
        if let Some(inner) = self.owner.upgrade() {
            inner.borrow_mut().release(self);
        }
    }
}

// A buffer suballocated from a pooled block. Owns the `VkBuffer` and its bytes;
// dropping the last clone queues both for destruction.
#[derive(Clone)]
pub(super) struct PooledBuffer {
    buffer: vk::Buffer,
    mapped: *mut u8,
    lease: Option<Rc<Lease>>,
}

impl PooledBuffer {
    pub(super) fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    // A pointer to this buffer's own bytes, or null when its memory type is not
    // host-visible. The block is mapped for its whole life, so this neither
    // maps nor needs a matching unmap.
    pub(super) fn mapped_ptr(&self) -> *mut u8 {
        self.mapped
    }

    // Byte length of the leased range backing this buffer: its aligned
    // allocation size, at least the size requested at creation. 0 for `null()`.
    pub(super) fn byte_len(&self) -> usize {
        self.lease
            .as_ref()
            .map_or(0, |l| usize::saturating_from(l.size))
    }

    // Records `first..first + len` of this buffer's mapping viewed as `T`s, for a
    // pack to write in place. `None` when the buffer has no host mapping, the
    // range runs past the leased bytes, or the mapping is misaligned for `T`.
    //
    // # Safety
    //
    // While the slice lives, nothing else may read or write those bytes: the
    // GPU must be done with them (a frame-in-flight copy whose fence has
    // signaled, or a bake buffer not yet submitted), and no other slice or
    // `write_*` call may touch the range.
    #[expect(
        clippy::mut_from_ref,
        reason = "the bytes are the device's mapping, lent out under the caller's contract"
    )]
    pub(super) unsafe fn records_mut<T: bytemuck::AnyBitPattern + bytemuck::NoUninit>(
        &self,
        first: usize,
        len: usize,
    ) -> Option<&mut [T]> {
        let stride = size_of::<T>();
        let end = first.checked_add(len)?.checked_mul(stride)?;
        if self.mapped.is_null() || end > self.byte_len() {
            return None;
        }
        // SAFETY: `first * stride <= end <= byte_len()`, and the lease's mapping
        // covers `byte_len()` bytes.
        let base = unsafe { self.mapped.add(first * stride) }.cast::<T>();
        if !base.is_aligned() {
            return None;
        }
        // SAFETY: the checks above proved `len` aligned `T`s fit the live mapping
        // from `base`; any bit pattern is a valid `T`; the caller guarantees no
        // other access to the range while the slice lives.
        Some(unsafe { std::slice::from_raw_parts_mut(base, len) })
    }

    // Assert the buffer is host-visible and `[offset, offset + len)` is within
    // the leased range, then hand back the write cursor.
    fn write_dst(&self, offset: usize, len: usize) -> *mut u8 {
        assert!(
            !self.mapped.is_null(),
            "write into a buffer with no host mapping"
        );
        let cap = self.byte_len();
        assert!(
            offset.checked_add(len).is_some_and(|end| end <= cap),
            "buffer write [{offset}, {}) exceeds mapped length {cap}",
            offset.saturating_add(len),
        );
        // SAFETY: `offset <= cap` per the assert above, and the lease's mapping
        // covers `cap` bytes, so the offset pointer stays in bounds.
        unsafe { self.mapped.add(offset) }
    }

    // Copy `bytes` into the buffer at `offset` bytes. Panics if the buffer is
    // not host-visible or the write does not fit the leased range.
    pub(super) fn write_bytes(&self, offset: usize, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let dst = self.write_dst(offset, bytes.len());
        // SAFETY: `write_dst` proved the destination holds `bytes.len()` bytes
        // of this buffer's live mapping; `bytes` is a separate borrow, so the
        // ranges cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        }
    }

    // Copy `value`'s bytes into the buffer at `offset` bytes. Panics if the
    // buffer is not host-visible or the write does not fit the leased range.
    pub(super) fn write_val<T: Copy>(&self, offset: usize, value: &T) {
        let len = size_of::<T>();
        if len == 0 {
            return;
        }
        let dst = self.write_dst(offset, len);
        // SAFETY: `write_dst` proved the destination holds `size_of::<T>()`
        // bytes of this buffer's live mapping, and `value` is a live borrow of
        // exactly that many bytes (padding is copied, never inspected). The
        // borrow is separate from the mapping, so the ranges cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(std::ptr::from_ref::<T>(value).cast::<u8>(), dst, len);
        }
    }

    // Copy `values`' bytes into the buffer at `offset` bytes. Panics if the
    // buffer is not host-visible or the write does not fit the leased range.
    pub(super) fn write_slice<T: Copy>(&self, offset: usize, values: &[T]) {
        let len = size_of_val(values);
        if len == 0 {
            return;
        }
        let dst = self.write_dst(offset, len);
        // SAFETY: `write_dst` proved the destination holds `size_of_val(values)`
        // bytes of this buffer's live mapping, and `values` is a live borrow of
        // exactly that many bytes (padding is copied, never inspected). The
        // borrow is separate from the mapping, so the ranges cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr().cast::<u8>(), dst, len);
        }
    }

    // Zero `len` bytes of the buffer at `offset` bytes. Panics if the buffer
    // is not host-visible or the range does not fit the leased range.
    pub(super) fn zero_bytes(&self, offset: usize, len: usize) {
        if len == 0 {
            return;
        }
        let dst = self.write_dst(offset, len);
        // SAFETY: `write_dst` proved the destination holds `len` bytes of this
        // buffer's live mapping.
        unsafe {
            std::ptr::write_bytes(dst, 0, len);
        }
    }

    // A placeholder naming no buffer, for a slot filled before its real
    // resource exists. Dropping it is a no-op.
    pub(super) fn null() -> Self {
        Self {
            buffer: vk::Buffer::null(),
            mapped: std::ptr::null_mut(),
            lease: None,
        }
    }

    pub(super) fn is_null(&self) -> bool {
        self.buffer == vk::Buffer::null()
    }
}

// An image suballocated from a pooled block. Owns the `VkImage`, its bytes, and
// any views attached to it; dropping the last clone queues them all for
// destruction, views first.
#[derive(Clone)]
pub(super) struct PooledImage {
    image: vk::Image,
    lease: Option<Rc<Lease>>,
}

impl PooledImage {
    pub(super) fn image(&self) -> vk::Image {
        self.image
    }

    // Tie `view` to this image's lifetime: it is destroyed just before the
    // image, at the same retirement.
    pub(super) fn attach_view(&self, view: vk::ImageView) {
        debug_assert!(self.lease.is_some(), "attach_view on a null PooledImage");
        if let Some(lease) = &self.lease {
            lease.views.borrow_mut().push(view);
        }
    }

    // A placeholder naming no image, for a slot filled before its real resource
    // exists. Dropping it is a no-op.
    pub(super) fn null() -> Self {
        Self {
            image: vk::Image::null(),
            lease: None,
        }
    }

    // Exercised by the unit tests; no live caller branches on an image's null
    // state yet.
    #[cfg(test)]
    pub(super) fn is_null(&self) -> bool {
        self.image == vk::Image::null()
    }
}

// What the allocator is holding, for the memory ledger and diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct AllocatorStats {
    // Bytes the device has committed across every block.
    pub(super) reserved_bytes: u64,
    // Bytes live resources occupy. The gap to `reserved_bytes` is alignment
    // padding, fragmentation, and unfilled block tails.
    pub(super) in_use_bytes: u64,
    // Live `vkAllocateMemory` results, i.e. what counts against
    // `maxMemoryAllocationCount`.
    pub(super) block_count: usize,
}

impl std::fmt::Display for AllocatorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} block(s), {} KiB reserved for {} KiB of resources",
            self.block_count,
            self.reserved_bytes / 1024,
            self.in_use_bytes / 1024,
        )
    }
}

// A reserved range plus the block it lives in, handed from `reserve` to the
// bind calls.
struct Reservation {
    memory: vk::DeviceMemory,
    // Base of the block's mapping, or null; the resource's own pointer is this
    // plus the placement offset.
    mapped: *mut u8,
    key: PoolKey,
    placement: Placement,
    size: u64,
}

// The allocator behind every pooled buffer and image. See the module comment.
// Clone is handle semantics: clones share one pool, so a live editor reload
// hands the outgoing context's allocator to its successor and the rebuilt
// world places into the blocks the old world's leases release.
#[derive(Clone)]
pub(super) struct DeviceAllocator {
    device: Device,
    inner: Rc<RefCell<Inner>>,
    memory_props: vk::PhysicalDeviceMemoryProperties,
    max_allocations: u32,
}

impl DeviceAllocator {
    pub(super) fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &Device,
        frames_in_flight: usize,
    ) -> Self {
        // SAFETY: `physical_device` was enumerated from this live instance; the query only reads.
        let memory_props =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        // SAFETY: same live instance and physical device as above; a read-only property query.
        let device_props = unsafe { instance.get_physical_device_properties(physical_device) };
        let max_allocations = device_props.limits.max_memory_allocation_count;
        Self {
            device: device.clone(),
            inner: Rc::new(RefCell::new(Inner {
                pools: HashMap::new(),
                retired: RetirePool::new(),
                frame: 0,
                idle_scopes: 0,
                // One tick beyond the frames in flight, matching the streamed
                // upload retire discipline: a resource replaced between frames
                // must outlive the submission that was already in flight.
                retire_depth: frames_in_flight as u64 + 1,
            })),
            memory_props,
            max_allocations,
        }
    }

    // Create a `size`-byte buffer with `usage`, placed in memory satisfying
    // `props`. A buffer created with SHADER_DEVICE_ADDRESS usage (the
    // ray-tracing acceleration-structure buffers and their build inputs) is
    // placed in a block allocated with VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    // or `get_buffer_device_address` would be invalid.
    pub(super) fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        props: vk::MemoryPropertyFlags,
    ) -> error::RenderResult<PooledBuffer> {
        let info = vk::BufferCreateInfo::default()
            .size(size.max(1))
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: the create-info and every slice it borrows are live for the call, and each handle
        // it names belongs to this device.
        let buffer = unsafe { self.device.create_buffer(&info, None) }
            .map_err(|e| super::error::map_vk_result(e, "create_buffer"))?;
        // SAFETY: a property query on a live handle; it only reads.
        let reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let device_address = usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
        let reservation = match self.reserve(reqs, props, ResourceKind::Linear, device_address) {
            Ok(r) => r,
            Err(e) => {
                // SAFETY: the handle was created from this device moments ago and never submitted,
                // so this cleanup is its only remaining use.
                unsafe { self.device.destroy_buffer(buffer, None) };
                return Err(e);
            }
        };
        // SAFETY: the resource and the memory were both created from this device, the reservation's
        // offset satisfies the alignment its memory requirements reported, and nothing is bound to
        // the resource yet.
        if let Err(e) = unsafe {
            self.device
                .bind_buffer_memory(buffer, reservation.memory, reservation.placement.offset)
        } {
            // SAFETY: the handle was created from this device moments ago and never submitted, so
            // this cleanup is its only remaining use.
            unsafe { self.device.destroy_buffer(buffer, None) };
            self.release(reservation);
            return Err(super::error::map_vk_result(e, "bind_buffer_memory"));
        }
        let mapped = resource_ptr(&reservation);
        Ok(PooledBuffer {
            buffer,
            mapped,
            lease: Some(Rc::new(
                self.lease(reservation, PooledHandle::Buffer(buffer)),
            )),
        })
    }

    // Create an image from `info`, placed in memory satisfying `props`. The
    // tiling in `info` picks the pool: a LINEAR image shares blocks with
    // buffers, an OPTIMAL one never does.
    pub(super) fn create_image(
        &self,
        info: &vk::ImageCreateInfo<'_>,
        props: vk::MemoryPropertyFlags,
    ) -> error::RenderResult<PooledImage> {
        // SAFETY: the create-info and every slice it borrows are live for the call, and each handle
        // it names belongs to this device.
        let image = unsafe { self.device.create_image(info, None) }
            .map_err(|e| super::error::map_vk_result(e, "create_image"))?;
        // SAFETY: a property query on a live handle; it only reads.
        let reqs = unsafe { self.device.get_image_memory_requirements(image) };
        let kind = if info.tiling == vk::ImageTiling::LINEAR {
            ResourceKind::Linear
        } else {
            ResourceKind::Optimal
        };
        let reservation = match self.reserve(reqs, props, kind, false) {
            Ok(r) => r,
            Err(e) => {
                // SAFETY: the handle was created from this device moments ago and never submitted,
                // so this cleanup is its only remaining use.
                unsafe { self.device.destroy_image(image, None) };
                return Err(e);
            }
        };
        // SAFETY: the resource and the memory were both created from this device, the reservation's
        // offset satisfies the alignment its memory requirements reported, and nothing is bound to
        // the resource yet.
        if let Err(e) = unsafe {
            self.device
                .bind_image_memory(image, reservation.memory, reservation.placement.offset)
        } {
            // SAFETY: the handle was created from this device moments ago and never submitted, so
            // this cleanup is its only remaining use.
            unsafe { self.device.destroy_image(image, None) };
            self.release(reservation);
            return Err(super::error::map_vk_result(e, "bind_image_memory"));
        }
        Ok(PooledImage {
            image,
            lease: Some(Rc::new(self.lease(reservation, PooledHandle::Image(image)))),
        })
    }

    // Advance the frame tick, destroy the handles whose retirement has passed,
    // make retired frees placeable again, and release any block that now holds
    // nothing back to the driver.
    pub(super) fn begin_frame(&self) {
        self.advance(1);
        self.release_empty_blocks();
    }

    // Retire everything pending at once. Only sound right after a queue or
    // device wait: the deferred window exists to outlast in-flight work, and a
    // wait leaves none. The synchronous upload helpers call this as their
    // staging drops so an init upload loop peaks at about one staging buffer,
    // instead of accumulating every upload's until `draw_frame` starts ticking
    // `begin_frame`. Emptied blocks are RETAINED, unlike `begin_frame`: the
    // callers are about to allocate again (the next upload's staging, a
    // reload's successor world), and refilling a kept block beats a
    // free-memory / allocate-memory round trip. Whatever stays empty is
    // released by `begin_frame` once frames run.
    pub(super) fn reclaim_idle(&self) {
        let depth = self.inner.borrow().retire_depth;
        self.advance(depth);
    }

    // Open a window in which dropped resources free their ranges immediately,
    // for a caller that has just idled the device and is about to replace
    // resources. Everything already awaiting retirement is retired on entry, and
    // whatever the scope drops is destroyed when the last open scope closes.
    // Close it before the next frame records: a submission that is not
    // waited on would break the idle premise.
    pub(super) fn idle_scope(&self) -> IdleScope {
        self.reclaim_idle();
        self.inner.borrow_mut().idle_scopes += 1;
        IdleScope {
            alloc: self.clone(),
        }
    }

    #[expect(
        clippy::iter_over_hash_type,
        reason = "each pool is visited on its own, so the order cannot show"
    )]
    fn advance(&self, ticks: u64) {
        let mut inner = self.inner.borrow_mut();
        inner.frame += ticks;
        let frame = inner.frame;
        let depth = inner.retire_depth;
        while let Some(retired) = inner.retired.pop_due(frame, depth) {
            self.destroy_retired(retired);
        }
        for pool in inner.pools.values_mut() {
            pool.reclaim(frame);
        }
    }

    #[expect(
        clippy::iter_over_hash_type,
        reason = "each pool is visited on its own, so the order cannot show"
    )]
    fn release_empty_blocks(&self) {
        let mut inner = self.inner.borrow_mut();
        for pool in inner.pools.values_mut() {
            pool.release_empty(|block_index, block| {
                tracing::debug!(
                    "allocator: released empty {} KiB block {block_index}",
                    block.size / 1024,
                );
                // SAFETY: the handle was created from this device and is destroyed exactly
                // once; the caller has already waited for the device to go idle, so no
                // submission still references it.
                unsafe { self.device.free_memory(block.memory, None) };
            });
        }
    }

    #[expect(
        clippy::iter_over_hash_type,
        reason = "each pool is visited on its own, so the order cannot show"
    )]
    pub(super) fn stats(&self) -> AllocatorStats {
        let inner = self.inner.borrow();
        let mut stats = AllocatorStats::default();
        for pool in inner.pools.values() {
            stats.reserved_bytes += pool.reserved_bytes();
            stats.in_use_bytes += pool.in_use_bytes();
            stats.block_count += pool.block_count();
        }
        stats
    }

    // The device's hard ceiling on live allocations, for readouts that want to
    // show how much of it the allocator is using.
    pub(super) fn max_allocations(&self) -> u32 {
        self.max_allocations
    }

    // Destroy everything still queued and free every block. The caller has
    // already idled the device and dropped every pooled resource; this is the
    // last allocator call before `destroy_device`.
    #[expect(
        clippy::iter_over_hash_type,
        reason = "each pool is visited on its own, so the order cannot show"
    )]
    pub(super) fn destroy(&self) {
        let mut inner = self.inner.borrow_mut();
        for retired in inner.retired.drain() {
            self.destroy_retired(retired);
        }
        for pool in inner.pools.values_mut() {
            for block in pool.drain() {
                // SAFETY: the handle was created from this device and is destroyed exactly once;
                // the caller has already waited for the device to go idle, so no submission still
                // references it.
                unsafe { self.device.free_memory(block.memory, None) };
            }
        }
        inner.pools.clear();
    }

    fn destroy_retired(&self, retired: Retired) {
        for view in retired.views {
            // SAFETY: the view was created from this device and is destroyed exactly once; the
            // caller has already waited for the device to go idle, so no submission references it.
            unsafe { self.device.destroy_image_view(view, None) };
        }
        match retired.handle {
            PooledHandle::Buffer(buffer) => {
                // SAFETY: the buffer was created from this device and retired once; the caller's
                // idle wait means no in-flight submission still reads it.
                unsafe { self.device.destroy_buffer(buffer, None) };
            }
            PooledHandle::Image(image) => {
                // SAFETY: the image's views were destroyed above, and the idle device no longer
                // references it, so this is its last use.
                unsafe { self.device.destroy_image(image, None) };
            }
        }
    }

    // Reserve a range for `reqs` in memory satisfying `props`, opening a block
    // when no existing one can host it.
    fn reserve(
        &self,
        reqs: vk::MemoryRequirements,
        props: vk::MemoryPropertyFlags,
        kind: ResourceKind,
        device_address: bool,
    ) -> error::RenderResult<Reservation> {
        let memory_type = self.find_memory_type(reqs.memory_type_bits, props)?;
        let key = PoolKey {
            memory_type,
            kind,
            device_address,
        };
        let align = reqs.alignment.max(1);
        let mut inner = self.inner.borrow_mut();
        let pool = inner.pools.entry(key).or_insert_with(|| Pool::new(GROWTH));
        let reserved = pool
            .reserve(reqs.size, align, |bytes| {
                let (memory, mapped) = Self::create_block(
                    &self.device,
                    &self.memory_props,
                    memory_type,
                    bytes,
                    device_address,
                )?;
                Ok(Block {
                    memory,
                    size: bytes,
                    mapped,
                })
            })
            .map_err(|e| match e {
                ReserveError::Open(e) => e,
                ReserveError::Released => {
                    error::RenderError::Other("allocator: placement named a released block".into())
                }
                ReserveError::Unhosted => error::RenderError::Other(
                    "allocator: a block sized for a request failed to host it".into(),
                ),
            })?;
        Ok(Reservation {
            memory: reserved.block.memory,
            mapped: reserved.block.mapped,
            key,
            placement: reserved.placement,
            size: reqs.size,
        })
    }

    // Return a reservation whose bind failed. The range was never visible to
    // the GPU, so it goes back through the normal deferred path for simplicity,
    // not because it needs the delay.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "consuming the reservation keeps it from being leased or released twice"
    )]
    fn release(&self, reservation: Reservation) {
        let mut inner = self.inner.borrow_mut();
        let retire = inner.frame + inner.retire_depth;
        if let Some(pool) = inner.pools.get_mut(&reservation.key) {
            pool.free(reservation.placement, reservation.size, retire);
        }
    }

    #[expect(
        clippy::needless_pass_by_value,
        reason = "consuming the reservation keeps it from being leased or released twice"
    )]
    fn lease(&self, reservation: Reservation, handle: PooledHandle) -> Lease {
        Lease {
            owner: Rc::downgrade(&self.inner),
            key: reservation.key,
            placement: reservation.placement,
            size: reservation.size,
            handle,
            views: RefCell::new(Vec::new()),
        }
    }

    fn find_memory_type(
        &self,
        type_filter: u32,
        props: vk::MemoryPropertyFlags,
    ) -> error::RenderResult<u32> {
        for i in 0..self.memory_props.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && self.memory_props.memory_types[i as usize]
                    .property_flags
                    .contains(props)
            {
                return Ok(i);
            }
        }
        Err(error::RenderError::Other(
            "no suitable memory type found".to_string(),
        ))
    }

    // Allocate one block and map it when its memory type is host-visible.
    fn create_block(
        device: &Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        memory_type: u32,
        size: u64,
        device_address: bool,
    ) -> error::RenderResult<(vk::DeviceMemory, *mut u8)> {
        let mut flags_info =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(memory_type);
        if device_address {
            info = info.push_next(&mut flags_info);
        }
        // SAFETY: the create-info and every slice it borrows are live for the call, and each handle
        // it names belongs to this device.
        let memory = unsafe { device.allocate_memory(&info, None) }.map_err(|e| {
            super::error::map_vk_result(e, &format!("allocator: block of {size} bytes"))
        })?;
        tracing::debug!(
            "allocator: new {} KiB block (memory type {memory_type}, device_address {device_address})",
            size / 1024,
        );

        let host_visible = memory_props.memory_types[memory_type as usize]
            .property_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
        let mapped = if host_visible {
            // SAFETY: `memory` was allocated from this device and is not already mapped; the
            // whole-size range is in bounds by construction.
            match unsafe {
                device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            } {
                Ok(ptr) => ptr.cast::<u8>(),
                Err(e) => {
                    // SAFETY: the handle was created from this device moments ago and never
                    // submitted, so this cleanup is its only remaining use.
                    unsafe { device.free_memory(memory, None) };
                    return Err(super::error::map_vk_result(e, "allocator: map block"));
                }
            }
        } else {
            std::ptr::null_mut()
        };
        Ok((memory, mapped))
    }
}

// An open `DeviceAllocator::idle_scope`. Closes on drop.
pub(super) struct IdleScope {
    alloc: DeviceAllocator,
}

impl Drop for IdleScope {
    fn drop(&mut self) {
        let closed = {
            let mut inner = self.alloc.inner.borrow_mut();
            inner.idle_scopes = inner.idle_scopes.saturating_sub(1);
            inner.idle_scopes == 0
        };
        if closed {
            self.alloc.reclaim_idle();
        }
    }
}

// The resource's own pointer into its block's mapping, or null when the block
// is not host-visible.
fn resource_ptr(reservation: &Reservation) -> *mut u8 {
    if reservation.mapped.is_null() {
        std::ptr::null_mut()
    } else {
        // SAFETY: `mapped` is the block's whole-size mapping and `placement.offset` is this
        // reservation's offset inside that same block, so the result stays within the mapping.
        unsafe {
            reservation
                .mapped
                .add(usize::saturating_from(reservation.placement.offset))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vulkan::test_gpu::{TestGpu, test_gpu, test_gpu_with_device_address};

    const FIRST_BLOCK_BYTES: u64 = GROWTH.first;
    const MAX_BLOCK_BYTES: u64 = GROWTH.max;

    // A block standing in for a device allocation, for policy-level tests.
    fn fake_block(size: u64) -> Block {
        Block {
            memory: vk::DeviceMemory::null(),
            size,
            mapped: std::ptr::null_mut(),
        }
    }

    #[test]
    fn blocks_double_from_first_to_max() {
        let mut pool = Pool::new(GROWTH);
        let mut expected = FIRST_BLOCK_BYTES;
        for _ in 0..6 {
            let bytes = pool.next_block_bytes(1024, 256);
            assert_eq!(bytes, expected);
            pool.add_block(bytes, fake_block(bytes));
            expected = (expected * 2).min(MAX_BLOCK_BYTES);
        }
        assert_eq!(pool.next_block_bytes(1024, 256), MAX_BLOCK_BYTES);
    }

    #[test]
    fn null_resources_are_inert() {
        let buffer = PooledBuffer::null();
        assert!(buffer.is_null());
        assert!(buffer.mapped_ptr().is_null());
        let image = PooledImage::null();
        assert!(image.is_null());
        drop(buffer);
        drop(image);
    }

    fn test_allocator(gpu: &TestGpu) -> DeviceAllocator {
        DeviceAllocator::new(&gpu.instance, gpu.physical_device, &gpu.device, 2)
    }

    const HOST: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
        vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
            | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
    );

    #[test]
    fn small_buffers_share_one_block_and_map_at_their_offsets() {
        let Some(gpu) = test_gpu() else {
            return;
        };
        let alloc = test_allocator(&gpu);
        let a = alloc
            .create_buffer(1024, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
            .unwrap();
        let b = alloc
            .create_buffer(1024, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
            .unwrap();
        assert_ne!(a.buffer(), b.buffer());
        assert_eq!(alloc.stats().block_count, 1);

        // Each maps at its own offset, and the ranges do not overlap.
        assert!(!a.mapped_ptr().is_null());
        assert!(!b.mapped_ptr().is_null());
        // SAFETY: `a` is HOST_VISIBLE | HOST_COHERENT and sized to 1024 bytes, so its mapping
        // covers the range being filled.
        unsafe { std::ptr::write_bytes(a.mapped_ptr(), 0xAA, 1024) };
        // SAFETY: `b` is a separate 1024-byte host-visible lease, so its mapping covers the fill.
        unsafe { std::ptr::write_bytes(b.mapped_ptr(), 0xBB, 1024) };
        // SAFETY: byte 0 of `a`'s mapping was written by the fill above.
        let a_first = unsafe { *a.mapped_ptr() };
        assert_eq!(a_first, 0xAA);
        // SAFETY: offset 1023 is the last byte inside `a`'s 1024-byte mapping.
        let a_last = unsafe { a.mapped_ptr().add(1023) };
        // SAFETY: `a_last` points at an initialized byte of `a`'s live mapping.
        let a_last = unsafe { *a_last };
        assert_eq!(a_last, 0xAA);
        // SAFETY: byte 0 of `b`'s mapping was written by its fill above.
        let b_first = unsafe { *b.mapped_ptr() };
        assert_eq!(b_first, 0xBB);

        drop(a);
        drop(b);
        alloc.destroy();
    }

    #[test]
    fn safe_writes_land_at_their_offsets() {
        let Some(gpu) = test_gpu() else {
            return;
        };
        let alloc = test_allocator(&gpu);
        let buffer = alloc
            .create_buffer(1024, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
            .unwrap();
        assert!(buffer.byte_len() >= 1024);

        buffer.write_bytes(0, &[0x11, 0x22]);
        buffer.write_val(4, &0x3344_5566_u32);
        buffer.write_slice(8, &[0x77_u8, 0x88]);
        buffer.write_bytes(2, &[]);
        buffer.write_bytes(10, &[0xEE, 0xEE]);
        buffer.zero_bytes(10, 2);
        buffer.zero_bytes(12, 0);
        // The buffer is HOST_VISIBLE | HOST_COHERENT, so its mapping covers the
        // 1024 requested bytes and every read below is in range.
        let bytes = buffer.mapped_ptr();
        // SAFETY: offset 4 plus a `u32` stays inside the 1024-byte mapping.
        let word = unsafe { bytes.add(4) }.cast::<[u8; 4]>();
        // SAFETY: `word` points at 4 initialized bytes of the live mapping.
        let word = unsafe { word.read() };
        assert_eq!(u32::from_ne_bytes(word), 0x3344_5566);
        for (offset, expected) in [(0, 0x11), (1, 0x22), (8, 0x77), (9, 0x88), (10, 0), (11, 0)] {
            // SAFETY: every offset in the table is below the 1024-byte mapping length.
            let byte = unsafe { bytes.add(offset) };
            // SAFETY: `byte` points into the live mapping and the write calls above initialized it.
            let byte = unsafe { *byte };
            assert_eq!(byte, expected);
        }

        drop(buffer);
        alloc.destroy();
    }

    #[test]
    fn an_out_of_bounds_write_panics() {
        let Some(gpu) = test_gpu() else {
            return;
        };
        let alloc = test_allocator(&gpu);
        let buffer = alloc
            .create_buffer(64, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
            .unwrap();
        let len = buffer.byte_len();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            buffer.write_bytes(len, &[0xFF]);
        }));
        assert!(result.is_err());
        drop(buffer);
        alloc.destroy();
    }

    #[test]
    fn writes_to_a_null_buffer_panic() {
        let buffer = PooledBuffer::null();
        assert_eq!(buffer.byte_len(), 0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            buffer.write_val(0, &1_u32);
        }));
        assert!(result.is_err());
    }

    #[test]
    fn a_dropped_buffer_frees_its_range_after_the_retire_window() {
        let Some(gpu) = test_gpu() else {
            return;
        };
        let alloc = test_allocator(&gpu);
        let buffer = alloc
            .create_buffer(1024, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
            .unwrap();
        let held = alloc.stats();
        assert!(held.in_use_bytes >= 1024);
        assert_eq!(held.block_count, 1);

        // The drop returns the bytes immediately but the block holds until the
        // retire window has passed, then goes back to the driver.
        drop(buffer);
        assert_eq!(alloc.stats().in_use_bytes, 0);
        assert_eq!(alloc.stats().block_count, 1);
        // frames_in_flight = 2, so the window is 3 ticks: still held mid-way.
        alloc.begin_frame();
        assert_eq!(alloc.stats().block_count, 1);
        alloc.begin_frame();
        alloc.begin_frame();
        assert_eq!(alloc.stats().block_count, 0);
        assert_eq!(alloc.stats().reserved_bytes, 0);
        alloc.destroy();
    }

    #[test]
    fn a_clone_keeps_the_resource_alive() {
        let Some(gpu) = test_gpu() else {
            return;
        };
        let alloc = test_allocator(&gpu);
        let a = alloc
            .create_buffer(1024, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
            .unwrap();
        let b = a.clone();
        drop(a);
        // The clone still holds the range: nothing has been freed.
        assert!(alloc.stats().in_use_bytes >= 1024);
        drop(b);
        assert_eq!(alloc.stats().in_use_bytes, 0);
        alloc.destroy();
    }

    #[test]
    fn an_image_and_its_views_retire_together() {
        let Some(gpu) = test_gpu() else {
            return;
        };
        let alloc = test_allocator(&gpu);
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: 4,
                height: 4,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::R8G8B8A8_UNORM)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);
        let image = alloc
            .create_image(&info, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .unwrap();
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image.image())
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        // SAFETY: the create-info and every slice it borrows are live for the call, and each handle
        // it names belongs to this device.
        let view = unsafe { gpu.device.create_image_view(&view_info, None) }.unwrap();
        image.attach_view(view);

        drop(image);
        for _ in 0..3 {
            alloc.begin_frame();
        }
        assert_eq!(alloc.stats().block_count, 0);
        alloc.destroy();
    }

    #[test]
    fn linear_and_optimal_images_never_share_a_block() {
        let Some(gpu) = test_gpu() else {
            return;
        };
        let alloc = test_allocator(&gpu);
        let base = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: 4,
                height: 4,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::R8G8B8A8_UNORM)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);
        let optimal = alloc
            .create_image(
                &base
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::SAMPLED),
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .unwrap();
        let linear = alloc
            .create_image(
                &base
                    .tiling(vk::ImageTiling::LINEAR)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST),
                HOST,
            )
            .unwrap();
        // Different tiling class and different memory type both force the
        // split; either alone would.
        assert_eq!(alloc.stats().block_count, 2);
        drop(optimal);
        drop(linear);
        alloc.destroy();
    }

    #[test]
    fn an_oversized_buffer_gets_a_dedicated_block() {
        let Some(gpu) = test_gpu() else {
            return;
        };
        let alloc = test_allocator(&gpu);
        let big = alloc
            .create_buffer(
                MAX_BLOCK_BYTES + 1024,
                vk::BufferUsageFlags::TRANSFER_SRC,
                HOST,
            )
            .unwrap();
        assert_eq!(alloc.stats().block_count, 1);
        // A dedicated block is never shared: the next resource opens its own.
        let small = alloc
            .create_buffer(1024, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
            .unwrap();
        assert_eq!(alloc.stats().block_count, 2);
        drop(big);
        drop(small);
        alloc.destroy();
    }

    #[test]
    fn a_reclaimed_range_is_reused_by_a_later_allocation() {
        let Some(gpu) = test_gpu() else {
            return;
        };
        let alloc = test_allocator(&gpu);
        // The anchor keeps the block alive; the big buffer fills most of it, so
        // a second big buffer only fits if the first one's range came back.
        let anchor = alloc
            .create_buffer(1024, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
            .unwrap();
        let big_size = 3 * 1024 * 1024;
        let big = alloc
            .create_buffer(big_size, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
            .unwrap();
        assert_eq!(alloc.stats().block_count, 1);
        drop(big);
        for _ in 0..3 {
            alloc.begin_frame();
        }
        let again = alloc
            .create_buffer(big_size, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
            .unwrap();
        assert_eq!(alloc.stats().block_count, 1);
        drop(anchor);
        drop(again);
        alloc.destroy();
    }

    #[test]
    fn reclaim_idle_retires_pending_frees_without_frame_ticks() {
        let Some(gpu) = test_gpu() else {
            return;
        };
        let alloc = test_allocator(&gpu);
        let anchor = alloc
            .create_buffer(1024, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
            .unwrap();
        let big_size = 3 * 1024 * 1024;
        let big = alloc
            .create_buffer(big_size, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
            .unwrap();
        drop(big);
        // No begin_frame between the drop and the next allocation, like an
        // init-time upload loop; reclaim_idle alone must make the range
        // placeable again.
        alloc.reclaim_idle();
        let again = alloc
            .create_buffer(big_size, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
            .unwrap();
        assert_eq!(alloc.stats().block_count, 1);
        drop(anchor);
        drop(again);
        alloc.destroy();
    }

    #[test]
    fn a_released_reservation_returns_its_range() {
        let Some(gpu) = test_gpu() else {
            return;
        };
        let alloc = test_allocator(&gpu);
        // Exercise the bind-failure path directly: reserve, then release
        // without a resource ever binding.
        let reqs = vk::MemoryRequirements {
            size: 1024,
            alignment: 256,
            memory_type_bits: !0,
        };
        let reservation = alloc
            .reserve(reqs, HOST, ResourceKind::Linear, false)
            .unwrap();
        assert!(alloc.stats().in_use_bytes >= 1024);
        alloc.release(reservation);
        assert_eq!(alloc.stats().in_use_bytes, 0);
        for _ in 0..3 {
            alloc.begin_frame();
        }
        assert_eq!(alloc.stats().block_count, 0);
        alloc.destroy();
    }

    #[test]
    fn an_emptied_pool_restarts_the_growth_ladder() {
        // Policy-level: climb the ladder, drain and release every block, and
        // the next block is back at `FIRST_BLOCK_BYTES`. Deliberate (see
        // `next_block_bytes`): an emptied pool recommits small rather than at
        // its high-water block size.
        let mut pool = Pool::new(GROWTH);
        let b1 = pool.next_block_bytes(1024, 1);
        assert_eq!(b1, FIRST_BLOCK_BYTES);
        let i1 = pool.add_block(b1, fake_block(b1));
        let p1 = pool.alloc_in(i1, 1024, 1).unwrap();
        let b2 = pool.next_block_bytes(1024, 1);
        assert_eq!(b2, FIRST_BLOCK_BYTES * 2);
        let i2 = pool.add_block(b2, fake_block(b2));
        let p2 = pool.alloc_in(i2, 1024, 1).unwrap();

        pool.free(p1, 1024, 0);
        pool.free(p2, 1024, 0);
        pool.reclaim(1);
        let mut released = 0;
        pool.release_empty(|_, _| released += 1);
        assert_eq!(released, 2);
        assert_eq!(pool.next_block_bytes(1024, 1), FIRST_BLOCK_BYTES);
    }

    // An `Inner` holding one pool with one block, and a lease on `size` bytes
    // of it. No device: `release` only touches placement state and the queue.
    fn leased_inner(size: u64) -> (Rc<RefCell<Inner>>, Lease) {
        let key = PoolKey {
            memory_type: 0,
            kind: ResourceKind::Linear,
            device_address: false,
        };
        let mut pool = Pool::new(GROWTH);
        let block = pool.add_block(FIRST_BLOCK_BYTES, fake_block(FIRST_BLOCK_BYTES));
        let placement = pool.alloc_in(block, size, 1).unwrap();
        let inner = Rc::new(RefCell::new(Inner {
            pools: HashMap::from([(key, pool)]),
            retired: RetirePool::new(),
            frame: 0,
            retire_depth: 3,
            idle_scopes: 0,
        }));
        let lease = Lease {
            owner: Rc::downgrade(&inner),
            key,
            placement,
            size,
            handle: PooledHandle::Buffer(vk::Buffer::null()),
            views: RefCell::new(Vec::new()),
        };
        (inner, lease)
    }

    fn placeable(inner: &Rc<RefCell<Inner>>, size: u64) -> bool {
        let mut inner = inner.borrow_mut();
        let pool = inner.pools.values_mut().next().unwrap();
        pool.alloc(size, 1).is_some()
    }

    #[test]
    fn a_release_outside_an_idle_scope_withholds_the_range() {
        let (inner, lease) = leased_inner(FIRST_BLOCK_BYTES);
        drop(lease);
        assert!(!placeable(&inner, FIRST_BLOCK_BYTES));
        assert_eq!(inner.borrow().retired.len(), 1);
    }

    #[test]
    fn a_release_inside_an_idle_scope_frees_the_range_at_once() {
        let (inner, lease) = leased_inner(FIRST_BLOCK_BYTES);
        inner.borrow_mut().idle_scopes = 1;
        drop(lease);
        assert!(placeable(&inner, FIRST_BLOCK_BYTES));
        // The handles still wait for the scope to close.
        assert_eq!(inner.borrow().retired.len(), 1);
    }

    #[test]
    fn an_idle_scope_refills_a_replaced_range_instead_of_opening_a_block() {
        let Some(gpu) = test_gpu() else {
            return;
        };
        let alloc = test_allocator(&gpu);
        // The anchor keeps the first block alive and the big buffer fills most
        // of it, so a replacement only fits where the old one was.
        let anchor = alloc
            .create_buffer(1024, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
            .unwrap();
        let big_size = 3 * 1024 * 1024;
        let mut big = alloc
            .create_buffer(big_size, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
            .unwrap();
        {
            let _idle = alloc.idle_scope();
            for _ in 0..4 {
                drop(big);
                big = alloc
                    .create_buffer(big_size, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
                    .unwrap();
                assert_eq!(alloc.stats().block_count, 1);
            }
        }
        // Without the scope the same replacement opens a second block.
        drop(big);
        let again = alloc
            .create_buffer(big_size, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
            .unwrap();
        assert_eq!(alloc.stats().block_count, 2);
        drop(anchor);
        drop(again);
        alloc.destroy();
    }

    #[test]
    fn opening_an_idle_scope_retires_frees_already_pending() {
        let Some(gpu) = test_gpu() else {
            return;
        };
        let alloc = test_allocator(&gpu);
        let anchor = alloc
            .create_buffer(1024, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
            .unwrap();
        let big_size = 3 * 1024 * 1024;
        let big = alloc
            .create_buffer(big_size, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
            .unwrap();
        drop(big);
        let idle = alloc.idle_scope();
        assert_eq!(alloc.inner.borrow().retired.len(), 0);
        let again = alloc
            .create_buffer(big_size, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
            .unwrap();
        assert_eq!(alloc.stats().block_count, 1);
        drop(anchor);
        drop(again);
        drop(idle);
        alloc.destroy();
    }

    #[test]
    fn only_the_outermost_idle_scope_destroys_dropped_handles() {
        let Some(gpu) = test_gpu() else {
            return;
        };
        let alloc = test_allocator(&gpu);
        let a = alloc
            .create_buffer(1024, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
            .unwrap();
        let b = alloc
            .create_buffer(1024, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
            .unwrap();
        let outer = alloc.idle_scope();
        {
            let _inner = alloc.idle_scope();
            drop(a);
        }
        // The outer scope is still open: the handles wait, and a later drop
        // still frees its range at once.
        assert_eq!(alloc.inner.borrow().retired.len(), 1);
        drop(b);
        assert_eq!(alloc.stats().in_use_bytes, 0);
        drop(outer);
        assert_eq!(alloc.inner.borrow().retired.len(), 0);
        assert_eq!(alloc.inner.borrow().idle_scopes, 0);
        alloc.destroy();
    }

    #[test]
    fn device_address_buffers_pool_apart_and_report_addresses() {
        let Some(gpu) = test_gpu_with_device_address() else {
            return;
        };
        let alloc = test_allocator(&gpu);
        let plain = alloc
            .create_buffer(1024, vk::BufferUsageFlags::TRANSFER_SRC, HOST)
            .unwrap();
        let a = alloc
            .create_buffer(1024, vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS, HOST)
            .unwrap();
        let b = alloc
            .create_buffer(1024, vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS, HOST)
            .unwrap();
        // The device-address buffers share one DEVICE_ADDRESS block; the plain
        // buffer never joins it.
        assert_eq!(alloc.stats().block_count, 2);

        // `get_buffer_device_address` must be valid for a pooled buffer at any
        // offset; `b` sits at a non-zero offset behind `a`.
        // SAFETY: `buffer` was created from this device with SHADER_DEVICE_ADDRESS usage and the
        // info struct borrows it for the call; the query only reads.
        let address = |buffer: vk::Buffer| unsafe {
            gpu.device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };
        let addr_a = address(a.buffer());
        let addr_b = address(b.buffer());
        assert_ne!(addr_a, 0);
        assert_ne!(addr_b, 0);
        assert_ne!(addr_a, addr_b);

        drop(plain);
        drop(a);
        drop(b);
        alloc.destroy();
    }
}
