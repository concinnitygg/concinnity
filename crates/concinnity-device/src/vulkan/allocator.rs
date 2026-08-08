// src/vulkan/allocator.rs
//
// The device-memory allocator every persistent Vulkan resource is created
// through. Buffers and images are suballocated out of a few large
// `VkDeviceMemory` blocks instead of each owning one.
//
// `vkAllocateMemory` is a scarce call. `maxMemoryAllocationCount` is commonly
// 4096 on desktop drivers and lower elsewhere, and it is a hard cap: past it
// allocation fails outright, however much memory is free. One allocation per
// resource spends that budget on resource count, so a world within the byte
// budget can still fail to load. Blocks make the cap a function of bytes
// rather than of how many things the world holds.
//
// `block_alloc::BlockAllocator` decides which block and what offset; this file
// is what makes those decisions Vulkan. The split is the same one the transient
// image pool uses: shared placement policy, backend-specific binding.
//
// Blocks are separated into pools by three properties, because each is fixed
// for the lifetime of an allocation and cannot be mixed within one:
//
//   memory type      what `vkAllocateMemory` was given; a resource can only
//                    bind to memory of a type its requirements permit
//   tiling class     Vulkan requires linear and optimal-tiling resources
//                    sharing an allocation to be separated by
//                    `bufferImageGranularity`. Separate pools remove the
//                    constraint rather than paying to honour it.
//   device address   `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT` is a property of
//                    the allocation, not the buffer, so ray-tracing buffers
//                    that need `vkGetBufferDeviceAddress` need their own blocks
//
// Host-visible blocks are mapped once, at block creation, and stay mapped.
// Vulkan forbids mapping one allocation twice, so a per-resource map is not
// even available once resources share a block; each resource carries a pointer
// to its own bytes at its own offset into the block's mapping. Every
// host-visible memory type this backend allocates is coherent, so the pointers
// need no flush discipline.
//
// Lifetime is RAII, like the Metal and D3D12 pools. Dropping a `PooledBuffer` /
// `PooledImage` returns its range to the pool and queues its Vulkan handles for
// destruction, both withheld until `frames_in_flight + 1` frame ticks have
// passed, which covers any command buffer that could still reference them. A
// caller therefore never destroys, frees, or reasons about GPU progress on
// teardown; assignment and drop are the whole discipline. The leases are
// cloneable so a resource can be held by a ring slot and the live field that
// reads it at once.
//
// The `Rc` behind the leases is main-thread state, on the same invariant as
// `unsafe impl Send for VkContext`: the context migrates between threads but is
// only ever used from one at a time, and workers given `&VkContext` only read
// handles, never drop or allocate.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

use ash::{Device, vk};

use crate::gfx::block_alloc::{BlockAllocator, Placement};

// Largest block the pool asks for. Big enough that a heavy world holds its
// persistent set in a handful of blocks, small enough that one block is not an
// absurd commitment. A resource too large for one gets a dedicated block sized
// to itself.
const MAX_BLOCK_BYTES: u64 = 64 * 1024 * 1024;

// Size of a pool's first block. Blocks double from here to `MAX_BLOCK_BYTES` as
// a pool fills, so a small world commits megabytes rather than a full-size
// block per pool for a handful of resources.
const FIRST_BLOCK_BYTES: u64 = 4 * 1024 * 1024;

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
struct Block {
    memory: vk::DeviceMemory,
    // Base of the block's mapping, or null when the memory type is not
    // host-visible.
    mapped: *mut u8,
}

// The blocks of one pool, indexed the way `BlockAllocator` names them.
struct Pool {
    placement: BlockAllocator,
    blocks: Vec<Option<Block>>,
}

impl Pool {
    fn new() -> Self {
        Self {
            placement: BlockAllocator::new(MAX_BLOCK_BYTES),
            blocks: Vec::new(),
        }
    }

    // How large the next block should be to host `size` bytes at `align`. The
    // standard size doubles with the pool's live block count up to
    // `MAX_BLOCK_BYTES`, and any request too large for that gets a block of its
    // own size (which `BlockAllocator::add_block` then marks dedicated). Keyed
    // off the live count, so a pool that empties out restarts the ladder: a
    // teardown that drops a large world recommits small for its successor
    // rather than at the old high-water block size.
    fn next_block_bytes(&self, size: u64, align: u64) -> u64 {
        let grown = FIRST_BLOCK_BYTES
            .saturating_mul(1 << self.placement.block_count().min(4))
            .min(MAX_BLOCK_BYTES);
        size.saturating_add(align.max(1).saturating_sub(1))
            .max(grown)
    }
}

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
    retire_at: u64,
}

struct Inner {
    pools: HashMap<PoolKey, Pool>,
    retired: Vec<Retired>,
    // Monotonic frame tick driving the deferred frees. Not the frame-in-flight
    // index, which wraps.
    frame: u64,
    retire_depth: u64,
}

impl Inner {
    // Return a lease's range to its pool and queue its handles, both withheld
    // until enough frames have ticked that no in-flight command buffer can
    // reference them.
    fn release(&mut self, lease: &mut Lease) {
        let retire = self.frame + self.retire_depth;
        if let Some(pool) = self.pools.get_mut(&lease.key) {
            pool.placement.free(lease.placement, lease.size, retire);
        }
        self.retired.push(Retired {
            handle: std::mem::replace(&mut lease.handle, PooledHandle::Buffer(vk::Buffer::null())),
            views: std::mem::take(&mut *lease.views.borrow_mut()),
            retire_at: retire,
        });
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
    _lease: Option<Rc<Lease>>,
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

    // A placeholder naming no buffer, for a slot filled before its real
    // resource exists. Dropping it is a no-op.
    pub(super) fn null() -> Self {
        Self {
            buffer: vk::Buffer::null(),
            mapped: std::ptr::null_mut(),
            _lease: None,
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
    mapped: *mut u8,
    _lease: Option<Rc<Lease>>,
}

impl PooledImage {
    pub(super) fn image(&self) -> vk::Image {
        self.image
    }

    // A pointer to this image's own bytes, or null when its memory type is not
    // host-visible. Only meaningful for LINEAR-tiled images; nothing pools one
    // yet, so this is exercised by the unit tests alone. Remove the allow with
    // its first live caller.
    #[allow(dead_code)]
    pub(super) fn mapped_ptr(&self) -> *mut u8 {
        self.mapped
    }

    // Tie `view` to this image's lifetime: it is destroyed just before the
    // image, at the same retirement.
    pub(super) fn attach_view(&self, view: vk::ImageView) {
        debug_assert!(self._lease.is_some(), "attach_view on a null PooledImage");
        if let Some(lease) = &self._lease {
            lease.views.borrow_mut().push(view);
        }
    }

    // A placeholder naming no image, for a slot filled before its real resource
    // exists. Dropping it is a no-op.
    pub(super) fn null() -> Self {
        Self {
            image: vk::Image::null(),
            mapped: std::ptr::null_mut(),
            _lease: None,
        }
    }

    // Exercised by the unit tests; no live caller branches on an image's null
    // state yet. Remove the allow with the first one.
    #[allow(dead_code)]
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
        let memory_props =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let max_allocations = unsafe { instance.get_physical_device_properties(physical_device) }
            .limits
            .max_memory_allocation_count;
        Self {
            device: device.clone(),
            inner: Rc::new(RefCell::new(Inner {
                pools: HashMap::new(),
                retired: Vec::new(),
                frame: 0,
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
    ) -> crate::gfx::error::RenderResult<PooledBuffer> {
        let info = vk::BufferCreateInfo::default()
            .size(size.max(1))
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.device.create_buffer(&info, None) }
            .map_err(|e| super::error::map_vk_result(e, "create_buffer"))?;
        let reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let device_address = usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
        let reservation = match self.reserve(reqs, props, ResourceKind::Linear, device_address) {
            Ok(r) => r,
            Err(e) => {
                unsafe { self.device.destroy_buffer(buffer, None) };
                return Err(e);
            }
        };
        if let Err(e) = unsafe {
            self.device
                .bind_buffer_memory(buffer, reservation.memory, reservation.placement.offset)
        } {
            unsafe { self.device.destroy_buffer(buffer, None) };
            self.release(reservation);
            return Err(super::error::map_vk_result(e, "bind_buffer_memory"));
        }
        let mapped = resource_ptr(&reservation);
        Ok(PooledBuffer {
            buffer,
            mapped,
            _lease: Some(Rc::new(
                self.lease(reservation, PooledHandle::Buffer(buffer)),
            )),
        })
    }

    // Create an image from `info`, placed in memory satisfying `props`. The
    // tiling in `info` picks the pool: a LINEAR image shares blocks with
    // buffers, an OPTIMAL one never does.
    pub(super) fn create_image(
        &self,
        info: &vk::ImageCreateInfo,
        props: vk::MemoryPropertyFlags,
    ) -> crate::gfx::error::RenderResult<PooledImage> {
        let image = unsafe { self.device.create_image(info, None) }
            .map_err(|e| super::error::map_vk_result(e, "create_image"))?;
        let reqs = unsafe { self.device.get_image_memory_requirements(image) };
        let kind = if info.tiling == vk::ImageTiling::LINEAR {
            ResourceKind::Linear
        } else {
            ResourceKind::Optimal
        };
        let reservation = match self.reserve(reqs, props, kind, false) {
            Ok(r) => r,
            Err(e) => {
                unsafe { self.device.destroy_image(image, None) };
                return Err(e);
            }
        };
        if let Err(e) = unsafe {
            self.device
                .bind_image_memory(image, reservation.memory, reservation.placement.offset)
        } {
            unsafe { self.device.destroy_image(image, None) };
            self.release(reservation);
            return Err(super::error::map_vk_result(e, "bind_image_memory"));
        }
        let mapped = resource_ptr(&reservation);
        Ok(PooledImage {
            image,
            mapped,
            _lease: Some(Rc::new(self.lease(reservation, PooledHandle::Image(image)))),
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

    fn advance(&self, ticks: u64) {
        let mut inner = self.inner.borrow_mut();
        inner.frame += ticks;
        let frame = inner.frame;
        let mut index = 0;
        while index < inner.retired.len() {
            if inner.retired[index].retire_at <= frame {
                let retired = inner.retired.swap_remove(index);
                self.destroy_retired(retired);
            } else {
                index += 1;
            }
        }
        for pool in inner.pools.values_mut() {
            pool.placement.reclaim(frame);
        }
    }

    fn release_empty_blocks(&self) {
        let mut inner = self.inner.borrow_mut();
        for pool in inner.pools.values_mut() {
            for block_index in pool.placement.take_empty_blocks() {
                if let Some(block) = pool.blocks.get_mut(block_index).and_then(Option::take) {
                    tracing::debug!("allocator: released empty block {block_index}");
                    unsafe { self.device.free_memory(block.memory, None) };
                }
            }
        }
    }

    pub(super) fn stats(&self) -> AllocatorStats {
        let inner = self.inner.borrow();
        let mut stats = AllocatorStats::default();
        for pool in inner.pools.values() {
            stats.reserved_bytes += pool.placement.reserved_bytes();
            stats.in_use_bytes += pool.placement.in_use_bytes();
            stats.block_count += pool.placement.block_count();
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
    pub(super) fn destroy(&self) {
        let mut inner = self.inner.borrow_mut();
        for retired in std::mem::take(&mut inner.retired) {
            self.destroy_retired(retired);
        }
        for pool in inner.pools.values_mut() {
            for block in pool.blocks.iter_mut().filter_map(Option::take) {
                unsafe { self.device.free_memory(block.memory, None) };
            }
        }
        inner.pools.clear();
    }

    fn destroy_retired(&self, retired: Retired) {
        unsafe {
            for view in retired.views {
                self.device.destroy_image_view(view, None);
            }
            match retired.handle {
                PooledHandle::Buffer(buffer) => self.device.destroy_buffer(buffer, None),
                PooledHandle::Image(image) => self.device.destroy_image(image, None),
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
    ) -> crate::gfx::error::RenderResult<Reservation> {
        let memory_type = self.find_memory_type(reqs.memory_type_bits, props)?;
        let key = PoolKey {
            memory_type,
            kind,
            device_address,
        };
        let align = reqs.alignment.max(1);
        let mut inner = self.inner.borrow_mut();
        let pool = inner.pools.entry(key).or_insert_with(Pool::new);

        // An existing block first; only open a new one when none can host it.
        if let Some(placement) = pool.placement.alloc(reqs.size, align) {
            let block = pool.blocks[placement.block]
                .as_ref()
                .ok_or("allocator: placement named a released block")?;
            return Ok(Reservation {
                memory: block.memory,
                mapped: block.mapped,
                key,
                placement,
                size: reqs.size,
            });
        }

        let block_bytes = pool.next_block_bytes(reqs.size, align);
        let (memory, mapped) = Self::create_block(
            &self.device,
            &self.memory_props,
            memory_type,
            block_bytes,
            device_address,
        )?;
        let index = pool.placement.add_block(block_bytes);
        if index == pool.blocks.len() {
            pool.blocks.push(Some(Block { memory, mapped }));
        } else {
            pool.blocks[index] = Some(Block { memory, mapped });
        }
        let placement = pool
            .placement
            .alloc_in(index, reqs.size, align)
            .ok_or("allocator: a block sized for a request failed to host it")?;
        Ok(Reservation {
            memory,
            mapped,
            key,
            placement,
            size: reqs.size,
        })
    }

    // Return a reservation whose bind failed. The range was never visible to
    // the GPU, so it goes back through the normal deferred path for simplicity,
    // not because it needs the delay.
    fn release(&self, reservation: Reservation) {
        let mut inner = self.inner.borrow_mut();
        let retire = inner.frame + inner.retire_depth;
        if let Some(pool) = inner.pools.get_mut(&reservation.key) {
            pool.placement
                .free(reservation.placement, reservation.size, retire);
        }
    }

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
    ) -> Result<u32, String> {
        for i in 0..self.memory_props.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && self.memory_props.memory_types[i as usize]
                    .property_flags
                    .contains(props)
            {
                return Ok(i);
            }
        }
        Err("no suitable memory type found".to_string())
    }

    // Allocate one block and map it when its memory type is host-visible.
    fn create_block(
        device: &Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        memory_type: u32,
        size: u64,
        device_address: bool,
    ) -> crate::gfx::error::RenderResult<(vk::DeviceMemory, *mut u8)> {
        let mut flags_info =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(memory_type);
        if device_address {
            info = info.push_next(&mut flags_info);
        }
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
            match unsafe {
                device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            } {
                Ok(ptr) => ptr as *mut u8,
                Err(e) => {
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

// The resource's own pointer into its block's mapping, or null when the block
// is not host-visible.
fn resource_ptr(reservation: &Reservation) -> *mut u8 {
    if reservation.mapped.is_null() {
        std::ptr::null_mut()
    } else {
        unsafe {
            reservation
                .mapped
                .add(reservation.placement.offset as usize)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_double_from_first_to_max() {
        let mut pool = Pool::new();
        let mut expected = FIRST_BLOCK_BYTES;
        for _ in 0..6 {
            let bytes = pool.next_block_bytes(1024, 256);
            assert_eq!(bytes, expected);
            pool.placement.add_block(bytes);
            expected = (expected * 2).min(MAX_BLOCK_BYTES);
        }
        assert_eq!(pool.next_block_bytes(1024, 256), MAX_BLOCK_BYTES);
    }

    #[test]
    fn an_oversized_request_sizes_its_own_block() {
        let pool = Pool::new();
        let size = MAX_BLOCK_BYTES * 2;
        assert_eq!(pool.next_block_bytes(size, 1), size);
        // Alignment slack is included so the block can place the request at
        // whatever offset it lands on.
        assert_eq!(pool.next_block_bytes(size, 4096), size + 4095);
    }

    #[test]
    fn null_resources_are_inert() {
        let buffer = PooledBuffer::null();
        assert!(buffer.is_null());
        assert!(buffer.mapped_ptr().is_null());
        let image = PooledImage::null();
        assert!(image.is_null());
        assert!(image.mapped_ptr().is_null());
        drop(buffer);
        drop(image);
    }

    // A headless instance and device for the allocation tests, or None where no
    // Vulkan driver is present (CI). Field order is drop order for the explicit
    // teardown in `Drop`.
    struct TestGpu {
        device: Device,
        instance: ash::Instance,
        physical_device: vk::PhysicalDevice,
        _entry: ash::Entry,
    }

    impl Drop for TestGpu {
        fn drop(&mut self) {
            unsafe {
                self.device.destroy_device(None);
                self.instance.destroy_instance(None);
            }
        }
    }

    fn test_gpu() -> Option<TestGpu> {
        let gpu = test_gpu_impl(false);
        if gpu.is_none() {
            eprintln!("skipped: no Vulkan driver");
        }
        gpu
    }

    // A device with `bufferDeviceAddress` enabled (core 1.2), for the
    // device-address pool tests, or None where the driver cannot provide one.
    fn test_gpu_with_device_address() -> Option<TestGpu> {
        let gpu = test_gpu_impl(true);
        if gpu.is_none() {
            eprintln!("skipped: no bufferDeviceAddress-capable Vulkan driver");
        }
        gpu
    }

    fn test_gpu_impl(device_address: bool) -> Option<TestGpu> {
        let entry = crate::vulkan::loader::load_entry().ok()?;
        let app = vk::ApplicationInfo::default().api_version(if device_address {
            vk::API_VERSION_1_2
        } else {
            vk::API_VERSION_1_0
        });
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
        }
        .ok()?;
        let destroy_instance = |instance: ash::Instance| {
            unsafe { instance.destroy_instance(None) };
            None
        };
        let physical_device = match unsafe { instance.enumerate_physical_devices() } {
            Ok(devices) if !devices.is_empty() => devices[0],
            _ => return destroy_instance(instance),
        };
        let mut enable = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
        if device_address {
            let props = unsafe { instance.get_physical_device_properties(physical_device) };
            if props.api_version < vk::API_VERSION_1_2 {
                return destroy_instance(instance);
            }
            let mut bda = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
            let mut feats = vk::PhysicalDeviceFeatures2::default().push_next(&mut bda);
            unsafe { instance.get_physical_device_features2(physical_device, &mut feats) };
            if bda.buffer_device_address == 0 {
                return destroy_instance(instance);
            }
            enable = enable.buffer_device_address(true);
        }
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(0)
            .queue_priorities(&[1.0])];
        let mut device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_infos);
        if device_address {
            device_info = device_info.push_next(&mut enable);
        }
        let device = match unsafe { instance.create_device(physical_device, &device_info, None) } {
            Ok(device) => device,
            Err(_) => return destroy_instance(instance),
        };
        Some(TestGpu {
            device,
            instance,
            physical_device,
            _entry: entry,
        })
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
        unsafe {
            std::ptr::write_bytes(a.mapped_ptr(), 0xAA, 1024);
            std::ptr::write_bytes(b.mapped_ptr(), 0xBB, 1024);
            assert_eq!(*a.mapped_ptr(), 0xAA);
            assert_eq!(*a.mapped_ptr().add(1023), 0xAA);
            assert_eq!(*b.mapped_ptr(), 0xBB);
        }

        drop(a);
        drop(b);
        alloc.destroy();
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
        let mut pool = Pool::new();
        let b1 = pool.next_block_bytes(1024, 1);
        assert_eq!(b1, FIRST_BLOCK_BYTES);
        let i1 = pool.placement.add_block(b1);
        let p1 = pool.placement.alloc_in(i1, 1024, 1).unwrap();
        let b2 = pool.next_block_bytes(1024, 1);
        assert_eq!(b2, FIRST_BLOCK_BYTES * 2);
        let i2 = pool.placement.add_block(b2);
        let p2 = pool.placement.alloc_in(i2, 1024, 1).unwrap();

        pool.placement.free(p1, 1024, 0);
        pool.placement.free(p2, 1024, 0);
        pool.placement.reclaim(1);
        assert_eq!(pool.placement.take_empty_blocks().len(), 2);
        assert_eq!(pool.next_block_bytes(1024, 1), FIRST_BLOCK_BYTES);
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
