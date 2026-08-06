// src/vulkan/allocator.rs
//
// The device-memory allocator every persistent Vulkan resource is placed
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
// even available once resources share a block; each allocation reads its own
// bytes at its own offset into the block's pointer. This is also faster than
// the map/unmap pair it replaces.
//
// Frees are deferred. A dedicated allocation freed while the GPU still reads it
// is merely undefined; a suballocation freed early is handed to a different
// resource almost immediately, so the retire discipline is load-bearing here in
// a way it was not before. `free` never releases bytes for reuse until
// `frames_in_flight + 1` frame ticks have passed, which covers any command
// buffer that could still reference them.

// Removed once `create_buffer` and `create_image` allocate through this module:
// until they do, nothing constructs a `DeviceAllocator` and every item here
// reads as dead.
#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::HashMap;

use ash::{Device, vk};

use crate::gfx::block_alloc::{BlockAllocator, Placement};

// Standard block size. Large enough that a normal world holds its whole
// resource set in a handful of blocks, small enough that a block is not an
// absurd commitment on a small device. A resource too large for one gets a
// dedicated block of its own size.
const BLOCK_BYTES: u64 = 64 * 1024 * 1024;

// Whether a resource is laid out linearly (buffers) or in an
// implementation-defined tiling (images). Kept apart so `bufferImageGranularity`
// never applies.
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
            placement: BlockAllocator::new(BLOCK_BYTES),
            blocks: Vec::new(),
        }
    }
}

// A placed resource. Plain data: it names its block and where in it the
// resource sits, and is what `bind_buffer_memory` / `bind_image_memory` and
// every later free are given in place of a bare `vk::DeviceMemory`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Allocation {
    memory: vk::DeviceMemory,
    offset: u64,
    size: u64,
    key: PoolKey,
    placement: Placement,
}

impl Allocation {
    // The block's device memory. What a bind call binds against, paired with
    // `offset`; never what a free call is given, since the block outlives any
    // one resource in it.
    pub(super) fn memory(&self) -> vk::DeviceMemory {
        self.memory
    }

    // Where this resource starts inside its block.
    pub(super) fn offset(&self) -> u64 {
        self.offset
    }

    pub(super) fn size(&self) -> u64 {
        self.size
    }

    // A placeholder naming no memory, for a slot filled before its real
    // allocation exists. Freeing it is a no-op.
    pub(super) fn null() -> Self {
        Self {
            memory: vk::DeviceMemory::null(),
            offset: 0,
            size: 0,
            key: PoolKey {
                memory_type: 0,
                kind: ResourceKind::Linear,
                device_address: false,
            },
            placement: Placement {
                block: usize::MAX,
                offset: 0,
            },
        }
    }

    pub(super) fn is_null(&self) -> bool {
        self.memory == vk::DeviceMemory::null()
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

struct Inner {
    pools: HashMap<PoolKey, Pool>,
    // Monotonic frame tick driving the deferred frees. Not the frame-in-flight
    // index, which wraps.
    frame: u64,
    retire_depth: u64,
}

// The allocator behind every persistent buffer and image. See the module
// comment.
pub(super) struct DeviceAllocator {
    inner: RefCell<Inner>,
    memory_props: vk::PhysicalDeviceMemoryProperties,
    max_allocations: u32,
}

impl DeviceAllocator {
    pub(super) fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        frames_in_flight: usize,
    ) -> Self {
        let memory_props =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let max_allocations = unsafe { instance.get_physical_device_properties(physical_device) }
            .limits
            .max_memory_allocation_count;
        Self {
            inner: RefCell::new(Inner {
                pools: HashMap::new(),
                frame: 0,
                // One tick beyond the frames in flight, matching the streamed
                // upload retire discipline: a resource replaced between frames
                // must outlive the submission that was already in flight.
                retire_depth: frames_in_flight as u64 + 1,
            }),
            memory_props,
            max_allocations,
        }
    }

    // Place a resource with `reqs` in memory satisfying `props`. `kind` is the
    // resource's tiling class and `device_address` whether its block must
    // permit `vkGetBufferDeviceAddress`.
    pub(super) fn alloc(
        &self,
        device: &Device,
        reqs: vk::MemoryRequirements,
        props: vk::MemoryPropertyFlags,
        kind: ResourceKind,
        device_address: bool,
    ) -> Result<Allocation, String> {
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
            return Ok(Allocation {
                memory: block.memory,
                offset: placement.offset,
                size: reqs.size,
                key,
                placement,
            });
        }

        let block_bytes = pool.placement.block_size_for(reqs.size, align);
        let (memory, mapped) = Self::create_block(
            device,
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
        Ok(Allocation {
            memory,
            offset: placement.offset,
            size: reqs.size,
            key,
            placement,
        })
    }

    // A pointer to this allocation's own bytes, or null when its memory type is
    // not host-visible. The block is mapped for its whole life, so this neither
    // maps nor needs a matching unmap.
    pub(super) fn mapped_ptr(&self, alloc: &Allocation) -> *mut u8 {
        if alloc.is_null() {
            return std::ptr::null_mut();
        }
        let inner = self.inner.borrow();
        let Some(pool) = inner.pools.get(&alloc.key) else {
            return std::ptr::null_mut();
        };
        match pool
            .blocks
            .get(alloc.placement.block)
            .and_then(Option::as_ref)
        {
            Some(block) if !block.mapped.is_null() => unsafe {
                block.mapped.add(alloc.offset as usize)
            },
            _ => std::ptr::null_mut(),
        }
    }

    // Release `alloc`. Its bytes are withheld until enough frames have ticked
    // that no in-flight command buffer can still reference them, so a caller
    // does not have to reason about whether the GPU is done with it.
    pub(super) fn free(&self, alloc: Allocation) {
        if alloc.is_null() {
            return;
        }
        let mut inner = self.inner.borrow_mut();
        let retire = inner.frame + inner.retire_depth;
        if let Some(pool) = inner.pools.get_mut(&alloc.key) {
            pool.placement.free(alloc.placement, alloc.size, retire);
        }
    }

    // Advance the frame tick, make retired frees placeable again, and release
    // any block that now holds nothing back to the driver.
    pub(super) fn begin_frame(&self, device: &Device) {
        let mut inner = self.inner.borrow_mut();
        inner.frame += 1;
        let frame = inner.frame;
        for pool in inner.pools.values_mut() {
            pool.placement.reclaim(frame);
            for index in pool.placement.take_empty_blocks() {
                if let Some(block) = pool.blocks.get_mut(index).and_then(Option::take) {
                    unsafe { device.free_memory(block.memory, None) };
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

    // Free every block. The caller has already idled the device and destroyed
    // every buffer and image bound into them.
    pub(super) fn destroy(&self, device: &Device) {
        let mut inner = self.inner.borrow_mut();
        for pool in inner.pools.values_mut() {
            for block in pool.blocks.iter_mut().filter_map(Option::take) {
                unsafe { device.free_memory(block.memory, None) };
            }
        }
        inner.pools.clear();
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
    ) -> Result<(vk::DeviceMemory, *mut u8), String> {
        let mut flags_info =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(memory_type);
        if device_address {
            info = info.push_next(&mut flags_info);
        }
        let memory = unsafe { device.allocate_memory(&info, None) }
            .map_err(|e| format!("allocator: block of {size} bytes: {e}"))?;

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
                    return Err(format!("allocator: map block: {e}"));
                }
            }
        } else {
            std::ptr::null_mut()
        };
        Ok((memory, mapped))
    }
}
