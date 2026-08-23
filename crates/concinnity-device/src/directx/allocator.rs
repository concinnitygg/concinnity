// src/directx/allocator.rs
//
// The device-memory allocator persistent D3D12 resources are placed through.
// Buffers and textures are suballocated out of a few large `ID3D12Heap`s with
// `CreatePlacedResource` instead of each owning a `CreateCommittedResource`
// allocation of its own.
//
// D3D12 caps nothing here the way Vulkan's `maxMemoryAllocationCount` does, so
// this is not a scalability cliff. What it buys is creation cost and
// fragmentation: a committed resource is a kernel-mode video-memory allocation
// and a separately residency-managed object, which a streaming world pays for
// on every texture swap, while a placed resource is a user-mode operation
// against memory its heap already owns.
//
// `block_alloc::BlockAllocator` decides which block and what offset; this file
// is what makes those decisions D3D12. The split is the one `transient_pool.rs`
// and the Metal / Vulkan `allocator.rs` already use: shared placement policy,
// backend-specific binding.
//
// Blocks are separated into pools by heap type, and below resource-heap tier 2
// also by heap class: tier 1 hardware cannot host buffers, RT/DS textures and
// plain textures in one heap, so each gets its own `ALLOW_ONLY_*` pool. Tier 2
// and above collapse the three into one `ALLOW_ALL_BUFFERS_AND_TEXTURES` pool.
//
// Only CPU-written, GPU-read-only resources are placed here. A GPU-written
// placed resource (render target, depth-stencil, UAV) must be re-initialized by
// a Clear / Discard / Copy every time it claims memory, and its compression
// metadata is what makes the aliasing rules bite; those are also a fixed
// handful rather than something that scales with world size. The render graph's
// transients already share memory through `transient_pool.rs`, which does that
// dance properly.
//
// Frees are deferred and leases are RAII. Dropping a `PooledBuffer` /
// `PooledTexture` returns its byte range to the pool tagged with a retire frame
// `FRAMES + 1` ticks out, so the bytes are not handed to another resource until
// no in-flight command list can still reference them. This matters more than it
// did before pooling: a committed resource released early is merely undefined,
// whereas a range released early is placed again almost immediately.
//
// A range handed out after another resource used it is activated with an
// aliasing barrier, submitted on its own one-shot list. Doing it in the
// allocator rather than folding it into a caller's command list means no call
// site has to know whether the range it got was fresh; the submit costs nothing
// in steady state, since a pool only recycles once a lease has dropped and its
// retire frame has passed.
//
// The `Rc` behind the leases is main-thread state. `DxContext` is `Send` and
// the parallel encoder hands workers a `&DxContext`, but a worker only ever
// dereferences a pooled resource to bind it, which touches no refcount; every
// allocation, clone and drop happens on the main thread.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Deref;
use std::rc::{Rc, Weak};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::suballoc::block_alloc::{BlockAllocator, Placement};

// Placement granularity every heap size is rounded to. D3D12 places resources
// at 64 KiB by default, so a block that is not a multiple of it ends in bytes
// nothing can be placed at.
const HEAP_GRANULARITY: u64 = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64;

// Largest block the pool asks for. Big enough that a heavy world holds its
// persistent set in a handful of heaps, small enough that one block is not an
// absurd commitment. A resource too large for one gets a dedicated block sized
// to itself.
const MAX_BLOCK_BYTES: u64 = 64 * 1024 * 1024;

// Size of a pool's first block. Blocks double from here to `MAX_BLOCK_BYTES` as
// a pool fills, so a small world commits megabytes rather than a full-size heap
// for a handful of resources. Smaller than the Metal pool's first block because
// a heap is committed up front and D3D12's 64 KiB placement granularity leaves
// more slack per small resource.
const FIRST_BLOCK_BYTES: u64 = 4 * 1024 * 1024;

// The heap types a pool can be opened for. `CUSTOM` carries its own memory-pool
// and CPU-page properties, which two resources would have to agree on to share
// a heap, and the backend never asks for one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum HeapKind {
    Default,
    Upload,
    Readback,
}

impl HeapKind {
    fn from_d3d12(heap_type: D3D12_HEAP_TYPE) -> Result<Self, String> {
        if heap_type == D3D12_HEAP_TYPE_DEFAULT {
            Ok(Self::Default)
        } else if heap_type == D3D12_HEAP_TYPE_UPLOAD {
            Ok(Self::Upload)
        } else if heap_type == D3D12_HEAP_TYPE_READBACK {
            Ok(Self::Readback)
        } else {
            Err(format!(
                "allocator: heap type {} cannot back a pool",
                heap_type.0
            ))
        }
    }

    fn to_d3d12(self) -> D3D12_HEAP_TYPE {
        match self {
            Self::Default => D3D12_HEAP_TYPE_DEFAULT,
            Self::Upload => D3D12_HEAP_TYPE_UPLOAD,
            Self::Readback => D3D12_HEAP_TYPE_READBACK,
        }
    }
}

// The category of resource a pool's heaps accept. Below resource-heap tier 2 a
// heap may hold only one of the three, so the class joins the pool key; from
// tier 2 up every resource lands in `All` and one heap serves them all.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum HeapClass {
    All,
    Buffers,
    NonRtDsTextures,
    RtDsTextures,
}

impl HeapClass {
    fn for_desc(desc: &D3D12_RESOURCE_DESC, tier: D3D12_RESOURCE_HEAP_TIER) -> Self {
        if tier.0 >= D3D12_RESOURCE_HEAP_TIER_2.0 {
            return Self::All;
        }
        if desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
            return Self::Buffers;
        }
        let rt_ds =
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0;
        if desc.Flags.0 & rt_ds != 0 {
            Self::RtDsTextures
        } else {
            Self::NonRtDsTextures
        }
    }

    fn heap_flags(self) -> D3D12_HEAP_FLAGS {
        match self {
            Self::All => D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES,
            Self::Buffers => D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
            Self::NonRtDsTextures => D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
            Self::RtDsTextures => D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES,
        }
    }
}

// The properties that must match for two resources to share a heap.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct PoolKey {
    kind: HeapKind,
    class: HeapClass,
}

// The blocks of one pool. `placement` names them by index; `heaps` holds the
// matching device allocation, with `None` for a slot whose heap was released.
struct Pool {
    placement: BlockAllocator,
    heaps: Vec<Option<ID3D12Heap>>,
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
            .saturating_mul(1 << self.placement.block_count().min(4))
            .min(MAX_BLOCK_BYTES);
        let needed = size.saturating_add(align.max(1).saturating_sub(1));
        round_up(needed.max(grown), HEAP_GRANULARITY)
    }
}

fn round_up(value: u64, granularity: u64) -> u64 {
    value.div_ceil(granularity).saturating_mul(granularity)
}

// A one-shot list submitted by the allocator itself, held until the GPU has
// provably retired it. Never read; dropping the entry releases the handles.
struct ParkedList {
    #[expect(
        dead_code,
        reason = "held until the GPU retires the list; dropping the entry releases the handle"
    )]
    allocator: ID3D12CommandAllocator,
    #[expect(
        dead_code,
        reason = "held until the GPU retires the list; dropping the entry releases the handle"
    )]
    cmd: ID3D12GraphicsCommandList,
    retire_at: u64,
}

struct Inner {
    pools: HashMap<PoolKey, Pool>,
    parked: Vec<ParkedList>,
    // Monotonic frame tick driving the deferred frees. Not the frame-in-flight
    // index, which wraps.
    frame: u64,
    retire_depth: u64,
}

impl Inner {
    // Return a lease's range to its pool, withheld from reuse until enough
    // frames have ticked that no in-flight command list can reference it.
    fn free(&mut self, key: PoolKey, placement: Placement, size: u64) {
        let retire = self.frame + self.retire_depth;
        if let Some(pool) = self.pools.get_mut(&key) {
            pool.placement.free(placement, size, retire);
        }
    }
}

// What the allocator is holding, for diagnostics and the memory ledger.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct AllocatorStats {
    // Bytes the device has committed across every heap.
    pub reserved_bytes: u64,
    // Bytes live resources occupy. The gap to `reserved_bytes` is alignment
    // padding, fragmentation, and unfilled block tails.
    pub in_use_bytes: u64,
    // Live heaps, i.e. how many device allocations back the pooled set.
    pub block_count: usize,
}

impl std::fmt::Display for AllocatorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} heap(s), {} KiB reserved for {} KiB of resources",
            self.block_count,
            self.reserved_bytes / 1024,
            self.in_use_bytes / 1024,
        )
    }
}

// A reserved range plus the heap to place into, handed from `reserve` to the
// `CreatePlacedResource` calls.
struct Reservation {
    heap: ID3D12Heap,
    key: PoolKey,
    placement: Placement,
    size: u64,
    // The range last belonged to another resource, so the new one must be
    // activated with an aliasing barrier before its first use.
    recycled: bool,
}

// A pooled range's claim on its block. Returns the range when the last holder
// drops, which is what lets a pooled resource be replaced by plain assignment.
// Shared, because D3D12 code hands the same resource to several owners (a ring
// slot plus the live field that reads it, a deduplicated morph buffer shared by
// every draw that sources it); a cloned handle therefore keeps the range alive
// rather than outliving it.
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

// A buffer placed inside a pooled heap. Derefs to the `ID3D12Resource` so
// `Map`, `GetGPUVirtualAddress` and every bind site read exactly as they did on
// a committed allocation.
#[derive(Clone)]
pub(super) struct PooledBuffer {
    resource: ID3D12Resource,
    // A placed resource does not keep its heap alive, and D3D12 requires the
    // heap to outlive it. Held rather than relied on through the lease so a
    // bare `ID3D12Resource` cloned out of this cannot outlive its memory.
    #[expect(
        dead_code,
        reason = "a placed resource does not keep its heap alive, so the heap is held to outlive it"
    )]
    heap: ID3D12Heap,
    _lease: Rc<Lease>,
}

impl Deref for PooledBuffer {
    type Target = ID3D12Resource;

    fn deref(&self) -> &Self::Target {
        &self.resource
    }
}

impl AsRef<ID3D12Resource> for PooledBuffer {
    fn as_ref(&self) -> &ID3D12Resource {
        &self.resource
    }
}

// A texture placed inside a pooled heap. Derefs to the `ID3D12Resource` so
// descriptor writes and copy targets read exactly as they did on a committed
// allocation.
#[derive(Clone)]
pub(super) struct PooledTexture {
    resource: ID3D12Resource,
    #[expect(
        dead_code,
        reason = "a placed resource does not keep its heap alive, so the heap is held to outlive it"
    )]
    heap: ID3D12Heap,
    _lease: Rc<Lease>,
}

impl Deref for PooledTexture {
    type Target = ID3D12Resource;

    fn deref(&self) -> &Self::Target {
        &self.resource
    }
}

impl AsRef<ID3D12Resource> for PooledTexture {
    fn as_ref(&self) -> &ID3D12Resource {
        &self.resource
    }
}

// The allocator behind every pooled buffer and texture. See the module comment.
pub(super) struct DeviceAllocator {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    heap_tier: D3D12_RESOURCE_HEAP_TIER,
    inner: Rc<RefCell<Inner>>,
}

impl DeviceAllocator {
    pub(super) fn new(
        device: &ID3D12Device,
        queue: &ID3D12CommandQueue,
        frames_in_flight: usize,
    ) -> Self {
        Self {
            device: device.clone(),
            queue: queue.clone(),
            heap_tier: resource_heap_tier(device),
            inner: Rc::new(RefCell::new(Inner {
                pools: HashMap::new(),
                parked: Vec::new(),
                frame: 0,
                // One tick beyond the frames in flight, matching the streamed
                // upload retire discipline: a resource replaced between frames
                // must outlive the submission that was already in flight.
                retire_depth: frames_in_flight as u64 + 1,
            })),
        }
    }

    // The device the pooled heaps are created on. Lets a caller that already
    // holds an allocator build the resources that stay committed (render
    // targets, descriptor heaps, pipelines) without carrying a second handle.
    pub(super) fn device(&self) -> &ID3D12Device {
        &self.device
    }

    // The queue the pooled resources' uploads are submitted on.
    pub(super) fn queue(&self) -> &ID3D12CommandQueue {
        &self.queue
    }

    // Place a `size`-byte buffer in `heap_type`. D3D12 ignores `initial_state`
    // for buffers on UPLOAD / READBACK heaps, matching the committed path.
    pub(super) fn alloc_buffer(
        &self,
        size: u64,
        heap_type: D3D12_HEAP_TYPE,
        initial_state: D3D12_RESOURCE_STATES,
    ) -> Result<PooledBuffer, String> {
        let desc = buffer_desc(size.max(1));
        let (resource, heap, lease) = self.place(&desc, heap_type, initial_state)?;
        Ok(PooledBuffer {
            resource,
            heap,
            _lease: lease,
        })
    }

    // Place a texture built from `desc` in `heap_type`. RT/DS/UAV descs are
    // rejected: a GPU-written placed resource needs re-initialization every time
    // it claims memory, which this pool does not do (see the module comment).
    pub(super) fn alloc_texture(
        &self,
        desc: &D3D12_RESOURCE_DESC,
        heap_type: D3D12_HEAP_TYPE,
        initial_state: D3D12_RESOURCE_STATES,
    ) -> Result<PooledTexture, String> {
        if is_gpu_written(desc) {
            return Err(format!(
                "allocator: resource flags {:#x} are GPU-written and stay committed",
                desc.Flags.0
            ));
        }
        let (resource, heap, lease) = self.place(desc, heap_type, initial_state)?;
        Ok(PooledTexture {
            resource,
            heap,
            _lease: lease,
        })
    }

    // Advance the frame tick, make retired frees placeable again, release any
    // heap that now holds nothing, and drop the activation lists the GPU has
    // finished with.
    pub(super) fn begin_frame(&self) {
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
        inner.parked.retain(|p| p.retire_at > frame);
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

    // Reserve a range for `desc`, place the resource in it, and activate the
    // range when another resource used it before.
    fn place(
        &self,
        desc: &D3D12_RESOURCE_DESC,
        heap_type: D3D12_HEAP_TYPE,
        initial_state: D3D12_RESOURCE_STATES,
    ) -> Result<(ID3D12Resource, ID3D12Heap, Rc<Lease>), String> {
        let key = PoolKey {
            kind: HeapKind::from_d3d12(heap_type)?,
            class: HeapClass::for_desc(desc, self.heap_tier),
        };
        let (desc, info) = self.allocation_info(desc)?;
        let reservation = self.reserve(key, info.SizeInBytes, info.Alignment)?;

        let mut placed: Option<ID3D12Resource> = None;
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
        // new COM object lands in a binding that owns it.
        let result = unsafe {
            self.device.CreatePlacedResource(
                &reservation.heap,
                reservation.placement.offset,
                &desc,
                initial_state,
                None,
                &mut placed,
            )
        };
        let resource = match result.map(|()| placed) {
            Ok(Some(resource)) => resource,
            Ok(None) => {
                self.release(reservation);
                return Err("allocator: CreatePlacedResource returned None".to_string());
            }
            Err(e) => {
                self.release(reservation);
                return Err(format!("allocator: place {} bytes: {e}", info.SizeInBytes));
            }
        };

        if reservation.recycled {
            self.activate(&resource)?;
        }
        let heap = reservation.heap.clone();
        Ok((resource, heap, Rc::new(self.lease(reservation))))
    }

    // The size and alignment `desc` needs, preferring the 4 KiB small-resource
    // alignment when the runtime grants it: a 1x1 fallback otherwise burns a
    // whole 64 KiB page of the block. The desc is returned alongside because the
    // granted alignment has to be the one the placement is made with.
    fn allocation_info(
        &self,
        desc: &D3D12_RESOURCE_DESC,
    ) -> Result<(D3D12_RESOURCE_DESC, D3D12_RESOURCE_ALLOCATION_INFO), String> {
        let mut standard = *desc;
        standard.Alignment = 0;
        // SAFETY: a query on a live COM object; the descriptor it reads and the out-parameters it
        // fills are live locals that outlive the call.
        let info = unsafe { self.device.GetResourceAllocationInfo(0, &[standard]) };
        if info.SizeInBytes == u64::MAX {
            return Err("allocator: resource has no valid allocation size".to_string());
        }
        // Only a resource whose whole footprint already fits one page can
        // qualify. Asking about a larger one is refused, and the debug layer
        // says so at length, so the standard size is the gate.
        if small_alignment_eligible(desc) && info.SizeInBytes <= HEAP_GRANULARITY {
            let mut small = *desc;
            small.Alignment = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT as u64;
            // SAFETY: a query on a live COM object; the descriptor it reads and the out-parameters
            // it fills are live locals that outlive the call.
            let small_info = unsafe { self.device.GetResourceAllocationInfo(0, &[small]) };
            if small_info.Alignment == D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT as u64
                && small_info.SizeInBytes != u64::MAX
            {
                return Ok((small, small_info));
            }
        }
        standard.Alignment = info.Alignment;
        Ok((standard, info))
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
                recycled: true,
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
            // A block the pool just created has never held a resource.
            recycled: false,
        })
    }

    // Claim memory another resource used before: an aliasing barrier on its own
    // one-shot list, ordered ahead of every later submission by the in-order
    // queue. The list is parked until the GPU retires it.
    fn activate(&self, resource: &ID3D12Resource) -> Result<(), String> {
        let (allocator, cmd) =
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            super::texture::one_shot_submit_nowait(&self.device, &self.queue, |cmd| unsafe {
                cmd.ResourceBarrier(&[super::texture::aliasing_barrier(resource)]);
            })?;
        let mut inner = self.inner.borrow_mut();
        let retire_at = inner.frame + inner.retire_depth;
        inner.parked.push(ParkedList {
            allocator,
            cmd,
            retire_at,
        });
        Ok(())
    }

    fn lease(&self, reservation: Reservation) -> Lease {
        Lease {
            owner: Rc::downgrade(&self.inner),
            key: reservation.key,
            placement: reservation.placement,
            size: reservation.size,
        }
    }

    // Hand a reservation back when the placement it was made for failed.
    // Retired immediately: no command list ever saw a resource there.
    fn release(&self, reservation: Reservation) {
        let mut inner = self.inner.borrow_mut();
        if let Some(pool) = inner.pools.get_mut(&reservation.key) {
            pool.placement
                .free(reservation.placement, reservation.size, 0);
        }
    }
}

// The resource desc for a pooled buffer, matching what the committed path used
// to build.
fn buffer_desc(size: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    }
}

// Whether the GPU writes this resource, which is what keeps it off the pool.
fn is_gpu_written(desc: &D3D12_RESOURCE_DESC) -> bool {
    let written = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0
        | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0
        | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0;
    desc.Flags.0 & written != 0
}

// Whether `desc` may be worth asking for the 4 KiB small-resource alignment.
// D3D12 grants it only to single-sample, non-RT/DS textures whose whole mip
// chain fits the smaller alignment, and never to buffers.
fn small_alignment_eligible(desc: &D3D12_RESOURCE_DESC) -> bool {
    desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
        && desc.SampleDesc.Count <= 1
        && !is_gpu_written(desc)
}

// The device's resource-heap tier, which decides whether buffers and textures
// may share a heap. Anything the query cannot answer is treated as tier 1, the
// restrictive case.
fn resource_heap_tier(device: &ID3D12Device) -> D3D12_RESOURCE_HEAP_TIER {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    // SAFETY: a query on a live COM object; the descriptor it reads and the out-parameters it fills
    // are live locals that outlive the call.
    let ok = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS,
            &mut options as *mut _ as *mut std::ffi::c_void,
            std::mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS>() as u32,
        )
    };
    if ok.is_ok() {
        options.ResourceHeapTier
    } else {
        D3D12_RESOURCE_HEAP_TIER_1
    }
}

// One placement heap of `size` bytes for `key`'s pool.
fn new_heap(device: &ID3D12Device, key: PoolKey, size: u64) -> Result<ID3D12Heap, String> {
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: size.max(HEAP_GRANULARITY),
        Properties: D3D12_HEAP_PROPERTIES {
            Type: key.kind.to_d3d12(),
            ..Default::default()
        },
        Alignment: HEAP_GRANULARITY,
        Flags: key.class.heap_flags(),
    };
    let mut heap: Option<ID3D12Heap> = None;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe { device.CreateHeap(&desc, &mut heap) }
        .map_err(|e| format!("allocator: create a {size}-byte heap: {e}"))?;
    heap.ok_or_else(|| "allocator: CreateHeap returned None".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;

    fn device() -> Option<ID3D12Device> {
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
        // new COM object lands in a binding that owns it.
        unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut device) }.ok()?;
        device
    }

    fn allocator() -> Option<DeviceAllocator> {
        let device = device()?;
        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        };
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
        // new COM object lands in a binding that owns it.
        let queue: ID3D12CommandQueue = unsafe { device.CreateCommandQueue(&queue_desc) }.ok()?;
        Some(DeviceAllocator::new(&device, &queue, 3))
    }

    fn texture_desc(width: u32) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: width as u64,
            Height: width,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        }
    }

    #[test]
    fn pooled_heap_types_round_trip() {
        for kind in [HeapKind::Default, HeapKind::Upload, HeapKind::Readback] {
            assert_eq!(HeapKind::from_d3d12(kind.to_d3d12()), Ok(kind));
        }
    }

    #[test]
    fn custom_heaps_are_rejected() {
        // A CUSTOM heap carries memory-pool and CPU-page properties the pool key
        // does not model, so it must fail at the allocation rather than
        // silently sharing a heap with mismatched properties.
        assert!(HeapKind::from_d3d12(D3D12_HEAP_TYPE_CUSTOM).is_err());
    }

    #[test]
    fn tier_one_separates_the_three_heap_classes() {
        // A tier 1 heap holds only one category, so a buffer, a plain texture
        // and a render target each need their own pool.
        let buffer = buffer_desc(1024);
        let plain = texture_desc(64);
        let mut target = texture_desc(64);
        target.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        let mut depth = texture_desc(64);
        depth.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

        let tier = D3D12_RESOURCE_HEAP_TIER_1;
        assert_eq!(HeapClass::for_desc(&buffer, tier), HeapClass::Buffers);
        assert_eq!(
            HeapClass::for_desc(&plain, tier),
            HeapClass::NonRtDsTextures
        );
        assert_eq!(HeapClass::for_desc(&target, tier), HeapClass::RtDsTextures);
        assert_eq!(HeapClass::for_desc(&depth, tier), HeapClass::RtDsTextures);
    }

    #[test]
    fn tier_two_collapses_every_class_into_one_pool() {
        // From tier 2 up one heap serves all three categories, so buffers and
        // textures share blocks instead of opening a pool each.
        let tier = D3D12_RESOURCE_HEAP_TIER_2;
        assert_eq!(
            HeapClass::for_desc(&buffer_desc(1024), tier),
            HeapClass::All
        );
        assert_eq!(HeapClass::for_desc(&texture_desc(64), tier), HeapClass::All);
        assert_eq!(
            HeapClass::All.heap_flags(),
            D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES
        );
    }

    #[test]
    fn each_class_asks_for_the_heap_flag_its_tier_one_pool_needs() {
        assert_eq!(
            HeapClass::Buffers.heap_flags(),
            D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS
        );
        assert_eq!(
            HeapClass::NonRtDsTextures.heap_flags(),
            D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES
        );
        assert_eq!(
            HeapClass::RtDsTextures.heap_flags(),
            D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES
        );
    }

    #[test]
    fn gpu_written_descs_stay_committed() {
        // The pool never re-initializes a range it hands out, so anything the
        // GPU writes has to be refused rather than silently placed.
        for flag in [
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ] {
            let mut desc = texture_desc(64);
            desc.Flags = flag;
            assert!(is_gpu_written(&desc), "{:#x}", flag.0);
            assert!(!small_alignment_eligible(&desc));
        }
        assert!(!is_gpu_written(&texture_desc(64)));
        assert!(!is_gpu_written(&buffer_desc(1024)));
    }

    #[test]
    fn only_plain_single_sample_textures_ask_for_small_alignment() {
        assert!(small_alignment_eligible(&texture_desc(64)));
        // D3D12 never grants the small alignment to a buffer or to MSAA.
        assert!(!small_alignment_eligible(&buffer_desc(1024)));
        let mut msaa = texture_desc(64);
        msaa.SampleDesc.Count = 4;
        assert!(!small_alignment_eligible(&msaa));
    }

    #[test]
    fn blocks_grow_from_the_first_size_up_to_the_cap() {
        // A small world should not commit a full-size heap for a few resources,
        // so the first blocks are small and the size doubles as the pool fills.
        let mut pool = Pool::new();
        let mut sizes = Vec::new();
        for _ in 0..6 {
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
                FIRST_BLOCK_BYTES * 8,
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
        let bytes = pool.next_block_bytes(huge, HEAP_GRANULARITY);
        assert!(bytes >= huge);
        assert_eq!(bytes % HEAP_GRANULARITY, 0);
    }

    #[test]
    fn block_sizes_stay_on_the_placement_granularity() {
        // A block that is not a multiple of 64 KiB ends in bytes no resource can
        // be placed at.
        let pool = Pool::new();
        for size in [1u64, 100, HEAP_GRANULARITY + 1, MAX_BLOCK_BYTES * 2 + 7] {
            assert_eq!(pool.next_block_bytes(size, 256) % HEAP_GRANULARITY, 0);
        }
    }

    #[test]
    fn the_small_alignment_is_asked_for_only_when_it_can_be_granted() {
        let Some(alloc) = allocator() else {
            return;
        };
        // 64x64 RGBA8 is 16 KiB, inside one page, so it costs a tile instead of
        // the page a committed resource would take.
        let (desc, info) = alloc
            .allocation_info(&texture_desc(64))
            .expect("small texture sizes");
        assert_eq!(
            info.Alignment,
            D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT as u64
        );
        assert_eq!(desc.Alignment, info.Alignment, "placement must match");

        // 512x512 RGBA8 is 1 MiB. Asking about the small alignment for it is
        // refused, and the debug layer says so at length, so it must not be
        // asked.
        let (desc, info) = alloc
            .allocation_info(&texture_desc(512))
            .expect("large texture sizes");
        assert_eq!(info.Alignment, HEAP_GRANULARITY);
        assert_eq!(desc.Alignment, info.Alignment, "placement must match");
    }

    #[test]
    fn many_buffers_share_few_heaps() {
        let Some(alloc) = allocator() else {
            return;
        };
        let buffers: Vec<PooledBuffer> = (0..512)
            .map(|_| {
                alloc
                    .alloc_buffer(
                        4096,
                        D3D12_HEAP_TYPE_UPLOAD,
                        D3D12_RESOURCE_STATE_GENERIC_READ,
                    )
                    .expect("upload buffer places")
            })
            .collect();
        let stats = alloc.stats();
        // 512 buffers at D3D12's 64 KiB buffer granularity is 32 MiB, which the
        // growing block sizes hold in a handful of heaps rather than 512
        // committed allocations.
        assert!(stats.block_count <= 8, "{stats:?}");
        assert!(stats.in_use_bytes >= 512 * HEAP_GRANULARITY, "{stats:?}");
        assert!(stats.reserved_bytes >= stats.in_use_bytes, "{stats:?}");
        drop(buffers);
    }

    #[test]
    fn placed_buffers_get_distinct_non_overlapping_storage() {
        let Some(alloc) = allocator() else {
            return;
        };
        let write = |buffer: &PooledBuffer, byte: u8| {
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { buffer.Map(0, None, Some(&mut ptr)) }.expect("upload buffer maps");
            // SAFETY: the map covers the whole 256-byte buffer.
            unsafe { std::ptr::write_bytes(ptr as *mut u8, byte, 256) };
        };
        let read = |buffer: &PooledBuffer| {
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { buffer.Map(0, None, Some(&mut ptr)) }.expect("upload buffer maps");
            // SAFETY: as above.
            unsafe { std::slice::from_raw_parts(ptr as *const u8, 256).to_vec() }
        };
        let a = alloc
            .alloc_buffer(
                256,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )
            .expect("upload buffer places");
        let b = alloc
            .alloc_buffer(
                256,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )
            .expect("upload buffer places");
        write(&a, 0xAA);
        write(&b, 0x55);
        // Overlapping ranges are the failure mode placement makes possible, so
        // check the contents rather than just the offsets.
        assert!(read(&a).iter().all(|&x| x == 0xAA));
        assert!(read(&b).iter().all(|&x| x == 0x55));
    }

    #[test]
    fn a_dropped_lease_is_withheld_until_its_retire_frame() {
        let Some(alloc) = allocator() else {
            return;
        };
        let first = alloc
            .alloc_buffer(
                4096,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )
            .expect("upload buffer places");
        assert_eq!(alloc.stats().block_count, 1);
        drop(first);
        // The bytes stop counting as in use at once, but a frame that may still
        // reference them has to retire before they are placed again.
        assert_eq!(alloc.stats().in_use_bytes, 0);
        assert_eq!(alloc.stats().block_count, 1);
        for _ in 0..5 {
            alloc.begin_frame();
        }
        let stats = alloc.stats();
        assert_eq!(stats.block_count, 0, "emptied heap is released");
        assert_eq!(stats.reserved_bytes, 0);
    }

    #[test]
    fn a_texture_larger_than_the_cap_gets_its_own_heap() {
        let Some(alloc) = allocator() else {
            return;
        };
        // 4096x4096 RGBA8 is 64 MiB, at the standard block cap; 8192 wide is
        // four times that, so it cannot share a block.
        let big = alloc
            .alloc_texture(
                &texture_desc(8192),
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )
            .expect("oversized texture places");
        let small = alloc
            .alloc_texture(
                &texture_desc(64),
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )
            .expect("small texture places");
        // The dedicated block is never shared, so the small texture opens a
        // standard block instead of stranding the remainder of the huge one.
        assert_eq!(alloc.stats().block_count, 2, "{:?}", alloc.stats());
        drop((big, small));
    }

    #[test]
    fn gpu_written_textures_are_refused_by_the_pool() {
        let Some(alloc) = allocator() else {
            return;
        };
        let mut desc = texture_desc(64);
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        assert!(
            alloc
                .alloc_texture(
                    &desc,
                    D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_STATE_RENDER_TARGET
                )
                .is_err()
        );
    }

    #[test]
    fn a_recycled_range_is_handed_out_again_after_it_retires() {
        let Some(alloc) = allocator() else {
            return;
        };
        let first = alloc
            .alloc_buffer(
                4096,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )
            .expect("upload buffer places");
        // Keep the block alive so the free is recycled rather than released
        // with its heap.
        let keep = alloc
            .alloc_buffer(
                4096,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )
            .expect("upload buffer places");
        drop(first);
        for _ in 0..5 {
            alloc.begin_frame();
        }
        let reused = alloc
            .alloc_buffer(
                4096,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )
            .expect("recycled range places and activates");
        // The pool reused the retired range instead of growing, and the
        // activation submit left the pool in one block.
        assert_eq!(alloc.stats().block_count, 1, "{:?}", alloc.stats());
        drop((keep, reused));
    }
}
