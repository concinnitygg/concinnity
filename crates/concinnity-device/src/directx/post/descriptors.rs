//! The descriptor slots the shared fullscreen post passes allocate from, in
//! place of the per-pass reservations `init/heap_layout.rs` used to carve for
//! each effect by name.
//!
//! A block of the shader-visible CBV/SRV/UAV heap and a block of the RTV heap,
//! handed out a slot (or a contiguous run of SRV slots) at a time. What the
//! block removes is the per-pass naming: a new post pass takes what it needs
//! from here instead of adding a `<effect>_srv_extra` to the heap cascade.
//!
//! A dropped lease does not free its slots at once. A command list reads a
//! shader-visible descriptor when it executes, so a slot freed while a frame
//! that names it is in flight would let the next target overwrite what that
//! frame samples. The block keeps its own frame tick and holds a freed slot for
//! `frames_in_flight + 1` ticks, the allocator's depth, so a feature rebuilt
//! between frames writes fresh slots while the frames in flight read the old.

use crate::directx::descriptor_slot::SrvSlot;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use windows::Win32::Graphics::Direct3D12::*;

// Descriptor sets the shared passes may hold at once: one per target, plus one
// per level of a target with several, plus one per pooled transient a pass
// writes, plus the RT output and the glass reflection windows. With every pass
// on they hold 38: the indirect-light trace's rings and depth pyramids 14, the
// bloom chain and its pooled top 7, the glass reflection windows 6, SSAO's
// depth copy and its raw and pooled occlusion 3, the temporal resolve's ring 2,
// the disocclusion snapshots 2, the reflection composite 2, the reflection
// target 1 and the RT output 1. A
// rebuilt feature takes fresh slots while its old ones wait out the frames in
// flight, so this leaves room for the widest of those to be held twice.
pub(in crate::directx) const POST_TARGET_SLOTS: usize = 64;

// The occupancy masks are one bit per slot.
const _: () = assert!(POST_TARGET_SLOTS <= u64::BITS as usize);

// The bits of `count` slots from `start`, or none when the run leaves the block.
fn run_mask(start: usize, count: usize) -> u64 {
    if count == 0 || start + count > POST_TARGET_SLOTS {
        return 0;
    }
    let ones = u64::MAX >> (u64::BITS as usize - count);
    ones << start
}

// Which slots are leased, which are waiting out the frames in flight and since
// when, with no D3D12 in it so the timing is testable on its own.
struct SlotTable {
    leased: u64,
    retiring: u64,
    released_at: [u64; POST_TARGET_SLOTS],
    // Monotonic frame tick, advanced once per frame past the fence.
    tick: u64,
    // Ticks a freed slot is withheld for.
    depth: u64,
    // Open idle scopes; while any is, a freed slot returns at once.
    idle_scopes: usize,
}

impl SlotTable {
    fn new(depth: u64) -> Self {
        Self {
            leased: 0,
            retiring: 0,
            released_at: [0; POST_TARGET_SLOTS],
            tick: 0,
            depth,
            idle_scopes: 0,
        }
    }

    // The first run of `count` slots neither leased nor retiring, now leased.
    fn allocate(&mut self, count: usize) -> Option<usize> {
        let busy = self.leased | self.retiring;
        let start = (0..POST_TARGET_SLOTS).find(|&start| {
            let mask = run_mask(start, count);
            mask != 0 && busy & mask == 0
        })?;
        self.leased |= run_mask(start, count);
        Some(start)
    }

    // Free a leased run; its slots come back once `depth` ticks have passed.
    fn release(&mut self, start: usize, count: usize) {
        let mask = run_mask(start, count) & self.leased;
        self.leased &= !mask;
        if self.idle_scopes > 0 {
            return;
        }
        self.retiring |= mask;
        for slot in start..start + count {
            if let Some(at) = self.released_at.get_mut(slot) {
                *at = self.tick;
            }
        }
    }

    // Advance one frame, freeing the slots whose window has closed.
    fn tick(&mut self) {
        self.tick += 1;
        let mut retiring = self.retiring;
        while retiring != 0 {
            let slot = retiring.trailing_zeros() as usize;
            retiring &= retiring - 1;
            if self.released_at[slot] + self.depth <= self.tick {
                self.retiring &= !(1u64 << slot);
            }
        }
    }

    // Return every retiring slot now. Only for a caller that has idled the
    // device, which leaves no frame in flight to read them.
    fn reclaim_idle(&mut self) {
        self.retiring = 0;
    }
}

// The shared table, poison-tolerant: a lease returning its slots must not fail
// because an unrelated thread panicked while holding the lock.
#[derive(Clone)]
struct SharedTable(Arc<Mutex<SlotTable>>);

impl SharedTable {
    fn lock(&self) -> MutexGuard<'_, SlotTable> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

// A held run of slots, released on drop.
struct SlotLease {
    table: SharedTable,
    start: usize,
    count: usize,
}

impl Drop for SlotLease {
    fn drop(&mut self) {
        self.table.lock().release(self.start, self.count);
    }
}

// One post target's descriptors: the shader-visible SRV a consumer samples it
// through, and the RTV the pass writes it through. The slot returns to the
// block, once the frames in flight have retired, when this is dropped.
pub(in crate::directx) struct PostTargetDescriptors {
    pub srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub srv_gpu: SrvSlot,
    pub rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    _lease: SlotLease,
}

// A contiguous run of SRV slots, for a descriptor table spanning several.
pub(in crate::directx) struct PostSrvRun {
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu: SrvSlot,
    srv_size: usize,
    count: usize,
    _lease: SlotLease,
}

impl PostSrvRun {
    // The CPU handle of the run's slot `i`, `None` past its end.
    pub(in crate::directx) fn cpu(&self, i: usize) -> Option<D3D12_CPU_DESCRIPTOR_HANDLE> {
        (i < self.count).then(|| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.srv_cpu.ptr + i * self.srv_size,
        })
    }

    // The shader-visible handle of the run's slot `i`, `None` past its end.
    pub(in crate::directx) fn gpu(&self, i: usize) -> Option<SrvSlot> {
        (i < self.count).then(|| self.srv_gpu.offset(i, self.srv_size))
    }
}

// An open `PostDescriptors::idle_scope`. Closes on drop.
pub(in crate::directx) struct PostIdleScope {
    table: SharedTable,
}

impl Drop for PostIdleScope {
    fn drop(&mut self) {
        let mut table = self.table.lock();
        table.idle_scopes = table.idle_scopes.saturating_sub(1);
    }
}

// The reserved blocks and which of their slots are held.
pub(in crate::directx) struct PostDescriptors {
    srv_cpu_base: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu_base: SrvSlot,
    srv_size: usize,
    rtv_base: D3D12_CPU_DESCRIPTOR_HANDLE,
    rtv_size: usize,
    table: SharedTable,
}

// Where the block's two heaps start and their strides.
#[derive(Clone, Copy)]
pub(in crate::directx) struct PostBlockBase {
    pub srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub srv_gpu: SrvSlot,
    pub srv_size: usize,
    pub rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub rtv_size: usize,
}

impl PostDescriptors {
    // The block starting at `base`, holding a freed slot for
    // `frames_in_flight + 1` frames.
    pub(in crate::directx) fn new(base: PostBlockBase, frames_in_flight: usize) -> Self {
        Self {
            srv_cpu_base: base.srv_cpu,
            srv_gpu_base: base.srv_gpu,
            srv_size: base.srv_size,
            rtv_base: base.rtv,
            rtv_size: base.rtv_size,
            table: SharedTable(Arc::new(Mutex::new(SlotTable::new(
                frames_in_flight as u64 + 1,
            )))),
        }
    }

    // Advance the block's frame tick, returning the slots no frame in flight
    // can still read. Called once per frame, past the frames-in-flight fence.
    pub(in crate::directx) fn begin_frame(&self) {
        self.table.lock().tick();
    }

    // Open a window, for a caller that has just idled the device, in which a
    // freed slot returns at once. Close it before the next frame records.
    pub(in crate::directx) fn idle_scope(&self) -> PostIdleScope {
        let mut table = self.table.lock();
        table.reclaim_idle();
        table.idle_scopes += 1;
        PostIdleScope {
            table: self.table.clone(),
        }
    }

    fn lease(&self, count: usize) -> RenderResult<SlotLease> {
        let start = self.table.lock().allocate(count).ok_or_else(|| {
            RenderError::Other(format!(
                "the shared post passes asked for more than {POST_TARGET_SLOTS} descriptor slots"
            ))
        })?;
        Ok(SlotLease {
            table: self.table.clone(),
            start,
            count,
        })
    }

    // The lowest free slot's descriptors.
    pub(in crate::directx) fn allocate(&self) -> RenderResult<PostTargetDescriptors> {
        let lease = self.lease(1)?;
        let i = lease.start;
        Ok(PostTargetDescriptors {
            srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: self.srv_cpu_base.ptr + i * self.srv_size,
            },
            srv_gpu: self.srv_gpu_base.offset(i, self.srv_size),
            rtv: D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: self.rtv_base.ptr + i * self.rtv_size,
            },
            _lease: lease,
        })
    }

    // The lowest run of `count` contiguous free SRV slots. Their RTVs go unused.
    pub(in crate::directx) fn allocate_srv_run(&self, count: usize) -> RenderResult<PostSrvRun> {
        let lease = self.lease(count)?;
        let i = lease.start;
        Ok(PostSrvRun {
            srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: self.srv_cpu_base.ptr + i * self.srv_size,
            },
            srv_gpu: self.srv_gpu_base.offset(i, self.srv_size),
            srv_size: self.srv_size,
            count,
            _lease: lease,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAMES_IN_FLIGHT: usize = 2;

    fn block() -> PostDescriptors {
        PostDescriptors::new(
            PostBlockBase {
                srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE { ptr: 4096 },
                srv_gpu: SrvSlot::for_test(8192),
                srv_size: 32,
                rtv: D3D12_CPU_DESCRIPTOR_HANDLE { ptr: 512 },
                rtv_size: 16,
            },
            FRAMES_IN_FLIGHT,
        )
    }

    #[test]
    fn slots_are_handed_out_in_order_at_the_heap_stride() {
        let b = block();
        let a = b.allocate().expect("first slot");
        let c = b.allocate().expect("second slot");
        assert_eq!(a.srv_cpu.ptr, 4096);
        assert_eq!(a.srv_gpu, SrvSlot::for_test(8192));
        assert_eq!(a.rtv.ptr, 512);
        assert_eq!(c.srv_cpu.ptr, 4096 + 32);
        assert_eq!(c.srv_gpu, SrvSlot::for_test(8192 + 32));
        assert_eq!(c.rtv.ptr, 512 + 16);
    }

    #[test]
    fn a_slot_freed_at_frame_n_is_reusable_at_n_plus_frames_in_flight_plus_one() {
        let b = block();
        for _ in 0..7 {
            b.begin_frame();
        }
        let freed = b.allocate().expect("slot");
        let slot = freed.srv_gpu;
        drop(freed);
        // Ticks N + 1 ..= N + frames_in_flight: a frame recorded at N may
        // still be reading the slot, so a new target lands elsewhere.
        let mut held = Vec::new();
        for _ in 0..FRAMES_IN_FLIGHT {
            b.begin_frame();
            let d = b.allocate().expect("slot");
            assert_ne!(d.srv_gpu, slot);
            held.push(d);
        }
        drop(held);
        b.begin_frame();
        assert_eq!(b.allocate().expect("slot").srv_gpu, slot);
    }

    #[test]
    fn a_held_slot_is_never_reissued() {
        // Two passes share the block: recreating one must not hand its slot to
        // a target of the other while that target is still live.
        let b = block();
        let taa = [b.allocate().expect("slot"), b.allocate().expect("slot")];
        let ssr = b.allocate().expect("slot");
        let ssr_slot = ssr.srv_gpu;
        drop(taa);
        for _ in 0..=FRAMES_IN_FLIGHT {
            b.begin_frame();
        }
        let rebuilt = [b.allocate().expect("slot"), b.allocate().expect("slot")];
        assert!(rebuilt.iter().all(|d| d.srv_gpu != ssr_slot));
        assert_eq!(
            rebuilt[0].srv_gpu,
            SrvSlot::for_test(8192),
            "the freed slots came back"
        );
        assert_eq!(ssr.srv_gpu, ssr_slot);
    }

    #[test]
    fn opening_an_idle_scope_returns_freed_slots_at_once() {
        // A resize idles the device first, so it gets the same slots back.
        let b = block();
        let first: Vec<_> = (0..3).map(|_| b.allocate().expect("slot")).collect();
        let ptrs: Vec<_> = first.iter().map(|d| d.srv_gpu).collect();
        drop(first);
        drop(b.idle_scope());
        let again: Vec<_> = (0..3).map(|_| b.allocate().expect("slot")).collect();
        assert_eq!(ptrs, again.iter().map(|d| d.srv_gpu).collect::<Vec<_>>());
    }

    #[test]
    fn a_slot_freed_inside_an_idle_scope_returns_at_once() {
        let b = block();
        let slot = {
            let _idle = b.idle_scope();
            let d = b.allocate().expect("slot");
            let slot = d.srv_gpu;
            drop(d);
            assert_eq!(b.allocate().expect("slot").srv_gpu, slot);
            slot
        };
        // Closed: freed slots wait again.
        drop(b.allocate().expect("slot"));
        assert_ne!(b.allocate().expect("slot").srv_gpu, slot);
    }

    #[test]
    fn a_run_is_contiguous_and_skips_a_gap_too_small_for_it() {
        let b = block();
        let _first = b.allocate().expect("slot");
        let gap = b.allocate().expect("slot");
        let _third = b.allocate().expect("slot");
        drop(gap);
        drop(b.idle_scope());
        let run = b.allocate_srv_run(3).expect("run");
        assert_eq!(run.gpu(0), Some(SrvSlot::for_test(8192 + 3 * 32)));
        assert_eq!(run.gpu(2), Some(SrvSlot::for_test(8192 + 5 * 32)));
        assert_eq!(run.cpu(1).map(|h| h.ptr), Some(4096 + 4 * 32));
        assert_eq!(run.gpu(3), None);
        // The one-slot gap is still there for a single target.
        assert_eq!(
            b.allocate().expect("slot").srv_gpu,
            SrvSlot::for_test(8192 + 32)
        );
    }

    #[test]
    fn running_out_of_slots_is_an_error_not_an_overrun() {
        // Past the reservation the next handle would address another feature's
        // descriptors, so the allocation fails instead.
        let b = block();
        let held: Vec<_> = (0..POST_TARGET_SLOTS)
            .map(|_| b.allocate().expect("reserved slot"))
            .collect();
        assert!(b.allocate().is_err());
        assert!(b.allocate_srv_run(2).is_err());
        drop(held);
        assert!(
            b.allocate().is_err(),
            "freed slots wait out the frames in flight"
        );
        drop(b.idle_scope());
        assert!(b.allocate().is_ok());
    }

    #[test]
    fn a_run_that_cannot_fit_the_block_is_refused() {
        assert_eq!(run_mask(POST_TARGET_SLOTS - 1, 2), 0);
        assert_eq!(run_mask(0, 0), 0);
        assert_eq!(run_mask(0, POST_TARGET_SLOTS), u64::MAX);
        let b = block();
        assert!(b.allocate_srv_run(POST_TARGET_SLOTS + 1).is_err());
    }
}
