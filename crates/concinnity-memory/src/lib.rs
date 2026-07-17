// concinnity-memory/src/lib.rs
//
// Allocator-side memory accounting: a `GlobalAlloc` wrapper that counts what
// the Rust heap is holding, and the snapshot type a readout consumes.
//
// This measures the Rust heap, not "the engine". It sees every allocation the
// process makes through Rust -- engine, tools, and third-party crates alike --
// and none of the memory Rust never allocated: GPU driver allocations, mapped
// asset files, thread stacks, and the binary image itself. The gap between
// `MemStats::live_bytes` and the process resident size is that non-Rust
// remainder, not untracked engine waste.
//
// A global allocator runs underneath everything, so nothing on the allocation
// path may allocate. The crate is `no_std` with no dependencies to keep that
// structural: plain relaxed atomics, which cannot re-enter the allocator.
//
// Installing it is the binary's call, since `#[global_allocator]` is a
// per-program item:
//
//     #[global_allocator]
//     static ALLOC: TrackingAlloc<System> = TrackingAlloc::new(System);
//
// Until some binary does, `stats()` reports `None` rather than a zero that
// would read as "the heap is empty".

#![no_std]

#[cfg(test)]
extern crate std;

use core::alloc::{GlobalAlloc, Layout};
use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};

// A snapshot of the tracked heap. Counters are relaxed, so the fields are
// individually accurate but need not agree with each other to the byte under
// concurrent allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemStats {
    // Bytes currently allocated and not yet freed.
    pub live_bytes: u64,
    // High-water mark of `live_bytes` since process start.
    pub peak_bytes: u64,
    // Allocations made since process start. With `free_count`, this is the
    // churn rate; a reallocation resizes an existing block and counts as
    // neither.
    pub alloc_count: u64,
    // Frees made since process start.
    pub free_count: u64,
}

// The counter block behind the global allocator. Split out from `TrackingAlloc`
// so the accounting is testable on its own instance: the allocator itself can
// only ever drive the one process-global block.
pub struct Counters {
    live: AtomicUsize,
    peak: AtomicUsize,
    allocs: AtomicUsize,
    frees: AtomicUsize,
}

impl Counters {
    pub const fn new() -> Self {
        Self {
            live: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            allocs: AtomicUsize::new(0),
            frees: AtomicUsize::new(0),
        }
    }

    fn record_alloc(&self, size: usize) {
        let live = self.live.fetch_add(size, Relaxed) + size;
        self.peak.fetch_max(live, Relaxed);
        self.allocs.fetch_add(1, Relaxed);
    }

    fn record_free(&self, size: usize) {
        self.live.fetch_sub(size, Relaxed);
        self.frees.fetch_add(1, Relaxed);
    }

    // A resize moves `live` by the delta only: the block was already counted at
    // `old_size` and no alloc/free pair reaches the counters.
    fn record_realloc(&self, old_size: usize, new_size: usize) {
        if new_size >= old_size {
            let grew = new_size - old_size;
            let live = self.live.fetch_add(grew, Relaxed) + grew;
            self.peak.fetch_max(live, Relaxed);
        } else {
            self.live.fetch_sub(old_size - new_size, Relaxed);
        }
    }

    // `None` until something allocates through this block, which for the global
    // block means "no binary installed the allocator".
    pub fn snapshot(&self) -> Option<MemStats> {
        let alloc_count = self.allocs.load(Relaxed);
        if alloc_count == 0 {
            return None;
        }
        Some(MemStats {
            live_bytes: self.live.load(Relaxed) as u64,
            peak_bytes: self.peak.load(Relaxed) as u64,
            alloc_count: alloc_count as u64,
            free_count: self.frees.load(Relaxed) as u64,
        })
    }
}

impl Default for Counters {
    fn default() -> Self {
        Self::new()
    }
}

static COUNTERS: Counters = Counters::new();

// The tracked heap as of now, or `None` when no binary installed
// `TrackingAlloc` as its `#[global_allocator]`.
pub fn stats() -> Option<MemStats> {
    COUNTERS.snapshot()
}

// A `GlobalAlloc` that counts allocations and forwards them to `A`.
pub struct TrackingAlloc<A> {
    inner: A,
}

impl<A> TrackingAlloc<A> {
    pub const fn new(inner: A) -> Self {
        Self { inner }
    }
}

// SAFETY: every method forwards to `inner`, which upholds the `GlobalAlloc`
// contract, and returns its pointer unchanged. The counter updates are relaxed
// atomics that never allocate, so the allocator cannot re-enter itself.
unsafe impl<A: GlobalAlloc> GlobalAlloc for TrackingAlloc<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is forwarded unchanged from our caller, who upholds
        // the `GlobalAlloc::alloc` contract.
        let ptr = unsafe { self.inner.alloc(layout) };
        if !ptr.is_null() {
            COUNTERS.record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: as `alloc`, forwarding our caller's obligations to `inner`.
        let ptr = unsafe { self.inner.alloc_zeroed(layout) };
        if !ptr.is_null() {
            COUNTERS.record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` come from our caller, who guarantees the
        // block was allocated by us with this layout; we allocated it through
        // `inner`, so `inner` is the right allocator to free it.
        unsafe { self.inner.dealloc(ptr, layout) };
        COUNTERS.record_free(layout.size());
    }

    // Forwarded rather than left to the default alloc-copy-dealloc so `inner`
    // can grow a block in place. A failed resize leaves the original block
    // allocated and unchanged, so the counters only move on success.
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `ptr`, `layout` and `new_size` are forwarded unchanged from
        // our caller, who upholds the `GlobalAlloc::realloc` contract.
        let new_ptr = unsafe { self.inner.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            COUNTERS.record_realloc(layout.size(), new_size);
        }
        new_ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::System;

    // A fresh block has never seen an allocation, which is how a readout tells
    // "not installed" from "installed and holding nothing".
    #[test]
    fn snapshot_is_none_until_something_allocates() {
        let counters = Counters::new();
        assert_eq!(counters.snapshot(), None);

        counters.record_alloc(64);
        assert!(counters.snapshot().is_some());
    }

    #[test]
    fn alloc_and_free_balance_back_to_zero() {
        let counters = Counters::new();
        counters.record_alloc(1024);
        counters.record_alloc(512);
        counters.record_free(1024);
        counters.record_free(512);

        let stats = counters.snapshot().expect("block has seen allocations");
        assert_eq!(stats.live_bytes, 0);
        assert_eq!(stats.alloc_count, 2);
        assert_eq!(stats.free_count, 2);
    }

    // Peak is a high-water mark: it holds the largest live total ever reached,
    // not the current one.
    #[test]
    fn peak_holds_the_high_water_mark() {
        let counters = Counters::new();
        counters.record_alloc(1000);
        counters.record_alloc(500);
        counters.record_free(1200);

        let stats = counters.snapshot().expect("block has seen allocations");
        assert_eq!(stats.live_bytes, 300);
        assert_eq!(stats.peak_bytes, 1500);
    }

    #[test]
    fn realloc_moves_live_bytes_by_the_delta_only() {
        let counters = Counters::new();
        counters.record_alloc(100);

        counters.record_realloc(100, 400);
        assert_eq!(counters.snapshot().unwrap().live_bytes, 400);

        counters.record_realloc(400, 250);
        let stats = counters.snapshot().unwrap();
        assert_eq!(stats.live_bytes, 250);
        assert_eq!(stats.peak_bytes, 400);
        // The resizes were neither allocations nor frees.
        assert_eq!(stats.alloc_count, 1);
        assert_eq!(stats.free_count, 0);
    }

    // Drives the real wrapper against the real system allocator. This is the
    // one test that touches the process-global counters, so it asserts on
    // deltas rather than absolute values.
    #[test]
    fn tracking_alloc_forwards_and_counts() {
        let alloc = TrackingAlloc::new(System);
        let layout = Layout::from_size_align(4096, 8).expect("valid layout");

        let before = stats().map(|s| s.live_bytes).unwrap_or(0);
        // SAFETY: `layout` has a non-zero size, and the block is freed below
        // through the same allocator and layout it was allocated with.
        let ptr = unsafe { alloc.alloc(layout) };
        assert!(!ptr.is_null(), "system allocator returned null for 4 KiB");

        let during = stats().expect("the wrapper just allocated").live_bytes;
        assert!(
            during >= before + 4096,
            "live bytes {during} did not rise by the allocation size from {before}"
        );

        // SAFETY: `ptr` came from `alloc.alloc(layout)` above and has not been
        // freed or reallocated since.
        unsafe { alloc.dealloc(ptr, layout) };
        assert!(stats().expect("counters are live").live_bytes >= before);
    }
}
