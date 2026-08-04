// concinnity-memory/src/alloc.rs
//
// The `GlobalAlloc` wrapper that counts what the Rust heap is holding.
//
// Installing it is the binary's call, since `#[global_allocator]` is a
// per-program item and an embedder may want an allocator of its own:
//
//     #[global_allocator]
//     static ALLOC: TrackingAlloc<System> = TrackingAlloc::new(System);
//
// Until some binary does, `stats()` reports `None` rather than a zero that would
// read as "the heap is empty".

use core::alloc::{GlobalAlloc, Layout};

use crate::counters::Counters;

pub(crate) static COUNTERS: Counters = Counters::new();

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
// atomics over statically allocated storage, so the allocator cannot re-enter
// itself.
unsafe impl<A: GlobalAlloc> GlobalAlloc for TrackingAlloc<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is forwarded unchanged from our caller, who upholds
        // the `GlobalAlloc::alloc` contract.
        let ptr = unsafe { self.inner.alloc(layout) };
        if !ptr.is_null() {
            COUNTERS.record_alloc(layout.size());
            crate::detail::record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: as `alloc`, forwarding our caller's obligations to `inner`.
        let ptr = unsafe { self.inner.alloc_zeroed(layout) };
        if !ptr.is_null() {
            COUNTERS.record_alloc(layout.size());
            crate::detail::record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` come from our caller, who guarantees the
        // block was allocated by us with this layout; we allocated it through
        // `inner`, so `inner` is the right allocator to free it.
        unsafe { self.inner.dealloc(ptr, layout) };
        COUNTERS.record_free(layout.size());
        crate::detail::record_free(layout.size());
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
            crate::detail::record_realloc(layout.size(), new_size);
        }
        new_ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::System;

    // Drives the real wrapper against the real system allocator. This is the
    // one test that touches the process-global counters, so it asserts on
    // deltas rather than absolute values.
    #[test]
    fn tracking_alloc_forwards_and_counts() {
        let alloc = TrackingAlloc::new(System);
        let layout = Layout::from_size_align(4096, 8).expect("valid layout");

        let before = crate::stats().map(|s| s.live_bytes).unwrap_or(0);
        // SAFETY: `layout` has a non-zero size, and the block is freed below
        // through the same allocator and layout it was allocated with.
        let ptr = unsafe { alloc.alloc(layout) };
        assert!(!ptr.is_null(), "system allocator returned null for 4 KiB");

        let during = crate::stats()
            .expect("the wrapper just allocated")
            .live_bytes;
        assert!(
            during >= before + 4096,
            "live bytes {during} did not rise by the allocation size from {before}"
        );

        // SAFETY: `ptr` came from `alloc.alloc(layout)` above and has not been
        // freed or reallocated since.
        unsafe { alloc.dealloc(ptr, layout) };
        assert!(crate::stats().expect("counters are live").live_bytes >= before);
    }
}
