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

use crate::memory::counters::Counters;

pub(crate) static COUNTERS: Counters = Counters::new();

/// A `GlobalAlloc` that counts allocations and forwards them to `A`.
pub struct TrackingAlloc<A> {
    inner: A,
    // The block this wrapper counts into, which for an installed allocator is
    // always the process-global one. Naming it rather than reaching for the
    // static is what lets a test weigh a wrapper of its own on its own scale.
    counters: &'static Counters,
}

impl<A> TrackingAlloc<A> {
    /// Wrap `inner` so every allocation through it is counted.
    pub const fn new(inner: A) -> Self {
        Self {
            inner,
            counters: &COUNTERS,
        }
    }

    // A wrapper counting into `counters` instead of the process-global block.
    #[cfg(test)]
    pub(crate) const fn with_counters(inner: A, counters: &'static Counters) -> Self {
        Self { inner, counters }
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
            self.counters.record_alloc(layout.size());
            crate::memory::detail::record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: as `alloc`, forwarding our caller's obligations to `inner`.
        let ptr = unsafe { self.inner.alloc_zeroed(layout) };
        if !ptr.is_null() {
            self.counters.record_alloc(layout.size());
            crate::memory::detail::record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` come from our caller, who guarantees the
        // block was allocated by us with this layout; we allocated it through
        // `inner`, so `inner` is the right allocator to free it.
        unsafe { self.inner.dealloc(ptr, layout) };
        self.counters.record_free(layout.size());
        crate::memory::detail::record_free(layout.size());
    }

    // Forwarded rather than left to the default alloc-copy-dealloc so `inner`
    // can grow a block in place. A failed resize leaves the original block
    // allocated and unchanged, so the counters only move on success.
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `ptr`, `layout` and `new_size` are forwarded unchanged from
        // our caller, who upholds the `GlobalAlloc::realloc` contract.
        let new_ptr = unsafe { self.inner.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            self.counters.record_realloc(layout.size(), new_size);
            crate::memory::detail::record_realloc(layout.size(), new_size);
        }
        new_ptr
    }
}

/// Install [`TrackingAlloc`] over the system allocator as this program's
/// `#[global_allocator]`.
///
/// `#[global_allocator]` is a per-program item, so every binary, test binary,
/// and benchmark that should count its own heap invokes this once at its crate
/// root. A program without it runs on Rust's default allocator and reports no
/// memory at all, which is what [`crate::memory::stats`] returning `None` means.
///
/// ```
/// concinnity_core::install_global_allocator!();
/// ```
#[macro_export]
macro_rules! install_global_allocator {
    () => {
        #[global_allocator]
        static CN_GLOBAL_ALLOC: $crate::memory::TrackingAlloc<std::alloc::System> =
            $crate::memory::TrackingAlloc::new(std::alloc::System);
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::ptr;
    use std::alloc::System;

    const SIZE: usize = 4096;
    const GROWN: usize = 64 * 1024;
    const SHRUNK: usize = 1024;

    fn layout(size: usize) -> Layout {
        Layout::from_size_align(size, 8).expect("valid layout")
    }

    // An inner allocator that always fails, for the paths that must count
    // nothing.
    struct Failing;

    // SAFETY: `alloc` hands out no memory, returning the null `GlobalAlloc`
    // defines as failure, so `dealloc` has no block it can be called with.
    unsafe impl GlobalAlloc for Failing {
        unsafe fn alloc(&self, _layout: Layout) -> *mut u8 {
            ptr::null_mut()
        }

        unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
            unreachable!("this allocator never hands out a block to free")
        }
    }

    // Drives the real wrapper against the real system allocator, counting into
    // a block of this test's own. The process-global counters are a net figure
    // that a free on any other test thread moves between two reads, so they
    // support only a threshold; a block nothing else counts into is exact.
    #[test]
    fn tracking_alloc_forwards_and_counts() {
        static BLOCK: Counters = Counters::new();
        let alloc = TrackingAlloc::with_counters(System, &BLOCK);
        let layout = layout(SIZE);
        assert_eq!(BLOCK.snapshot(), None, "nothing has allocated yet");

        // SAFETY: `layout` has a non-zero size, and the block is freed below
        // through the same allocator and layout it was allocated with.
        let ptr = unsafe { alloc.alloc(layout) };
        assert!(!ptr.is_null(), "system allocator returned null for 4 KiB");
        // SAFETY: `ptr` is a live block of `SIZE` bytes from the line above.
        unsafe { ptr.write_bytes(0xAB, SIZE) };

        let during = BLOCK.snapshot().expect("the wrapper just allocated");
        assert_eq!(during.live_bytes, SIZE as u64);
        assert_eq!(during.alloc_count, 1);
        assert_eq!(during.free_count, 0);

        // SAFETY: `ptr` came from `alloc.alloc(layout)` above and has not been
        // freed or reallocated since.
        unsafe { alloc.dealloc(ptr, layout) };
        let after = BLOCK.snapshot().expect("counters are live");
        assert_eq!(after.live_bytes, 0);
        assert_eq!(after.alloc_count, 1);
        assert_eq!(after.free_count, 1);
        assert_eq!(after.peak_bytes, SIZE as u64);
    }

    // The zeroing path is the inner allocator's, and it counts like any other
    // allocation.
    #[test]
    fn alloc_zeroed_forwards_the_zeroing_and_counts() {
        static BLOCK: Counters = Counters::new();
        let alloc = TrackingAlloc::with_counters(System, &BLOCK);
        let layout = layout(SIZE);

        // SAFETY: `layout` has a non-zero size, and the block is freed below
        // through the same allocator and layout it was allocated with.
        let ptr = unsafe { alloc.alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "system allocator returned null for 4 KiB");
        // SAFETY: `ptr` is a live block of `SIZE` bytes the allocator just
        // initialized, so a shared slice over it is valid.
        let bytes = unsafe { core::slice::from_raw_parts(ptr, SIZE) };
        assert!(bytes.iter().all(|&b| b == 0), "the block was not zeroed");

        let during = BLOCK.snapshot().expect("the wrapper just allocated");
        assert_eq!(during.live_bytes, SIZE as u64);
        assert_eq!(during.alloc_count, 1);

        // SAFETY: `ptr` came from `alloc.alloc_zeroed(layout)` above and has
        // not been freed or reallocated since.
        unsafe { alloc.dealloc(ptr, layout) };
        assert_eq!(BLOCK.snapshot().expect("counters are live").live_bytes, 0);
    }

    // A resize moves live bytes by the delta in either direction, and is
    // neither an allocation nor a free.
    #[test]
    fn realloc_counts_the_resize_only() {
        static BLOCK: Counters = Counters::new();
        let alloc = TrackingAlloc::with_counters(System, &BLOCK);

        // SAFETY: the layout has a non-zero size, and the block is resized and
        // freed below through the same allocator.
        let ptr = unsafe { alloc.alloc(layout(SIZE)) };
        assert!(!ptr.is_null(), "system allocator returned null for 4 KiB");

        // SAFETY: `ptr` came from `alloc.alloc` with `layout(SIZE)`, and
        // `GROWN` is a non-zero size valid for that layout's alignment.
        let ptr = unsafe { alloc.realloc(ptr, layout(SIZE), GROWN) };
        assert!(!ptr.is_null(), "system allocator returned null for 64 KiB");
        let grown = BLOCK.snapshot().expect("the wrapper just allocated");
        assert_eq!(grown.live_bytes, GROWN as u64);
        assert_eq!(grown.alloc_count, 1, "a resize is not an allocation");
        assert_eq!(grown.free_count, 0, "a resize is not a free");

        // SAFETY: the resize above left `ptr` holding `GROWN` bytes at that
        // layout's alignment, and `SHRUNK` is a non-zero size.
        let ptr = unsafe { alloc.realloc(ptr, layout(GROWN), SHRUNK) };
        assert!(!ptr.is_null(), "system allocator returned null for 1 KiB");
        let shrunk = BLOCK.snapshot().expect("counters are live");
        assert_eq!(shrunk.live_bytes, SHRUNK as u64);
        assert_eq!(shrunk.peak_bytes, GROWN as u64);

        // SAFETY: `ptr` holds `SHRUNK` bytes from the resize above.
        unsafe { alloc.dealloc(ptr, layout(SHRUNK)) };
        let after = BLOCK.snapshot().expect("counters are live");
        assert_eq!(after.live_bytes, 0);
        assert_eq!(after.alloc_count, 1);
        assert_eq!(after.free_count, 1);
    }

    // A null return is a block that was never handed out. Counting it would
    // leave live bytes nothing can ever free.
    #[test]
    fn a_failed_allocation_counts_nothing() {
        static BLOCK: Counters = Counters::new();
        let alloc = TrackingAlloc::with_counters(Failing, &BLOCK);

        // SAFETY: the inner allocator returns null without allocating, so
        // there is no block to free.
        assert!(unsafe { alloc.alloc(layout(SIZE)) }.is_null());
        // SAFETY: as above.
        assert!(unsafe { alloc.alloc_zeroed(layout(SIZE)) }.is_null());
        assert_eq!(BLOCK.snapshot(), None, "a failed allocation was counted");
    }
}
