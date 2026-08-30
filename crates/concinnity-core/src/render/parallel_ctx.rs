//! Generic Send/Sync shim for parallel per-pass command recording, shared by all
//! three backend executors (`{metal,directx,vulkan}/graph_exec.rs`). Each backend
//! fans its non-composite render-graph passes onto worker threads; every worker
//! records into its own command buffer/list and reaches the immutable subset of
//! the backend context it needs through a `ParallelCtxRef`.
//!
//! The backend context types are not Send/Sync in Rust's type system: they hold
//! COM smart pointers, objc2 protocol objects, RefCells, and the like. The
//! graphics APIs nonetheless permit shared, read-only access to device-derived
//! resources from many threads. A backend adopts that claim for its own context
//! type with a single `unsafe impl ParallelEncodeCtx`, where it documents the
//! audit of every interior-mutable field reachable during encode. The Send/Sync
//! impls on `ParallelCtxRef` below are then keyed off that marker, so the unsafe
//! reasoning lives in one auditable place per backend instead of being repeated
//! on three structurally identical wrapper types.

/// Marker for a backend context that may be shared, read-only, across the
/// parallel-encode worker fan-out.
///
/// # Safety
///
/// Implementing this is a claim that concurrent `&Self` access during command
/// recording is sound: the graphics API allows shared read of device-derived
/// resources, and every interior-mutable field reachable during encode is
/// either atomic or hoisted out of the fan-out before it begins. Each backend's
/// impl carries that audit (see the module docs in each
/// `*/parallel_encoder.rs`).
pub unsafe trait ParallelEncodeCtx {}

/// A handle to a `&'a T` borrow that is Send + Sync when `T: ParallelEncodeCtx`.
/// Worker closures use it to reach the immutable subset of the backend context
/// they need while recording commands into their own command buffer/list.
///
/// The borrow is held directly, so its lifetime is enforced by the type. The
/// wrapper is only used inside each backend's parallel-encoder fan-out in
/// `graph_exec.rs`, which joins all workers before the outer borrow returns. The
/// Send/Sync claim rests entirely on the `T: ParallelEncodeCtx` marker.
pub struct ParallelCtxRef<'a, T> {
    inner: &'a T,
}

impl<'a, T> ParallelCtxRef<'a, T> {
    /// Wrap `ctx` so worker threads can share it.
    pub fn new(ctx: &'a T) -> Self {
        Self { inner: ctx }
    }

    /// Borrow the wrapped context.
    pub fn as_ctx(&self) -> &T {
        self.inner
    }
}

impl<T> Clone for ParallelCtxRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for ParallelCtxRef<'_, T> {}

// SAFETY: `T: ParallelEncodeCtx` is the backend's audited claim that shared,
// read-only `&T` access across the encode fan-out is sound. The wrapper only
// ever hands out `&T` (via `as_ctx`), so Send + Sync follow from that claim.
unsafe impl<T: ParallelEncodeCtx> Send for ParallelCtxRef<'_, T> {}
// SAFETY: as for `Send` above -- the wrapper only ever hands out `&T`.
unsafe impl<T: ParallelEncodeCtx> Sync for ParallelCtxRef<'_, T> {}

#[cfg(test)]
mod tests {
    use super::*;

    struct Ctx(u32);

    // SAFETY: a plain integer with no interior mutability, so concurrent `&Ctx`
    // access during the fan-out is trivially sound.
    unsafe impl ParallelEncodeCtx for Ctx {}

    // The wrapper hands back the borrow it was built from, and copying one
    // hands back the same borrow: workers each hold their own copy of the
    // handle and reach one shared context through it.
    #[test]
    fn every_copy_of_the_handle_reaches_the_same_context() {
        let ctx = Ctx(7);
        let handle = ParallelCtxRef::new(&ctx);
        let copied = handle;
        // Spelled through the trait: the wrapper is `Copy`, so method-call
        // syntax would take the copy path and never reach this impl.
        let cloned = Clone::clone(&handle);
        assert_eq!(handle.as_ctx().0, 7);
        assert_eq!(copied.as_ctx().0, 7);
        assert_eq!(cloned.as_ctx().0, 7);
        assert!(core::ptr::eq(handle.as_ctx(), cloned.as_ctx()));
    }

    // The Send/Sync claim is what lets a worker thread hold one, so borrow a
    // handle across a scoped thread to prove the impls are in force.
    #[test]
    fn a_handle_crosses_a_thread_boundary() {
        let ctx = Ctx(3);
        let handle = ParallelCtxRef::new(&ctx);
        std::thread::scope(|scope| {
            let worker = scope.spawn(move || handle.as_ctx().0);
            assert_eq!(worker.join().expect("the worker finished"), 3);
        });
    }
}
