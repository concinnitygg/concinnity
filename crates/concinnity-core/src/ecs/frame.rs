// src/ecs/frame.rs
//
// Per-frame state handed to every system, carried as one field on
// `PipelineContext` so later frame-scoped facilities arrive without touching
// every system and every context construction again.

use concinnity_memory::{Arena, ArenaVec};

/// Frame-scoped facilities a system may use for the duration of its `step`.
///
/// The scratch arena is reset by the frame loop, so nothing allocated from it
/// may outlive the `step` that allocated it. That is enforced rather than
/// documented: `reset` takes `&mut Arena`, so the borrow checker will not let
/// the loop reclaim the arena while any allocation from it is still live.
///
/// Take the arena out before using the rest of the context:
///
/// ```ignore
/// let scratch = ctx.frame.scratch;   // an &Arena that outlives this borrow
/// let mut ids = scratch.vec::<Entity>(count)?;
/// // `ctx` is still free to be used mutably here
/// ```
///
/// The copy matters. `scratch` is a shared reference with the context's own
/// lifetime, so copying it out detaches it from the borrow of `ctx` and leaves
/// the context usable. Reaching through `ctx.frame.scratch` at each call site
/// instead would hold `ctx` borrowed for as long as the allocation lives.
#[derive(Clone, Copy)]
pub struct FrameContext<'a> {
    /// Bump scratch for temporaries that do not outlive this frame. Returns
    /// `None` when full, which is the caller's cue to fall back to the heap;
    /// the frame loop reports that it happened rather than absorbing it.
    pub scratch: &'a Arena,
}

impl<'a> FrameContext<'a> {
    pub fn new(scratch: &'a Arena) -> Self {
        Self { scratch }
    }

    /// Gather `items` into frame scratch, falling back to the heap if the
    /// reserve is exhausted.
    ///
    /// This is the shape most frame temporaries take: a system reads something
    /// out of the context, needs the context mutably to act on it, and so has
    /// to copy the values out first to end the borrow. That copy is what the
    /// arena is for.
    /// Reserves from the iterator's upper size bound, so anything that knows
    /// how much it can yield qualifies -- not just `ExactSizeIterator`. An
    /// unbounded iterator goes to the heap rather than guessing a reservation.
    pub fn collect<T, I>(&self, items: I) -> FrameVec<'a, T>
    where
        T: Copy,
        I: IntoIterator<Item = T>,
    {
        let items = items.into_iter();
        let reservation = items
            .size_hint()
            .1
            .and_then(|upper| self.scratch.vec::<T>(upper));
        match reservation {
            Some(mut out) => {
                out.extend(items);
                FrameVec::Scratch(out)
            }
            None => FrameVec::Heap(items.collect()),
        }
    }
}

/// A frame temporary: in the scratch arena when it fit, on the heap when it did
/// not. Reads as `&[T]` either way, so a caller never branches on which it got.
///
/// The heap arm is a correctness fallback, not a failure. The arena counts the
/// decline and the frame loop reports it, so an undersized reserve surfaces
/// instead of quietly costing allocations again.
pub enum FrameVec<'a, T: Copy> {
    Scratch(ArenaVec<'a, T>),
    Heap(Vec<T>),
}

impl<T: Copy> core::ops::Deref for FrameVec<'_, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match self {
            FrameVec::Scratch(v) => v,
            FrameVec::Heap(v) => v,
        }
    }
}

impl<'v, T: Copy> IntoIterator for &'v FrameVec<'_, T> {
    type Item = &'v T;
    type IntoIter = core::slice::Iter<'v, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gather_that_fits_lands_in_scratch() {
        let arena = Arena::with_capacity(4096);
        let frame = FrameContext::new(&arena);

        let out = frame.collect([1u32, 2, 3]);
        assert!(matches!(out, FrameVec::Scratch(_)));
        assert_eq!(&*out, &[1, 2, 3]);
        assert_eq!(arena.overflows(), 0);
        assert!(arena.used() > 0, "it came out of the reserve");
    }

    // Running out is a fallback, not a failure: the values still arrive, and
    // the decline is recorded so the frame loop can report the reserve is small.
    #[test]
    fn a_gather_too_large_falls_back_to_the_heap_and_is_recorded() {
        let arena = Arena::with_capacity(8);
        let frame = FrameContext::new(&arena);

        let out = frame.collect([1u64, 2, 3, 4]);
        assert!(matches!(out, FrameVec::Heap(_)));
        assert_eq!(&*out, &[1, 2, 3, 4], "the fallback holds the same values");
        assert_eq!(arena.overflows(), 1);
    }

    // Callers read a slice and never branch on where it came from.
    #[test]
    fn both_arms_read_the_same_way() {
        let roomy = Arena::with_capacity(4096);
        let tight = Arena::with_capacity(0);
        let items = [7u16, 8, 9];

        let from_scratch = FrameContext::new(&roomy).collect(items);
        let from_heap = FrameContext::new(&tight).collect(items);

        assert_eq!(&*from_scratch, &*from_heap);
        assert_eq!(from_scratch.len(), 3);
        assert_eq!(from_heap.iter().copied().sum::<u16>(), 24);
        for (a, b) in (&from_scratch).into_iter().zip(&from_heap) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn an_empty_gather_costs_nothing() {
        let arena = Arena::with_capacity(4096);
        let out = FrameContext::new(&arena).collect([0u8; 0]);
        assert!(out.is_empty());
        assert_eq!(arena.overflows(), 0);
    }

    // The context is Copy so a system can take it out and keep using the
    // pipeline context mutably; both copies must name the same arena.
    #[test]
    fn a_copied_context_shares_one_reserve() {
        let arena = Arena::with_capacity(4096);
        let frame = FrameContext::new(&arena);
        let copy = frame;

        let _a = frame.collect([1u32; 4]);
        let used = arena.used();
        let _b = copy.collect([2u32; 4]);
        assert!(arena.used() > used, "the copy drew from the same reserve");
    }
}
