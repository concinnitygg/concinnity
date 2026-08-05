// concinnity-memory/src/arena.rs
//
// A linear (bump) allocator for working memory that is thrown away wholesale.
//
// Per-frame scratch is the case it exists for: a list built to be walked once
// and dropped costs a heap allocation and a free every frame, and a thousand of
// those is a thousand trips through the global allocator plus the churn they
// leave behind. An arena hands out slices of one buffer by moving a cursor, and
// `reset` gives the whole frame back at once.
//
// Everything stored must be `Copy`. The arena never runs a destructor -- `reset`
// only rewinds the cursor -- so a type that owns anything would silently leak
// what it owns. `Copy` makes that a compile error instead of a slow leak.
//
// `alloc` takes `&self` while `reset` takes `&mut self`: handing out memory is
// what a frame does from many call sites, and giving it back is only legal once
// every borrow has ended, which is exactly what the borrow checker already
// proves.

use core::alloc::Layout;
use core::cell::Cell;
use core::mem::MaybeUninit;
use core::ptr::NonNull;

use crate::tag::{MemTag, Realm};

// The backing buffer's alignment, and so the strongest alignment the arena can
// satisfy. Covers everything up to a cache line, which is past any vector type
// the engine stores.
const ARENA_ALIGN: usize = 64;

pub struct Arena {
    ptr: NonNull<u8>,
    cap: usize,
    used: Cell<usize>,
    peak: Cell<usize>,
    // Requests this arena could not satisfy. A caller falling back to the heap
    // is correct but means the reserve is too small, and a silent fallback
    // reads as "the arena is sized right" when it is not.
    overflows: Cell<u32>,
    // Where the reservation is accounted, for as long as the arena lives.
    tag: Option<MemTag>,
}

// Handing out `&mut` from `&self` is the arena's design, not an oversight:
// allocations never overlap (the cursor only moves forward) and the memory
// cannot be reclaimed while one is borrowed (`reset` takes `&mut self`). The
// lint is right about the general case and wrong about this one.
#[allow(clippy::mut_from_ref)]
impl Arena {
    // Reserve `bytes` up front. The buffer is taken from the global allocator
    // once and held until the arena drops; nothing here allocates again.
    pub fn with_capacity(bytes: usize) -> Self {
        Self::new(bytes, None)
    }

    // As `with_capacity`, reporting the reservation under `tag` in host memory
    // for as long as the arena lives. An arena's whole cost is its reservation,
    // so it can account for itself rather than making its owner do it.
    pub fn tagged(bytes: usize, tag: MemTag) -> Self {
        crate::ledger().add(tag, Realm::Host, bytes as u64);
        Self::new(bytes, Some(tag))
    }

    fn new(bytes: usize, tag: Option<MemTag>) -> Self {
        if bytes == 0 {
            return Self {
                ptr: NonNull::dangling(),
                cap: 0,
                used: Cell::new(0),
                peak: Cell::new(0),
                overflows: Cell::new(0),
                tag,
            };
        }
        let layout = Layout::from_size_align(bytes, ARENA_ALIGN).expect("arena layout");
        // SAFETY: `layout` has a non-zero size.
        let ptr = unsafe { alloc::alloc::alloc(layout) };
        let Some(ptr) = NonNull::new(ptr) else {
            alloc::alloc::handle_alloc_error(layout)
        };
        Self {
            ptr,
            cap: bytes,
            used: Cell::new(0),
            peak: Cell::new(0),
            overflows: Cell::new(0),
            tag,
        }
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    pub fn used(&self) -> usize {
        self.used.get()
    }

    pub fn remaining(&self) -> usize {
        self.cap - self.used.get()
    }

    // The most this arena has held between resets: what to size it from.
    pub fn peak(&self) -> usize {
        self.peak.get()
    }

    // Requests declined since the last `clear_overflows`. Non-zero means
    // callers fell back to the heap and the reserve wants raising.
    pub fn overflows(&self) -> u32 {
        self.overflows.get()
    }

    pub fn clear_overflows(&self) {
        self.overflows.set(0);
    }

    // Give back everything handed out. Taking `&mut self` is the safety
    // argument: no allocation from this arena can still be borrowed.
    //
    // Deliberately leaves `peak` and `overflows` alone: both describe the worst
    // frame so far, which is what sizes the reserve, and a per-frame reset would
    // erase exactly the evidence they exist to carry.
    pub fn reset(&mut self) {
        self.used.set(0);
    }

    // Move `value` into the arena. `None` when the arena is full, which is the
    // caller's cue to fall back to the heap rather than a failure.
    pub fn alloc<T: Copy>(&self, value: T) -> Option<&mut T> {
        let ptr = self.bump(size_of::<T>(), align_of::<T>())?.cast::<T>();
        // SAFETY: `bump` returned a region of `size_of::<T>()` bytes aligned for
        // `T`, inside the arena and not overlapping any other live allocation
        // (the cursor only ever moves forward until `reset`, which needs
        // `&mut self` and so cannot run while this borrow lives).
        unsafe {
            ptr.write(value);
            Some(&mut *ptr.as_ptr())
        }
    }

    // A slice of `len` copies of `value`.
    pub fn alloc_slice<T: Copy>(&self, len: usize, value: T) -> Option<&mut [T]> {
        let slice = self.uninit_slice::<T>(len)?;
        for slot in slice.iter_mut() {
            slot.write(value);
        }
        // SAFETY: every element was just written.
        Some(unsafe { assume_init_mut(slice) })
    }

    // A copy of `src` in the arena.
    pub fn alloc_slice_copy<T: Copy>(&self, src: &[T]) -> Option<&mut [T]> {
        let slice = self.uninit_slice::<T>(src.len())?;
        for (slot, value) in slice.iter_mut().zip(src) {
            slot.write(*value);
        }
        // SAFETY: `slice` and `src` have the same length, so every element was
        // written.
        Some(unsafe { assume_init_mut(slice) })
    }

    // An empty vector holding `capacity` elements' worth of arena. Pushing past
    // that capacity does not grow -- the caller reserves the bound it knows.
    pub fn vec<T: Copy>(&self, capacity: usize) -> Option<ArenaVec<'_, T>> {
        Some(ArenaVec {
            buf: self.uninit_slice::<T>(capacity)?,
            len: 0,
        })
    }

    fn uninit_slice<T>(&self, len: usize) -> Option<&mut [MaybeUninit<T>]> {
        // A length that overflows is asking for more than exists, so it takes
        // the same declined-and-counted path as any other oversized request.
        let bytes = size_of::<T>().saturating_mul(len);
        let ptr = self.bump(bytes, align_of::<T>())?.cast::<MaybeUninit<T>>();
        // SAFETY: `bump` returned `len * size_of::<T>()` bytes aligned for `T`
        // inside the arena, and no other live allocation overlaps them.
        // `MaybeUninit<T>` is valid for any bit pattern, so the region needs no
        // initialization to be read as this type.
        Some(unsafe { core::slice::from_raw_parts_mut(ptr.as_ptr(), len) })
    }

    // Carve `size` bytes aligned to `align` off the front of what is left,
    // counting anything this arena could not satisfy.
    fn bump(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        let carved = self.try_bump(size, align);
        if carved.is_none() {
            self.overflows.set(self.overflows.get().saturating_add(1));
        }
        carved
    }

    fn try_bump(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        if align > ARENA_ALIGN || self.cap == 0 {
            return None;
        }
        let start = self.used.get().checked_next_multiple_of(align)?;
        let end = start.checked_add(size)?;
        if end > self.cap {
            return None;
        }
        self.used.set(end);
        if end > self.peak.get() {
            self.peak.set(end);
        }
        // SAFETY: `start <= end <= cap`, so the offset stays inside the one
        // allocation `ptr` owns.
        Some(unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(start)) })
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        if let Some(tag) = self.tag {
            crate::ledger().release(tag, Realm::Host, self.cap as u64);
        }
        if self.cap == 0 {
            return;
        }
        let layout = Layout::from_size_align(self.cap, ARENA_ALIGN).expect("arena layout");
        // SAFETY: `ptr` came from `alloc` with this exact layout in
        // `with_capacity`, and nothing can still borrow it: `Drop` takes
        // `&mut self`.
        unsafe { alloc::alloc::dealloc(self.ptr.as_ptr(), layout) };
    }
}

// SAFETY: an `Arena` owns its buffer outright and hands out borrows tied to
// itself, so moving one to another thread moves the whole allocation with it.
// The `Cell` cursor makes it (correctly) `!Sync`, which is what stops two
// threads bumping the same cursor.
unsafe impl Send for Arena {}

// A vector over a reservation in an arena: pushes cost a write, and the whole
// thing disappears when the arena resets.
pub struct ArenaVec<'a, T: Copy> {
    buf: &'a mut [MaybeUninit<T>],
    len: usize,
}

impl<T: Copy> ArenaVec<'_, T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    pub fn is_full(&self) -> bool {
        self.len == self.buf.len()
    }

    // Append `value`, reporting whether it fit. The reservation is fixed, so a
    // `false` means the caller reserved less than it pushed.
    #[must_use]
    pub fn push(&mut self, value: T) -> bool {
        if self.is_full() {
            return false;
        }
        self.buf[self.len].write(value);
        self.len += 1;
        true
    }

    // Append until the iterator ends or the reservation fills, returning how
    // many were appended.
    pub fn extend(&mut self, values: impl IntoIterator<Item = T>) -> usize {
        let before = self.len;
        for value in values {
            if !self.push(value) {
                break;
            }
        }
        self.len - before
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn as_slice(&self) -> &[T] {
        // SAFETY: the first `len` elements were written by `push`.
        unsafe { assume_init_ref(&self.buf[..self.len]) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: the first `len` elements were written by `push`.
        unsafe { assume_init_mut(&mut self.buf[..self.len]) }
    }
}

impl<T: Copy> core::ops::Deref for ArenaVec<'_, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Copy> core::ops::DerefMut for ArenaVec<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Copy + core::fmt::Debug> core::fmt::Debug for ArenaVec<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_slice().fmt(f)
    }
}

// SAFETY: every element of `slice` must have been initialized.
unsafe fn assume_init_ref<T>(slice: &[MaybeUninit<T>]) -> &[T] {
    // SAFETY: `MaybeUninit<T>` has the same layout as `T`, and the caller
    // guarantees every element holds an initialized value.
    unsafe { &*(slice as *const [MaybeUninit<T>] as *const [T]) }
}

// SAFETY: every element of `slice` must have been initialized.
unsafe fn assume_init_mut<T>(slice: &mut [MaybeUninit<T>]) -> &mut [T] {
    // SAFETY: as `assume_init_ref`, for a unique borrow.
    unsafe { &mut *(slice as *mut [MaybeUninit<T>] as *mut [T]) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocations_come_back_with_their_values() {
        let arena = Arena::with_capacity(4096);
        let a = arena.alloc(7u32).expect("fits");
        let b = arena.alloc_slice(4, 1u16).expect("fits");
        let c = arena.alloc_slice_copy(&[9u64, 8, 7]).expect("fits");

        assert_eq!(*a, 7);
        assert_eq!(b, &[1, 1, 1, 1]);
        assert_eq!(c, &[9, 8, 7]);
    }

    // Separate allocations must not overlap: writing through one may not be
    // visible through another.
    #[test]
    fn separate_allocations_do_not_overlap() {
        let arena = Arena::with_capacity(4096);
        let first = arena.alloc_slice(8, 0u32).expect("fits");
        let second = arena.alloc_slice(8, 0u32).expect("fits");
        first.fill(0xAAAA_AAAA);
        second.fill(0x5555_5555);
        assert!(first.iter().all(|&v| v == 0xAAAA_AAAA));
        assert!(second.iter().all(|&v| v == 0x5555_5555));
    }

    #[test]
    fn allocations_are_aligned_for_their_type() {
        let arena = Arena::with_capacity(4096);
        // A byte first, so the cursor sits at an odd offset.
        let _ = arena.alloc(1u8).expect("fits");
        let wide = arena.alloc(1u128).expect("fits");
        assert!((wide as *const u128).is_aligned());

        let _ = arena.alloc(1u8).expect("fits");
        let slice = arena.alloc_slice(3, 0u64).expect("fits");
        assert!(slice.as_ptr().is_aligned());
    }

    // Running out is a `None`, not a panic: the caller falls back to the heap.
    #[test]
    fn a_full_arena_declines_rather_than_panicking() {
        let arena = Arena::with_capacity(64);
        assert!(arena.alloc_slice(8, 0u64).is_some());
        assert!(arena.alloc(0u8).is_none());
        assert_eq!(arena.remaining(), 0);
    }

    #[test]
    fn an_empty_arena_declines_everything() {
        let arena = Arena::with_capacity(0);
        assert_eq!(arena.capacity(), 0);
        assert!(arena.alloc(1u8).is_none());
    }

    #[test]
    fn reset_hands_the_whole_arena_back() {
        let mut arena = Arena::with_capacity(128);
        {
            let slice = arena.alloc_slice(16, 0u64).expect("fits");
            assert_eq!(slice.len(), 16);
        }
        assert_eq!(arena.used(), 128);
        assert!(arena.alloc(0u8).is_none());

        arena.reset();
        assert_eq!(arena.used(), 0);
        assert!(arena.alloc_slice(16, 0u64).is_some());
    }

    // Peak survives resets: it is what sizes the arena, so it must describe the
    // worst frame rather than the current one.
    #[test]
    fn peak_survives_a_reset() {
        let mut arena = Arena::with_capacity(1024);
        let _ = arena.alloc_slice(64, 0u8).expect("fits");
        arena.reset();
        let _ = arena.alloc_slice(8, 0u8).expect("fits");

        assert_eq!(arena.used(), 8);
        assert_eq!(arena.peak(), 64);
    }

    // An over-aligned type the buffer cannot satisfy is declined, not
    // mis-aligned.
    #[test]
    fn an_over_aligned_type_is_declined() {
        #[repr(align(128))]
        #[derive(Clone, Copy)]
        struct Overaligned(u8);

        let arena = Arena::with_capacity(4096);
        let value = Overaligned(7);
        assert_eq!(value.0, 7);
        assert!(arena.alloc(value).is_none());
    }

    #[test]
    fn a_vector_pushes_into_its_reservation() {
        let arena = Arena::with_capacity(4096);
        let mut v = arena.vec::<u32>(4).expect("fits");
        assert!(v.is_empty());
        for i in 0..4 {
            assert!(v.push(i));
        }
        assert!(v.is_full());
        assert_eq!(v.as_slice(), &[0, 1, 2, 3]);
        assert_eq!(v.len(), 4);
    }

    // The reservation is the whole story: pushing past it reports `false`
    // rather than growing into the rest of the arena.
    #[test]
    fn a_vector_declines_pushes_past_its_reservation() {
        let arena = Arena::with_capacity(4096);
        let mut v = arena.vec::<u8>(2).expect("fits");
        assert!(v.push(1));
        assert!(v.push(2));
        assert!(!v.push(3));
        assert_eq!(v.as_slice(), &[1, 2]);
    }

    #[test]
    fn extend_reports_what_it_took() {
        let arena = Arena::with_capacity(4096);
        let mut v = arena.vec::<u16>(3).expect("fits");
        assert_eq!(v.extend([1, 2, 3, 4, 5]), 3);
        assert_eq!(v.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn a_vector_sorts_and_reads_back_through_the_slice() {
        let arena = Arena::with_capacity(4096);
        let mut v = arena.vec::<u32>(5).expect("fits");
        assert_eq!(v.extend([5, 3, 1, 4, 2]), 5);
        v.sort_unstable();
        assert_eq!(&*v, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn clearing_a_vector_keeps_its_reservation() {
        let arena = Arena::with_capacity(4096);
        let mut v = arena.vec::<u8>(2).expect("fits");
        assert!(v.push(1));
        v.clear();
        assert!(v.is_empty());
        assert!(v.push(2));
        assert_eq!(v.as_slice(), &[2]);
    }

    // Two vectors alive at once must own disjoint reservations.
    #[test]
    fn two_vectors_hold_separate_reservations() {
        let arena = Arena::with_capacity(4096);
        let mut a = arena.vec::<u32>(2).expect("fits");
        let mut b = arena.vec::<u32>(2).expect("fits");
        assert_eq!(a.extend([1, 2]), 2);
        assert_eq!(b.extend([3, 4]), 2);
        assert_eq!(a.as_slice(), &[1, 2]);
        assert_eq!(b.as_slice(), &[3, 4]);
    }

    // A tagged arena accounts for itself: its reservation appears under its tag
    // while it lives and is given back when it drops. Asserted as a delta,
    // since the ledger it reports into is the process-wide one.
    #[test]
    fn a_tagged_arena_reports_its_reservation_for_as_long_as_it_lives() {
        const BYTES: usize = 8192;
        let held = || crate::ledger().usage(MemTag::Scratch, Realm::Host).bytes;

        let before = held();
        {
            let arena = Arena::tagged(BYTES, MemTag::Scratch);
            assert_eq!(held(), before + BYTES as u64);
            // What it hands out does not change what it cost.
            let _ = arena.alloc_slice(16, 0u8).expect("fits");
            assert_eq!(held(), before + BYTES as u64);
        }
        assert_eq!(held(), before);
    }

    #[test]
    fn an_untagged_arena_reports_nothing() {
        let held = || crate::ledger().usage(MemTag::Ui, Realm::Host).bytes;
        let before = held();
        let _arena = Arena::with_capacity(8192);
        assert_eq!(held(), before);
    }

    #[test]
    fn a_reservation_larger_than_the_arena_is_declined() {
        let arena = Arena::with_capacity(64);
        assert!(arena.vec::<u64>(1024).is_none());
    }

    // A declined request is counted, so a caller falling back to the heap
    // leaves evidence the reserve is too small instead of hiding it.
    #[test]
    fn declined_requests_are_counted() {
        let arena = Arena::with_capacity(64);
        assert_eq!(arena.overflows(), 0);

        assert!(arena.alloc_slice(8, 0u64).is_some());
        assert_eq!(arena.overflows(), 0, "a request that fits counts nothing");

        assert!(arena.alloc(0u8).is_none());
        assert!(arena.vec::<u32>(4).is_none());
        assert_eq!(arena.overflows(), 2);

        arena.clear_overflows();
        assert_eq!(arena.overflows(), 0);
    }

    // Sizing evidence has to outlive the frame that produced it, so neither
    // counter is cleared by the per-frame reset.
    #[test]
    fn reset_keeps_the_sizing_evidence() {
        let mut arena = Arena::with_capacity(64);
        let _ = arena.alloc_slice(8, 0u64).expect("fits");
        assert!(arena.alloc(0u8).is_none());

        arena.reset();
        assert_eq!(arena.used(), 0, "the cursor rewinds");
        assert_eq!(arena.peak(), 64, "the peak does not");
        assert_eq!(arena.overflows(), 1, "nor does the overflow count");
    }

    // An over-aligned type and an oversized length are both declines, and both
    // reach the counter rather than returning early past it.
    #[test]
    fn every_decline_path_reaches_the_counter() {
        #[repr(align(128))]
        #[derive(Clone, Copy)]
        struct Overaligned(u8);

        let arena = Arena::with_capacity(4096);
        let value = Overaligned(7);
        assert_eq!(value.0, 7);
        assert!(arena.alloc(value).is_none());
        assert!(arena.vec::<u64>(usize::MAX).is_none());
        assert_eq!(arena.overflows(), 2);

        let empty = Arena::with_capacity(0);
        assert!(empty.alloc(1u8).is_none());
        assert_eq!(empty.overflows(), 1);
    }
}
