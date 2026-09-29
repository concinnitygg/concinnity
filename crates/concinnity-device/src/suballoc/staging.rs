//! Staging-memory policy for geometry uploads recorded rather than waited on.
//!
//! `StagingRing` places each upload's bytes in one persistently-mapped
//! CPU-visible buffer, first in first out. Everything placed before a submit is
//! one batch, tagged with the frame tick from which its bytes may be
//! overwritten; `reclaim` releases batches oldest first as ticks pass. The
//! backend owns the buffer and records the copies, so this is pure policy like
//! the rest of `suballoc`.
//!
//! `Recycler` holds the command lists those submits recorded into until their
//! tick passes, so a list is reused rather than created per submit.

use concinnity_core::render::fullscreen::align_up;
use std::collections::VecDeque;

// Placement alignment in a ring. Buffer copies need none; 16 keeps every
// source offset on a vertex-friendly boundary.
pub(crate) const STAGING_ALIGN: u64 = 16;

// The smallest ring a backend creates.
pub(crate) const STAGING_MIN_CAPACITY: u64 = 8 << 20;

// The largest ring created ahead of need. A burst past it grows the ring.
pub(crate) const STAGING_RESERVE_CAP: u64 = 64 << 20;

// A submitted run of placements, released whole once `retire_at` has passed.
#[derive(Clone, Copy, Debug)]
struct Batch {
    end: u64,
    retire_at: u64,
}

// A FIFO byte ring over a buffer of `capacity` bytes. Positions are monotonic
// byte counts; a position's offset in the buffer is it modulo `capacity`.
pub(crate) struct StagingRing {
    capacity: u64,
    // Next free position.
    head: u64,
    // Oldest position a batch in flight, or an unsubmitted placement, still owns.
    tail: u64,
    // Where the placements not yet sealed into a batch begin.
    open: u64,
    batches: VecDeque<Batch>,
}

impl StagingRing {
    // A ring over `capacity` bytes, a power of two no smaller than any
    // alignment later asked of `alloc`.
    pub(crate) fn new(capacity: u64) -> Self {
        debug_assert!(capacity.is_power_of_two());
        Self {
            capacity,
            head: 0,
            tail: 0,
            open: 0,
            batches: VecDeque::new(),
        }
    }

    pub(crate) fn capacity(&self) -> u64 {
        self.capacity
    }

    // Place `size` bytes at an `align`ed offset, or `None` when the ring cannot
    // hold them until older batches retire. A placement never wraps: one that
    // would run past the end starts over at offset 0 and gives up the tail.
    pub(crate) fn alloc(&mut self, size: u64, align: u64) -> Option<u64> {
        if size == 0 || size > self.capacity {
            return None;
        }
        let mut start = align_up(self.head, align.max(1));
        if start % self.capacity + size > self.capacity {
            start = align_up(start, self.capacity);
        }
        let end = start + size;
        if end - self.tail > self.capacity {
            return None;
        }
        self.head = end;
        Some(start % self.capacity)
    }

    // Close everything placed since the last seal into one batch whose bytes
    // may be overwritten from tick `retire_at`. A no-op when nothing was placed.
    pub(crate) fn seal(&mut self, retire_at: u64) {
        if self.head == self.open {
            return;
        }
        self.batches.push_back(Batch {
            end: self.head,
            retire_at,
        });
        self.open = self.head;
    }

    // Release every batch whose tick has passed, oldest first. Batches retire
    // in order, so a later batch waits on an earlier one even if its own tick
    // came sooner.
    pub(crate) fn reclaim(&mut self, tick: u64) {
        while let Some(batch) = self.batches.front() {
            if batch.retire_at > tick {
                break;
            }
            self.tail = batch.end;
            self.batches.pop_front();
        }
    }

    // Bytes neither free nor reclaimable yet, alignment and wrap padding included.
    #[cfg(test)]
    fn in_use(&self) -> u64 {
        self.head - self.tail
    }
}

// The capacity a ring must grow to so one `size`-byte placement fits: at least
// double `capacity`, rounded to a power of two.
pub(crate) fn grown_capacity(capacity: u64, size: u64) -> u64 {
    capacity
        .saturating_mul(2)
        .max(size)
        .max(STAGING_MIN_CAPACITY)
        .next_power_of_two()
}

// The capacity to create a ring at ahead of need, for a stream whose single
// frame may stage up to `expected` bytes.
pub(crate) fn reserved_capacity(expected: u64) -> u64 {
    expected
        .clamp(STAGING_MIN_CAPACITY, STAGING_RESERVE_CAP)
        .next_power_of_two()
}

// Objects held until a frame tick, then handed back for reuse.
pub(crate) struct Recycler<T> {
    held: Vec<(T, u64)>,
}

impl<T> Recycler<T> {
    pub(crate) fn new() -> Self {
        Self { held: Vec::new() }
    }

    // Hold `object` until tick `retire_at`.
    pub(crate) fn release(&mut self, object: T, retire_at: u64) {
        self.held.push((object, retire_at));
    }

    // One object whose tick has passed, if any.
    pub(crate) fn acquire(&mut self, tick: u64) -> Option<T> {
        let index = self.held.iter().position(|(_, at)| *at <= tick)?;
        Some(self.held.swap_remove(index).0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placements_are_aligned_and_sequential() {
        let mut ring = StagingRing::new(1024);
        assert_eq!(ring.alloc(10, 16), Some(0));
        assert_eq!(ring.alloc(10, 16), Some(16));
        assert_eq!(ring.alloc(1, 1), Some(26));
        assert_eq!(ring.in_use(), 27);
    }

    #[test]
    fn a_full_ring_refuses_until_its_batch_retires() {
        let mut ring = StagingRing::new(256);
        assert_eq!(ring.alloc(200, 16), Some(0));
        ring.seal(5);
        assert_eq!(ring.alloc(100, 16), None);
        ring.reclaim(4);
        assert_eq!(ring.alloc(100, 16), None, "tick 4 precedes the retire tick");
        ring.reclaim(5);
        assert_eq!(
            ring.alloc(100, 16),
            Some(0),
            "wraps rather than straddling the end"
        );
    }

    #[test]
    fn an_unsealed_placement_is_never_reclaimed() {
        let mut ring = StagingRing::new(256);
        ring.alloc(200, 16);
        ring.reclaim(u64::MAX);
        assert_eq!(ring.alloc(100, 16), None);
        assert_eq!(ring.in_use(), 200);
    }

    #[test]
    fn a_placement_never_straddles_the_end() {
        let mut ring = StagingRing::new(256);
        assert_eq!(ring.alloc(160, 16), Some(0));
        ring.seal(1);
        ring.reclaim(1);
        // 96 bytes remain before the end; 128 does not fit there, so it starts
        // at 0 and the skipped tail counts as used until its batch retires.
        assert_eq!(ring.alloc(128, 16), Some(0));
        assert_eq!(ring.in_use(), 96 + 128);
        ring.seal(2);
        ring.reclaim(2);
        assert_eq!(ring.in_use(), 0);
    }

    #[test]
    fn batches_retire_in_submission_order() {
        let mut ring = StagingRing::new(256);
        ring.alloc(64, 16);
        ring.seal(10);
        ring.alloc(64, 16);
        ring.seal(3);
        ring.reclaim(5);
        assert_eq!(
            ring.in_use(),
            128,
            "the older batch holds the newer one back"
        );
        ring.reclaim(10);
        assert_eq!(ring.in_use(), 0);
    }

    #[test]
    fn sealing_nothing_adds_no_batch() {
        let mut ring = StagingRing::new(256);
        ring.seal(1);
        ring.alloc(64, 16);
        ring.reclaim(1);
        assert_eq!(
            ring.in_use(),
            64,
            "the empty seal did not capture the later placement"
        );
    }

    #[test]
    fn oversized_and_empty_placements_are_refused() {
        let mut ring = StagingRing::new(256);
        assert_eq!(ring.alloc(257, 1), None);
        assert_eq!(ring.alloc(0, 1), None);
        assert_eq!(ring.alloc(256, 16), Some(0));
    }

    #[test]
    fn growth_doubles_and_covers_the_request() {
        let min = STAGING_MIN_CAPACITY;
        assert_eq!(grown_capacity(0, 10), min);
        assert_eq!(grown_capacity(min, 10), min * 2);
        assert_eq!(grown_capacity(min, min * 3), min * 4);
    }

    #[test]
    fn a_reserve_covers_the_stream_up_to_the_cap() {
        let min = STAGING_MIN_CAPACITY;
        assert_eq!(reserved_capacity(0), min);
        assert_eq!(reserved_capacity(min * 3), min * 4);
        assert_eq!(reserved_capacity(u64::MAX), STAGING_RESERVE_CAP);
    }

    #[test]
    fn a_recycled_object_waits_for_its_tick() {
        let mut pool = Recycler::new();
        pool.release("a", 4);
        assert_eq!(pool.acquire(3), None);
        assert_eq!(pool.acquire(4), Some("a"));
        assert_eq!(pool.acquire(4), None);
    }

    #[test]
    fn the_first_retired_object_is_handed_back() {
        let mut pool = Recycler::new();
        pool.release(1, 9);
        pool.release(2, 1);
        assert_eq!(pool.acquire(5), Some(2));
        assert_eq!(pool.acquire(5), None);
        assert_eq!(pool.acquire(9), Some(1));
    }
}
