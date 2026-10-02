//! The rings an acceleration-structure update rebuilds in place instead of
//! allocating fresh.
//!
//! A ring slot can be rewritten only once the frames-in-flight fence has retired
//! every frame that traced it. [`FrameRing`] gets that by being indexed by the
//! frame's own in-flight index: frame `R` writes slot `R % depth`, so it suits
//! work that runs on every frame. [`StaticRing`] advances one slot per published
//! rebuild instead, for work that may skip frames: a scene that rarely moves keeps
//! tracing one structure across many frames, so a frame-keyed slot could be
//! rewritten while a live trace still reads it, while the slot after the live one
//! has always been retired.
//!
//! Both rest on the live structure moving on: the slot a published rebuild wrote
//! stays live, and traced, until another rebuild publishes. A rebuild that fails
//! leaves it live, so neither ring hands out the live slot: a [`StaticRing`]
//! advances its cursor only on a publish, and a [`FrameRing`] refuses its live
//! slot until something else publishes.
//!
//! Slots are lent out by value so a backend can rebuild one while borrowing the
//! rest of its state, and put back on every exit path, so a failed rebuild leaves
//! the slots it did not publish as they were.

use alloc::vec::Vec;
use core::mem;

use crate::render::error::{RenderError, RenderResult};

/// Which slot of a frame-indexed ring the live TLAS was built from, and when
/// each stopped being live, so a slot is rewritten only once no frame still in
/// flight can trace it.
///
/// Frames trace the live slot until the update tick another publish (or an
/// unpublish) replaces it, and the frame-begin wait on tick `now` retires frames
/// up to `now - depth`. In steady state every frame publishes its own slot, so
/// each slot was last live a full ring of frames ago and is always writable. A
/// failed rebuild leaves its predecessor live, and that slot then waits.
#[derive(Debug)]
pub struct SlotLiveness {
    live: Option<usize>,
    // Per slot, the update tick it last stopped being live.
    released: Vec<Option<u64>>,
}

impl SlotLiveness {
    /// Tracking for a ring of `depth` slots, one per frame in flight.
    pub fn new(depth: usize) -> Self {
        Self {
            live: None,
            released: alloc::vec![None; depth.max(1)],
        }
    }

    /// Whether `slot` can be rewritten on update tick `now`. A single-slot ring
    /// always can, since the frame-begin wait retires the one frame that could
    /// read it.
    pub fn writable(&self, slot: usize, now: u64) -> bool {
        let depth = self.released.len() as u64;
        if depth <= 1 {
            return true;
        }
        if self.live == Some(slot) {
            return false;
        }
        match self.released.get(slot).copied().flatten() {
            Some(at) => now + 1 >= at + depth,
            None => true,
        }
    }

    /// The live TLAS was built from `slot` on update tick `now`.
    pub fn publish(&mut self, slot: usize, now: u64) {
        if self.live != Some(slot) {
            self.unpublish(now);
        }
        self.live = Some(slot);
    }

    /// The live TLAS stopped reading any slot on update tick `now`.
    pub fn unpublish(&mut self, now: u64) {
        if let Some(released) = self
            .live
            .take()
            .and_then(|live| self.released.get_mut(live))
        {
            *released = Some(now);
        }
    }
}

/// Slots indexed by the frame's in-flight index.
pub struct FrameRing<S> {
    slots: Vec<S>,
    liveness: SlotLiveness,
}

impl<S: Default> FrameRing<S> {
    /// One default slot per frame in flight (at least one).
    pub fn new(frames_in_flight: usize) -> Self {
        let depth = frames_in_flight.max(1);
        Self {
            slots: (0..depth).map(|_| S::default()).collect(),
            liveness: SlotLiveness::new(depth),
        }
    }

    /// Lend out frame `frame_idx`'s slot on update tick `now`, leaving a default
    /// in its place. `None` while a frame still in flight may trace the slot (see
    /// [`SlotLiveness`]).
    pub fn take(&mut self, frame_idx: usize, now: u64) -> RenderResult<Option<S>> {
        let len = self.slots.len();
        if !self.liveness.writable(frame_idx, now) {
            return Ok(None);
        }
        self.slots
            .get_mut(frame_idx)
            .map(|s| Some(mem::take(s)))
            .ok_or_else(|| out_of_range(frame_idx, len))
    }

    /// Return a slot lent out by [`Self::take`] whose rebuild did not publish.
    pub fn put(&mut self, frame_idx: usize, slot: S) {
        if let Some(s) = self.slots.get_mut(frame_idx) {
            *s = slot;
        }
    }

    /// Return a slot lent out by [`Self::take`] whose structures became live on
    /// update tick `now`.
    pub fn publish(&mut self, frame_idx: usize, slot: S, now: u64) {
        self.put(frame_idx, slot);
        self.liveness.publish(frame_idx, now);
    }
}

impl<S> FrameRing<S> {
    /// The live TLAS stopped reading any slot on update tick `now`.
    pub fn unpublish(&mut self, now: u64) {
        self.liveness.unpublish(now);
    }

    /// Every slot, for teardown or for invalidating them all at once.
    pub fn slots_mut(&mut self) -> impl Iterator<Item = &mut S> {
        self.slots.iter_mut()
    }
}

/// Slots revisited one per published rebuild, in order.
pub struct StaticRing<S> {
    slots: Vec<S>,
    // The live slot: the one the last published rebuild wrote.
    cursor: usize,
}

impl<S: Default> StaticRing<S> {
    /// A ring of `len` slots (at least one) whose live slot, 0, holds `first`:
    /// the structures the initial build published. The first rebuild moves past
    /// it, so it is rewritten only a full cycle later like every other slot.
    pub fn new(len: usize, first: S) -> Self {
        let mut slots: Vec<S> = (0..len.max(1)).map(|_| S::default()).collect();
        slots[0] = first;
        Self { slots, cursor: 0 }
    }

    /// Lend out the slot after the live one, returning its index for
    /// [`Self::publish`] or [`Self::put`].
    pub fn take_next(&mut self) -> (usize, S) {
        let next = next_slot(self.cursor, self.slots.len());
        (next, mem::take(&mut self.slots[next]))
    }

    /// Return a slot lent out by [`Self::take_next`] whose rebuild did not
    /// publish; the live slot stays where it was.
    pub fn put(&mut self, index: usize, slot: S) {
        if let Some(s) = self.slots.get_mut(index) {
            *s = slot;
        }
    }

    /// Return a slot lent out by [`Self::take_next`] whose structures are now
    /// live.
    pub fn publish(&mut self, index: usize, slot: S) {
        if let Some(s) = self.slots.get_mut(index) {
            *s = slot;
            self.cursor = index;
        }
    }
}

impl<S> StaticRing<S> {
    /// Every slot, for teardown.
    pub fn slots_mut(&mut self) -> impl Iterator<Item = &mut S> {
        self.slots.iter_mut()
    }
}

// The slot after `cursor`, wrapping at `len`.
fn next_slot(cursor: usize, len: usize) -> usize {
    (cursor + 1) % len.max(1)
}

fn out_of_range(index: usize, len: usize) -> RenderError {
    RenderError::Other(alloc::format!(
        "ring slot {index} out of range for {len} frames in flight"
    ))
}

/// Acceleration-structure build scratch, one buffer per frame in flight.
///
/// Scratch is written by the builds that name it and read by nothing after, so
/// it only has to outlive the frame that recorded it, and the in-flight fence
/// retires a slot's previous writer before the next frame with that index
/// records. One shared buffer cannot promise that: frame N's builds would write
/// the bytes frame N-1's builds are still working in.
///
/// A slot is replaced only when a build outgrows it. The replaced buffer is
/// dropped in place, so each backend's buffer type must keep its memory alive
/// past the builds already recorded against it this frame and any still in
/// flight, or the backend must record at most one replacing path per frame.
pub struct ScratchRing<B> {
    slots: Vec<Option<Sized<B>>>,
}

struct Sized<B> {
    buffer: B,
    capacity: u64,
}

impl<B> ScratchRing<B> {
    /// One slot per frame in flight (at least one), each allocated at
    /// `capacity` bytes by `alloc`.
    pub fn filled(
        frames_in_flight: usize,
        capacity: u64,
        mut alloc: impl FnMut(u64) -> RenderResult<B>,
    ) -> RenderResult<Self> {
        let mut slots = Vec::with_capacity(frames_in_flight.max(1));
        for _ in 0..frames_in_flight.max(1) {
            slots.push(Some(Sized {
                buffer: alloc(capacity)?,
                capacity,
            }));
        }
        Ok(Self { slots })
    }

    /// Frame `frame_idx`'s scratch, holding at least `capacity` bytes. The slot
    /// is reallocated through `alloc` only when it is smaller; a failed
    /// allocation leaves the old one in place.
    pub fn ensure(
        &mut self,
        frame_idx: usize,
        capacity: u64,
        alloc: impl FnOnce(u64) -> RenderResult<B>,
    ) -> RenderResult<&B> {
        let len = self.slots.len();
        let slot = self
            .slots
            .get_mut(frame_idx)
            .ok_or_else(|| out_of_range(frame_idx, len))?;
        let fits = slot.as_ref().is_some_and(|s| s.capacity >= capacity);
        if !fits {
            let buffer = alloc(capacity)?;
            *slot = Some(Sized { buffer, capacity });
        }
        slot.as_ref()
            .map(|s| &s.buffer)
            .ok_or_else(|| out_of_range(frame_idx, len))
    }

    /// Frame `frame_idx`'s scratch as last allocated.
    pub fn get(&self, frame_idx: usize) -> Option<&B> {
        self.slots.get(frame_idx)?.as_ref().map(|s| &s.buffer)
    }

    /// Every allocated buffer, leaving the slots empty, for a backend that hands
    /// them to a deferred free.
    pub fn drain(&mut self) -> impl Iterator<Item = B> + '_ {
        self.slots
            .iter_mut()
            .filter_map(|s| s.take())
            .map(|s| s.buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn the_cursor_wraps_around_the_ring() {
        assert_eq!(next_slot(0, 3), 1);
        assert_eq!(next_slot(1, 3), 2);
        assert_eq!(next_slot(2, 3), 0);
        // A single-slot ring always returns slot 0.
        assert_eq!(next_slot(0, 1), 0);
        assert_eq!(next_slot(0, 0), 0);
    }

    #[test]
    fn a_static_slot_is_revisited_only_after_a_full_cycle() {
        let mut ring = StaticRing::new(3, 10u32);
        let mut visited = vec![];
        for _ in 0..4 {
            let (i, slot) = ring.take_next();
            visited.push((i, slot));
            ring.publish(i, slot + 1);
        }
        // Slot 0 holds the initial build, so it is the last of the first cycle.
        assert_eq!(visited, [(1, 0), (2, 0), (0, 10), (1, 1)]);
    }

    #[test]
    fn a_failed_static_rebuild_never_reaches_the_live_slot() {
        let mut ring = StaticRing::new(2, 10u32);
        // However many rebuilds fail, each lends the slot after the live one.
        for _ in 0..5 {
            let (i, slot) = ring.take_next();
            assert_eq!(i, 1);
            ring.put(i, slot);
        }
        let (i, slot) = ring.take_next();
        ring.publish(i, slot);
        assert_eq!(ring.take_next().0, 0);
    }

    #[test]
    fn a_lent_slot_comes_back_intact() {
        let mut ring = FrameRing::<u32>::new(2);
        let slot = ring.take(1, 0).expect("in range").expect("not live");
        ring.put(1, slot + 5);
        assert_eq!(ring.take(1, 0).expect("in range"), Some(5));
        assert!(ring.take(2, 0).is_err());
    }

    // Publishing on every tick, each frame's slot was last live a full ring of
    // frames ago.
    #[test]
    fn a_frame_ring_publishing_every_frame_always_lends() {
        let mut ring = FrameRing::<u32>::new(3);
        for tick in 0..12u64 {
            let idx = (tick % 3) as usize;
            let slot = ring.take(idx, tick).expect("in range").expect("retired");
            ring.publish(idx, slot, tick);
        }
    }

    #[test]
    fn the_live_frame_slot_is_refused_until_its_tracers_retire() {
        let mut ring = FrameRing::<u32>::new(2);
        let slot = ring.take(0, 0).expect("in range").expect("not live");
        ring.publish(0, slot, 0);
        // Tick 1 (slot 1) failed to publish, so tick 2 finds slot 0 still live.
        assert_eq!(ring.take(0, 2).expect("in range"), None);
        let slot = ring.take(1, 3).expect("in range").expect("not live");
        ring.publish(1, slot, 3);
        // Ticks 0..=2 traced slot 0; the wait on tick 4 retires tick 2.
        assert!(ring.take(0, 4).expect("in range").is_some());
    }

    #[test]
    fn an_unpublished_slot_waits_out_the_frames_that_traced_it() {
        let mut ring = FrameRing::<u32>::new(2);
        let slot = ring.take(0, 0).expect("in range").expect("not live");
        ring.publish(0, slot, 0);
        // Tick 1 failed; tick 2 publishes a TLAS without skinned BLAS, but tick 1
        // traced slot 0 and may still be in flight.
        ring.unpublish(2);
        assert_eq!(ring.take(0, 2).expect("in range"), None);
        assert!(ring.take(0, 3).expect("in range").is_some());
    }

    // Every slot remembers its own release, so a deep ring that releases two
    // slots in a row still waits out the frames that traced the first.
    #[test]
    fn each_released_slot_waits_out_its_own_tracers() {
        let mut ring = FrameRing::<u32>::new(4);
        let slot = ring.take(0, 0).expect("in range").expect("not live");
        ring.publish(0, slot, 0);
        // Tick 1 (slot 1) failed; slot 0 stays live until slot 2 publishes.
        let slot = ring.take(2, 2).expect("in range").expect("not live");
        ring.publish(2, slot, 2);
        let slot = ring.take(3, 3).expect("in range").expect("not live");
        ring.publish(3, slot, 3);
        // Ticks up to 2 traced slot 0, and the wait on tick 4 retires only tick 0.
        assert_eq!(ring.take(0, 4).expect("in range"), None);
        assert!(ring.take(0, 5).expect("in range").is_some());
        assert_eq!(ring.take(2, 5).expect("in range"), None);
    }

    #[test]
    fn a_single_slot_frame_ring_always_lends() {
        let mut ring = FrameRing::<u32>::new(0);
        let slot = ring.take(0, 0).expect("in range").expect("lends");
        ring.publish(0, slot, 0);
        assert!(ring.take(0, 1).expect("in range").is_some());
    }

    #[test]
    fn scratch_is_replaced_only_when_a_build_outgrows_it() {
        let mut allocs = 0;
        let mut ring = ScratchRing::filled(2, 1000, |c| {
            allocs += 1;
            Ok(c)
        })
        .expect("allocates");
        assert_eq!(allocs, 2);
        // The build it was sized for, and a smaller one, reuse it.
        assert_eq!(ring.ensure(0, 1000, |_| unreachable!()).copied(), Ok(1000));
        assert_eq!(ring.ensure(0, 1, |_| unreachable!()).copied(), Ok(1000));
        // One byte more replaces only this frame's slot.
        assert_eq!(ring.ensure(0, 1001, Ok).copied(), Ok(1001));
        assert_eq!(ring.get(1), Some(&1000));
        assert_eq!(ring.drain().collect::<Vec<_>>(), [1001, 1000]);
        assert_eq!(ring.get(0), None);
    }

    #[test]
    fn a_failed_scratch_grow_keeps_the_old_slot() {
        let mut ring = ScratchRing::filled(1, 64, Ok).expect("allocates");
        let grown = ring.ensure(0, 128, |_| Err(RenderError::Other("no memory".into())));
        assert!(grown.is_err());
        assert_eq!(ring.get(0), Some(&64));
        assert!(ring.ensure(3, 1, Ok).is_err());
    }
}
