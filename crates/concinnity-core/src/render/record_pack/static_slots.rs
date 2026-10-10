//! Which ring slots hold a current copy of records written once per change.

use crate::render::frame_dirty::{FrameDirty, MAX_TRACKED_FRAMES};

// The key recorded for a slot nothing has been written into.
const UNWRITTEN: u64 = u64::MAX;

/// Tracks, per ring slot, whether a block of data that changes only with the
/// world (the instance tail, the probe records) is already in that slot's
/// buffer as this frame needs it. A slot that is current is skipped.
///
/// What else the copy depends on rides in a key: where in the buffer the block
/// sits, or the revision of its source. A slot whose copy was written under a
/// different key is rewritten.
///
/// ```rust
/// # use concinnity_core::render::record_pack::StaticSlots;
/// let mut slots = StaticSlots::new(2);
/// assert!(slots.take(0, 100, false), "first use writes");
/// assert!(!slots.take(0, 100, false), "then it is current");
/// assert!(slots.take(0, 120, false), "a moved block is rewritten");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaticSlots {
    dirty: FrameDirty,
    // The key each slot's copy was written under.
    keys: [u64; MAX_TRACKED_FRAMES],
}

impl StaticSlots {
    /// Track `slots` ring slots, none of them written yet.
    pub const fn new(slots: usize) -> Self {
        Self {
            dirty: FrameDirty::new(slots),
            keys: [UNWRITTEN; MAX_TRACKED_FRAMES],
        }
    }

    /// The data changed: every slot's copy is out of date.
    pub fn mark_all(&mut self) {
        self.dirty.mark_all();
    }

    /// Whether `slot` needs the data written under `key`, counting it as
    /// written from here on. `fresh` says the slot's buffer was just
    /// (re)allocated, so it holds nothing. A slot past what can be tracked is
    /// always written.
    pub fn take(&mut self, slot: usize, key: u64, fresh: bool) -> bool {
        let pending = self.dirty.take(slot);
        let Some(written) = self.keys.get_mut(slot) else {
            return true;
        };
        let changed = *written != key;
        *written = key;
        pending || changed || fresh
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_slot_is_written_once_then_skipped() {
        let mut slots = StaticSlots::new(3);
        for slot in 0..3 {
            assert!(slots.take(slot, 10, false), "slot {slot} seeds");
        }
        for _ in 0..4 {
            for slot in 0..3 {
                assert!(!slots.take(slot, 10, false), "slot {slot} is current");
            }
        }
    }

    #[test]
    fn a_change_rewrites_every_slot_exactly_once() {
        let mut slots = StaticSlots::new(3);
        for slot in 0..3 {
            slots.take(slot, 10, false);
        }
        slots.mark_all();
        for slot in [2, 0, 1] {
            assert!(slots.take(slot, 10, false), "slot {slot} after the change");
        }
        for slot in 0..3 {
            assert!(!slots.take(slot, 10, false), "slot {slot} only once");
        }
    }

    // A draw list that grew moves the block within the buffer; each slot is
    // rewritten the first time it is used at the new offset.
    #[test]
    fn a_moved_block_rewrites_each_slot_at_its_next_use() {
        let mut slots = StaticSlots::new(2);
        slots.take(0, 10, false);
        slots.take(1, 10, false);
        assert!(slots.take(0, 11, false));
        assert!(!slots.take(0, 11, false));
        assert!(slots.take(1, 11, false));
        assert!(!slots.take(1, 11, false));
    }

    #[test]
    fn a_reallocated_slot_is_rewritten() {
        let mut slots = StaticSlots::new(2);
        slots.take(0, 10, false);
        assert!(slots.take(0, 10, true));
        assert!(!slots.take(0, 10, false));
        assert!(slots.take(1, 10, false), "the other slot still seeds");
    }

    #[test]
    fn a_slot_past_the_tracked_range_is_always_written() {
        let mut slots = StaticSlots::new(2);
        assert!(slots.take(MAX_TRACKED_FRAMES, 10, false));
        assert!(slots.take(MAX_TRACKED_FRAMES, 10, false));
    }
}
