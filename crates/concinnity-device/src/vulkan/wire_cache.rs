//! Per-frame-slot memo of what a descriptor set already points at.
//!
//! Several sets are re-pointed at the top of every frame so a slot that missed a
//! rebuild still binds live handles rather than retired ones. The handles only
//! actually move on a rebuild, so the steady state is a run of byte-identical
//! writes. This records what each slot was last given and reports whether a new
//! value is a change, letting the caller skip the write while keeping the
//! guarantee: any handle that differs from the one the set holds still fires.

// One entry per frame in flight. `T` is whatever bundle of handles the set's
// dynamic bindings are written from.
pub(in crate::vulkan) struct WireCache<T> {
    wired: Vec<Option<T>>,
}

impl<T: Copy + PartialEq> WireCache<T> {
    pub(in crate::vulkan) fn new(frames: usize) -> Self {
        Self {
            wired: (0..frames).map(|_| None).collect(),
        }
    }

    // True when `value` differs from what `frame_idx` was last wired with, which
    // also records it as the slot's new contents. A frame index past the end
    // reports stale and memoizes nothing, so an out-of-range caller degrades to
    // the unconditional write rather than silently skipping one.
    pub(in crate::vulkan) fn changed(&mut self, frame_idx: usize, value: T) -> bool {
        match self.wired.get_mut(frame_idx) {
            Some(slot) if *slot == Some(value) => false,
            Some(slot) => {
                *slot = Some(value);
                true
            }
            None => true,
        }
    }

    // What `frame_idx` was last wired with, or `None` when it has never been
    // wired (or is out of range). For callers that want to assert a set holds
    // what they are about to draw against.
    pub(in crate::vulkan) fn current(&self, frame_idx: usize) -> Option<T> {
        self.wired.get(frame_idx).copied().flatten()
    }

    // Forget every slot, so the next frame rewires unconditionally. Called when
    // the sets themselves are reallocated or their bindings rewritten out of
    // band.
    pub(in crate::vulkan) fn reset(&mut self) {
        self.wired.fill(None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_write_of_each_slot_is_a_change() {
        let mut cache = WireCache::new(2);
        assert!(cache.changed(0, 7u32));
        assert!(cache.changed(1, 7u32));
    }

    #[test]
    fn repeating_a_value_in_the_same_slot_is_not() {
        let mut cache = WireCache::new(2);
        assert!(cache.changed(0, 7u32));
        assert!(!cache.changed(0, 7u32));
        assert!(!cache.changed(0, 7u32));
    }

    #[test]
    fn slots_memoize_independently() {
        let mut cache = WireCache::new(2);
        assert!(cache.changed(0, 7u32));
        // Slot 1 has never seen the value even though slot 0 holds it.
        assert!(cache.changed(1, 7u32));
        assert!(!cache.changed(0, 7u32));
    }

    #[test]
    fn a_new_value_fires_again() {
        let mut cache = WireCache::new(1);
        assert!(cache.changed(0, 7u32));
        assert!(cache.changed(0, 8u32));
        assert!(!cache.changed(0, 8u32));
        assert!(cache.changed(0, 7u32));
    }

    #[test]
    fn current_reports_what_a_slot_holds() {
        let mut cache = WireCache::new(2);
        assert_eq!(cache.current(0), None);
        cache.changed(0, 7u32);
        assert_eq!(cache.current(0), Some(7));
        assert_eq!(cache.current(1), None);
        assert_eq!(cache.current(9), None);
        cache.reset();
        assert_eq!(cache.current(0), None);
    }

    #[test]
    fn reset_forgets_every_slot() {
        let mut cache = WireCache::new(2);
        cache.changed(0, 7u32);
        cache.changed(1, 9u32);
        cache.reset();
        assert!(cache.changed(0, 7u32));
        assert!(cache.changed(1, 9u32));
    }

    #[test]
    fn an_out_of_range_slot_always_reports_stale() {
        let mut cache = WireCache::new(1);
        assert!(cache.changed(4, 7u32));
        assert!(cache.changed(4, 7u32));
    }
}
