//! Write tracking for a uniform block a backend rings over its frames in
//! flight. The CPU owns the values; every slot is marked when they change and
//! one slot is cleared per frame, so a world whose values are steady writes
//! nothing after the ring has caught up.

/// Which frame-in-flight copies of a ringed uniform block still need a write.
///
/// ```rust
/// # use concinnity_core::render::frame_dirty::FrameDirty;
/// let mut dirty = FrameDirty::new(2);
/// assert!(dirty.take(0));
/// assert!(dirty.take(1));
/// assert!(!dirty.take(0));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameDirty {
    all: u32,
    pending: u32,
}

/// Frames in flight a single [`FrameDirty`] can track.
pub const MAX_TRACKED_FRAMES: usize = 32;

impl FrameDirty {
    /// Track `frames` slots, every one pending so the ring seeds itself.
    /// Frames beyond [`MAX_TRACKED_FRAMES`] are not tracked.
    pub const fn new(frames: usize) -> Self {
        let all = if frames >= MAX_TRACKED_FRAMES {
            u32::MAX
        } else {
            (1u32 << frames) - 1
        };
        Self { all, pending: all }
    }

    /// Mark every slot pending, after a change to the CPU-side values.
    pub fn mark_all(&mut self) {
        self.pending = self.all;
    }

    /// Whether `frame` still needs a write, clearing it if so.
    pub fn take(&mut self, frame: usize) -> bool {
        if frame >= MAX_TRACKED_FRAMES {
            return true;
        }
        let bit = 1u32 << frame;
        let pending = self.pending & bit != 0;
        self.pending &= !bit;
        pending
    }

    /// Whether any slot is still pending.
    pub const fn any_pending(&self) -> bool {
        self.pending != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_slot_is_pending_until_it_is_taken_once() {
        let mut dirty = FrameDirty::new(3);
        assert!(dirty.any_pending());
        for frame in 0..3 {
            assert!(dirty.take(frame), "slot {frame} seeds pending");
        }
        assert!(!dirty.any_pending());
        for frame in 0..3 {
            assert!(!dirty.take(frame), "slot {frame} stays clear");
        }
    }

    #[test]
    fn a_change_re_arms_every_slot() {
        let mut dirty = FrameDirty::new(2);
        dirty.take(0);
        dirty.take(1);
        dirty.mark_all();
        assert!(dirty.take(1));
        assert!(dirty.take(0));
        assert!(!dirty.any_pending());
    }

    #[test]
    fn taking_one_slot_leaves_the_others_alone() {
        let mut dirty = FrameDirty::new(3);
        assert!(dirty.take(1));
        assert!(dirty.any_pending());
        assert!(dirty.take(0));
        assert!(dirty.take(2));
        assert!(!dirty.any_pending());
    }

    #[test]
    fn an_untracked_slot_always_reports_pending() {
        let mut dirty = FrameDirty::new(2);
        assert!(dirty.take(MAX_TRACKED_FRAMES));
        assert!(dirty.take(MAX_TRACKED_FRAMES));
    }

    #[test]
    fn a_ring_wider_than_the_mask_tracks_every_bit_it_has() {
        let dirty = FrameDirty::new(MAX_TRACKED_FRAMES);
        assert_eq!(dirty, FrameDirty::new(MAX_TRACKED_FRAMES + 4));
        assert!(dirty.any_pending());
    }
}
