//! The accumulation ring a temporal post pass reads its own previous output
//! through, and the gate that says whether that output means anything yet.
//!
//! Pure state: no device, no targets, just which slot this frame writes, which
//! slot it samples, and whether the sampled one has ever been written. It is the
//! part of a temporal pass that was hand-duplicated on every backend and the
//! part most able to go quietly wrong, since nothing about a wrong slot choice
//! is visible except as ghosting a frame later.
//!
//! Two moduli, not one. `slots` is how many targets exist; `stride` is how far
//! the write index walks before it wraps. They differ where a backend's
//! consumers are pre-bound per frame in flight: the write index is then the
//! frame slot, which cycles under the frame count, while the ring is floored at
//! two targets so a pass never samples the target it is writing. A single frame
//! in flight is the case that separates them.

/// Which target a temporal pass writes this frame, which it samples as history,
/// and whether that history is meaningful.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct HistoryRing {
    slots: usize,
    stride: usize,
    write: usize,
    valid: bool,
}

impl HistoryRing {
    /// A ring of `slots` targets whose write index advances under `stride`.
    ///
    /// `slots` is floored at two: one target cannot be both the pass's colour
    /// attachment and one of its sampled sources. `stride` is clamped into
    /// `1..=slots`; a stride below `slots` leaves the tail targets unwritten,
    /// which is what a single frame in flight does to a two-target ring.
    ///
    /// The ring starts at slot zero with its history invalid, so the first
    /// frame passes the scene through instead of blending against a target that
    /// has never been rendered to.
    pub fn new(slots: usize, stride: usize) -> Self {
        let slots = slots.max(2);
        Self {
            slots,
            stride: stride.clamp(1, slots),
            write: 0,
            valid: false,
        }
    }

    /// A two-target ping-pong that alternates every frame. What a backend uses
    /// when nothing downstream is pre-bound to a particular slot.
    pub fn ping_pong() -> Self {
        Self::new(2, 2)
    }

    /// How many targets the ring holds.
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// The target this frame writes.
    pub fn write(&self) -> usize {
        self.write
    }

    /// The target this frame samples as history: the one the previous advance
    /// wrote.
    pub fn history(&self) -> usize {
        (self.write + self.slots - 1) % self.slots
    }

    /// Whether the history target holds an accumulated result. False on the
    /// first frame and after [`HistoryRing::invalidate`]; the pass then ignores
    /// history and seeds it from the current frame.
    pub fn valid(&self) -> bool {
        self.valid
    }

    /// Step to the next frame's write target, and mark the target just written
    /// as usable history. Called once per frame the pass actually ran.
    pub fn advance(&mut self) {
        self.write = (self.write + 1) % self.stride;
        self.valid = true;
    }

    /// Forget the accumulated history: the next frame passes through and
    /// accumulation restarts. Called when the targets are recreated, since a
    /// history rendered at another resolution cannot be reprojected into this
    /// one.
    pub fn invalidate(&mut self) {
        self.write = 0;
        self.valid = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn a_fresh_ring_has_no_history() {
        let r = HistoryRing::ping_pong();
        assert_eq!(r.write(), 0);
        assert!(!r.valid());
    }

    #[test]
    fn a_ping_pong_alternates_and_never_reads_what_it_writes() {
        let mut r = HistoryRing::ping_pong();
        for _ in 0..8 {
            assert_ne!(r.write(), r.history());
            r.advance();
        }
    }

    #[test]
    fn history_is_the_slot_the_previous_frame_wrote() {
        let mut r = HistoryRing::ping_pong();
        for _ in 0..8 {
            let wrote = r.write();
            r.advance();
            assert_eq!(r.history(), wrote);
        }
    }

    #[test]
    fn history_becomes_valid_after_one_frame_and_stays_valid() {
        let mut r = HistoryRing::ping_pong();
        assert!(!r.valid());
        r.advance();
        for _ in 0..8 {
            assert!(r.valid());
            r.advance();
        }
    }

    #[test]
    fn invalidating_restarts_accumulation_from_the_first_slot() {
        // A resize: the targets are recreated, so the accumulated history was
        // rendered at a resolution this frame cannot reproject from.
        let mut r = HistoryRing::ping_pong();
        r.advance();
        r.advance();
        r.invalidate();
        assert_eq!(r.write(), 0);
        assert!(!r.valid());
    }

    #[test]
    fn a_deeper_ring_walks_every_slot_before_repeating() {
        let mut r = HistoryRing::new(3, 3);
        let seen: Vec<usize> = (0..6)
            .map(|_| {
                let w = r.write();
                r.advance();
                w
            })
            .collect();
        assert_eq!(seen, [0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn a_stride_below_the_slot_count_pins_the_write_target() {
        // One frame in flight against a two-target ring: the write index never
        // leaves slot 0, so the pass samples a slot nothing writes.
        let mut r = HistoryRing::new(2, 1);
        for _ in 0..4 {
            assert_eq!(r.write(), 0);
            assert_eq!(r.history(), 1);
            r.advance();
        }
    }

    #[test]
    fn a_single_slot_request_is_floored_to_a_pair() {
        // A pass cannot sample the target it is writing, whatever it asks for.
        let r = HistoryRing::new(1, 1);
        assert_eq!(r.slots(), 2);
        assert_ne!(r.write(), r.history());
    }

    #[test]
    fn a_stride_past_the_slot_count_is_clamped_to_it() {
        let mut r = HistoryRing::new(2, 9);
        r.advance();
        assert_eq!(r.write(), 1);
        r.advance();
        assert_eq!(r.write(), 0);
    }

    #[test]
    fn a_frame_slot_driven_pass_lands_on_the_same_walk() {
        // A backend whose consumers are pre-bound per frame in flight passes the
        // frame slot to the pass rather than reading the ring's index, so the
        // two have to agree: a ring as deep as the frame count walks the frame
        // slots in the same order.
        let mut r = HistoryRing::new(3, 3);
        for frame in 0..6usize {
            assert_eq!(r.write(), frame % 3);
            r.advance();
        }
    }
}
