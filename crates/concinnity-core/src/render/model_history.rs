//! Decides, per cull record, whether the model-history ring holds a usable
//! previous-frame transform for the G-buffer pre-pass's motion vectors.
//!
//! The history ring is filled on the GPU by `model_history.slang`, which copies
//! this frame's model matrices straight out of the bindless object buffer. That
//! makes a history entry meaningful only while its record keeps its occupant: a
//! recycled draw slot, a runtime reserve that repacked around a
//! streamed-in chunk, or the frames before the ring has been written at all
//! would otherwise reproject through a stranger's transform and smear under TAA.
//!
//! Each backend runs [`ModelHistory::begin`] once per frame and then asks for
//! every record's flag bits while it builds the draw-args buffer. A record this
//! reports stale carries [`crate::gfx::render_types::draw_args_no_history`], and
//! the pre-pass reprojects it through its own current model, which is a zero
//! model-delta rather than a wrong one.

use alloc::vec::Vec;

use crate::gfx::render_types::draw_args_no_history;

/// How a frame's draw-args build treats the model-history ring.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HistoryMode {
    /// The pre-pass fills and reads the ring this frame: flag only the records
    /// whose occupant moved since the snapshot was taken.
    Track,
    /// The ring is not being filled this frame -- the pre-pass is off, or no
    /// consumer reads motion -- so nothing in it can be trusted when it returns:
    /// flag every record and re-prime.
    #[default]
    Stale,
    /// A reflection-probe bake, building its own records into its own buffers:
    /// flag every record and leave the frame's tracker alone.
    Untracked,
}

// Occupant kinds, in the token's top bits, so a draw slot and a skinned slot
// that share an index never compare equal.
const KIND_DRAW: u64 = 0;
const KIND_SKINNED: u64 = 1;

// The token stored for a record no frame has observed yet. Distinct from every
// real token, whose kind occupies only the low two bits of the top byte.
const UNOBSERVED: u64 = u64::MAX;

/// Per-record previous-frame-transform validity for the GPU-filled model
/// history ring.
#[derive(Debug, Default)]
pub struct ModelHistory {
    // Bumped whenever a draw slot takes a new occupant, so a record that keeps
    // its index still reads as changed.
    draw_gen: Vec<u32>,
    // The same for the skinned tail's instance pool.
    skinned_gen: Vec<u32>,
    // Occupant token per cull record, as of the frame that last observed it.
    observed: Vec<u64>,
    // Set by `reset`: the ring holds nothing this frame's records were written
    // for, so every slot wants filling before it is read.
    prime: bool,
    // This frame's mode, from `begin`.
    mode: HistoryMode,
}

// `kind`'s occupant at `index`, at generation `generation`.
fn token(kind: u64, generation: u32, index: usize) -> u64 {
    (kind << 62) | ((generation as u64 & 0x3FFF_FFFF) << 32) | (index as u64 & 0xFFFF_FFFF)
}

impl ModelHistory {
    /// An empty tracker; [`Self::reset`] sizes it to the world.
    pub fn new() -> Self {
        Self::default()
    }

    /// Size the tracker to a world of `n_cull` records and invalidate every
    /// one: the history ring's buffers have just been (re)allocated, so nothing
    /// in them was written for these records. Occupant generations survive, so
    /// a slot reused across the rebuild still reads as changed.
    pub fn reset(&mut self, n_cull: usize) {
        self.observed.clear();
        self.observed.resize(n_cull, UNOBSERVED);
        self.prime = true;
    }

    /// Open a draw-args build over `n_cull` records. Call once per build, before
    /// any of the flag queries; a `Track` build in steady state is a no-op.
    pub fn begin(&mut self, mode: HistoryMode, n_cull: usize) {
        self.mode = mode;
        match mode {
            HistoryMode::Track if self.observed.len() == n_cull => {}
            HistoryMode::Track | HistoryMode::Stale => self.reset(n_cull),
            HistoryMode::Untracked => {}
        }
    }

    /// The `GpuDrawArgs::flags` bits cull record `record` needs for the draw
    /// slot `draw_idx` now filling it: `NO_HISTORY` when the ring's entry was
    /// written for a different occupant, nothing when it can be trusted. Call
    /// exactly once per record per build.
    pub fn draw_flags(&mut self, record: usize, draw_idx: usize) -> u32 {
        if self.mode != HistoryMode::Track {
            return draw_args_no_history();
        }
        let generation = self.draw_gen.get(draw_idx).copied().unwrap_or(0);
        self.flags(record, token(KIND_DRAW, generation, draw_idx))
    }

    /// [`Self::draw_flags`] for a record in the skinned tail.
    pub fn skinned_flags(&mut self, record: usize, skinned_idx: usize) -> u32 {
        if self.mode != HistoryMode::Track {
            return draw_args_no_history();
        }
        let generation = self.skinned_gen.get(skinned_idx).copied().unwrap_or(0);
        self.flags(record, token(KIND_SKINNED, generation, skinned_idx))
    }

    fn flags(&mut self, record: usize, token: u64) -> u32 {
        match self.observe(record, token) {
            true => draw_args_no_history(),
            false => 0,
        }
    }

    /// Whether the ring's slots must all be filled before this frame reads one.
    /// True once per [`Self::reset`]; the caller dispatches the history kernel
    /// into every slot on that frame instead of only its own.
    pub fn take_prime(&mut self) -> bool {
        core::mem::take(&mut self.prime)
    }

    /// Note that draw slot `draw_idx` now holds a different object. Call
    /// wherever a slot is written for a new occupant -- a reused or appended
    /// draw slot, a streamed chunk moving in, a spawned clone.
    pub fn reoccupy_draw(&mut self, draw_idx: usize) {
        bump(&mut self.draw_gen, draw_idx);
    }

    /// Note that skinned instance `skinned_idx` now holds a different object,
    /// as a revealed instance-pool slot does.
    pub fn reoccupy_skinned(&mut self, skinned_idx: usize) {
        bump(&mut self.skinned_gen, skinned_idx);
    }

    fn observe(&mut self, record: usize, token: u64) -> bool {
        let Some(slot) = self.observed.get_mut(record) else {
            // A record past the tracked range has no history to trust.
            return true;
        };
        let stale = *slot != token;
        *slot = token;
        stale
    }
}

// Bump `slots[index]`, growing the vec to reach it.
fn bump(slots: &mut Vec<u32>, index: usize) {
    if index >= slots.len() {
        slots.resize(index + 1, 0);
    }
    slots[index] = slots[index].wrapping_add(1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::render_types::draw_args_no_history;

    const KEEP: u32 = 0;

    // A record keeping its occupant is stale on the first observation (nothing
    // was ever written for it) and trusted from the second frame on.
    #[test]
    fn a_settled_record_is_stale_once_then_trusted() {
        let mut h = ModelHistory::new();
        h.begin(HistoryMode::Track, 4);
        assert_eq!(h.draw_flags(2, 2), draw_args_no_history());
        h.begin(HistoryMode::Track, 4);
        assert_eq!(h.draw_flags(2, 2), KEEP);
        h.begin(HistoryMode::Track, 4);
        assert_eq!(h.draw_flags(2, 2), KEEP);
    }

    // Reusing a draw slot in place keeps the record index but changes the
    // occupant, which is exactly the ghosting case the flag exists for.
    #[test]
    fn reoccupying_a_slot_invalidates_its_record_for_one_frame() {
        let mut h = ModelHistory::new();
        h.begin(HistoryMode::Track, 4);
        h.draw_flags(1, 1);
        h.begin(HistoryMode::Track, 4);
        assert_eq!(h.draw_flags(1, 1), KEEP);
        h.reoccupy_draw(1);
        h.begin(HistoryMode::Track, 4);
        assert_eq!(h.draw_flags(1, 1), draw_args_no_history());
        h.begin(HistoryMode::Track, 4);
        assert_eq!(h.draw_flags(1, 1), KEEP);
    }

    // The runtime reserve repacks when a chunk streams in or out, so a record
    // can be handed a different draw slot without either slot being reused.
    #[test]
    fn a_repacked_record_is_stale_even_though_neither_slot_changed() {
        let mut h = ModelHistory::new();
        h.begin(HistoryMode::Track, 8);
        h.draw_flags(5, 40);
        h.begin(HistoryMode::Track, 8);
        assert_eq!(h.draw_flags(5, 40), KEEP);
        // The reserve shifted: record 5 now carries draw slot 41.
        h.begin(HistoryMode::Track, 8);
        assert_eq!(h.draw_flags(5, 41), draw_args_no_history());
        h.begin(HistoryMode::Track, 8);
        assert_eq!(h.draw_flags(5, 41), KEEP);
    }

    // The two occupant pools index from zero independently, so a skinned tail
    // record must not be settled by the static prefix's observation.
    #[test]
    fn draw_and_skinned_occupants_never_alias() {
        let mut h = ModelHistory::new();
        h.begin(HistoryMode::Track, 4);
        assert_eq!(h.draw_flags(0, 3), draw_args_no_history());
        assert_eq!(h.skinned_flags(1, 3), draw_args_no_history());
        h.begin(HistoryMode::Track, 4);
        assert_eq!(h.draw_flags(0, 3), KEEP);
        assert_eq!(h.skinned_flags(1, 3), KEEP);
        h.reoccupy_skinned(3);
        h.begin(HistoryMode::Track, 4);
        // Only the skinned pool moved.
        assert_eq!(h.draw_flags(0, 3), KEEP);
        assert_eq!(h.skinned_flags(1, 3), draw_args_no_history());
    }

    // A record count change reallocates the ring, so every record is stale
    // again and every slot wants filling before it is read.
    #[test]
    fn a_record_count_change_invalidates_everything_and_asks_for_a_prime() {
        let mut h = ModelHistory::new();
        h.begin(HistoryMode::Track, 3);
        assert!(h.take_prime());
        for r in 0..3 {
            assert_eq!(h.draw_flags(r, r), draw_args_no_history());
        }
        h.begin(HistoryMode::Track, 3);
        assert!(!h.take_prime());
        for r in 0..3 {
            assert_eq!(h.draw_flags(r, r), KEEP);
        }
        h.begin(HistoryMode::Track, 5);
        assert!(h.take_prime());
        for r in 0..3 {
            assert_eq!(h.draw_flags(r, r), draw_args_no_history());
        }
    }

    // A frame the pre-pass sits out leaves the ring stale, so every record is
    // flagged and the next tracked frame starts over from a prime.
    #[test]
    fn a_stale_frame_flags_everything_and_re_primes() {
        let mut h = ModelHistory::new();
        h.begin(HistoryMode::Track, 2);
        h.draw_flags(0, 0);
        h.begin(HistoryMode::Track, 2);
        assert_eq!(h.draw_flags(0, 0), KEEP);
        h.take_prime();
        h.begin(HistoryMode::Stale, 2);
        assert_eq!(h.draw_flags(0, 0), draw_args_no_history());
        assert!(h.take_prime());
        h.begin(HistoryMode::Track, 2);
        assert_eq!(h.draw_flags(0, 0), draw_args_no_history());
    }

    // A probe bake builds its own records into its own buffers: it flags
    // everything but must not disturb the frame's own tracking.
    #[test]
    fn an_untracked_build_leaves_the_frames_tracker_alone() {
        let mut h = ModelHistory::new();
        h.begin(HistoryMode::Track, 2);
        h.draw_flags(0, 0);
        assert!(h.take_prime());
        h.begin(HistoryMode::Untracked, 2);
        assert_eq!(h.draw_flags(0, 0), draw_args_no_history());
        assert!(!h.take_prime());
        h.begin(HistoryMode::Track, 2);
        assert_eq!(h.draw_flags(0, 0), KEEP);
    }

    // A record past the tracked range (a world that grew its reserve without a
    // rebuild) never claims a history it does not have.
    #[test]
    fn a_record_past_the_tracked_range_is_always_stale() {
        let mut h = ModelHistory::new();
        h.begin(HistoryMode::Track, 2);
        assert_eq!(h.draw_flags(7, 7), draw_args_no_history());
        h.begin(HistoryMode::Track, 2);
        assert_eq!(h.draw_flags(7, 7), draw_args_no_history());
    }

    // Generations are bumped for slots the tracker has never sized for, so a
    // reoccupy that arrives before the first observation still counts.
    #[test]
    fn reoccupying_an_untracked_slot_grows_the_generation_table() {
        let mut h = ModelHistory::new();
        h.begin(HistoryMode::Track, 4);
        h.reoccupy_draw(9);
        assert_eq!(h.draw_flags(0, 9), draw_args_no_history());
        h.begin(HistoryMode::Track, 4);
        assert_eq!(h.draw_flags(0, 9), KEEP);
        h.reoccupy_draw(9);
        h.begin(HistoryMode::Track, 4);
        assert_eq!(h.draw_flags(0, 9), draw_args_no_history());
    }
}
