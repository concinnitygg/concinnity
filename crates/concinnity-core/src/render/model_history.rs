//! Decides, per cull record, whether the model-history ring holds a usable
//! previous-frame transform for the G-buffer pre-pass's motion vectors.
//!
//! The history ring is filled on the GPU by `model_history.hlsl`, which copies
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

use crate::gfx::render_types::{DrawIndex, SkinnedIndex, draw_args_no_history};

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
    // Set by `forget`: the next tracked build treats the ring as stale.
    forget: bool,
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
        let mode = match mode {
            HistoryMode::Track if core::mem::take(&mut self.forget) => HistoryMode::Stale,
            mode => mode,
        };
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
        self.window().draw_flags(record, draw_idx)
    }

    /// [`Self::draw_flags`] for a record in the skinned tail.
    pub fn skinned_flags(&mut self, record: usize, skinned_idx: usize) -> u32 {
        self.window().skinned_flags(record, skinned_idx)
    }

    /// Every record, as one window that [`HistoryWindow::split_at`] divides
    /// into disjoint runs a build can query from separate threads.
    pub fn window(&mut self) -> HistoryWindow<'_> {
        HistoryWindow {
            mode: self.mode,
            draw_gen: &self.draw_gen,
            skinned_gen: &self.skinned_gen,
            observed: &mut self.observed,
            first: 0,
        }
    }

    /// Whether the ring's slots must all be filled before this frame reads one.
    /// True once per [`Self::reset`]; the caller dispatches the history kernel
    /// into every slot on that frame instead of only its own.
    pub fn take_prime(&mut self) -> bool {
        core::mem::take(&mut self.prime)
    }

    /// Ask for a prime again: a frame that took one failed before its history
    /// snapshot reached the GPU, so the ring may still hold unwritten slots.
    pub fn request_prime(&mut self) {
        self.prime = true;
    }

    /// Distrust every transform in the ring for the next tracked build: the
    /// world moved to another frame, so last frame's transforms place last
    /// frame's objects in the old one. Each record reprojects through its own
    /// current transform for that frame, which is exact for anything that
    /// did not move.
    pub fn forget(&mut self) {
        self.forget = true;
    }

    /// Note that draw slot `draw_idx` now holds a different object. Call
    /// wherever a slot is written for a new occupant -- a reused or appended
    /// draw slot, a streamed chunk moving in, a spawned clone.
    pub fn reoccupy_draw(&mut self, draw_idx: DrawIndex) {
        bump(&mut self.draw_gen, draw_idx.index());
    }

    /// Note that skinned instance `skinned_idx` now holds a different object,
    /// as a revealed instance-pool slot does.
    pub fn reoccupy_skinned(&mut self, skinned_idx: SkinnedIndex) {
        bump(&mut self.skinned_gen, skinned_idx.index());
    }
}

/// A run of cull records lent out of a [`ModelHistory`] for one build: the
/// flag queries for records `first..`, writing only this run's entries.
#[derive(Debug)]
pub struct HistoryWindow<'a> {
    mode: HistoryMode,
    draw_gen: &'a [u32],
    skinned_gen: &'a [u32],
    // Entries for records `first..first + observed.len()`. Shorter than the
    // run when the tracker is: a record past the tracked range reads stale.
    observed: &'a mut [u64],
    first: usize,
}

impl<'a> HistoryWindow<'a> {
    /// The first record this window answers for.
    pub fn first(&self) -> usize {
        self.first
    }

    /// Split into records `first..first + mid` and the rest.
    pub fn split_at(self, mid: usize) -> (Self, Self) {
        let (mode, draw_gen, skinned_gen, first) =
            (self.mode, self.draw_gen, self.skinned_gen, self.first);
        let (head, tail) = self.observed.split_at_mut(mid.min(self.observed.len()));
        let part = |observed, first| Self {
            mode,
            draw_gen,
            skinned_gen,
            observed,
            first,
        };
        (part(head, first), part(tail, first + mid))
    }

    /// [`ModelHistory::draw_flags`], for a record in this window.
    pub fn draw_flags(&mut self, record: usize, draw_idx: usize) -> u32 {
        if self.mode != HistoryMode::Track {
            return draw_args_no_history();
        }
        let generation = self.draw_gen.get(draw_idx).copied().unwrap_or(0);
        self.flags(record, token(KIND_DRAW, generation, draw_idx))
    }

    /// [`ModelHistory::skinned_flags`], for a record in this window.
    pub fn skinned_flags(&mut self, record: usize, skinned_idx: usize) -> u32 {
        if self.mode != HistoryMode::Track {
            return draw_args_no_history();
        }
        let generation = self.skinned_gen.get(skinned_idx).copied().unwrap_or(0);
        self.flags(record, token(KIND_SKINNED, generation, skinned_idx))
    }

    fn flags(&mut self, record: usize, token: u64) -> u32 {
        let slot = record
            .checked_sub(self.first)
            .and_then(|i| self.observed.get_mut(i));
        // A record past the tracked range has no history to trust.
        let Some(slot) = slot else {
            return draw_args_no_history();
        };
        let stale = *slot != token;
        *slot = token;
        match stale {
            true => draw_args_no_history(),
            false => 0,
        }
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
        h.reoccupy_draw(DrawIndex(1));
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
        h.reoccupy_skinned(SkinnedIndex(3));
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

    // A forgotten ring is stale for the next tracked build only, and a probe
    // bake in between does not use the forgetting up.
    #[test]
    fn a_forgotten_ring_is_stale_for_one_tracked_build() {
        let mut h = ModelHistory::new();
        h.begin(HistoryMode::Track, 2);
        h.draw_flags(0, 0);
        h.begin(HistoryMode::Track, 2);
        assert_eq!(h.draw_flags(0, 0), KEEP);
        h.take_prime();
        h.forget();
        h.begin(HistoryMode::Untracked, 2);
        h.begin(HistoryMode::Track, 2);
        assert_eq!(h.draw_flags(0, 0), draw_args_no_history());
        assert!(h.take_prime(), "every slot is refilled in the new frame");
        h.begin(HistoryMode::Track, 2);
        h.draw_flags(0, 0);
        h.begin(HistoryMode::Track, 2);
        assert_eq!(h.draw_flags(0, 0), KEEP);
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

    // A frame that runs no history snapshot leaves the prime untaken, and it
    // must still be pending for the first frame that does, across both steady
    // tracked builds and probe bakes in between.
    #[test]
    fn an_untaken_prime_stays_pending_until_taken() {
        let mut h = ModelHistory::new();
        h.begin(HistoryMode::Track, 3);
        h.begin(HistoryMode::Track, 3);
        h.begin(HistoryMode::Untracked, 3);
        h.begin(HistoryMode::Track, 3);
        assert!(h.take_prime());
        h.begin(HistoryMode::Track, 3);
        assert!(!h.take_prime());
    }

    // A frame that fails after taking the prime hands it back, and the next
    // frame takes it as if the failed one had never run.
    #[test]
    fn a_requested_prime_is_taken_once_by_the_next_frame() {
        let mut h = ModelHistory::new();
        h.begin(HistoryMode::Track, 2);
        assert!(h.take_prime());
        h.request_prime();
        h.begin(HistoryMode::Track, 2);
        assert!(h.take_prime());
        assert!(!h.take_prime());
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
        h.reoccupy_draw(DrawIndex(9));
        assert_eq!(h.draw_flags(0, 9), draw_args_no_history());
        h.begin(HistoryMode::Track, 4);
        assert_eq!(h.draw_flags(0, 9), KEEP);
        h.reoccupy_draw(DrawIndex(9));
        h.begin(HistoryMode::Track, 4);
        assert_eq!(h.draw_flags(0, 9), draw_args_no_history());
    }

    // Two windows split from one build observe exactly the records a single
    // build over the whole tracker would, each touching only its own run.
    #[test]
    fn split_windows_observe_like_one_build() {
        let mut whole = ModelHistory::new();
        let mut split = ModelHistory::new();
        whole.reoccupy_draw(DrawIndex(2));
        split.reoccupy_draw(DrawIndex(2));
        for frame in 0..3 {
            whole.begin(HistoryMode::Track, 6);
            split.begin(HistoryMode::Track, 6);
            if frame == 2 {
                whole.reoccupy_draw(DrawIndex(4));
                split.reoccupy_draw(DrawIndex(4));
            }
            let expected: Vec<u32> = (0..6).map(|i| whole.draw_flags(i, i)).collect();
            let (mut head, mut tail) = split.window().split_at(4);
            assert_eq!((head.first(), tail.first()), (0, 4));
            let mut got: Vec<u32> = (0..4).map(|i| head.draw_flags(i, i)).collect();
            got.extend((4..6).map(|i| tail.draw_flags(i, i)));
            assert_eq!(got, expected, "frame {frame}");
        }
    }

    // A window answers only for its own run: a record before it, or past the
    // tracked range, reads stale without touching another window's entries.
    #[test]
    fn a_window_reads_records_outside_its_run_as_stale() {
        let mut h = ModelHistory::new();
        h.begin(HistoryMode::Track, 4);
        h.window().draw_flags(1, 1);
        h.begin(HistoryMode::Track, 4);
        let (_, mut tail) = h.window().split_at(2);
        assert_eq!(tail.draw_flags(1, 1), draw_args_no_history());
        let (_, mut past) = h.window().split_at(9);
        assert_eq!(past.draw_flags(9, 9), draw_args_no_history());
        assert_eq!(h.draw_flags(1, 1), KEEP);
    }

    #[test]
    fn an_untracked_window_flags_everything() {
        let mut h = ModelHistory::new();
        h.begin(HistoryMode::Track, 2);
        h.draw_flags(0, 0);
        h.begin(HistoryMode::Untracked, 2);
        let (mut head, _) = h.window().split_at(1);
        assert_eq!(head.draw_flags(0, 0), draw_args_no_history());
        assert_eq!(head.skinned_flags(0, 0), draw_args_no_history());
    }
}
