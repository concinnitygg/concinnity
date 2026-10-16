//! Per-pass GPU timing slot arithmetic, shared by every backend that times
//! passes with a timestamp query pool.
//!
//! The pool holds one block of [`SLOTS_PER_FRAME`] u64 slots per in-flight
//! frame. Within a block the whole-frame timer owns slots [0, 1] and one
//! (start, end) pair per [`PassId`] follows, so `pass` occupies
//! `2 + 2 * pass` and `3 + 2 * pass`. Keeping the whole-frame pair at the front
//! lets a whole-frame readback stay the first pair of the block, independent of
//! how many passes exist.
//!
//! This is index arithmetic with no device type in it, so it lives here rather
//! than in a backend and its layout tests run on every platform's CI. The
//! DirectX and Vulkan backends decode their readback through
//! [`decode_frame_block`].

use crate::profile::{MAX_PASS_TIMINGS, PassTiming};
use crate::render::render_graph::{PASS_COUNT, PASS_NAMES, PassId};

// Every pass has a slot in the per-pass timing table.
const _: () = assert!(PASS_COUNT <= MAX_PASS_TIMINGS);

/// Per-frame block: `[whole_frame_start, whole_frame_end, pass0_start,
/// pass0_end, ..., pass(PASS_COUNT-1)_end]`.
pub const SLOTS_PER_FRAME: usize = 2 * (PASS_COUNT + 1);

/// Bytes one frame's block occupies in a readback buffer of u64 timestamps.
pub const FRAME_BLOCK_BYTES: u64 = (SLOTS_PER_FRAME * 8) as u64;

/// First slot of `frame`'s block. Also the start of the range a per-frame
/// resolve should walk, paired with [`SLOTS_PER_FRAME`] as the count.
pub const fn frame_block_base(frame: usize) -> u32 {
    (frame * SLOTS_PER_FRAME) as u32
}

/// Byte offset into the readback buffer where `frame`'s block begins.
pub const fn frame_readback_byte_offset(frame: usize) -> u64 {
    frame as u64 * FRAME_BLOCK_BYTES
}

/// (start, end) slots for `frame`'s whole-frame pair.
pub const fn whole_frame_pair(frame: usize) -> (u32, u32) {
    let base = frame_block_base(frame);
    (base, base + 1)
}

/// (start, end) slots for `pass` within `frame`'s block.
pub const fn pass_pair(frame: usize, pass: PassId) -> (u32, u32) {
    let base = frame_block_base(frame) + 2 + 2 * (pass as u32);
    (base, base + 1)
}

/// Decode one frame's block into the whole-frame micros and a per-pass table in
/// [`PassId`] order. `pair_micros(start_slot, end_slot)` is the backend's reading
/// of one pair, with slots relative to the block; unused table entries keep the
/// `("", 0)` sentinel.
pub fn decode_frame_block(
    pair_micros: impl Fn(usize, usize) -> u32,
) -> (u32, [PassTiming; MAX_PASS_TIMINGS]) {
    let frame_us = pair_micros(0, 1);
    let mut passes = [("", 0u32); MAX_PASS_TIMINGS];
    for (i, name) in PASS_NAMES.iter().enumerate() {
        passes[i] = (*name, pair_micros(2 + 2 * i, 3 + 2 * i));
    }
    (frame_us, passes)
}

/// One frame's GPU span on the queue that paces frames, split against the
/// frame before it.
///
/// Consecutive frames overlap on the GPU: the next frame's leading compute
/// starts while the previous frame's last passes still run. A frame's span is
/// then its latency, and summing spans counts the overlap twice, so the span
/// is clipped at the previous frame's end. The clipped part is what the frame
/// added to the GPU timeline, which in steady state is at most the frame
/// interval.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameSpan {
    /// Span time after the previous frame's end.
    pub own: u64,
    /// Span time shared with the previous frame.
    pub overlap: u64,
}

impl FrameSpan {
    /// Split the span `start..end` against `previous_end`, the end of the
    /// frame before it, in the same timestamp units.
    pub fn after(start: u64, end: u64, previous_end: u64) -> Self {
        let span = end.saturating_sub(start);
        let overlap = previous_end.saturating_sub(start).min(span);
        Self {
            own: span - overlap,
            overlap,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_after_the_previous_one_keeps_its_whole_span() {
        assert_eq!(
            FrameSpan::after(100, 130, 90),
            FrameSpan {
                own: 30,
                overlap: 0
            }
        );
    }

    #[test]
    fn a_frame_started_under_the_previous_one_owns_only_what_follows_it() {
        // Spans of 30 every 25: each frame shares 5 with the one before, and
        // what it owns is the interval.
        assert_eq!(
            FrameSpan::after(125, 155, 130),
            FrameSpan {
                own: 25,
                overlap: 5
            }
        );
    }

    #[test]
    fn a_frame_inside_the_previous_one_owns_nothing() {
        assert_eq!(
            FrameSpan::after(100, 120, 150),
            FrameSpan {
                own: 0,
                overlap: 20
            }
        );
        assert_eq!(FrameSpan::after(100, 90, 0), FrameSpan::default());
    }

    #[test]
    fn whole_frame_pair_heads_each_block() {
        assert_eq!(whole_frame_pair(0), (0, 1));
        assert_eq!(
            whole_frame_pair(1),
            (SLOTS_PER_FRAME as u32, SLOTS_PER_FRAME as u32 + 1)
        );
        assert_eq!(frame_block_base(0), 0);
        assert_eq!(frame_block_base(2), 2 * SLOTS_PER_FRAME as u32);
        assert_eq!(frame_readback_byte_offset(0), 0);
        assert_eq!(frame_readback_byte_offset(1), FRAME_BLOCK_BYTES);
    }

    #[test]
    fn pass_pair_skips_the_whole_frame_pair() {
        assert_eq!(pass_pair(0, PassId::Cull), (2, 3));
    }

    #[test]
    fn every_pass_owns_a_distinct_pair_inside_the_block() {
        // Driven by `PassId::ALL`, so a new pass is covered here the moment it
        // is registered rather than when someone remembers to extend a list.
        let mut seen = hashbrown::HashSet::new();
        // The whole-frame pair owns slots 0, 1.
        assert!(seen.insert(0) && seen.insert(1));
        for pass in PassId::ALL {
            let (s, e) = pass_pair(0, pass);
            assert!(seen.insert(s), "duplicate start slot for {pass:?}");
            assert!(seen.insert(e), "duplicate end slot for {pass:?}");
            assert!((e as usize) < SLOTS_PER_FRAME);
        }
        // The block is exactly the whole-frame pair plus one pair per pass.
        assert_eq!(seen.len(), SLOTS_PER_FRAME);
    }

    #[test]
    fn decode_frame_block_pairs_every_name_with_its_slots() {
        // Encodes both slots into the reading so a swapped or shifted pair shows.
        let (frame_us, passes) = decode_frame_block(|s, e| (s * 1000 + e) as u32);
        assert_eq!(frame_us, 1);
        for pass in PassId::ALL {
            let (s, e) = pass_pair(0, pass);
            assert_eq!(passes[pass as usize], (pass.name(), s * 1000 + e));
        }
        assert!(passes[PASS_COUNT..].iter().all(|&t| t == ("", 0)));
    }
}
