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
//! DirectX and Vulkan backends re-export it under their own `pass_timing`.

use crate::render_graph::{PASS_COUNT, PassId};

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
