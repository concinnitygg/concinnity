//! GPU buffer memory every backend manages the same way: range and block
//! sub-allocation, growth policy, frame-retired frees, per-frame upload rings
//! and recorded staging uploads.

mod block_alloc;
pub mod block_pool;
pub mod growth;
pub mod range_alloc;
pub mod retire_pool;
pub mod staging;
pub mod staging_uploads;
pub mod upload_ring;

/// Round `offset` up to the next multiple of `align` (a power of two). An
/// `align` of 0 or 1 returns `offset` unchanged.
pub fn align_up(offset: u64, align: u64) -> u64 {
    if align <= 1 {
        return offset;
    }
    debug_assert!(
        align.is_power_of_two(),
        "align_up: {align} is not a power of two"
    );
    (offset + align - 1) & !(align - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_rounds_to_multiple() {
        assert_eq!(align_up(0, 16), 0);
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
        assert_eq!(align_up(123, 1), 123);
        assert_eq!(align_up(123, 0), 123);
    }
}
