//! How a reusable buffer grows when a request outgrows it.

/// The capacity a buffer holding `have` bytes must be reallocated to so it holds
/// `needed`, or `None` when it already does. Growth rounds up to a power of two,
/// so a slowly growing request stops reallocating, and never lands below
/// `floor`, which keeps small buffers off the allocator's smallest classes. A
/// request past the largest power of two gets exactly what it asked for.
pub fn grow_capacity(have: u64, needed: u64, floor: u64) -> Option<u64> {
    if have >= needed {
        return None;
    }
    Some(
        needed
            .checked_next_power_of_two()
            .unwrap_or(needed)
            .max(floor),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_buffer_that_fits_is_kept() {
        assert_eq!(grow_capacity(256, 200, 256), None);
        assert_eq!(grow_capacity(256, 256, 256), None);
        assert_eq!(grow_capacity(0, 0, 256), None);
    }

    #[test]
    fn growth_starts_at_the_floor() {
        assert_eq!(grow_capacity(0, 1, 256), Some(256));
        assert_eq!(grow_capacity(0, 1, 64 * 1024), Some(64 * 1024));
    }

    #[test]
    fn a_request_past_the_largest_power_of_two_is_met_exactly() {
        let needed = (1u64 << 63) + 1;
        assert_eq!(grow_capacity(0, needed, 256), Some(needed));
    }

    #[test]
    fn growth_rounds_up_to_a_power_of_two() {
        assert_eq!(grow_capacity(0, 300, 256), Some(512));
        assert_eq!(grow_capacity(512, 513, 256), Some(1024));
        assert_eq!(grow_capacity(1024, 4096, 256), Some(4096));
    }

    // Doubling a power-of-two capacity until it fits lands on the same size as
    // rounding the request up, which is what lets one rule serve both callers.
    #[test]
    fn rounding_up_matches_doubling_a_power_of_two_capacity() {
        let floor = 64 * 1024;
        for have in [0u64, floor, floor * 4] {
            for needed in [1u64, floor - 1, floor + 1, floor * 3 + 1, floor * 9] {
                let mut doubled = have.max(floor);
                while doubled < needed {
                    doubled *= 2;
                }
                let expected = (have < needed).then_some(doubled);
                assert_eq!(
                    grow_capacity(have, needed, floor),
                    expected,
                    "{have} {needed}"
                );
            }
        }
    }
}
