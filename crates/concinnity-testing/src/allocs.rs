// Pinning a step's heap allocations under a harness that runs other tests on
// parallel threads.
//
// The allocation counters are process-wide, so one step's delta is only an
// upper bound on its own cost. Concurrency can only add allocations, never hide
// them, so stepping until one delta lands at or under the budget proves the
// step's own cost is within it, while a regression never produces such a step.

use std::time::{Duration, Instant};

// Generous: a steady step lands within a few tries once the concurrent noise
// drains, so only a real regression spends the whole window.
const DEADLINE: Duration = Duration::from_secs(10);

/// Run `step` until one call allocates at most `budget` times, and return the
/// fewest allocations any call made: at most `budget` when the pin holds, or
/// the closest ten seconds of calls came when it does not.
///
/// Panics when the binary has not installed the tracking allocator.
pub fn quietest_allocs(budget: u64, step: impl FnMut()) -> u64 {
    quietest_allocs_within(DEADLINE, budget, step)
}

fn quietest_allocs_within(timeout: Duration, budget: u64, mut step: impl FnMut()) -> u64 {
    let deadline = Instant::now() + timeout;
    let mut min = u64::MAX;
    loop {
        let before = concinnity_core::memory::alloc_count().expect("the binary tracks its heap");
        step();
        let after = concinnity_core::memory::alloc_count().expect("the allocator stays installed");
        min = min.min(after - before);
        if min <= budget || Instant::now() >= deadline {
            return min;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    concinnity_core::install_global_allocator!();

    #[test]
    fn a_step_that_allocates_nothing_holds_a_zero_pin() {
        assert_eq!(quietest_allocs(0, || {}), 0);
    }

    // Other threads can only add to a delta, so the fewest is never below the
    // step's own two.
    #[test]
    fn a_step_over_its_budget_never_reports_less_than_it_made() {
        let min = quietest_allocs_within(Duration::from_millis(20), 0, || {
            std::hint::black_box(vec![0u8; 16]);
            std::hint::black_box(vec![0u8; 16]);
        });
        assert!(min >= 2, "{min}");
    }
}
