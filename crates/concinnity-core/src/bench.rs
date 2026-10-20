//! The timing harness in-module benchmarks report through: a calibrated
//! per-item cost beside the allocations one item causes. Shared by every
//! crate's ignored benchmark tests, so it ships behind the `test-support` feature
//! (and in this crate's own test build) rather than in the `no_std` surface.

use std::println;
use std::time::Instant;

// One measured pass runs at least this long before its time is trusted.
const TARGET_NS: u128 = 200_000_000;
const MAX_ITERS: u64 = 1 << 20;

/// How hard a benchmark pass drives its body.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Pace {
    /// Calibrate an iteration count, time the body, and report its per-item cost.
    Timed,
    /// Run the body exactly once and report nothing. What a non-ignored test
    /// drives, so a fixture that stopped building or started panicking fails on
    /// an ordinary run instead of waiting for someone to pass `--ignored`.
    Once,
}

/// Time `body` over a calibrated iteration count and report its per-item cost
/// beside the allocations one item causes. `items` is how many units of work one
/// call performs, so a number is comparable across fixture sizes.
///
/// The allocation pass is separate, so the counter reads sit outside the timed
/// window. A test binary that installs the tracking allocator
/// ([`crate::install_global_allocator`]) has live counters; one that does not
/// reports zero allocations. They are process-global, which is why every
/// benchmark asks for `--test-threads=1`: one running beside another reads the
/// other's allocations as its own.
pub fn bench<R>(pace: Pace, name: &str, items: u64, body: impl FnMut() -> R) {
    bench_over(TARGET_NS, pace, name, items, body);
}

// `bench` against an explicit measurement window, so a test can drive the
// calibration loop without waiting out a real one.
fn bench_over<R>(target_ns: u128, pace: Pace, name: &str, items: u64, mut body: impl FnMut() -> R) {
    if pace == Pace::Once {
        core::hint::black_box(body());
        return;
    }

    let mut iters: u64 = 1;
    loop {
        let start = Instant::now();
        for _ in 0..iters {
            core::hint::black_box(body());
        }
        if start.elapsed().as_nanos() >= target_ns || iters >= MAX_ITERS {
            break;
        }
        iters = iters.saturating_mul(4).min(MAX_ITERS);
    }

    let start = Instant::now();
    for _ in 0..iters {
        core::hint::black_box(body());
    }
    let elapsed = start.elapsed();

    let before = crate::memory::alloc_count().unwrap_or(0);
    for _ in 0..iters {
        core::hint::black_box(body());
    }
    let allocated = crate::memory::alloc_count()
        .unwrap_or(0)
        .saturating_sub(before);

    let units = (iters * items.max(1)) as f64;
    let per_item_ns = elapsed.as_secs_f64() * 1e9 / units;
    let allocs = allocated as f64 / units;
    println!("  {name:<40} {per_item_ns:>10.2} ns/item {allocs:>10.3} allocs/item");
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicU32, Ordering};

    // A single-run pass drives the body exactly once and reports nothing.
    #[test]
    fn a_single_run_pass_drives_the_body_once() {
        let runs = AtomicU32::new(0);
        bench(Pace::Once, "test/once", 4, || {
            runs.fetch_add(1, Ordering::Relaxed)
        });
        assert_eq!(runs.load(Ordering::Relaxed), 1);
    }

    // A measured pass calibrates an iteration count against its window, then
    // runs that count twice more: once timed, once counting allocations. The
    // window here is a microsecond rather than the real fifth of a second, so
    // the loop is driven rather than waited out.
    #[test]
    fn a_measured_pass_calibrates_then_times_and_counts() {
        let runs = AtomicU32::new(0);
        bench_over(1_000, Pace::Timed, "test/timed", 1, || {
            runs.fetch_add(1, Ordering::Relaxed)
        });
        assert!(
            runs.load(Ordering::Relaxed) >= 3,
            "a measured pass runs the body far more than a single-run one"
        );
    }

    // A body slow enough to fill the window on its first iteration never
    // multiplies the count, so the calibration cannot run away.
    #[test]
    fn a_body_that_fills_the_window_stays_at_one_iteration() {
        let runs = AtomicU32::new(0);
        bench_over(0, Pace::Timed, "test/wide", 0, || {
            runs.fetch_add(1, Ordering::Relaxed)
        });
        assert_eq!(
            runs.load(Ordering::Relaxed),
            3,
            "one calibration pass, one timed, one counted"
        );
    }
}
