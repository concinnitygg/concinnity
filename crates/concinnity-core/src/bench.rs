//! The timing harness in-module benchmarks report through: a calibrated
//! per-item cost beside the allocations one item causes. Shared by every
//! crate's ignored benchmark tests, so it ships behind the `test-support` feature
//! (and in this crate's own test build) rather than in the `no_std` surface.

use crate::math::SaturatingFrom;
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

/// Deterministic xorshift generator, so benchmark fixtures lay out identically
/// across runs and machines.
pub struct Rng(u64);

impl Rng {
    /// A generator seeded with `seed`; zero, the xorshift fixed point, is
    /// replaced by a fixed odd constant.
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    /// The next 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A value in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        usize::saturating_from(self.next_u64() % n as u64)
    }

    /// Shuffle `items` in place.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
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

    #[test]
    fn rng_shuffle_is_a_deterministic_permutation() {
        let mut a: std::vec::Vec<u32> = (0..64).collect();
        let mut b = a.clone();
        Rng::new(7).shuffle(&mut a);
        Rng::new(7).shuffle(&mut b);
        assert_eq!(a, b, "same seed must produce the same order");
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..64).collect::<std::vec::Vec<u32>>());
        assert_ne!(a, sorted, "shuffle must move something");
    }

    #[test]
    fn rng_zero_seed_still_advances() {
        let mut rng = Rng::new(0);
        assert_ne!(rng.next_u64(), rng.next_u64());
        assert!((0..100).all(|_| rng.below(3) < 3));
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
