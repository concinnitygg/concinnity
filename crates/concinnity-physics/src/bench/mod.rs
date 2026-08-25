// In-crate microbenchmarks for the simulation, and the fan-out and timing
// helpers they share.
//
// These live inside the crate rather than under a bench target because what
// they measure is `pub(crate)`: the overflow counters, the pose and sleep
// readouts, and the kinematic and shape-cast paths are not part of the
// simulation's public surface, and widening them for a benchmark would be the
// wrong trade. Ignored by default, so a normal test run never pays for them.
// `--test-threads=1` is required rather than tidy: a benchmark sharing a
// machine with another reads the other's contention as its own.
//
//     cargo test -p concinnity-physics --release -- --ignored --nocapture \
//         --test-threads=1 bench

mod sim;

use std::time::Instant;

use crate::Fanout;

// Workers a stepping benchmark is handed, standing in for the pool the driver
// lends. Fixed rather than machine-derived so a number is comparable across
// machines at the same split.
pub(crate) const WORKERS: usize = 8;

/// A fan-out that gives every unit of work its own thread, as the parallel
/// determinism tests do. Not a pool: this crate owns no scheduler, and what
/// the benchmarks measure is the split, not the scheduler behind it.
pub(crate) struct Pool;

impl Fanout for Pool {
    fn workers(&self) -> usize {
        WORKERS
    }

    fn for_each<T, F>(&self, items: &mut [T], body: F)
    where
        T: Send,
        F: Fn(&mut T) + Send + Sync,
    {
        if items.len() < 2 {
            items.iter_mut().for_each(body);
            return;
        }
        let body = &body;
        std::thread::scope(|scope| {
            for item in items.iter_mut() {
                scope.spawn(move || body(item));
            }
        });
    }
}

// One measured pass runs at least this long before its time is trusted.
const TARGET_NS: u128 = 200_000_000;
const MAX_ITERS: u64 = 1 << 20;

// Time `body` over a calibrated iteration count and report its per-item cost.
// `items` is how many units of work one call performs, so a number is
// comparable across fixture sizes.
pub(crate) fn bench<R>(name: &str, items: u64, mut body: impl FnMut() -> R) {
    let mut iters: u64 = 1;
    loop {
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(body());
        }
        if start.elapsed().as_nanos() >= TARGET_NS || iters >= MAX_ITERS {
            break;
        }
        iters = iters.saturating_mul(4).min(MAX_ITERS);
    }

    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(body());
    }
    let elapsed = start.elapsed();

    let units = (iters * items.max(1)) as f64;
    let per_item_ns = elapsed.as_secs_f64() * 1e9 / units;
    std::println!("  {name:<40} {per_item_ns:>10.2} ns/item");
}
