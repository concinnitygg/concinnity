// In-crate microbenchmarks for the simulation, and the fan-out and timing
// helpers they share.
//
// The measured pass is ignored by default, so a normal test run
// pays only for the single-run pass beside it, which drives the same
// fixtures once at a fraction of the size.
// `--test-threads=1` is required rather than tidy: a benchmark sharing a
// machine with another reads the other's contention as its own.
//
//     cargo test -p concinnity-engine --release -- --ignored --nocapture \
//         --test-threads=1 physics::bench

mod float;
mod pen;
mod sim;

use concinnity_core::fanout::test_pools::Threads;

// Workers a stepping benchmark is handed, standing in for the pool the driver
// lends. Fixed rather than machine-derived so a number is comparable across
// machines at the same split.
pub(crate) const WORKERS: usize = 8;

// The fan-out a stepping benchmark is handed: every unit on its own thread.
pub(crate) const POOL: Threads = Threads(WORKERS);
