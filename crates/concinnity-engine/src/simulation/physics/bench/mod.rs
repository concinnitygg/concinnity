// Simulation microbenchmarks and their shared helpers. The measured pass is
// ignored by default; a single-run pass drives the same fixtures small. Use
// `--test-threads=1`, since a concurrent benchmark reads as contention.
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
