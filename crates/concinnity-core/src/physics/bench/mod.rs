// In-crate microbenchmarks for the simulation, and the fan-out and timing
// helpers they share.
//
// These live inside the crate rather than under a bench target because what
// they measure is `pub(crate)`: the overflow counters, the pose and sleep
// readouts, and the kinematic and shape-cast paths are not part of the
// simulation's public surface, and widening them for a benchmark would be the
// wrong trade. The measured pass is ignored by default, so a normal test run
// pays only for the single-run pass beside it, which drives the same
// fixtures once at a fraction of the size.
// `--test-threads=1` is required rather than tidy: a benchmark sharing a
// machine with another reads the other's contention as its own.
//
//     cargo test -p concinnity-core --release -- --ignored --nocapture \
//         --test-threads=1 physics::bench

mod pen;
mod sim;

use crate::fanout::test_pools::Threads;

// Workers a stepping benchmark is handed, standing in for the pool the driver
// lends. Fixed rather than machine-derived so a number is comparable across
// machines at the same split.
pub(crate) const WORKERS: usize = 8;

// The fan-out a stepping benchmark is handed: every unit on its own thread.
pub(crate) const POOL: Threads = Threads(WORKERS);
