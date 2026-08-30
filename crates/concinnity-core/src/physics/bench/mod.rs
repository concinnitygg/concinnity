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

mod sim;

use crate::physics::Fanout;

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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    // The fan-out gives every unit of work its own thread and joins them all,
    // so each item is touched exactly once however the split lands.
    #[test]
    fn the_fan_out_touches_every_item_once() {
        let mut items: Vec<u32> = (0..16).collect();
        Pool.for_each(&mut items, |item| *item += 1);
        assert_eq!(items, (1..17).collect::<Vec<u32>>());
        assert_eq!(Pool.workers(), WORKERS);
    }

    // Below two items there is nothing to split, so the body runs on the
    // calling thread rather than paying for a scope.
    #[test]
    fn a_split_of_one_or_none_runs_on_the_calling_thread() {
        let mut one = [7u32];
        Pool.for_each(&mut one, |item| *item += 1);
        assert_eq!(one, [8]);

        let mut none: [u32; 0] = [];
        Pool.for_each(&mut none, |item| *item += 1);
        assert!(none.is_empty());
    }
}
