//! Spreads a set of independent work items across the shared job pool.
//!
//! `concinnity-core` decomposes its environment-map convolutions into rows that
//! share nothing, and leaves the schedule to whoever has threads. This is that
//! schedule for the backends: the probe bake hands its rows here, and the
//! fan-out buys wall clock without changing a byte.

use crate::build::environment_map::RowScheduler;
use crate::jobs;

pub(crate) struct PoolRows;

impl RowScheduler for PoolRows {
    fn run<T: Send>(&self, items: &mut [T], compute: &(dyn Fn(&mut T) + Send + Sync)) {
        jobs::pool().parallel_for(items, compute);
    }
}
