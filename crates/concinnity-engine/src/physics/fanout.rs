// The engine's job pool, lent to the simulation.
//
// The driver lives in concinnity-core and names no scheduler; it asks for a
// fan-out and a host decides what that is. This is this host's answer, and it
// is the only place the two meet.
//
// Which pool a step runs on follows `ScheduleMode`, the same way every other
// system's internal fan-out does, so the determinism oracle runs the identical
// code path with one worker rather than a different path with none.

use concinnity_core::ecs::ScheduleMode;
use concinnity_core::physics::PhysicsFanout;
use concinnity_core::physics::{Fanout, Simulation};
use concinnity_host::thread::jobs::{self, JobPool};

/// Lends the engine's job pool to a stepping simulation.
#[derive(Debug)]
pub(crate) struct PoolFanout;

// The pool `mode` runs on: the bounded worker pool under a parallel schedule,
// and its single-worker twin under a serial one.
fn pool_for(mode: ScheduleMode) -> &'static JobPool {
    match mode {
        ScheduleMode::Parallel => jobs::pool(),
        ScheduleMode::Serial => jobs::serial_pool(),
    }
}

impl PhysicsFanout for PoolFanout {
    fn worker_count(&self, mode: ScheduleMode) -> usize {
        pool_for(mode).thread_count()
    }

    fn step(&self, sim: &mut Simulation, dt: f32, mode: ScheduleMode) {
        sim.step_with(dt, &PoolLease(pool_for(mode)));
    }
}

// One step's borrow of a pool, in the shape the simulation asks for.
struct PoolLease(&'static JobPool);

impl Fanout for PoolLease {
    fn workers(&self) -> usize {
        self.0.thread_count()
    }

    fn scope<R, F>(&self, work: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.0.install(work)
    }

    fn for_each<T, F>(&self, items: &mut [T], body: F)
    where
        T: Send,
        F: Fn(&mut T) + Send + Sync,
    {
        self.0.parallel_for(items, body);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A serial schedule lends exactly one worker, which is what makes the
    // determinism oracle's run the same path a host with no pool takes.
    #[test]
    fn a_serial_schedule_lends_one_worker() {
        assert_eq!(PoolFanout.worker_count(ScheduleMode::Serial), 1);
        assert!(PoolFanout.worker_count(ScheduleMode::Parallel) >= 1);
    }
}
