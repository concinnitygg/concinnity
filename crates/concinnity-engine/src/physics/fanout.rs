// src/physics/fanout.rs
//
// The engine's job pool, lent to the simulation.
//
// The simulation crate names no scheduler: it is `#![no_std]` and depends on
// two leaves, so it asks for a fan-out and a caller decides what that is. This
// is the engine's answer, and it is the only place the two meet.
//
// Which pool it wraps follows `ScheduleMode`, the same way every other system's
// internal fan-out does, so the determinism oracle runs the identical code path
// with one worker rather than a different path with none.

use concinnity_core::ecs::ScheduleMode;
use concinnity_cpu::jobs::{self, JobPool};
use concinnity_physics::Fanout;

/// Lends a job pool to a stepping simulation.
pub(super) struct PoolFanout {
    pool: &'static JobPool,
}

impl PoolFanout {
    /// The pool `mode` runs on: the bounded worker pool under a parallel
    /// schedule, and its single-worker twin under a serial one.
    pub(super) fn for_mode(mode: ScheduleMode) -> PoolFanout {
        PoolFanout {
            pool: match mode {
                ScheduleMode::Parallel => jobs::pool(),
                ScheduleMode::Serial => jobs::serial_pool(),
            },
        }
    }

    /// Workers to reserve the simulation's per-worker scratch for.
    pub(super) fn worker_count(&self) -> usize {
        self.pool.thread_count()
    }
}

impl Fanout for PoolFanout {
    fn workers(&self) -> usize {
        self.pool.thread_count()
    }

    fn scope<R, F>(&self, work: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.pool.install(work)
    }

    fn for_each<T, F>(&self, items: &mut [T], body: F)
    where
        T: Send,
        F: Fn(&mut T) + Send + Sync,
    {
        self.pool.parallel_for(items, body);
    }
}
