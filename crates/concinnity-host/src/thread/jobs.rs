//! Backend-agnostic job pool for parallelizing expensive per-frame CPU work.
//!
//! Systems run serially in the frame loop, each holding `&mut PipelineContext`.
//! This pool does not change that: it lets a single system fan its own
//! data-parallel work (per-skeleton pose sampling, particle update, ...) across
//! worker threads and join before `step` returns. It is not a way to run whole
//! systems concurrently.
//!
//! The pool wraps a dedicated `rayon::ThreadPool` rather than rayon's global
//! pool so the worker count and thread names are controlled. It is process-wide
//! and built once, by whichever of `configure` and `pool()` is reached first.

use std::sync::OnceLock;

use rayon::prelude::*;

/// A dedicated thread pool for per-frame data-parallel work.
pub struct JobPool {
    pool: rayon::ThreadPool,
}

impl JobPool {
    /// Build a pool with an explicit worker count (floored at one), for work
    /// that must not size the process-wide pool before the runtime configures it.
    pub fn new(threads: usize) -> JobPool {
        let threads = threads.max(1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("cn-job-{i}"))
            .build()
            .expect("failed to build job thread pool");
        tracing::info!("JobPool: {threads} worker thread(s)");
        JobPool { pool }
    }

    /// Number of worker threads in this pool.
    pub fn thread_count(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Apply `f` to every item in parallel, blocking until all are done.
    ///
    /// Each item must be independent: `f` runs concurrently across items in
    /// no defined order. Inputs shorter than two items skip the pool and run
    /// inline to avoid dispatch overhead.
    pub fn parallel_for<T, F>(&self, items: &mut [T], f: F)
    where
        T: Send,
        F: Fn(&mut T) + Send + Sync,
    {
        if items.len() < 2 {
            items.iter_mut().for_each(f);
            return;
        }
        self.pool.install(|| items.par_iter_mut().for_each(f));
    }

    /// Run a closure inside this pool's scope so any nested rayon
    /// `par_iter` / `par_iter_mut` calls dispatch to JobPool's bounded thread
    /// count (`available_parallelism() - 1`) instead of rayon's global pool
    /// (which defaults to every core and would starve the render thread when
    /// invoked from a worker that is itself competing for CPU).
    ///
    /// Used by the DirectX / Metal parallel command-buffer recording; the Vulkan
    /// backend records single-threaded, so it is unused under `backend_vk`.
    pub fn install<R, F>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.pool.install(f)
    }
}

/// Spreads a bake's independent rows across the given job pool.
///
/// The environment-map convolutions decompose into rows that share nothing --
/// each reads only the immutable source and writes only its own texels -- so
/// fanning them out buys wall clock without changing a byte. Handed to
/// `concinnity_core::bake` wherever a build has a pool behind it; a caller
/// without one uses `Serial` instead.
pub struct PoolRows<'a>(pub &'a JobPool);

impl concinnity_core::bake::environment_map::RowScheduler for PoolRows<'_> {
    fn run<T: Send>(&self, items: &mut [T], compute: &(dyn Fn(&mut T) + Send + Sync)) {
        self.0.parallel_for(items, compute);
    }
}

// The process-wide pool, built by whichever of `configure` and `pool` gets
// there first.
static POOL: OnceLock<JobPool> = OnceLock::new();

/// Worker count when nothing configures the pool: one per logical core, less
/// one for the main thread, floored at one.
pub fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .unwrap_or(1)
}

/// Build the process-wide job pool at `threads` workers, returning whether
/// this call sized it. The runtime calls this from its `ThreadBudget` at start,
/// before any system uses the pool. The pool is built once, so a call that
/// finds it already there sizes nothing and is reported as `false`; a value
/// below one is clamped.
pub fn configure(threads: usize) -> bool {
    size_pool(&POOL, threads)
}

// Sizing a pool cell, shared with the test so both outcomes are reachable
// without depending on what else in the process has touched `POOL`.
fn size_pool(cell: &OnceLock<JobPool>, threads: usize) -> bool {
    // A cheap refusal before paying for the worker threads a full `set` would
    // build and then drop.
    if cell.get().is_some() {
        return false;
    }
    cell.set(JobPool::new(threads.max(1))).is_ok()
}

/// The process-wide job pool, built at the auto default worker count if
/// `configure` has not already sized it.
pub fn pool() -> &'static JobPool {
    POOL.get_or_init(|| JobPool::new(default_threads()))
}

/// A single-worker pool: the same execution shape as `pool()` with the jobs
/// run one at a time. The serial schedule installs solver work here so the
/// determinism oracle exercises the identical code path minus the
/// concurrency.
pub fn serial_pool() -> &'static JobPool {
    static SERIAL: OnceLock<JobPool> = OnceLock::new();
    SERIAL.get_or_init(|| JobPool::new(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_is_a_singleton() {
        assert!(std::ptr::eq(pool(), pool()));
    }

    #[test]
    fn sizing_a_pool_reports_whether_it_built_one() {
        let cell = OnceLock::new();
        assert!(size_pool(&cell, 3));
        assert_eq!(cell.get().expect("the sized pool").thread_count(), 3);

        // The pool is built once, so a later call sizes nothing and says so
        // rather than recording a count nobody reads.
        assert!(!size_pool(&cell, 7));
        assert_eq!(cell.get().expect("the sized pool").thread_count(), 3);
    }

    #[test]
    fn configure_is_refused_once_the_pool_exists() {
        // Order-independent: reaching `pool()` is what closes configuration,
        // whichever test in this process got there first.
        let workers = pool().thread_count();
        assert!(!configure(workers + 1));
        assert_eq!(pool().thread_count(), workers);
    }

    // An explicit worker count is honored (floored at one). Tested via
    // `new` directly: the process-wide `pool()` is a `OnceLock` built
    // once, so its size cannot be asserted deterministically alongside the
    // other tests that also touch it.
    #[test]
    fn new_sets_the_worker_count() {
        assert_eq!(JobPool::new(3).thread_count(), 3);
        assert_eq!(JobPool::new(0).thread_count(), 1);
    }

    #[test]
    fn pool_rows_visit_every_row_exactly_once() {
        use concinnity_core::bake::environment_map::RowScheduler;

        let pool = JobPool::new(2);
        let mut rows = vec![0u32; 257];
        PoolRows(&pool).run(&mut rows, &|visits| *visits += 1);
        assert!(rows.iter().all(|&visits| visits == 1));
    }

    // The auto default always leaves at least one worker.
    #[test]
    fn default_threads_is_at_least_one() {
        assert!(default_threads() >= 1);
    }

    #[test]
    fn parallel_for_visits_every_item() {
        let mut data: Vec<u32> = (0..10_000).collect();
        pool().parallel_for(&mut data, |x| *x += 1);
        assert!(data.iter().enumerate().all(|(i, &x)| x == i as u32 + 1));
    }

    #[test]
    fn parallel_for_handles_empty_and_single() {
        let mut empty: Vec<u32> = Vec::new();
        pool().parallel_for(&mut empty, |x| *x += 1);
        assert!(empty.is_empty());

        let mut single = vec![41u32];
        pool().parallel_for(&mut single, |x| *x += 1);
        assert_eq!(single, vec![42]);
    }
}
