//! Backend-agnostic job pool for parallelising expensive per-frame CPU work.
//!
//! Systems run serially in the frame loop, each holding `&mut PipelineContext`.
//! This pool does not change that: it lets a single system fan its own
//! data-parallel work (per-skeleton pose sampling, particle update, ...) across
//! worker threads and join before `step` returns. It is not a way to run whole
//! systems concurrently.
//!
//! The pool wraps a dedicated `rayon::ThreadPool` rather than rayon's global
//! pool so the worker count and thread names are controlled. It is process-wide
//! and lazily built on first use via `pool()`.

use std::sync::OnceLock;

use rayon::prelude::*;

/// A dedicated thread pool for per-frame data-parallel work.
pub struct JobPool {
    pool: rayon::ThreadPool,
}

impl JobPool {
    // Build the pool at the worker count `configure` set, or the auto default
    // (`available_parallelism() - 1`) when unconfigured. The App sizes it from
    // its `ThreadBudget` before the first `pool()` use.
    fn build() -> JobPool {
        Self::with_threads(
            CONFIGURED_THREADS
                .get()
                .copied()
                .unwrap_or_else(default_threads),
        )
    }

    // Build a pool with an explicit worker count (floored at one).
    fn with_threads(threads: usize) -> JobPool {
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    pub fn install<R, F>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.pool.install(f)
    }
}

// Worker count set by `configure`, consulted by `JobPool::build` on first use.
static CONFIGURED_THREADS: OnceLock<usize> = OnceLock::new();

// Auto worker count: one per logical core, less one for the main thread,
// floored at one.
fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .unwrap_or(1)
}

/// Set the process-wide job pool's worker count. The App calls this from its
/// `ThreadBudget` at start, before any system uses the pool. It takes effect
/// only if called before the first `pool()` access (the pool is built once);
/// a later call, or a value below one, is ignored/clamped.
pub fn configure(threads: usize) {
    let _ = CONFIGURED_THREADS.set(threads.max(1));
}

/// The process-wide job pool, built on first access.
pub fn pool() -> &'static JobPool {
    static POOL: OnceLock<JobPool> = OnceLock::new();
    POOL.get_or_init(JobPool::build)
}

/// A single-worker pool: the same execution shape as `pool()` with the jobs
/// run one at a time. The serial schedule installs solver work here so the
/// determinism oracle exercises the identical code path minus the
/// concurrency.
pub fn serial_pool() -> &'static JobPool {
    static POOL: OnceLock<JobPool> = OnceLock::new();
    POOL.get_or_init(|| JobPool::with_threads(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_is_a_singleton() {
        assert!(std::ptr::eq(pool(), pool()));
    }

    // An explicit worker count is honored (floored at one). Tested via
    // `with_threads` directly: the process-wide `pool()` is a `OnceLock` built
    // once, so its size cannot be asserted deterministically alongside the
    // other tests that also touch it.
    #[test]
    fn with_threads_sets_the_worker_count() {
        assert_eq!(JobPool::with_threads(3).thread_count(), 3);
        assert_eq!(JobPool::with_threads(0).thread_count(), 1);
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
