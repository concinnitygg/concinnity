//! Backend-agnostic job pool for parallelizing expensive per-frame CPU work.
//!
//! Systems run serially in the frame loop, each holding `&mut PipelineContext`.
//! This pool does not change that: it lets a single system fan its own
//! data-parallel work (per-skeleton pose sampling, particle update, ...) across
//! worker threads and join before `step` returns. It is not a way to run whole
//! systems concurrently.
//!
//! The pool wraps a dedicated `rayon::ThreadPool` rather than rayon's global
//! pool so the worker count, thread names and scheduling are controlled: every
//! worker runs as a [`ThreadRole::FrameWorker`] thread at the pool's
//! [`FramePriority`]. It is process-wide and built once, by whichever of
//! `configure` and `pool()` is reached first.

use std::sync::OnceLock;
use std::time::Instant;

use concinnity_core::components::FramePriority;
use concinnity_core::profile::{FanOutClock, FanOutTimer, FanOutTiming};
use rayon::prelude::*;

use super::role::{ThreadRole, set_current_thread_role};

/// A dedicated thread pool for per-frame data-parallel work.
pub struct JobPool {
    pool: rayon::ThreadPool,
    priority: FramePriority,
}

impl JobPool {
    /// Build a pool with an explicit worker count (floored at one) whose
    /// workers are scheduled as `priority` asks, for work that must not size
    /// the process-wide pool before the runtime configures it.
    pub fn new(threads: usize, priority: FramePriority) -> JobPool {
        let threads = threads.max(1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("cn-job-{i}"))
            .start_handler(move |_| set_current_thread_role(ThreadRole::FrameWorker(priority)))
            .build()
            .expect("failed to build job thread pool");
        tracing::info!("JobPool: {threads} worker thread(s), priority {priority:?}");
        JobPool { pool, priority }
    }

    /// Number of worker threads in this pool.
    pub fn thread_count(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// The priority setting the workers were started with.
    pub fn priority(&self) -> FramePriority {
        self.priority
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

    /// [`parallel_for`](Self::parallel_for), reporting how long the calling
    /// thread waited against how much work the items took.
    pub fn parallel_for_timed<T, F>(&self, items: &mut [T], f: F) -> FanOutTiming
    where
        T: Send,
        F: Fn(&mut T) + Send + Sync,
    {
        let timer = fan_out_timer();
        self.parallel_for(items, |item| timer.job(|| f(item)));
        timer.finish()
    }

    /// Run a closure inside this pool's scope so any nested rayon
    /// `par_iter` / `par_iter_mut` calls dispatch to this pool's bounded
    /// workers instead of rayon's global pool (which defaults to every core
    /// and would starve the render thread when invoked from a worker that is
    /// itself competing for CPU).
    ///
    /// Used by every backend's parallel command-buffer recording.
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

/// The wall clock a fan-out is timed against: nanoseconds since it started.
#[derive(Debug, Clone, Copy)]
pub struct SinceStart(Instant);

impl FanOutClock for SinceStart {
    fn elapsed_ns(&self) -> u64 {
        self.0.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
    }
}

/// A timer for a fan-out that starts now.
pub fn fan_out_timer() -> FanOutTimer<SinceStart> {
    FanOutTimer::new(SinceStart(Instant::now()))
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

/// Build the process-wide job pool at `threads` workers scheduled as
/// `priority` asks, returning whether this call built it. The runtime calls
/// this from its `ThreadBudget` at start, before any system uses the pool. The
/// pool is built once, so a call that finds it already there changes nothing
/// and is reported as `false`; a value below one is clamped.
pub fn configure(threads: usize, priority: FramePriority) -> bool {
    size_pool(&POOL, threads, priority)
}

// Sizing a pool cell, shared with the test so both outcomes are reachable
// without depending on what else in the process has touched `POOL`.
fn size_pool(cell: &OnceLock<JobPool>, threads: usize, priority: FramePriority) -> bool {
    // A cheap refusal before paying for the worker threads a full `set` would
    // build and then drop.
    if cell.get().is_some() {
        return false;
    }
    cell.set(JobPool::new(threads.max(1), priority)).is_ok()
}

/// The process-wide job pool, built at the auto default worker count and the
/// default priority if `configure` has not already built it.
pub fn pool() -> &'static JobPool {
    POOL.get_or_init(|| JobPool::new(default_threads(), FramePriority::default()))
}

/// A single-worker pool: the same execution shape as `pool()` with the jobs
/// run one at a time. The serial schedule installs solver work here so the
/// determinism oracle exercises the identical code path minus the
/// concurrency.
pub fn serial_pool() -> &'static JobPool {
    static SERIAL: OnceLock<JobPool> = OnceLock::new();
    SERIAL.get_or_init(|| JobPool::new(1, FramePriority::default()))
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
        assert!(size_pool(&cell, 3, FramePriority::AllThreads));
        let sized = cell.get().expect("the sized pool");
        assert_eq!(sized.thread_count(), 3);
        assert_eq!(sized.priority(), FramePriority::AllThreads);

        // The pool is built once, so a later call changes nothing and says so
        // rather than recording a count nobody reads.
        assert!(!size_pool(&cell, 7, FramePriority::Normal));
        let kept = cell.get().expect("the sized pool");
        assert_eq!(kept.thread_count(), 3);
        assert_eq!(kept.priority(), FramePriority::AllThreads);
    }

    #[test]
    fn configure_is_refused_once_the_pool_exists() {
        // Order-independent: reaching `pool()` is what closes configuration,
        // whichever test in this process got there first.
        let workers = pool().thread_count();
        assert!(!configure(workers + 1, FramePriority::Normal));
        assert_eq!(pool().thread_count(), workers);
    }

    // An explicit worker count is honored (floored at one). Tested via
    // `new` directly: the process-wide `pool()` is a `OnceLock` built
    // once, so its size cannot be asserted deterministically alongside the
    // other tests that also touch it.
    #[test]
    fn new_sets_the_worker_count() {
        assert_eq!(JobPool::new(3, FramePriority::default()).thread_count(), 3);
        assert_eq!(JobPool::new(0, FramePriority::default()).thread_count(), 1);
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn workers_run_at_the_frame_qos_class() {
        let class = JobPool::new(1, FramePriority::Normal).install(|| {
            let mut class = libc::qos_class_t::QOS_CLASS_UNSPECIFIED;
            let mut relative = 0;
            // SAFETY: both out-pointers are live locals, and `pthread_self`
            // names the calling worker, which outlives the call.
            let status = unsafe {
                libc::pthread_get_qos_class_np(libc::pthread_self(), &mut class, &mut relative)
            };
            (status, class as u32)
        });
        assert_eq!(
            class,
            (0, libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE as u32)
        );
    }

    #[cfg(windows)]
    #[test]
    fn workers_run_at_the_priority_the_pool_was_built_with() {
        use windows::Win32::System::Threading::{
            GetCurrentThread, GetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL,
            THREAD_PRIORITY_NORMAL,
        };

        let worker_priority = |priority| {
            JobPool::new(1, priority).install(|| {
                // SAFETY: `GetCurrentThread` is a pseudo-handle valid for the
                // calling worker.
                unsafe { GetThreadPriority(GetCurrentThread()) }
            })
        };
        assert_eq!(
            worker_priority(FramePriority::MainThreads),
            THREAD_PRIORITY_NORMAL.0
        );
        assert_eq!(
            worker_priority(FramePriority::AllThreads),
            THREAD_PRIORITY_ABOVE_NORMAL.0
        );
    }

    #[test]
    fn pool_rows_visit_every_row_exactly_once() {
        use concinnity_core::bake::environment_map::RowScheduler;

        let pool = JobPool::new(2, FramePriority::default());
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
    fn a_timed_parallel_for_reports_the_work_inside_the_wait() {
        let pool = JobPool::new(2, FramePriority::default());
        let mut items = [0u32; 4];
        let timing = pool.parallel_for_timed(&mut items, |x| {
            std::thread::sleep(std::time::Duration::from_millis(2));
            *x += 1;
        });
        assert_eq!(items, [1; 4]);
        // Four 2 ms jobs: at least 8 ms of work, each job at least 2 ms, and
        // the wait covers the longest job, its start latency and its tail.
        assert!(timing.job_sum_us >= 8_000, "{timing:?}");
        assert!(timing.longest_job_us >= 2_000, "{timing:?}");
        assert!(
            timing.wall_us >= timing.first_job_us + timing.longest_job_us,
            "{timing:?}"
        );
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
