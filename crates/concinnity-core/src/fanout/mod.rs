//! How a host lends this crate its threads.
//!
//! The crate is `no_std` and owns no scheduler, so work it can split is offered
//! rather than taken: a caller says what it holds that is independent, and the
//! [`FanOut`] it was handed decides whether that runs on one thread or several.
//! A host implements it once on its pool; physics steps, behavior evaluation,
//! pose sampling, bake rows and render-graph recording all reach the same pool
//! through it. A caller that lends nothing gets [`Inline`].

mod gate;
mod judge;
mod lent;
mod pooled;
#[cfg(any(test, feature = "test-support"))]
pub mod test_pools;
mod timing;

pub use gate::FanOutGate;
pub use judge::FanOutJudge;
pub use lent::Lent;
pub use pooled::Pooled;
pub use timing::{FanOutClock, FanOutTimer, FanOutTiming};

/// How a tick's independent work executes. `Parallel` lets systems fan their
/// safe internal work across the job pool; `Serial` keeps every system's work
/// on the stepping thread. Both modes must produce identical results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScheduleMode {
    /// Keep every system's work on the stepping thread.
    Serial,
    #[default]
    /// Let systems fan safe internal work across the job pool.
    Parallel,
}

/// A caller's way of running independent work at the same time.
///
/// Generic in the item and the body rather than object safe, so the work is
/// monomorphized into the caller and nothing is boxed per item. A system that
/// has to hold one keeps a [`Lent`] instead.
///
/// # Examples
///
/// ```
/// use concinnity_core::fanout::{FanOut, Inline};
///
/// let mut work = [1u32, 2, 3];
/// Inline.for_each(&mut work, |item| *item *= 10);
/// assert_eq!(work, [10, 20, 30]);
/// assert_eq!(Inline.workers(), 1);
/// ```
pub trait FanOut: Sync {
    /// Units of work this fan-out can run at once. One means the work runs on
    /// the calling thread.
    fn workers(&self) -> usize;

    /// Run `work` with this fan-out's workers already gathered, and return
    /// what it produced. Reaching a sleeping pool from outside costs far more
    /// than reaching it again from inside, so a caller handing out several
    /// batches does so inside one of these. The default runs `work` in place.
    fn scope<R, F>(&self, work: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        work()
    }

    /// Run `body` over every item and return once all of them are done. Items
    /// are independent, so they may run in any order and on any thread.
    fn for_each<T, F>(&self, items: &mut [T], body: F)
    where
        T: Send,
        F: Fn(&mut T) + Send + Sync;

    /// [`for_each`](Self::for_each) for a few long items: each starts as soon
    /// as a worker is free rather than being split into even chunks. The
    /// default is `for_each`.
    fn spawn_each<T, F>(&self, items: &mut [T], body: F)
    where
        T: Send,
        F: Fn(&mut T) + Send + Sync,
    {
        self.for_each(items, body);
    }

    /// Nanoseconds on a monotonic clock, read from any worker. Zero
    /// throughout (the default) leaves every fan-out untimed.
    fn clock_ns(&self) -> u64 {
        0
    }

    /// A timer for a fan-out starting now, on this fan-out's clock.
    fn timer(&self) -> FanOutTimer<Since<'_, Self>>
    where
        Self: Sized,
    {
        FanOutTimer::new(Since {
            fanout: self,
            start: self.clock_ns(),
        })
    }

    /// [`for_each`](Self::for_each), reporting how long the calling thread
    /// waited against how much work the items took.
    fn for_each_timed<T, F>(&self, items: &mut [T], body: F) -> FanOutTiming
    where
        Self: Sized,
        T: Send,
        F: Fn(&mut T) + Send + Sync,
    {
        let timer = self.timer();
        self.for_each(items, |item| timer.job(|| body(item)));
        timer.finish()
    }
}

/// A fan-out's clock read relative to the moment its timer started.
#[derive(Debug)]
pub struct Since<'a, F: ?Sized> {
    fanout: &'a F,
    start: u64,
}

impl<F: FanOut + ?Sized> FanOutClock for Since<'_, F> {
    fn elapsed_ns(&self) -> u64 {
        self.fanout.clock_ns().saturating_sub(self.start)
    }
}

/// The fan-out for a caller that has no threads to lend: every unit runs on the
/// calling thread, in order.
#[derive(Debug, Clone, Copy, Default)]
pub struct Inline;

impl FanOut for Inline {
    fn workers(&self) -> usize {
        1
    }

    fn for_each<T, F>(&self, items: &mut [T], body: F)
    where
        T: Send,
        F: Fn(&mut T) + Send + Sync,
    {
        for item in items {
            body(item);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FanOut, FanOutTiming, Inline};
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicU64, Ordering};

    #[test]
    fn inline_visits_every_item_in_order() {
        let mut items: Vec<u32> = (0..8).collect();
        Inline.for_each(&mut items, |item| *item += 100);
        assert_eq!(items, (0..8).map(|i| i + 100).collect::<Vec<_>>());
        Inline.spawn_each(&mut items, |item| *item -= 100);
        assert_eq!(items, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn inline_lends_one_worker_and_runs_its_scope_in_place() {
        assert_eq!(Inline.workers(), 1);
        assert_eq!(Inline.scope(|| 7), 7);
    }

    #[test]
    fn an_untimed_fanout_reports_zero_timing() {
        let mut items = [1u32, 2];
        assert_eq!(
            Inline.for_each_timed(&mut items, |item| *item += 1),
            FanOutTiming::default()
        );
        assert_eq!(items, [2, 3]);
    }

    // A clock that advances 1 us on every read.
    struct Ticking(AtomicU64);

    impl FanOut for Ticking {
        fn workers(&self) -> usize {
            1
        }

        fn for_each<T, F>(&self, items: &mut [T], body: F)
        where
            T: Send,
            F: Fn(&mut T) + Send + Sync,
        {
            items.iter_mut().for_each(body);
        }

        fn clock_ns(&self) -> u64 {
            self.0.fetch_add(1_000, Ordering::Relaxed)
        }
    }

    // The timer reads the fan-out's own clock relative to its start: the
    // start read, one start and one end per job, and the finish read.
    #[test]
    fn a_timed_fanout_measures_against_its_own_clock() {
        let clock = Ticking(AtomicU64::new(5_000_000));
        let mut items = [0u32; 2];
        let timing = clock.for_each_timed(&mut items, |_| ());
        assert_eq!(timing.wall_us, 5);
        assert_eq!(timing.first_job_us, 1);
        assert_eq!(timing.job_sum_us, 2);
    }
}
