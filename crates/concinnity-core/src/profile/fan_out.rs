//! Timing for one parallel fan-out: how long the thread that waits on it was
//! held, against how much work its jobs contained.
//!
//! A fan-out whose wall time is well above its longest job is not limited by
//! its work. The gap splits into the wait before the first job starts (workers
//! waking) and the tail after the last one ends (the waiting thread resuming),
//! which is what this records alongside the job times.

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};

/// A monotonic clock reading nanoseconds since the fan-out it times began.
///
/// Injected so the timer itself needs no `std`: a host passes a wall clock, a
/// test passes a scripted one.
pub trait FanOutClock {
    /// Nanoseconds since the fan-out began. Never decreases.
    fn elapsed_ns(&self) -> u64;
}

/// One fan-out's timing, in microseconds. All zero for a fan-out that did not
/// run.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FanOutTiming {
    /// How long the waiting thread was held, from just before it handed out
    /// the jobs to just after it resumed.
    pub wall_us: u32,
    /// From the start of the fan-out to the moment the first job began.
    pub first_job_us: u32,
    /// Every job's duration, summed: the work the fan-out contained.
    pub job_sum_us: u32,
    /// The longest single job, which bounds the wall time from below however
    /// many workers there are.
    pub longest_job_us: u32,
    /// From the moment the last job ended to the waiting thread resuming.
    pub tail_us: u32,
}

impl FanOutTiming {
    /// Two fan-outs the same thread waited on in turn, as one: the waits and
    /// the work add, and the longest job is the longer of the two.
    pub fn merge(self, other: Self) -> Self {
        Self {
            wall_us: self.wall_us.saturating_add(other.wall_us),
            first_job_us: self.first_job_us.saturating_add(other.first_job_us),
            job_sum_us: self.job_sum_us.saturating_add(other.job_sum_us),
            longest_job_us: self.longest_job_us.max(other.longest_job_us),
            tail_us: self.tail_us.saturating_add(other.tail_us),
        }
    }
}

/// Times one fan-out. Create it just before the jobs are handed out, wrap each
/// job's body in [`job`](Self::job), and call [`finish`](Self::finish) once the
/// waiting thread resumes.
///
/// Shared by reference across the jobs: every reading is one relaxed atomic,
/// and the join the waiting thread returns from orders them before `finish`.
#[derive(Debug)]
pub struct FanOutTimer<C> {
    clock: C,
    first_start_ns: AtomicU64,
    last_end_ns: AtomicU64,
    job_sum_ns: AtomicU64,
    longest_job_ns: AtomicU64,
}

impl<C: FanOutClock> FanOutTimer<C> {
    /// A timer for a fan-out beginning at `clock`'s zero.
    pub const fn new(clock: C) -> Self {
        Self {
            clock,
            first_start_ns: AtomicU64::new(u64::MAX),
            last_end_ns: AtomicU64::new(0),
            job_sum_ns: AtomicU64::new(0),
            longest_job_ns: AtomicU64::new(0),
        }
    }

    /// Run one job, recording when it started and how long it took.
    pub fn job<R>(&self, job: impl FnOnce() -> R) -> R {
        self.timed(job).0
    }

    /// Run one job as [`job`](Self::job) does, also storing its duration in
    /// microseconds in `slot`, so the caller can say which job took the time.
    pub fn job_into<R>(&self, slot: &AtomicU32, job: impl FnOnce() -> R) -> R {
        let (value, duration) = self.timed(job);
        slot.store(micros(duration), Relaxed);
        value
    }

    fn timed<R>(&self, job: impl FnOnce() -> R) -> (R, u64) {
        let start = self.clock.elapsed_ns();
        let value = job();
        let end = self.clock.elapsed_ns();
        let duration = end.saturating_sub(start);
        self.first_start_ns.fetch_min(start, Relaxed);
        self.last_end_ns.fetch_max(end, Relaxed);
        self.job_sum_ns.fetch_add(duration, Relaxed);
        self.longest_job_ns.fetch_max(duration, Relaxed);
        (value, duration)
    }

    /// The fan-out's timing, read now. Call it once the waiting thread has
    /// resumed; a fan-out that ran no job reports its wall time alone.
    pub fn finish(&self) -> FanOutTiming {
        let wall = self.clock.elapsed_ns();
        let first_start = self.first_start_ns.load(Relaxed);
        if first_start == u64::MAX {
            return FanOutTiming {
                wall_us: micros(wall),
                ..FanOutTiming::default()
            };
        }
        FanOutTiming {
            wall_us: micros(wall),
            first_job_us: micros(first_start),
            job_sum_us: micros(self.job_sum_ns.load(Relaxed)),
            longest_job_us: micros(self.longest_job_ns.load(Relaxed)),
            tail_us: micros(wall.saturating_sub(self.last_end_ns.load(Relaxed))),
        }
    }
}

// Saturates rather than wrapping: a pathological stall reports the ceiling.
fn micros(ns: u64) -> u32 {
    (ns / 1_000).min(u64::from(u32::MAX)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;

    // A clock that reads the next scripted value on every call.
    struct Scripted<'a> {
        readings: &'a [u64],
        next: Cell<usize>,
    }

    impl<'a> Scripted<'a> {
        fn new(readings: &'a [u64]) -> Self {
            Self {
                readings,
                next: Cell::new(0),
            }
        }
    }

    impl FanOutClock for Scripted<'_> {
        fn elapsed_ns(&self) -> u64 {
            let i = self.next.get();
            self.next.set(i + 1);
            self.readings[i]
        }
    }

    #[test]
    fn a_fan_out_reports_its_wait_against_its_work() {
        // Two jobs: 400..1_400 and 900..2_100 us, then the waiter resumes at
        // 2_600 us.
        let timer = FanOutTimer::new(Scripted::new(&[
            400_000, 1_400_000, 900_000, 2_100_000, 2_600_000,
        ]));
        assert_eq!(timer.job(|| 7), 7);
        timer.job(|| ());
        assert_eq!(
            timer.finish(),
            FanOutTiming {
                wall_us: 2_600,
                first_job_us: 400,
                job_sum_us: 2_200,
                longest_job_us: 1_200,
                tail_us: 500,
            }
        );
    }

    #[test]
    fn the_first_job_is_the_earliest_start_whatever_order_jobs_report_in() {
        // The job that started later finishes reporting first.
        let timer = FanOutTimer::new(Scripted::new(&[
            300_000, 500_000, 100_000, 200_000, 600_000,
        ]));
        timer.job(|| ());
        timer.job(|| ());
        let t = timer.finish();
        assert_eq!((t.first_job_us, t.tail_us), (100, 100));
    }

    #[test]
    fn a_job_run_into_a_slot_stores_its_own_duration_there() {
        let timer = FanOutTimer::new(Scripted::new(&[
            100_000, 350_000, 200_000, 1_200_000, 1_300_000,
        ]));
        let (a, b) = (AtomicU32::new(0), AtomicU32::new(0));
        assert_eq!(timer.job_into(&a, || 3), 3);
        timer.job_into(&b, || ());
        assert_eq!((a.into_inner(), b.into_inner()), (250, 1_000));
        let t = timer.finish();
        assert_eq!((t.job_sum_us, t.longest_job_us), (1_250, 1_000));
    }

    #[test]
    fn a_fan_out_with_no_jobs_reports_its_wall_time_alone() {
        let timer = FanOutTimer::new(Scripted::new(&[50_000]));
        assert_eq!(
            timer.finish(),
            FanOutTiming {
                wall_us: 50,
                ..FanOutTiming::default()
            }
        );
    }

    #[test]
    fn readings_saturate_instead_of_wrapping() {
        let timer = FanOutTimer::new(Scripted::new(&[0, u64::MAX, u64::MAX]));
        timer.job(|| ());
        let t = timer.finish();
        assert_eq!(
            (t.wall_us, t.job_sum_us, t.tail_us),
            (u32::MAX, u32::MAX, 0)
        );
    }

    #[test]
    fn merging_adds_the_waits_and_keeps_the_longest_job() {
        let a = FanOutTiming {
            wall_us: 100,
            first_job_us: 10,
            job_sum_us: 150,
            longest_job_us: 60,
            tail_us: 5,
        };
        let b = FanOutTiming {
            wall_us: 300,
            first_job_us: 20,
            job_sum_us: 250,
            longest_job_us: 40,
            tail_us: 7,
        };
        assert_eq!(
            a.merge(b),
            FanOutTiming {
                wall_us: 400,
                first_job_us: 30,
                job_sum_us: 400,
                longest_job_us: 60,
                tail_us: 12,
            }
        );
        assert_eq!(a.merge(FanOutTiming::default()), a);
    }
}
