// src/app/syscpu.rs
//
// Host-CPU queries, the sibling of `app::sysmem`. One value: the CPU time this
// process has burned across all its threads. Utilization is a rate, not a
// reading, so `CpuSampler` turns successive queries into one.
//
// Deliberately a small hand-rolled platform shim rather than a dependency like
// `sysinfo`, matching sysmem: one syscall per platform, `None` when the
// platform call is unavailable or fails, and callers degrade to "unknown".

use std::time::{Duration, Instant};

// Total CPU time consumed by this process since it started, summed over every
// thread and over user + kernel time. `None` if the platform query is
// unsupported or fails.
pub fn process_cpu_time() -> Option<Duration> {
    imp::process_cpu_time()
}

// Turns successive `process_cpu_time` readings into a utilization rate.
//
// The unit is cores: 1.0 means one core saturated for the whole interval, 4.0
// means four. It is deliberately not a percentage, because the useful
// comparison is against `ThreadBudget::total_cores` rather than against 100.
#[derive(Debug, Default)]
pub struct CpuSampler {
    last: Option<(Duration, Instant)>,
}

impl CpuSampler {
    pub fn new() -> Self {
        Self { last: None }
    }

    // Samples the process clock now. `None` on the first call (a rate needs two
    // readings), when the platform query fails, or when no time has passed.
    pub fn sample(&mut self) -> Option<f32> {
        self.fold(process_cpu_time()?, Instant::now())
    }

    // The rate itself, split out from the clock reads so it can be tested
    // against known intervals.
    fn fold(&mut self, cpu: Duration, at: Instant) -> Option<f32> {
        let rate = match self.last {
            Some((last_cpu, last_at)) => {
                let wall = at.saturating_duration_since(last_at).as_secs_f64();
                // Two samples in the same instant have no rate to report; the
                // stored reading stays put so the next one spans a real gap.
                if wall <= 0.0 {
                    return None;
                }
                // CPU time is monotonic, but a failed platform read must not be
                // able to produce a negative rate.
                let busy = cpu.saturating_sub(last_cpu).as_secs_f64();
                Some((busy / wall) as f32)
            }
            None => None,
        };
        self.last = Some((cpu, at));
        rate
    }
}

#[cfg(unix)]
mod imp {
    use std::time::Duration;

    // getrusage(RUSAGE_SELF) sums user + kernel time over every thread in the
    // process on both macOS and Linux, so the two share one implementation.
    pub(super) fn process_cpu_time() -> Option<Duration> {
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        // SAFETY: `usage` is a correctly sized rusage output buffer and
        // RUSAGE_SELF is a valid `who` value.
        let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        (rc == 0).then(|| timeval(usage.ru_utime) + timeval(usage.ru_stime))
    }

    fn timeval(t: libc::timeval) -> Duration {
        let secs = t.tv_sec.max(0) as u64;
        let micros = t.tv_usec.clamp(0, 999_999) as u32;
        Duration::new(secs, micros * 1_000)
    }
}

#[cfg(windows)]
mod imp {
    use std::time::Duration;
    use windows::Win32::Foundation::FILETIME;
    use windows::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

    // GetProcessTimes reports kernel + user time across every thread. The
    // creation/exit times come back in the same call and are unused.
    pub(super) fn process_cpu_time() -> Option<Duration> {
        let mut creation = FILETIME::default();
        let mut exit = FILETIME::default();
        let mut kernel = FILETIME::default();
        let mut user = FILETIME::default();
        // SAFETY: all four out-params are valid FILETIME buffers, as the API
        // requires; the current-process pseudo-handle needs no close.
        unsafe {
            GetProcessTimes(
                GetCurrentProcess(),
                &mut creation,
                &mut exit,
                &mut kernel,
                &mut user,
            )
        }
        .ok()?;
        Some(filetime(kernel) + filetime(user))
    }

    // FILETIME counts 100-nanosecond ticks.
    fn filetime(ft: FILETIME) -> Duration {
        let ticks = ((ft.dwHighDateTime as u64) << 32) | ft.dwLowDateTime as u64;
        Duration::from_nanos(ticks.saturating_mul(100))
    }
}

#[cfg(not(any(unix, windows)))]
mod imp {
    use std::time::Duration;

    pub(super) fn process_cpu_time() -> Option<Duration> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // On a supported platform the query must return a clock that advances as the
    // process burns CPU. A single reading can land inside one scheduler tick
    // (Windows `GetProcessTimes` is ~15ms granular), so spin until it moves
    // rather than assuming any time has accumulated yet.
    #[test]
    fn query_returns_a_plausible_value() {
        if !cfg!(any(unix, windows)) {
            return;
        }

        let start = process_cpu_time().expect("CPU query works on this platform");

        // Burn CPU until the reading advances, bounded so a broken query fails
        // the test instead of hanging it.
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut work: u64 = 0;
        let latest = loop {
            for _ in 0..4096 {
                work = work.wrapping_mul(2_654_435_761).wrapping_add(1);
            }
            let now = process_cpu_time().expect("CPU query keeps working");
            assert!(now >= start, "process CPU time must be non-decreasing");
            if now > start || Instant::now() >= deadline {
                break now;
            }
        };
        std::hint::black_box(work);

        assert!(
            latest > start,
            "process CPU time should advance while the test burns CPU"
        );
    }

    // A rate needs two readings, so the first sample only primes the sampler.
    #[test]
    fn first_fold_reports_no_rate() {
        let mut sampler = CpuSampler::new();
        assert_eq!(sampler.fold(Duration::from_secs(1), Instant::now()), None);
    }

    // One core busy for half of a two-second interval is 0.5 cores.
    #[test]
    fn fold_reports_cpu_time_over_wall_time() {
        let t0 = Instant::now();
        let mut sampler = CpuSampler::new();
        sampler.fold(Duration::ZERO, t0);

        let rate = sampler
            .fold(Duration::from_secs(1), t0 + Duration::from_secs(2))
            .expect("second fold spans a real interval");
        assert!((rate - 0.5).abs() < 1e-5, "expected 0.5 cores, got {rate}");
    }

    // Cores are not capped at one: four cores saturated for an interval reads
    // as 4.0, which is what makes the value comparable to the core count.
    #[test]
    fn fold_reports_more_than_one_core() {
        let t0 = Instant::now();
        let mut sampler = CpuSampler::new();
        sampler.fold(Duration::ZERO, t0);

        let rate = sampler
            .fold(Duration::from_secs(4), t0 + Duration::from_secs(1))
            .expect("second fold spans a real interval");
        assert!((rate - 4.0).abs() < 1e-5, "expected 4.0 cores, got {rate}");
    }

    // Successive folds each measure their own interval rather than accumulating
    // against the first reading.
    #[test]
    fn fold_measures_each_interval_independently() {
        let t0 = Instant::now();
        let mut sampler = CpuSampler::new();
        sampler.fold(Duration::ZERO, t0);
        sampler.fold(Duration::from_secs(1), t0 + Duration::from_secs(1));

        let rate = sampler
            .fold(Duration::from_millis(1500), t0 + Duration::from_secs(2))
            .expect("third fold spans a real interval");
        assert!((rate - 0.5).abs() < 1e-5, "expected 0.5 cores, got {rate}");
    }

    // Two reads in the same instant have no interval to divide by.
    #[test]
    fn fold_with_no_elapsed_time_reports_no_rate() {
        let t0 = Instant::now();
        let mut sampler = CpuSampler::new();
        sampler.fold(Duration::ZERO, t0);
        assert_eq!(sampler.fold(Duration::from_secs(1), t0), None);
    }
}
