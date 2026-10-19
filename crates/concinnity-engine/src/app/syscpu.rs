//! Host-CPU queries, the sibling of `app::sysmem`. One value: the CPU time this
//! process has burned across all its threads. Utilization is a rate, not a
//! reading, so `CpuSampler` turns successive queries into one.
//!
//! Deliberately a small hand-rolled platform shim rather than a dependency like
//! `sysinfo`, matching sysmem: one syscall per platform, `None` when the
//! platform call is unavailable or fails, and callers degrade to "unknown".

use concinnity_core::budget::cpu::CpuRate;
use std::time::{Duration, Instant};

// Total CPU time consumed by this process since it started, summed over every
// thread and over user + kernel time. `None` if the platform query is
// unsupported or fails.
fn process_cpu_time() -> Option<Duration> {
    imp::process_cpu_time()
}

/// Samples the process CPU clock into core's `CpuRate`.
#[derive(Debug)]
pub struct CpuSampler {
    origin: Instant,
    rate: CpuRate,
}

impl Default for CpuSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuSampler {
    /// A sampler with no prior reading.
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
            rate: CpuRate::default(),
        }
    }

    /// Samples the process clock now, in cores. `None` on the first call (a
    /// rate needs two readings), when the platform query fails, or when no
    /// time has passed.
    pub fn sample(&mut self) -> Option<f32> {
        self.rate.fold(process_cpu_time()?, self.origin.elapsed())
    }
}

#[cfg(unix)]
mod imp {
    use std::time::Duration;

    // getrusage(RUSAGE_SELF) sums user + kernel time over every thread in the
    // process on both macOS and Linux, so the two share one implementation.
    pub(super) fn process_cpu_time() -> Option<Duration> {
        // SAFETY: `rusage` is a plain C struct of integer fields, so all-zero
        // is a valid inhabitant.
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

        // Burn CPU until the reading advances.
        let mut work: u64 = 0;
        concinnity_testing::eventually("the process CPU time to advance", || {
            for _ in 0..4096 {
                work = work.wrapping_mul(2_654_435_761).wrapping_add(1);
            }
            let now = process_cpu_time().expect("CPU query keeps working");
            assert!(now >= start, "process CPU time must be non-decreasing");
            (now > start).then_some(())
        });
        std::hint::black_box(work);
    }
}
