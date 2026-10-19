//! Process CPU utilization from successive readings of its CPU-time clock.

use crate::math::narrow_f32;
use core::time::Duration;

/// Turns successive process CPU-time readings into a utilization rate.
///
/// The unit is cores: 1.0 means one core saturated for the whole interval, 4.0
/// means four. It is deliberately not a percentage, because the useful
/// comparison is against `ThreadBudget::total_cores` rather than against 100.
#[derive(Debug, Default)]
pub struct CpuRate {
    last: Option<(Duration, Duration)>,
}

impl CpuRate {
    /// Fold in the process's CPU time `cpu`, read at `at` (an offset from any
    /// fixed origin). `None` on the first reading (a rate needs two) or when
    /// no time has passed.
    pub fn fold(&mut self, cpu: Duration, at: Duration) -> Option<f32> {
        let rate = match self.last {
            Some((last_cpu, last_at)) => {
                let wall = at.saturating_sub(last_at).as_secs_f64();
                // Two samples in the same instant have no rate to report; the
                // stored reading stays put so the next one spans a real gap.
                if wall <= 0.0 {
                    return None;
                }
                // CPU time is monotonic, but a failed platform read must not be
                // able to produce a negative rate.
                let busy = cpu.saturating_sub(last_cpu).as_secs_f64();
                Some(narrow_f32(busy / wall))
            }
            None => None,
        };
        self.last = Some((cpu, at));
        rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A rate needs two readings, so the first sample only primes the sampler.
    #[test]
    fn first_fold_reports_no_rate() {
        let mut sampler = CpuRate::default();
        assert_eq!(sampler.fold(Duration::from_secs(1), Duration::ZERO), None);
    }

    // One core busy for half of a two-second interval is 0.5 cores.
    #[test]
    fn fold_reports_cpu_time_over_wall_time() {
        let t0 = Duration::from_secs(10);
        let mut sampler = CpuRate::default();
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
        let t0 = Duration::from_secs(10);
        let mut sampler = CpuRate::default();
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
        let t0 = Duration::from_secs(10);
        let mut sampler = CpuRate::default();
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
        let t0 = Duration::from_secs(10);
        let mut sampler = CpuRate::default();
        sampler.fold(Duration::ZERO, t0);
        assert_eq!(sampler.fold(Duration::from_secs(1), t0), None);
    }
}
