// Whether a tick's evaluation is worth fanning out, judged by what a run cost
// the last time the host measured one.
//
// A run's cost is decided at run time rather than by its program: a spatial
// search behind a false guard costs nothing, so the same programs over the same
// instances can cost microseconds one tick and milliseconds the next. The last
// measured tick is the estimate. Scheduling never changes results, so a wrong
// guess costs one tick's time and nothing else.

use crate::profile::FanOutTiming;

// Below this much estimated work a split does not pay. Measured on a 12-core
// Mac with the workers asleep between ticks, as they are at frame rate: the
// fan-out loses below about 0.3 ms of work, breaks even near 0.4 ms, and halves
// the wait from about 0.8 ms.
pub(super) const MIN_FANOUT_WORK_NS: u64 = 500_000;

#[derive(Debug, Default)]
pub(super) struct FanOutGate {
    // `None` until a tick with runs has been measured.
    per_job_ns: Option<u64>,
}

impl FanOutGate {
    // Whether `jobs` runs are estimated to cost enough to split across
    // `workers`. Unmeasured work stays on the calling thread, which is what
    // measures it.
    pub(super) fn fans_out(&self, jobs: usize, workers: usize) -> bool {
        workers > 1
            && jobs > 1
            && self
                .per_job_ns
                .is_some_and(|ns| ns.saturating_mul(jobs as u64) >= MIN_FANOUT_WORK_NS)
    }

    // Take the work a tick of `jobs` runs measured as the estimate for the next.
    pub(super) fn observe(&mut self, jobs: usize, timing: FanOutTiming) {
        if jobs > 0 {
            self.per_job_ns = Some(u64::from(timing.job_sum_us) * 1_000 / jobs as u64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worked(job_sum_us: u32) -> FanOutTiming {
        FanOutTiming {
            job_sum_us,
            ..FanOutTiming::default()
        }
    }

    #[test]
    fn unmeasured_work_stays_on_the_calling_thread() {
        assert!(!FanOutGate::default().fans_out(100_000, 8));
    }

    // The benchmark's two shapes: the same 2160 runs asleep (a guard reading
    // the clock) and awake (each searching its neighbors).
    #[test]
    fn the_same_runs_fan_out_only_while_they_are_expensive() {
        let mut gate = FanOutGate::default();
        gate.observe(2_160, worked(150));
        assert!(!gate.fans_out(2_160, 8));
        gate.observe(2_160, worked(3_000));
        assert!(gate.fans_out(2_160, 8));
        gate.observe(2_160, worked(150));
        assert!(!gate.fans_out(2_160, 8));
    }

    #[test]
    fn the_estimate_scales_with_the_job_count() {
        let mut gate = FanOutGate::default();
        // 1 us a run.
        gate.observe(100, worked(100));
        let threshold = (MIN_FANOUT_WORK_NS / 1_000) as usize;
        assert!(!gate.fans_out(threshold - 1, 8));
        assert!(gate.fans_out(threshold, 8));
    }

    #[test]
    fn one_worker_or_one_job_never_fans_out() {
        let mut gate = FanOutGate::default();
        gate.observe(1, worked(10_000));
        assert!(!gate.fans_out(1, 8));
        assert!(!gate.fans_out(1_000, 1));
        assert!(gate.fans_out(2, 8));
    }

    #[test]
    fn a_tick_with_no_runs_keeps_the_last_estimate() {
        let mut gate = FanOutGate::default();
        gate.observe(10, worked(10_000));
        gate.observe(0, worked(0));
        assert!(gate.fans_out(10, 8));
    }
}
