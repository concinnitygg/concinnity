//! Whether a fan-out is worth its wake-up cost, judged by what its work cost
//! the last time it was measured.
//!
//! Work cost is often decided at run time rather than by its shape: a spatial
//! search behind a false guard costs nothing, so the same jobs can cost
//! microseconds one frame and milliseconds the next. The last measured run is
//! the estimate. Scheduling never changes results, so a wrong guess costs one
//! run's time and nothing else.

use super::FanOutTiming;

/// Fans work out once it is estimated at `MIN_WORK_NS` nanoseconds or more,
/// from the per-unit cost the previous run measured. Below that, splitting
/// does not pay for waking the workers.
///
/// ```rust
/// # use concinnity_core::profile::{FanOutGate, FanOutTiming};
/// let mut gate = FanOutGate::<500_000>::default();
/// assert!(!gate.fans_out(10_000, 8), "unmeasured work stays inline");
/// gate.observe(10_000, FanOutTiming { job_sum_us: 1_000, ..FanOutTiming::default() });
/// assert!(gate.fans_out(10_000, 8));
/// assert!(!gate.fans_out(4_000, 8));
/// ```
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FanOutGate<const MIN_WORK_NS: u64> {
    // `None` until a run with units has been measured.
    per_unit_ns: Option<u64>,
}

impl<const MIN_WORK_NS: u64> FanOutGate<MIN_WORK_NS> {
    /// Whether `units` of work are estimated to cost enough to split across
    /// `workers`. Unmeasured work stays on the calling thread, which is what
    /// measures it.
    pub fn fans_out(&self, units: usize, workers: usize) -> bool {
        workers > 1
            && units > 1
            && self
                .per_unit_ns
                .is_some_and(|ns| ns.saturating_mul(units as u64) >= MIN_WORK_NS)
    }

    /// Take the work a run of `units` measured as the estimate for the next.
    /// A run with no units keeps the last estimate.
    pub fn observe(&mut self, units: usize, timing: FanOutTiming) {
        if units > 0 {
            self.per_unit_ns = Some(u64::from(timing.job_sum_us) * 1_000 / units as u64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN_WORK_NS: u64 = 500_000;
    type Gate = FanOutGate<MIN_WORK_NS>;

    fn worked(job_sum_us: u32) -> FanOutTiming {
        FanOutTiming {
            job_sum_us,
            ..FanOutTiming::default()
        }
    }

    #[test]
    fn unmeasured_work_stays_on_the_calling_thread() {
        assert!(!Gate::default().fans_out(100_000, 8));
    }

    // The same units cheap one run and expensive the next, as behaviors asleep
    // behind a guard and then awake.
    #[test]
    fn the_same_units_fan_out_only_while_they_are_expensive() {
        let mut gate = Gate::default();
        gate.observe(2_160, worked(150));
        assert!(!gate.fans_out(2_160, 8));
        gate.observe(2_160, worked(3_000));
        assert!(gate.fans_out(2_160, 8));
        gate.observe(2_160, worked(150));
        assert!(!gate.fans_out(2_160, 8));
    }

    #[test]
    fn the_estimate_scales_with_the_unit_count() {
        let mut gate = Gate::default();
        // 1 us a unit.
        gate.observe(100, worked(100));
        let threshold = (MIN_WORK_NS / 1_000) as usize;
        assert!(!gate.fans_out(threshold - 1, 8));
        assert!(gate.fans_out(threshold, 8));
    }

    #[test]
    fn one_worker_or_one_unit_never_fans_out() {
        let mut gate = Gate::default();
        gate.observe(1, worked(10_000));
        assert!(!gate.fans_out(1, 8));
        assert!(!gate.fans_out(1_000, 1));
        assert!(gate.fans_out(2, 8));
    }

    #[test]
    fn a_run_with_no_units_keeps_the_last_estimate() {
        let mut gate = Gate::default();
        gate.observe(10, worked(10_000));
        gate.observe(0, worked(0));
        assert!(gate.fans_out(10, 8));
    }

    #[test]
    fn the_threshold_is_the_gates_own() {
        let mut low = FanOutGate::<1_000>::default();
        let mut high = FanOutGate::<1_000_000>::default();
        low.observe(10, worked(5));
        high.observe(10, worked(5));
        assert!(low.fans_out(10, 8));
        assert!(!high.fans_out(10, 8));
    }
}
