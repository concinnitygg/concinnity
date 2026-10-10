//! Whether to fan work out, judged by measuring it both ways.
//!
//! A work estimate alone cannot say whether a split pays: work bound by memory
//! bandwidth (packing into write-combined or device memory) finishes barely
//! sooner across many workers while each of them stalls on the same bus, so
//! the summed work balloons for a small saving. The judge keeps what the last
//! run of each kind cost per unit and fans out only while the split is both
//! clearly faster for the waiting thread and not much more work. Scheduling
//! never changes results, so a wrong call costs one run's time.

use super::FanOutTiming;

// A split must hold the waiting thread for at most this share of the inline
// time, in quarters, to count as faster.
const WIN_QUARTERS: u64 = 3;

// A split may contain at most this many times the inline work: past it, the
// workers' time it takes is worth more than the wait it saves.
const MAX_WORK_RATIO: u64 = 2;

// Runs between re-measurements of a split that lost, so a judgment made under
// other conditions does not stand forever.
const PROBE_INTERVAL: u32 = 240;

// Within a run of one kind, each new measurement moves its estimate by this
// share, in quarters, so a single noisy frame cannot flip the judgment.
const BLEND_QUARTERS: u64 = 1;

// Per-unit cost of one kind of run, in nanoseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Cost {
    wall_ns: u64,
    work_ns: u64,
}

impl Cost {
    fn per_unit(units: usize, timing: FanOutTiming) -> Self {
        let per = |us: u32| u64::from(us) * 1_000 / units as u64;
        Self {
            wall_ns: per(timing.wall_us),
            work_ns: per(timing.job_sum_us),
        }
    }

    fn blend(self, sample: Self) -> Self {
        let mix = |old: u64, new: u64| (old * (4 - BLEND_QUARTERS) + new * BLEND_QUARTERS) / 4;
        Self {
            wall_ns: mix(self.wall_ns, sample.wall_ns),
            work_ns: mix(self.work_ns, sample.work_ns),
        }
    }
}

/// Chooses between running `units` of work inline and fanning them out, from
/// the measured cost of each. Work estimated below `MIN_WORK_NS` nanoseconds
/// always runs inline.
///
/// ```rust
/// # use concinnity_core::profile::{FanOutJudge, FanOutTiming};
/// let timing = |wall: u32, work: u32| FanOutTiming { wall_us: wall, job_sum_us: work, ..Default::default() };
/// let mut judge = FanOutJudge::<300_000>::default();
/// assert!(!judge.fans_out(10_000, 8), "inline is measured first");
/// judge.observe(10_000, false, timing(1_000, 1_000));
/// assert!(judge.fans_out(10_000, 8), "then the split is tried");
/// judge.observe(10_000, true, timing(300, 1_200));
/// assert!(judge.fans_out(10_000, 8), "faster and not much more work");
/// ```
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FanOutJudge<const MIN_WORK_NS: u64> {
    inline: Option<Cost>,
    fanned: Option<Cost>,
    // Whether the last measured run was split, `None` before the first.
    last_fanned: Option<bool>,
    // Runs judged since the split last ran while losing.
    since_probe: u32,
}

impl<const MIN_WORK_NS: u64> FanOutJudge<MIN_WORK_NS> {
    /// Whether to split `units` of work across `workers`. Counts toward the
    /// next re-measurement of a split that lost, so call it once per run.
    pub fn fans_out(&mut self, units: usize, workers: usize) -> bool {
        if workers < 2 || units < 2 {
            return false;
        }
        let Some(inline) = self.inline else {
            return false;
        };
        if inline.wall_ns.saturating_mul(units as u64) < MIN_WORK_NS {
            return false;
        }
        let Some(fanned) = self.fanned else {
            return true;
        };
        if Self::split_wins(inline, fanned) {
            return true;
        }
        self.since_probe += 1;
        if self.since_probe >= PROBE_INTERVAL {
            self.since_probe = 0;
            return true;
        }
        false
    }

    /// Fold a run of `units` into the estimate for its kind: `fanned` when it
    /// was split, inline otherwise. The first run after a switch of kind (or a
    /// re-measurement) replaces that kind's estimate; later runs of the same
    /// kind are blended in. A run with no units changes nothing.
    pub fn observe(&mut self, units: usize, fanned: bool, timing: FanOutTiming) {
        if units == 0 {
            return;
        }
        let sample = Cost::per_unit(units, timing);
        let streak = self.last_fanned == Some(fanned);
        let estimate = match fanned {
            true => &mut self.fanned,
            false => &mut self.inline,
        };
        *estimate = Some(match (*estimate, streak) {
            (Some(old), true) => old.blend(sample),
            _ => sample,
        });
        self.last_fanned = Some(fanned);
    }

    fn split_wins(inline: Cost, fanned: Cost) -> bool {
        fanned.wall_ns * 4 <= inline.wall_ns * WIN_QUARTERS
            && fanned.work_ns <= inline.work_ns.saturating_mul(MAX_WORK_RATIO)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Judge = FanOutJudge<300_000>;
    const UNITS: usize = 10_000;

    fn run(wall_us: u32, job_sum_us: u32) -> FanOutTiming {
        FanOutTiming {
            wall_us,
            job_sum_us,
            ..FanOutTiming::default()
        }
    }

    // A judge that has measured an inline run of `wall_us` over UNITS.
    fn measured_inline(wall_us: u32) -> Judge {
        let mut judge = Judge::default();
        judge.observe(UNITS, false, run(wall_us, wall_us));
        judge
    }

    #[test]
    fn unmeasured_work_runs_inline() {
        assert!(!Judge::default().fans_out(UNITS, 8));
    }

    #[test]
    fn small_work_runs_inline_and_is_never_split() {
        let mut judge = measured_inline(200);
        for _ in 0..2 * PROBE_INTERVAL {
            assert!(!judge.fans_out(UNITS, 8));
        }
    }

    #[test]
    fn large_work_tries_the_split_once_measured_inline() {
        assert!(measured_inline(1_000).fans_out(UNITS, 8));
    }

    // Mac at 32k records: a quarter of the wait for about the same work.
    #[test]
    fn a_split_that_is_faster_and_cheap_keeps_running() {
        let mut judge = measured_inline(1_094);
        judge.observe(UNITS, true, run(274, 992));
        for _ in 0..2 * PROBE_INTERVAL {
            assert!(judge.fans_out(UNITS, 8));
        }
    }

    // DirectX at 64k records into an upload heap: a 30% shorter wait for four
    // times the work.
    #[test]
    fn a_split_that_multiplies_the_work_loses() {
        let mut judge = measured_inline(810);
        judge.observe(UNITS, true, run(576, 3_180));
        assert!(!judge.fans_out(UNITS, 8));
    }

    #[test]
    fn a_split_that_barely_shortens_the_wait_loses() {
        let mut judge = measured_inline(1_000);
        judge.observe(UNITS, true, run(800, 1_100));
        assert!(!judge.fans_out(UNITS, 8));
    }

    #[test]
    fn a_losing_split_is_measured_again_after_the_interval() {
        let mut judge = measured_inline(1_000);
        judge.observe(UNITS, true, run(900, 9_000));
        let probes = (0..3 * PROBE_INTERVAL)
            .filter(|_| judge.fans_out(UNITS, 8))
            .count();
        assert_eq!(probes, 3);
    }

    // One slow split among fast ones is noise, not a verdict.
    #[test]
    fn a_single_slow_split_does_not_flip_the_judgment() {
        let mut judge = measured_inline(1_000);
        for _ in 0..4 {
            judge.observe(UNITS, true, run(300, 1_100));
        }
        judge.observe(UNITS, true, run(1_500, 1_500));
        assert!(judge.fans_out(UNITS, 8));
    }

    // The machine got busy: the split keeps measuring slower, and the work
    // goes back inline.
    #[test]
    fn a_split_that_stops_paying_returns_inline() {
        let mut judge = measured_inline(1_000);
        judge.observe(UNITS, true, run(300, 1_100));
        assert!(judge.fans_out(UNITS, 8));
        let mut runs = 0;
        while judge.fans_out(UNITS, 8) {
            judge.observe(UNITS, true, run(950, 1_100));
            runs += 1;
            assert!(runs < 20, "the split never lost");
        }
        assert!(runs > 1, "one slow split flipped it");
    }

    // A re-measurement after a loss replaces the stale estimate outright, so
    // a split that pays again is taken back at once.
    #[test]
    fn a_probe_that_now_pays_resumes_the_split() {
        let mut judge = measured_inline(1_000);
        judge.observe(UNITS, true, run(900, 9_000));
        for _ in 0..PROBE_INTERVAL {
            if judge.fans_out(UNITS, 8) {
                break;
            }
            judge.observe(UNITS, false, run(1_000, 1_000));
        }
        judge.observe(UNITS, true, run(300, 1_100));
        assert!(judge.fans_out(UNITS, 8));
    }

    #[test]
    fn one_worker_or_one_unit_never_splits() {
        let mut judge = measured_inline(1_000);
        judge.observe(UNITS, true, run(100, 1_000));
        assert!(!judge.fans_out(UNITS, 1));
        assert!(!judge.fans_out(1, 8));
    }

    #[test]
    fn a_run_with_no_units_changes_nothing() {
        let mut judge = measured_inline(1_000);
        let before = judge;
        judge.observe(0, true, run(5, 5));
        assert_eq!(judge, before);
    }
}
