// The distribution summary reported over one column of samples.

use crate::frame_report::histogram::Histogram;
use crate::math::SaturatingFrom;
use serde::Serialize;

/// How one measured quantity was distributed over a run, in microseconds.
///
/// The shape rather than the average: a mean frame time hides exactly the
/// stutter a renderer is judged on, so the tail percentiles and the count of
/// samples that missed their budget are the numbers to read first.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub(super) struct Distribution {
    /// How many samples the summary is over.
    pub count: usize,
    /// Arithmetic mean.
    pub mean_us: u32,
    /// Median.
    pub p50_us: u32,
    /// 95th percentile.
    pub p95_us: u32,
    /// 99th percentile.
    pub p99_us: u32,
    /// Largest sample.
    pub max_us: u32,
    /// How many samples were over the budget the run was measured against.
    pub over_budget: usize,
}

impl Distribution {
    /// The share of samples that missed the budget, in `0..=1`.
    pub(super) fn over_budget_share(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        self.over_budget as f32 / self.count as f32
    }
}

/// One measured quantity over a stretch of frames, counted against the
/// budget as it is recorded.
#[derive(Debug, Clone)]
pub(super) struct Column {
    values: Histogram,
    budget_us: u32,
    over_budget: usize,
}

impl Column {
    pub(super) fn new(budget_us: u32) -> Self {
        Self {
            values: Histogram::new(),
            budget_us,
            over_budget: 0,
        }
    }

    pub(super) fn record(&mut self, value: u32) {
        self.values.record(value);
        self.over_budget += usize::from(value > self.budget_us);
    }

    /// How many readings were recorded.
    pub(super) fn count(&self) -> usize {
        self.values.count()
    }

    /// Summarize what was recorded. Percentiles are nearest-rank, read from
    /// the histogram: a reading some frame had, within 1/64 of the exact one.
    pub(super) fn distribution(&self) -> Distribution {
        let values = &self.values;
        Distribution {
            count: values.count(),
            mean_us: values.mean(),
            p50_us: values.percentile(0.50),
            p95_us: values.percentile(0.95),
            p99_us: values.percentile(0.99),
            max_us: values.max(),
            over_budget: self.over_budget,
        }
    }
}

/// A running mean of microsecond readings.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Mean {
    total: u64,
    count: u64,
}

impl Mean {
    pub(super) fn record(&mut self, value: u32) {
        self.total += u64::from(value);
        self.count += 1;
    }

    /// The mean, or zero when nothing was recorded.
    pub(super) fn get(&self) -> u32 {
        u32::saturating_from(self.total.checked_div(self.count).unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn column(values: &[u32], budget_us: u32) -> Distribution {
        let mut c = Column::new(budget_us);
        for v in values {
            c.record(*v);
        }
        c.distribution()
    }

    #[test]
    fn an_empty_column_summarizes_to_nothing() {
        let d = column(&[], 16_667);
        assert_eq!(d, Distribution::default());
        assert_eq!(d.over_budget_share(), 0.0);
    }

    #[test]
    fn a_single_sample_is_every_percentile() {
        let d = column(&[500], 16_667);
        assert_eq!((d.count, d.mean_us, d.max_us), (1, 500, 500));
        assert_eq!((d.p50_us, d.p95_us, d.p99_us), (500, 500, 500));
    }

    #[test]
    fn the_summary_does_not_depend_on_the_order_it_arrives_in() {
        let ascending: Vec<u32> = (1..=50).map(|n| n * 1_000).collect();
        let descending: Vec<u32> = ascending.iter().rev().copied().collect();
        assert_eq!(column(&ascending, 25_000), column(&descending, 25_000));
    }

    #[test]
    fn the_budget_counts_frames_over_it_and_not_frames_on_it() {
        // A frame exactly at budget made it, so the boundary is exclusive.
        let d = column(&[10, 20, 20, 21, 40], 20);
        assert_eq!(d.over_budget, 2);
        assert!((d.over_budget_share() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn a_tail_that_a_mean_would_hide_shows_up_in_the_percentiles() {
        // Ninety-nine good frames and one terrible one: the mean barely moves
        // while the max and the budget count both name the stutter.
        let mut values = vec![10_000; 99];
        values.push(100_000);
        let d = column(&values, 16_667);
        assert!(d.mean_us < 11_000, "{}", d.mean_us);
        assert_eq!(d.p50_us, 10_000);
        assert_eq!(d.max_us, 100_000);
        assert_eq!(d.over_budget, 1);
    }

    #[test]
    fn a_mean_of_nothing_is_zero() {
        assert_eq!(Mean::default().get(), 0);
        let mut m = Mean::default();
        for v in [2, 4, 6] {
            m.record(v);
        }
        assert_eq!(m.get(), 4);
    }
}
