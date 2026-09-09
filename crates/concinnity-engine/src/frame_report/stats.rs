// The distribution summary reported over one column of samples.

use serde::Serialize;

/// How one measured quantity was distributed over a run, in microseconds.
///
/// The shape rather than the average: a mean frame time hides exactly the
/// stutter a renderer is judged on, so the tail percentiles and the count of
/// samples that missed their budget are the numbers to read first.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Distribution {
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
    /// Summarize `values` against `budget_us`. Sorts the slice in place, since
    /// the caller's copy is scratch either way.
    ///
    /// Percentiles are nearest-rank: the smallest sample at or above the
    /// quantile, never interpolated between two. That keeps every reported
    /// number a frame that actually happened.
    pub fn of(values: &mut [u32], budget_us: u32) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        values.sort_unstable();
        let count = values.len();
        let total: u64 = values.iter().map(|v| u64::from(*v)).sum();
        Self {
            count,
            mean_us: (total / count as u64) as u32,
            p50_us: nearest_rank(values, 0.50),
            p95_us: nearest_rank(values, 0.95),
            p99_us: nearest_rank(values, 0.99),
            max_us: values[count - 1],
            over_budget: values.iter().filter(|v| **v > budget_us).count(),
        }
    }

    /// The share of samples that missed the budget, in `0..=1`.
    pub fn over_budget_share(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        self.over_budget as f32 / self.count as f32
    }
}

// The smallest sample at or above quantile `q` of an already sorted slice.
fn nearest_rank(sorted: &[u32], q: f32) -> u32 {
    let rank = (q * sorted.len() as f32).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

// The mean of `values`, or zero when there are none.
pub(super) fn mean_u32(values: impl Iterator<Item = u32>) -> u32 {
    let mut total: u64 = 0;
    let mut count: u64 = 0;
    for v in values {
        total += u64::from(v);
        count += 1;
    }
    total.checked_div(count).unwrap_or(0) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_column_summarizes_to_nothing() {
        let d = Distribution::of(&mut [], 16_667);
        assert_eq!(d, Distribution::default());
        assert_eq!(d.over_budget_share(), 0.0);
    }

    #[test]
    fn a_single_sample_is_every_percentile() {
        let d = Distribution::of(&mut [500], 16_667);
        assert_eq!((d.count, d.mean_us, d.max_us), (1, 500, 500));
        assert_eq!((d.p50_us, d.p95_us, d.p99_us), (500, 500, 500));
    }

    #[test]
    fn percentiles_are_samples_that_actually_happened() {
        // 1..=100, so the q-th percentile is the value q by construction. An
        // interpolating definition would report 95.5 for p95, which is a frame
        // time no frame had.
        let mut values: Vec<u32> = (1..=100).collect();
        let d = Distribution::of(&mut values, 1_000);
        assert_eq!(d.p50_us, 50);
        assert_eq!(d.p95_us, 95);
        assert_eq!(d.p99_us, 99);
        assert_eq!(d.max_us, 100);
        assert_eq!(d.mean_us, 50);
    }

    #[test]
    fn the_summary_does_not_depend_on_the_order_it_arrives_in() {
        let mut ascending: Vec<u32> = (1..=50).collect();
        let mut descending: Vec<u32> = (1..=50).rev().collect();
        assert_eq!(
            Distribution::of(&mut ascending, 25),
            Distribution::of(&mut descending, 25)
        );
    }

    #[test]
    fn the_budget_counts_frames_over_it_and_not_frames_on_it() {
        // A frame exactly at budget made it, so the boundary is exclusive.
        let mut values = vec![10, 20, 20, 21, 40];
        let d = Distribution::of(&mut values, 20);
        assert_eq!(d.over_budget, 2);
        assert!((d.over_budget_share() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn a_tail_that_a_mean_would_hide_shows_up_in_the_percentiles() {
        // Ninety-nine good frames and one terrible one: the mean barely moves
        // while the max and the budget count both name the stutter.
        let mut values = vec![10_000; 99];
        values.push(100_000);
        let d = Distribution::of(&mut values, 16_667);
        assert!(d.mean_us < 11_000, "{}", d.mean_us);
        assert_eq!(d.p50_us, 10_000);
        assert_eq!(d.max_us, 100_000);
        assert_eq!(d.over_budget, 1);
    }

    #[test]
    fn the_mean_helper_folds_an_empty_iterator_to_zero() {
        assert_eq!(mean_u32(core::iter::empty()), 0);
        assert_eq!(mean_u32([2, 4, 6].into_iter()), 4);
    }
}
