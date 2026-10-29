// A fixed-size distribution of microsecond readings, so a run of any length
// summarizes in the memory it started with.

use alloc::boxed::Box;
use alloc::vec;

use concinnity_core::math::SaturatingFrom;

// Readings below `1 << LINEAR_BITS` get a bucket each, so small values are
// exact. Above that every power of two splits into `OCTAVE_BUCKETS`, which
// keeps a bucket's width within 1/64 of its lowest reading.
const LINEAR_BITS: u32 = 7;
const OCTAVE_BUCKETS: usize = 1 << (LINEAR_BITS - 1);

// Readings at or past `1 << TOP_BIT` microseconds (about 16.8 s) share the
// last bucket; `max` still holds the largest exactly.
const TOP_BIT: u32 = 24;

const BUCKETS: usize = (1 << LINEAR_BITS) + (TOP_BIT - LINEAR_BITS) as usize * OCTAVE_BUCKETS;

#[derive(Debug, Clone, Copy)]
struct Bucket {
    count: u32,
    // The smallest reading that landed here, so a percentile is a value some
    // frame actually had rather than a bucket edge.
    least: u32,
}

const EMPTY: Bucket = Bucket {
    count: 0,
    least: u32::MAX,
};

/// Counts of readings in log-spaced buckets, with their exact count, total
/// and maximum.
///
/// Every bucket is allocated up front, so recording never allocates. A
/// percentile reads back as the smallest reading in the bucket holding it,
/// which is within 1/64 below the exact nearest-rank value.
#[derive(Debug, Clone)]
pub(super) struct Histogram {
    buckets: Box<[Bucket]>,
    count: usize,
    total: u64,
    max: u32,
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

impl Histogram {
    pub(super) fn new() -> Self {
        Self {
            buckets: vec![EMPTY; BUCKETS].into_boxed_slice(),
            count: 0,
            total: 0,
            max: 0,
        }
    }

    pub(super) fn record(&mut self, value: u32) {
        let bucket = &mut self.buckets[bucket_of(value)];
        bucket.count = bucket.count.saturating_add(1);
        bucket.least = bucket.least.min(value);
        self.count += 1;
        self.total += u64::from(value);
        self.max = self.max.max(value);
    }

    /// How many readings were recorded.
    pub(super) fn count(&self) -> usize {
        self.count
    }

    /// The mean reading, or zero when there are none.
    pub(super) fn mean(&self) -> u32 {
        u32::saturating_from(self.total.checked_div(self.count as u64).unwrap_or(0))
    }

    /// The largest reading, or zero when there are none.
    pub(super) fn max(&self) -> u32 {
        self.max
    }

    /// The nearest-rank reading at quantile `q`, or zero when there are none.
    pub(super) fn percentile(&self, q: f32) -> u32 {
        if self.count == 0 {
            return 0;
        }
        let rank = usize::saturating_from(concinnity_core::math::ceil(q * self.count as f32))
            .clamp(1, self.count);
        let mut seen = 0;
        for bucket in &self.buckets {
            seen += bucket.count as usize;
            if seen >= rank {
                return bucket.least;
            }
        }
        self.max
    }
}

fn bucket_of(value: u32) -> usize {
    if value < 1 << LINEAR_BITS {
        return value as usize;
    }
    let top = u32::BITS - 1 - value.leading_zeros();
    if top >= TOP_BIT {
        return BUCKETS - 1;
    }
    let within = (value >> (top - (LINEAR_BITS - 1))) as usize - OCTAVE_BUCKETS;
    (1 << LINEAR_BITS) + (top - LINEAR_BITS) as usize * OCTAVE_BUCKETS + within
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn of(values: impl IntoIterator<Item = u32>) -> Histogram {
        let mut h = Histogram::new();
        for v in values {
            h.record(v);
        }
        h
    }

    #[test]
    fn an_empty_histogram_reads_zero() {
        let h = Histogram::new();
        assert_eq!(
            (h.count(), h.mean(), h.max(), h.percentile(0.5)),
            (0, 0, 0, 0)
        );
    }

    #[test]
    fn small_readings_are_exact() {
        let h = of(1..=100);
        assert_eq!(h.percentile(0.50), 50);
        assert_eq!(h.percentile(0.95), 95);
        assert_eq!(h.percentile(0.99), 99);
        assert_eq!((h.max(), h.mean()), (100, 50));
    }

    #[test]
    fn a_percentile_is_a_reading_that_happened() {
        let h = of([10_003, 10_007, 10_011]);
        assert_eq!(h.percentile(0.5), 10_003);
        assert_eq!(of([6_000]).percentile(0.5), 6_000);
    }

    #[test]
    fn a_percentile_is_within_a_sixty_fourth_of_the_exact_one() {
        let values: Vec<u32> = (0..10_000).map(|n| 1_000 + n * 37).collect();
        let h = of(values.iter().copied());
        for q in [0.05, 0.5, 0.95, 0.99] {
            let exact =
                values[usize::saturating_from(concinnity_core::math::ceil(q * 10_000.0)) - 1];
            let read = h.percentile(q);
            assert!(read <= exact, "{q}: {read} > {exact}");
            assert!(exact - read <= exact / 64, "{q}: {read} vs {exact}");
        }
    }

    #[test]
    fn buckets_are_contiguous_and_ordered() {
        let mut previous = 0;
        for value in 1..(1 << 20) {
            let bucket = bucket_of(value);
            assert!(bucket == previous || bucket == previous + 1, "{value}");
            previous = bucket;
        }
        assert_eq!(bucket_of((1 << TOP_BIT) - 1), BUCKETS - 1);
        assert_eq!(bucket_of(u32::MAX), BUCKETS - 1);
    }

    #[test]
    fn a_reading_past_the_top_keeps_its_exact_maximum() {
        let h = of([100, u32::MAX]);
        assert_eq!(h.max(), u32::MAX);
        assert_eq!(h.percentile(1.0), u32::MAX);
    }

    #[test]
    fn zero_is_a_reading_like_any_other() {
        let h = of([0, 0, 0, 5_000]);
        assert_eq!(h.percentile(0.5), 0);
        assert_eq!(h.percentile(1.0), 5_000);
    }
}
