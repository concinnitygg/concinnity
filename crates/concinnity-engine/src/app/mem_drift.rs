// src/app/mem_drift.rs
//
// Long-session memory drift: whether the process's growth came from the Rust
// heap or from somewhere the Rust heap cannot see.
//
// Resident set size alone cannot tell a leak from a fragmenting allocator, and
// the tracked heap alone cannot either, yet the two failures have opposite
// remedies. Read together across a session they separate: a heap that grows
// while the rest holds steady is ours to fix, and a resident set that grows
// while the heap holds steady is not.
//
// The instantaneous pair says nothing worth reading. A healthy process holds
// most of its resident set outside the Rust heap -- the binary image, thread
// stacks, driver allocations, mapped asset blobs -- so the ratio between them
// has no value to threshold against. Only its movement over a long session
// does, which is why this tracks growth from a baseline rather than a ratio.
//
// What the growth outside the heap does not do is name its own cause. It is
// fragmentation, driver growth and newly mapped assets together; separating
// those further is the ledger's job, not this module's.

use std::time::Instant;

// How far a term must move, as a percentage of the memory budget, before it
// counts as drift, and how far back it must fall before it stops counting. Any
// long session jitters by a few megabytes and none of it means anything, and a
// figure sitting on a single threshold would alternate its reading twice a
// second; the gap between the two is the hysteresis band that stops it, the
// same way the streaming valve's engage and release marks do.
const SIGNIFICANT_PCT: u64 = 2;
const RELEASE_PCT: u64 = 1;

// Per-sample RSS growth, as a percentage of the previous sample, under which the
// session counts as flat, and how many flat samples in a row settle it. Startup
// climbs steeply while blobs load and pipelines build, and the driver keeps
// allocating for seconds after the first frame; one flat sample lands in the
// gaps of that, so the baseline waits for a run of them.
const SETTLE_GROWTH_PCT: u64 = 1;
const SETTLE_STREAK: u32 = 4;

// Samples after which the baseline is captured whether or not RSS ever settled.
// A world that streams continuously may never settle, and drift measured from a
// busy baseline still beats no drift at all.
const SETTLE_DEADLINE_SAMPLES: u32 = 240;

// Which terms moved, once both are read against the budget.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum DriftVerdict {
    // Neither term has moved against the budget.
    #[default]
    Settled,
    // The tracked heap grew: the engine is holding more than it was.
    Heap,
    // The resident set grew and the tracked heap did not, so the growth is
    // memory Rust never allocated.
    OutsideHeap,
    // Both terms grew.
    Both,
}

impl DriftVerdict {
    pub fn label(self) -> &'static str {
        match self {
            DriftVerdict::Settled => "settled",
            DriftVerdict::Heap => "heap",
            DriftVerdict::OutsideHeap => "outside-heap",
            DriftVerdict::Both => "heap and outside-heap",
        }
    }
}

// Process memory movement since the session settled. Growth is signed: a term
// that shrank reads negative.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MemoryDrift {
    // Bytes the tracked heap has moved since the baseline.
    pub heap_growth_bytes: i64,
    // Bytes of resident-set movement the tracked heap does not account for.
    pub outside_heap_growth_bytes: i64,
    // Seconds the movement is measured over, so a growth figure has a rate
    // behind it: 400 MB over three hours and over three minutes are different
    // problems.
    pub window_secs: u64,
    pub verdict: DriftVerdict,
}

// The baseline the drift is measured from, captured once the session settles.
#[derive(Clone, Copy, Debug)]
struct Baseline {
    at: Instant,
    rss_bytes: u64,
    heap_live_bytes: u64,
}

// Holds the baseline and turns each sample into a `MemoryDrift`. Reports
// nothing until the session settles, because a drift measured from a startup
// figure is noise wearing a number's clothes.
#[derive(Debug, Default)]
pub struct DriftTracker {
    baseline: Option<Baseline>,
    // Previous RSS, for the settle test that runs before a baseline exists.
    last_rss: Option<u64>,
    // Consecutive flat samples so far, and every sample seen before the
    // baseline was captured (which the deadline is measured against).
    flat_streak: u32,
    samples_before_baseline: u32,
    // Whether each term currently counts as drifting. Latched, so a figure
    // hovering at the threshold holds its reading instead of alternating.
    heap_moved: bool,
    outside_heap_moved: bool,
}

impl DriftTracker {
    // Fold one sample in, returning the drift once a baseline exists.
    pub fn sample(
        &mut self,
        rss_bytes: u64,
        heap_live_bytes: u64,
        budget_bytes: u64,
    ) -> Option<MemoryDrift> {
        self.sample_at(rss_bytes, heap_live_bytes, budget_bytes, Instant::now())
    }

    // The clock split out so a test can drive the window without waiting on it.
    fn sample_at(
        &mut self,
        rss_bytes: u64,
        heap_live_bytes: u64,
        budget_bytes: u64,
        now: Instant,
    ) -> Option<MemoryDrift> {
        let Some(base) = self.baseline else {
            self.settle(rss_bytes, heap_live_bytes, now);
            return None;
        };
        let heap_growth = growth(heap_live_bytes, base.heap_live_bytes);
        let outside_heap_growth = growth(rss_bytes, base.rss_bytes) - heap_growth;
        self.heap_moved = moved(heap_growth, budget_bytes, self.heap_moved);
        self.outside_heap_moved = moved(outside_heap_growth, budget_bytes, self.outside_heap_moved);
        Some(MemoryDrift {
            heap_growth_bytes: heap_growth,
            outside_heap_growth_bytes: outside_heap_growth,
            window_secs: now.saturating_duration_since(base.at).as_secs(),
            verdict: verdict(self.heap_moved, self.outside_heap_moved),
        })
    }

    // Capture the baseline once RSS has held flat for a run of samples, or once
    // the deadline passes for a world that never stops loading.
    fn settle(&mut self, rss_bytes: u64, heap_live_bytes: u64, now: Instant) {
        self.samples_before_baseline = self.samples_before_baseline.saturating_add(1);
        let flat = self.last_rss.is_some_and(|prev| {
            rss_bytes.saturating_sub(prev) <= prev.saturating_mul(SETTLE_GROWTH_PCT) / 100
        });
        self.flat_streak = if flat { self.flat_streak + 1 } else { 0 };
        self.last_rss = Some(rss_bytes);
        if self.flat_streak >= SETTLE_STREAK
            || self.samples_before_baseline >= SETTLE_DEADLINE_SAMPLES
        {
            self.baseline = Some(Baseline {
                at: now,
                rss_bytes,
                heap_live_bytes,
            });
        }
    }
}

// Signed movement from `base` to `now`, saturating rather than wrapping at the
// extremes a bogus platform reading could produce.
fn growth(now: u64, base: u64) -> i64 {
    (now as i128 - base as i128).clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

// Whether one term counts as drifting, given whether it already did. A term
// that has not moved must clear the significance mark to start counting, and one
// that has must fall back under the release mark to stop; between them it holds
// whatever it was. Shrinkage never counts, and a zero budget has no scale to
// judge against so nothing counts.
fn moved(growth: i64, budget_bytes: u64, was_moved: bool) -> bool {
    let pct = if was_moved {
        RELEASE_PCT
    } else {
        SIGNIFICANT_PCT
    };
    let threshold = budget_bytes.saturating_mul(pct) / 100;
    threshold > 0 && growth > 0 && growth as u64 > threshold
}

fn verdict(heap_moved: bool, outside_heap_moved: bool) -> DriftVerdict {
    match (heap_moved, outside_heap_moved) {
        (true, true) => DriftVerdict::Both,
        (true, false) => DriftVerdict::Heap,
        (false, true) => DriftVerdict::OutsideHeap,
        (false, false) => DriftVerdict::Settled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const MIB: u64 = 1024 * 1024;
    const BUDGET: u64 = 1000 * MIB;
    // 2% of the budget: the smallest movement that counts as drift.
    const SIGNIFICANT: u64 = 20 * MIB;

    // A tracker already past its baseline, so a test can drive drift directly.
    // One sample to compare against plus a full flat run settles it.
    fn settled(rss: u64, heap: u64, at: Instant) -> DriftTracker {
        let mut t = DriftTracker::default();
        for _ in 0..=SETTLE_STREAK {
            assert_eq!(t.sample_at(rss, heap, BUDGET, at), None);
        }
        assert!(
            t.baseline.is_some(),
            "a run of steady samples settles the baseline"
        );
        t
    }

    // The reading the whole module exists for: RSS climbing while the heap holds
    // steady is growth Rust never made, and the remedy is not a leak hunt.
    #[test]
    fn a_climbing_rss_against_a_flat_heap_reads_as_outside_the_heap() {
        let start = Instant::now();
        let mut t = settled(2000 * MIB, 400 * MIB, start);

        let d = t
            .sample_at(
                2400 * MIB,
                400 * MIB,
                BUDGET,
                start + Duration::from_secs(3600),
            )
            .expect("the baseline is captured");
        assert_eq!(d.verdict, DriftVerdict::OutsideHeap);
        assert_eq!(d.heap_growth_bytes, 0);
        assert_eq!(d.outside_heap_growth_bytes, (400 * MIB) as i64);
        assert_eq!(d.window_secs, 3600);
    }

    // The opposite reading, which is the one that *is* a leak hunt: the heap
    // took every byte of the growth.
    #[test]
    fn a_climbing_heap_carrying_the_rss_reads_as_ours() {
        let start = Instant::now();
        let mut t = settled(2000 * MIB, 400 * MIB, start);

        let d = t
            .sample_at(
                2400 * MIB,
                800 * MIB,
                BUDGET,
                start + Duration::from_secs(60),
            )
            .expect("the baseline is captured");
        assert_eq!(d.verdict, DriftVerdict::Heap);
        assert_eq!(d.heap_growth_bytes, (400 * MIB) as i64);
        // The heap explains all of it, so nothing is left outside it.
        assert_eq!(d.outside_heap_growth_bytes, 0);
    }

    #[test]
    fn both_terms_growing_are_reported_as_both() {
        let start = Instant::now();
        let mut t = settled(2000 * MIB, 400 * MIB, start);
        let d = t
            .sample_at(2600 * MIB, 700 * MIB, BUDGET, start)
            .expect("the baseline is captured");
        assert_eq!(d.verdict, DriftVerdict::Both);
        assert_eq!(d.heap_growth_bytes, (300 * MIB) as i64);
        assert_eq!(d.outside_heap_growth_bytes, (300 * MIB) as i64);
    }

    // A few megabytes either way is what every long session does; the verdict
    // must not call that drift.
    #[test]
    fn movement_under_the_threshold_stays_settled() {
        let start = Instant::now();
        let mut t = settled(2000 * MIB, 400 * MIB, start);
        let d = t
            .sample_at(
                2000 * MIB + SIGNIFICANT - 1,
                400 * MIB + SIGNIFICANT - 1,
                BUDGET,
                start,
            )
            .expect("the baseline is captured");
        assert_eq!(d.verdict, DriftVerdict::Settled);
    }

    // A figure sitting on the threshold must not alternate its reading. A live
    // run flapped outside-heap / settled in 250 ms across a growth of 350 then
    // 317 MiB against a 328 MiB mark, which is a log line twice a second saying
    // nothing. Once a term counts as drifting it holds until it falls back to
    // the release mark.
    #[test]
    fn a_growth_hovering_at_the_threshold_holds_its_reading() {
        let start = Instant::now();
        let mut t = settled(2000 * MIB, 400 * MIB, start);
        let rss_at = |outside: u64| 2000 * MIB + outside;

        // Just over the significance mark: the term starts counting.
        let d = t
            .sample_at(rss_at(SIGNIFICANT + MIB), 400 * MIB, BUDGET, start)
            .expect("the baseline is captured");
        assert_eq!(d.verdict, DriftVerdict::OutsideHeap);

        // Back under it, but nowhere near the release mark: the reading holds
        // rather than flipping back.
        for outside in [SIGNIFICANT - MIB, SIGNIFICANT + MIB, SIGNIFICANT - MIB] {
            let d = t
                .sample_at(rss_at(outside), 400 * MIB, BUDGET, start)
                .expect("the baseline is captured");
            assert_eq!(
                d.verdict,
                DriftVerdict::OutsideHeap,
                "a term at {outside} bytes flipped inside the hysteresis band"
            );
        }

        // All the way back under the release mark: it stops counting.
        let released = BUDGET * RELEASE_PCT / 100 - MIB;
        let d = t
            .sample_at(rss_at(released), 400 * MIB, BUDGET, start)
            .expect("the baseline is captured");
        assert_eq!(d.verdict, DriftVerdict::Settled);
    }

    // Shrinking is not drift. Growth is signed so the numbers stay honest, but
    // a process handing memory back is not a fault to report.
    #[test]
    fn shrinking_is_reported_but_never_read_as_drift() {
        let start = Instant::now();
        let mut t = settled(2000 * MIB, 400 * MIB, start);
        let d = t
            .sample_at(1500 * MIB, 300 * MIB, BUDGET, start)
            .expect("the baseline is captured");
        assert_eq!(d.verdict, DriftVerdict::Settled);
        assert_eq!(d.heap_growth_bytes, -((100 * MIB) as i64));
        assert_eq!(d.outside_heap_growth_bytes, -((400 * MIB) as i64));
    }

    // The failure a fixed warm-up timer produces: a baseline captured while
    // assets are still loading makes every later reading a measurement against
    // a number that was never steady.
    #[test]
    fn a_climbing_startup_does_not_become_the_baseline() {
        let start = Instant::now();
        let mut t = DriftTracker::default();
        // Each sample adds a tenth of the last, far above the settle threshold.
        let mut rss = 100 * MIB;
        for _ in 0..20 {
            assert_eq!(t.sample_at(rss, 50 * MIB, BUDGET, start), None);
            rss += rss / 10;
        }
        assert!(t.baseline.is_none(), "a climbing session has not settled");

        // Levelling off is recognised over a run of flat samples. The first
        // sample here still carries the last climb, so it takes one more than
        // the streak, and the baseline is the sample that completes it.
        for _ in 0..=SETTLE_STREAK {
            assert_eq!(t.sample_at(rss, 50 * MIB, BUDGET, start), None);
        }
        assert!(t.baseline.is_some(), "a flat run settles the baseline");
        assert!(t.sample_at(rss, 50 * MIB, BUDGET, start).is_some());
    }

    // A single flat sample in the middle of a climb is not a settled session.
    // The driver keeps allocating for seconds after the first frame and lands a
    // flat sample in the gaps; a baseline captured on one of those measures
    // every later reading against a number that was still moving.
    #[test]
    fn one_flat_sample_inside_a_climb_does_not_settle_it() {
        let start = Instant::now();
        let mut t = DriftTracker::default();
        let mut rss = 100 * MIB;
        for _ in 0..10 {
            // A flat pair, then a jump: the streak restarts every time.
            t.sample_at(rss, 50 * MIB, BUDGET, start);
            t.sample_at(rss, 50 * MIB, BUDGET, start);
            rss += rss / 4;
            t.sample_at(rss, 50 * MIB, BUDGET, start);
        }
        assert!(
            t.baseline.is_none(),
            "a climb interrupted by single flat samples is not settled"
        );
    }

    // A world that streams forever never settles, and no baseline at all would
    // mean no signal for exactly the long sessions this is built for.
    #[test]
    fn a_session_that_never_settles_captures_a_baseline_at_the_deadline() {
        let start = Instant::now();
        let mut t = DriftTracker::default();
        let mut rss = 100 * MIB;
        for _ in 0..SETTLE_DEADLINE_SAMPLES {
            t.sample_at(rss, 50 * MIB, BUDGET, start);
            rss += rss / 10;
        }
        assert!(
            t.baseline.is_some(),
            "the deadline captures a baseline even while RSS climbs"
        );
    }

    // Without a budget there is no scale to call a movement significant against,
    // so the numbers are still reported and the verdict withholds judgement.
    #[test]
    fn a_zero_budget_reports_movement_without_a_verdict() {
        let start = Instant::now();
        let mut t = DriftTracker::default();
        for _ in 0..=SETTLE_STREAK {
            t.sample_at(2000 * MIB, 400 * MIB, 0, start);
        }
        let d = t
            .sample_at(9000 * MIB, 400 * MIB, 0, start)
            .expect("the baseline is captured");
        assert_eq!(d.verdict, DriftVerdict::Settled);
        assert_eq!(d.outside_heap_growth_bytes, (7000 * MIB) as i64);
    }

    #[test]
    fn every_verdict_has_a_label() {
        for v in [
            DriftVerdict::Settled,
            DriftVerdict::Heap,
            DriftVerdict::OutsideHeap,
            DriftVerdict::Both,
        ] {
            assert!(!v.label().is_empty());
        }
    }
}
