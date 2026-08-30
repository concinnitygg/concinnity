// Where the churn is: live blocks and lifetime allocation counts per size
// class, behind the `detail` cargo feature.
//
// This is the tier a shipped build does not pay for. The global counters and
// the tagged ledger both stay on everywhere -- a console budget is fixed and the
// engine's own streaming already makes residency decisions from byte budgets --
// but a per-allocation histogram is a development instrument, so the feature is
// off unless a dev binary asks for it.
//
// Size classes rather than call sites: a class is one shift on the allocation
// path, where a call site would mean capturing a stack, and a class already
// answers the questions that matter at a glance. A live count that climbs and
// never comes back down is a leak in that class; a lifetime count far above the
// live one is churn.
//
// The reading side is always present and reports `None` when the feature is
// off, so nothing downstream has to be compiled twice.

// Class `c` above zero holds allocations of `2^(c-1) .. 2^c - 1` bytes. The top
// class is a catch-all, so a 64-bit size cannot run off the end of the table.
pub(crate) const CLASS_COUNT: usize = 33;

/// One size class as read at a moment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SizeClass {
    /// Smallest allocation size in this class, in bytes.
    pub min_bytes: u64,
    /// Inclusive. `u64::MAX` in the catch-all top class.
    pub max_bytes: u64,
    /// Allocations of this size made since process start.
    pub allocs: u64,
    /// Blocks of this size allocated and not yet freed.
    pub live_blocks: u64,
}

// Every size class, read in one pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SizeClasses {
    classes: [SizeClass; CLASS_COUNT],
}

impl SizeClasses {
    pub fn classes(&self) -> &[SizeClass; CLASS_COUNT] {
        &self.classes
    }

    // The class holding the most live blocks: where the heap's population sits,
    // and where a leak shows first. `None` when nothing is live.
    pub fn busiest(&self) -> Option<SizeClass> {
        self.classes
            .iter()
            .copied()
            .filter(|c| c.live_blocks > 0)
            .max_by_key(|c| c.live_blocks)
    }

    pub fn live_blocks(&self) -> u64 {
        self.classes.iter().map(|c| c.live_blocks).sum()
    }
}

// The class an allocation of `size` falls in.
#[cfg(any(feature = "detail", test))]
const fn class_of(size: usize) -> usize {
    let bits = (usize::BITS - size.leading_zeros()) as usize;
    if bits >= CLASS_COUNT {
        CLASS_COUNT - 1
    } else {
        bits
    }
}

// The byte range class `class` covers, inclusive.
#[cfg(any(feature = "detail", test))]
const fn class_bounds(class: usize) -> (u64, u64) {
    match class {
        0 => (0, 0),
        c if c == CLASS_COUNT - 1 => (1 << (CLASS_COUNT - 2), u64::MAX),
        c => (1 << (c - 1), (1 << c) - 1),
    }
}

#[cfg(not(feature = "detail"))]
mod imp {
    pub(crate) fn record_alloc(_size: usize) {}
    pub(crate) fn record_free(_size: usize) {}
    pub(crate) fn record_realloc(_old_size: usize, _new_size: usize) {}
    pub(crate) fn snapshot() -> Option<super::SizeClasses> {
        None
    }
}

#[cfg(feature = "detail")]
mod imp {
    use core::sync::atomic::{AtomicU64, Ordering::Relaxed};

    use super::{CLASS_COUNT, SizeClass, SizeClasses, class_bounds, class_of};

    struct Class {
        allocs: AtomicU64,
        live: AtomicU64,
    }

    impl Class {
        const fn new() -> Self {
            Self {
                allocs: AtomicU64::new(0),
                live: AtomicU64::new(0),
            }
        }
    }

    // The histogram behind the global allocator. Split out from the free
    // functions so the bookkeeping is testable on its own instance: the
    // process-global one is fed by every thread at once, so a delta read across
    // it is a race rather than a measurement.
    pub(crate) struct Histogram {
        classes: [Class; CLASS_COUNT],
    }

    impl Histogram {
        pub(crate) const fn new() -> Self {
            Self {
                classes: [const { Class::new() }; CLASS_COUNT],
            }
        }

        pub(crate) fn record_alloc(&self, size: usize) {
            let class = &self.classes[class_of(size)];
            class.allocs.fetch_add(1, Relaxed);
            class.live.fetch_add(1, Relaxed);
        }

        pub(crate) fn record_free(&self, size: usize) {
            let live = &self.classes[class_of(size)].live;
            let _ = live.fetch_update(Relaxed, Relaxed, |n| Some(n.saturating_sub(1)));
        }

        // A resize moves a block between classes without being an allocation of
        // its own, matching how the global counters treat it.
        pub(crate) fn record_realloc(&self, old_size: usize, new_size: usize) {
            let (old, new) = (class_of(old_size), class_of(new_size));
            if old != new {
                self.record_free(old_size);
                self.classes[new].live.fetch_add(1, Relaxed);
            }
        }

        pub(crate) fn snapshot(&self) -> SizeClasses {
            SizeClasses {
                classes: core::array::from_fn(|i| {
                    let (min_bytes, max_bytes) = class_bounds(i);
                    SizeClass {
                        min_bytes,
                        max_bytes,
                        allocs: self.classes[i].allocs.load(Relaxed),
                        live_blocks: self.classes[i].live.load(Relaxed),
                    }
                }),
            }
        }
    }

    static CLASSES: Histogram = Histogram::new();

    pub(crate) fn record_alloc(size: usize) {
        CLASSES.record_alloc(size);
    }

    pub(crate) fn record_free(size: usize) {
        CLASSES.record_free(size);
    }

    pub(crate) fn record_realloc(old_size: usize, new_size: usize) {
        CLASSES.record_realloc(old_size, new_size);
    }

    pub(crate) fn snapshot() -> Option<SizeClasses> {
        Some(CLASSES.snapshot())
    }
}

pub(crate) use imp::{record_alloc, record_free, record_realloc};

/// The heap's size-class histogram, or `None` when the crate was built without
/// the `detail` feature.
pub fn size_classes() -> Option<SizeClasses> {
    imp::snapshot()
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "detail")]
    use super::imp::Histogram;
    use super::*;

    #[test]
    fn classes_partition_the_size_range_without_gaps_or_overlap() {
        let (mut prev_min, mut prev_max) = class_bounds(0);
        assert_eq!((prev_min, prev_max), (0, 0));
        for class in 1..CLASS_COUNT {
            let (min, max) = class_bounds(class);
            assert_eq!(min, prev_max + 1, "class {class} does not follow the last");
            assert!(max >= min);
            (prev_min, prev_max) = (min, max);
        }
        assert!(prev_min > 0);
        assert_eq!(prev_max, u64::MAX, "the top class must catch every size");
    }

    #[test]
    fn a_size_lands_in_the_class_that_covers_it() {
        for size in [0usize, 1, 2, 3, 4, 7, 8, 64, 1023, 1024, 1 << 20] {
            let class = class_of(size);
            let (min, max) = class_bounds(class);
            assert!(
                (min..=max).contains(&(size as u64)),
                "{size} landed outside class {class} ({min}..={max})"
            );
        }
    }

    #[test]
    fn an_enormous_size_lands_in_the_catch_all_class() {
        assert_eq!(class_of(usize::MAX), CLASS_COUNT - 1);
        assert_eq!(class_of(1 << 40), CLASS_COUNT - 1);
    }

    // Without the feature the instrument is absent, not zeroed: a readout must
    // be able to tell "not measured" from "measured nothing".
    #[cfg(not(feature = "detail"))]
    #[test]
    fn the_histogram_is_absent_without_the_feature() {
        record_alloc(64);
        assert_eq!(size_classes(), None);
    }

    #[cfg(feature = "detail")]
    #[test]
    fn allocations_and_frees_move_their_class() {
        const SIZE: usize = 1 << 17;
        let class = class_of(SIZE);
        let histogram = Histogram::new();

        histogram.record_alloc(SIZE);
        let during = histogram.snapshot().classes[class];
        assert_eq!(during.allocs, 1);
        assert_eq!(during.live_blocks, 1);

        histogram.record_free(SIZE);
        let after = histogram.snapshot().classes[class];
        assert_eq!(after.live_blocks, 0);
        assert_eq!(after.allocs, 1, "a free is not an alloc");
    }

    // A free of a class that never allocated cannot take the live count below
    // zero: the allocator sees frees of blocks older than the histogram.
    #[cfg(feature = "detail")]
    #[test]
    fn a_free_without_a_matching_alloc_leaves_the_class_empty() {
        const SIZE: usize = 1 << 11;
        let histogram = Histogram::new();

        histogram.record_free(SIZE);

        let class = histogram.snapshot().classes[class_of(SIZE)];
        assert_eq!(class.live_blocks, 0);
        assert_eq!(class.allocs, 0);
    }

    // A resize moves the block's live count between classes and counts as no
    // new allocation.
    #[cfg(feature = "detail")]
    #[test]
    fn a_resize_moves_the_block_between_classes() {
        const SMALL: usize = 1 << 13;
        const LARGE: usize = 1 << 19;
        let (small, large) = (class_of(SMALL), class_of(LARGE));
        let histogram = Histogram::new();

        histogram.record_alloc(SMALL);
        histogram.record_realloc(SMALL, LARGE);
        let after = histogram.snapshot();

        assert_eq!(after.classes[small].live_blocks, 0);
        assert_eq!(after.classes[large].live_blocks, 1);
        assert_eq!(after.classes[large].allocs, 0, "a resize is not an alloc");
        assert_eq!(after.classes[small].allocs, 1);
    }

    // A resize inside one class is not a move at all.
    #[cfg(feature = "detail")]
    #[test]
    fn a_resize_within_a_class_leaves_the_counts_alone() {
        const SMALL: usize = 1 << 13;
        const LARGER: usize = (1 << 13) + 64;
        let class = class_of(SMALL);
        let histogram = Histogram::new();

        histogram.record_alloc(SMALL);
        histogram.record_realloc(SMALL, LARGER);

        let after = histogram.snapshot().classes[class];
        assert_eq!(after.live_blocks, 1);
        assert_eq!(after.allocs, 1);
    }

    #[cfg(feature = "detail")]
    #[test]
    fn the_busiest_class_is_the_one_holding_the_most_live_blocks() {
        const BUSY: usize = 1 << 9;
        const QUIET: usize = 1 << 3;
        let histogram = Histogram::new();

        histogram.record_alloc(QUIET);
        for _ in 0..3 {
            histogram.record_alloc(BUSY);
        }

        let snapshot = histogram.snapshot();
        let busiest = snapshot.busiest().expect("blocks are live");
        assert_eq!(busiest.min_bytes, class_bounds(class_of(BUSY)).0);
        assert_eq!(busiest.live_blocks, 3);
        assert_eq!(snapshot.live_blocks(), 4);
    }

    #[cfg(feature = "detail")]
    #[test]
    fn an_empty_histogram_has_no_busiest_class() {
        let histogram = Histogram::new();
        assert_eq!(histogram.snapshot().busiest(), None);
        assert_eq!(histogram.snapshot().live_blocks(), 0);
    }

    // The global histogram is the one the allocator feeds, and every other test
    // thread feeds it too, so this asserts only what stays true under that
    // traffic: a recorded allocation reaches it, and the lifetime count only
    // ever climbs.
    #[cfg(feature = "detail")]
    #[test]
    fn the_global_histogram_takes_what_the_allocator_records() {
        const SIZE: usize = 1 << 17;
        let class = class_of(SIZE);
        let before = size_classes().expect("the feature is on").classes[class];

        record_alloc(SIZE);
        let after = size_classes().expect("the feature is on").classes[class];
        assert!(after.allocs > before.allocs);

        record_free(SIZE);
    }

    // The whole partition is readable as one array, which is what a readout
    // walks: every class in ascending order, covering the range without a gap.
    // Only a build carrying the histogram has one to read.
    #[cfg(feature = "detail")]
    #[test]
    fn the_classes_are_readable_as_one_ascending_array() {
        let snapshot = size_classes().expect("this build tracks size classes");
        let classes = snapshot.classes();
        assert_eq!(classes.len(), CLASS_COUNT);
        for pair in classes.windows(2) {
            assert!(
                pair[0].max_bytes < pair[1].min_bytes || pair[0].max_bytes + 1 == pair[1].min_bytes,
                "{:?} then {:?} do not meet",
                pair[0],
                pair[1]
            );
        }
        assert_eq!(classes[CLASS_COUNT - 1].max_bytes, u64::MAX);
    }
}
