// The tracked heap's global counters, sharded so concurrent allocation does not
// serialize on one cache line.
//
// A job system has every worker allocating at once, and a single atomic block
// turns each allocation into a contended read-modify-write on the same line. The
// counters are split across cache-line-sized shards picked by the calling
// thread, and summed only when something reads them.
//
// Threads are told apart by a stack address: every thread has its own stack, and
// reading one costs nothing. Thread-local storage would say it directly but is
// not available in a `no_std` crate, and is unwelcome underneath a global
// allocator in any case -- its lazy initialization allocates, which re-enters
// the allocator. The mapping is approximate: a deep call stack can land a thread
// in a neighbouring shard, and a block allocated on one thread is often freed on
// another. `live` is therefore a wrapping counter, which leaves the individual
// shards free to disagree and their sum exact -- and the sum is the only figure
// anyone reads.

use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};

// Enough shards to keep a workstation's workers off each other's lines without
// making a read walk a large table.
const SHARDS: usize = 16;
const SHARD_BITS: u32 = SHARDS.trailing_zeros();

// Stack addresses separate at 64 KiB granularity: coarse enough that a thread
// keeps its shard as its stack grows and shrinks, fine enough that threads whose
// stacks sit close together still land apart.
const STACK_SHIFT: u32 = 16;

// How often an allocating thread re-sums the shards to refresh the peak.
// Amortized over a thousand allocations the walk costs nothing, and a rise built
// from small allocations always carries enough of them to be sampled.
const PEAK_SAMPLE_ALLOCS: usize = 1024;

// A single allocation this large moves the total by itself, so it refreshes the
// peak on the spot rather than waiting for the count to come round.
const PEAK_SAMPLE_BYTES: usize = 1 << 20;

/// A snapshot of the tracked heap. Counters are relaxed, so the fields are
/// individually accurate but need not agree with each other to the byte under
/// concurrent allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemStats {
    /// Bytes currently allocated and not yet freed.
    pub live_bytes: u64,
    /// High-water mark of `live_bytes`, sampled as the shards are re-summed.
    pub peak_bytes: u64,
    /// Allocations made since process start. With `free_count`, this is the
    /// churn rate; a reallocation resizes an existing block and counts as
    /// neither.
    pub alloc_count: u64,
    /// Frees made since process start.
    pub free_count: u64,
}

// One shard's counters, padded to a cache line so a thread updating its own
// shard never invalidates another's.
#[repr(align(64))]
struct Shard {
    live: AtomicUsize,
    allocs: AtomicUsize,
    frees: AtomicUsize,
}

impl Shard {
    const fn new() -> Self {
        Self {
            live: AtomicUsize::new(0),
            allocs: AtomicUsize::new(0),
            frees: AtomicUsize::new(0),
        }
    }
}

// The counter block behind the global allocator. Split out from `TrackingAlloc`
// so the accounting is testable on its own instance: the allocator itself can
// only ever drive the one process-global block.
pub(crate) struct Counters {
    shards: [Shard; SHARDS],
    peak: AtomicUsize,
}

impl Counters {
    pub(crate) const fn new() -> Self {
        Self {
            shards: [const { Shard::new() }; SHARDS],
            peak: AtomicUsize::new(0),
        }
    }

    fn shard(&self) -> &Shard {
        &self.shards[current_shard()]
    }

    pub(crate) fn record_alloc(&self, size: usize) {
        let shard = self.shard();
        shard.live.fetch_add(size, Relaxed);
        let allocs = shard.allocs.fetch_add(1, Relaxed).wrapping_add(1);
        if size >= PEAK_SAMPLE_BYTES || allocs.is_multiple_of(PEAK_SAMPLE_ALLOCS) {
            self.refresh_peak();
        }
    }

    pub(crate) fn record_free(&self, size: usize) {
        let shard = self.shard();
        shard.live.fetch_sub(size, Relaxed);
        shard.frees.fetch_add(1, Relaxed);
    }

    // A resize moves `live` by the delta only: the block was already counted at
    // `old_size` and no alloc/free pair reaches the counters.
    pub(crate) fn record_realloc(&self, old_size: usize, new_size: usize) {
        let shard = self.shard();
        if new_size >= old_size {
            let grew = new_size - old_size;
            shard.live.fetch_add(grew, Relaxed);
            if grew >= PEAK_SAMPLE_BYTES {
                self.refresh_peak();
            }
        } else {
            shard.live.fetch_sub(old_size - new_size, Relaxed);
        }
    }

    // Live bytes across every shard. Wrapping, because a block freed on a
    // different thread than it was allocated on decrements a shard that never
    // counted it; only the total is meaningful.
    fn live(&self) -> usize {
        self.shards
            .iter()
            .fold(0usize, |sum, s| sum.wrapping_add(s.live.load(Relaxed)))
    }

    fn refresh_peak(&self) -> usize {
        let live = self.live();
        self.peak.fetch_max(live, Relaxed);
        live
    }

    // How many shards have counted anything. The claim a stack probe makes is
    // that real threads land apart; this is what lets a test check it.
    #[cfg(test)]
    fn touched_shards(&self) -> usize {
        self.shards
            .iter()
            .filter(|s| s.allocs.load(Relaxed) > 0)
            .count()
    }

    // Allocations counted by this block since process start, without the peak
    // refresh a full `snapshot` pays. Cheap enough to sample around every
    // system step; `None` under the same condition as `snapshot`.
    pub(crate) fn alloc_count(&self) -> Option<u64> {
        let count = self
            .shards
            .iter()
            .fold(0u64, |sum, s| sum + s.allocs.load(Relaxed) as u64);
        (count > 0).then_some(count)
    }

    // `None` until something allocates through this block, which for the global
    // block means "no binary installed the allocator".
    pub(crate) fn snapshot(&self) -> Option<MemStats> {
        let (alloc_count, free_count) = self.shards.iter().fold((0u64, 0u64), |(a, f), s| {
            (
                a + s.allocs.load(Relaxed) as u64,
                f + s.frees.load(Relaxed) as u64,
            )
        });
        if alloc_count == 0 {
            return None;
        }
        let live = self.refresh_peak();
        Some(MemStats {
            live_bytes: live as u64,
            peak_bytes: self.peak.load(Relaxed) as u64,
            alloc_count,
            free_count,
        })
    }
}

impl Default for Counters {
    fn default() -> Self {
        Self::new()
    }
}

// Which shard an address on the calling thread's stack belongs to.
//
// Hashed rather than masked. Thread stacks are laid out at a fixed stride, and
// a stride that is a multiple of the table size would put every thread on one
// shard -- which is the failure this whole file exists to avoid. Multiplying by
// the golden ratio and keeping the high bits spreads any stride.
const fn shard_of(stack_addr: usize) -> usize {
    const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
    let key = (stack_addr >> STACK_SHIFT) as u64;
    (key.wrapping_mul(GOLDEN) >> (u64::BITS - SHARD_BITS)) as usize
}

// The calling thread's shard. `black_box` keeps the probe a real stack slot
// rather than something the optimizer folds away, and the probe is
// uninitialized because only its address is ever read.
fn current_shard() -> usize {
    let probe = core::mem::MaybeUninit::<u8>::uninit();
    shard_of(core::hint::black_box(probe.as_ptr()) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    // A fresh block has never seen an allocation, which is how a readout tells
    // "not installed" from "installed and holding nothing".
    #[test]
    fn snapshot_is_none_until_something_allocates() {
        let counters = Counters::new();
        assert_eq!(counters.snapshot(), None);

        counters.record_alloc(64);
        assert!(counters.snapshot().is_some());
    }

    // The cheap count agrees with the full snapshot and shares its
    // None-until-installed probe, so a delta reader can rely on either.
    #[test]
    fn alloc_count_matches_the_snapshot() {
        let counters = Counters::new();
        assert_eq!(counters.alloc_count(), None);

        counters.record_alloc(64);
        counters.record_alloc(32);
        counters.record_free(64);
        let stats = counters.snapshot().expect("block has seen allocations");
        assert_eq!(counters.alloc_count(), Some(stats.alloc_count));
        assert_eq!(counters.alloc_count(), Some(2));
    }

    #[test]
    fn alloc_and_free_balance_back_to_zero() {
        let counters = Counters::new();
        counters.record_alloc(1024);
        counters.record_alloc(512);
        counters.record_free(1024);
        counters.record_free(512);

        let stats = counters.snapshot().expect("block has seen allocations");
        assert_eq!(stats.live_bytes, 0);
        assert_eq!(stats.alloc_count, 2);
        assert_eq!(stats.free_count, 2);
    }

    // Peak is a high-water mark: it holds the largest live total seen, not the
    // current one. Reading is one of the moments it is sampled, so a readout
    // never reports a peak below the live bytes beside it.
    #[test]
    fn peak_holds_the_high_water_mark() {
        let counters = Counters::new();
        counters.record_alloc(1000);
        counters.record_alloc(500);
        assert_eq!(counters.snapshot().unwrap().live_bytes, 1500);

        counters.record_free(1200);
        let stats = counters.snapshot().expect("block has seen allocations");
        assert_eq!(stats.live_bytes, 300);
        assert_eq!(stats.peak_bytes, 1500);
    }

    // A single large allocation refreshes the peak where it happens, so a block
    // taken and dropped between two reads is still seen.
    #[test]
    fn a_large_allocation_refreshes_the_peak_where_it_happens() {
        let counters = Counters::new();
        counters.record_alloc(PEAK_SAMPLE_BYTES);
        counters.record_free(PEAK_SAMPLE_BYTES);

        let stats = counters.snapshot().expect("block has seen allocations");
        assert_eq!(stats.live_bytes, 0);
        assert_eq!(stats.peak_bytes as usize, PEAK_SAMPLE_BYTES);
    }

    #[test]
    fn realloc_moves_live_bytes_by_the_delta_only() {
        let counters = Counters::new();
        counters.record_alloc(100);

        counters.record_realloc(100, 400);
        assert_eq!(counters.snapshot().unwrap().live_bytes, 400);

        counters.record_realloc(400, 250);
        let stats = counters.snapshot().unwrap();
        assert_eq!(stats.live_bytes, 250);
        assert_eq!(stats.peak_bytes, 400);
        // The resizes were neither allocations nor frees.
        assert_eq!(stats.alloc_count, 1);
        assert_eq!(stats.free_count, 0);
    }

    // Threads must spread across the table whatever stride the platform lays
    // their stacks out at. Perfect separation is not the claim -- shards are a
    // hash, and collisions only cost contention -- but collapsing a whole
    // process onto one shard is the failure this file exists to avoid, and a
    // masked index does exactly that at any stride that divides the table.
    #[test]
    fn threads_spread_across_the_table_at_every_plausible_stack_stride() {
        const KIB: usize = 1024;
        for stride in [64 * KIB, 512 * KIB, 1 << 20, 2 << 20, 8 << 20] {
            let shards: std::collections::BTreeSet<usize> = (0..SHARDS)
                .map(|t| shard_of(0x7000_0000_0000 + t * stride))
                .collect();
            assert!(
                shards.len() >= SHARDS / 2,
                "{SHARDS} stacks {stride} bytes apart used only {} of {SHARDS} shards",
                shards.len()
            );
        }
    }

    // A thread keeps its shard as its stack grows and shrinks, so its counters
    // stay on the line its own core already holds.
    #[test]
    fn one_stack_keeps_its_shard_as_it_grows() {
        let base = 0x7000_0000_0000usize;
        for depth in [0, 1, 64, 4096, (1 << STACK_SHIFT) - 1] {
            assert_eq!(shard_of(base), shard_of(base + depth));
        }
    }

    #[test]
    fn every_address_maps_into_the_shard_table() {
        for addr in [0usize, 1, usize::MAX, 0x7fff_ffff_ffff, 1 << 47] {
            assert!(shard_of(addr) < SHARDS);
        }
    }

    // The whole point of sharding: threads counting at once must still sum to
    // the exact total, including blocks freed on a thread other than the one
    // that allocated them.
    #[test]
    fn concurrent_threads_sum_to_the_exact_total() {
        use std::sync::Arc;
        use std::thread;

        const THREADS: usize = 8;
        const PER_THREAD: usize = 4096;
        const SIZE: usize = 128;

        let counters = Arc::new(Counters::new());
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let counters = Arc::clone(&counters);
                thread::spawn(move || {
                    for _ in 0..PER_THREAD {
                        counters.record_alloc(SIZE);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("counting thread");
        }

        let stats = counters.snapshot().expect("threads allocated");
        assert_eq!(stats.alloc_count as usize, THREADS * PER_THREAD);
        assert_eq!(stats.live_bytes as usize, THREADS * PER_THREAD * SIZE);
        assert!(
            counters.touched_shards() > 1,
            "every thread landed on one shard, which is the contention sharding removes"
        );

        // Free every block from threads that never allocated one, which is what
        // makes the per-shard counters disagree and the sum still hold.
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let counters = Arc::clone(&counters);
                thread::spawn(move || {
                    for _ in 0..PER_THREAD {
                        counters.record_free(SIZE);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("freeing thread");
        }

        let stats = counters.snapshot().expect("threads allocated");
        assert_eq!(stats.live_bytes, 0);
        assert_eq!(stats.free_count as usize, THREADS * PER_THREAD);
        assert_eq!(stats.peak_bytes as usize, THREADS * PER_THREAD * SIZE);
    }
}
