// Measures the allocation layer: the frame arena against the heap path it
// replaces, pool churn, the inline-vec single-element case, the tracking
// wrapper's per-allocation price, and the tagged ledger's report cost.
//
// In-crate rather than under a bench target because `Arena::alloc_slice` and
// the ledger's per-realm readout have no consumer outside this crate, and
// widening them for a benchmark would be the wrong trade. The counters are
// process-global, so `--test-threads=1` is required rather than tidy.
//
//     cargo test -p concinnity-core --release -- --ignored --nocapture \
//         --test-threads=1 memory::bench

use std::alloc::{GlobalAlloc, Layout, System};
use std::vec;
use std::vec::Vec;

use crate::memory::{Arena, InlineVec, MemTag, Pool, Realm, TrackingAlloc};
use crate::test_support::{Pace, bench};

const FRAME_ITEMS: usize = 4_096;
const POOL_CHURN: usize = 1_024;
const LEDGER_PAIRS: usize = 1_024;

fn run(pace: Pace) {
    {
        let mut arena = Arena::with_capacity(64 * 1024);
        bench(
            pace,
            "memory/arena_frame_slice/4k",
            FRAME_ITEMS as u64,
            move || {
                arena.reset();
                let slab = arena
                    .alloc_slice(FRAME_ITEMS, 0u64)
                    .expect("arena sized for the frame");
                for (i, v) in slab.iter_mut().enumerate() {
                    *v = i as u64;
                }
                slab.iter().sum::<u64>()
            },
        );
    }

    bench(pace, "memory/heap_frame_vec/4k", FRAME_ITEMS as u64, || {
        let v: Vec<u64> = (0..FRAME_ITEMS as u64).collect();
        v.iter().sum::<u64>()
    });

    {
        let mut pool: Pool<[u64; 4]> = Pool::with_capacity(POOL_CHURN);
        let mut handles = Vec::with_capacity(POOL_CHURN);
        bench(pace, "memory/pool_churn/1k", POOL_CHURN as u64, move || {
            for i in 0..POOL_CHURN {
                handles.push(
                    pool.insert([i as u64; 4])
                        .expect("pool sized for the churn"),
                );
            }
            for handle in handles.drain(..) {
                pool.remove(handle);
            }
            pool.len()
        });
    }

    // black_box makes each container's address escape, so the optimizer cannot
    // elide the heap allocation this pair of benchmarks exists to compare.
    bench(pace, "memory/inline_vec_single/1", 1, || {
        let mut v: InlineVec<u64> = InlineVec::default();
        v.push(7);
        core::hint::black_box(&v);
        v.as_slice()[0]
    });

    bench(pace, "memory/vec_single/1", 1, || {
        let v: Vec<u64> = vec![7];
        core::hint::black_box(&v);
        v[0]
    });

    // What the counters cost the binaries that run on them: the same system
    // allocator with and without the tracking wrapper in front of it. This is
    // the per-allocation price of the memory instrumentation, and the delta
    // between these two rows is the whole of it.
    {
        let tracked = TrackingAlloc::new(System);
        let layout = Layout::from_size_align(64, 16).expect("valid layout");

        // black_box on the pointer is load-bearing: without it the untracked
        // pair has no observable effect and LLVM deletes it outright, while
        // the tracked pair survives on its counter writes -- which would
        // compare real work against nothing.
        bench(pace, "memory/alloc_tracked/64B", 1, || {
            // SAFETY: the layout has a non-zero size, and the block is freed
            // through the same allocator and layout it was allocated with.
            unsafe {
                let ptr = core::hint::black_box(tracked.alloc(layout));
                assert!(!ptr.is_null(), "system allocator returned null");
                tracked.dealloc(ptr, layout);
            }
        });

        bench(pace, "memory/alloc_untracked/64B", 1, || {
            // SAFETY: as above, against the system allocator directly.
            unsafe {
                let ptr = core::hint::black_box(System.alloc(layout));
                assert!(!ptr.is_null(), "system allocator returned null");
                System.dealloc(ptr, layout);
            }
        });
    }

    {
        let ledger = crate::memory::ledger();
        bench(
            pace,
            "memory/ledger_report_pair/1k",
            LEDGER_PAIRS as u64,
            || {
                for _ in 0..LEDGER_PAIRS {
                    ledger.add(MemTag::Meshes, Realm::Device, 4_096);
                    ledger.release(MemTag::Meshes, Realm::Device, 4_096);
                }
            },
        );
    }
}

#[test]
#[ignore = "benchmark; run with --ignored --test-threads=1"]
fn bench_allocation_layer() {
    run(Pace::Timed);
}

#[test]
fn allocation_layer_fixtures_build_and_run() {
    run(Pace::Once);
}
