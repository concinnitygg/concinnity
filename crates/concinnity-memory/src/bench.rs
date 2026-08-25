// concinnity-memory/src/bench.rs
//
// Measures the allocation layer: the frame arena against the heap path it
// replaces, pool churn, the inline-vec single-element case, the tracking
// wrapper's per-allocation price, and the tagged ledger's report cost.
//
// In-crate rather than under a bench target because `Arena::alloc_slice` and
// the ledger's per-realm readout have no consumer outside this crate, and
// widening them for a benchmark would be the wrong trade. The counters are
// process-global, so `--test-threads=1` is required rather than tidy.
//
//     cargo test -p concinnity-memory --release -- --ignored --nocapture \
//         --test-threads=1 bench_allocation_layer

use std::alloc::{GlobalAlloc, Layout, System};
use std::println;
use std::time::Instant;
use std::vec;
use std::vec::Vec;

use crate::{Arena, InlineVec, MemTag, Pool, Realm, TrackingAlloc};

const FRAME_ITEMS: usize = 4_096;
const POOL_CHURN: usize = 1_024;
const LEDGER_PAIRS: usize = 1_024;

// One measured pass runs at least this long before its time is trusted.
const TARGET_NS: u128 = 200_000_000;
const MAX_ITERS: u64 = 1 << 20;

// Time `body` over a calibrated iteration count and report its per-item cost.
fn bench<R>(name: &str, items: u64, mut body: impl FnMut() -> R) {
    let mut iters: u64 = 1;
    loop {
        let start = Instant::now();
        for _ in 0..iters {
            core::hint::black_box(body());
        }
        if start.elapsed().as_nanos() >= TARGET_NS || iters >= MAX_ITERS {
            break;
        }
        iters = iters.saturating_mul(4).min(MAX_ITERS);
    }

    let start = Instant::now();
    for _ in 0..iters {
        core::hint::black_box(body());
    }
    let elapsed = start.elapsed();
    let per_item_ns = elapsed.as_secs_f64() * 1e9 / (iters * items.max(1)) as f64;
    println!("  {name:<40} {per_item_ns:>10.2} ns/item");
}

#[test]
#[ignore = "benchmark; run with --ignored --test-threads=1"]
fn bench_allocation_layer() {
    {
        let mut arena = Arena::with_capacity(64 * 1024);
        bench(
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

    bench("memory/heap_frame_vec/4k", FRAME_ITEMS as u64, || {
        let v: Vec<u64> = (0..FRAME_ITEMS as u64).collect();
        v.iter().sum::<u64>()
    });

    {
        let mut pool: Pool<[u64; 4]> = Pool::with_capacity(POOL_CHURN);
        let mut handles = Vec::with_capacity(POOL_CHURN);
        bench("memory/pool_churn/1k", POOL_CHURN as u64, move || {
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
    bench("memory/inline_vec_single/1", 1, || {
        let mut v: InlineVec<u64> = InlineVec::default();
        v.push(7);
        core::hint::black_box(&v);
        v.as_slice()[0]
    });

    bench("memory/vec_single/1", 1, || {
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
        bench("memory/alloc_tracked/64B", 1, || {
            // SAFETY: the layout has a non-zero size, and the block is freed
            // through the same allocator and layout it was allocated with.
            unsafe {
                let ptr = core::hint::black_box(tracked.alloc(layout));
                assert!(!ptr.is_null(), "system allocator returned null");
                tracked.dealloc(ptr, layout);
            }
        });

        bench("memory/alloc_untracked/64B", 1, || {
            // SAFETY: as above, against the system allocator directly.
            unsafe {
                let ptr = core::hint::black_box(System.alloc(layout));
                assert!(!ptr.is_null(), "system allocator returned null");
                System.dealloc(ptr, layout);
            }
        });
    }

    {
        let ledger = crate::ledger();
        bench("memory/ledger_report_pair/1k", LEDGER_PAIRS as u64, || {
            for _ in 0..LEDGER_PAIRS {
                ledger.add(MemTag::Meshes, Realm::Device, 4_096);
                ledger.release(MemTag::Meshes, Realm::Device, 4_096);
            }
        });
    }
}
