// benches/memory.rs
//
// Benchmarks over the engine's allocation layer: the frame arena against the
// heap path it replaces, pool churn, the inline-vec single-element case, and
// the tagged ledger's report cost. The ledger benchmark reports into the
// device realm, which also exercises the harness's vram accounting end to end.
//
// Run with `cargo bench -p concinnity-bench --bench memory`.

use std::alloc::{GlobalAlloc, Layout, System};

use concinnity_bench::Bench;
use concinnity_memory::{Arena, InlineVec, MemTag, Pool, Realm, TrackingAlloc};

const FRAME_ITEMS: usize = 4_096;
const POOL_CHURN: usize = 1_024;
const LEDGER_PAIRS: usize = 1_024;

fn main() {
    let mut bench = Bench::from_env();

    {
        let mut arena = Arena::with_capacity(64 * 1024);
        bench.run(
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

    bench.run("memory/heap_frame_vec/4k", FRAME_ITEMS as u64, || {
        let v: Vec<u64> = (0..FRAME_ITEMS as u64).collect();
        v.iter().sum::<u64>()
    });

    {
        let mut pool: Pool<[u64; 4]> = Pool::with_capacity(POOL_CHURN);
        let mut handles = Vec::with_capacity(POOL_CHURN);
        bench.run("memory/pool_churn/1k", POOL_CHURN as u64, move || {
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
    bench.run("memory/inline_vec_single/1", 1, || {
        let mut v: InlineVec<u64> = InlineVec::default();
        v.push(7);
        core::hint::black_box(&v);
        v.as_slice()[0]
    });

    bench.run("memory/vec_single/1", 1, || {
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
        bench.run("memory/alloc_tracked/64B", 1, || {
            // SAFETY: the layout has a non-zero size, and the block is freed
            // through the same allocator and layout it was allocated with.
            unsafe {
                let ptr = core::hint::black_box(tracked.alloc(layout));
                assert!(!ptr.is_null(), "system allocator returned null");
                tracked.dealloc(ptr, layout);
            }
        });

        bench.run("memory/alloc_untracked/64B", 1, || {
            // SAFETY: as above, against the system allocator directly.
            unsafe {
                let ptr = core::hint::black_box(System.alloc(layout));
                assert!(!ptr.is_null(), "system allocator returned null");
                System.dealloc(ptr, layout);
            }
        });
    }

    {
        let ledger = concinnity_memory::ledger();
        bench.run("memory/ledger_report_pair/1k", LEDGER_PAIRS as u64, || {
            for _ in 0..LEDGER_PAIRS {
                ledger.add(MemTag::Meshes, Realm::Device, 4_096);
                ledger.release(MemTag::Meshes, Realm::Device, 4_096);
            }
        });
    }

    bench.finish();
}
