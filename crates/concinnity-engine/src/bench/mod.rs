// src/bench/mod.rs
//
// In-crate microbenchmarks for engine internals, the fixture they share, and
// the per-frame allocation pins (`alloc_budget`, which DO run un-ignored).
//
// These live inside the crate rather than in `benches/` because what is worth
// measuring here is `pub(crate)`: the per-frame transform propagation and the
// behavior tick are not part of the engine's public surface, and widening them
// for a benchmark would be the wrong trade. Ignored by default, so a normal
// test run never pays for them. `--test-threads=1` is required rather than
// tidy: the allocation counters are process-global, so a benchmark running
// beside another reads the other's allocations as its own:
//
//     cargo test -p concinnity-engine --release -- --ignored --nocapture \
//         --test-threads=1 bench
//
// The timing helper is deliberately not `concinnity-bench`: that crate installs
// a global allocator and depends on this one, so importing it here would both
// collide with this test binary's allocator and close a dependency cycle. The
// numbers come from the same instruments either way.

pub(crate) mod alloc_budget;
pub(crate) mod extraction;
pub(crate) mod transforms;

use std::time::Instant;

use crate::blob::BlobData;
use crate::ecs::{Arena, ComponentStorage, FrameContext, PipelineContext, Resources};
use crate::gfx::profile::FrameProfile;

// One measured pass runs at least this long before its time is trusted.
const TARGET_NS: u128 = 200_000_000;
const MAX_ITERS: u64 = 1 << 20;

// The hand-assembled world the benches drive, mirroring `behavior::tests`'
// fixture: a component storage plus the pieces a `PipelineContext` borrows.
pub(crate) struct BenchWorld {
    pub components: ComponentStorage,
    blob: BlobData,
    profile: FrameProfile,
    resources: Resources,
    scratch: Arena,
}

impl BenchWorld {
    pub(crate) fn new() -> BenchWorld {
        BenchWorld {
            components: ComponentStorage::default(),
            blob: BlobData::empty(),
            profile: FrameProfile::default(),
            resources: Resources::default(),
            scratch: Arena::with_capacity(1 << 20),
        }
    }

    pub(crate) fn ctx(&mut self) -> PipelineContext<'_> {
        PipelineContext {
            components: &mut self.components,
            blob: &mut self.blob,
            profile: &mut self.profile,
            resources: &mut self.resources,
            frame: FrameContext::new(&self.scratch),
        }
    }
}

// Time `body` over a calibrated iteration count and report its per-item cost
// beside the allocations one item causes. `items` is how many units of work one
// call performs, so a number is comparable across fixture sizes.
pub(crate) fn bench<R>(name: &str, items: u64, mut body: impl FnMut() -> R) {
    let mut iters: u64 = 1;
    loop {
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(body());
        }
        if start.elapsed().as_nanos() >= TARGET_NS || iters >= MAX_ITERS {
            break;
        }
        iters = iters.saturating_mul(4).min(MAX_ITERS);
    }

    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(body());
    }
    let elapsed = start.elapsed();

    // A separate pass for allocations, so the counter reads sit outside the
    // timed window. The counters are process-global, so this figure is only
    // this benchmark's when nothing else is running (see the module doc).
    let before = concinnity_memory::stats().expect("the test binary tracks its heap");
    for _ in 0..iters {
        std::hint::black_box(body());
    }
    let after = concinnity_memory::stats().expect("the allocator stays installed");

    let units = (iters * items.max(1)) as f64;
    let per_item_ns = elapsed.as_secs_f64() * 1e9 / units;
    let allocs = (after.alloc_count - before.alloc_count) as f64 / units;
    println!("  {name:<40} {per_item_ns:>10.2} ns/item {allocs:>10.3} allocs/item");
}
