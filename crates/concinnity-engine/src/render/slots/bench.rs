// Measures draw-slot recycling: freeing a scattered run of slots out of a
// full table and allocating the same count back.
//
//     cargo test -p concinnity-engine --release -- --ignored --nocapture \
//         --test-threads=1 bench_draw_slot_churn

use crate::gfx::render_types::DrawIndex;
use crate::render::slots::draw_slot::DrawSlotAllocator;
use crate::test_support::{Pace, bench};

const OBJECTS: usize = 10_000;
const SLOT_CHURN: usize = 1_024;

fn run(pace: Pace) {
    let mut slots = DrawSlotAllocator::with_len(OBJECTS);
    bench(pace, "render/draw_slot_churn/1k", SLOT_CHURN as u64, || {
        for slot in 0..SLOT_CHURN {
            slots.free(DrawIndex::from_usize(slot * 7 % OBJECTS));
        }
        let mut acc = 0usize;
        for _ in 0..SLOT_CHURN {
            acc += slots.allocate().slot().index();
        }
        acc
    });
}

#[test]
#[ignore = "benchmark; run with --ignored --test-threads=1"]
fn bench_draw_slot_churn() {
    run(Pace::Timed);
}

#[test]
fn draw_slot_churn_fixtures_build_and_run() {
    run(Pace::Once);
}
