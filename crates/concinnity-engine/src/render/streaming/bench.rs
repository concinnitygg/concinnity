// Measures the streaming planner's per-frame re-rank under pool pressure in
// the stress-world shape: twice as many items as the residency cap, scores
// tracking a moving hotspot so the planner keeps loading toward it and
// evicting behind it every frame.
//
//     cargo test -p concinnity-engine --release -- --ignored --nocapture \
//         --test-threads=1 bench_stream_plan

use crate::render::streaming::StreamPlanner;
use crate::test_support::{Pace, bench};

const STREAM_ITEMS: usize = 4_096;

// One streaming frame: re-score every item against the hotspot, plan, and
// complete the scheduled loads. Returns the number of load/evict decisions.
fn stream_frame(planner: &mut StreamPlanner, center: usize, frame: u64) -> usize {
    for id in 0..STREAM_ITEMS {
        let direct = id.abs_diff(center);
        planner.set_score(id, direct.min(STREAM_ITEMS - direct) as f32);
    }
    let plan = planner.plan();
    for &id in &plan.to_load {
        planner.mark_resident(id, frame, 1);
    }
    plan.to_load.len() + plan.to_evict.len()
}

fn run(pace: Pace) {
    let mut planner = StreamPlanner::new(STREAM_ITEMS, 4, STREAM_ITEMS / 2);
    for id in 0..STREAM_ITEMS / 2 {
        planner.mark_resident(id, 0, 1);
    }
    let mut frame = 1u64;
    let mut center = 0usize;
    bench(pace, "render/stream_plan/4k", STREAM_ITEMS as u64, || {
        center = (center + 97) % STREAM_ITEMS;
        frame += 1;
        stream_frame(&mut planner, center, frame)
    });
    let churn = stream_frame(&mut planner, (center + 97) % STREAM_ITEMS, frame + 1);
    assert!(churn > 0, "stream fixture stopped churning while measured");
}

#[test]
#[ignore = "benchmark; run with --ignored --test-threads=1"]
fn bench_stream_plan() {
    run(Pace::Timed);
}

#[test]
fn stream_plan_fixtures_build_and_run() {
    run(Pace::Once);
}
