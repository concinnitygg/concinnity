//! Benchmarks over the GPU-free render-prep layer: light packing for the
//! clustered forward pass, the streaming planner's per-frame re-rank under
//! pool pressure, and draw-slot recycling. Everything here runs on the CPU
//! side of the render / device split; no backend is involved.
//!
//! The BVH build and frustum query are benchmarked inside `core::render`,
//! where the item type they need lives.

use crate::support::Bench;
use concinnity_core::components::{PointLight, RectAreaLight, SpotLight};
use concinnity_core::render::draw_slot::{DrawSlotAllocator, SlotAlloc};
use concinnity_core::render::lights::build_light_data;
use concinnity_core::render::streaming::StreamPlanner;

const OBJECTS: usize = 10_000;
const STREAM_ITEMS: usize = 4_096;
const SLOT_CHURN: usize = 1_024;

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

pub(crate) fn benches(bench: &mut Bench) {
    {
        let grid = |i: usize| {
            [
                (i % 32) as f32 * 4.0 - 64.0,
                2.5,
                (i / 32) as f32 * 4.0 - 64.0,
            ]
        };
        let points: Vec<PointLight> = (0..700)
            .map(|i| PointLight {
                position: grid(i),
                ..Default::default()
            })
            .collect();
        let spots: Vec<SpotLight> = (0..200)
            .map(|i| SpotLight {
                position: grid(i + 700),
                cast_shadows: i % 4 == 0,
                ..Default::default()
            })
            .collect();
        let rects: Vec<RectAreaLight> = (0..100)
            .map(|i| RectAreaLight {
                centre: grid(i + 900),
                ..Default::default()
            })
            .collect();
        bench.run("render/light_pack/1k", 1_000, || {
            build_light_data(&points, &spots, &rects).lights.len()
        });
    }

    // Pool pressure in the stress-world shape: twice as many items as the
    // residency cap, scores tracking a moving hotspot so the planner keeps
    // loading toward it and evicting behind it every frame.
    {
        let mut planner = StreamPlanner::new(STREAM_ITEMS, 4, STREAM_ITEMS / 2);
        for id in 0..STREAM_ITEMS / 2 {
            planner.mark_resident(id, 0, 1);
        }
        let mut frame = 1u64;
        let mut center = 0usize;
        bench.run("render/stream_plan/4k", STREAM_ITEMS as u64, || {
            center = (center + 97) % STREAM_ITEMS;
            frame += 1;
            stream_frame(&mut planner, center, frame)
        });
        let churn = stream_frame(&mut planner, (center + 97) % STREAM_ITEMS, frame + 1);
        assert!(churn > 0, "stream fixture stopped churning while measured");
    }

    {
        let mut slots = DrawSlotAllocator::with_len(OBJECTS);
        bench.run("render/draw_slot_churn/1k", SLOT_CHURN as u64, || {
            for slot in 0..SLOT_CHURN {
                slots.free(slot * 7 % OBJECTS);
            }
            let mut acc = 0usize;
            for _ in 0..SLOT_CHURN {
                acc += match slots.allocate() {
                    SlotAlloc::Reuse(slot) => slot,
                    SlotAlloc::Append(slot) => slot,
                };
            }
            acc
        });
    }
}
