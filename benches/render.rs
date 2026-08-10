// benches/render.rs
//
// Benchmarks over the GPU-free render-prep layer: BVH build and frustum
// query (the visibility path), light packing for the clustered forward
// pass, the streaming planner's per-frame re-rank under pool pressure, and
// draw-slot recycling. Everything here runs on the CPU side of the
// render / device split; no backend is involved.
//
// Run with `cargo bench -p concinnity-bench --bench render`.

use concinnity_asset::{PointLight, RectAreaLight, SpotLight};
use concinnity_bench::Bench;
use concinnity_render::bvh::{Bvh, BvhItem};
use concinnity_render::draw_slot::{DrawSlotAllocator, SlotAlloc};
use concinnity_render::frustum::Frustum;
use concinnity_render::lights::build_light_data;
use concinnity_render::streaming::StreamPlanner;

const OBJECTS: usize = 10_000;
const STREAM_ITEMS: usize = 4_096;
const SLOT_CHURN: usize = 1_024;

// Unit boxes on a 100x100 ground grid straddling the camera plane, so the
// frustum query sees a real mix of accepted, rejected, and straddling nodes.
fn scene_items() -> Vec<BvhItem> {
    (0..OBJECTS)
        .map(|i| {
            let x = (i % 100) as f32 * 3.0 - 150.0;
            let z = (i / 100) as f32 * 6.0 - 300.0;
            BvhItem {
                bb_min: [x - 0.5, 0.0, z - 0.5],
                bb_max: [x + 0.5, 1.0, z + 0.5],
                cull_distance: 0.0,
                index: i as u32,
            }
        })
        .collect()
}

// Perspective view-projection (70 degree fov, 16:9, camera at the origin
// looking down -Z), column-major as the renderer's ViewUniforms lay it out.
fn camera_frustum() -> Frustum {
    let f = 1.0 / 35.0f32.to_radians().tan();
    let aspect = 16.0 / 9.0;
    let (near, far) = (0.1, 400.0);
    let mut vp = [[0.0f32; 4]; 4];
    vp[0][0] = f / aspect;
    vp[1][1] = f;
    vp[2][2] = (far + near) / (near - far);
    vp[2][3] = -1.0;
    vp[3][2] = 2.0 * far * near / (near - far);
    Frustum::from_view_projection(vp)
}

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

fn main() {
    let mut bench = Bench::from_env();

    let items = scene_items();
    let frustum = camera_frustum();

    bench.run("render/bvh_build/10k", OBJECTS as u64, || {
        Bvh::build(&items)
    });

    let bvh = Bvh::build(&items);
    let mut visible = 0u32;
    bvh.query(&frustum, [0.0; 3], |_| visible += 1);
    assert!(
        visible > 0 && (visible as usize) < OBJECTS,
        "the query fixture must accept some objects and reject others, saw {visible}"
    );
    bench.run("render/bvh_query/10k", OBJECTS as u64, || {
        let mut seen = 0u32;
        bvh.query(&frustum, [0.0; 3], |_| seen += 1);
        seen
    });

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

    bench.finish();
}
