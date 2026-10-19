// Per-frame render extraction: the pass that fills the
// RenderSnapshot from world state before submission. The pairs that matter
// are static-vs-moved: a static scene must cost only the change-gated join
// walk and allocate nothing, and a scene where everything moved pays the
// snapshot copy the boundary introduces. The pose row measures the one real
// copy extraction added over the old borrow-through path (joint matrices
// into the snapshot's span buffer).
//
//     cargo test -p concinnity-core --release -- --ignored --nocapture \
//         --test-threads=1 render_extraction

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use std::println;

use super::FrameExtractor;
use crate::animation::skeleton;
use crate::components::{GlobalTransform, Prop, RenderHandle, SkeletonPose};
use crate::ecs::{Entity, SkinnedMeshHandle, World};
use crate::gfx::render_types::{DrawIndex, SkinnedIndex};
use crate::render::snapshot::RenderSnapshot;
use crate::test_support::{Pace, bench};

const SMALL: usize = 100;
const LARGE: usize = 20_000;
const JOINTS: usize = 64;

fn model_at(i: usize) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [i as f32, 0.0, 0.0, 1.0],
    ]
}

// `count` entities each holding one draw slot, with GlobalTransform authored
// directly so the propagation pass (benched on its own in `transforms`) stays
// out of the measurement.
fn draw_world(count: usize) -> (World, Vec<Entity>) {
    let mut world = World::new();
    let entities = (0..count)
        .map(|i| {
            let entity = world.push(Prop::default());
            world.insert(entity, GlobalTransform(model_at(i)));
            world.insert(
                entity,
                RenderHandle {
                    draws: [DrawIndex::from_usize(i)].into(),
                },
            );
            entity
        })
        .collect();
    (world, entities)
}

// `count` skinned poses at a production joint count.
fn pose_world(count: usize) -> World {
    let mut world = World::new();
    for i in 0..count {
        let entity = world.push(Prop::default());
        world.insert(
            entity,
            SkeletonPose {
                mesh_id: SkinnedMeshHandle::new(i as u32),
                skinned_index: SkinnedIndex::from_usize(i),
                skeleton: skeleton::Skeleton::new(Vec::new()),
                joint_matrices: vec![model_at(i); JOINTS],
                morph_weights: Vec::new(),
                morph_base: Vec::new(),
                proportions: Default::default(),
                updated: true,
                scratch: Default::default(),
            },
        );
    }
    world
}

fn run(pace: Pace) {
    if pace == Pace::Timed {
        println!("\nrender extraction");
    }

    // A world nothing touches: the change gate drops every slot, so this is
    // the steady-state floor a static scene pays every frame. The allocs
    // column is the boundary's contract: 0 after warmup.
    for (name, count) in [("extract_static/100", SMALL), ("extract_static/20k", LARGE)] {
        let (mut world, _) = draw_world(count);
        let mut extractor = FrameExtractor::default();
        let mut snap = RenderSnapshot::default();
        extractor.extract(&mut world.context(), &mut snap, 0);
        bench(pace, name, count as u64, || {
            extractor.extract(&mut world.context(), &mut snap, 0);
        });
    }

    // Every entity moved: each frame re-copies every model matrix into the
    // snapshot. The in-body mutation that defeats the gate is part of the
    // measured cost, as in the transforms bench.
    for (name, count) in [("extract_moved/100", SMALL), ("extract_moved/20k", LARGE)] {
        let (mut world, _) = draw_world(count);
        let mut extractor = FrameExtractor::default();
        let mut snap = RenderSnapshot::default();
        extractor.extract(&mut world.context(), &mut snap, 0);
        let mut pass = 0.0f32;
        bench(pace, name, count as u64, || {
            pass += 0.001;
            for g in world.context().query_slice_mut::<GlobalTransform>() {
                g.0[3][1] = pass;
            }
            extractor.extract(&mut world.context(), &mut snap, 0);
        });
    }

    // Every pose updated each frame: the joint-matrix copy into the span
    // buffer, the one real copy the snapshot boundary introduced.
    {
        let mut world = pose_world(SMALL);
        let mut extractor = FrameExtractor::default();
        let mut snap = RenderSnapshot::default();
        extractor.extract(&mut world.context(), &mut snap, 0);
        bench(
            pace,
            &format!("extract_poses_{JOINTS}j/100"),
            SMALL as u64,
            || {
                for pose in world.context().query_mut::<SkeletonPose>() {
                    pose.updated = true;
                }
                extractor.extract(&mut world.context(), &mut snap, 0);
            },
        );
    }
}

// Static extraction is allocation-free: the change gate drops every slot, so a
// frame where nothing moved must not touch the heap. The allocation counters
// are process-wide, so it keeps extracting until one pass lands at zero; only a
// whole deadline without one fails.
#[test]
fn static_extraction_allocates_nothing() {
    const WARMUP_FRAMES: usize = 64;
    const DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);

    let (mut world, _) = draw_world(SMALL);
    let mut extractor = FrameExtractor::default();
    let mut snap = RenderSnapshot::default();
    for _ in 0..WARMUP_FRAMES {
        extractor.extract(&mut world.context(), &mut snap, 0);
    }
    let deadline = std::time::Instant::now() + DEADLINE;
    let mut min = u64::MAX;
    while min > 0 && std::time::Instant::now() < deadline {
        let before = crate::memory::alloc_count().expect("the test binary tracks its heap");
        extractor.extract(&mut world.context(), &mut snap, 0);
        let after = crate::memory::alloc_count().expect("the allocator stays installed");
        min = min.min(after - before);
    }
    assert_eq!(
        min, 0,
        "static extraction's quietest pass allocated {min} times; \
         the zero-alloc extraction contract broke"
    );
}

#[test]
#[ignore = "microbench; run it by name with --ignored --nocapture --test-threads=1"]
fn render_extraction() {
    run(Pace::Timed);
}

#[test]
fn render_extraction_fixtures_build_and_run() {
    run(Pace::Once);
}
