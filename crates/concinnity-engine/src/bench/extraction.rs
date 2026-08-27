// src/bench/extraction.rs
//
// Per-frame render extraction: the pass GraphicsSystem runs to fill the
// RenderSnapshot from world state before submission. The pairs that matter
// are static-vs-moved: a static scene must cost only the change-gated join
// walk and allocate nothing, and a scene where everything moved pays the
// snapshot copy the boundary introduces. The pose row measures the one real
// copy extraction added over the old borrow-through path (joint matrices
// into the snapshot's span buffer).

use super::{BenchWorld, bench};
use crate::components::{GlobalTransform, Prop, RenderHandle, SkeletonPose};
use crate::ecs::{Entity, SkinnedMeshHandle};
use crate::gfx::graphics_system::GraphicsSystem;
use crate::gfx::snapshot::RenderSnapshot;

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
fn draw_world(count: usize) -> (BenchWorld, Vec<Entity>) {
    let mut world = BenchWorld::new();
    let entities = (0..count)
        .map(|i| {
            let entity = world.components.push_typed(Prop::default());
            world
                .components
                .insert_typed(entity, GlobalTransform(model_at(i)));
            world.components.insert_typed(
                entity,
                RenderHandle {
                    draws: [i as u32].into(),
                },
            );
            entity
        })
        .collect();
    (world, entities)
}

// `count` skinned poses at a production joint count.
fn pose_world(count: usize) -> BenchWorld {
    let mut world = BenchWorld::new();
    for i in 0..count {
        let entity = world.components.push_typed(Prop::default());
        world.components.insert_typed(
            entity,
            SkeletonPose {
                mesh_id: SkinnedMeshHandle(i as u32),
                skinned_index: i,
                skeleton: crate::gfx::skeleton::Skeleton::new(Vec::new()),
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

#[test]
#[ignore = "microbench; see crate::bench for the command"]
fn render_extraction() {
    println!("\nrender extraction");

    // A world nothing touches: the change gate drops every slot, so this is
    // the steady-state floor a static scene pays every frame. The allocs
    // column is the boundary's contract: 0 after warmup.
    for (name, count) in [("extract_static/100", SMALL), ("extract_static/20k", LARGE)] {
        let (mut world, _) = draw_world(count);
        let mut gs = GraphicsSystem::new();
        let mut snap = RenderSnapshot::default();
        gs.extract(&mut world.ctx(), &mut snap);
        bench(name, count as u64, || {
            gs.extract(&mut world.ctx(), &mut snap);
        });
    }

    // Every entity moved: each frame re-copies every model matrix into the
    // snapshot. The in-body mutation that defeats the gate is part of the
    // measured cost, as in the transforms bench.
    for (name, count) in [("extract_moved/100", SMALL), ("extract_moved/20k", LARGE)] {
        let (mut world, _) = draw_world(count);
        let mut gs = GraphicsSystem::new();
        let mut snap = RenderSnapshot::default();
        gs.extract(&mut world.ctx(), &mut snap);
        let mut pass = 0.0f32;
        bench(name, count as u64, || {
            pass += 0.001;
            for g in world.ctx().query_slice_mut::<GlobalTransform>() {
                g.0[3][1] = pass;
            }
            gs.extract(&mut world.ctx(), &mut snap);
        });
    }

    // Every pose updated each frame: the joint-matrix copy into the span
    // buffer, the one real copy the snapshot boundary introduced.
    {
        let mut world = pose_world(SMALL);
        let mut gs = GraphicsSystem::new();
        let mut snap = RenderSnapshot::default();
        gs.extract(&mut world.ctx(), &mut snap);
        bench(
            &format!("extract_poses_{JOINTS}j/100"),
            SMALL as u64,
            || {
                for pose in world.ctx().query_mut::<SkeletonPose>() {
                    pose.updated = true;
                }
                gs.extract(&mut world.ctx(), &mut snap);
            },
        );
    }
}
