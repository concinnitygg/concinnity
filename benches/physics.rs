// benches/physics.rs
//
// Benchmarks over the physics wrapper and rapier: solver cost under
// sustained contacts, the sleeping-island idle path, contact-free
// integration, world construction, body churn, and the query paths
// (raycast, character move). The whole-frame probe cannot hold this
// number still -- an un-settled solver sees a different contact set every
// session -- so the stepping benches rebuild the same world and step it
// the same fixed count every iteration, making the measured work
// bit-identical run to run. Fixtures mirror the CPU stress world's
// physics axis: stacked columns of bouncing cuboids that never sleep.
//
// Run with `cargo bench -p concinnity-bench --bench physics`.

use concinnity_bench::Bench;
use concinnity_physics::{
    BodyHandle, CharacterMoveInput, CharacterShape, ColliderShape, DynamicParams, LayerMask,
    PhysicsWorld,
};

const TICK: f32 = 1.0 / 60.0;
// Steps per iteration for the deterministic drop benches: enough to cover
// the fall, the impacts, and sustained stack jostle.
const DROP_STEPS: usize = 60;
const SIZES: [(usize, &str); 2] = [(256, "256"), (1024, "1k")];
const STACK: usize = 8;
const CUBE: ColliderShape = ColliderShape::Cuboid {
    half_extents: [0.4, 0.4, 0.4],
};

fn cube_params(restitution: f32, damping: f32) -> DynamicParams {
    DynamicParams {
        mass: 1.0,
        friction: 0.4,
        restitution,
        gravity_scale: 1.0,
        linear_damping: damping,
    }
}

// Stacked columns of dynamic cuboids over a fixed floor. High restitution
// and zero damping keep the stacks jostling (the stress-world shape); low
// restitution with damping lets them settle and sleep.
fn stacked_world(
    bodies: usize,
    restitution: f32,
    damping: f32,
    floor: bool,
) -> (PhysicsWorld, Vec<BodyHandle>) {
    let mut world = PhysicsWorld::new(9.81);
    if floor {
        world.add_fixed(
            &ColliderShape::Cuboid {
                half_extents: [200.0, 0.5, 200.0],
            },
            [0.0, -0.5, 0.0],
            [0.0; 3],
            0.8,
            LayerMask::ALL,
        );
    }
    let columns = bodies.div_ceil(STACK);
    let per_row = (columns as f32).sqrt().ceil().max(1.0) as usize;
    let handles = (0..bodies)
        .map(|i| {
            let (column, level) = (i / STACK, i % STACK);
            let pos = [
                (column % per_row) as f32 * 2.0,
                0.5 + level as f32 * 1.05,
                (column / per_row) as f32 * 2.0,
            ];
            world.add_dynamic(
                &CUBE,
                pos,
                [0.0; 3],
                cube_params(restitution, damping),
                LayerMask::ALL,
            )
        })
        .collect();
    (world, handles)
}

fn step_and_drain(world: &mut PhysicsWorld) {
    world.step(TICK);
    world.drain_contact_hits();
    world.drain_sensor_crossings();
}

fn poses(world: &PhysicsWorld, handles: &[BodyHandle]) -> Vec<([f32; 3], [f32; 3])> {
    handles.iter().map(|&h| world.body_pose(h)).collect()
}

// The premise the stepping benches rest on: identical worlds stepped the
// same count report bit-identical poses. If a rapier upgrade breaks this,
// fail loudly instead of quietly turning into a noisy benchmark.
fn assert_deterministic() {
    let run = || {
        let (mut world, handles) = stacked_world(256, 0.85, 0.0, true);
        for _ in 0..120 {
            step_and_drain(&mut world);
        }
        poses(&world, &handles)
    };
    assert_eq!(run(), run(), "physics step is no longer deterministic");
}

fn main() {
    let mut bench = Bench::from_env();

    for (n, label) in SIZES {
        let body_steps = (n * DROP_STEPS) as u64;

        // Same build, same steps, every iteration: drop, impact, jostle.
        bench.run(
            &format!("physics/step_settling/{label}"),
            body_steps,
            || {
                let (mut world, handles) = stacked_world(n, 0.85, 0.0, true);
                for _ in 0..DROP_STEPS {
                    step_and_drain(&mut world);
                }
                world.body_pose(handles[0]).0[1].to_bits()
            },
        );

        // No floor: the bodies never touch anything, so this is broad-phase
        // plus integration alone; the settling delta above it is the solver.
        bench.run(
            &format!("physics/step_free_fall/{label}"),
            body_steps,
            || {
                let (mut world, handles) = stacked_world(n, 0.85, 0.0, false);
                for _ in 0..DROP_STEPS {
                    step_and_drain(&mut world);
                }
                world.body_pose(handles[0]).0[1].to_bits()
            },
        );

        bench.run(&format!("physics/build_world/{label}"), n as u64, || {
            stacked_world(n, 0.85, 0.0, true).1.len()
        });

        // A world stepped to sleep: each measured step is idle-island cost
        // and leaves the world unchanged, so iterations stay identical.
        let (mut settled, handles) = stacked_world(n, 0.0, 0.5, true);
        for _ in 0..600 {
            step_and_drain(&mut settled);
        }
        let poses_before = poses(&settled, &handles);
        bench.run(&format!("physics/step_settled/{label}"), n as u64, || {
            step_and_drain(&mut settled);
            handles.len()
        });
        assert_eq!(
            poses(&settled, &handles),
            poses_before,
            "settled fixture was still moving while measured"
        );
    }

    {
        let (mut world, _handles) = stacked_world(1024, 0.0, 0.5, true);
        const RAYS: usize = 1024;
        bench.run("physics/raycast/1k", RAYS as u64, || {
            let mut hits = 0u32;
            for i in 0..RAYS {
                let angle = i as f32 * core::f32::consts::TAU / RAYS as f32;
                let hit = world.raycast(
                    [12.0, 6.0, 12.0],
                    [angle.cos(), -0.6, angle.sin()],
                    100.0,
                    None,
                    LayerMask::ALL,
                );
                hits += hit.is_some() as u32;
            }
            hits
        });

        let capsule = world.add_character(0.6, 0.3, [1.0, 1.1, 1.0], LayerMask::ALL);
        let shape = CharacterShape::capsule(0.6, 0.3);
        bench.run("physics/character_move/1", 1, || {
            let moved = world.move_character(
                &shape,
                &CharacterMoveInput {
                    center: [1.0, 1.1, 1.0],
                    desired: [0.05, 0.0, 0.02],
                    dt: TICK,
                    exclude: capsule,
                    mask: LayerMask::ALL,
                },
            );
            moved.grounded
        });

        const CHURN: usize = 64;
        let mut churned = Vec::with_capacity(CHURN);
        bench.run("physics/body_churn/64", CHURN as u64, || {
            for i in 0..CHURN {
                churned.push(world.add_dynamic(
                    &CUBE,
                    [80.0 + (i % 8) as f32, 4.0 + (i / 8) as f32 * 1.1, 80.0],
                    [0.0; 3],
                    cube_params(0.2, 0.0),
                    LayerMask::ALL,
                ));
            }
            let added = churned.len();
            for handle in churned.drain(..) {
                world.remove_body(handle);
            }
            added
        });
    }

    assert_deterministic();
    bench.finish();
}
