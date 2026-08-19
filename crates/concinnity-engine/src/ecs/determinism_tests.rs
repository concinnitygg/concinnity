// src/ecs/determinism_tests.rs
//
// The schedule-determinism gate: the same world stepped N ticks under the
// serial and parallel schedule modes must land in bit-identical state. This
// is the acceptance test every parallel execution path (behavior evaluation,
// physics solve) must keep green; a hash mismatch here means completion order
// leaked into world state.
//
// The hash covers component state (transforms), the entity population, and
// the event traffic. Change ticks are excluded by design: their values are
// interleaving-dependent under the atomic counter and must never be treated
// as world state.

use crate::assets::{
    Behavior, BehaviorSource, BodyDynamics, Collider, Expr, Node, PhysicsConfig, Prop,
    PropCollider, Transform,
};
use crate::ecs::{ScheduleMode, StepResult, World};

fn fnv(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash ^= b as u64;
        *hash = hash.wrapping_mul(0x100000001b3);
    }
}

fn hash_f32s(hash: &mut u64, values: &[f32]) {
    for v in values {
        fnv(hash, &v.to_bits().to_le_bytes());
    }
}

fn hash_world(world: &World) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for t in world.query::<Transform>() {
        hash_f32s(&mut h, &t.position);
        hash_f32s(&mut h, &t.rotation_deg);
        hash_f32s(&mut h, &t.scale);
    }
    fnv(&mut h, &(world.component_count() as u64).to_le_bytes());
    if let Some(events) = world.events::<crate::assets::ContactEvent>() {
        fnv(&mut h, &(events.len() as u64).to_le_bytes());
    }
    h
}

// A world exercising the two heavy sim paths: scoped behaviors that both
// accumulate shared variables and move their entity, and dynamic bodies
// falling onto a floor. No graphics band (no GPU in tests).
fn build_world() -> World {
    let mut world = World::new();

    // Independent per-prop movers: each Prop drifts along +X every tick.
    world.add_component(Behavior {
        on: BehaviorSource::Tick,
        scope: vec!["Prop".into()],
        body: vec![Node::SetTransform {
            entity: Expr::SelfEntity,
            position: Some(Expr::Add(
                Box::new(Expr::Position(Box::new(Expr::SelfEntity))),
                Box::new(Expr::Vec3([0.01, 0.0, 0.0])),
            )),
            rotation_deg: None,
            scale: None,
        }],
        ..Default::default()
    });
    // A dependent chain through shared variables: b reads what a wrote this
    // tick, so their relative order is observable in `pace`.
    world.add_component(Behavior {
        on: BehaviorSource::Tick,
        body: vec![Node::Set {
            var: "beat".into(),
            value: Expr::Int(1),
            add: true,
        }],
        ..Default::default()
    });
    world.add_component(Behavior {
        on: BehaviorSource::Tick,
        body: vec![
            Node::Set {
                var: "pace".into(),
                value: Expr::Mul(Box::new(Expr::Var("beat".into())), Box::new(Expr::Int(3))),
                add: false,
            },
            Node::SetTransform {
                entity: Expr::First("props".into()),
                position: None,
                rotation_deg: Some(Expr::Vec3([0.0, 1.0, 0.0])),
                scale: None,
            },
        ],
        queries: vec![crate::assets::QueryDecl {
            name: "props".into(),
            has: vec!["Prop".into()],
        }],
        ..Default::default()
    });

    // Enough props that the scoped mover fires well past the parallel-eval
    // job threshold, so the Parallel run exercises the pooled path for real.
    for i in 0..200 {
        world.add_component(Prop {
            position: [i as f32 * 2.0, 0.5, 0.0],
            ..Default::default()
        });
    }

    // Dynamic boxes above a flat floor: eight well-separated jostling stacks,
    // so the solver has several independent islands and its parallel path has
    // real work to reorder if it ever could.
    world.add_component(PhysicsConfig::default());
    for i in 0..32 {
        let (stack, level) = (i / 4, i % 4);
        let e = world.push(Transform {
            position: [
                stack as f32 * 25.0,
                0.6 + level as f32 * 1.05,
                0.05 * level as f32,
            ],
            rotation_deg: [0.0; 3],
            scale: [1.0; 3],
        });
        world.insert(
            e,
            Collider(PropCollider {
                shape: "cuboid".into(),
                half_extents: [0.4, 0.4, 0.4],
                ..Default::default()
            }),
        );
        world.insert(
            e,
            BodyDynamics {
                restitution: 0.6,
                ..Default::default()
            },
        );
    }

    world
}

fn run(mode: ScheduleMode, ticks: u32) -> u64 {
    let mut world = build_world();
    world.insert_resource(mode);
    world.start().expect("world starts");
    for _ in 0..ticks {
        assert_eq!(world.step(), StepResult::Continue);
    }
    hash_world(&world)
}

#[test]
fn parallel_schedule_state_matches_serial() {
    let serial = run(ScheduleMode::Serial, 240);
    let parallel = run(ScheduleMode::Parallel, 240);
    assert_eq!(
        serial, parallel,
        "parallel schedule diverged from serial world state",
    );
}

#[test]
fn repeated_serial_runs_are_reproducible() {
    assert_eq!(
        run(ScheduleMode::Serial, 120),
        run(ScheduleMode::Serial, 120)
    );
}

#[test]
fn the_world_actually_moves() {
    let start = {
        let mut world = build_world();
        world.start().expect("world starts");
        hash_world(&world)
    };
    assert_ne!(
        start,
        run(ScheduleMode::Serial, 120),
        "the gate world must produce motion, or the hash proves nothing",
    );
}
