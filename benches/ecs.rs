//! Benchmarks over the EAS storage primitives: spawn and despawn churn, column
//! scans, multi-component joins, targeted lookups, deferred commands, the
//! sparse column, and the event queue. Fixtures are deterministic, and partner
//! columns are laid out in shuffled entity order so join probes pay a real
//! scattered read rather than a coincidentally sequential one.
//!
//! Run with `cargo bench -p concinnity-bench --bench ecs`, optionally followed
//! by `-- <substring>` to select benchmarks.

extern crate alloc;

use std::collections::VecDeque;

use concinnity_bench::{Bench, Rng};
use concinnity_eas::{
    CommandQueue, CommandTarget, Entities, Entity, EventCursor, Events, SparseColumn,
    define_component_storage,
};

/// A world matrix, the bulky per-entity datum the render prep walks.
#[derive(Clone, Copy, Default, Debug)]
pub struct Transform {
    m: [f32; 16],
}

impl Transform {
    fn keyed(i: usize) -> Transform {
        Transform { m: [i as f32; 16] }
    }
}

/// The small renderable descriptor joined onto each transform.
#[derive(Clone, Copy, Default, Debug)]
pub struct Renderer {
    mesh: u32,
    material: u32,
}

#[derive(Clone, Copy, Default, Debug)]
/// A bench component carrying a velocity vector.
pub struct Motion {
    velocity: [f32; 4],
}

define_component_storage! {
    storage: BenchWorld,
    slot: BenchComponent,
    transforms => Transform, 1,
    renderers => Renderer, 2,
    motions => Motion, 3,
}

impl CommandTarget for BenchWorld {
    fn despawn_entity(&mut self, entity: Entity) {
        self.despawn(entity);
    }
}

const SIZES: [(usize, &str); 2] = [(10_000, "10k"), (100_000, "100k")];
const RENDERABLE_PERMIL: usize = 750;
const CHURN: usize = 1_024;
const LOOKUPS: usize = 4_096;
const COMMANDS: usize = 512;
const EVENTS: usize = 1_024;

// A world of `n` transform entities, `renderable_permil`/1000 of which also
// carry a Renderer and a Motion, inserted in shuffled order so their rows do
// not track the transform rows.
fn build_world(n: usize, renderable_permil: usize, seed: u64) -> (BenchWorld, Vec<Entity>) {
    let mut world = BenchWorld::default();
    let mut entities = Vec::with_capacity(n);
    for i in 0..n {
        entities.push(world.push_typed(Transform::keyed(i)));
    }
    let mut shuffled = entities.clone();
    Rng::new(seed).shuffle(&mut shuffled);
    for (i, &e) in shuffled
        .iter()
        .take(n * renderable_permil / 1000)
        .enumerate()
    {
        world.insert_typed(
            e,
            Renderer {
                mesh: i as u32,
                material: i as u32 * 2,
            },
        );
        world.insert_typed(
            e,
            Motion {
                velocity: [0.0, 1.0, 0.0, 0.0],
            },
        );
    }
    (world, entities)
}

fn main() {
    let mut bench = Bench::from_env();

    for (n, label) in SIZES {
        bench.run(&format!("ecs/spawn_prop/{label}"), n as u64, || {
            let mut world = BenchWorld::default();
            for i in 0..n {
                let e = world.push_typed(Transform::keyed(i));
                world.insert_typed(
                    e,
                    Renderer {
                        mesh: i as u32,
                        material: i as u32,
                    },
                );
            }
            world.len()
        });

        {
            let (mut world, entities) = build_world(n, 0, 1);
            let mut live: VecDeque<Entity> = entities.into();
            bench.run(
                &format!("ecs/despawn_spawn/{label}"),
                CHURN as u64,
                move || {
                    for i in 0..CHURN {
                        let dead = live.pop_front().expect("churn fixture stays populated");
                        world.despawn(dead);
                        live.push_back(world.push_typed(Transform::keyed(i)));
                    }
                    world.len()
                },
            );
        }

        let (mut world, entities) = build_world(n, RENDERABLE_PERMIL, 2);
        let renderable = n * RENDERABLE_PERMIL / 1000;
        assert_eq!(
            world.join2::<Transform, Renderer>().count(),
            renderable,
            "join fixture must match its renderable fraction"
        );
        assert_eq!(
            world.join3::<Transform, Renderer, Motion>().count(),
            renderable
        );

        bench.run(&format!("ecs/scan_transforms/{label}"), n as u64, || {
            let mut acc = 0.0f32;
            for t in world.transforms.iter() {
                acc += t.m[0];
            }
            acc
        });

        bench.run(&format!("ecs/join2/{label}"), renderable as u64, || {
            let mut acc = 0u64;
            for (e, t, r) in world.join2::<Transform, Renderer>() {
                acc = acc
                    .wrapping_add(t.m[0].to_bits() as u64)
                    .wrapping_add(r.mesh as u64)
                    .wrapping_add(e.index() as u64);
            }
            acc
        });

        bench.run(&format!("ecs/join3/{label}"), renderable as u64, || {
            let mut acc = 0u64;
            for (e, t, r, m) in world.join3::<Transform, Renderer, Motion>() {
                acc = acc
                    .wrapping_add(t.m[0].to_bits() as u64)
                    .wrapping_add(r.material as u64)
                    .wrapping_add(m.velocity[1].to_bits() as u64)
                    .wrapping_add(e.index() as u64);
            }
            acc
        });

        let mut handles = entities.clone();
        Rng::new(3).shuffle(&mut handles);
        handles.truncate(LOOKUPS);
        bench.run(&format!("ecs/get_random/{label}"), LOOKUPS as u64, || {
            let mut acc = 0.0f32;
            for &e in &handles {
                if let Some(t) = world.get::<Transform>(e) {
                    acc += t.m[0];
                }
            }
            acc
        });

        bench.run(&format!("ecs/write_transforms/{label}"), n as u64, || {
            for t in world.values_mut::<Transform>() {
                t.m[0] += 1.0;
            }
        });
    }

    {
        let ids = Entities::new();
        let mut queue: CommandQueue<BenchWorld> = CommandQueue::new();
        let mut world = BenchWorld::default();
        bench.run("ecs/command_spawn_apply/512", COMMANDS as u64, move || {
            {
                let mut commands = queue.recorder(&ids);
                for i in 0..COMMANDS {
                    commands.run(move |w: &mut BenchWorld| {
                        w.push_typed(Transform::keyed(i));
                    });
                }
            }
            queue.apply(&mut world);
            world.drain::<Transform>().len()
        });
    }

    {
        let mut ids = Entities::new();
        let mut sparse: SparseColumn<Motion> = SparseColumn::new();
        let handles: Vec<Entity> = (0..10_000).map(|_| ids.alloc()).collect();
        for &e in &handles {
            sparse.insert(e, Motion::default());
        }
        let mut next = 0usize;
        bench.run("ecs/sparse_churn/10k", CHURN as u64, move || {
            for _ in 0..CHURN {
                let e = handles[next];
                next = (next + 1) % handles.len();
                let value = sparse.remove(e).expect("churned entity stays resident");
                sparse.insert(e, value);
            }
            sparse.len()
        });
    }

    {
        let mut events: Events<Motion> = Events::new();
        let mut cursor = EventCursor::default();
        bench.run("ecs/events_pump/1k", EVENTS as u64, move || {
            for i in 0..EVENTS {
                events.send(Motion {
                    velocity: [i as f32, 0.0, 0.0, 0.0],
                });
            }
            let seen = events.read(&mut cursor).count();
            events.update();
            seen
        });
    }

    bench.finish();
}
