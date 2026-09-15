// Measures the storage primitives a world is built out of: spawn and despawn
// churn, column scans, multi-component joins, targeted lookups, and the event
// queue. Fixtures are
// deterministic, and partner columns are laid out in shuffled entity order so
// join probes pay a real scattered read rather than a coincidentally
// sequential one.
//
// In-module rather than under a bench target because the storage macro it
// drives is expanded by its consumers rather than called across a boundary.
//
//     cargo test -p concinnity-core --release -- --ignored --nocapture \
//         --test-threads=1 bench_storage

#![expect(
    dead_code,
    unreachable_pub,
    reason = "BenchWorld is module-private, so the generated pub items are unreachable and dead_code fires on whichever ones these benchmarks skip"
)]

use std::collections::VecDeque;
use std::format;
use std::vec::Vec;

use crate::define_component_storage;
use crate::ecs::{Entity, EventCursor, Events};
use crate::test_support::{Pace, bench};

// Deterministic xorshift so fixtures lay out identically across runs.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            items.swap(i, j);
        }
    }
}

// A world matrix, the bulky per-entity datum the render prep walks.
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct Transform {
    m: [f32; 16],
}

impl Transform {
    fn keyed(i: usize) -> Transform {
        Transform { m: [i as f32; 16] }
    }
}

// The small renderable descriptor joined onto each transform.
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct Renderer {
    mesh: u32,
    material: u32,
}

#[derive(Clone, Copy, Default, Debug)]
// A bench component carrying a velocity vector.
pub(crate) struct Motion {
    velocity: [f32; 4],
}

define_component_storage! {
    storage: BenchWorld,
    slot: BenchComponent,
    Transform => Transform, 1,
    Renderer => Renderer, 2,
    Motion => Motion, 3,
}

const RENDERABLE_PERMIL: usize = 750;

// The world sizes and per-body counts a pass drives, per pace. A single-run
// pass keeps every shape and shrinks every count: it proves the fixtures still
// build and the joins still walk, and measures nothing, so a 100k-entity world
// in an unoptimized build would buy it nothing.
struct Fixture {
    sizes: [(usize, &'static str); 2],
    churn: usize,
    lookups: usize,
    events: usize,
}

fn fixture(pace: Pace) -> Fixture {
    match pace {
        Pace::Timed => Fixture {
            sizes: [(10_000, "10k"), (100_000, "100k")],
            churn: 1_024,
            lookups: 4_096,
            events: 1_024,
        },
        Pace::Once => Fixture {
            sizes: [(64, "64"), (128, "128")],
            churn: 8,
            lookups: 16,
            events: 8,
        },
    }
}

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

fn run(pace: Pace) {
    let Fixture {
        sizes,
        churn,
        lookups,
        events: event_count,
    } = fixture(pace);

    for (n, label) in sizes {
        bench(pace, &format!("ecs/spawn_prop/{label}"), n as u64, || {
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
            bench(
                pace,
                &format!("ecs/despawn_spawn/{label}"),
                churn as u64,
                move || {
                    for i in 0..churn {
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

        bench(
            pace,
            &format!("ecs/scan_transforms/{label}"),
            n as u64,
            || {
                let mut acc = 0.0f32;
                for t in world.Transform.iter() {
                    acc += t.m[0];
                }
                acc
            },
        );

        bench(
            pace,
            &format!("ecs/join2/{label}"),
            renderable as u64,
            || {
                let mut acc = 0u64;
                for (e, t, r) in world.join2::<Transform, Renderer>() {
                    acc = acc
                        .wrapping_add(t.m[0].to_bits() as u64)
                        .wrapping_add(r.mesh as u64)
                        .wrapping_add(e.index() as u64);
                }
                acc
            },
        );

        bench(
            pace,
            &format!("ecs/join3/{label}"),
            renderable as u64,
            || {
                let mut acc = 0u64;
                for (e, t, r, m) in world.join3::<Transform, Renderer, Motion>() {
                    acc = acc
                        .wrapping_add(t.m[0].to_bits() as u64)
                        .wrapping_add(r.material as u64)
                        .wrapping_add(m.velocity[1].to_bits() as u64)
                        .wrapping_add(e.index() as u64);
                }
                acc
            },
        );

        let mut handles = entities.clone();
        Rng::new(3).shuffle(&mut handles);
        handles.truncate(lookups);
        bench(
            pace,
            &format!("ecs/get_random/{label}"),
            lookups as u64,
            || {
                let mut acc = 0.0f32;
                for &e in &handles {
                    if let Some(t) = world.get::<Transform>(e) {
                        acc += t.m[0];
                    }
                }
                acc
            },
        );

        bench(
            pace,
            &format!("ecs/write_transforms/{label}"),
            n as u64,
            || {
                for t in world.values_mut::<Transform>() {
                    t.m[0] += 1.0;
                }
            },
        );
    }

    {
        let mut events: Events<Motion> = Events::new();
        let mut cursor = EventCursor::default();
        bench(pace, "ecs/events_pump/1k", event_count as u64, move || {
            for i in 0..event_count {
                events.send(Motion {
                    velocity: [i as f32, 0.0, 0.0, 0.0],
                });
            }
            let seen = events.read(&mut cursor).count();
            events.update();
            seen
        });
    }
}

#[test]
#[ignore = "benchmark; run with --ignored --test-threads=1"]
fn bench_storage() {
    run(Pace::Timed);
}

#[test]
fn storage_fixtures_build_and_run() {
    run(Pace::Once);
}
