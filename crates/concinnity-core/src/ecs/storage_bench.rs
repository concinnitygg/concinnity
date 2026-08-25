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
use std::println;
use std::time::Instant;
use std::vec::Vec;

use crate::define_component_storage;
use crate::ecs::{Entity, EventCursor, Events};

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

// One measured pass runs at least this long before its time is trusted.
const TARGET_NS: u128 = 200_000_000;
const MAX_ITERS: u64 = 1 << 20;

// Time `body` over a calibrated iteration count and report its per-item cost.
fn bench<R>(name: &str, items: u64, mut body: impl FnMut() -> R) {
    let mut iters: u64 = 1;
    loop {
        let start = Instant::now();
        for _ in 0..iters {
            core::hint::black_box(body());
        }
        if start.elapsed().as_nanos() >= TARGET_NS || iters >= MAX_ITERS {
            break;
        }
        iters = iters.saturating_mul(4).min(MAX_ITERS);
    }

    let start = Instant::now();
    for _ in 0..iters {
        core::hint::black_box(body());
    }
    let elapsed = start.elapsed();
    let per_item_ns = elapsed.as_secs_f64() * 1e9 / (iters * items.max(1)) as f64;
    println!("  {name:<40} {per_item_ns:>10.2} ns/item");
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

const SIZES: [(usize, &str); 2] = [(10_000, "10k"), (100_000, "100k")];
const RENDERABLE_PERMIL: usize = 750;
const CHURN: usize = 1_024;
const LOOKUPS: usize = 4_096;
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

#[test]
#[ignore = "benchmark; run with --ignored --test-threads=1"]
fn bench_storage() {
    for (n, label) in SIZES {
        bench(&format!("ecs/spawn_prop/{label}"), n as u64, || {
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

        bench(&format!("ecs/scan_transforms/{label}"), n as u64, || {
            let mut acc = 0.0f32;
            for t in world.Transform.iter() {
                acc += t.m[0];
            }
            acc
        });

        bench(&format!("ecs/join2/{label}"), renderable as u64, || {
            let mut acc = 0u64;
            for (e, t, r) in world.join2::<Transform, Renderer>() {
                acc = acc
                    .wrapping_add(t.m[0].to_bits() as u64)
                    .wrapping_add(r.mesh as u64)
                    .wrapping_add(e.index() as u64);
            }
            acc
        });

        bench(&format!("ecs/join3/{label}"), renderable as u64, || {
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
        bench(&format!("ecs/get_random/{label}"), LOOKUPS as u64, || {
            let mut acc = 0.0f32;
            for &e in &handles {
                if let Some(t) = world.get::<Transform>(e) {
                    acc += t.m[0];
                }
            }
            acc
        });

        bench(&format!("ecs/write_transforms/{label}"), n as u64, || {
            for t in world.values_mut::<Transform>() {
                t.m[0] += 1.0;
            }
        });
    }

    {
        let mut events: Events<Motion> = Events::new();
        let mut cursor = EventCursor::default();
        bench("ecs/events_pump/1k", EVENTS as u64, move || {
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
}
