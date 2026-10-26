// Measures the public World surface against the real registered component
// set rather than the synthetic one the storage benchmarks drive: populating a
// world the way a blob load does, iterating a column, targeted lookups,
// draining a column, and despawning. Per-entity costs carry the real
// `ComponentAsset` dispatch and the real column count.
//
//     cargo test -p concinnity-engine --release -- --ignored --nocapture \
//         --test-threads=1 bench_world

use concinnity_core::math::SaturatingFrom;
use std::format;
use std::vec::Vec;

use crate::components::Prop;
use crate::ecs::{ComponentSlot, Entity, World};
use crate::test_support::{Pace, Rng, bench};

const SIZES: [(usize, &str); 2] = [(10_000, "10k"), (100_000, "100k")];
const LOOKUPS: usize = 4_096;

fn prop(i: usize) -> Prop {
    Prop {
        position: [i as f32, 0.45, 0.0],
        ..Default::default()
    }
}

// A world holding `n` Props, with the entity handle for each.
fn populated(n: usize) -> (World, Vec<Entity>) {
    let mut world = World::new();
    world.reserve_components(&[(
        <Prop as ComponentSlot>::DISCRIMINANT,
        u32::saturating_from(n),
    )]);
    let entities = (0..n).map(|i| world.push(prop(i))).collect();
    (world, entities)
}

fn run(pace: Pace) {
    // A single-run pass only proves the fixtures, so it skips the large size.
    let sizes = match pace {
        Pace::Timed => &SIZES[..],
        Pace::Once => &SIZES[..1],
    };
    for &(n, label) in sizes {
        // The blob-load shape: pre-size the column from the manifest count,
        // then bulk-push. `reserve_components` is what keeps this from
        // reallocating mid-load.
        bench(
            pace,
            &format!("ecs/world_populate/{label}"),
            n as u64,
            || populated(n).0.component_count(),
        );

        // The same load without the manifest pre-size, so the reserve's worth
        // is a subtraction rather than a claim.
        bench(
            pace,
            &format!("ecs/world_populate_unreserved/{label}"),
            n as u64,
            || {
                let mut world = World::new();
                for i in 0..n {
                    world.push(prop(i));
                }
                world.component_count()
            },
        );

        let (mut world, entities) = populated(n);
        assert_eq!(world.component_count(), n, "fixture holds every prop");

        bench(pace, &format!("ecs/world_query/{label}"), n as u64, || {
            let mut acc = 0.0f32;
            for p in world.query::<Prop>() {
                acc += p.position[0];
            }
            acc
        });

        let mut handles = entities.clone();
        Rng::new(11).shuffle(&mut handles);
        handles.truncate(LOOKUPS);
        bench(
            pace,
            &format!("ecs/world_get/{label}"),
            LOOKUPS as u64,
            || {
                let mut acc = 0.0f32;
                for &e in &handles {
                    if let Some(p) = world.get::<Prop>(e) {
                        acc += p.position[0];
                    }
                }
                acc
            },
        );

        bench(pace, &format!("ecs/world_write/{label}"), n as u64, || {
            for p in world.query_mut::<Prop>() {
                p.position[1] += 1.0;
            }
        });

        // Draining the column despawns every owner whose last component it
        // was, so this is the teardown half of a scene swap.
        bench(pace, &format!("ecs/world_drain/{label}"), n as u64, || {
            let (mut world, _) = populated(n);
            world.remove_all::<Prop>();
            world.component_count()
        });

        // Despawning one entity at a time walks every registered column, so
        // this less the populate row is the per-entity despawn cost.
        bench(
            pace,
            &format!("ecs/world_populate_despawn/{label}"),
            n as u64,
            || {
                let (mut world, entities) = populated(n);
                for e in entities {
                    world.despawn(e);
                }
                world.component_count()
            },
        );
    }
}

#[test]
#[ignore = "benchmark; run with --ignored --test-threads=1"]
fn bench_world() {
    run(Pace::Timed);
}

#[test]
fn world_fixtures_build_and_run() {
    run(Pace::Once);
}
