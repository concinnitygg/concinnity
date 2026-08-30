//! Benchmarks over the engine's public World surface: populating a world the
//! way a blob load does, iterating a component column, targeted lookups, and
//! draining a column. Unlike the `ecs` target -- which drives the storage
//! macro through a synthetic three-component set -- these run against the
//! engine's real registered component set, so per-entity costs carry the real
//! `ComponentAsset` dispatch and the real column count.
//!
//! `World::despawn` is `#[cfg(test)]`-gated and so is not reachable here; it
//! walks every registered column, which is exactly where the two component
//! sets diverge most. That measurement belongs to an in-crate bench.

use crate::support::{Bench, Rng};
use concinnity_engine::components::Prop;
use concinnity_engine::ecs::{ComponentSlot, Entity, World};

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
    world.reserve_components(&[(<Prop as ComponentSlot>::DISCRIMINANT, n as u32)]);
    let entities = (0..n).map(|i| world.push(prop(i))).collect();
    (world, entities)
}

pub(crate) fn benches(bench: &mut Bench) {
    for (n, label) in SIZES {
        // The blob-load shape: pre-size the column from the manifest count,
        // then bulk-push. `reserve_components` is what keeps this from
        // reallocating mid-load.
        bench.run(&format!("engine/world_populate/{label}"), n as u64, || {
            populated(n).0.component_count()
        });

        // The same load without the manifest pre-size, so the reserve's worth
        // is a subtraction rather than a claim.
        bench.run(
            &format!("engine/world_populate_unreserved/{label}"),
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

        bench.run(&format!("engine/world_query/{label}"), n as u64, || {
            let mut acc = 0.0f32;
            for p in world.query::<Prop>() {
                acc += p.position[0];
            }
            acc
        });

        let mut handles = entities.clone();
        Rng::new(11).shuffle(&mut handles);
        handles.truncate(LOOKUPS);
        bench.run(&format!("engine/world_get/{label}"), LOOKUPS as u64, || {
            let mut acc = 0.0f32;
            for &e in &handles {
                if let Some(p) = world.get::<Prop>(e) {
                    acc += p.position[0];
                }
            }
            acc
        });

        bench.run(&format!("engine/world_write/{label}"), n as u64, || {
            for p in world.query_mut::<Prop>() {
                p.position[1] += 1.0;
            }
        });

        // Draining the column despawns every owner whose last component it
        // was, so this is the teardown half of a scene swap.
        bench.run(&format!("engine/world_drain/{label}"), n as u64, || {
            let (mut world, _) = populated(n);
            world.remove_all::<Prop>();
            world.component_count()
        });
    }
}
