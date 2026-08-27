// BehaviorSystem's per-frame cost. This lives beside the system rather than
// under a bench target because `gather` is private: it is measured directly,
// and it is the reason these benchmarks exist. `gather` used to build
// whole-world ordered containers every tick, which made BehaviorSystem's cost a
// function of world size rather than of how many behaviors were declared;
// scoping it to what the bodies can actually read took 23-31x off the tick.
//
// The guard against that returning is the pair of `tick` rows at one behavior
// count and two world sizes: they must stay close. A cost that tracks the world
// instead of the behaviors is the regression.
//
//     cargo test -p concinnity-core --release -- --ignored --nocapture \
//         --test-threads=1 behavior_tick

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use std::println;

use super::BehaviorSystem;
use super::eval::Snapshot;
use super::test_world::{TestWorld, world_with};
use crate::components::{
    Behavior, BehaviorExpr, BehaviorNode, BehaviorSource, PropInstance, Transform,
};
use crate::ecs::System;
use crate::test_support::bench;

const BEHAVIORS: usize = 256;
const SMALL_WORLD: usize = 1_000;
const LARGE_WORLD: usize = 20_000;

// A world-scoped tick behavior that only does arithmetic on its own variable,
// so the measurement is evaluation cost and never transform writes.
fn counter(i: usize) -> Behavior {
    Behavior {
        on: BehaviorSource::Tick,
        body: vec![BehaviorNode::Set {
            var: format!("acc{i}"),
            value: BehaviorExpr::Int(1),
            add: true,
        }],
        ..Default::default()
    }
}

// `behaviors` tick behaviors over a world padded to `props` entities, each
// carrying a Transform (what `gather` would have been scanning).
fn padded_world(behaviors: usize, props: usize) -> TestWorld {
    let mut world = world_with((0..behaviors).map(counter).collect::<Vec<_>>());
    for i in 0..props {
        let entity = world.components.push_typed(PropInstance);
        world.components.insert_typed(
            entity,
            Transform {
                position: [i as f32, 0.0, 0.0],
                rotation_deg: [0.0; 3],
                scale: [1.0; 3],
            },
        );
    }
    world
}

fn started(world: &mut TestWorld) -> BehaviorSystem {
    let mut sys = BehaviorSystem::new();
    sys.init(&mut world.ctx());
    sys
}

#[test]
#[ignore = "microbench; run it by name with --ignored --nocapture --test-threads=1"]
fn behavior_tick() {
    println!("\nbehavior evaluation ({BEHAVIORS} tick behaviors)");

    // The two rows that matter: same behaviors, 20x the world. If these
    // diverge, the tick has gone back to scanning the world.
    for (props, label) in [(SMALL_WORLD, "1k"), (LARGE_WORLD, "20k")] {
        let mut world = padded_world(BEHAVIORS, props);
        let mut sys = started(&mut world);
        let mut elapsed = 0.0f32;
        bench(&format!("tick_world{label}"), BEHAVIORS as u64, || {
            elapsed += 0.016;
            sys.tick(&mut world.ctx(), 0.016, elapsed);
        });
    }

    // `gather` on its own, at the same two world sizes: the snapshot build is
    // the half that used to carry the world-size term.
    for (props, label) in [(SMALL_WORLD, "1k"), (LARGE_WORLD, "20k")] {
        let mut world = padded_world(BEHAVIORS, props);
        let mut sys = started(&mut world);
        let mut snapshot = Snapshot::default();
        bench(&format!("gather_world{label}"), BEHAVIORS as u64, || {
            sys.gather(&world.ctx(), &mut snapshot)
        });
    }
}
