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
// The `swarm` row is the neighbor-query load: a dense drift of scoped tokens,
// each asking three rules' `nearest` and `count_within` of every prop in a long
// corridor. Its per-run cost should track the neighbors a token has, not the
// props the corridor holds.
//
//     cargo test -p concinnity-core --release -- --ignored --nocapture \
//         --test-threads=1 behavior_tick

use alloc::boxed::Box;
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use std::println;

use super::BehaviorSystem;
use super::eval::Snapshot;
use super::test_world::world_with;
use crate::components::{
    Behavior, BehaviorExpr, BehaviorNode, BehaviorQuery, BehaviorSource, Interactable,
    PropInstance, Transform,
};
use crate::ecs::{System, World};
use crate::test_support::{Pace, bench};

const BEHAVIORS: usize = 256;
const SMALL_WORLD: usize = 1_000;
const LARGE_WORLD: usize = 20_000;

// How many behaviors and how large the two worlds are, per pace. A single-run
// pass drives the same shapes at a fraction of the size: it proves the fixtures
// still build and tick, and compares nothing, so paying for 20k entities in an
// unoptimized build would buy it nothing.
fn fixture(pace: Pace) -> (usize, [(usize, &'static str); 2]) {
    match pace {
        Pace::Timed => (BEHAVIORS, [(SMALL_WORLD, "1k"), (LARGE_WORLD, "20k")]),
        Pace::Once => (4, [(8, "8"), (16, "16")]),
    }
}

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
fn padded_world(behaviors: usize, props: usize) -> World {
    let mut world = world_with((0..behaviors).map(counter).collect::<Vec<_>>());
    for i in 0..props {
        let entity = world.push(PropInstance);
        world.insert(
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

// The swarm fixture's shape: tokens along each axis and the props padding the
// corridor around them, per pace.
fn swarm_fixture(pace: Pace) -> ([usize; 3], usize) {
    match pace {
        Pace::Timed => ([12, 6, 10], 1_280),
        Pace::Once => ([3, 2, 2], 16),
    }
}

const SWARM_SPACING: f32 = 1.5;
const CORRIDOR_LENGTH: f32 = 640.0;

// One swarm rule: find the nearest prop, count the props within `radius`, and
// record whether the crowd passes `threshold`. It records rather than moves, so
// every timed tick asks the same questions of the same layout.
fn swarm_rule(k: usize, radius: f32, threshold: i32) -> Behavior {
    let here = || Box::new(BehaviorExpr::Position(Box::new(BehaviorExpr::SelfEntity)));
    Behavior {
        on: BehaviorSource::Tick,
        scope: vec!["Interactable".into()],
        queries: vec![BehaviorQuery {
            name: "nearby".into(),
            has: vec!["Prop".into()],
        }],
        body: vec![
            BehaviorNode::Let {
                name: "closest".into(),
                value: BehaviorExpr::Nearest {
                    query: "nearby".into(),
                    of: here(),
                },
            },
            BehaviorNode::Let {
                name: "crowd".into(),
                value: BehaviorExpr::CountWithin {
                    query: "nearby".into(),
                    of: here(),
                    radius: Box::new(BehaviorExpr::Float(radius)),
                },
            },
            BehaviorNode::If {
                cond: BehaviorExpr::All(vec![
                    BehaviorExpr::Alive(Box::new(BehaviorExpr::Bind("closest".into()))),
                    BehaviorExpr::Gt(
                        Box::new(BehaviorExpr::Bind("crowd".into())),
                        Box::new(BehaviorExpr::Int(threshold)),
                    ),
                ]),
                then: vec![BehaviorNode::Set {
                    var: format!("crowded{k}"),
                    value: BehaviorExpr::Int(1),
                    add: true,
                }],
                otherwise: Vec::new(),
            },
        ],
        ..Default::default()
    }
}

// A drift of interactable tokens at the middle of a corridor of props, and the
// three rules the benchmark world's swarm station runs against it.
fn swarm_world(drift: [usize; 3], padding: usize) -> World {
    let mut world = world_with(vec![
        swarm_rule(0, 4.5, 6),
        swarm_rule(1, 4.5, 6),
        swarm_rule(2, 1.1, 0),
    ]);
    let place = |world: &mut World, position: [f32; 3]| {
        let entity = world.push(PropInstance);
        world.insert(
            entity,
            Transform {
                position,
                rotation_deg: [0.0; 3],
                scale: [1.0; 3],
            },
        );
        entity
    };
    let middle = CORRIDOR_LENGTH * 0.5;
    for x in 0..drift[0] {
        for y in 0..drift[1] {
            for z in 0..drift[2] {
                let position = [
                    middle + x as f32 * SWARM_SPACING,
                    2.0 + y as f32 * SWARM_SPACING,
                    z as f32 * SWARM_SPACING,
                ];
                let entity = place(&mut world, position);
                world.insert(entity, Interactable);
            }
        }
    }
    for i in 0..padding {
        let x = CORRIDOR_LENGTH * i as f32 / padding as f32;
        let side = if i.is_multiple_of(2) { -6.0 } else { 6.0 };
        place(&mut world, [x, 0.0, side]);
    }
    world
}

fn started(world: &mut World) -> BehaviorSystem {
    let mut sys = BehaviorSystem::new();
    sys.init(&mut world.context());
    sys
}

fn run(pace: Pace) {
    let (behaviors, worlds) = fixture(pace);
    if pace == Pace::Timed {
        println!("\nbehavior evaluation ({behaviors} tick behaviors)");
    }

    // The two rows that matter: same behaviors, 20x the world. If these
    // diverge, the tick has gone back to scanning the world.
    for (props, label) in worlds {
        let mut world = padded_world(behaviors, props);
        let mut sys = started(&mut world);
        let mut elapsed = 0.0f32;
        bench(
            pace,
            &format!("tick_world{label}"),
            behaviors as u64,
            || {
                elapsed += 0.016;
                sys.tick(&mut world.context(), 0.016, elapsed);
            },
        );
    }

    // `gather` on its own, at the same two world sizes: the snapshot build is
    // the half that used to carry the world-size term.
    for (props, label) in worlds {
        let mut world = padded_world(behaviors, props);
        let mut sys = started(&mut world);
        let mut snapshot = Snapshot::default();
        bench(
            pace,
            &format!("gather_world{label}"),
            behaviors as u64,
            || sys.gather(&world.context(), &mut snapshot),
        );
    }

    // Per run: one token's pass through one rule.
    let (drift, padding) = swarm_fixture(pace);
    let runs = (drift.iter().product::<usize>() * 3) as u64;
    if pace == Pace::Timed {
        println!("\nneighbor queries ({runs} runs over {padding} padding props)");
    }
    let mut world = swarm_world(drift, padding);
    let mut sys = started(&mut world);
    let mut elapsed = 0.0f32;
    bench(pace, "swarm_tick", runs, || {
        elapsed += 0.016;
        sys.tick(&mut world.context(), 0.016, elapsed);
    });
}

#[test]
#[ignore = "microbench; run it by name with --ignored --nocapture --test-threads=1"]
fn behavior_tick() {
    run(Pace::Timed);
}

#[test]
fn behavior_tick_fixtures_build_and_run() {
    run(Pace::Once);
}
