//! Benchmarks over the engine's rigid-body simulation. Solver cost under
//! sustained contacts, the sleeping-island idle path, contact-free
//! integration, world construction, body churn, and the query paths
//! (raycast, shape cast, character move). The whole-frame probe cannot hold
//! this number still -- an un-settled solver sees a different contact set
//! every session -- so the stepping benches rebuild the same world and step it
//! the same fixed count every iteration, making the measured work
//! bit-identical run to run. Fixtures mirror the CPU stress world's physics
//! axis: stacked columns of bouncing cuboids that never sleep.
//!
//! `physics/step_ccd/*` is the continuous-collision stage's worst case: every
//! body in the world moving fast enough to be swept on every step, which no
//! ordinary world does. The stacked entries beside it are the control -- the
//! gate must keep them out of that path entirely, so their numbers must not
//! move when the stage changes.
//!
//! The simulation reserves every buffer at construction and a query keeps one
//! hit rather than a list, so `physics/step_settled/*`, `physics/raycast/1k`,
//! `physics/character_move/1`, `physics/step_joints/*`,
//! `physics/step_sensors/*`, `physics/step_contacts/*`, and every
//! `physics/*terrain*` entry must all report zero allocations per item.
//! Joints, height grids and regions are built before the measured loop begins,
//! which is the only place any of them allocates. The two event entries drain
//! into a caller's buffer reserved up front, which is what a driver on a fixed
//! tick does.
//!
//! Every query is asked on a stepped world, which is the state the driver asks
//! one in: the sweep order is current, so a traversal is a window over the
//! sorted proxies rather than a walk over all of them.
//!
//! The stepping entries are measured on the engine's job pool, which is what
//! the driver steps on. Each contact- or joint-bound one has a `_serial` twin
//! stepped on the calling thread, and the pair is the scaling report:
//! what a lent pool is worth on this machine, and what the split machinery
//! costs a caller that lends nothing. The two must land in the same place,
//! which `assert_deterministic` checks rather than assumes.
//!
//! Run with `cargo bench -p concinnity-bench --bench physics`.

use concinnity_bench::Bench;
use concinnity_cpu::jobs::{self, JobPool};
use concinnity_physics::{
    BodyHandle, CharacterMoveInput, ColliderShape, DynamicParams, Fanout, Inline, JointMotor,
    JointSpec, LayerMask, ShapeCast, SimConfig, Simulation,
};

/// The engine's job pool, lent to a stepping simulation the way the driver
/// lends it.
struct Pool(&'static JobPool);

impl Fanout for Pool {
    fn workers(&self) -> usize {
        self.0.thread_count()
    }

    fn scope<R, F>(&self, work: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.0.install(work)
    }

    fn for_each<T, F>(&self, items: &mut [T], body: F)
    where
        T: Send,
        F: Fn(&mut T) + Send + Sync,
    {
        self.0.parallel_for(items, body);
    }
}

fn pooled() -> Pool {
    Pool(jobs::pool())
}

// Reserve the per-worker scratch for the pool a bench may lend, the way the
// driver does at init. A world stepped serially keeps the reservation and
// never touches it.
fn reserve(sim: &mut Simulation) {
    sim.reserve_workers(jobs::pool().thread_count());
}

// The capsule the character-move benches drive, and where it starts: between
// two stacks, a little above the floor, so the move has ground to find.
const CHARACTER_HALF_HEIGHT: f32 = 0.6;
const CHARACTER_RADIUS: f32 = 0.3;
const CHARACTER_CENTER: [f32; 3] = [1.0, 1.1, 1.0];
const CHARACTER_STEP: [f32; 3] = [0.05, 0.0, 0.02];

const TICK: f32 = 1.0 / 60.0;
// Steps per iteration for the deterministic drop benches: enough to cover
// the fall, the impacts, and sustained stack jostle.
const DROP_STEPS: usize = 60;
const SIZES: [(usize, &str); 2] = [(256, "256"), (1024, "1k")];
const STACK: usize = 8;
const CUBE: ColliderShape = ColliderShape::Cuboid {
    half_extents: [0.4, 0.4, 0.4],
};
// Small enough that a tick of ordinary falling outruns it, which is what the
// continuous-collision fixture needs.
const PELLET: ColliderShape = ColliderShape::Ball { radius: 0.1 };

fn cube_params(restitution: f32, damping: f32) -> DynamicParams {
    DynamicParams {
        mass: 1.0,
        friction: 0.4,
        restitution,
        gravity_scale: 1.0,
        linear_damping: damping,
    }
}

// Stacked columns of dynamic cuboids over a fixed floor. High restitution and
// zero damping keep the stacks jostling (the stress-world shape); low
// restitution with damping lets them settle and sleep. `spare` is the room
// left over for the bodies a bench adds to the built fixture.
fn stacked_world(
    bodies: usize,
    restitution: f32,
    damping: f32,
    floor: bool,
    spare: usize,
) -> (Simulation, Vec<BodyHandle>) {
    let mut sim = Simulation::new(
        SimConfig {
            gravity: 9.81,
            ..SimConfig::default()
        },
        bodies + 1 + spare,
    );
    if floor {
        sim.add_fixed(
            &ColliderShape::Cuboid {
                half_extents: [200.0, 0.5, 200.0],
            },
            [0.0, -0.5, 0.0],
            [0.0; 3],
            0.8,
            LayerMask::ALL,
        )
        .expect("room for the floor");
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
            sim.add_dynamic(
                &CUBE,
                pos,
                [0.0; 3],
                cube_params(restitution, damping),
                LayerMask::ALL,
            )
            .expect("room for a body")
        })
        .collect();
    reserve(&mut sim);
    (sim, handles)
}

// A chain of hinged links hanging off a fixed post, which is the fixture the
// joint solver is measured on: every link is in the same island, so a step
// costs a full pass over the whole chain rather than over isolated pairs.
fn jointed_world(chains: usize, links: usize) -> (Simulation, Vec<BodyHandle>) {
    let mut sim = Simulation::new(
        SimConfig {
            gravity: 9.81,
            allow_sleep: false,
            ..SimConfig::default()
        },
        chains * (links + 1),
    );
    let mut handles = Vec::with_capacity(chains * links);
    for chain in 0..chains {
        let z = chain as f32 * 2.0;
        let post = sim
            .add_fixed(
                &ColliderShape::Ball { radius: 0.05 },
                [0.0, 8.0, z],
                [0.0; 3],
                0.5,
                LayerMask::ALL,
            )
            .expect("room for the post");
        let mut previous = post;
        for link in 0..links {
            let body = sim
                .add_dynamic(
                    &CUBE,
                    [(link + 1) as f32 * 0.9, 8.0, z],
                    [0.0; 3],
                    cube_params(0.0, 0.0),
                    LayerMask::ALL,
                )
                .expect("room for a link");
            let anchor_a = if link == 0 {
                [0.0; 3]
            } else {
                [0.45, 0.0, 0.0]
            };
            assert!(
                sim.add_joint(
                    previous,
                    body,
                    anchor_a,
                    [-0.45, 0.0, 0.0],
                    JointSpec::Revolute {
                        axis: [0.0, 0.0, 1.0],
                        limits: Some([-1.2, 1.2]),
                        motor: Some(JointMotor {
                            target_velocity: 1.0,
                            max_force: 4.0,
                        }),
                    },
                ),
                "the joint has to be made"
            );
            previous = body;
            handles.push(body);
        }
    }
    reserve(&mut sim);
    (sim, handles)
}

// Rolling terrain with a grid of boxes settled on it: the fixture both the
// terrain narrow phase and the terrain queries are measured on.
fn terrain_world(bodies: usize) -> (Simulation, Vec<BodyHandle>) {
    const SIDE: usize = 65;
    const EXTENT: f32 = 200.0;
    let mut sim = Simulation::new(
        SimConfig {
            gravity: 9.81,
            ..SimConfig::default()
        },
        // One spare slot for the character capsule the move bench adds.
        bodies + 2,
    );
    let mut heights = Vec::with_capacity(SIDE * SIDE);
    for row in 0..SIDE {
        let z = row as f32 / (SIDE - 1) as f32;
        for col in 0..SIDE {
            let x = col as f32 / (SIDE - 1) as f32;
            heights.push((x * 9.0).sin() * 1.5 + (z * 7.0).cos() * 1.2);
        }
    }
    sim.add_heightfield(
        SIDE,
        SIDE,
        heights,
        [EXTENT, 1.0, EXTENT],
        [0.0; 3],
        LayerMask::ALL,
    )
    .expect("room for the terrain");

    let per_row = (bodies as f32).sqrt().ceil().max(1.0) as usize;
    let handles = (0..bodies)
        .map(|i| {
            let x = (i % per_row) as f32 * 1.5 - 40.0;
            let z = (i / per_row) as f32 * 1.5 - 40.0;
            sim.add_dynamic(
                &CUBE,
                [x, 8.0, z],
                [0.0; 3],
                cube_params(0.0, 0.5),
                LayerMask::ALL,
            )
            .expect("room for a body")
        })
        .collect();
    reserve(&mut sim);
    (sim, handles)
}

// The stacked fixture with a sensor region standing over each column, which
// is what a world of authored trigger volumes looks like from the simulation's
// side: every resting body is inside one, and no crossing happens again after
// they have all settled.
fn sensor_world(bodies: usize) -> (Simulation, Vec<BodyHandle>) {
    let columns = bodies.div_ceil(STACK);
    let per_row = (columns as f32).sqrt().ceil().max(1.0) as usize;
    let mut sim = Simulation::new(
        SimConfig {
            gravity: 9.81,
            ..SimConfig::default()
        },
        bodies + columns + 1,
    );
    sim.add_fixed(
        &ColliderShape::Cuboid {
            half_extents: [200.0, 0.5, 200.0],
        },
        [0.0, -0.5, 0.0],
        [0.0; 3],
        0.8,
        LayerMask::ALL,
    )
    .expect("room for the floor");
    for column in 0..columns {
        let (x, z) = (
            (column % per_row) as f32 * 2.0,
            (column / per_row) as f32 * 2.0,
        );
        sim.add_sensor(
            &ColliderShape::Cuboid {
                half_extents: [0.6, 4.5, 0.6],
            },
            [x, 4.0, z],
            [0.0; 3],
            column as u64,
            LayerMask::ALL,
        )
        .expect("room for a region");
    }
    let handles = (0..bodies)
        .map(|i| {
            let (column, level) = (i / STACK, i % STACK);
            let pos = [
                (column % per_row) as f32 * 2.0,
                0.5 + level as f32 * 1.05,
                (column / per_row) as f32 * 2.0,
            ];
            sim.add_dynamic(&CUBE, pos, [0.0; 3], cube_params(0.0, 0.5), LayerMask::ALL)
                .expect("room for a body")
        })
        .collect();
    reserve(&mut sim);
    (sim, handles)
}

// A grid of small bodies falling at sixty units a second onto a slab a tenth
// of a unit thick: one unit of travel per tick against a tenth of a unit of
// surface, so every body arms the sweep on every step of the fall. Started
// high enough that most of the measured window is the sweep itself rather
// than the contacts afterwards.
fn hail_world(bodies: usize) -> (Simulation, Vec<BodyHandle>) {
    let mut sim = Simulation::new(
        SimConfig {
            gravity: 9.81,
            ..SimConfig::default()
        },
        bodies + 1,
    );
    sim.add_fixed(
        &ColliderShape::Cuboid {
            half_extents: [200.0, 0.05, 200.0],
        },
        [0.0; 3],
        [0.0; 3],
        0.8,
        LayerMask::ALL,
    )
    .expect("room for the slab");
    let per_row = (bodies as f32).sqrt().ceil().max(1.0) as usize;
    let handles = (0..bodies)
        .map(|i| {
            let x = (i % per_row) as f32 * 0.5;
            let z = (i / per_row) as f32 * 0.5;
            let body = sim
                .add_dynamic(
                    &PELLET,
                    [x, 40.0 + (i % 4) as f32 * 0.5, z],
                    [0.0; 3],
                    cube_params(0.0, 0.0),
                    LayerMask::ALL,
                )
                .expect("room for a body");
            sim.set_linear_velocity(body, [0.0, -60.0, 0.0]);
            body
        })
        .collect();
    reserve(&mut sim);
    (sim, handles)
}

fn poses(sim: &Simulation, handles: &[BodyHandle]) -> Vec<([f32; 3], [f32; 3])> {
    handles
        .iter()
        .map(|&h| sim.body_pose(h).expect("live"))
        .collect()
}

// The premise the stepping benches rest on: identical worlds stepped the
// same count report bit-identical poses. If a change to the simulation breaks
// this, fail loudly instead of quietly turning into a noisy benchmark.
fn assert_deterministic() {
    let stacked = |fanout: &dyn Fn(&mut Simulation)| {
        let (mut sim, handles) = stacked_world(256, 0.85, 0.0, true, 1);
        for _ in 0..120 {
            fanout(&mut sim);
        }
        poses(&sim, &handles)
    };
    let on_pool = |sim: &mut Simulation| sim.step_with(TICK, &pooled());
    let alone = |sim: &mut Simulation| sim.step_with(TICK, &Inline);
    assert_eq!(
        stacked(&on_pool),
        stacked(&on_pool),
        "the physics step is no longer deterministic"
    );
    // The claim the whole fan-out rests on: what a step produces cannot depend
    // on how many workers it was handed, so this is `==` on the raw bits and
    // not a tolerance.
    assert_eq!(
        stacked(&alone),
        stacked(&on_pool),
        "a split step no longer lands where a serial one does"
    );

    let jointed = |fanout: &dyn Fn(&mut Simulation)| {
        let (mut sim, handles) = jointed_world(8, 6);
        for _ in 0..120 {
            fanout(&mut sim);
        }
        poses(&sim, &handles)
    };
    assert_eq!(
        jointed(&on_pool),
        jointed(&on_pool),
        "the joint solve is no longer deterministic"
    );
    assert_eq!(
        jointed(&alone),
        jointed(&on_pool),
        "a split joint solve no longer lands where a serial one does"
    );

    // A scene the sweep is doing real work in: candidate order, impact
    // resolution and the stop itself all have to be repeatable, and none of
    // them is exercised by a world slow enough to stay off that path.
    let hail = || {
        let (mut sim, handles) = hail_world(256);
        for _ in 0..90 {
            sim.step_with(TICK, &pooled());
        }
        assert_eq!(sim.ccd_overflows(), 0, "the sweep was declined");
        poses(&sim, &handles)
    };
    assert_eq!(
        hail(),
        hail(),
        "the continuous-collision step is no longer deterministic"
    );

    let terrain = |fanout: &dyn Fn(&mut Simulation)| {
        let (mut sim, handles) = terrain_world(256);
        for _ in 0..120 {
            fanout(&mut sim);
        }
        poses(&sim, &handles)
    };
    assert_eq!(
        terrain(&on_pool),
        terrain(&on_pool),
        "the terrain step is no longer deterministic"
    );
    assert_eq!(
        terrain(&alone),
        terrain(&on_pool),
        "a split terrain step no longer lands where a serial one does"
    );

    // What a step reports has to be as repeatable as where it leaves a body:
    // a crossing sequence or a hit order that shifts between runs is a hash
    // or a set that crept onto the step path.
    let reported = |fanout: &dyn Fn(&mut Simulation)| {
        let (mut sim, _) = sensor_world(256);
        sim.set_contact_min_impulse(2.0, TICK);
        let (mut crossings, mut hits) = (Vec::new(), Vec::new());
        let mut recorded = Vec::new();
        for _ in 0..120 {
            fanout(&mut sim);
            sim.drain_sensor_crossings_into(&mut crossings);
            sim.drain_contact_hits_into(&mut hits);
            recorded.extend(crossings.iter().map(|c| (c.tag, c.entered)));
            recorded.extend(hits.iter().map(|h| (h.impulse.to_bits() as u64, false)));
        }
        assert_eq!(sim.sensor_overflows(), 0, "the sensor queue overflowed");
        assert_eq!(sim.contact_hit_overflows(), 0, "the hit queue overflowed");
        recorded
    };
    let first = reported(&on_pool);
    assert!(!first.is_empty(), "the fixture has to report something");
    assert_eq!(
        first,
        reported(&on_pool),
        "the step no longer reports deterministically"
    );
    assert_eq!(
        reported(&alone),
        first,
        "a split step no longer reports what a serial one does"
    );
}

fn main() {
    let mut bench = Bench::from_env();
    // Built before anything is measured, so the one-off cost of standing the
    // workers up is not charged to whichever bench ran first.
    let workers = jobs::pool().thread_count();
    assert!(workers >= 1);

    for (n, label) in SIZES {
        let body_steps = (n * DROP_STEPS) as u64;

        bench.run(
            &format!("physics/step_settling/{label}"),
            body_steps,
            || {
                let (mut sim, handles) = stacked_world(n, 0.85, 0.0, true, 1);
                for _ in 0..DROP_STEPS {
                    sim.step_with(TICK, &pooled());
                }
                sim.body_pose(handles[0]).expect("live").0[1].to_bits()
            },
        );

        // The same world with nothing lent: what the split is worth on this
        // machine is the distance between this entry and the one above.
        bench.run(
            &format!("physics/step_settling_serial/{label}"),
            body_steps,
            || {
                let (mut sim, handles) = stacked_world(n, 0.85, 0.0, true, 1);
                for _ in 0..DROP_STEPS {
                    sim.step_with(TICK, &Inline);
                }
                sim.body_pose(handles[0]).expect("live").0[1].to_bits()
            },
        );

        bench.run(
            &format!("physics/step_free_fall/{label}"),
            body_steps,
            || {
                let (mut sim, handles) = stacked_world(n, 0.85, 0.0, false, 1);
                for _ in 0..DROP_STEPS {
                    sim.step_with(TICK, &pooled());
                }
                sim.body_pose(handles[0]).expect("live").0[1].to_bits()
            },
        );

        bench.run(
            &format!("physics/step_free_fall_serial/{label}"),
            body_steps,
            || {
                let (mut sim, handles) = stacked_world(n, 0.85, 0.0, false, 1);
                for _ in 0..DROP_STEPS {
                    sim.step_with(TICK, &Inline);
                }
                sim.body_pose(handles[0]).expect("live").0[1].to_bits()
            },
        );

        bench.run(&format!("physics/step_ccd/{label}"), body_steps, || {
            let (mut sim, _handles) = hail_world(n);
            let mut swept = 0usize;
            for _ in 0..DROP_STEPS {
                sim.step_with(TICK, &pooled());
                swept = swept.max(sim.swept_body_count());
            }
            assert_eq!(swept, n, "every body has to have been swept");
            assert_eq!(sim.ccd_overflows(), 0, "the sweep was declined");
            swept
        });

        bench.run(&format!("physics/build_world/{label}"), n as u64, || {
            stacked_world(n, 0.85, 0.0, true, 1).1.len()
        });

        // Stepped to sleep, then measured: this is the idle path, and the
        // reservation the simulation was built with means it must report zero
        // allocations per iteration.
        let (mut settled, handles) = stacked_world(n, 0.0, 0.5, true, 1);
        for _ in 0..600 {
            settled.step_with(TICK, &pooled());
        }
        let poses_before = poses(&settled, &handles);
        bench.run(&format!("physics/step_settled/{label}"), n as u64, || {
            settled.step_with(TICK, &pooled());
            handles.len()
        });
        bench.run(
            &format!("physics/step_settled_serial/{label}"),
            n as u64,
            || {
                settled.step_with(TICK, &Inline);
                handles.len()
            },
        );
        assert_eq!(
            poses(&settled, &handles),
            poses_before,
            "settled fixture was still moving while measured"
        );
    }

    for (n, label) in SIZES {
        // Settled under regions: every resting body is inside one, which is
        // the sensor stage's worst case rather than a typical world's. The
        // measured step is a full pass over one overlapping pair per body,
        // finding nothing new to report.
        let (mut regions, handles) = sensor_world(n);
        for _ in 0..600 {
            regions.step_with(TICK, &pooled());
        }
        let poses_before = poses(&regions, &handles);
        let mut crossings = Vec::with_capacity(n);
        bench.run(&format!("physics/step_sensors/{label}"), n as u64, || {
            regions.step_with(TICK, &pooled());
            regions.drain_sensor_crossings_into(&mut crossings);
            crossings.len()
        });
        assert_eq!(
            poses(&regions, &handles),
            poses_before,
            "settled region fixture was still moving while measured"
        );
        assert_eq!(regions.sensor_overflows(), 0, "the sensor queue overflowed");
        assert!(
            regions.sensor_overlap_count() >= n,
            "every body has to be inside a region, {} of {n} were",
            regions.sensor_overlap_count()
        );

        // Jostling with the contact gate armed: bouncing stacks never settle,
        // so the measured step is a real solve with the impact pass reading
        // what it delivered and a drain that has hits to move every tick.
        //
        // Both reservations are sized past the body count rather than at it.
        // A stack of eight has more contacting pairs than bodies, and the
        // tick a whole field of them lands on reports every one of those
        // pairs at once, so a queue reserved from the body count alone would
        // measure an overflow rather than the drain.
        let (mut jostling, handles) = stacked_world(n, 0.85, 0.0, true, n);
        // Held awake rather than merely bouncy. A stack this lively still
        // settles eventually, and a fixture that falls asleep partway through
        // the measured window reports how fast it settled instead of what a
        // step under sustained contact costs.
        jostling.set_config(SimConfig {
            allow_sleep: false,
            ..*jostling.config()
        });
        jostling.set_contact_min_impulse(5.0, TICK);
        for _ in 0..60 {
            jostling.step_with(TICK, &pooled());
        }
        let mut hits = Vec::with_capacity(n * 4);
        let mut reported = 0usize;
        bench.run(&format!("physics/step_contacts/{label}"), n as u64, || {
            jostling.step_with(TICK, &pooled());
            jostling.drain_contact_hits_into(&mut hits);
            reported += hits.len();
            hits.len()
        });
        bench.run(
            &format!("physics/step_contacts_serial/{label}"),
            n as u64,
            || {
                jostling.step_with(TICK, &Inline);
                jostling.drain_contact_hits_into(&mut hits);
                reported += hits.len();
                hits.len()
            },
        );
        assert!(reported > 0, "the fixture reported no impacts to measure");
        assert!(
            handles
                .iter()
                .all(|&h| jostling.is_sleeping(h) == Some(false)),
            "the jostling fixture settled while it was being measured"
        );
        assert_eq!(
            jostling.contact_hit_overflows(),
            0,
            "the hit queue overflowed"
        );
    }

    for (n, label) in [(64usize, "64"), (256, "256")] {
        // A chain of hinged links never settles while its motors are driving
        // it, so the measured step is a real joint solve every iteration.
        let chains = n / 8;
        let (mut jointed, handles) = jointed_world(chains, 8);
        for _ in 0..60 {
            jointed.step_with(TICK, &pooled());
        }
        assert_eq!(jointed.joint_count(), chains * 8);
        bench.run(
            &format!("physics/step_joints/{label}"),
            handles.len() as u64,
            || {
                jointed.step_with(TICK, &pooled());
                handles.len()
            },
        );
        bench.run(
            &format!("physics/step_joints_serial/{label}"),
            handles.len() as u64,
            || {
                jointed.step_with(TICK, &Inline);
                handles.len()
            },
        );

        // Settled on terrain: the idle path with a height grid under it, and
        // the one that must report no allocation now that a grid can hand a
        // pair several manifolds.
        let (mut terrain, handles) = terrain_world(n);
        for _ in 0..600 {
            terrain.step_with(TICK, &pooled());
        }
        let poses_before = poses(&terrain, &handles);
        bench.run(&format!("physics/step_terrain/{label}"), n as u64, || {
            terrain.step_with(TICK, &pooled());
            handles.len()
        });
        assert_eq!(
            poses(&terrain, &handles),
            poses_before,
            "settled terrain fixture was still moving while measured"
        );
        assert_eq!(
            terrain.heightfield_overflows(),
            0,
            "the terrain step gave up on part of the surface"
        );
    }

    {
        // Terrain queries, on a world stepped once so the sweep order is
        // current. A ray walks the grid cell by cell and a sweep clips against
        // the cells its swept box names, so neither allocates.
        let (mut sim, _handles) = terrain_world(256);
        sim.step_with(TICK, &pooled());
        const TERRAIN_RAYS: usize = 1024;
        let fan = |sim: &Simulation| {
            let mut hits = 0u32;
            for i in 0..TERRAIN_RAYS {
                let angle = i as f32 * core::f32::consts::TAU / TERRAIN_RAYS as f32;
                let hit = sim.raycast(
                    [angle.cos() * 60.0, 30.0, angle.sin() * 60.0],
                    [0.0, -1.0, 0.0],
                    100.0,
                    None,
                    LayerMask::ALL,
                );
                hits += hit.is_some() as u32;
            }
            hits
        };
        assert!(fan(&sim) > 0, "the fan has to meet the terrain it measures");
        bench.run("physics/terrain_raycast/1k", TERRAIN_RAYS as u64, || {
            fan(&sim)
        });

        const SWEEPS: usize = 256;
        let capsule = ColliderShape::Capsule {
            half_height: CHARACTER_HALF_HEIGHT,
            radius: CHARACTER_RADIUS,
        };
        let drops = |sim: &Simulation| {
            let mut hits = 0u32;
            for i in 0..SWEEPS {
                let angle = i as f32 * core::f32::consts::TAU / SWEEPS as f32;
                let hit = sim.shape_cast(&ShapeCast::new(
                    capsule,
                    [angle.cos() * 60.0, 30.0, angle.sin() * 60.0],
                    [0.0, -40.0, 0.0],
                ));
                hits += hit.is_some() as u32;
            }
            hits
        };
        assert!(drops(&sim) > 0, "the sweeps have to meet the terrain");
        bench.run("physics/terrain_sweep/256", SWEEPS as u64, || drops(&sim));
        assert_eq!(
            sim.heightfield_overflows(),
            0,
            "a measured terrain query gave up on part of the surface"
        );
    }

    {
        // Stepped once so the sweep order is current: that is the state a
        // query is asked in, and it is what makes the traversal a window over
        // the sorted proxies rather than a walk over all of them.
        let (mut sim, _handles) = stacked_world(1024, 0.0, 0.5, true, 1);
        sim.step_with(TICK, &pooled());
        const RAYS: usize = 1024;
        let fan = |sim: &Simulation| {
            let mut hits = 0u32;
            for i in 0..RAYS {
                let angle = i as f32 * core::f32::consts::TAU / RAYS as f32;
                let hit = sim.raycast(
                    [12.0, 6.0, 12.0],
                    [angle.cos(), -0.6, angle.sin()],
                    100.0,
                    None,
                    LayerMask::ALL,
                );
                hits += hit.is_some() as u32;
            }
            hits
        };
        assert!(fan(&sim) > 0, "the fan has to meet the world it measures");
        bench.run("physics/raycast/1k", RAYS as u64, || fan(&sim));

        let capsule = sim
            .add_kinematic(
                &ColliderShape::Capsule {
                    half_height: CHARACTER_HALF_HEIGHT,
                    radius: CHARACTER_RADIUS,
                },
                CHARACTER_CENTER,
                [0.0; 3],
                0.8,
                LayerMask::ALL,
            )
            .expect("room for the capsule");
        // Stepped again so the sweep order is current with the capsule in it:
        // the driver asks for a move between steps, never on a world whose
        // broad phase has an unsorted body in it.
        sim.step_with(TICK, &pooled());
        let shape = Simulation::character_shape(CHARACTER_HALF_HEIGHT, CHARACTER_RADIUS);
        let walk = |sim: &Simulation| {
            sim.move_character(
                &shape,
                &CharacterMoveInput {
                    center: CHARACTER_CENTER,
                    desired: CHARACTER_STEP,
                    dt: TICK,
                    exclude: capsule,
                    mask: LayerMask::ALL,
                },
            )
        };
        assert!(
            walk(&sim).grounded,
            "the move has to find the floor it measures"
        );
        bench.run("physics/character_move/1", 1, || walk(&sim).grounded);
    }

    {
        // Runtime spawn and despawn: the same bodies added and removed against
        // a settled world, on a reservation that has room for them. The pool
        // hands back the slots it just freed, so an iteration ends where it
        // began and none of it allocates.
        const CHURN: usize = 64;
        let (mut sim, _handles) = stacked_world(1024, 0.0, 0.5, true, CHURN);
        sim.step_with(TICK, &pooled());
        let mut churned = Vec::with_capacity(CHURN);
        bench.run("physics/body_churn/64", CHURN as u64, || {
            for i in 0..CHURN {
                churned.push(
                    sim.add_dynamic(
                        &CUBE,
                        [80.0 + (i % 8) as f32, 4.0 + (i / 8) as f32 * 1.1, 80.0],
                        [0.0; 3],
                        cube_params(0.2, 0.0),
                        LayerMask::ALL,
                    )
                    .expect("room for a churned body"),
                );
            }
            let added = churned.len();
            for handle in churned.drain(..) {
                assert!(sim.remove_body(handle), "a churned body has to come out");
            }
            added
        });
    }

    assert_deterministic();
    bench.finish();
}
