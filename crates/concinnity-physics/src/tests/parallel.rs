//! A step handed real threads must land where the same step on one thread
//! lands, to the bit.
//!
//! That is the whole licence for the fan-out. The stages a step offers are
//! split so their results cannot depend on who did what -- islands share no
//! body the solve moves, narrow-phase ranges write their own buffers and are
//! appended in range order, and the sweep's pairs are sorted by slot whichever
//! range found them -- so the comparison below is `==` on the raw bits rather
//! than a tolerance. A tolerance would pass on a solve that had quietly become
//! order dependent, which is exactly the failure this file exists to catch.
//!
//! One thread per work unit rather than a pool: it maximises the interleaving
//! and it is what makes this file worth running under a thread sanitizer.

use crate::{
    BodyHandle, ColliderShape, DynamicParams, Fanout, JointMotor, JointSpec, LayerMask, SimConfig,
    Simulation,
};
use alloc::vec::Vec;

const TICK: f32 = 1.0 / 60.0;
const WORKERS: usize = 8;
const CUBE: ColliderShape = ColliderShape::Cuboid {
    half_extents: [0.4, 0.4, 0.4],
};

/// A fan-out that gives every unit of work its own thread.
struct Threads(usize);

impl Fanout for Threads {
    fn workers(&self) -> usize {
        self.0
    }

    fn for_each<T, F>(&self, items: &mut [T], body: F)
    where
        T: Send,
        F: Fn(&mut T) + Send + Sync,
    {
        if items.len() < 2 {
            items.iter_mut().for_each(body);
            return;
        }
        let body = &body;
        std::thread::scope(|scope| {
            for item in items.iter_mut() {
                scope.spawn(move || body(item));
            }
        });
    }
}

fn params(restitution: f32, damping: f32) -> DynamicParams {
    DynamicParams {
        mass: 1.0,
        friction: 0.4,
        restitution,
        gravity_scale: 1.0,
        linear_damping: damping,
    }
}

fn config() -> SimConfig {
    SimConfig {
        gravity: 9.81,
        ..SimConfig::default()
    }
}

/// Where every body ended up, as raw bits. Rotations are taken as quaternions
/// rather than Euler angles so the comparison reads the state the step left
/// rather than a conversion of it.
fn poses(sim: &Simulation, bodies: &[BodyHandle]) -> Vec<[u32; 7]> {
    bodies
        .iter()
        .map(|&handle| {
            let (position, rotation) = sim.body_pose_quat(handle).expect("a live body");
            [
                position[0].to_bits(),
                position[1].to_bits(),
                position[2].to_bits(),
                rotation[0].to_bits(),
                rotation[1].to_bits(),
                rotation[2].to_bits(),
                rotation[3].to_bits(),
            ]
        })
        .collect()
}

/// Step `build`'s world `steps` times, on the given number of workers, and
/// report where every body ended up.
fn walked(
    build: impl Fn() -> (Simulation, Vec<BodyHandle>),
    steps: usize,
    workers: usize,
) -> Vec<[u32; 7]> {
    let (mut sim, bodies) = build();
    sim.reserve_workers(workers);
    let fanout = Threads(workers);
    for _ in 0..steps {
        sim.step_with(TICK, &fanout);
    }
    poses(&sim, &bodies)
}

/// Columns of jostling cuboids over a shared floor: many islands, all leaning
/// on the same immovable body, which is the shape the split is built for.
fn stacked(bodies: usize, restitution: f32, damping: f32) -> (Simulation, Vec<BodyHandle>) {
    const STACK: usize = 8;
    let mut sim = Simulation::new(config(), bodies + 1);
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
    let columns = bodies.div_ceil(STACK);
    let per_row = (columns as f32).sqrt().ceil().max(1.0) as usize;
    let handles = (0..bodies)
        .map(|i| {
            let (column, level) = (i / STACK, i % STACK);
            sim.add_dynamic(
                &CUBE,
                [
                    (column % per_row) as f32 * 2.0,
                    0.5 + level as f32 * 1.05,
                    (column / per_row) as f32 * 2.0,
                ],
                [0.0; 3],
                params(restitution, damping),
                LayerMask::ALL,
            )
            .expect("room for a body")
        })
        .collect();
    (sim, handles)
}

/// Hinged chains hanging off fixed posts: the joint solve, split the same way
/// the contacts are.
fn jointed(chains: usize, links: usize) -> (Simulation, Vec<BodyHandle>) {
    let mut sim = Simulation::new(
        SimConfig {
            allow_sleep: false,
            ..config()
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
                    params(0.0, 0.0),
                    LayerMask::ALL,
                )
                .expect("room for a link");
            let anchor_a = if link == 0 {
                [0.0; 3]
            } else {
                [0.45, 0.0, 0.0]
            };
            assert!(sim.add_joint(
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
            ));
            previous = body;
            handles.push(body);
        }
    }
    (sim, handles)
}

/// Boxes dropped onto rolling terrain: the narrow phase's other branch, where
/// one pair can produce several manifolds.
fn terrain(bodies: usize) -> (Simulation, Vec<BodyHandle>) {
    const SIDE: usize = 65;
    let mut sim = Simulation::new(config(), bodies + 1);
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
        [200.0, 1.0, 200.0],
        [0.0; 3],
        LayerMask::ALL,
    )
    .expect("room for the terrain");
    let per_row = (bodies as f32).sqrt().ceil().max(1.0) as usize;
    let handles = (0..bodies)
        .map(|i| {
            sim.add_dynamic(
                &CUBE,
                [
                    (i % per_row) as f32 * 1.5 - 40.0,
                    8.0,
                    (i / per_row) as f32 * 1.5 - 40.0,
                ],
                [0.0; 3],
                params(0.0, 0.5),
                LayerMask::ALL,
            )
            .expect("room for a body")
        })
        .collect();
    (sim, handles)
}

/// Small bodies falling fast enough to be swept every step, with regions to
/// cross on the way down: the continuous-collision and sensor paths, which
/// stay on the calling thread and must not notice the ones that do not.
fn hail(bodies: usize) -> (Simulation, Vec<BodyHandle>) {
    let mut sim = Simulation::new(config(), bodies * 2 + 1);
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
    let mut handles = Vec::with_capacity(bodies);
    for i in 0..bodies {
        let (x, z) = ((i % per_row) as f32 * 0.5, (i / per_row) as f32 * 0.5);
        if i % 8 == 0 {
            sim.add_sensor(
                &ColliderShape::Cuboid {
                    half_extents: [0.3, 2.0, 0.3],
                },
                [x, 20.0, z],
                [0.0; 3],
                i as u64,
                LayerMask::ALL,
            )
            .expect("room for a region");
        }
        let body = sim
            .add_dynamic(
                &ColliderShape::Ball { radius: 0.1 },
                [x, 40.0 + (i % 4) as f32 * 0.5, z],
                [0.0; 3],
                params(0.0, 0.0),
                LayerMask::ALL,
            )
            .expect("room for a body");
        sim.set_linear_velocity(body, [0.0, -60.0, 0.0]);
        handles.push(body);
    }
    (sim, handles)
}

#[test]
fn jostling_stacks_land_in_the_same_place_on_one_thread_and_on_many() {
    let build = || stacked(512, 0.85, 0.0);
    assert_eq!(walked(build, 90, 1), walked(build, 90, WORKERS));
}

#[test]
fn a_settling_world_sleeps_the_same_way_however_it_was_split() {
    let build = || stacked(512, 0.0, 0.5);
    assert_eq!(walked(build, 300, 1), walked(build, 300, WORKERS));
}

#[test]
fn a_driven_joint_chain_is_unmoved_by_the_split() {
    let build = || jointed(16, 8);
    assert_eq!(walked(build, 120, 1), walked(build, 120, WORKERS));
}

#[test]
fn terrain_manifolds_survive_the_narrow_phase_split() {
    let build = || terrain(256);
    let (one, many) = (walked(build, 150, 1), walked(build, 150, WORKERS));
    assert_eq!(one, many);
}

#[test]
fn a_swept_world_reports_the_same_crossings_however_it_was_split() {
    let reported = |workers: usize| {
        let (mut sim, bodies) = hail(256);
        sim.reserve_workers(workers);
        sim.set_contact_min_impulse(2.0, TICK);
        let fanout = Threads(workers);
        let (mut crossings, mut hits) = (Vec::new(), Vec::new());
        let mut recorded = Vec::new();
        for _ in 0..90 {
            sim.step_with(TICK, &fanout);
            sim.drain_sensor_crossings_into(&mut crossings);
            sim.drain_contact_hits_into(&mut hits);
            recorded.extend(crossings.iter().map(|c| (c.tag, c.entered)));
            recorded.extend(hits.iter().map(|h| (u64::from(h.impulse.to_bits()), false)));
        }
        assert_eq!(sim.ccd_overflows(), 0, "the sweep was declined");
        (recorded, poses(&sim, &bodies))
    };
    let (events, poses) = reported(1);
    assert!(!events.is_empty(), "the fixture has to report something");
    assert_eq!((events, poses), reported(WORKERS));
}

// Splitting a step is a ceiling, not a promise: a caller lending more workers
// than the world reserved for gets the reserved number and a count of the
// difference, and lands in exactly the same place either way.
#[test]
fn a_wider_fanout_than_was_reserved_is_declined_and_counted() {
    let build = || stacked(256, 0.85, 0.0);
    let (mut sim, bodies) = build();
    let reserved = sim.reserve_workers(4);
    assert_eq!(reserved, 4);
    assert_eq!(sim.workers(), 4);
    for _ in 0..60 {
        sim.step_with(TICK, &Threads(32));
    }
    assert_eq!(sim.worker_overflows(), 60);
    sim.clear_worker_overflows();
    assert_eq!(sim.worker_overflows(), 0);
    assert_eq!(poses(&sim, &bodies), walked(build, 60, 4));
}

// A caller that lends nothing steps the same world the same way, which is what
// keeps a host with no threads on the same code path.
#[test]
fn a_world_that_reserved_nothing_steps_serially() {
    let (mut sim, bodies) = stacked(256, 0.85, 0.0);
    assert_eq!(sim.workers(), 1);
    for _ in 0..60 {
        sim.step(TICK);
    }
    assert_eq!(sim.worker_overflows(), 0);
    assert_eq!(
        poses(&sim, &bodies),
        walked(|| stacked(256, 0.85, 0.0), 60, 1)
    );
}
