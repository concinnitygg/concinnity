// The scene pieces the scenario tests share: the fixed tick, a floor, a ball,
// and stepping.

use crate::physics::{BodyHandle, ColliderShape, DynamicParams, LayerMask, SimConfig, Simulation};

pub(super) const TICK: f32 = 1.0 / 60.0;

pub(super) fn sim(capacity: usize) -> Simulation {
    Simulation::new(SimConfig::default(), capacity)
}

// A configuration whose bodies never sleep, so a slow drift keeps integrating.
pub(super) fn awake() -> SimConfig {
    SimConfig {
        allow_sleep: false,
        ..SimConfig::default()
    }
}

/// A 40 x 40 floor whose top surface is exactly `y = 0`.
pub(super) fn add_floor(sim: &mut Simulation) -> BodyHandle {
    sim.add_fixed(
        &ColliderShape::Cuboid {
            half_extents: [20.0, 1.0, 20.0],
        },
        [0.0, -1.0, 0.0],
        [0.0; 3],
        0.8,
        LayerMask::ALL,
    )
    .expect("room for the floor")
}

/// A half-meter-radius ball released at `pos`.
pub(super) fn drop_ball(sim: &mut Simulation, pos: [f32; 3], params: DynamicParams) -> BodyHandle {
    sim.add_dynamic(
        &ColliderShape::Ball { radius: 0.5 },
        pos,
        [0.0; 3],
        params,
        LayerMask::ALL,
    )
    .expect("room for the ball")
}

pub(super) fn step_for(sim: &mut Simulation, ticks: usize) {
    for _ in 0..ticks {
        sim.step(TICK);
    }
}

pub(super) fn position(sim: &Simulation, handle: BodyHandle) -> [f32; 3] {
    sim.body_pose(handle).expect("a live body").0
}
