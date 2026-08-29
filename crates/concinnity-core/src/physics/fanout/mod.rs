// How a host lends the simulation its threads.
//
// `crate::physics::Fanout` is generic in the item and the body so a step's
// work units are monomorphised into it rather than boxed, which is what a
// system holding one as a trait object cannot be. So the seam is drawn one
// level up: a host answers "advance this simulation by dt", and the fan-out it
// reaches for on the way stays inside its own answer.
//
// The ordering contract is not this seam's to keep. A step hands out
// independent work and loads every result back in manifold order afterwards,
// so which thread solved what is not observable in the world state either way;
// what a fan-out decides is only how long the step takes.

mod inline;

pub use inline::{Fanout, Inline};

use crate::physics::Simulation;

use crate::ecs::ScheduleMode;

/// A host's way of lending the simulation the threads it has.
///
/// A world with none runs its steps on the calling thread through
/// [`SerialFanout`], which is the default a [`PhysicsSystem`] is built with.
///
/// [`PhysicsSystem`]: crate::physics::PhysicsSystem
pub trait PhysicsFanout: core::fmt::Debug + Send {
    /// Workers the step's per-worker scratch is reserved for under `mode`.
    /// Read once, at world start.
    fn worker_count(&self, mode: ScheduleMode) -> usize;

    /// Advance `sim` by `dt`, lending it whatever `mode` names.
    fn step(&self, sim: &mut Simulation, dt: f32, mode: ScheduleMode);
}

/// The fan-out for a host with no threads to lend: every step runs on the
/// calling thread.
#[derive(Debug, Clone, Copy, Default)]
pub struct SerialFanout;

impl PhysicsFanout for SerialFanout {
    fn worker_count(&self, _mode: ScheduleMode) -> usize {
        1
    }

    fn step(&self, sim: &mut Simulation, dt: f32, _mode: ScheduleMode) {
        sim.step_with(dt, &Inline);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::{ColliderShape, DynamicParams, LayerMask};

    fn falling() -> (Simulation, crate::physics::BodyHandle) {
        let mut sim = Simulation::with_capacity(4);
        let ball = sim
            .add_dynamic(
                &ColliderShape::Ball { radius: 0.5 },
                [0.0, 10.0, 0.0],
                [0.0; 3],
                DynamicParams {
                    mass: 1.0,
                    friction: 0.5,
                    restitution: 0.0,
                    gravity_scale: 1.0,
                    linear_damping: 0.0,
                },
                LayerMask::ALL,
            )
            .expect("room for one body");
        (sim, ball)
    }

    // The serial fan-out reserves one worker's scratch and advances the world,
    // which is the whole of what a host with no pool needs from it.
    #[test]
    fn the_serial_fanout_lends_one_worker_and_steps() {
        assert_eq!(SerialFanout.worker_count(ScheduleMode::Serial), 1);
        assert_eq!(SerialFanout.worker_count(ScheduleMode::Parallel), 1);

        let (mut sim, ball) = falling();
        for _ in 0..10 {
            SerialFanout.step(&mut sim, 1.0 / 60.0, ScheduleMode::Serial);
        }
        assert!(
            sim.body_pose_quat(ball).expect("a live body").0[1] < 10.0,
            "the fan-out advanced the simulation"
        );
    }

    // The mode a serial fan-out is asked for changes nothing: it has one place
    // to run the step, so both land on identical state.
    #[test]
    fn the_schedule_mode_does_not_change_a_serial_step() {
        let run = |mode| {
            let (mut sim, ball) = falling();
            for _ in 0..30 {
                SerialFanout.step(&mut sim, 1.0 / 60.0, mode);
            }
            sim.body_pose_quat(ball).expect("a live body").0
        };
        assert_eq!(run(ScheduleMode::Serial), run(ScheduleMode::Parallel));
    }
}
