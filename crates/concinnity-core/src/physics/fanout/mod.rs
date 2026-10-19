// How a step reaches the threads a host lends.
//
// `crate::fanout::Fanout` is generic in the item and the body, so a system
// holding one as a trait object cannot be. The seam is drawn one level up: the
// system holds "advance this simulation by dt", and the fan-out it reaches for
// on the way is monomorphized inside that answer.
//
// A step hands out independent work and loads every result back in manifold
// order afterwards, so which thread solved what is not observable in the world
// state; a fan-out decides only how long the step takes.

use crate::ecs::ScheduleMode;
use crate::fanout::{Fanout, Inline, Lent};
use crate::physics::Simulation;

// Where a system's steps run.
pub(crate) trait PhysicsFanout: core::fmt::Debug + Send {
    // Workers the step's per-worker scratch is reserved for under `mode`.
    // Read once, at world start.
    fn worker_count(&self, mode: ScheduleMode) -> usize;

    // Advance `sim` by `dt` on whatever `mode` names.
    fn step(&self, sim: &mut Simulation, dt: f32, mode: ScheduleMode);
}

// Every step on the calling thread: a system no host lent a pool.
impl PhysicsFanout for Inline {
    fn worker_count(&self, _mode: ScheduleMode) -> usize {
        1
    }

    fn step(&self, sim: &mut Simulation, dt: f32, _mode: ScheduleMode) {
        sim.step_with(dt, &Inline);
    }
}

impl<P: Fanout> PhysicsFanout for Lent<P> {
    fn worker_count(&self, mode: ScheduleMode) -> usize {
        self.pool(mode).workers()
    }

    fn step(&self, sim: &mut Simulation, dt: f32, mode: ScheduleMode) {
        sim.step_with(dt, self.pool(mode));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fanout::test_pools::Reversed;
    use crate::physics::{ColliderShape, DynamicParams, LayerMask};

    static WIDE: Reversed = Reversed(4);

    // A lent pool reserves its own worker count and steps to the same state
    // the calling thread does.
    #[test]
    fn a_lent_pool_reserves_its_workers_and_steps_like_the_calling_thread() {
        let lent = Lent::new(|_| &WIDE);
        assert_eq!(lent.worker_count(ScheduleMode::Parallel), 4);
        assert_eq!(Inline.worker_count(ScheduleMode::Parallel), 1);
        let run = |fanout: &dyn PhysicsFanout| {
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
            sim.reserve_workers(fanout.worker_count(ScheduleMode::Parallel));
            for _ in 0..30 {
                fanout.step(&mut sim, 1.0 / 60.0, ScheduleMode::Parallel);
            }
            sim.body_pose_quat(ball).expect("a live body").0
        };
        let stepped = run(&lent);
        assert!(stepped[1] < 10.0, "the step advanced the body");
        assert_eq!(stepped, run(&Inline));
    }
}
