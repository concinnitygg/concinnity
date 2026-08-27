// What this crate lends `concinnity_core::physics::PhysicsSystem`: the one
// thing the driver needs that only a host has.
//
//   fanout.rs   the job pool a step's independent work is offered to
//
// The simulation, the driver around it, and everything a tick does to the
// world are in concinnity-core; the gate in `ecs::schedule` builds the system
// with the pool attached.

pub(crate) mod fanout;

use concinnity_core::physics::PhysicsSystem;

// The physics system as this host runs it: steps fanned out across the job
// pool the frame's schedule names.
pub(crate) fn build(config: concinnity_core::components::PhysicsConfig) -> PhysicsSystem {
    PhysicsSystem::new(config).with_fanout(Box::new(fanout::PoolFanout))
}
