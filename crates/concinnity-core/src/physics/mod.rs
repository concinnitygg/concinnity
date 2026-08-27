//! The rigid-body simulation driver: it builds a simulation from the world's
//! physics content at init, steps it on the fixed tick, and writes the results
//! back as component data.
//!
//! The simulation itself is [`concinnity_physics::Simulation`], whose
//! vocabulary is `[f32; 3]` / Euler-degree data addressed by an opaque
//! `BodyHandle`. This module holds the driver around it: prop bodies, character
//! rigs, probes, contact shaping, and layer resolution.
//!
//! Nothing here reads a clock, a file, or a device. The one thing only a host
//! can supply is threads, which it lends through [`PhysicsFanout`]; a world
//! with none steps on the calling thread.

// What the simulation reserves for a world, and the ledger row reporting it.
mod budget;
// Contact-event shaping: per-frame batching and the per-pair refractory.
mod contacts;
// The authored assets turned into the simulation's shapes and parameters.
mod convert;
// The seam a host lends the step's independent work its threads through.
mod fanout;
// The sorted lookup containers the driver's indices are kept in.
mod index;
// Prev/curr pose snapshots blended by the frame's accumulator alpha.
mod interp;
// Named collision layers over the simulation's 32-bit interaction groups.
mod layers;
// Raycast probe answering for animation IK and the follow camera.
mod probes;
// Prop bodies with their handle -> entity and tracked-entity indices.
mod props;
// Root-motion character rigs: one kinematic capsule per `CharacterRig`.
mod rig;
// The system that builds and steps the simulation from the world's bodies,
// driven by an optional `PhysicsConfig`.
mod system;
// The world's floor: the heightfield collider and the noise it is sampled from.
mod terrain;

#[cfg(test)]
mod test_world;

pub use fanout::{PhysicsFanout, SerialFanout};
pub use system::PhysicsSystem;
