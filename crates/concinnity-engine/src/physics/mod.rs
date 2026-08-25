// src/physics/mod.rs
//
// The engine's rigid-body simulation driver: it builds a simulation from the
// world's physics content at init, steps it on the fixed tick, and writes the
// results back as component data.
//
// The simulation itself is `concinnity_physics::Simulation`, whose vocabulary
// is engine-native `[f32; 3]` / Euler-degree data addressed by an opaque
// `BodyHandle`. This module holds the driver around it: prop bodies, character
// rigs, probes, contact shaping, and layer resolution.

// What the simulation reserves for a world, and the ledger row reporting it.
mod budget;
// Contact-event shaping: per-frame batching and the per-pair refractory.
mod contacts;
// The authored assets turned into the simulation's shapes and parameters.
mod convert;
// The job pool the step's independent work is offered to.
mod fanout;
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
// The internal physics system that builds and steps the simulation from the
// world's bodies, driven by an optional `PhysicsConfig`.
mod system;

pub(crate) use system::PhysicsSystem;
