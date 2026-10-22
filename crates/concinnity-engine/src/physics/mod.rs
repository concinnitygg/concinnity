//! The rigid-body simulation, and the driver that runs it over a world.
//!
//! Two halves that stay apart all the way down. The simulation is the solver:
//! the shapes, body parameters, joints, layer masks, and step results that
//! cross the boundary between a caller and the [`Simulation`], plus the
//! [`PhysicsBudget`] a world reserves its bodies against. Everything crossing
//! that boundary is plain data in the engine's `[f32; 3]` and Euler-degree
//! representation, addressed by an opaque [`BodyHandle`]. That is what lets the
//! simulation hold whatever math types it likes without those types reaching a
//! caller, and it names no engine domain type in either direction.
//!
//! The driver around it builds a simulation from the world's physics content at
//! init, steps it on the fixed tick, and writes the results back as component
//! data: prop bodies, character rigs, probes, contact shaping, and layer
//! resolution. Where a module holds both halves they sit in separate files,
//! never one.
//!
//! Nothing here reads a clock, a file, or a device. The one thing only a host
//! can supply is threads, which it lends through a [`Lent`](concinnity_core::fanout::Lent)
//! pool; a world with none steps on the calling thread.

// What the simulation reserves for a world, derived from the counts the world
// is measured by, and the ledger row reporting it.
mod capacity;
// The move input a character rig is driven by and the result it reports.
mod character;
// Contact-event shaping: per-frame batching and the per-pair refractory.
mod contacts;
// The authored assets turned into the simulation's shapes and parameters.
mod convert;
pub(crate) mod entity_ray;
// What a step reports back: contact, ray, and sensor crossings.
mod events;
// Where a step's independent work runs.
mod fanout;
// Buoyant bodies pushed by the water they float in.
mod float;
// The opaque body identity a caller addresses the simulation by.
mod handle;
// The sorted lookup containers the driver's indices are kept in.
mod index;
// Prev/curr pose snapshots blended by the frame's accumulator alpha.
mod interp;
// The constraints two bodies can be tied together by, and their motors.
mod joints;
// The membership/filter bit pair, and the named layers resolved onto it.
mod layers;
// Bodies held where they are while a planet's ground patch is elsewhere.
mod off_ground;
// Raycast probe answering for animation IK and the follow camera.
mod probes;
// Prop bodies with their handle -> entity and tracked-entity indices.
mod props;
// Root-motion character rigs: one kinematic capsule per `CharacterRig`.
mod rig;
// The solver: broadphase, narrowphase, islands, contacts, joints, CCD.
mod sim;
// The system that builds and steps the simulation from the world's bodies,
// driven by an optional `PhysicsConfig`.
mod system;
// The world's floor: the heightfield collider and the noise it is sampled from.
mod terrain;
// The collider shapes and dynamic body parameters a caller builds bodies from.
mod types;

#[cfg(test)]
mod bench;
#[cfg(test)]
mod tests;
// Per-instance components an entity is composed from: its placement, render
// description, collision, hierarchy, and gameplay tags.
pub(crate) mod body_dynamics;
pub(crate) mod buoyancy;
pub(crate) mod collider;
pub(crate) mod config;
pub(crate) mod contact_event;
pub(crate) mod joint;
pub(crate) mod prop_body;
pub(crate) mod rigid_body;
pub(crate) mod trigger_volume;
pub(crate) mod volume_event;
pub(crate) mod wake_request;

pub use capacity::{PhysicsBudget, PhysicsCounts};
pub use character::{CharacterMove, CharacterMoveInput};
pub use events::{ContactHit, RayHit, SensorCrossing};
pub use handle::BodyHandle;
pub use joints::{JointMotor, JointSpec};
pub use layers::LayerMask;
pub use sim::{
    CharacterCapsule, FluidLoad, FluidPiece, FluidShape, MAX_FLUID_PIECES, ShapeCast, ShapeCastHit,
    SimConfig, Simulation, Submerged, submerged,
};
pub use system::PhysicsSystem;
pub use types::{BodyState, ColliderShape, DynamicParams};

/// Acceleration due to gravity in world units per second squared. Shared with
/// the third-person controller so its jump takeoff matches the rig's fall.
pub const GRAVITY: f32 = 20.0;
