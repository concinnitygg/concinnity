//! Concinnity physics: the engine's rigid-body simulation and the vocabulary
//! it is driven through, in its own crate so it names no engine domain type.
//! It provides the shapes, body parameters, joints, layer masks, and step
//! results that cross the boundary between a caller and the [`Simulation`],
//! and the [`PhysicsBudget`] a world reserves its bodies against.
//!
//! Everything crossing that boundary is plain data in the engine's `[f32; 3]`
//! and Euler-degree representation, addressed by an opaque [`BodyHandle`].
//! That is what lets the simulation hold whatever math types it likes without
//! those types reaching a caller.
//!
//! The crate is `#![no_std]` because nothing here needs an operating system or
//! an allocator beyond `alloc`: it opens no file, spawns no thread, and reads
//! no clock. Its two dependencies are leaves and neither is an engine domain
//! type: `libm` for the float functions `core` lacks, preferred to std's even
//! where std exists because a software implementation is bit-identical across
//! platforms where a system libm is not, and `concinnity-memory` for the
//! fixed-capacity pool the simulation stores its bodies in. The dependency
//! arrow is concinnity-physics <- concinnity-engine.

#![no_std]

extern crate alloc;

// The test harness is a std program.
#[cfg(test)]
extern crate std;

#[cfg(test)]
mod bench;
mod budget;
mod character;
mod events;
mod fanout;
mod handle;
mod joints;
mod layers;
mod sim;
#[cfg(test)]
mod tests;
mod types;

pub use budget::{PhysicsBudget, PhysicsCounts};
pub use character::{CharacterMove, CharacterMoveInput};
pub use events::{ContactHit, RayHit, SensorCrossing};
pub use fanout::{Fanout, Inline};
pub use handle::BodyHandle;
pub use joints::{JointMotor, JointSpec};
pub use layers::LayerMask;
pub use sim::{
    CharacterCapsule, ShapeCast, ShapeCastHit, SimConfig, Simulation, euler_deg_from_quat,
    quat_from_euler_deg,
};
pub use types::{ColliderShape, DynamicParams};

/// Acceleration due to gravity in world units per second squared. Shared with
/// the third-person controller so its jump takeoff matches the rig's fall.
pub const GRAVITY: f32 = 20.0;
