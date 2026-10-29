//! The per-frame pipeline: extraction into an owned snapshot, the frame plan,
//! submission, and what a frame reports back to the simulation.

pub mod extract;
pub mod feedback;
pub mod ops;
pub mod outcome;
pub mod plan;
pub mod queues;
pub mod snapshot;
pub mod submit;
