//! Synthetic asset bytes, built in memory.
//!
//! A unit test never reads a checked-in binary, so every format the pipeline
//! accepts needs a way to produce a valid one of its own. These are the
//! smallest files each decoder will accept, assembled here once rather than in
//! each crate that needs one.
//!
//! A format whose encoder lives in the workspace keeps its fixture beside that
//! encoder instead: the fixture and the code it feeds should move together.

pub mod glb;
pub mod png;
pub mod wav;
