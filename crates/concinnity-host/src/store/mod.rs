//! The project's state tree on disk: where it is anchored (`paths`), how a source
//! asset is found inside it (`source`), how the compiled blob is read out of it
//! (`blob`), how the runtime's regenerable artifacts are kept in it (`cache`),
//! and how a regenerable container is published without a reader ever seeing it
//! part-written (`atomic`).
//!
//! This is the engine's only home for filesystem knowledge below the engine and
//! cook crates. `concinnity_core::blob` owns the pure bytes <-> metadata
//! transforms and performs no I/O; concinnity-core owns the runtime blob types
//! and the `PayloadStore` seam and knows neither paths nor files. The reads that
//! join the two live here.

pub mod atomic;
pub mod blob;
pub mod cache;
pub mod paths;
pub mod source;
