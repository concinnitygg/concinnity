// src/gfx/streaming/mod.rs
//
// The asset-streaming home: the `std` drivers and the system that schedules
// them. The policy core they consult (`StreamPlanner` / `StreamState` and its
// LRU scoring) is `no_std` and lives in `concinnity_core::render::streaming`,
// so a future `no_std` client runtime can share it.

// Async asset-streaming drivers, scheduled by `StreamingSystem` against
// whichever backend the world is running on.
pub(crate) mod chunk;
mod file_range;
pub(crate) mod mesh;
pub(crate) mod shader;
pub(crate) mod texture;
mod worker;

/// Asset-streaming drive (texture / mesh / voxel-world chunk pools) + the
/// camera-relative view publish. Internal system, constructed alongside
/// GraphicsSystem (same gate) and scheduled immediately before it. `pub` so the
/// editor's debug server can name `StreamingStats` (its state lives in the
/// parked `StreamingState` resource, read via `World::streaming_stats`).
pub mod system;
