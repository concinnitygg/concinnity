// src/gfx/streaming/mod.rs
//
// The asset-streaming home. The `no_std` policy core (`StreamPlanner` /
// `StreamState` and its LRU scoring) lives in `core::render` so a future
// `no_std` client runtime can share it; it is re-exported here so the `std`
// drivers below and the rest of `gfx` name it as `crate::gfx::streaming::*`.
pub(crate) use concinnity_core::render::streaming::*;

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
