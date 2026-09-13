//! Runtime GPU resource management for `MtlContext`. The methods are split
//! across sibling files by resource family:
//!
//!   textures.rs   albedo + normal-map pool slot updates, IBL envmap +
//!                 color-grading LUT hot-swap
//!   geometry.rs   Streamed-mesh upload / eviction via the sub-allocators, and
//!                 in-place per-slot geometry replacement
//!   streaming.rs  `VoxelWorld` chunk streaming
//!   skinning.rs   skinned pipelines + buffer setup, per-frame pose updates,
//!                 and skinned hot-reload paths
//!   geometry_rebuild.rs  `rebuild_static_geometry` -- hot-reload rebuild of the
//!                 shared static vertex + index buffers
//!
//! Each file is a single `impl MtlContext { pub fn ... }` block; nothing is
//! re-exported here -- callers reach the methods directly through `MtlContext`.

mod geometry;
mod geometry_rebuild;
pub(crate) mod skinning;
mod streaming;
mod textures;
