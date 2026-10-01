//! Runtime GPU resource management for `MtlContext`. The methods are split
//! across sibling files by resource family:
//!
//!   textures.rs   albedo + normal-map pool slot updates, IBL envmap +
//!                 color-grading LUT hot-swap
//!   geometry.rs   the scene lent to the `SceneHost` defaults, and the writer
//!                 that lands streamed geometry in the shared buffers
//!   streaming.rs  `VoxelWorld` chunk headroom
//!   skinning.rs   skinned pipelines + buffer setup, per-frame pose updates,
//!                 and skinned hot-reload paths
//!   geometry_rebuild.rs  `rebuild_static_geometry` -- hot-reload rebuild of the
//!                 shared static vertex + index buffers
//!
//! Each file is an `impl` block on `MtlContext`; nothing is re-exported here --
//! callers reach the methods directly through `MtlContext`.

mod geometry;
mod geometry_rebuild;
pub(crate) mod skinning;
mod streaming;
mod textures;
