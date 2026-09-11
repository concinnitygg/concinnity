// src/directx/resources/mod.rs
//
// Runtime GPU resource management for `DxContext`. The methods are split
// across sibling files by resource family, mirroring `metal/resources/` and
// `vulkan/resources/`:
//
//   textures.rs   Texture-pool slot updates, envmap + color-grading LUT
//                 hot-swap, and the runtime static-draw-object clone
//   geometry.rs   Streamed-mesh upload / eviction and in-place per-slot
//                 geometry replacement
//   streaming.rs  `VoxelWorld` chunk streaming
//   skinning.rs   Skinned pipelines + upload, per-frame pose and morph uploads
//   geometry_rebuild.rs  Size-changing static + skinned VB/IB rebuilds driven
//                 by asset hot-reload
//
// Each file is a single `impl DxContext` block; nothing is re-exported here --
// callers reach the methods directly through `DxContext`.

mod geometry;
pub(in crate::directx) mod geometry_rebuild;
mod skinning;
mod streaming;
mod textures;
