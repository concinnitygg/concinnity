// src/asset_impls/mod.rs
//
// The `BuildAsset` trait implementations for each compiled asset type. The asset
// data types, their `Component` and `SourceBacked` impls, and their runtime
// helpers stay in concinnity-core; only the build-time `compile_payload` /
// `source_files` impls live here, calling the compile pipeline in this crate.
// These are trait impls only, so the modules need no re-exports.

mod file;
mod procedural_mesh;
mod room;
mod sdf_volume;
mod shader;
mod voxel_chunk;
