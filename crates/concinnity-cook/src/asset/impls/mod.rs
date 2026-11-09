// The `BuildAsset` trait implementations for each compiled asset type,
// components and resources alike. The asset data types and their runtime
// helpers stay in concinnity-engine; only the build-time compile and cache-key
// impls live here, calling the compile pipeline in this crate.
// These are trait impls only, so the modules need no re-exports.

mod file;
mod material;
mod planet_ground;
mod procedural_mesh;
mod resources;
mod room;
mod sdf_volume;
mod shader;
mod skinned_mesh;
mod terrain;
mod voxel_chunk;
