//! Source-file readers: the artist-supplied formats a build starts from, and
//! the expansion that turns one of them into asset entries.
//!
//! `scene` is the entry point, dispatching a `SceneImport` on its source
//! extension into the container readers below it (`fbx`, `glb` / `gltf` over
//! `gltf_source`, `wavefront`). `panorama` recognises the one `.glb` shape that
//! is an environment image rather than scene geometry, and `mesh_reimport`
//! reads a mesh back out of a parsed document for hot reload. The container and
//! image formats these hand off to live in `crate::codec`.

// Neutral grey vertex colour for imported geometry, so a mesh takes its
// material albedo unmodified. Shared by every source-format decoder.
pub(crate) const NEUTRAL_COLOR: [f32; 3] = [0.75, 0.74, 0.72];

pub mod fbx;
pub mod glb;
pub(crate) mod gltf;
pub mod gltf_source;
pub mod mesh_reimport;
/// Recognises a `.glb` that packages an environment image as a sphere you stand
/// inside, so it imports as an EnvironmentMap instead of scene geometry.
pub mod panorama;
pub(crate) mod scene;
pub(crate) mod wavefront;
