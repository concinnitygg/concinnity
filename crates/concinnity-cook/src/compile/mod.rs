//! The compile path: what a validated world is turned into once nothing
//! authoring-only is left in it. Most of these map one asset type's args and
//! sources onto that type's binary payload; the rest are whole-world passes
//! that the same stage runs (`physics_budget` counts what the runtime must
//! reserve, `scene_partition` decides which scene a payload packs into,
//! `thumbnail` renders previews from finished payloads, and `character_shape`
//! warns on names that resolve against nothing).
//!
//! `pipeline` sequences these; the source readers they call live in
//! `crate::import` and the format decoders in `crate::codec`.

pub(crate) mod audio_clip;
pub mod character;
pub(crate) mod character_shape;
pub mod color_lut;
pub(crate) mod cubemap;
pub mod environment_map;
/// Referenced-file assets: paths and their compiled payloads.
pub(crate) mod file;
pub(crate) mod font;
/// Build-time mesh generators + payload compilers. The runtime-side mesh helpers
/// they share (tangents, the voxel mesher, chunk streaming) stay in
/// `concinnity_core::geometry`; this module re-exports what cook code names.
pub(crate) mod geometry;
pub mod mesh_compile;
pub(crate) mod physics_budget;
pub(crate) mod root_motion;
pub(crate) mod scene_partition;
pub mod shader;
pub(crate) mod spawn_population;
pub mod texture;
pub(crate) mod thumbnail;
