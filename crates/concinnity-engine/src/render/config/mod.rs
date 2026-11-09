//! The renderer's configuration: the `GraphicsConfig` a world declares and
//! its per-feature sections, and the [`ResolvedGraphics`] snapshot the
//! renderer is built with, which the settings resolve from them against the
//! quality preset and the user's choices.

// Component data types.
pub(crate) mod ambient_occlusion;
pub(crate) mod anti_aliasing;
pub mod apply;
pub(crate) mod background;
pub(crate) mod bloom;
pub(crate) mod culling;
pub(crate) mod display;
pub(crate) mod exposure;
pub(crate) mod frame_pacing;
pub(crate) mod graphics_config;
pub(crate) mod indirect_lighting;
pub(crate) mod pass_resolution;
pub(crate) mod planet_detail;
pub(crate) mod reflections;
mod resolved;
pub(crate) mod shadows;
pub(crate) mod streaming;
pub(crate) mod texture_filtering;
mod tunables;
pub(crate) mod vignette;

pub use resolved::ResolvedGraphics;
pub use tunables::post_process_tunables;
