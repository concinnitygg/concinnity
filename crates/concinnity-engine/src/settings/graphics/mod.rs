//! The graphics settings a world declares: the `GraphicsConfig` and its
//! per-feature sections. [`super::resolve`] reads them against the quality
//! preset and the user's choices into the snapshot the renderer is built with.

// Component data types.
pub(crate) mod ambient_occlusion;
pub(crate) mod anti_aliasing;
pub(crate) mod background;
pub(crate) mod bloom;
pub(crate) mod config;
pub(crate) mod culling;
pub(crate) mod display;
pub(crate) mod exposure;
pub(crate) mod frame_pacing;
pub(crate) mod indirect_lighting;
pub(crate) mod pass_resolution;
pub(crate) mod planet_detail;
pub(crate) mod reflections;
pub(crate) mod shadows;
pub(crate) mod streaming;
pub(crate) mod texture_filtering;
pub(crate) mod vignette;
