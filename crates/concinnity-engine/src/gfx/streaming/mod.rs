//! The asset-streaming home: the load threads and payload sources behind
//! core's streaming pools (`concinnity_core::render::stream_pools`), and the
//! system that runs them each frame.

mod file_range;
pub(crate) mod mesh;
pub(crate) mod planet;
pub(crate) mod shader;
pub(crate) mod texture;

// The streaming drive (texture / mesh / chunk / planet-tile pools) and the
// camera-relative view publish, scheduled immediately before GraphicsSystem.
pub(crate) mod system;
