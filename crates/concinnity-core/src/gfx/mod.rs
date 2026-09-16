//! The GPU-facing half of the runtime vocabulary: the `#[repr(C)]` layouts the
//! CPU and the shaders both name (`render_types`), and the pure math kernels
//! over them: culling and LOD primitives, the screen-overlay and chunk-grid
//! coordinate spaces, software rasterization, payload decoding, line expansion.
//!
//! The rule that separates this from [`crate::render`]: gfx holds `repr(C)`
//! layouts and pure math kernels; render holds the per-frame builders, the
//! passes, and the backend seam. Nothing here owns a backend handle or drives a
//! frame.
pub mod auto_exposure;
pub mod camera;
pub mod chunk_coord;
pub mod cubemap;
pub mod cull_status;
pub mod font;
pub mod frustum;
pub mod image_decode;
pub mod jitter;
pub mod lines;
pub mod lod;
pub mod mesh_payload;
pub mod mesh_seed;
pub mod overlay;
pub mod projection;
pub mod raster;
pub mod render_types;
pub mod view_modes;
