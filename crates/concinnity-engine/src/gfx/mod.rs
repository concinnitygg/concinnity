//! The GPU-facing half of the runtime vocabulary: the `#[repr(C)]` layouts the
//! CPU and the shaders both name (`render_types`), and the pure math kernels
//! over them: culling and LOD primitives, view and projection math, the
//! screen-overlay coordinate space, software rasterization, payload decoding,
//! line expansion.
//!
//! What separates this from [`crate::render`]: gfx holds the per-object and
//! per-draw `repr(C)` layouts and pure math kernels; render holds the per-frame
//! builders, the passes, the backend seam, and the uniform blocks a pass uploads
//! (`render::uniforms`). Nothing here owns a backend handle or drives a frame,
//! though a layout may pack a type or apply a rule render defines (the depth
//! convention, a capture's tiles, the daylight block, a canopy's range, when
//! light clustering is active).
pub mod auto_exposure;
pub mod camera;
pub mod cubemap;
pub mod cull_status;
pub mod frustum;
pub mod image_decode;
pub mod jitter;
pub mod lines;
pub mod lod;
pub mod mesh_payload;
pub mod mipmap;
pub mod morph_targets;
pub mod overlay;
pub mod pick;
pub mod projection;
pub mod raster;
pub mod render_types;
pub mod view_modes;
