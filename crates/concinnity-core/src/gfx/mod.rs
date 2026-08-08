// src/gfx/mod.rs
//
// CPU-side render-prep compute: LOD decimation, skinning and pose blending, IK,
// software rasterisation, the chunk-streaming allocators, and the UI layout
// helpers. No backend handles and no rendering logic: the render graph, draw
// lists, and per-backend executors stay in the client crate's own `gfx` module.
//
// The data these compute over -- the GPU layouts, the transform and skeleton
// types, the frustum and camera math, the post-process setting structs -- lives
// in concinnity-types and is re-exported below under its historical paths.
pub mod anim_graph;
pub mod auto_exposure;
pub mod block_alloc;
pub mod chunk_coord;
pub mod dropdown;
pub mod ik;
pub mod image_decode;
pub mod lines;
pub mod lod;
pub mod mesh_payload;
pub mod mesh_seed;
pub mod overlay;
pub mod pick;
pub mod range_alloc;
pub mod raster;
pub mod scroll_layout;
pub mod skinning;

pub use concinnity_types::gfx::{
    camera, frustum, profile, render_types, root_motion, rt_reflections, settings, ssao, ssgi, ssr,
    view_modes,
};
