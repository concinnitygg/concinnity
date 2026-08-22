//! CPU-side render-prep compute: LOD decimation, skinning and pose blending, IK,
//! software rasterisation, payload decoding, and auto-exposure metering. No
//! backend handles and no rendering logic: the render graph, draw lists, and
//! per-backend executors stay in the client crate's own `gfx` module.
//!
//! The data these compute over -- the GPU layouts, the transform and skeleton
//! types, the frustum and camera math, the post-process setting structs -- lives
//! in concinnity-core, which callers name directly.
pub mod anim_graph;
pub mod auto_exposure;
pub mod ik;
pub mod image_decode;
pub mod lines;
pub mod lod;
pub mod mesh_payload;
pub mod mesh_seed;
pub mod pick;
pub mod raster;
pub mod skinning;
