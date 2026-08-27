//! The GPU-facing half of the runtime vocabulary and the CPU compute over it:
//! the `#[repr(C)]` structs the CPU and the shaders both name (`render_types`),
//! the transform and skeleton types animation is expressed in, the culling and
//! LOD primitives the backends test against, the screen-overlay and chunk-grid
//! coordinate spaces, the post-process / quality setting structs, and the
//! render-prep kernels that feed them: skinning and pose blending, IK, LOD
//! decimation, software rasterisation, payload decoding, line expansion.
//!
//! No backend handles and no rendering logic: the render graph, draw lists, and
//! per-backend executors stay in the client crate's own `gfx` module.
pub mod anim_graph;
pub mod auto_exposure;
pub mod camera;
pub mod chunk_coord;
pub mod font;
pub mod frustum;
pub mod ik;
pub mod image_decode;
pub mod lines;
pub mod lod;
pub mod mesh_payload;
pub mod mesh_seed;
pub mod morph_targets;
pub mod morph_weights;
pub mod overlay;
pub mod pick;
pub mod pose_blend;
pub mod pose_scratch;
pub mod profile;
pub mod projection;
pub mod proportions;
pub mod raster;
pub mod render_types;
pub mod root_motion;
pub mod rt_reflections;
pub mod settings;
pub mod skeleton;
pub mod ssao;
pub mod ssgi;
pub mod ssr;
pub mod transform;
pub mod transform_propagation;
pub mod view_modes;
