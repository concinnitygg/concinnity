// src/gfx/mod.rs
//
// The GPU-facing half of the runtime vocabulary: the `#[repr(C)]` structs the
// CPU and the shaders both name (`render_types`), the transform and skeleton
// types animation is expressed in, the culling and LOD-selection primitives the
// backends test against, and the post-process / quality setting structs. Data
// and the small total functions over it; the CPU compute that consumes it
// (skinning, decimation, rasterisation, IBL convolution) lives in
// concinnity-core.
pub mod anim_graph;
pub mod auto_exposure;
pub mod camera;
pub mod frustum;
pub mod lines;
pub mod lod_select;
pub mod profile;
pub mod render_types;
pub mod root_motion;
pub mod rt_reflections;
pub mod settings;
pub mod skeleton;
pub mod ssao;
pub mod ssgi;
pub mod ssr;
pub mod transform;
pub mod view_modes;
