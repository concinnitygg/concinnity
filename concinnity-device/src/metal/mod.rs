// Metal rendering backend. Gated by #[cfg(backend_metal)] on the mod
// declaration in lib.rs; compiled on macOS only.

mod auto_exposure;
mod backend;
mod context;
mod cull;
mod decal;
mod display_mode;
// pub(in crate::metal) so the render-graph executor, planar mirror, and probe
// bake can name the shared main-pass param structs (MainPassCamera, DrawInputs,
// GpuFrameBuffers, FaceTargets) defined in draw/main.rs.
pub(in crate::metal) mod draw;
mod fog;
mod frame_pacing;
mod glass;
mod gpu_profile;
mod graph_exec;
mod hiz;
mod hot_reload;
mod init;
mod input;
mod instanced;
mod parallel_encoder;
mod particle;
mod pass_timing;
mod pipeline;
mod planar;
mod post;
mod probe;
mod quality;
mod raymarch;
mod raytrace;
mod resources;
mod scoped_encoder;
mod screenshot;
mod streaming;
mod texture;
mod transient;
mod transient_pool;
mod transparent;
mod water;
mod window_delegate;

// GPU-free host-side pieces live in the concinnity-render crate (compiled
// unconditionally so their unit tests count toward coverage); re-exported here
// so the backend keeps its `super::{math,uniforms}` / `crate::metal::shader_layout`
// paths. The `shader_layout` re-export is `pub` so the concinnity-app
// shader-reflection adapter can drive `validate_stage` against the engine layouts.
pub use concinnity_render::metal::shader_layout;
pub(crate) use concinnity_render::metal::{math, uniforms};

pub use context::{MtlContext, set_embedded_pump_events, set_preview_view};
pub(crate) use gpu_profile::probe_gpu_profile;
