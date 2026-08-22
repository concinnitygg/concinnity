//! Metal rendering backend. Gated by #[cfg(backend_metal)] on the mod
//! declaration in lib.rs; compiled on macOS only.

mod allocator;
mod auto_exposure;
mod backend;
mod context;
mod cull;
mod decal;
mod descriptors;
mod encode;
mod error;
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
mod instanced;
mod light_cull;
mod line;
mod metallib;
mod msl_cache;
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
mod rt_ring;
mod scoped_encoder;
mod screenshot;
mod shader_reflect;
mod slang_shaders;
mod streaming;
mod text_upload;
mod texture;
mod transient;
mod transient_pool;
mod transparent;
mod water;
mod world_shaders;

// GPU-free host-side pieces live in the concinnity-render crate (compiled
// unconditionally so their unit tests count toward coverage); re-exported here
// so the backend keeps its `super::{math,uniforms}` / `crate::metal::shader_layout`
// paths. The `shader_layout` re-export is `pub` so the out-of-tree Swift app's
// FFI crate can drive `validate_stage` against the engine layouts.
pub use concinnity_render::metal::shader_layout;
pub(crate) use concinnity_render::metal::{math, uniforms};

// `set_preview_view` / `set_embedded_pump_events` are the play-in-view
// embedding hooks the out-of-tree Swift app drives through its FFI crate; no
// crate in this workspace calls them.
pub(crate) use context::MtlContext;
pub use context::{set_embedded_pump_events, set_preview_view};
pub(crate) use gpu_profile::probe_gpu_profile;
// Build-time Metal shader-layout reflection, driven by the cook pipeline through
// the thin `ShaderBuildValidator` bridge in concinnity-editor.
pub use shader_reflect::{
    ShaderLayoutIssue, metal_device_available, metal_source_defines, validate_metal_shader_layout,
};
