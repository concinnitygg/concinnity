//! Metal rendering backend. Gated by #[cfg(backend_metal)] on the mod
//! declaration in lib.rs; compiled on macOS only.

mod allocator;
mod auto_exposure;
mod backend;
mod bindless_args;
mod context;
mod cull;
mod cull_readback;
mod decal;
mod descriptors;
mod encode;
mod error;
mod fault_log;
// pub(in crate::metal) so the render-graph executor, planar mirror, and probe
// bake can name the shared main-pass param structs (MainPassCamera, DrawInputs,
// GpuFrameBuffers, FaceTargets) defined in draw/main.rs.
pub(in crate::metal) mod draw;
mod fog;
mod frame_pacing;
mod frame_rings;
mod glass;
mod gpu_profile;
mod graph_events;
mod graph_exec;
mod graph_queues;
mod hiz;
mod hot_reload;
mod init;
mod light_cull;
mod lights;
mod line;
mod metallib;
mod model_history;
mod msl_cache;
mod parallel_encoder;
mod particle;
mod pass_timing;
mod pipeline;
mod planar;
mod post;
mod probe;
mod probe_cubes;
mod probe_prefilter;
mod quality;
mod raymarch;
mod raytrace;
mod resources;
mod rt_ring;
mod scoped_encoder;
mod screenshot;
mod slang_builtins;
mod text_upload;
mod texture;
mod transient_pool;
mod transparent;
mod water;
mod world_shaders;

pub(crate) use context::MtlContext;
pub(crate) use gpu_profile::probe_gpu_profile;
