// D3D12 rendering backend. Gated by #[cfg(backend_dx)] on the mod declaration
// in lib.rs; compiled on Windows unless the `vulkan` feature is enabled.

mod agility;
mod allocator;
mod auto_exposure;
mod backend;
mod barrier_translate;
mod com;
mod context;
mod cull;
mod cull_readback;
mod decal;
mod draw;
mod error;
mod fog;
mod glass;
mod gpu_profile;
mod graph_exec;
mod hiz;
mod hot_reload;
mod init;
mod light_cull;
mod line;
mod parallel_encoder;
mod particle;
mod pipeline;
mod planar;
mod post;
mod probe;
mod probe_prefilter;
mod pso_library;
mod quality;
mod raymarch;
mod raytrace;
mod resize;
mod resources;
mod screenshot;
pub(crate) mod slang_builtins;
mod texture;
mod transient_pool;
mod transparent;
mod upload_ring;
mod water;
mod wireframe;
mod world_shaders;

pub(crate) use context::DxContext;
pub(crate) use gpu_profile::probe_gpu_profile;
