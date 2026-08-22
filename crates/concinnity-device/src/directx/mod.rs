// D3D12 rendering backend. Gated by #[cfg(backend_dx)] on the mod declaration
// in lib.rs; compiled on Windows unless the `vulkan` feature is enabled.

mod agility;
mod allocator;
mod auto_exposure;
mod backend;
mod barrier_translate;
pub(crate) mod builtins;
mod com;
mod context;
mod cull;
mod decal;
mod draw;
mod draw_iter;
mod dxc;
mod error;
mod fog;
mod geometry_rebuild;
mod glass;
mod gpu_profile;
mod graph_exec;
mod hiz;
mod hot_reload;
mod init;
mod light_cull;
mod line;
mod math;
mod parallel_encoder;
mod particle;
mod pipeline;
mod planar;
mod post;
mod probe;
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
mod upload_ring;
mod wireframe;
mod world_shaders;

pub(crate) use context::DxContext;
pub(crate) use gpu_profile::probe_gpu_profile;

// GPU-free host structs live in concinnity-render (counted for coverage); the
// backend keeps its existing `crate::directx::{pass_timing,uniforms}`
// paths through these re-exports. `uniforms` holds the per-pass repr(C) structs;
// each pass file re-exports the struct(s) it fills so their paths are unchanged.
pub(crate) use concinnity_render::directx::{pass_timing, uniforms};
