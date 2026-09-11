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
mod geometry_rebuild;
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

// GPU-free host structs live in `core::render` (counted for coverage); the
// backend keeps its existing `crate::directx::{pass_timing,uniforms}`
// paths through these re-exports. `uniforms` holds the per-pass repr(C) structs;
// each pass file re-exports the struct(s) it fills so their paths are unchanged.
//
// Timing here: `execute_graph` issues an EndQuery before and after each pass's
// encode, and the resolve at the end of the command list copies the whole block
// into the persistently-mapped readback buffer. The CPU reads the previous
// frame's block at the top of `draw_frame`, after the matching fence wait gates
// the GPU writes. SsaoPrepass and SsaoKernel are bundled inside their parent
// encoder, and the FogFroxel / Upscale / Transparent / Raymarch arms are no-ops
// here, so those slots stay zero and drop out of the on-screen chip.
pub(crate) use concinnity_core::render::{directx::uniforms, pass_timing};
