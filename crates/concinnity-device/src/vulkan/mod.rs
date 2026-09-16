// Vulkan rendering backend. Gated by #[cfg(backend_vk)] on the mod declaration
// in lib.rs; compiled on Linux always and on macOS / Windows with the `vulkan`
// feature. macOS runs over the MoltenVK portability driver.

mod allocator;
mod auto_exposure;
mod backend;
mod barrier_translate;
mod context;
mod cull;
mod cull_readback;
mod decal;
mod descriptor_layout;
mod device;
mod draw;
mod error;
mod fog;
mod glass;
mod gpu_profile;
mod graph_exec;
mod hiz;
mod hot_reload;
mod init;
mod instance_exts;
mod light_cull;
mod line;
mod loader;
mod owned;
mod parallel_encoder;
mod particle;
mod pipeline;
mod pipeline_cache;
mod planar;
mod post;
mod probe;
mod probe_prefilter;
mod quality;
mod raymarch;
mod raytrace;
mod record;
mod render_pass;
mod resources;
mod screenshot;
pub(crate) mod slang_builtins;
mod swapchain;
mod texture;
mod transient_pool;
mod transparent;
mod upload_ring;
mod water;
mod window;
mod wire_cache;
mod wireframe;
mod world_shaders;

pub(crate) use context::VkContext;
pub(crate) use gpu_profile::probe_gpu_profile;
