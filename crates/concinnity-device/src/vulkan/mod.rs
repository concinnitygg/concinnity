// Vulkan rendering backend. Gated by #[cfg(backend_vk)] on the mod declaration
// in lib.rs; compiled on Linux always and on Windows with the `vulkan` feature.

mod auto_exposure;
mod backend;
mod barrier_translate;
mod composite;
mod context;
mod cull;
mod decal;
mod descriptor_layout;
mod device;
mod draw;
mod fog;
mod glass;
mod gpu_profile;
mod graph_exec;
mod hiz;
mod hot_reload;
mod init;
mod input;
mod light_cull;
mod main;
mod math;
mod parallel_encoder;
mod particle;
mod pipeline;
mod planar;
mod post;
mod probe;
mod quality;
mod raymarch;
mod raytrace;
mod render_pass;
mod resources;
mod screenshot;
mod shadow;
mod swapchain;
mod texture;
mod transient_pool;
#[cfg(target_os = "windows")]
mod win32_window;
#[cfg(not(target_os = "windows"))]
pub(crate) mod window;

// The platform window VkContext owns: the shared native Win32 layer on
// Windows (one window/input implementation with the DirectX backend), GLFW on
// Linux.
#[cfg(target_os = "windows")]
pub(crate) use win32_window::Win32Window as PlatformWindow;
#[cfg(not(target_os = "windows"))]
pub(crate) use window::GlfwWindow as PlatformWindow;

pub use context::VkContext;
pub(crate) use gpu_profile::probe_gpu_profile;

// GPU-free host structs live in concinnity-render (counted for coverage); the
// backend keeps its existing `crate::vulkan::{pass_timing,probe_uniforms,uniforms}`
// paths through these re-exports. `uniforms` holds the per-pass repr(C) structs;
// each pass file re-exports the struct(s) it fills so their paths are unchanged.
pub(crate) use concinnity_render::vulkan::{pass_timing, probe_uniforms, uniforms};
