// Vulkan rendering backend. Gated by #[cfg(backend_vk)] on the mod declaration
// in lib.rs; compiled on Linux always and on macOS / Windows with the `vulkan`
// feature. macOS runs over the MoltenVK portability driver.

mod allocator;
#[cfg(target_os = "macos")]
mod appkit_window;
mod auto_exposure;
mod backend;
mod barrier_translate;
pub(crate) mod builtins;
mod composite;
mod context;
mod cull;
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
mod input;
mod instance_exts;
mod light_cull;
mod line;
mod loader;
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
mod spot_shadow;
mod swapchain;
mod texture;
mod transient_pool;
#[cfg(target_os = "windows")]
mod win32_window;
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub(crate) mod window;
mod wireframe;
mod world_shaders;

// The platform window VkContext owns: the shared native Win32 layer on Windows
// and the shared native AppKit layer on macOS (one window/input implementation
// with the DirectX and Metal backends respectively), GLFW on Linux.
#[cfg(target_os = "macos")]
pub(crate) use appkit_window::AppKitVkWindow as PlatformWindow;
#[cfg(target_os = "windows")]
pub(crate) use win32_window::Win32Window as PlatformWindow;
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub(crate) use window::GlfwWindow as PlatformWindow;

pub use context::VkContext;
pub(crate) use gpu_profile::probe_gpu_profile;

// GPU-free host structs live in concinnity-render (counted for coverage); the
// backend keeps its existing `crate::vulkan::{pass_timing,probe_uniforms,uniforms}`
// paths through these re-exports. `uniforms` holds the per-pass repr(C) structs;
// each pass file re-exports the struct(s) it fills so their paths are unchanged.
pub(crate) use concinnity_render::vulkan::{pass_timing, probe_uniforms, uniforms};
