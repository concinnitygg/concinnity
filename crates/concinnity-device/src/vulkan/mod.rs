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
mod input;
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
#[cfg(target_os = "windows")]
mod win32_window;
#[cfg(all(unix, not(target_vendor = "apple"), not(target_os = "android")))]
pub(crate) mod window;
mod wire_cache;
mod wireframe;
mod world_shaders;

// The platform window VkContext owns: the shared native Win32 layer on Windows
// and the shared native AppKit layer on macOS (one window/input implementation
// with the DirectX and Metal backends respectively), GLFW on the desktop Unix
// tier. The gate matches the manifest's, so a target with no window layer
// fails naming this alias rather than naming a missing crate.
#[cfg(target_os = "macos")]
pub(crate) use appkit_window::AppKitVkWindow as PlatformWindow;
#[cfg(target_os = "windows")]
pub(crate) use win32_window::Win32Window as PlatformWindow;
#[cfg(all(unix, not(target_vendor = "apple"), not(target_os = "android")))]
pub(crate) use window::GlfwWindow as PlatformWindow;

pub(crate) use context::VkContext;
pub(crate) use gpu_profile::probe_gpu_profile;

// GPU-free host structs live in `core::render` (counted for coverage); the
// backend keeps its existing `crate::vulkan::{pass_timing,uniforms}`
// paths through these re-exports. `uniforms` holds the per-pass repr(C) structs;
// each pass file re-exports the struct(s) it fills so their paths are unchanged.
//
// Timing here: the start buffer resets the whole block and writes the
// whole-frame start; each per-pass command buffer writes its own pair around
// its encode; the end buffer writes the whole-frame end. Unlike D3D12, which
// can pre-write every slot so a pass that did not run still reads a value,
// Vulkan forbids writing a timestamp to a query already written without an
// intervening reset. A pass absent from this frame's graph therefore leaves its
// reset-but-unwritten slots unavailable; the readback uses WITH_AVAILABILITY
// and reports 0 for any pair that is not both available.
pub(crate) use concinnity_core::render::{pass_timing, vulkan::uniforms};
