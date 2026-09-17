//! The device backends. The proprietary, hardware-facing renderers - Metal
//! (macOS), DirectX 12 (Windows), Vulkan (Windows/Linux) - plus the shared native
//! Win32 window/input layer. At most one backend compiles per build, resolved by
//! build.rs from the backend features into a single backend_* cfg; a build that
//! names none compiles no GPU code at all. Depends on concinnity-core, whose
//! `render` module holds the RenderBackend/SceneControl trait seam these
//! implement plus the render-prep feeding it; owns no gameplay, ECS-runtime,
//! audio, or physics. The client drives these through a
//! `Box<dyn RenderBackend>` obtained from `init_backend`, never naming a concrete
//! context type.

#[cfg(backend_dx)]
pub(crate) mod directx;
// The forwarding macro the three backends write their RenderBackend family
// impls with.
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) mod forward;
// The CPU's blocked-on-GPU time, shared by the three backends so
// `RenderStats::gpu_wait_us` means the same thing on each.
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) mod gpu_wait;
#[cfg(backend_metal)]
pub mod metal;
// The `objects` / `skinned_visible` render stats, shared so they mean the same
// thing on every backend.
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) mod object_counts;
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) mod png_encode;
#[cfg(backend_vk)]
pub(crate) mod vulkan;
// Native Win32 window/input/display-mode layer shared by the HWND-rendering
// backends (DirectX always; Vulkan on Windows instead of GLFW).
#[cfg(all(target_os = "windows", any(backend_dx, backend_vk)))]
pub(crate) mod win32;
// Native AppKit window/input/display-mode layer shared by the NSView-rendering
// backends (Metal always; Vulkan on macOS instead of GLFW).
#[cfg(all(target_os = "macos", any(backend_metal, backend_vk)))]
pub(crate) mod appkit;

// Shader source assembly, compile and pipeline caches, and export-time
// precompilation shared by the backends.
pub mod shader;

// Reflection-driven layout guard for the `#[repr(C)]` structs the CPU uploads
// into the single-source `.slang` shaders: the expected offsets come from
// slangc, per target, rather than from a hand-written number. Reads the source
// assembly a backend brings with it, so a build with none has nothing to check.
#[cfg(all(test, any(backend_metal, backend_dx, backend_vk)))]
mod shader_layout;

// Source-scanning guards over the explicit backends' resource barriers.
#[cfg(test)]
mod audit;

// Device-memory placement policy shared by the backends' allocators.
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) mod suballoc;

mod factory;
pub use factory::{init_backend, probe_gpu_profile};
