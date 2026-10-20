//! The device backends. The proprietary, hardware-facing renderers - Metal
//! (macOS), DirectX 12 (Windows), Vulkan (Windows/Linux) - plus the shared native
//! Win32 window/input layer. build.rs resolves the backends a build compiles in
//! (the target's native one, plus Vulkan with the `vulkan` feature) into one
//! backend_* cfg each; a target with none compiles no GPU code at all. Depends on concinnity-engine, whose
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
// Copying presented frames out and writing them as PNGs off the frame path.
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) mod frame_capture;
// The CPU's blocked-on-GPU time, shared by the three backends so
// `RenderStats::gpu_wait_us` means the same thing on each.
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) mod gpu_wait;
// How a grass frame reaching a placement limit is logged, the same on every
// backend.
#[cfg(any(backend_metal, backend_dx, backend_vk))]
#[cfg(backend_metal)]
pub mod metal;
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) mod png_encode;
// Consuming a temporal upscaler's history reset, the same on every backend.
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) mod upscale_reset;
// Packing the per-draw cull records, inline or across the job pool.
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) mod record_pack;
// How the reflection-probe bake is logged, the same on every backend.
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) mod probe_report;
// The API-independent half of the vendor temporal upscalers (FSR, DLSS, XeSS)
// the DirectX and Vulkan backends drive.
#[cfg(any(backend_dx, backend_vk))]
pub(crate) mod upscale_sdk;
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
// into the single-source shaders: the expected offsets come from the compiler,
// per target, rather than from a hand-written number. Reads the source
// assembly a backend brings with it, so a build with none has nothing to check.
#[cfg(all(test, any(backend_metal, backend_dx, backend_vk)))]
mod shader_layout;

// Source-scanning guards over the explicit backends' resource barriers.
#[cfg(test)]
mod audit;

// How a failed ray-tracing BVH update is logged, the same on every backend.
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) mod rt_report;

mod factory;
pub use factory::{COMPILED_PLATFORMS, init_backend, probe_device};
