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
// The CPU's blocked-on-GPU time, shared by the three backends so
// `RenderStats::gpu_wait_us` means the same thing on each.
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) mod gpu_wait;
#[cfg(backend_metal)]
pub mod metal;
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

// The runtime cache segment both caches below write into, and the checkpoints
// at which it reaches disk.
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) mod runtime_cache;

// Disk cache for shader binaries compiled after build time: the built-ins the
// DirectX and Vulkan backends compile at init, and the Metal raymarch
// libraries assembled from world-authored SdfVolume fragments (the rest of
// Metal precompiles into the binary via the toolchain crate).
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) mod shader_cache;

// Scratch directory for the shader compilers that work on files.
#[cfg(any(backend_dx, backend_vk, backend_metal))]
pub(crate) mod compiler_work;

// Shared source assembly for the single-source `.slang` shaders every backend
// draws from.
#[cfg(any(backend_dx, backend_vk, backend_metal))]
pub(crate) mod raymarch_source;
pub(crate) mod slang_source;
pub(crate) mod surface_source;

// Disk persistence for driver pipeline blobs (VkPipelineCache, D3D12 pipeline
// library). Metal needs none: its libraries are precompiled or cached above,
// and the OS maintains the per-app pipeline binary cache.
#[cfg(any(backend_dx, backend_vk))]
pub(crate) mod pipeline_cache;

// Export-time precompilation of the built-in shaders into the cache segment a
// bundle ships. Backends whose shaders compile at renderer init (DX, VK)
// declare their compile set as data; `cn export` compiles it here, in-process,
// with no GPU device. Metal precompiles at build time and needs none of this.
#[cfg(any(backend_dx, backend_vk))]
pub mod precompile;

// Test-only probe for the shader compiler the single-source `.slang` shaders
// need, so the compile checks skip a host without one instead of failing. Only
// a backend has shaders to compile.
#[cfg(all(test, any(backend_metal, backend_dx, backend_vk)))]
// Reflection-driven layout guard for the `#[repr(C)]` structs the CPU uploads
// into the single-source `.slang` shaders: the expected offsets come from
// slangc, per target, rather than from a hand-written number. Reads the source
// assembly a backend brings with it, so a build with none has nothing to check.
#[cfg(all(test, any(backend_metal, backend_dx, backend_vk)))]
mod shader_layout;

// Ownership guard for the explicit backends' resource barriers. Test-only and
// backend-agnostic for the same reason as the fragment guard above: the call
// sites are counted as text, so one build audits both explicit backends.
#[cfg(test)]
mod barrier_audit;

// The companion guard: `barrier_audit` proves every barrier is classified, this
// one proves a classified barrier is not redundant with one the graph executor
// already emits. Same text-scanning rationale, so it also covers DirectX from a
// macOS build.
#[cfg(test)]
mod double_drive_audit;

// Device-memory placement policy shared by the backends' allocators.
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) mod suballoc;

mod factory;
pub use factory::{init_backend, probe_gpu_profile};
