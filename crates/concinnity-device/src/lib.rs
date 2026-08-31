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

// Bridge so the backends' historical `crate::gfx::<X>` paths resolve: the GPU
// data layouts, render math, and CPU kernels (`concinnity_core::gfx`) plus the
// render-prep modules (`concinnity_core::render`). Each
// backend consumes a different subset and one backend compiles per build, so a
// portion of these re-exports is unused on any given build - suppress it
// crate-wide rather than gate every item per backend.
#[expect(
    unused_imports,
    reason = "one backend compiles per build, so each consumes only a subset of these re-exports"
)]
pub(crate) mod gfx {
    pub(crate) use concinnity_core::gfx::{
        auto_exposure, frustum, image_decode, lod, mesh_payload, morph_targets, profile,
        render_types, rt_reflections, ssao, ssgi, ssr,
    };
    pub(crate) use concinnity_core::render::{
        backend, backend_init, bvh, csm, decal, display_mode, draw_slot, error, fullscreen,
        hdr_output, input, keymap, lights, ltc, mipmap, parallel_ctx, particles, planar_reflection,
        reflection_probe, render_graph, rt_geom, rt_refit, rt_topology, scene_flow,
        shadow_schedule, skinned_pool, slot_rewrites, spot_shadow, transparent, volumetric_fog,
    };
}

// Asset data types, the runtime build helpers, and the mesh/chunk geometry the
// backends reach by their historical `crate::` paths, plus the shared rayon job
// pool.
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) use concinnity_core::components;
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) use concinnity_core::{bake, geometry};
#[cfg(any(backend_metal, backend_dx, backend_vk))]
pub(crate) use concinnity_host::thread::jobs;

#[cfg(backend_dx)]
pub(crate) mod directx;
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
pub(crate) mod slang_source;

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
mod slangc_gate;

// Cross-backend drift guard for the shared `GpuObjectData` shader fragments.
// Test-only and backend-agnostic on purpose: the fragments are checked as text,
// so one build validates all three languages.
#[cfg(test)]
mod object_data_layout;

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
