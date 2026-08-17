// src/lib.rs
//
// The device backends. The proprietary, hardware-facing renderers - Metal
// (macOS), DirectX 12 (Windows), Vulkan (Windows/Linux) - plus the shared native
// Win32 window/input layer. Exactly one backend compiles per build (resolved by
// build.rs into a single backend_* cfg). Depends on concinnity-render (the
// RenderBackend/SceneControl trait seam + render-prep) and concinnity-cpu; owns
// no gameplay, ECS-runtime, audio, or physics. The client drives these through a
// `Box<dyn RenderBackend>` obtained from `init_backend`, never naming a concrete
// context type.

// Bridge so the backends' historical `crate::gfx::<X>` paths resolve: the GPU
// data layouts and render math (concinnity-core), the CPU kernels over them
// (concinnity-cpu), and the render-prep modules (concinnity-render). Each
// backend consumes a different subset and one backend compiles per build, so a
// portion of these re-exports is unused on any given build - allow it
// crate-wide rather than gate every item per backend.
#[allow(unused_imports)]
pub(crate) mod gfx {
    pub use concinnity_core::gfx::lod_select as lod;
    pub use concinnity_core::gfx::{
        frustum, profile, render_types, rt_reflections, ssao, ssgi, ssr,
    };
    pub use concinnity_cpu::gfx::{auto_exposure, image_decode, mesh_payload};
    pub use concinnity_render::{
        backend, backend_init, bvh, csm, decal, display_mode, draw_slot, error, fullscreen,
        hdr_output, input, keymap, ltc, mipmap, parallel_ctx, particles, planar_reflection,
        reflection_probe, render_graph, rt_geom, rt_topology, scene_flow, shadow_schedule,
        skinned_pool, slot_rewrites, spot_shadow, transparent, volumetric_fog,
    };
}

// Asset data types, the runtime build helpers, and the mesh/chunk geometry the
// backends reach by their historical `crate::` paths; plus the shared rayon job
// pool (now in concinnity-render).
pub(crate) use concinnity_core::assets;
pub(crate) use concinnity_cpu::{build, geometry};
pub(crate) use concinnity_render::jobs;

#[cfg(backend_dx)]
pub mod directx;
#[cfg(backend_metal)]
pub mod metal;
#[cfg(backend_vk)]
pub mod vulkan;
// Native Win32 window/input/display-mode layer shared by the HWND-rendering
// backends (DirectX always; Vulkan on Windows instead of GLFW).
#[cfg(all(target_os = "windows", any(backend_dx, backend_vk)))]
pub(crate) mod win32;
// Native AppKit window/input/display-mode layer shared by the NSView-rendering
// backends (Metal always; Vulkan on macOS instead of GLFW).
#[cfg(all(target_os = "macos", any(backend_metal, backend_vk)))]
pub(crate) mod appkit;

// Disk cache for shader binaries compiled after build time: the built-ins the
// DirectX and Vulkan backends compile at init, and the Metal raymarch
// libraries assembled from world-authored SdfVolume fragments (the rest of
// Metal precompiles into the binary via the toolchain crate).
pub(crate) mod shader_cache;

// Shared source assembly for the single-source `.slang` shaders every backend
// draws from.
#[cfg(any(backend_dx, backend_vk, backend_metal))]
pub(crate) mod slang_source;

// Disk persistence for driver pipeline blobs (VkPipelineCache, D3D12 pipeline
// library). Metal needs none: its libraries are precompiled or cached above,
// and the OS maintains the per-app pipeline binary cache.
#[cfg(any(backend_dx, backend_vk))]
pub(crate) mod pipeline_cache;

// Export-time precompilation of the built-in shaders into a bundle's
// shader-cache/. Backends whose shaders compile at renderer init (DX, VK)
// declare their compile set as data; `cn export` compiles it here, in-process,
// with no GPU device. Metal precompiles at build time and needs none of this.
#[cfg(any(backend_dx, backend_vk))]
pub mod precompile;

// Cross-backend drift guard for the shared `GpuObjectData` shader fragments.
// Test-only and backend-agnostic on purpose: the fragments are checked as text,
// so one build validates all three languages.
#[cfg(test)]
mod object_data_layout;

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
pub(crate) mod suballoc;

mod factory;
pub use factory::{init_backend, probe_gpu_profile};
