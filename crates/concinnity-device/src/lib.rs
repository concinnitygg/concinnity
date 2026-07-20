// src/lib.rs
//
// The device backends. The proprietary, hardware-facing renderers - Metal
// (macOS), DirectX 12 (Windows), Vulkan (Windows/Linux) - plus the shared native
// Win32 window/input layer. Exactly one backend compiles per build (resolved by
// build.rs into a single backend_* cfg). Depends on concinnity-render (the
// RenderBackend/SceneControl trait seam + render-prep) and concinnity-core; owns
// no gameplay, ECS-runtime, audio, or physics. The client drives these through a
// `Box<dyn RenderBackend>` obtained from `init_backend`, never naming a concrete
// context type.

// Bridge so the backends' historical `crate::gfx::<X>` paths resolve: the
// GPU-free GPU-layout / render-math types (concinnity-core) plus the render-prep
// modules (concinnity-render). Each backend consumes a different subset and one
// backend compiles per build, so a portion of these re-exports is unused on any
// given build - allow it crate-wide rather than gate every item per backend.
#[allow(unused_imports)]
pub(crate) mod gfx {
    pub use concinnity_core::gfx::{
        auto_exposure, frustum, lod, mesh_payload, profile, range_alloc, render_types,
        rt_reflections, ssao, ssgi, ssr,
    };
    pub use concinnity_render::{
        backend, backend_init, bvh, csm, decal, display_mode, draw_slot, fullscreen, hdr_output,
        input, keymap, ltc, mipmap, parallel_ctx, particles, planar_reflection, reflection_probe,
        render_graph, rt_topology, scene_reel, shadow_schedule, skinned_pool, slot_rewrites,
        spot_shadow, transparent, volumetric_fog,
    };
}

// Asset data types, the runtime build helpers, and the mesh/chunk geometry the
// backends reach by their historical `crate::` paths; plus the shared rayon job
// pool (now in concinnity-render).
pub(crate) use concinnity_core::{assets, build, geometry};
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

mod factory;
pub use factory::{init_backend, probe_gpu_profile};
