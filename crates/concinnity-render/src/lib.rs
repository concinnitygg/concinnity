//! The backend-agnostic, GPU-free render-prep layer. Holds the
//! `RenderBackend`/`SceneControl` trait seam the device backends implement, plus
//! the record builders, render graph, and CPU-side math that turn asset
//! components into GPU-ready data. Depends on the vocabulary (concinnity-core)
//! and the CPU compute over it (concinnity-cpu); owns no device or window
//! handle. The device backends (concinnity-device) and the runtime driver
//! (concinnity-engine) both build on top of this crate.

// GPU-layout + render-math vocabulary, re-exported at the crate root so the
// render-prep modules reach them as `crate::<type>`.
pub use concinnity_core::gfx::{
    auto_exposure, chunk_coord, frustum, profile, render_types, rt_reflections, ssao, ssgi, ssr,
};
// The mesh payload codec, which is CPU work rather than vocabulary.
pub use concinnity_cpu::gfx::mesh_payload;

// The rest of what the render-prep modules reach by their `crate::` paths:
// asset data types (`Decal`, `ParticleEmitter`, `InputKey`, ...) and the stable
// `AssetId` newtype from the vocabulary, and from the compute crate the
// environment-map bake the reflection-probe payload builder calls plus the
// glass-quad generator.
pub(crate) use concinnity_core::{assets, ecs};
pub(crate) use concinnity_cpu::{build, geometry};

pub mod area_light;
pub mod backend;
pub mod backend_init;
pub mod bvh;
pub mod call_buffer;
pub mod chunk_window;
pub mod csm;
pub mod cursor;
pub mod decal;
pub mod display_mode;
pub mod draw_slot;
pub mod error;
pub mod feedback;
pub mod fullscreen;
pub mod hdr_output;
pub mod input;
// The job pool lives in concinnity-cpu so sim-side crates (physics) can share
// it; re-exported here for the backends and the engine's historical path.
pub use concinnity_cpu::jobs;
pub mod keymap;
pub mod lights;
pub mod ltc;
pub mod mat;
pub mod mipmap;
pub mod ops;
pub mod parallel_ctx;
pub mod particles;
pub mod planar_reflection;
pub mod reflection_probe;
pub mod render_graph;
pub mod rt_geom;
pub mod rt_refit;
pub mod rt_topology;
pub mod scene_flow;
pub mod scene_residency;
pub mod shadow_schedule;
pub mod skinned_pool;
pub mod slot_rewrites;
pub mod snapshot;
pub mod spot_shadow;
pub mod sprite;
pub mod streaming;
pub mod text;
pub mod transparent;

/// The `#[repr(C)]` blocks the CPU uploads into the single-source `.slang`
/// shaders, declared once for every backend (see `uniforms/mod.rs`).
pub mod uniforms;

pub mod volumetric_fog;

/// GPU-free host-side layout contract for the Metal backend's shader structs
/// (uniform structs, math, shader-layout asserts). Metal-specific but device-free,
/// so it is compiled unconditionally and its layout tests run on every platform's
/// CI. The Metal backend (concinnity-device) re-exports it under its own `metal`.
pub mod metal;

/// The same for the DirectX and Vulkan backends: their repr(C) uniform / probe
/// structs + GPU-timing slot arithmetic (mirrored in the HLSL / GLSL shaders).
/// Backend-specific but device-free (plain repr(C), no windows/ash types), so
/// they compile unconditionally and their layout tests count toward coverage.
/// The DirectX / Vulkan backends (concinnity-device) re-export them under their
/// own `directx` / `vulkan`.
pub mod directx;
pub mod vulkan;
