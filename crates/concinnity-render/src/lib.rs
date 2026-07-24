// src/lib.rs
//
// The backend-agnostic, GPU-free render-prep layer. Holds the
// `RenderBackend`/`SceneControl` trait seam the device backends implement, plus
// the record builders, render graph, and CPU-side math that turn asset
// components into GPU-ready data. Depends on concinnity-core alone; owns no
// device or window handle. The device backends (concinnity-device) and the
// runtime driver (concinnity-engine) both build on top of this crate.

// Core GPU-layout + render-math types, re-exported at the crate root so the
// render-prep modules reach them as `crate::<type>` (they were `crate::gfx::<type>`
// in the client, where gfx/mod.rs re-exported the same core modules).
pub use concinnity_core::gfx::{
    auto_exposure, chunk_coord, frustum, mesh_payload, profile, render_types, rt_reflections, ssao,
    ssgi, ssr,
};

// Backend-agnostic foundation the moved modules reach by their historical
// crate:: paths: asset data types (`Decal`, `ParticleEmitter`, `Key`, ...), the
// stable `AssetId` newtype (`ecs::asset_id`), and the runtime environment-map
// bake helpers the reflection-probe payload builder calls (`build::environment_map`).
pub(crate) use concinnity_core::{assets, build, ecs, geometry};

pub mod area_light;
pub mod backend;
pub mod backend_init;
pub mod bvh;
pub mod chunk_window;
pub mod csm;
pub mod cursor;
pub mod decal;
pub mod display_mode;
pub mod draw_slot;
pub mod fullscreen;
pub mod hdr_output;
pub mod input;
pub mod jobs;
pub mod keymap;
pub mod lights;
pub mod ltc;
pub mod mat;
pub mod mipmap;
pub mod parallel_ctx;
pub mod particles;
pub mod planar_reflection;
pub mod reflection_probe;
// Each backend's barrier + resource-aliasing path consumes only a subset of the
// graph today (buffer resources and version tracking are not wired into a
// backend executor yet), so a portion of the builder internals stays unused.
// Drop this allow once every backend's barrier + aliasing path consumes the
// full builder surface.
#[allow(dead_code)]
pub mod render_graph;
pub mod rt_geom;
pub mod rt_topology;
pub mod scene_flow;
pub mod shadow_schedule;
pub mod skinned_pool;
pub mod slot_rewrites;
pub mod spot_shadow;
pub mod sprite;
pub mod streaming;
pub mod text;
pub mod transparent;
pub mod volumetric_fog;

// GPU-free host-side layout contract for the Metal backend's shader structs
// (uniform structs, math, shader-layout asserts). Metal-specific but device-free,
// so it is compiled unconditionally and its layout tests run on every platform's
// CI. The Metal backend (concinnity-device) re-exports it under its own `metal`.
pub mod metal;

// The same for the DirectX and Vulkan backends: their repr(C) uniform / probe
// structs + GPU-timing slot arithmetic (mirrored in the HLSL / GLSL shaders).
// Backend-specific but device-free (plain repr(C), no windows/ash types), so
// they compile unconditionally and their layout tests count toward coverage.
// The DirectX / Vulkan backends (concinnity-device) re-export them under their
// own `directx` / `vulkan`.
pub mod directx;
pub mod vulkan;
