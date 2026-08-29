//! The backend-agnostic, GPU-free render-prep layer: the
//! `RenderBackend`/`SceneControl` trait seam the device backends implement, plus
//! the record builders, render graph, and CPU-side math that turn components
//! into GPU-ready data.
//!
//! Sits above [`crate::gfx`], which holds the layouts and the kernels this
//! prepares into, and owns no device or window handle of its own: a frame's
//! work is built here and handed to whichever backend implements the seam. The
//! device backends (concinnity-device) and the runtime driver
//! (concinnity-engine) are the two consumers.

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
pub mod keymap;
pub mod lights;
pub mod ltc;
pub mod mipmap;
pub mod ops;
pub mod overlay_maps;
pub mod parallel_ctx;
pub mod particles;
pub mod pass_timing;
pub mod planar_reflection;
pub mod reflection_probe;
pub mod render_graph;
pub mod rt_geom;
pub mod rt_refit;
pub mod rt_topology;
pub mod scene_flow;
pub mod scene_residency;
pub mod shaders;
pub mod shadow_schedule;
pub mod skinned_pool;
pub mod slang_programs;
pub mod slang_source;
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
