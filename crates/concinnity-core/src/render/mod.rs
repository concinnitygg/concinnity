//! The backend-agnostic, GPU-free render-prep layer: the
//! `RenderBackend`/`SceneControl` trait seam the device backends implement, plus
//! the per-frame record builders, render graph, and passes that turn components
//! into GPU-ready data.
//!
//! The rule that separates this from [`crate::gfx`]: gfx holds `repr(C)`
//! layouts and pure math kernels; render holds the per-frame builders, the
//! passes, and the backend seam. This layer owns no device or window handle of
//! its own: a frame's work is built here and handed to whichever backend
//! implements the seam. The one layout exception is [`uniforms`]: a block only
//! one backend binds, or whose shader is still per backend, stays beside the
//! shared blocks in a per-backend child.
//!
//! Three crates consume it. The device backends (concinnity-device) implement
//! it, and the runtime driver (concinnity-engine) drives a frame through it,
//! with simulation systems queueing their GPU mutations as [`ops`] so the
//! submit path replays them in record order. The dev tooling
//! (concinnity-dev) is the third: asset hot-reload and the debug verbs call
//! the backend directly from `DebugHook::tick` on the render thread, outside
//! that ordering guarantee, because they run between frames rather than inside
//! one.

pub mod area_light;
pub mod backend;
pub mod backend_init;
pub mod call_buffer;
pub mod chunk_window;
pub mod csm;
pub mod cursor;
pub mod decal;
pub mod draw_slot;
pub mod error;
pub mod feedback;
pub mod frame_dirty;
pub mod fullscreen;
pub mod hdr_output;
pub mod hiz_spd;
pub mod lights;
pub mod ltc;
pub mod mipmap;
pub mod model_history;
pub mod ops;
pub mod overlay_maps;
pub mod parallel_ctx;
pub mod particles;
pub mod pass_timing;
pub mod planar_reflection;
pub mod post;
pub mod probe_book;
pub mod reflection_probe;
pub mod render_graph;
pub mod rt_geom;
pub mod rt_refit;
pub mod rt_topology;
pub mod scene_flow;
pub mod scene_residency;
pub mod shader_programs;
pub mod shader_source;
pub mod shaders;
pub mod shadow_bias;
pub mod shadow_schedule;
pub mod skinned_pool;
pub mod skinned_slots;
pub mod slot_rewrites;
pub mod snapshot;
pub mod spot_shadow;
pub mod sprite;
pub mod streaming;
pub mod text;
pub mod transparent;

/// The `#[repr(C)]` blocks the CPU uploads into the shaders, declared once for
/// every backend except where only one backend binds a block.
pub mod uniforms;

pub mod volumetric_fog;
