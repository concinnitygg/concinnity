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
//! it, and the runtime driver (concinnity-runtime) drives a frame through it,
//! with simulation systems queueing their GPU mutations as [`ops`] so the
//! submit path replays them in record order. The dev tooling
//! (concinnity-dev) is the third: asset hot-reload and the debug verbs call
//! the backend directly from its per-frame hook on the render thread, outside
//! that ordering guarantee, because they run between frames rather than inside
//! one.

pub mod backend;
pub mod buffers;
pub mod decal;
pub mod depth;
pub mod draw_list;
pub mod environment;
pub mod frame;
pub mod fullscreen;
pub mod hiz;
pub mod light;
pub mod live_preview;
pub mod material_entry;
pub mod material_params;
pub mod mipmap;
pub mod output;
pub mod overlay;
pub mod particles;
pub mod planar_reflection;
pub mod post;
pub mod probe;
pub mod record_pack;
pub mod render_graph;
pub mod rt_accel;
pub mod scene_flow;
pub mod scene_state;
pub mod shader;
pub mod shadow;
pub mod slots;
pub mod streaming;
pub mod temporal;
pub mod transparent;

/// The `#[repr(C)]` blocks the CPU uploads into the shaders, declared once for
/// every backend except where only one backend binds a block.
pub mod uniforms;
pub mod world_load;
