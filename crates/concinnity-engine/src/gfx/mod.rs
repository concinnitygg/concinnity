//! The client's render layer. Everything below the client sits in
//! concinnity-core and is re-exported here under the historical
//! `crate::gfx::<module>` paths: the GPU data layouts and render math (camera,
//! frustum, post-process settings) plus the CPU kernels over them from
//! `concinnity_core::gfx`, and the backend-agnostic render-prep (record
//! builders, render graph, trait seam, and the GPU-free cursor / sprite / text /
//! lights / streaming layout helpers) from `concinnity_core::render`.
//!
//! What remains declared here are the runtime render systems (the renderer
//! driver, animation, camera controllers, draw list) and the client-only
//! settings/quality-preset resolution. The re-exports are `pub` so the editor
//! crate can reach them through `concinnity_engine::gfx::*` (e.g. shader-layout
//! reflection); `chunk_coord` is named only by the chunk-streaming drive, so it
//! stays crate-private.
pub(crate) use concinnity_core::gfx::chunk_coord;
pub use concinnity_core::gfx::{
    anim_graph, auto_exposure, camera, font, frustum, ik, lines, lod, mesh_payload, mesh_seed,
    morph_weights, pose_blend, pose_scratch, profile, proportions, render_types, root_motion,
    rt_reflections, skeleton, ssao, ssgi, ssr, transform, transform_propagation, view_modes,
};

// Render-prep from `concinnity_core::render` that the client's own systems consume (the
// device backends reach the rest through concinnity-device's own bridge, not
// this crate). `pub` for the pieces the editor / app crates name, `pub(crate)`
// for the rest.
pub use concinnity_core::render::{
    backend, backend_init, decal, error, feedback, input, ops, particles, scene_flow,
    scene_residency, snapshot, volumetric_fog,
};
pub(crate) use concinnity_core::render::{
    call_buffer, chunk_window, cursor, display_mode, keymap, lights, overlay_maps, sprite, text,
};
// Seeded / driven by the client's GraphicsSystem.
pub use concinnity_core::render::draw_slot;
pub(crate) use concinnity_core::render::{planar_reflection, reflection_probe};

// The bundled glyph atlas baked into the binary: the face the startup error
// screen draws with, and the fallback for a world whose labels name no Font.
pub(crate) mod builtin_font;

/// Skeletal animation playback. Internal system, constructed by `World::start`
/// when the world declares any `Animation`; produces per-frame skinning matrices.
/// `pub` so the editor crate can drive the clip hot-reload through the
/// `AnimationSystem` setter API.
pub mod animation;
/// First-person / fly-through camera controller. Internal system, constructed by
/// `World::start` from a `Camera3D`'s controller settings. `pub` so the editor
/// crate can zero the controller's velocity behind an externally driven pose.
pub mod camera_controller;
pub(crate) mod draw_list;
/// Live reassignment of a running world's draw slots (their material and cull
/// distance), for an editor previewing a Prop edit without a rebuild.
pub mod draw_preview;
/// The renderer driver. An internal system (not a declarable asset), constructed
/// by `World::start` when the world declares a `GraphicsConfig`.
pub mod graphics_system;
/// Live application of the world's lighting assets to a running world, for an
/// editor previewing sun / fog / shadow / post-process edits without a rebuild.
pub mod lighting_preview;
/// The renderer's reading of one compiled `Material`: GPU uniforms plus the
/// texture-pool slots its references resolve to.
pub(crate) mod material_entry;
/// Live re-resolution of a `CharacterShape` against a running world's poses,
/// for an editor previewing slider edits without a rebuild.
pub mod shape_preview;
// Per-frame input sampling + FrameInput publish. Internal system, constructed
// alongside GraphicsSystem (same gate) and scheduled immediately after it.
pub(crate) mod input_system;
// 2D overlay draw-list build + menu-state publish. Internal system,
// constructed alongside GraphicsSystem (same gate) and scheduled first.
pub(crate) mod overlay;
// Engine-side allocation authority for backend draw slots + pre-reserved
// skinned instances (the `RenderSlots` resource).
pub(crate) mod render_slots;
// SettingCommand / SceneCommand application + settings snapshot ownership.
// Internal system, constructed alongside GraphicsSystem (same gate) and
// scheduled just before it.
pub(crate) mod settings_system;
// Asset-streaming home: the re-exported `no_std` policy core (`StreamPlanner`)
// plus the `std` texture / mesh / chunk drivers it schedules.
pub(crate) mod streaming;
/// Asset-streaming drive (texture / mesh / voxel-world chunk pools) + the
/// camera-relative view publish. Internal system, constructed alongside
/// GraphicsSystem (same gate) and scheduled immediately before it. `pub` so the
/// editor's debug server can name `StreamingStats` (its state lives in the parked
/// `StreamingState` resource, read via `World::streaming_stats`).
pub mod streaming_system;
// Recording mock RenderBackend + the GraphicsSystem test-injection hooks,
// compiled only into the unit-test binary. Implements `core::render`'s
// RenderBackend seam on a client-local type and carries a `config::Settings`,
// so it stays with the GraphicsSystem tests that consume it.
pub(crate) mod look_controls;
#[cfg(test)]
pub(crate) mod mock_backend;
pub(crate) mod quality_preset;
// How the world's authored render settings resolve against the user's persisted
// settings-menu choices and the active quality preset's ceiling.
pub(crate) mod render_config;
pub(crate) mod setting_action;
pub(crate) mod settings;
// Handle -> asset id bridge for SkinnedMesh correlation references, published by
// GraphicsSystem and read by the animation / third-person systems.
pub(crate) mod skinned_mesh_map;
// Third-person character controller. Internal system, constructed instead of
// Camera3DSystem when the controlling camera's controller has a `follow` block.
pub(crate) mod third_person;
