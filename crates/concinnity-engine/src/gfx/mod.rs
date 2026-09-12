//! The client's render layer: the runtime render systems (the renderer driver,
//! animation, camera controllers, draw list) and the client-only settings /
//! quality-preset resolution.
//!
//! What these drive lives below the client, in concinnity-core: the GPU data
//! layouts and render math in `concinnity_core::gfx`, and the
//! backend-agnostic render-prep (record builders, render graph, the
//! `RenderBackend` trait seam, and the GPU-free cursor / sprite / text /
//! lights / streaming layout helpers) in `concinnity_core::render`. Each is
//! named where it is used, by its owning crate.

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
/// Live application of the world's lighting assets to a running world, for an
/// editor previewing sun / fog / shadow / post-process edits without a rebuild.
pub mod lighting_preview;
/// The renderer's reading of one compiled `Material`: GPU uniforms plus the
/// texture-pool slots its references resolve to.
pub(crate) mod material_entry;
/// Live re-resolution of a `CharacterShape` against a running world's poses,
/// for an editor previewing slider edits without a rebuild.
pub mod shape_preview;
/// The renderer driver. An internal system (not a declarable asset), constructed
/// by `World::start` when the world declares a `GraphicsConfig`.
pub mod system;
// 2D overlay draw-list build + menu-state publish. Internal system,
// constructed alongside GraphicsSystem (same gate) and scheduled first.
pub(crate) mod overlay;
// Engine-side allocation authority for backend draw slots + pre-reserved
// skinned instances (the `RenderSlots` resource).
pub(crate) mod render_slots;
/// Asset-streaming home: the re-exported `no_std` policy core (`StreamPlanner`),
/// the `std` texture / mesh / chunk drivers it schedules, and the system that
/// drives them. `pub` for that system alone; everything else inside is
/// crate-private.
pub mod streaming;
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
