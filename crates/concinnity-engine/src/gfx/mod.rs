// src/gfx.rs
//
// The client's render layer. The backend-agnostic render-prep helpers (record
// builders, render graph, trait seam, CPU-side render math, plus the GPU-free
// cursor / sprite / text / lights / streaming layout helpers) now live in
// concinnity-render and are re-exported below under the historical
// crate::gfx::<module> paths so the rest of the client and the device backends
// keep resolving. What remains declared here are the game/runtime render systems
// (the renderer driver, animation, camera controllers, draw list) and the
// client-only settings/quality-preset resolution.
//
// The GPU data layouts and render math (camera, frustum, post-process settings)
// live in concinnity-core; the CPU kernels over them (mesh payloads, skinning,
// IK, line expansion, the animation cursor) in concinnity-cpu. Both are
// re-exported here so the crate::gfx::<module> paths keep resolving.
// `pub` so the editor crate can reach them through `concinnity_engine::gfx::*`
// (e.g. shader-layout reflection); `chunk_coord` is named only by the
// chunk-streaming drive, so it stays crate-private.
pub(crate) use concinnity_core::gfx::chunk_coord;
pub use concinnity_core::gfx::lod_select as lod;
pub use concinnity_core::gfx::{
    camera, frustum, profile, render_types, root_motion, rt_reflections, ssao, ssgi, ssr,
    view_modes,
};
pub use concinnity_cpu::gfx::{
    anim_graph, auto_exposure, ik, lines, mesh_payload, mesh_seed, skinning,
};

// Render-prep from concinnity-render that the client's own systems consume (the
// device backends reach the rest through concinnity-device's own bridge, not
// this crate). `pub` for the pieces the editor / app crates name, `pub(crate)`
// for the rest.
pub use concinnity_render::{
    backend, backend_init, decal, error, input, particles, scene_flow, scene_residency,
    volumetric_fog,
};
pub(crate) use concinnity_render::{
    chunk_window, cursor, display_mode, keymap, lights, sprite, text,
};
// Seeded / driven by the client's GraphicsSystem on Metal today; the other
// backends' probe + planar ports have not landed, so unused off macOS.
#[cfg_attr(not(target_os = "macos"), allow(unused_imports))]
pub(crate) use concinnity_render::{planar_reflection, reflection_probe};
// Consumed only by the runtime-spawn unit tests (draw-slot allocator +
// skinned-instance pool); the production spawn path lives in the backends.
#[cfg(test)]
pub(crate) use concinnity_render::{draw_slot, skinned_pool};

// Skeletal animation playback. Internal system, constructed by `World::start`
// when the world declares any `Animation`; produces per-frame skinning matrices.
// `pub` so the editor crate can drive the clip hot-reload through the
// `AnimationSystem` setter API.
pub mod animation;
// First-person / fly-through camera controller. Internal system, constructed by
// `World::start` from a `Camera3D`'s controller settings.
pub(crate) mod camera_controller;
pub mod draw_list;
// Transform -> GlobalTransform propagation down the Parent hierarchy, and the
// runtime reparent that recomposes it. Driven per frame by GraphicsSystem.
pub(crate) mod transform_propagation;
// The renderer driver. An internal system (not a declarable asset), constructed
// by `World::start` when the world declares a `GraphicsConfig`.
pub mod graphics_system;
// Per-frame input sampling + FrameInput publish. Internal system, constructed
// alongside GraphicsSystem (same gate) and scheduled immediately after it.
pub(crate) mod input_system;
// 2D overlay draw-list build + menu-state publish. Internal system,
// constructed alongside GraphicsSystem (same gate) and scheduled first.
pub(crate) mod overlay;
// SettingCommand / SceneCommand application + settings snapshot ownership.
// Internal system, constructed alongside GraphicsSystem (same gate) and
// scheduled just before it.
pub(crate) mod settings_system;
// Asset-streaming home: the re-exported `no_std` policy core (`StreamPlanner`)
// plus the `std` texture / mesh / chunk drivers it schedules.
pub(crate) mod streaming;
// Asset-streaming drive (texture / mesh / voxel-world chunk pools) + the
// camera-relative view publish. Internal system, constructed alongside
// GraphicsSystem (same gate) and scheduled immediately before it. `pub` so the
// editor's debug server can name `StreamingStats` (its state lives in the parked
// `StreamingState` resource, read via `World::streaming_stats`).
pub mod streaming_system;
// Recording mock RenderBackend + the GraphicsSystem test-injection hooks,
// compiled only into the unit-test binary. Implements concinnity-render's
// RenderBackend seam on a client-local type and carries a `config::Settings`,
// so it stays with the GraphicsSystem tests that consume it.
pub(crate) mod look_controls;
#[cfg(test)]
pub(crate) mod mock_backend;
pub(crate) mod quality_preset;
pub(crate) mod setting_action;
pub(crate) mod settings;
// Handle -> asset id bridge for SkinnedMesh correlation references, published by
// GraphicsSystem and read by the animation / third-person systems.
pub(crate) mod skinned_mesh_map;
// Third-person character controller. Internal system, constructed instead of
// Camera3DSystem when the controlling camera's controller has a `follow` block.
pub(crate) mod third_person;

// First-person camera controller. Currently unreferenced scaffolding
// (Camera3DSystem drives the camera); kept for a future stateful controller.
#[cfg(not(target_os = "macos"))]
pub(crate) mod fps_controller;
