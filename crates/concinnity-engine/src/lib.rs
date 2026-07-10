// src/lib.rs
//
// The runtime crate. Holds the world loop, the ECS, the GraphicsSystem renderer
// driver, audio, and physics. The GPU-free render-prep lives in
// concinnity-render and the hardware backends (Metal/DirectX/Vulkan/Win32) in
// concinnity-device; this crate drives them through a `Box<dyn RenderBackend>`
// from `concinnity_device::init_backend` and never names a concrete backend.
// Depends on concinnity-core/render/device (no concinnity-cook, no image
// decoders). The editor crate (concinnity-editor) drives this crate's App /
// renderer through the public API widened here; the modules the editor reaches
// into are `pub` so it can name their paths, but individual internals stay
// `pub(crate)` unless the editor specifically needs them.
pub mod assets;
pub mod blob;
pub mod ecs;

// Renderer-free foundation shared with the build/validate pipeline lives in
// concinnity-core. Re-export its modules under the historical crate::* paths so
// the rest of the client keeps resolving (crate::result / crate::gfx are the
// pre-existing slices; build / check / geometry / world join them here).
pub(crate) use concinnity_core::result;
pub(crate) use concinnity_core::{build, geometry, world};

pub mod app;
// Flat entry point for a shipped player: run a compiled world from a state dir.
// The runtime bin calls `concinnity_engine::run_from` rather than reaching
// through the `app::run` module path.
pub use app::run::run_from;
pub(crate) mod audio;
pub mod config;
pub mod gfx;
pub(crate) mod hud;
// The rayon job pool now lives in concinnity-render (the lowest layer the device
// backends and the client's animation fan-out share); re-export it under the
// historical crate::jobs path. `pub` so the editor's hot-reload decoder keeps
// reaching it through `concinnity_engine::jobs`.
pub use concinnity_render::jobs;
pub(crate) mod physics;
pub(crate) mod story;
pub(crate) mod text_input_system;
pub(crate) mod ui;

// Asset API
pub use assets::*;
