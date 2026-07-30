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
// pre-existing slices; build / geometry join them here). world.jsonl I/O moved
// to concinnity-cook (authoring), which the runtime does not link.
pub(crate) use concinnity_core::result;
pub(crate) use concinnity_core::{build, geometry};

pub mod app;
// Flat entry point for a shipped player: run a compiled world from a state dir.
// The runtime bin calls `concinnity_engine::run_from` rather than reaching
// through the `app::run` module path.
pub use app::run::run_from;
// Redirect runtime-writable state (`saves/` + `settings`) before `run_from`
// when the content dir is read-only. Exported beside `run_from` so the runtime
// bin's entire entry API lives on this crate.
pub use concinnity_core::paths::set_writable_state_dir;

// Export-time compilation of the built-in shaders into a bundle's
// shader-cache/, for backends that compile them at renderer init. Re-exported
// so `cn export` reaches it without a direct device dependency.
#[cfg(any(backend_dx, backend_vk))]
pub use concinnity_device::precompile::{
    Report as ShaderPrecompileReport, precompile_builtin_shaders,
};
pub mod config;
pub mod gfx;
pub(crate) mod hud;
pub(crate) mod input;
// Declarative logic (Behavior components + the shared world variables
// store), scheduled before SpawnSystem so its requests apply the same tick.
pub(crate) mod behavior;
// The rayon job pool now lives in concinnity-render (the lowest layer the device
// backends and the client's animation fan-out share); re-export it under the
// historical crate::jobs path. `pub` so the editor's hot-reload decoder keeps
// reaching it through `concinnity_engine::jobs`.
pub use concinnity_render::jobs;
// Runtime resource tables (per-kind, handle-indexed views of the blob's resource
// stream). `pub` so the editor's in-memory build path can construct the tables it
// inserts into the world, mirroring the shipped-runtime loader.
pub mod resource;
// Runtime entity churn (Lifetime/Spawner ticks + spawn/despawn/reparent
// drains), scheduled immediately before GraphicsSystem.
pub(crate) mod spawn;
pub(crate) mod story;
pub(crate) mod text_input_system;
pub(crate) mod ui;

// Asset API
pub use assets::*;
