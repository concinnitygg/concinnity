//! The runtime crate. Holds the world loop, the ECS, the GraphicsSystem renderer
//! driver, audio, and physics. The GPU-free render-prep lives in
//! `concinnity_core::render` and the hardware backends (Metal/DirectX/Vulkan/
//! Win32) in concinnity-device; this crate drives them through a
//! `Box<dyn RenderBackend>` from `concinnity_device::init_backend` and never
//! names a concrete backend. Depends on concinnity-core and concinnity-device
//! (no concinnity-cook, no image decoders). The editor crate (concinnity-dev)
//! drives this crate's App / renderer through the public API widened here; the
//! modules the editor reaches
//! into are `pub` so it can name their paths, but individual internals stay
//! `pub(crate)` unless the editor specifically needs them.
pub mod blob;
pub mod components;
pub mod ecs;
mod heap;

#[cfg(test)]
mod bench;

// Renderer-free foundation shared with the build/validate pipeline: the result
// vocabulary, the payload decoders, and the runtime geometry generators, all
// from concinnity-core. Re-exported under the historical crate::* paths so the
// rest of the client keeps resolving. world.jsonl I/O lives in concinnity-cook
// (authoring), which the runtime does not link.
pub(crate) use concinnity_core::{bake, geometry, result};

// The access-declaration mask builders, reached crate-wide as
// `crate::component_mask!` / `crate::resource_mask!`.
pub(crate) use ecs::access_ids::{component_mask, resource_mask};

pub mod app;
// Flat entry point for a shipped player: run a compiled world from a state dir.
// The runtime bin calls `concinnity_engine::run_from` rather than reaching
// through the `app::run` module path.
pub use app::run::{BlobSource, run_from};
// The application surface a host embeds: construct an App, populate its world,
// and drive it with `App::run` / `App::run_with`. Exported flat so the
// `concinnity` facade crate re-exports these under its own root.
pub use app::run::{PipelineMode, RunOptions, init_logging};
pub use app::startup_error::StartupError;
pub use app::state::App;
// Redirect runtime-writable state (`saves/` + `settings`) before `run_from`
// when the content dir is read-only. Exported beside `run_from` so the runtime
// bin's entire entry API lives on this crate.
pub use concinnity_host::store::paths;
pub use concinnity_host::store::paths::set_writable_state_dir;

/// The shader platform this build's rendering backend consumes. The backend is
/// this crate's own compile-time choice, so the crate that knows it is the one
/// that states it: the editor and the cook are handed the value rather than
/// resolving a backend of their own.
pub mod platform;

// Export-time compilation of the built-in shaders into the cache segment a
// bundle ships, for backends that compile them at renderer init. Re-exported
// so `cn export` reaches it without a direct device dependency.
#[cfg(any(backend_dx, backend_vk))]
pub use concinnity_device::precompile::{
    Report as ShaderPrecompileReport, precompile_builtin_shaders,
};
mod device;
/// Whether this build links a rendering backend. A build with no backend
/// feature has none, so the only loop that can run a world is a headless one.
pub use device::AVAILABLE as HAS_RENDER_BACKEND;
pub(crate) mod cbor_file;
pub(crate) mod config;
/// Crash reporting: panic hook, native fault capture, local report files.
/// `pub` so the binaries install the hooks and the editor composes the
/// recent-log ring layer into its tracing subscriber.
pub mod crash;
pub mod shutdown;
// 3D positional sound and screen-triggered cues, and the one module that names
// kira.
pub(crate) mod audio;
// The standalone startup-error window, shown when a fatal startup failure
// happens before any world exists.
pub(crate) mod error_screen;
pub mod gfx;
pub(crate) mod hud;
pub(crate) mod input;
// The rigid-body simulation driver: builds a `concinnity_core::physics::Simulation`
// from the world's physics content and steps it on the fixed tick.
pub(crate) mod physics;
// Declarative logic (Behavior components + the shared world variables
// store), scheduled before SpawnSystem so its requests apply the same tick.
pub(crate) mod behavior;
// The rayon job pool, re-exported under the historical crate::jobs path. `pub`
// so the editor's hot-reload decoder keeps reaching it through
// `concinnity_engine::jobs`.
pub use concinnity_host::thread::jobs;
/// Runtime resource tables (per-kind, handle-indexed views of the blob's resource
/// stream). `pub` so the editor's in-memory build path can construct the tables it
/// inserts into the world, mirroring the shipped-runtime loader.
pub mod resource;
// Runtime entity churn (Lifetime/Spawner ticks + spawn/despawn/reparent
// drains), scheduled immediately before GraphicsSystem.
pub(crate) mod spawn;
pub(crate) mod story;
pub(crate) mod text_input_system;
pub(crate) mod ui;

// Asset API
pub use components::*;
