//! Asset / shader / world.jsonl hot-reload machinery for dev sessions
//! (`cn debug` and `cn editor`). Moved out of the library into the binary
//! tree: the watcher, off-thread decode, and the reload passes are driven once
//! per frame from a `DebugHook::tick` (see `driver::HotReloadDriver`). The
//! passive source catalogs these consume are captured at
//! `GraphicsSystem::init` and live in the library
//! (`concinnity_engine::gfx::system::hot_reload_sources`); the per-frame backend
//! and pushed fog come from `concinnity_engine::ecs::render_handoff`.
//!
//! Split by responsibility:
//!   driver     `HotReloadDriver`, the per-frame drive + ECS effect apply
//!   state      `AssetHotReloadState` + decode result types + `run_frame` entry
//!   watcher    the `notify` filesystem watcher
//!   decode     off-thread payload decode + poll/apply (textures, meshes, IBL)
//!   passes     world.jsonl / ProceduralMesh / VolumetricFog / story reload
//!   shader     per-Shader recompile: off-thread compile, newest save wins,
//!              pipeline swap on the frame thread, live override for a Shader
//!              whose scene is not loaded, and each Shader's latest outcome
//!              for an editor session to show
//!   animation  file-backed Animation clip re-import into the AnimationSystem
//!   pending    process-wide world.jsonl / story / Animation "changed" flags and
//!              the pending Shader set
//!   world_path the session's world.jsonl path, shared with the host that switches worlds

mod animation;
mod decode;
mod driver;
mod passes;
mod pending;
mod shader;
mod state;
mod watcher;
mod world_path;

#[cfg(test)]
mod tests;

pub(crate) use driver::HotReloadDriver;
pub(crate) use pending::{
    mark_all_shaders_pending, set_pending_animations, set_pending_stories, set_pending_world,
};
pub(crate) use shader::{ReportBoard, ShaderReloadFailure, ShaderReloadOutcome, ShaderReports};
pub(crate) use world_path::WorldPathHandle;
// The editor's tests publish reports of their own.
#[cfg(test)]
pub(crate) use shader::ShaderReloadReport;
// The `reload-assets` dispatch test drains the sibling reload flags the handler
// raises so they don't leak into other tests; only that test needs them.
#[cfg(test)]
pub(crate) use pending::{
    take_pending_animations, take_pending_shaders, take_pending_stories, take_pending_world,
};
