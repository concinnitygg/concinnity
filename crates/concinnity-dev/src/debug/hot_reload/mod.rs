//! Asset / shader / world.jsonl hot-reload machinery for dev sessions
//! (`cn debug` and `cn editor`). Moved out of the library into the binary
//! tree: the watcher, off-thread decode, and the reload passes are driven once
//! per frame from a `FrameHook::tick` (see `driver::HotReloadDriver`). The
//! passive source catalogs these consume are captured by graphics init and
//! reached through the engine's live-edit contract
//! (`concinnity_engine::live_edit::hot_reload_sources`); the per-frame backend
//! and pushed fog come from `concinnity_engine::live_edit::render_handoff`.
//!
//! Split by responsibility:
//!   driver     `HotReloadDriver`, the per-frame drive + ECS effect apply
//!   state      `AssetHotReloadState` + decode result types + `run_frame` entry
//!   watcher    the `notify` filesystem watcher
//!   decode     off-thread payload decode + poll/apply (textures, meshes, IBL)
//!   passes     world.jsonl / ProceduralMesh / VolumetricFog / story reload
//!   shader     per-Shader recompile: pipeline swap on the frame thread, and a
//!              live override for a Shader whose scene is not loaded
//!   sdf        per-SdfVolume field recompile: one compile per field and flag
//!              set, swapped into every volume reading the field
//!   compile_queue  the off-thread compiles both run, newest save wins
//!   files      which subjects a filesystem event touches
//!   report     what became of each reload: log, toast, and the board an
//!              editor session's panels read
//!   animation  file-backed Animation clip re-import into the AnimationSystem
//!   signals    `ReloadSignals`, the reload requests the watcher and the
//!              `reload-assets` tool call raise for the frame thread
//!   world_path the session's world.jsonl path, shared with the host that switches worlds

mod animation;
mod compile_queue;
mod decode;
mod driver;
mod files;
mod passes;
mod report;
mod sdf;
mod shader;
mod signals;
mod state;
mod watcher;
mod world_path;

#[cfg(test)]
mod tests;

pub(crate) use driver::HotReloadDriver;
pub(crate) use report::{
    Latest, ReloadFailure, ReloadOutcome, ReloadReports, ReloadSubject, ReportBoard,
};
pub(crate) use signals::ReloadSignals;
pub(crate) use world_path::WorldPathHandle;
// The editor's tests publish reports of their own.
#[cfg(test)]
pub(crate) use report::ReloadReport;
