// src/debug/hot_reload/mod.rs
//
// Asset / shader / world.jsonl hot-reload machinery for dev sessions
// (`cn debug` and `cn editor`). Moved out of the library into the binary
// tree: the watcher, off-thread decode, and the reload passes are driven once
// per frame from a `DebugHook::tick` (see `driver::HotReloadDriver`). The
// passive source catalogues these consume are captured at
// `GraphicsSystem::init` and live in the library
// (`crate::gfx::graphics_system::hot_reload_sources`); the per-frame backend
// + Prop-tracking handle comes from `GraphicsSystem::hot_reload_apply_parts`.
//
// Split by responsibility:
//   driver   `HotReloadDriver`, the per-frame drive + ECS effect apply
//   state    `AssetHotReloadState` + decode result types + `run_frame` entry
//   watcher  the `notify` filesystem watcher
//   decode   off-thread payload decode + poll/apply (textures, meshes, IBL)
//   passes   world.jsonl / ProceduralMesh / VolumetricFog / Shader reload
//   pending  process-wide world.jsonl / Shader "changed" flags

mod decode;
mod driver;
mod passes;
mod pending;
mod state;
mod watcher;

#[cfg(test)]
mod tests;

pub(crate) use driver::HotReloadDriver;
pub(crate) use pending::{set_pending_shader_stages, set_pending_stories, set_pending_world};

// The `reload-assets` dispatch test drains the sibling reload flags the handler
// raises so they don't leak into other tests; only that test needs them.
#[cfg(test)]
pub(crate) use pending::{take_pending_shader_stages, take_pending_stories, take_pending_world};
