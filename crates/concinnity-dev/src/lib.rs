//! concinnity-dev: the dev tooling library.
//!
//! Everything the `concinnity` binary does, minus the argv it does it from:
//! the world authoring / in-memory build code, the implementations behind each
//! subcommand, the asset-reference generator, bundle packaging, the in-engine
//! editor HUD, the localhost debug server, and the interpreted (`cn debug`) run
//! loop.
//!
//! The binary is the clap command tree and nothing else. Everything it dispatches
//! into is here, which is what lets the same entry points serve an out-of-tree
//! host that has no argv at all.

// Bridge: re-export the runtime/core modules the authoring, editor, and debug
// code names under crate::* so their `crate::<module>` import paths resolve.
// world.jsonl I/O lives in the compiler (concinnity-cook), not core.
pub(crate) use concinnity_cook::authoring::world;
pub(crate) use concinnity_engine::{app, blob, components, ecs, gfx, jobs, resource};

/// The shader platform `cn` cooks worlds for: the backend the runtime linked
/// into this same binary consumes, so a world built here plays here.
pub fn cook_platform() -> concinnity_core::platform::Platform {
    concinnity_engine::platform::current()
}

// Authoring / in-memory build. Its exports below are the surface the
// out-of-tree Swift app's FFI crate embeds; the `cn` binary uses only part of
// it, so some entry points have no in-workspace caller.
mod authoring;

// The in-engine editor HUD, the localhost debug server, the MCP bridge over
// it, the interpreted run loop, and animation clip hot-reload.
mod anim_reload;
mod debug;
mod debug_hook;
mod editor;
mod mcp;
mod run;

/// The implementations behind each `cn` subcommand, and the only place in this
/// crate that writes to stdout.
pub mod command;
/// The asset reference pages, generated from the engine's own schema sources.
pub mod docs;
/// Packaging a built world into a distributable bundle.
pub mod export;
/// The project a dev session works on: the one state tree its builds, runs, and
/// editor sessions address, opened by the binary at startup.
pub mod project;

// Process-global test serialization lock; test builds only.
#[cfg(test)]
mod test_support;

// Dev-session entry points, consumed by the `concinnity` binary: the debug
// server + interpreted run (`cn debug`), the in-engine editor (`cn editor`),
// the debug-server WebSocket client (`cn debug send/screenshot/...`), and the
// MCP stdio bridge that serves that same protocol to an agent (`cn mcp`).
pub use debug::{WatchTarget, client as debug_client};
pub use editor::run_editor;
pub use mcp::run as run_mcp;
pub use run::run_debug;

// The authoring API
pub use authoring::{
    add_to_path, arg_value_to_json, build_world_from_path, build_world_from_str,
    build_world_to_disk, check_at_path, check_from_str, rm_at_path, spec_args, spec_to_value,
    world_from_loaded, world_template_entries,
};
pub use concinnity_cook::authoring::world::{parse_world_jsonl, write_world_jsonl};
pub use concinnity_cook::{build_pipeline_from_str, validate_asset, validate_world_jsonl};
