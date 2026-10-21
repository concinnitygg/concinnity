//! concinnity-dev: the dev tooling library.
//!
//! Everything the `concinnity` binary is, apart from the process it runs in:
//! the command line it parses, the world authoring / in-memory build code, the
//! implementations behind each subcommand, the asset-reference generator,
//! bundle packaging, the in-engine editor HUD, the localhost debug server, and
//! the interpreted (`cn debug`) run loop.
//!
//! The binary declares the global allocator, resolves the project's state tree,
//! installs the crash hooks, and hands the rest to [`cli::Invocation`].

/// The shader platform `cn` cooks a world for when it plays in this process:
/// the backend this process renders with, so a world built here plays here.
pub fn cook_platform() -> concinnity_engine::platform::Platform {
    concinnity_runtime::platform::current()
}

/// The shader platforms `cn` cooks a world's blobs for: every backend this
/// binary can render with, so the blobs play on whichever one a launch selects.
pub fn cook_targets() -> concinnity_engine::platform::PlatformSet {
    concinnity_runtime::platform::compiled_set()
}

// Authoring / in-memory build.
mod authoring;
// The live status a world build shows on the terminal.
mod build_status;

// The in-engine editor HUD, the localhost debug server, the MCP transport it
// speaks, and the interpreted run loop.
mod debug;
mod editor;
mod frame_hook;
mod mcp;
mod run;

/// The `cn` command line: the arguments a run was given, and the command they
/// name.
pub mod cli;
/// The implementations behind each `cn` subcommand, and the only place in this
/// crate that writes to stdout.
pub(crate) mod command;
/// The asset reference pages, generated from the engine's own schema sources.
pub(crate) mod docs;
/// Packaging a built world into a distributable bundle.
pub(crate) mod export;
pub mod project;

// Process-global test serialization lock; test builds only.
#[cfg(test)]
mod test_support;

// Dev-session entry points, dispatched by `cli`: the debug
// server + interpreted run (`cn debug`), the in-engine editor (`cn editor`),
// and the MCP stdio bridge that forwards an agent's tool calls to a running
// app (`cn mcp`).
pub(crate) use editor::run_editor;
pub(crate) use mcp::run as run_mcp;
pub(crate) use run::run_debug;
