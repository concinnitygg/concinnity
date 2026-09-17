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

/// The shader platform `cn` cooks worlds for: the backend the runtime linked
/// into this same binary consumes, so a world built here plays here.
pub fn cook_platform() -> concinnity_core::platform::Platform {
    concinnity_engine::platform::current()
}

// Authoring / in-memory build.
mod authoring;

// The in-engine editor HUD, the localhost debug server, the MCP transport it
// speaks, and the interpreted run loop.
mod debug;
mod debug_hook;
mod editor;
mod mcp;
mod run;

/// The `cn` command line: the arguments a run was given, and the command they
/// name.
pub mod cli;
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
// and the MCP stdio bridge that forwards an agent's tool calls to a running
// app (`cn mcp`).
pub use editor::run_editor;
pub use mcp::run as run_mcp;
pub use run::run_debug;
