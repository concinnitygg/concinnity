//! concinnity-cli: the `concinnity` dev CLI.
//!
//! Owns the clap command tree + dispatch (entry.rs) and the subcommand
//! implementations (cli/), driving the dev-session entry points the
//! concinnity-editor library exposes and the concinnity-cook compile pipeline.
//! The binary itself is a target of the workspace's root package, which links
//! this library behind its `editor` feature.
//!
//! Also owns where a dev project keeps its state. The engine crates have no
//! default: they read whatever root a host installs, so the `.concinnity/`
//! directory a project's build outputs, caches, worlds, and settings live in is
//! this crate's convention and nobody else's.

use std::path::{Path, PathBuf};

mod cli;
mod entry;

// The directory a dev project keeps its state in, hidden inside the project so
// a checkout carries its build state without a stray visible folder.
const STATE_DIR: &str = ".concinnity";

/// Anchor the project state directory, install the crash hooks, and dispatch
/// the command line. The whole of the `concinnity` binary.
pub fn run() -> std::io::Result<()> {
    // Before the crash hooks, so a report written from here on lands in the
    // project rather than nowhere.
    concinnity_cook::paths::set_state_dir(project_state_dir());
    concinnity_engine::crash::install();
    entry::run()
}

// `.concinnity/` under the directory the command was run from, resolved once so
// every subcommand addresses the same tree. Falls back to the relative name
// when the working directory cannot be read, which keeps the layout correct
// for the process that is already sitting in it.
fn project_state_dir() -> PathBuf {
    std::env::current_dir().map_or_else(
        |_| PathBuf::from(STATE_DIR),
        |cwd| cwd.join(Path::new(STATE_DIR)),
    )
}
