//! concinnity-cli: the `concinnity` dev CLI binary.
//!
//! Owns the clap command tree + dispatch (entry.rs) and the subcommand
//! implementations (cli/), driving the dev-session entry points the
//! concinnity-editor library exposes and the concinnity-cook compile pipeline.
//!
//! Also owns where a dev project keeps its state. The engine crates have no
//! default: they read whatever root a host installs, so the `.concinnity/`
//! directory a project's build outputs, caches, worlds, and settings live in is
//! this binary's convention and nobody else's.

use std::path::{Path, PathBuf};

concinnity_memory::install_global_allocator!();

mod cli;
mod entry;

// The directory a dev project keeps its state in, hidden inside the project so
// a checkout carries its build state without a stray visible folder.
const STATE_DIR: &str = ".concinnity";

fn main() -> std::io::Result<()> {
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

#[cfg(test)]
mod tests {
    // `cn debug` and `cn editor` report heap figures through the editor's health
    // output, which the probe harness reads. Nothing forces the declaration at
    // the top of this file to exist, so this is what catches its removal:
    // without it the CLI would run correctly while reporting no memory at all.
    #[test]
    fn the_dev_cli_tracks_its_own_heap() {
        const MIB: usize = 1 << 20;

        let before = concinnity_memory::stats()
            .expect("this binary declares the tracking allocator")
            .alloc_count;
        let held: Vec<u8> = core::hint::black_box(vec![0; MIB]);
        let after = concinnity_memory::stats().expect("the allocator stays installed");

        assert!(
            after.alloc_count > before,
            "allocation count did not move ({before} -> {}) across a megabyte",
            after.alloc_count
        );
        assert!(after.peak_bytes >= after.live_bytes);
        drop(core::hint::black_box(held));
    }
}
