//! concinnity: the dev CLI binary, built by the `editor` feature.
//!
//! Three files: [`cli`] is the clap command tree, [`dispatch`] turns a parsed
//! command into a call, and this one owns what a process owns -- the tracking
//! allocator, where the project keeps its state, and the crash hooks. Every
//! command's actual work lives in concinnity-dev.

mod cli;
mod dispatch;

use clap::Parser;
use std::path::{Path, PathBuf};

// A global allocator belongs to a final artifact, not to a library every
// consumer would inherit it from.
concinnity_core::install_global_allocator!();

// The directory a dev project keeps its state in, hidden inside the project so
// a checkout carries its build state without a stray visible folder. The engine
// crates have no default: they read whatever root a host installs, so this is
// the binary's convention and nobody else's.
const STATE_DIR: &str = ".concinnity";

fn main() -> std::io::Result<()> {
    let parsed = cli::Cli::parse();

    // Must run before any thread spawns or the Metal framework initialises.
    cli::reexec_with_metal_validation(&parsed);

    // Before the crash hooks, so a report written from here on lands in the
    // project rather than nowhere.
    concinnity_cook::paths::set_state_dir(project_state_dir());
    concinnity_engine::crash::install();

    dispatch::dispatch(&parsed)
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

        let before = concinnity_core::memory::stats()
            .expect("this binary declares the tracking allocator")
            .alloc_count;
        let held: Vec<u8> = core::hint::black_box(vec![0; MIB]);
        let after = concinnity_core::memory::stats().expect("the allocator stays installed");

        assert!(
            after.alloc_count > before,
            "allocation count did not move ({before} -> {}) across a megabyte",
            after.alloc_count
        );
        assert!(after.peak_bytes >= after.live_bytes);
        drop(core::hint::black_box(held));
    }

    // The project state root is anchored under the working directory, not left
    // relative: every subcommand resolves the same tree however it chdirs.
    #[test]
    fn the_state_dir_is_anchored_absolutely() {
        let dir = super::project_state_dir();
        assert!(dir.ends_with(super::STATE_DIR));
        assert!(dir.is_absolute(), "{dir:?} should be anchored to the cwd");
    }
}
