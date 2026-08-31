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

// The directory a dev project keeps its derived state in, hidden inside the
// project so a checkout carries its build output without a stray visible
// folder. What a person authors stays beside it, in sight. The engine crates
// have no default: they are handed whatever tree a host builds, so this is the
// binary's convention and nobody else's.
const STATE_DIR: &str = ".concinnity";

fn main() -> std::io::Result<()> {
    let parsed = cli::Cli::parse();

    // Must run before any thread spawns or the Metal framework initialises.
    cli::reexec_with_metal_validation(&parsed);

    // Where this run reads and writes. Resolved before the crash hooks, so a
    // report written from here on lands in the project rather than nowhere,
    // and handed to the dev library, which builds and runs against it.
    let tree = project_tree();
    concinnity_engine::crash::install(Some(&tree.crashes_dir()));
    concinnity_dev::project::open(tree.clone());

    dispatch::dispatch(&parsed, &tree)
}

// The tree rooted at the directory the command was run from: `assets/` and
// `worlds/` in sight there, everything a build derives under `.concinnity/`
// beside them. Resolved once so every subcommand addresses the same tree.
//
// Falls back to the working directory as a relative path when it cannot be
// read, which keeps the layout correct for the process already sitting in it.
fn project_tree() -> concinnity_engine::StateTree {
    let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let state = root.join(Path::new(STATE_DIR));
    concinnity_engine::StateTree::at(root)
        .with_writable(&state)
        .with_build(state)
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

    // The project's roots are anchored under the working directory, not left
    // relative: every subcommand resolves the same tree however it chdirs.
    #[test]
    fn the_project_roots_are_anchored_absolutely() {
        let tree = super::project_tree();
        assert!(
            tree.content_root().is_absolute(),
            "{:?} should be anchored to the cwd",
            tree.content_root()
        );
        assert!(tree.build_root().ends_with(super::STATE_DIR));
        assert!(tree.writable_root().ends_with(super::STATE_DIR));
    }

    // What the CLI's layout is: authored content in sight at the project root,
    // every derived byte under the one hidden directory.
    #[test]
    fn authored_content_is_visible_and_derived_state_is_hidden() {
        let tree = super::project_tree();
        let root = tree.content_root();

        assert_eq!(tree.assets_dir(), root.join("assets"));
        assert_eq!(tree.worlds_dir(), root.join("worlds"));

        let hidden = root.join(super::STATE_DIR);
        assert_eq!(tree.data_dir(), hidden.join("data"));
        assert_eq!(tree.world_lock_path(), hidden.join("world-lock.json"));
        assert_eq!(tree.settings_path(), hidden.join("settings"));
        assert_eq!(tree.crashes_dir(), hidden.join("crashes"));
        assert_eq!(tree.editor_session_path(), hidden.join("editor"));
        assert_eq!(tree.build_cache_path(), hidden.join("cache").join("1"));
        assert_eq!(tree.runtime_cache_path(), hidden.join("cache").join("0"));
    }
}
