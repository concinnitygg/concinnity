//! concinnity: the dev CLI binary, built by the `editor` feature.
//!
//! One file, owning what a process owns -- where the project keeps its state,
//! and the crash hooks. The command line it reads and every command's actual
//! work live in concinnity-dev.

use std::path::PathBuf;

use concinnity::player;

fn main() -> std::process::ExitCode {
    // Parses argv, and on macOS may replace this process image to bring up
    // Metal validation, so nothing above it may spawn a thread.
    let invocation = concinnity_dev::cli::Invocation::from_args();

    // Where this run reads and writes. Resolved before the crash hooks, so a
    // report written from here on lands in the project rather than nowhere,
    // and handed to the dev library, which builds and runs against it.
    let tree = project_tree();
    player::install_crash_hooks(Some(&tree.crashes_dir()));
    concinnity_dev::project::open(tree.clone());

    // Printed with Display rather than the Debug a returned error gets, so a
    // multi-line report such as a shader diagnostic reads as written.
    match invocation.dispatch(&tree) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

// The unsplit tree rooted at the directory the command was run from: the
// authored `assets/` and .jsonl files, and everything a build or run writes,
// side by side. Resolved once so every subcommand addresses the same tree.
//
// Falls back to the working directory as a relative path when it cannot be
// read, which keeps the layout correct for the process already sitting in it.
fn project_tree() -> player::StateTree {
    let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    player::StateTree::at(root)
}

#[cfg(test)]
mod tests {
    // `cn run -f` and `cn edit` report heap figures through the editor's health
    // output, which the probe harness reads. They come from the library's
    // counting allocator; without it the CLI would run correctly while
    // reporting no memory at all.
    #[test]
    fn the_dev_cli_tracks_its_own_heap() {
        const MIB: usize = 1 << 20;

        let before = concinnity::memory::stats()
            .expect("this binary declares the tracking allocator")
            .alloc_count;
        let held: Vec<u8> = core::hint::black_box(vec![0; MIB]);
        let after = concinnity::memory::stats().expect("the allocator stays installed");

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
    }

    // What the CLI's layout is: authored content and every byte a build or run
    // writes share the working directory.
    #[test]
    fn every_output_lands_in_the_working_directory() {
        let tree = super::project_tree();
        let root = tree.content_root();

        assert_eq!(tree.build_root(), root);
        assert_eq!(tree.writable_root(), root);
        assert_eq!(tree.assets_dir(), root.join("assets"));
        assert_eq!(tree.data_dir(), root.join("data"));
        assert_eq!(tree.cook_lock_path().parent(), Some(root));
        assert_eq!(tree.settings_path(), root.join("settings"));
        assert_eq!(tree.crashes_dir(), root.join("crashes"));
        assert_eq!(tree.editor_session_path(), root.join("editor"));
        assert_eq!(tree.build_cache_path(), root.join("cache").join("1"));
        assert_eq!(tree.runtime_cache_path(), root.join("cache").join("0"));
    }
}
