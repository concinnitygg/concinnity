//! Project state root: where the engine's state tree is anchored, and the names
//! of the directories hanging off it.
//!
//! Everything the engine writes for a project lives under one state directory:
//! the compiled blobs (`data/`), the regenerable cache (`cache/`: `0` for the
//! running application, `1` for a build, the baked asset thumbnails included),
//! fetched source assets (`assets/`),
//! named worlds (`worlds/`), the runtime save files (`saves/`), and the
//! mutable settings file (`settings`).
//!
//! Nothing here has a default. A host installs the state directory via
//! [`set_state_dir`] before anything reads the tree, and until it does every
//! path below resolves to `None`. The naming of that directory is the host's
//! business, not this crate's: the dev CLI hides it inside the project, a
//! shipped application puts it beside its executable, and an embedder points
//! it wherever its own layout implies. Reads that cannot proceed without a
//! state tree report [`CnResult::NoStateRoot`](concinnity_core::result::CnResult);
//! the caches and the settings file simply do nothing.
//!
//! The read-only content of the tree (`data/`) and the runtime-writable state
//! (`saves/` + `settings`) usually share one root, but a shipped application
//! installed in a read-only location (Program Files) cannot write beside its
//! data. Such an application installs a separate writable root via
//! [`set_writable_state_dir`] so only `saves/` and `settings` relocate to a
//! per-user directory while `data/` stays beside the executable.
//!
//! Resolution touches no files: these functions compute paths. Reading the tree
//! is `super::source` (finding a source asset) and `super::blob` (the compiled
//! blob).

use std::path::{Path, PathBuf};

mod root;

pub use root::{
    clear_state_dir, clear_writable_state_dir, set_state_dir, set_writable_state_dir, state_dir,
    writable_state_dir,
};

/// The state root's `assets/` directory.
pub fn assets_dir() -> Option<PathBuf> {
    state_dir().map(|d| d.join("assets"))
}

/// The state root's `data/` directory.
pub fn data_dir() -> Option<PathBuf> {
    state_dir().map(|d| d.join("data"))
}

/// Directory holding the runtime save files (`auto`, `save1` ..). Created on
/// first write by the running application, never by a build. Resolves under the
/// writable-state dir, which is the content root unless an application redirected it.
pub fn saves_dir() -> Option<PathBuf> {
    writable_state_dir().map(|d| d.join("saves"))
}

/// Sandboxed sibling of [saves_dir] for preview sessions (see the
/// `TransientSaves` protocol resource): the save UI keeps working against this
/// directory, but the real saves are never touched and the sandbox is wiped at
/// each session start.
pub fn preview_saves_dir() -> Option<PathBuf> {
    writable_state_dir().map(|d| d.join("preview-saves"))
}

/// The mutable settings file (CBOR). Written by the in-engine settings menu,
/// never by a build. A sibling of `data/` in the common case, or under the
/// writable-state dir when a read-only install redirected it.
pub fn settings_path() -> Option<PathBuf> {
    writable_state_dir().map(|d| d.join("settings"))
}

/// Directory holding crash reports (and minidumps) written by the crash
/// reporting machinery. Resolves under the writable-state dir like `saves/`,
/// since a shipped install's content root may be read-only. Created on first
/// write; capped by the writer's retention pruning, never by a build.
pub fn crashes_dir() -> Option<PathBuf> {
    writable_state_dir().map(|d| d.join("crashes"))
}

/// The state root's `worlds/` directory.
pub fn worlds_dir() -> Option<PathBuf> {
    state_dir().map(|d| d.join("worlds"))
}

/// The subdirectory a state tree keeps its cache segments in.
///
/// Named so a caller that has to reason about the tree's shape -- a test
/// harness deciding what may survive between runs -- asks this crate rather
/// than spelling the layout itself.
pub const CACHE_DIR: &str = "cache";

/// The runtime cache segment inside `state_dir`, for a caller naming a state
/// tree other than the installed one: `cn export` warms the segment it writes
/// into a bundle before that bundle is ever launched.
pub fn runtime_cache_in(state_dir: &Path) -> PathBuf {
    state_dir.join(CACHE_DIR).join("0")
}

/// The runtime cache segment, `cache/0`: one container holding every
/// regenerable artifact the running application produces for its own later
/// launches, indexed by producer and key. Resolves under the writable-state
/// dir, since a shipped install's content root may be read-only.
///
/// Deletable at any time; whatever is missing is recomputed. The running
/// application writes this file and no other, so a concurrent build writing a
/// segment of its own never shares a file with it.
pub fn runtime_cache_path() -> Option<PathBuf> {
    writable_state_dir().map(|d| runtime_cache_in(&d))
}

/// The runtime cache segment a bundle ships, read-only. `cn export` warms it
/// with the shader binaries a first launch would otherwise compile; because
/// those artifacts are backend IR (DXBC / SPIR-V) rather than machine code, one
/// warmed at package time is valid on any machine.
///
/// Resolves against the content root, so it stays readable on a read-only
/// install. That is also the only layout where this differs from
/// [`runtime_cache_path`]: a bundle the player can write to has one segment
/// serving both roles.
pub fn bundled_runtime_cache_path() -> Option<PathBuf> {
    state_dir().map(|d| runtime_cache_in(&d))
}

/// The build cache segment, `cache/1`: one container holding every payload,
/// expansion, and baked thumbnail a cook produced, indexed by producer and
/// key. A build writes this
/// file and no other, so a cook running against a live application never
/// shares a file with the segment that application writes.
///
/// Resolves against the content root rather than the writable one: a build
/// writes the `data/` beside it, so a tree it cannot write is a tree it cannot
/// cook into either.
///
/// Deletable at any time; whatever is missing is recompiled.
pub fn build_cache_path() -> Option<PathBuf> {
    state_dir().map(|d| d.join(CACHE_DIR).join("1"))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Exercises the process-global roots end to end. The single test that drives
    // the globals, so its mutations never race another test that reads them.
    #[test]
    fn installed_roots_redirect_every_state_dir() {
        let flat = Path::new("/tmp/flat-probe");
        set_state_dir(flat);
        assert_eq!(state_dir().as_deref(), Some(flat));
        assert_eq!(data_dir().unwrap(), flat.join("data"));
        assert_eq!(runtime_cache_path().unwrap(), flat.join("cache").join("0"));
        assert_eq!(build_cache_path().unwrap(), flat.join("cache").join("1"));
        assert_eq!(assets_dir().unwrap(), flat.join("assets"));
        assert_eq!(worlds_dir().unwrap(), flat.join("worlds"));
        // With no writable override, writable state stays beside the data.
        assert_eq!(writable_state_dir().as_deref(), Some(flat));
        assert_eq!(saves_dir().unwrap(), flat.join("saves"));
        assert_eq!(settings_path().unwrap(), flat.join("settings"));
        assert_eq!(crashes_dir().unwrap(), flat.join("crashes"));

        // A writable override relocates only the runtime-writable state
        // (`saves/`, `settings`, `crashes/`); `data/` (and assets/worlds) stay
        // at the content root, and so does the segment a build writes.
        let writable = Path::new("/tmp/per-user-probe");
        set_writable_state_dir(writable);
        assert_eq!(writable_state_dir().as_deref(), Some(writable));
        assert_eq!(saves_dir().unwrap(), writable.join("saves"));
        assert_eq!(settings_path().unwrap(), writable.join("settings"));
        assert_eq!(crashes_dir().unwrap(), writable.join("crashes"));
        assert_eq!(
            runtime_cache_path().unwrap(),
            writable.join("cache").join("0")
        );
        assert_eq!(build_cache_path().unwrap(), flat.join("cache").join("1"));
        // The bundle's warmed segment stays with the content, which is what
        // makes a read-only install's shipped artifacts still readable.
        assert_eq!(
            bundled_runtime_cache_path().unwrap(),
            flat.join("cache").join("0")
        );
        assert_eq!(data_dir().unwrap(), flat.join("data"));
        clear_writable_state_dir();
        assert_eq!(saves_dir().unwrap(), flat.join("saves"));

        // With nothing installed there is no state tree at all: no guess
        // against the working directory, so a library writes nowhere.
        clear_state_dir();
        assert_eq!(state_dir(), None);
        for path in [
            data_dir(),
            assets_dir(),
            worlds_dir(),
            saves_dir(),
            preview_saves_dir(),
            settings_path(),
            crashes_dir(),
            runtime_cache_path(),
            bundled_runtime_cache_path(),
            build_cache_path(),
            writable_state_dir(),
        ] {
            assert_eq!(path, None);
        }
    }
}
