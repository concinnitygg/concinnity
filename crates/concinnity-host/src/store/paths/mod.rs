//! Project state root: where the engine's state tree is anchored, and the names
//! of the directories hanging off it.
//!
//! Everything the engine writes for a project lives under one state directory:
//! the compiled blobs (`data/`), the payload cache (`cache/`), fetched source
//! assets (`assets/`), named worlds (`worlds/`), the runtime save files
//! (`saves/`), and the mutable settings file (`settings`).
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

use std::path::PathBuf;

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

/// The state root's `cache/` directory.
pub fn cache_dir() -> Option<PathBuf> {
    state_dir().map(|d| d.join("cache"))
}

/// Directory holding baked asset thumbnails: content-addressed `<sha256>.png`
/// files plus an `index.json` mapping asset names to keys. Deterministic
/// products of the build like [`cache_dir`]'s payloads, but kept apart so they
/// can be listed and cleared independently (and never ship: `cn export` copies
/// neither).
pub fn thumbnails_dir() -> Option<PathBuf> {
    state_dir().map(|d| d.join("thumbnails"))
}

/// Directory the renderer writes compiled built-in shader binaries to, keyed by
/// a hash of their compile inputs. Resolves under the writable-state dir, since
/// a shipped install's content root may be read-only. Distinct from
/// [`cache_dir`], which holds cooked asset payloads: these artifacts belong to
/// the machine's shader compiler, not to the build.
pub fn shader_cache_dir() -> Option<PathBuf> {
    writable_state_dir().map(|d| d.join("shader-cache"))
}

/// Directory the renderer persists driver pipeline blobs to (a serialized
/// VkPipelineCache, a D3D12 pipeline library), keyed per adapter. Unlike
/// [`shader_cache_dir`] artifacts these are machine code tied to one GPU and
/// driver, so they resolve under the writable-state dir only and never ship in
/// a bundle. A sibling of `shader-cache/` rather than a subdirectory, since the
/// shader cache prunes its directory by age and would reclaim these.
pub fn pipeline_cache_dir() -> Option<PathBuf> {
    writable_state_dir().map(|d| d.join("pipeline-cache"))
}

/// Directory holding shader binaries shipped inside a bundle, read-only. `cn
/// export` warms this so a player's first launch does not pay the compile;
/// because the artifacts are backend IR (DXBC / SPIR-V) rather than machine
/// code, one warmed at package time is valid on any machine.
///
/// Equal to [`shader_cache_dir`] whenever the content root is writable (the
/// portable-folder case). The two diverge only for a read-only install, which
/// redirects writable state to a per-user directory: the bundled artifacts then
/// stay readable here while new ones land in the writable dir.
pub fn bundled_shader_cache_dir() -> Option<PathBuf> {
    state_dir().map(|d| d.join("shader-cache"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    // Exercises the process-global roots end to end. The single test that drives
    // the globals, so its mutations never race another test that reads them.
    #[test]
    fn installed_roots_redirect_every_state_dir() {
        let flat = Path::new("/tmp/flat-probe");
        set_state_dir(flat);
        assert_eq!(state_dir().as_deref(), Some(flat));
        assert_eq!(data_dir().unwrap(), flat.join("data"));
        assert_eq!(cache_dir().unwrap(), flat.join("cache"));
        assert_eq!(assets_dir().unwrap(), flat.join("assets"));
        assert_eq!(worlds_dir().unwrap(), flat.join("worlds"));
        // With no writable override, writable state stays beside the data.
        assert_eq!(writable_state_dir().as_deref(), Some(flat));
        assert_eq!(saves_dir().unwrap(), flat.join("saves"));
        assert_eq!(settings_path().unwrap(), flat.join("settings"));
        assert_eq!(crashes_dir().unwrap(), flat.join("crashes"));

        // A writable override relocates only the runtime-writable state
        // (`saves/`, `settings`, `crashes/`); `data/` (and cache/assets/worlds)
        // stay at the content root.
        let writable = Path::new("/tmp/per-user-probe");
        set_writable_state_dir(writable);
        assert_eq!(writable_state_dir().as_deref(), Some(writable));
        assert_eq!(saves_dir().unwrap(), writable.join("saves"));
        assert_eq!(settings_path().unwrap(), writable.join("settings"));
        assert_eq!(crashes_dir().unwrap(), writable.join("crashes"));
        assert_eq!(shader_cache_dir().unwrap(), writable.join("shader-cache"));
        // The bundled shader cache stays with the content, which is what makes
        // a read-only install's warmed artifacts still readable.
        assert_eq!(
            bundled_shader_cache_dir().unwrap(),
            flat.join("shader-cache")
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
            cache_dir(),
            assets_dir(),
            worlds_dir(),
            saves_dir(),
            preview_saves_dir(),
            settings_path(),
            crashes_dir(),
            thumbnails_dir(),
            shader_cache_dir(),
            pipeline_cache_dir(),
            bundled_shader_cache_dir(),
            writable_state_dir(),
        ] {
            assert_eq!(path, None);
        }
    }
}
