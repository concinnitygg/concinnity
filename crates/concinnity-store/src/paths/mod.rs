// Project state root: where the engine's state tree is anchored, and the names
// of the directories hanging off it.
//
// Everything the engine writes for a project lives under one state directory:
// the compiled blobs (`data/`), the payload cache (`cache/`), fetched source
// assets (`assets/`), named worlds (`worlds/`), the runtime save files
// (`saves/`), and the mutable settings file (`settings`). In a dev project that
// tree is wrapped in a `.concinnity/` directory addressed relative to the
// current working directory, so `.concinnity/` sits wherever a command runs.
// That is the historical behavior and is unchanged when no root is installed.
//
// A host that must change the working directory for an unrelated reason (an
// example that chdirs so its world's relative asset paths resolve against the
// example directory) would otherwise drag `.concinnity/` along with it. Such a
// host captures the invocation directory and installs it here before it chdirs,
// so state stays put while content resolution follows the working directory.
//
// A shipped application installs a flat state root via `set_state_dir`: the state tree
// then sits directly at that directory with no `.concinnity/` wrapper, so
// `data/`, `saves/`, and `settings` resolve beside the executable or inside the
// app bundle.
//
// The read-only content of the tree (`data/`) and the runtime-writable state
// (`saves/` + `settings`) usually share one root, but a shipped application installed
// in a read-only location (Program Files) cannot write beside its data. Such a
// application installs a separate writable root via `set_writable_state_dir` so only
// `saves/` and `settings` relocate to a per-user directory while `data/` stays
// beside the executable. When no writable root is installed, writable state
// stays with the content, so dev and portable installs are unaffected.
//
// Resolution touches no files: these functions compute paths. Reading the tree
// is `crate::source` (finding a source asset) and `crate::blob` (the compiled
// blob).

use std::path::PathBuf;

mod root;

pub use root::{
    HOME_ENV, STATE_DIR, clear_root, clear_state_dir, clear_writable_state_dir, set_root,
    set_state_dir, set_writable_state_dir, state_dir, writable_state_dir,
};

pub fn assets_dir() -> PathBuf {
    state_dir().join("assets")
}

pub fn data_dir() -> PathBuf {
    state_dir().join("data")
}

// Directory holding the runtime save files (`auto`, `save1` ..). Created on
// first write by the running application, never by a build. Resolves under the
// writable-state dir, which is the content root unless an application redirected it.
pub fn saves_dir() -> PathBuf {
    writable_state_dir().join("saves")
}

// Sandboxed sibling of [saves_dir] for preview sessions (see the
// `TransientSaves` protocol resource): the save UI keeps working against this
// directory, but the real saves are never touched and the sandbox is wiped at
// each session start.
pub fn preview_saves_dir() -> PathBuf {
    writable_state_dir().join("preview-saves")
}

// The mutable settings file (CBOR). Written by the in-engine settings menu,
// never by a build. A sibling of `data/` in the common case, or under the
// writable-state dir when a read-only install redirected it.
pub fn settings_path() -> PathBuf {
    writable_state_dir().join("settings")
}

// Directory holding crash reports (and minidumps) written by the crash
// reporting machinery. Resolves under the writable-state dir like `saves/`,
// since a shipped install's content root may be read-only. Created on first
// write; capped by the writer's retention pruning, never by a build.
pub fn crashes_dir() -> PathBuf {
    writable_state_dir().join("crashes")
}

pub fn worlds_dir() -> PathBuf {
    state_dir().join("worlds")
}

pub fn cache_dir() -> PathBuf {
    state_dir().join("cache")
}

/// Directory holding baked asset thumbnails: content-addressed `<sha256>.png`
/// files plus an `index.json` mapping asset names to keys. Deterministic
/// products of the build like [`cache_dir`]'s payloads, but kept apart so they
/// can be listed and cleared independently (and never ship: `cn export` copies
/// neither).
pub fn thumbnails_dir() -> PathBuf {
    state_dir().join("thumbnails")
}

/// Directory the renderer writes compiled built-in shader binaries to, keyed by
/// a hash of their compile inputs. Resolves under the writable-state dir, since
/// a shipped install's content root may be read-only. Distinct from
/// [`cache_dir`], which holds cooked asset payloads: these artifacts belong to
/// the machine's shader compiler, not to the build.
pub fn shader_cache_dir() -> PathBuf {
    writable_state_dir().join("shader-cache")
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
pub fn bundled_shader_cache_dir() -> PathBuf {
    state_dir().join("shader-cache")
}

#[cfg(test)]
mod tests {
    use super::root::anchor;
    use super::*;
    use std::path::Path;

    #[test]
    fn subdirs_hang_off_the_state_dir() {
        // The layout under the state dir is stable regardless of the anchor.
        // Exercised through the pure `anchor` helper so the assertion does not
        // depend on the process-global root.
        let base = anchor(Some(Path::new("/proj")), Path::new(STATE_DIR));
        for sub in ["assets", "data", "saves", "worlds", "cache"] {
            assert_eq!(base.join(sub), Path::new("/proj").join(STATE_DIR).join(sub));
        }
        // `settings` is a file directly under the state dir, not a directory.
        assert_eq!(
            base.join("settings"),
            Path::new("/proj").join(STATE_DIR).join("settings")
        );
    }

    // Exercises the process-global roots end to end. The single test that drives
    // the globals, so its mutations never race another test that reads them.
    #[test]
    fn installed_roots_redirect_every_state_dir() {
        // A flat state dir wins verbatim, with no `.concinnity` segment.
        let flat = Path::new("/tmp/flat-probe");
        set_state_dir(flat);
        assert_eq!(state_dir(), flat);
        assert_eq!(data_dir(), flat.join("data"));
        // With no writable override, writable state stays beside the data.
        assert_eq!(writable_state_dir(), flat);
        assert_eq!(saves_dir(), flat.join("saves"));
        assert_eq!(settings_path(), flat.join("settings"));
        assert_eq!(crashes_dir(), flat.join("crashes"));

        // A writable override relocates only the runtime-writable state
        // (`saves/`, `settings`, `crashes/`); `data/` (and
        // cache/assets/worlds) stay at the content root.
        let writable = Path::new("/tmp/per-user-probe");
        set_writable_state_dir(writable);
        assert_eq!(writable_state_dir(), writable);
        assert_eq!(saves_dir(), writable.join("saves"));
        assert_eq!(settings_path(), writable.join("settings"));
        assert_eq!(crashes_dir(), writable.join("crashes"));
        assert_eq!(data_dir(), flat.join("data"));
        clear_writable_state_dir();
        assert_eq!(saves_dir(), flat.join("saves"));
        clear_state_dir();

        // With no flat dir, an installed root anchors the `.concinnity` tree and
        // takes precedence over CN_HOME and the cwd default.
        let root = Path::new("/tmp/anchor-probe");
        set_root(root);
        let expected = root.join(STATE_DIR);
        assert_eq!(state_dir(), expected);
        assert_eq!(cache_dir(), expected.join("cache"));
        assert_eq!(data_dir(), expected.join("data"));
        assert_eq!(saves_dir(), expected.join("saves"));
        assert_eq!(settings_path(), expected.join("settings"));
        assert_eq!(assets_dir(), expected.join("assets"));
        assert_eq!(worlds_dir(), expected.join("worlds"));

        // Clearing restores cwd-relative resolution, unless CN_HOME is set.
        clear_root();
        if std::env::var_os(HOME_ENV).is_none() {
            assert_eq!(cache_dir(), Path::new(STATE_DIR).join("cache"));
        }
    }
}
