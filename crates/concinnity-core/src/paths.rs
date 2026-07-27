// Project state root: where the engine's state tree is anchored.
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
// A shipped game installs a flat state root via `set_state_dir`: the state tree
// then sits directly at that directory with no `.concinnity/` wrapper, so
// `data/`, `saves/`, and `settings` resolve beside the executable or inside the
// app bundle.
//
// Resolution order, highest precedence first:
//   1. a flat state dir installed via `set_state_dir` (no `.concinnity` segment)
//   2. a root installed via `set_root` (wraps in `.concinnity`)
//   3. the `CN_HOME` environment variable (wraps in `.concinnity`)
//   4. none: `.concinnity` relative to the current directory (the default)
//
// The read-only content of the tree (`data/`) and the runtime-writable state
// (`saves/` + `settings`) usually share one root, but a shipped game installed
// in a read-only location (Program Files) cannot write beside its data. Such a
// game installs a separate writable root via `set_writable_state_dir` so only
// `saves/` and `settings` relocate to a per-user directory while `data/` stays
// beside the executable. When no writable root is installed, writable state
// stays with the content, so dev and portable installs are unaffected.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

// Name of the state directory joined onto the resolved root in the default and
// `set_root`/`CN_HOME` modes. A flat state dir (`set_state_dir`) omits it.
pub const STATE_DIR: &str = ".concinnity";

// Environment variable that anchors the state root when no root is installed.
pub const HOME_ENV: &str = "CN_HOME";

fn installed_root() -> &'static Mutex<Option<PathBuf>> {
    static ROOT: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
    ROOT.get_or_init(|| Mutex::new(None))
}

fn flat_state_dir() -> &'static Mutex<Option<PathBuf>> {
    static FLAT: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
    FLAT.get_or_init(|| Mutex::new(None))
}

fn writable_state_override() -> &'static Mutex<Option<PathBuf>> {
    static WRITABLE: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
    WRITABLE.get_or_init(|| Mutex::new(None))
}

// Anchor `.concinnity/` to `dir` for the rest of the process, taking precedence
// over `CN_HOME` and the working-directory default. A host that chdirs for
// content resolution installs the invocation directory here before it chdirs.
pub fn set_root<P: Into<PathBuf>>(dir: P) {
    *installed_root().lock().unwrap() = Some(dir.into());
}

// Remove an installed root, restoring environment/working-directory resolution.
pub fn clear_root() {
    *installed_root().lock().unwrap() = None;
}

// Anchor the state tree directly at `dir` with no `.concinnity` segment, taking
// precedence over `set_root`, `CN_HOME`, and the default. A shipped game
// installs this so `data/`, `saves/`, and `settings` resolve beside its
// executable (or inside its app bundle) rather than under a `.concinnity/`
// wrapper.
pub fn set_state_dir<P: Into<PathBuf>>(dir: P) {
    *flat_state_dir().lock().unwrap() = Some(dir.into());
}

// Remove an installed flat state dir, restoring the wrapped resolution.
pub fn clear_state_dir() {
    *flat_state_dir().lock().unwrap() = None;
}

// Anchor the runtime-writable state (`saves/` + `settings`) at `dir`, leaving
// the read-only content (`data/`) at the resolved state dir. A shipped game
// installs this when its content dir is not writable (a read-only install such
// as Program Files), redirecting only what it writes at runtime to a per-user
// directory. When unset, writable state stays beside `data/`.
pub fn set_writable_state_dir<P: Into<PathBuf>>(dir: P) {
    *writable_state_override().lock().unwrap() = Some(dir.into());
}

// Remove an installed writable-state dir, restoring writable state to the
// content root beside `data/`.
pub fn clear_writable_state_dir() {
    *writable_state_override().lock().unwrap() = None;
}

// The resolved wrapping root, or `None` when paths should stay relative to the
// cwd. Only consulted when no flat state dir is installed.
fn root() -> Option<PathBuf> {
    installed_root()
        .lock()
        .unwrap()
        .clone()
        .or_else(|| std::env::var_os(HOME_ENV).map(PathBuf::from))
}

// The state directory: the flat state dir verbatim when one is installed,
// otherwise `<root>/.concinnity` (or the relative `.concinnity` against cwd).
pub fn state_dir() -> PathBuf {
    let flat = flat_state_dir().lock().unwrap().clone();
    resolve_state_dir(flat.as_deref(), root().as_deref())
}

// Pure resolution split out so the precedence rule is unit-testable without
// touching the process-global state. A flat dir wins verbatim; otherwise the
// `.concinnity` segment is anchored onto the wrapping root.
fn resolve_state_dir(flat: Option<&Path>, root: Option<&Path>) -> PathBuf {
    match flat {
        Some(f) => f.to_path_buf(),
        None => anchor(root, Path::new(STATE_DIR)),
    }
}

// The directory holding runtime-writable state (`saves/` + `settings`): the
// writable override when one is installed, otherwise the state dir (writable
// state sits beside `data/`).
pub fn writable_state_dir() -> PathBuf {
    let over = writable_state_override().lock().unwrap().clone();
    resolve_writable_dir(over.as_deref(), &state_dir())
}

// Pure resolution split out so the fallback rule is unit-testable without the
// process-global override: the override verbatim, else the content state dir.
fn resolve_writable_dir(over: Option<&Path>, state: &Path) -> PathBuf {
    over.map_or_else(|| state.to_path_buf(), Path::to_path_buf)
}

// Join `rel` onto `root`, or return it unchanged (relative) when `root` is
// `None`. Split out so the anchoring rule is unit-testable without touching the
// process-global root.
fn anchor(root: Option<&Path>, rel: &Path) -> PathBuf {
    match root {
        Some(r) => r.join(rel),
        None => rel.to_path_buf(),
    }
}

pub fn assets_dir() -> PathBuf {
    state_dir().join("assets")
}

pub fn data_dir() -> PathBuf {
    state_dir().join("data")
}

// Directory holding the runtime save files (`auto`, `save1` ..). Created on
// first write by the running game, never by a build. Resolves under the
// writable-state dir, which is the content root unless a game redirected it.
pub fn saves_dir() -> PathBuf {
    writable_state_dir().join("saves")
}

// The mutable settings file (CBOR). Written by the in-engine settings menu,
// never by a build. A sibling of `data/` in the common case, or under the
// writable-state dir when a read-only install redirected it.
pub fn settings_path() -> PathBuf {
    writable_state_dir().join("settings")
}

pub fn worlds_dir() -> PathBuf {
    state_dir().join("worlds")
}

pub fn cache_dir() -> PathBuf {
    state_dir().join("cache")
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

// Recursively search `assets_dir()` for a file matching the given bare
// filename, returning the first match. Source strings that are bare filenames
// (a texture, HDRI, `.cube` LUT, or shader named without a directory) are
// resolved against `.concinnity/assets/` this way. Used only by source-path
// resolution: the compile pipeline (cook) and the `cn debug` hot-reload watcher
// (which reads the resolved on-disk path to subscribe to changes). The shipped
// runtime plays compiled blobs and never resolves a source file.
pub fn find_in_assets(filename: &str) -> Option<String> {
    fn walk(dir: &Path, filename: &str) -> Option<PathBuf> {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return None;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.file_name().and_then(|n| n.to_str()) == Some(filename) {
                return Some(path);
            }
            if path.is_dir()
                && let Some(found) = walk(&path, filename)
            {
                return Some(found);
            }
        }
        None
    }
    walk(&assets_dir(), filename).map(|p| p.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Only the miss path is exercised: a hit resolves against the process-global
    // assets-dir anchor, which the roots test below redirects, so pointing it at
    // a temp tree here would race that test.
    #[test]
    fn find_in_assets_misses_cleanly() {
        assert_eq!(find_in_assets("cn_test_no_such_asset.json"), None);
    }

    #[test]
    fn anchor_without_root_stays_relative() {
        let p = anchor(None, Path::new(STATE_DIR));
        assert_eq!(p, Path::new(STATE_DIR));
        assert!(p.is_relative());
    }

    #[test]
    fn anchor_with_root_is_under_it() {
        let root = Path::new("/proj/game");
        let p = anchor(Some(root), Path::new(STATE_DIR));
        assert!(p.starts_with(root));
        assert_eq!(p, root.join(STATE_DIR));
        assert_eq!(p.file_name().unwrap(), STATE_DIR);
    }

    #[test]
    fn resolve_state_dir_flat_wins_verbatim() {
        // A flat state dir is used exactly as given, with no `.concinnity`
        // segment, and ignores any wrapping root.
        let flat = Path::new("/game/MyGame");
        let p = resolve_state_dir(Some(flat), Some(Path::new("/ignored")));
        assert_eq!(p, flat);
    }

    #[test]
    fn resolve_state_dir_without_flat_anchors_concinnity() {
        let root = Path::new("/proj");
        assert_eq!(resolve_state_dir(None, Some(root)), root.join(STATE_DIR));

        let rel = resolve_state_dir(None, None);
        assert_eq!(rel, Path::new(STATE_DIR));
        assert!(rel.is_relative());
    }

    #[test]
    fn resolve_writable_dir_prefers_override_then_falls_back() {
        // An installed writable override wins verbatim, relocating only the
        // writable state; without one, writable state stays with the content.
        let state = Path::new("/game/MyGame");
        let over = Path::new("/users/me/AppData/Local/MyGame");
        assert_eq!(resolve_writable_dir(Some(over), state), over);
        assert_eq!(resolve_writable_dir(None, state), state);
    }

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

        // A writable override relocates only `saves/` + `settings`; `data/`
        // (and cache/assets/worlds) stay at the content root.
        let writable = Path::new("/tmp/per-user-probe");
        set_writable_state_dir(writable);
        assert_eq!(writable_state_dir(), writable);
        assert_eq!(saves_dir(), writable.join("saves"));
        assert_eq!(settings_path(), writable.join("settings"));
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
