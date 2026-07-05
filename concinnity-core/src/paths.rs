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
// first write by the running game, never by a build.
pub fn saves_dir() -> PathBuf {
    state_dir().join("saves")
}

// The mutable settings file (CBOR). Written by the in-engine settings menu,
// never by a build. A sibling of `data/`, not a directory inside it.
pub fn settings_path() -> PathBuf {
    state_dir().join("settings")
}

pub fn worlds_dir() -> PathBuf {
    state_dir().join("worlds")
}

pub fn cache_dir() -> PathBuf {
    state_dir().join("cache")
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(saves_dir(), flat.join("saves"));
        assert_eq!(settings_path(), flat.join("settings"));
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
