// Anchoring: which directory the state tree hangs off, and the precedence
// between the ways a host can install one.
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
pub(crate) const STATE_DIR: &str = ".concinnity";

// Environment variable that anchors the state root when no root is installed.
pub(crate) const HOME_ENV: &str = "CN_HOME";

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

/// Anchor `.concinnity/` to `dir` for the rest of the process, taking precedence
/// over `CN_HOME` and the working-directory default. A host that chdirs for
/// content resolution installs the invocation directory here before it chdirs.
pub fn set_root<P: Into<PathBuf>>(dir: P) {
    *installed_root().lock().unwrap() = Some(dir.into());
}

// Remove an installed root, restoring environment/working-directory resolution.
#[cfg(test)]
pub(crate) fn clear_root() {
    *installed_root().lock().unwrap() = None;
}

/// Anchor the state tree directly at `dir` with no `.concinnity` segment, taking
/// precedence over `set_root`, `CN_HOME`, and the default. A shipped application
/// installs this so `data/`, `saves/`, and `settings` resolve beside its
/// executable (or inside its app bundle) rather than under a `.concinnity/`
/// wrapper.
pub fn set_state_dir<P: Into<PathBuf>>(dir: P) {
    *flat_state_dir().lock().unwrap() = Some(dir.into());
}

/// Remove an installed flat state dir, restoring the wrapped resolution.
pub fn clear_state_dir() {
    *flat_state_dir().lock().unwrap() = None;
}

/// Anchor the runtime-writable state (`saves/` + `settings`) at `dir`, leaving
/// the read-only content (`data/`) at the resolved state dir. A shipped application
/// installs this when its content dir is not writable (a read-only install such
/// as Program Files), redirecting only what it writes at runtime to a per-user
/// directory. When unset, writable state stays beside `data/`.
pub fn set_writable_state_dir<P: Into<PathBuf>>(dir: P) {
    *writable_state_override().lock().unwrap() = Some(dir.into());
}

// Remove an installed writable-state dir, restoring writable state to the
// content root beside `data/`.
#[cfg(test)]
pub(crate) fn clear_writable_state_dir() {
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

/// The state directory: the flat state dir verbatim when one is installed,
/// otherwise `<root>/.concinnity` (or the relative `.concinnity` against cwd).
pub fn state_dir() -> PathBuf {
    let flat = flat_state_dir().lock().unwrap().clone();
    resolve_state_dir(flat.as_deref(), root().as_deref())
}

// Pure resolution split out so the precedence rule is unit-testable without
// touching the process-global state. A flat dir wins verbatim; otherwise the
// `.concinnity` segment is anchored onto the wrapping root.
fn resolve_state_dir(flat: Option<&Path>, root: Option<&Path>) -> PathBuf {
    flat.map_or_else(|| anchor(root, Path::new(STATE_DIR)), Path::to_path_buf)
}

/// The directory holding runtime-writable state (`saves/` + `settings`): the
/// writable override when one is installed, otherwise the state dir (writable
/// state sits beside `data/`).
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
pub(super) fn anchor(root: Option<&Path>, rel: &Path) -> PathBuf {
    root.map_or_else(|| rel.to_path_buf(), |r| r.join(rel))
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
    fn resolve_writable_dir_prefers_override_then_falls_back() {
        // An installed writable override wins verbatim, relocating only the
        // writable state; without one, writable state stays with the content.
        let state = Path::new("/game/MyGame");
        let over = Path::new("/users/me/AppData/Local/MyGame");
        assert_eq!(resolve_writable_dir(Some(over), state), over);
        assert_eq!(resolve_writable_dir(None, state), state);
    }
}
