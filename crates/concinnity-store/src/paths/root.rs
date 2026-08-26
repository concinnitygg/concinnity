// Anchoring: which directory the state tree hangs off.
//
// There is no default. A host installs a state directory before anything reads
// the tree, and an uninstalled root resolves to `None` rather than a guess
// against the working directory: a library that guessed would scatter a
// project's settings and saves beside whatever directory its caller happened
// to launch from.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

fn installed_state_dir() -> &'static Mutex<Option<PathBuf>> {
    static STATE: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(None))
}

fn writable_state_override() -> &'static Mutex<Option<PathBuf>> {
    static WRITABLE: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
    WRITABLE.get_or_init(|| Mutex::new(None))
}

/// Anchor the state tree at `dir` for the rest of the process, so `data/`,
/// `saves/`, and `settings` resolve under it. Every host installs one before
/// reading project state: the dev CLI its project directory, a shipped
/// application the directory beside its executable (or inside its app bundle),
/// an embedder whatever its own layout implies.
pub fn set_state_dir<P: Into<PathBuf>>(dir: P) {
    *installed_state_dir().lock().unwrap() = Some(dir.into());
}

/// Remove an installed state dir, leaving the process with no state tree.
pub fn clear_state_dir() {
    *installed_state_dir().lock().unwrap() = None;
}

/// Anchor the runtime-writable state (`saves/` + `settings`) at `dir`, leaving
/// the read-only content (`data/`) at the installed state dir. A shipped
/// application installs this when its content dir is not writable (a read-only
/// install such as Program Files), redirecting only what it writes at runtime
/// to a per-user directory. When unset, writable state stays beside `data/`.
pub fn set_writable_state_dir<P: Into<PathBuf>>(dir: P) {
    *writable_state_override().lock().unwrap() = Some(dir.into());
}

/// Remove an installed writable-state dir, restoring writable state to the
/// content root beside `data/`.
pub fn clear_writable_state_dir() {
    *writable_state_override().lock().unwrap() = None;
}

/// The state directory, or `None` when no host installed one.
pub fn state_dir() -> Option<PathBuf> {
    installed_state_dir().lock().unwrap().clone()
}

/// The directory holding runtime-writable state (`saves/` + `settings`): the
/// writable override when one is installed, otherwise the state dir (writable
/// state sits beside `data/`).
pub fn writable_state_dir() -> Option<PathBuf> {
    let over = writable_state_override().lock().unwrap().clone();
    resolve_writable_dir(over.as_deref(), state_dir().as_deref())
}

// Pure resolution split out so the fallback rule is unit-testable without the
// process-global override: the override verbatim, else the content state dir.
fn resolve_writable_dir(over: Option<&Path>, state: Option<&Path>) -> Option<PathBuf> {
    over.or(state).map(Path::to_path_buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_writable_dir_prefers_override_then_falls_back() {
        // An installed writable override wins verbatim, relocating only the
        // writable state; without one, writable state stays with the content.
        let state = Path::new("/game/MyGame");
        let over = Path::new("/users/me/AppData/Local/MyGame");
        assert_eq!(
            resolve_writable_dir(Some(over), Some(state)).as_deref(),
            Some(over)
        );
        assert_eq!(
            resolve_writable_dir(None, Some(state)).as_deref(),
            Some(state)
        );
    }

    // An override with no content root behind it still resolves: a host may
    // redirect its writable state without ever installing a state dir.
    #[test]
    fn resolve_writable_dir_without_a_state_dir() {
        let over = Path::new("/users/me/MyGame");
        assert_eq!(
            resolve_writable_dir(Some(over), None).as_deref(),
            Some(over)
        );
        assert_eq!(resolve_writable_dir(None, None), None);
    }
}
