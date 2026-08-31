//! Where a player run reads its world from and writes its state to.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use concinnity_engine::StateTree;

// The state tree this player runs against: the content beside the executable,
// with the runtime-writable state (`saves/`, `settings`, `crashes/`) redirected
// to a per-user directory when the content dir cannot be written -- a read-only
// install such as Program Files. In the portable case (content dir writable)
// both stay beside the data, preserving the single-folder layout. The world's
// own `AppConfig.home`, applied once the blob is read, overrides either.
pub(crate) fn tree_for_exe(exe: &Path, exe_dir: &Path) -> StateTree {
    let content = state_dir_for_exe(exe_dir);
    let writable = (!dir_is_writable(&content))
        .then(|| per_user_state_dir(&app_name_from_exe(exe)))
        .flatten();
    match writable {
        Some(dir) => StateTree::at(content).with_writable(dir),
        None => StateTree::at(content),
    }
}

// Resolve the state root that holds the world's `data` (and, unless redirected,
// the `saves/` + `settings` written at runtime) from the executable's
// directory. Inside a macOS `.app` the executable sits at `Contents/MacOS/<exe>`
// and the data lives in `Contents/Resources/`; everywhere else the data sits
// directly beside the executable.
pub(crate) fn state_dir_for_exe(exe_dir: &Path) -> PathBuf {
    let in_app_bundle = exe_dir.file_name() == Some(OsStr::new("MacOS"))
        && exe_dir.parent().and_then(Path::file_name) == Some(OsStr::new("Contents"));
    match exe_dir.parent() {
        Some(contents) if in_app_bundle => contents.join("Resources"),
        _ => exe_dir.to_path_buf(),
    }
}

// The application name used to key the per-user writable directory: the
// executable's file stem (the export slug), falling back to a generic name.
pub(crate) fn app_name_from_exe(exe: &Path) -> String {
    exe.file_stem()
        .and_then(OsStr::to_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("concinnity")
        .to_string()
}

// Whether `dir` accepts new files. Probes by creating (and removing) a uniquely
// named file; a read-only install (Program Files) fails here. A missing dir is
// treated as writable -- the runtime creates `saves/` under it on first save.
pub(crate) fn dir_is_writable(dir: &Path) -> bool {
    if !dir.exists() {
        return true;
    }
    let probe = dir.join(format!(".cn-write-probe-{}", std::process::id()));
    match std::fs::File::create(&probe) {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

// A per-user, always-writable directory for `saves/` + `settings`, keyed by the
// app name. `None` only when the platform's base directory cannot be resolved
// from the environment, in which case the caller leaves writable state beside
// the data.
pub(crate) fn per_user_state_dir(app: &str) -> Option<PathBuf> {
    per_user_base().map(|base| base.join(app))
}

// The platform base for per-user application state.
#[cfg(windows)]
fn per_user_base() -> Option<PathBuf> {
    // %LOCALAPPDATA% (e.g. C:\Users\<user>\AppData\Local), falling back to the
    // roaming %APPDATA% if the local one is somehow unset.
    non_empty_env("LOCALAPPDATA")
        .or_else(|| non_empty_env("APPDATA"))
        .map(PathBuf::from)
}

#[cfg(target_os = "macos")]
fn per_user_base() -> Option<PathBuf> {
    non_empty_env("HOME").map(|h| PathBuf::from(h).join("Library").join("Application Support"))
}

#[cfg(all(unix, not(target_os = "macos")))]
fn per_user_base() -> Option<PathBuf> {
    // The XDG base-directory spec: $XDG_DATA_HOME, else ~/.local/share.
    non_empty_env("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| non_empty_env("HOME").map(|h| PathBuf::from(h).join(".local").join("share")))
}

// An environment variable's value when set and non-empty. Keeps the base
// resolvers from returning a base rooted at "" (which would place per-user
// state at the filesystem root).
#[cfg(any(windows, unix))]
fn non_empty_env(key: &str) -> Option<std::ffi::OsString> {
    std::env::var_os(key).filter(|v| !v.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_layout_uses_the_executable_directory() {
        // A portable folder (Windows/Linux, or a bare macOS binary): the state
        // tree sits directly beside the executable.
        let dir = Path::new("/apps/MyApp");
        assert_eq!(state_dir_for_exe(dir), Path::new("/apps/MyApp"));
    }

    #[test]
    fn macos_app_bundle_uses_resources() {
        // Contents/MacOS/<exe> -> data under Contents/Resources.
        let dir = Path::new("/Applications/MyGame.app/Contents/MacOS");
        assert_eq!(
            state_dir_for_exe(dir),
            Path::new("/Applications/MyGame.app/Contents/Resources")
        );
    }

    #[test]
    fn macos_like_path_not_in_bundle_stays_beside_exe() {
        // A `MacOS` directory that is not under `Contents` is not a bundle.
        let dir = Path::new("/home/user/MacOS");
        assert_eq!(state_dir_for_exe(dir), Path::new("/home/user/MacOS"));
    }

    #[test]
    fn app_name_falls_back_when_stem_missing() {
        assert_eq!(app_name_from_exe(Path::new("/apps/MyGame")), "MyGame");
        assert_eq!(app_name_from_exe(Path::new("MyGame.exe")), "MyGame");
        // No file name at all: the generic fallback keeps the path well-formed.
        assert_eq!(app_name_from_exe(Path::new("/")), "concinnity");
    }

    // A writable content dir keeps the single-folder layout: everything the
    // player reads and writes stays beside the executable.
    #[test]
    fn a_writable_install_keeps_one_folder() {
        let tmp = concinnity_testing::TempTree::new();
        let exe = tmp.path().join("MyGame");
        let tree = tree_for_exe(&exe, tmp.path());

        assert_eq!(tree.content_root(), tmp.path());
        assert_eq!(tree.writable_root(), tmp.path());
        assert_eq!(tree.data_dir(), tmp.path().join("data"));
        assert_eq!(tree.saves_dir(), tmp.path().join("saves"));
    }

    #[test]
    fn a_writable_dir_probes_true_and_leaves_nothing_behind() {
        let tmp = concinnity_testing::TempTree::new();
        assert!(dir_is_writable(tmp.path()));
        // The probe file is cleaned up.
        assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 0);
    }

    #[test]
    fn a_missing_dir_is_treated_as_writable() {
        let tmp = concinnity_testing::TempTree::new();
        let missing = tmp.path().join("not-created-yet");
        assert!(dir_is_writable(&missing));
    }

    #[test]
    fn per_user_dir_appends_the_app_name_under_a_base() {
        // The host always has a resolvable base (HOME / LOCALAPPDATA), so the
        // per-user dir is Some and ends with the app name.
        let dir = per_user_state_dir("MyGame").expect("a per-user base on the test host");
        assert_eq!(dir.file_name().and_then(OsStr::to_str), Some("MyGame"));
        assert!(dir.is_absolute());
    }
}
