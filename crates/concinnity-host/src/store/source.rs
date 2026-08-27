//! Finding an authored source asset on disk.
//!
//! A `source` string in world.jsonl may be a bare filename (a texture, HDRI,
//! `.cube` LUT, or shader named without a directory), which resolves against
//! an `assets/` directory by recursive search. Anything carrying a directory
//! component is a path already and is used verbatim, so absolute and relative
//! paths still work.
//!
//! The directory to search is the caller's: a build threads the root it was
//! given, and the hot-reload watcher passes the running project's. Nothing
//! here reads a process-global anchor, so two builds may resolve against
//! different trees in one process. The shipped runtime plays compiled blobs
//! and never resolves a source file.

use std::path::{Path, PathBuf};

/// Recursively search `assets_dir` for a file matching the given bare
/// filename, returning the first match.
pub fn find_in(assets_dir: &Path, filename: &str) -> Option<String> {
    walk(assets_dir, filename).map(|p| p.to_string_lossy().into_owned())
}

/// Resolve a source string into a filesystem path: bare filenames are searched
/// for under `assets_dir` and fall back to sitting directly in it, so a miss
/// still names the file the read error will report. With no assets dir the
/// source is returned verbatim, which is again the path the read error names.
pub fn resolve_source_path(source: &str, assets_dir: Option<&Path>) -> String {
    let Some(assets_dir) = assets_dir else {
        return source.to_string();
    };
    if !is_bare(source) {
        return source.to_string();
    }
    walk(assets_dir, source)
        .unwrap_or_else(|| assets_dir.join(source))
        .to_string_lossy()
        .into_owned()
}

// True when `source` names a file with no directory component.
fn is_bare(source: &str) -> bool {
    Path::new(source)
        .parent()
        .map(|d| d.as_os_str().is_empty())
        .unwrap_or(true)
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_filenames_are_distinguished_from_paths() {
        assert!(is_bare("sky.hdr"));
        assert!(!is_bare("hdri/sky.hdr"));
        assert!(!is_bare("/abs/sky.hdr"));
        assert!(!is_bare("../sky.hdr"));
    }

    #[test]
    fn a_path_with_a_directory_component_is_used_verbatim() {
        let dir = tempfile::tempdir().unwrap();
        // Even a path that does not exist is returned unchanged: it is already
        // a path, so no search applies.
        assert_eq!(
            resolve_source_path("hdri/sky.hdr", Some(dir.path())),
            "hdri/sky.hdr"
        );
        assert_eq!(
            resolve_source_path("/abs/sky.hdr", Some(dir.path())),
            "/abs/sky.hdr"
        );
    }

    #[test]
    fn a_bare_filename_is_found_in_a_nested_subdirectory() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("hdri").join("outdoor");
        std::fs::create_dir_all(&nested).unwrap();
        let target = nested.join("sky.hdr");
        std::fs::write(&target, b"x").unwrap();

        assert_eq!(
            resolve_source_path("sky.hdr", Some(dir.path())),
            target.to_string_lossy().into_owned()
        );
    }

    #[test]
    fn a_missing_bare_filename_falls_back_to_the_assets_dir() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            resolve_source_path("cn_no_such_asset.hdr", Some(dir.path())),
            dir.path()
                .join("cn_no_such_asset.hdr")
                .to_string_lossy()
                .into_owned()
        );
    }

    // With no assets dir there is nowhere to search, so a bare filename comes
    // back untouched rather than anchored to a guessed root.
    #[test]
    fn resolving_without_an_assets_dir_returns_the_source_verbatim() {
        assert_eq!(resolve_source_path("sky.hdr", None), "sky.hdr");
        assert_eq!(resolve_source_path("hdri/sky.hdr", None), "hdri/sky.hdr");
    }

    #[test]
    fn resolving_against_a_missing_assets_dir_still_names_the_file() {
        // An unreadable assets dir makes the search miss rather than fail, so
        // the caller gets a path whose read error names the file.
        let missing = Path::new("/nonexistent/cn/assets");
        assert_eq!(
            resolve_source_path("sky.hdr", Some(missing)),
            missing.join("sky.hdr").to_string_lossy().into_owned()
        );
    }

    #[test]
    fn walk_misses_cleanly_on_an_unreadable_directory() {
        assert_eq!(walk(Path::new("/nonexistent/cn/assets"), "x.png"), None);
    }

    #[test]
    fn find_in_returns_the_first_match_and_misses_cleanly() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("hdri");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("sky.hdr"), b"x").unwrap();

        assert_eq!(
            find_in(dir.path(), "sky.hdr").as_deref(),
            nested.join("sky.hdr").to_str()
        );
        assert_eq!(find_in(dir.path(), "cn_test_no_such_asset.json"), None);
    }
}
