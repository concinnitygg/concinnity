//! Where a path a line names points on disk: one home for the rule, so every
//! reader of a .jsonl file resolves a path the same way.
//!
//! A relative path is relative to a root. A file and each file it includes
//! take theirs from the nearest directory at or above them, and within the
//! build's root, holding a [`LIBRARY_MARKER`], so a library of .jsonl files
//! names its own files the same whether it is run on its own or sits inside a
//! project. Once a file is read, every path in it is [`rebase`]d onto the one
//! root the build was given, and everything after that [`resolve`]s against
//! that root. An absolute path names itself everywhere.
//!
//! The root is the caller's: nothing here picks one, so two builds may resolve
//! against different trees in one process.

use std::path::{Component, Path, PathBuf};

/// The directory whose presence marks its parent as a library root.
pub const LIBRARY_MARKER: &str = ".concinnity";

/// The nearest directory at or above `file` holding a [`LIBRARY_MARKER`]
/// directory, looking no higher than `within` (the build's root), so a marker
/// outside the tree being built never takes a file over.
pub fn marked_root(file: &Path, within: &Path) -> Option<PathBuf> {
    let within = normalized(within);
    normalized(file)
        .ancestors()
        .skip(1)
        .take_while(|dir| dir.starts_with(&within))
        .find(|dir| dir.join(LIBRARY_MARKER).is_dir())
        .map(Path::to_path_buf)
}

/// The file `path` names under `root`: an absolute path verbatim, a relative
/// one joined onto `root`. With no root a relative path is returned as is.
pub fn resolve(path: &str, root: Option<&Path>) -> PathBuf {
    let path = Path::new(path);
    match root {
        Some(root) if path.is_relative() => root.join(path),
        _ => path.to_path_buf(),
    }
}

/// [`resolve`], when the file is there. `None` otherwise, so the caller's read
/// error names the path it meant.
pub fn find_existing(path: &str, root: Option<&Path>) -> Option<PathBuf> {
    Some(resolve(path, root)).filter(|p| p.exists())
}

/// `path`, written relative to `from`, rewritten to name the same file relative
/// to `to`, with `/` separators. An absolute or empty path, or one whose roots
/// are the same, is returned unchanged.
pub fn rebase(path: &str, from: &Path, to: &Path) -> String {
    if path.is_empty() || Path::new(path).is_absolute() {
        return path.to_string();
    }
    let (from, to) = (normalized(from), normalized(to));
    if from == to {
        return path.to_string();
    }
    relative_from(&to, &normalized(&from.join(path)))
}

/// `path` made absolute against the working directory, with every `.` and `..`
/// folded away lexically: two spellings of one file compare equal.
pub fn normalized(path: &Path) -> PathBuf {
    let absolute = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let mut out = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other),
        }
    }
    out
}

/// How a path relative to `dir` names `target`, with `/` separators, so a
/// file reads the same on every host.
pub fn relative_from(dir: &Path, target: &Path) -> String {
    let dir = normalized(dir);
    let target = normalized(target);
    let shared = dir
        .components()
        .zip(target.components())
        .take_while(|(a, b)| a == b)
        .count();
    let ups = dir.components().count() - shared;
    std::iter::repeat_n("..".to_string(), ups)
        .chain(
            target
                .components()
                .skip(shared)
                .map(|c| c.as_os_str().to_string_lossy().into_owned()),
        )
        .collect::<Vec<_>>()
        .join("/")
}

/// Recursively search `dir` for a file named `filename`, returning the first
/// match: how a preset is found by name.
pub fn find_in(dir: &Path, filename: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && path.file_name().and_then(|n| n.to_str()) == Some(filename) {
            return Some(path);
        }
        if path.is_dir()
            && let Some(found) = find_in(&path, filename)
        {
            return Some(found);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn marked(root: &Path) {
        std::fs::create_dir_all(root.join(LIBRARY_MARKER)).unwrap();
    }

    #[test]
    fn a_relative_path_joins_onto_the_root() {
        let root = Path::new("/p");
        assert_eq!(
            resolve("shaders/a.hlsl", Some(root)),
            root.join("shaders/a.hlsl")
        );
        assert_eq!(resolve("a.hlsl", Some(root)), root.join("a.hlsl"));
        assert_eq!(resolve("/abs/a.hlsl", Some(root)), Path::new("/abs/a.hlsl"));
        assert_eq!(resolve("a.hlsl", None), Path::new("a.hlsl"));
    }

    #[test]
    fn find_existing_names_only_a_file_that_is_there() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("shaders")).unwrap();
        std::fs::write(dir.path().join("shaders/blob.hlsl"), b"x").unwrap();
        assert_eq!(
            find_existing("shaders/blob.hlsl", Some(dir.path())),
            Some(dir.path().join("shaders/blob.hlsl"))
        );
        assert_eq!(find_existing("blob.hlsl", Some(dir.path())), None);
    }

    #[test]
    fn the_nearest_marked_directory_is_the_library_root() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path();
        let library = project.join("assets");
        marked(project);
        marked(&library);
        let kit = library.join("shaders/kit.jsonl");
        assert_eq!(marked_root(&kit, project), Some(normalized(&library)));
        assert_eq!(
            marked_root(&project.join("worlds/w.jsonl"), project),
            Some(normalized(project))
        );
    }

    #[test]
    fn a_file_with_no_marked_directory_above_has_no_library_root() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            marked_root(&dir.path().join("worlds/w.jsonl"), dir.path()),
            None
        );
    }

    // A marker above the build's root, such as a checkout the project sits
    // in, is not the project's.
    #[test]
    fn a_marker_above_the_build_root_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        marked(dir.path());
        let project = dir.path().join("project");
        let world = project.join("worlds/w.jsonl");
        assert_eq!(marked_root(&world, &project), None);
        assert_eq!(
            marked_root(&world, dir.path()),
            Some(normalized(dir.path()))
        );
    }

    // A marker that is a file, not a directory, marks nothing.
    #[test]
    fn a_marker_file_is_not_a_library_root() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(LIBRARY_MARKER), b"").unwrap();
        assert_eq!(marked_root(&dir.path().join("w.jsonl"), dir.path()), None);
    }

    #[test]
    fn a_path_is_rebased_onto_another_root() {
        let project = Path::new("/p");
        let library = Path::new("/p/assets");
        assert_eq!(
            rebase("shaders/a.hlsl", library, project),
            "assets/shaders/a.hlsl"
        );
        assert_eq!(rebase("../x/a.hlsl", library, project), "x/a.hlsl");
        assert_eq!(
            rebase("worlds/a.jsonl", project, library),
            "../worlds/a.jsonl"
        );
    }

    #[test]
    fn an_absolute_empty_or_same_root_path_is_not_rebased() {
        let (a, b) = (Path::new("/p/assets"), Path::new("/p"));
        assert_eq!(rebase("/abs/a.hlsl", a, b), "/abs/a.hlsl");
        assert_eq!(rebase("", a, b), "");
        assert_eq!(rebase("./a.hlsl", a, Path::new("/p/./assets")), "./a.hlsl");
    }

    #[test]
    fn spellings_of_one_file_normalize_alike() {
        assert_eq!(
            normalized(Path::new("nested/../assets/g.jsonl")),
            normalized(Path::new("./assets/g.jsonl"))
        );
    }

    #[test]
    fn a_relative_path_climbs_out_of_its_directory() {
        let root = Path::new("/p");
        assert_eq!(
            relative_from(&root.join("nested"), &root.join("assets/ground/a.jsonl")),
            "../assets/ground/a.jsonl"
        );
        assert_eq!(
            relative_from(root, &root.join("assets/a.jsonl")),
            "assets/a.jsonl"
        );
    }

    #[test]
    fn find_in_returns_the_first_match_and_misses_cleanly() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("hdri");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("sky.hdr"), b"x").unwrap();

        assert_eq!(find_in(dir.path(), "sky.hdr"), Some(nested.join("sky.hdr")));
        assert_eq!(find_in(dir.path(), "cn_test_no_such_asset.json"), None);
        assert_eq!(find_in(Path::new("/nonexistent/cn/assets"), "x.png"), None);
    }
}
