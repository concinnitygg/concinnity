//! Path arithmetic for `Include` lines, done on the spelling alone: a file an
//! entry is added to may not exist until the save that writes it.

use std::path::{Component, Path, PathBuf};

// The file an `Include` line in `file` names by `path`, spelled as the build
// spells it: joined onto the directory of the file the line is in.
pub(super) fn include_target(file: &Path, path: &str) -> PathBuf {
    file.parent().unwrap_or_else(|| Path::new("")).join(path)
}

// `path` made absolute against the working directory, with every `.` and `..`
// folded away lexically.
pub(super) fn normalized(path: &Path) -> PathBuf {
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

// Whether two spellings name the same file.
pub(super) fn same_file(a: &Path, b: &Path) -> bool {
    a == b || normalized(a) == normalized(b)
}

// How an `Include` line in a file under `dir` names `target`: relative, with
// `/` separators, so the line reads the same on every host.
pub(super) fn relative_from(dir: &Path, target: &Path) -> String {
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

// `path` as the editor shows it: relative to the working directory when it
// sits under it.
pub(crate) fn shown(path: &Path) -> String {
    let path = normalized(path);
    let cwd = std::env::current_dir().map(|d| normalized(&d));
    let shown = cwd
        .as_deref()
        .ok()
        .and_then(|cwd| path.strip_prefix(cwd).ok())
        .unwrap_or(&path);
    shown
        .components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_target_joins_onto_the_including_files_directory() {
        assert_eq!(
            include_target(Path::new("worlds/w.jsonl"), "../assets/g.jsonl"),
            Path::new("worlds/../assets/g.jsonl")
        );
        assert_eq!(
            include_target(Path::new("w.jsonl"), "g.jsonl"),
            Path::new("g.jsonl")
        );
    }

    #[test]
    fn spellings_of_one_file_are_the_same_file() {
        assert!(same_file(
            Path::new("worlds/../assets/g.jsonl"),
            Path::new("./assets/g.jsonl")
        ));
        assert!(!same_file(
            Path::new("assets/g.jsonl"),
            Path::new("assets/h.jsonl")
        ));
    }

    #[test]
    fn a_relative_path_climbs_out_of_the_including_directory() {
        let root = Path::new("/p");
        assert_eq!(
            relative_from(&root.join("worlds"), &root.join("assets/ground/a.jsonl")),
            "../assets/ground/a.jsonl"
        );
        assert_eq!(
            relative_from(root, &root.join("assets/a.jsonl")),
            "assets/a.jsonl"
        );
        assert_eq!(
            relative_from(Path::new("worlds"), Path::new("worlds/../assets/a.jsonl")),
            "../assets/a.jsonl",
            "relative spellings resolve against the working directory"
        );
    }

    #[test]
    fn a_path_under_the_working_directory_shows_relative() {
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(shown(&cwd.join("assets/./a.jsonl")), "assets/a.jsonl");
        assert_eq!(
            shown(Path::new("worlds/../assets/a.jsonl")),
            "assets/a.jsonl"
        );
    }
}
