//! Path arithmetic for the files an entry list holds, done on the spelling
//! alone: a file an entry is added to may not exist until the save that writes
//! it.

use std::path::Path;

use concinnity_host::store::source::normalized;

// Whether two spellings name the same file.
pub(super) fn same_file(a: &Path, b: &Path) -> bool {
    a == b || normalized(a) == normalized(b)
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
    fn spellings_of_one_file_are_the_same_file() {
        assert!(same_file(
            Path::new("nested/../assets/g.jsonl"),
            Path::new("./assets/g.jsonl")
        ));
        assert!(!same_file(
            Path::new("assets/g.jsonl"),
            Path::new("assets/h.jsonl")
        ));
    }

    #[test]
    fn a_path_under_the_working_directory_shows_relative() {
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(shown(&cwd.join("assets/./a.jsonl")), "assets/a.jsonl");
        assert_eq!(
            shown(Path::new("nested/../assets/a.jsonl")),
            "assets/a.jsonl"
        );
    }
}
