//! Every `.jsonl` file under a directory, the walk `check` validates and
//! application discovery reads. It leaves out what a build or run writes, so
//! a project's outputs and a Cargo `target/` cost nothing to skip past.

use std::collections::HashSet;
use std::io::Error;
use std::path::{Path, PathBuf};

/// The extension of a file of assets.
pub(crate) const DATA_EXTENSION: &str = "jsonl";

// The file a directory of regenerable output carries (a Cargo `target/` among
// them), by the Cache Directory Tagging convention.
const CACHE_TAG: &str = "CACHEDIR.TAG";

/// What a walk for `.jsonl` files turned up: the files, and each directory that
/// could not be read with the reason.
#[derive(Default)]
pub(crate) struct DataFiles {
    pub(crate) files: Vec<PathBuf>,
    pub(crate) unreadable: Vec<(PathBuf, Error)>,
}

/// Every `.jsonl` file under `root`, sorted, each file and directory once.
/// Hidden entries are skipped, which leaves out `.git` and the CLI's own
/// `.concinnity`, as are the directories in `skip` and any tagged as a cache
/// with a `CACHEDIR.TAG`. Symlinks are followed.
pub(crate) fn data_files(root: &Path, skip: &[PathBuf]) -> DataFiles {
    let mut found = DataFiles::default();
    let mut seen: HashSet<PathBuf> = skip
        .iter()
        .filter_map(|d| std::fs::canonicalize(d).ok())
        .collect();
    collect(root, &mut seen, &mut found);
    let mut seen = HashSet::new();
    found
        .files
        .retain(|f| seen.insert(std::fs::canonicalize(f).unwrap_or_else(|_| f.clone())));
    found
}

/// The open project's output directories, which a walk skips.
pub(crate) fn project_outputs() -> Vec<PathBuf> {
    crate::project::tree().map_or_else(Vec::new, |tree| tree.output_dirs().into())
}

// Entries are walked in name order, so which spelling of a file reached twice
// is kept does not depend on the file system.
fn collect(dir: &Path, seen: &mut HashSet<PathBuf>, found: &mut DataFiles) {
    let entries = std::fs::canonicalize(dir).and_then(|canonical| {
        if !seen.insert(canonical) {
            return Ok(Vec::new());
        }
        let mut paths = std::fs::read_dir(dir)?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        paths.sort();
        Ok(paths)
    });
    let paths = match entries {
        Ok(paths) => paths,
        Err(e) => {
            found.unreadable.push((dir.to_path_buf(), e));
            return;
        }
    };
    for path in paths {
        if path
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with('.'))
        {
            continue;
        }
        if path.is_dir() {
            if !path.join(CACHE_TAG).exists() {
                collect(&path, seen, found);
            }
        } else if path.extension().is_some_and(|e| e == DATA_EXTENSION) {
            found.files.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(root: &Path, rel: &str) {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, "").unwrap();
    }

    fn relative(root: &Path) -> Vec<String> {
        data_files(root, &[])
            .files
            .iter()
            .map(|f| {
                f.strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect()
    }

    #[test]
    fn data_files_are_every_jsonl_below_the_root_in_order() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "app.jsonl");
        write(dir.path(), "props/b.jsonl");
        write(dir.path(), "props/a.jsonl");
        write(dir.path(), "props/deep/c.jsonl");
        write(dir.path(), "effects/embers.json");
        write(dir.path(), "notes.txt");
        assert_eq!(
            relative(dir.path()),
            [
                "app.jsonl",
                "props/a.jsonl",
                "props/b.jsonl",
                "props/deep/c.jsonl",
            ]
        );
    }

    #[test]
    fn data_files_skip_hidden_entries() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), ".git/objects.jsonl");
        write(dir.path(), ".hidden.jsonl");
        write(dir.path(), "app.jsonl");
        assert_eq!(relative(dir.path()), ["app.jsonl"]);
    }

    #[test]
    fn data_files_skip_the_named_and_the_tagged_directories() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "app.jsonl");
        write(dir.path(), "data/stale.jsonl");
        write(dir.path(), "target/CACHEDIR.TAG");
        write(dir.path(), "target/debug/fixture.jsonl");
        let found = data_files(
            dir.path(),
            &[dir.path().join("data"), dir.path().join("absent")],
        );
        assert_eq!(found.files, [dir.path().join("app.jsonl")]);
        assert!(found.unreadable.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn data_files_follow_symlinks_without_looping() {
        let dir = tempfile::tempdir().unwrap();
        let kits = tempfile::tempdir().unwrap();
        write(kits.path(), "kit.jsonl");
        std::os::unix::fs::symlink(kits.path(), dir.path().join("assets")).unwrap();
        std::os::unix::fs::symlink(dir.path(), dir.path().join("loop")).unwrap();
        assert_eq!(relative(dir.path()), ["assets/kit.jsonl"]);
    }

    #[cfg(unix)]
    #[test]
    fn a_file_reached_through_a_symlink_is_listed_once() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "app.jsonl");
        std::os::unix::fs::symlink(dir.path().join("app.jsonl"), dir.path().join("alias.jsonl"))
            .unwrap();
        assert_eq!(relative(dir.path()), ["alias.jsonl"]);
    }

    // An unreadable folder is reported by its path and the rest is still
    // walked.
    #[test]
    fn an_unreadable_directory_is_reported_and_the_walk_goes_on() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "app.jsonl");
        let missing = dir.path().join("gone");
        let mut found = DataFiles::default();
        collect(&missing, &mut HashSet::new(), &mut found);
        collect(dir.path(), &mut HashSet::new(), &mut found);
        assert_eq!(found.unreadable.len(), 1);
        assert_eq!(found.unreadable[0].0, missing);
        assert_eq!(found.files, [dir.path().join("app.jsonl")]);
    }
}
