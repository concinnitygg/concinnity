//! `cn check`: validate one named world, or every world file under the working
//! directory.

use std::collections::HashSet;
use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};

use crate::authoring::{ValidationFailed, check_at_path};
use crate::command::resolve_world_path;

const WORLD_EXTENSION: &str = "jsonl";

/// Validate worlds and report their errors without building blobs.
///
/// `json_path` names one world, which must exist; `None` checks every `.jsonl`
/// file under the working directory.
pub(crate) fn check(json_path: Option<&str>) -> std::io::Result<()> {
    match json_path {
        Some(_) => {
            let path = resolve_world_path(json_path)?;
            check_at_path(&path, &path.display().to_string())
        }
        None => check_tree(Path::new(".")),
    }
}

// Check every world file under `root`, each on its own, and fail when any of
// them does or a directory could not be read.
fn check_tree(root: &Path) -> std::io::Result<()> {
    let found = world_files(root);
    let total = found.files.len() + found.unreadable.len();
    if total == 0 {
        return Err(Error::new(
            ErrorKind::NotFound,
            format!("no .{WORLD_EXTENSION} files under {}", root.display()),
        ));
    }
    let label = |path: &Path| {
        path.strip_prefix(root)
            .unwrap_or(path)
            .display()
            .to_string()
    };
    for (dir, e) in &found.unreadable {
        eprintln!("failed: {}\nerror:   {e}", label(dir));
    }
    let mut failed = found.unreadable.len();
    for file in &found.files {
        let label = label(file);
        if let Err(e) = check_at_path(file, &label) {
            if ValidationFailed::of(&e).is_none() {
                eprintln!("failed: {label}\nerror:   {e}");
            }
            failed += 1;
        }
    }
    if failed > 0 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("{failed} of {total} world file(s) or folder(s) failed"),
        ));
    }
    println!("ok: {total} world file(s) checked");
    Ok(())
}

// What a walk for world files turned up: the files, and each directory that
// could not be read with the reason.
#[derive(Default)]
struct WorldFiles {
    files: Vec<PathBuf>,
    unreadable: Vec<(PathBuf, Error)>,
}

// Every `.jsonl` file under `root`, sorted, each file and directory once.
// Hidden entries are skipped, which leaves out `.git` and the CLI's own
// `.concinnity`. Symlinks are followed.
fn world_files(root: &Path) -> WorldFiles {
    let mut found = WorldFiles::default();
    collect(root, &mut HashSet::new(), &mut found);
    let mut seen = HashSet::new();
    found
        .files
        .retain(|f| seen.insert(std::fs::canonicalize(f).unwrap_or_else(|_| f.clone())));
    found
}

// Entries are walked in name order, so which spelling of a file reached twice
// is kept does not depend on the file system.
fn collect(dir: &Path, seen: &mut HashSet<PathBuf>, found: &mut WorldFiles) {
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
            collect(&path, seen, found);
        } else if path.extension().is_some_and(|e| e == WORLD_EXTENSION) {
            found.files.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = "[\"PhysicsConfig\",{\"$id\":\"phys\"}]\n";
    const INVALID: &str = "[\"NotARealAssetType\",{\"$id\":\"x\"}]\n";

    fn write(root: &Path, rel: &str, contents: &str) {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, contents).unwrap();
    }

    fn relative(root: &Path) -> Vec<String> {
        world_files(root)
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

    // An explicit, existing path is validated in place -- the discovery branch
    // (and its process-global path anchors) is never touched.
    #[test]
    fn check_validates_an_explicit_existing_world() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("world.jsonl");
        std::fs::write(&path, VALID).unwrap();
        check(path.to_str()).unwrap();
    }

    #[test]
    fn check_reports_an_invalid_explicit_world() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("world.jsonl");
        std::fs::write(&path, INVALID).unwrap();
        assert!(check(path.to_str()).is_err());
    }

    #[test]
    fn check_of_a_missing_explicit_world_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("absent.jsonl");
        let err = check(path.to_str()).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::NotFound);
    }

    #[test]
    fn world_files_are_every_jsonl_below_the_root_in_order() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "world.jsonl", "");
        write(dir.path(), "props/b.jsonl", "");
        write(dir.path(), "props/a.jsonl", "");
        write(dir.path(), "props/deep/c.jsonl", "");
        write(dir.path(), "effects/embers.json", "");
        write(dir.path(), "notes.txt", "");
        assert_eq!(
            relative(dir.path()),
            [
                "props/a.jsonl",
                "props/b.jsonl",
                "props/deep/c.jsonl",
                "world.jsonl"
            ]
        );
    }

    #[test]
    fn world_files_skip_hidden_entries() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), ".git/objects.jsonl", "");
        write(dir.path(), ".hidden.jsonl", "");
        write(dir.path(), "world.jsonl", "");
        assert_eq!(relative(dir.path()), ["world.jsonl"]);
    }

    #[cfg(unix)]
    #[test]
    fn world_files_follow_symlinks_without_looping() {
        let dir = tempfile::tempdir().unwrap();
        let kits = tempfile::tempdir().unwrap();
        write(kits.path(), "kit.jsonl", "");
        std::os::unix::fs::symlink(kits.path(), dir.path().join("assets")).unwrap();
        std::os::unix::fs::symlink(dir.path(), dir.path().join("loop")).unwrap();
        assert_eq!(relative(dir.path()), ["assets/kit.jsonl"]);
    }

    #[test]
    fn check_tree_passes_when_every_file_does() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "world.jsonl", VALID);
        write(dir.path(), "kits/physics.jsonl", VALID);
        check_tree(dir.path()).unwrap();
    }

    #[test]
    fn check_tree_counts_the_files_that_fail() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "world.jsonl", VALID);
        write(dir.path(), "kits/bad.jsonl", INVALID);
        write(dir.path(), "kits/worse.jsonl", INVALID);
        let err = check_tree(dir.path()).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        assert!(err.to_string().contains("2 of 3"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn a_file_reached_through_a_symlink_is_listed_once() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "world.jsonl", "");
        std::os::unix::fs::symlink(
            dir.path().join("world.jsonl"),
            dir.path().join("alias.jsonl"),
        )
        .unwrap();
        assert_eq!(relative(dir.path()), ["alias.jsonl"]);
    }

    // An unreadable folder is reported by its path and the rest is still
    // walked and checked.
    #[test]
    fn an_unreadable_directory_is_reported_and_the_walk_goes_on() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "world.jsonl", VALID);
        let missing = dir.path().join("gone");
        let mut found = WorldFiles::default();
        collect(&missing, &mut HashSet::new(), &mut found);
        collect(dir.path(), &mut HashSet::new(), &mut found);
        assert_eq!(found.unreadable.len(), 1);
        assert_eq!(found.unreadable[0].0, missing);
        assert_eq!(found.files, [dir.path().join("world.jsonl")]);
    }

    #[cfg(unix)]
    #[test]
    fn check_tree_fails_on_an_unreadable_directory_but_checks_the_rest() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "world.jsonl", VALID);
        write(dir.path(), "locked/kit.jsonl", VALID);
        let locked = dir.path().join("locked");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        let result = check_tree(dir.path());
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
        let err = result.unwrap_err();
        assert!(err.to_string().contains("1 of 2"), "{err}");
    }

    #[test]
    fn check_tree_resolves_includes_from_each_file() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "kits/physics.jsonl", VALID);
        write(
            dir.path(),
            "worlds/world.jsonl",
            "[\"Include\",{\"path\":\"../kits/physics.jsonl\"}]\n",
        );
        check_tree(dir.path()).unwrap();
    }

    #[test]
    fn check_tree_without_world_files_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "notes.txt", "");
        let err = check_tree(dir.path()).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::NotFound);
    }
}
