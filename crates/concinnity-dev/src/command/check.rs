//! `cn check`: validate one named file, or every `.jsonl` file under the
//! working directory.

use std::io::{Error, ErrorKind};
use std::path::Path;

use crate::authoring::{ValidationFailed, check_at_path};
use crate::command::Target;
use crate::command::data_files::{DATA_EXTENSION, data_files, project_outputs};

/// Validate files of assets and report their errors without building blobs.
///
/// `json_path` names one file, which must exist; `None` checks every `.jsonl`
/// file under the working directory.
pub(crate) fn check(json_path: Option<&str>) -> std::io::Result<()> {
    match json_path {
        Some(file) => {
            let path = Target::file(file).resolve()?;
            let root = crate::project::content_root();
            check_at_path(&path, root.as_deref(), &path.display().to_string())
        }
        None => check_tree(Path::new(".")),
    }
}

// Check every `.jsonl` file under `root`, each on its own with its includes
// resolved against `root`, and fail when any of them does or a directory could
// not be read.
fn check_tree(root: &Path) -> std::io::Result<()> {
    let found = data_files(root, &project_outputs());
    let total = found.files.len() + found.unreadable.len();
    if total == 0 {
        return Err(Error::new(
            ErrorKind::NotFound,
            format!("no .{DATA_EXTENSION} files under {}", root.display()),
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
        if let Err(e) = check_at_path(file, Some(root), &label) {
            if ValidationFailed::of(&e).is_none() {
                eprintln!("failed: {label}\nerror:   {e}");
            }
            failed += 1;
        }
    }
    if failed > 0 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("{failed} of {total} file(s) or folder(s) failed"),
        ));
    }
    println!("ok: {total} file(s) checked");
    Ok(())
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

    // An explicit, existing path is validated in place -- the discovery branch
    // (and its process-global path anchors) is never touched.
    #[test]
    fn check_validates_an_explicit_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("app.jsonl");
        std::fs::write(&path, VALID).unwrap();
        check(path.to_str()).unwrap();
    }

    #[test]
    fn check_reports_an_invalid_explicit_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("app.jsonl");
        std::fs::write(&path, INVALID).unwrap();
        assert!(check(path.to_str()).is_err());
    }

    #[test]
    fn check_of_a_missing_explicit_file_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("absent.jsonl");
        let err = check(path.to_str()).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::NotFound);
    }

    #[test]
    fn check_tree_passes_when_every_file_does() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "app.jsonl", VALID);
        write(dir.path(), "kits/physics.jsonl", VALID);
        check_tree(dir.path()).unwrap();
    }

    #[test]
    fn check_tree_counts_the_files_that_fail() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "app.jsonl", VALID);
        write(dir.path(), "kits/bad.jsonl", INVALID);
        write(dir.path(), "kits/worse.jsonl", INVALID);
        let err = check_tree(dir.path()).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        assert!(err.to_string().contains("2 of 3"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn check_tree_fails_on_an_unreadable_directory_but_checks_the_rest() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "app.jsonl", VALID);
        write(dir.path(), "locked/kit.jsonl", VALID);
        let locked = dir.path().join("locked");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        let result = check_tree(dir.path());
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
        let err = result.unwrap_err();
        assert!(err.to_string().contains("1 of 2"), "{err}");
    }

    #[test]
    fn check_tree_resolves_includes_against_the_root() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "kits/physics.jsonl", VALID);
        write(
            dir.path(),
            "apps/arena.jsonl",
            "[\"Include\",{\"path\":\"kits/physics.jsonl\"}]\n",
        );
        check_tree(dir.path()).unwrap();
    }

    #[test]
    fn check_tree_without_jsonl_files_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "notes.txt", "");
        let err = check_tree(dir.path()).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::NotFound);
    }
}
