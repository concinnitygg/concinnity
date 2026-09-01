// src/editor/world_files.rs
//
// The world files a project offers the Worlds panel, and the naming rules a
// new one has to pass. Pure filesystem work over paths the caller resolves, so
// the panel's listing, creation, and deletion are testable without a session.
//
// A project keeps its worlds in `worlds/*.jsonl`. A `world.jsonl` sitting at
// the project root is where worlds lived before that, and is still listed so a
// legacy project stays openable from the panel.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::world::WORLD_JSONL;

// The extension every world file carries.
const WORLD_EXT: &str = "jsonl";

// Characters a world name cannot carry, because a file named with them is
// either impossible or means something else to a path. The `file_stem` check in
// `validate_name` backstops anything this list misses.
const ILLEGAL: [char; 9] = ['/', '\\', ':', '*', '?', '"', '<', '>', '|'];

// One listed world: the stem it shows as, the file behind it, and when that
// file was last written (the panel's sort key).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WorldFile {
    pub name: String,
    pub path: PathBuf,
    pub modified: SystemTime,
}

// Every world a project offers, most recently edited first. Ties break by name
// so a listing is stable when two files share an mtime. Directories that do not
// exist contribute nothing rather than failing the listing.
pub(crate) fn list(worlds_dir: Option<&Path>, content_root: Option<&Path>) -> Vec<WorldFile> {
    let mut found: Vec<WorldFile> = Vec::new();
    if let Some(dir) = worlds_dir
        && let Ok(entries) = std::fs::read_dir(dir)
    {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some(WORLD_EXT) {
                continue;
            }
            if let Some(world) = world_at(&path) {
                found.push(world);
            }
        }
    }
    if let Some(root) = content_root {
        let legacy = root.join(WORLD_JSONL);
        if !found.iter().any(|w| w.path == legacy)
            && let Some(world) = world_at(&legacy)
        {
            found.push(world);
        }
    }
    found.sort_by(|a, b| {
        b.modified
            .cmp(&a.modified)
            .then_with(|| a.name.cmp(&b.name))
    });
    found
}

// The world at `path`, or `None` when it is not a readable file.
fn world_at(path: &Path) -> Option<WorldFile> {
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_file() {
        return None;
    }
    Some(WorldFile {
        name: path.file_stem()?.to_string_lossy().into_owned(),
        path: path.to_path_buf(),
        modified: meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
    })
}

// Check a typed world name, returning the name to create or the reason it was
// rejected (shown on the panel's status line). `existing` is every listed
// world's name; the comparison ignores case, since the filesystems the editor
// runs on mostly do.
pub(crate) fn validate_name(name: &str, existing: &[String]) -> Result<String, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("Enter a world name".to_string());
    }
    if name.starts_with('.') {
        return Err("A world name cannot start with a dot".to_string());
    }
    if name.contains(ILLEGAL) || name.chars().any(char::is_control) {
        return Err("A world name cannot contain / \\ : * ? \" < > |".to_string());
    }
    // What the name has to survive to be addressable: the file it produces has
    // to read back as the same stem, which is how every other path here (the
    // listing, the session store) names a world.
    let file = format!("{name}.{WORLD_EXT}");
    if Path::new(&file).file_stem().and_then(|s| s.to_str()) != Some(name) {
        return Err(format!("'{name}' is not a usable world name"));
    }
    if existing.iter().any(|e| e.eq_ignore_ascii_case(name)) {
        return Err(format!("A world named '{name}' already exists"));
    }
    Ok(name.to_string())
}

// Create an empty world file under `dir`, failing rather than clobbering when
// something already sits at the path. The directory is created first: a fresh
// project has no `worlds/` until its first world.
pub(crate) fn create(dir: &Path, name: &str) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{name}.{WORLD_EXT}"));
    std::fs::File::options()
        .write(true)
        .create_new(true)
        .open(&path)?;
    Ok(path)
}

pub(crate) fn delete(path: &Path) -> std::io::Result<()> {
    std::fs::remove_file(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn touch(path: &Path, secs: u64) {
        std::fs::write(path, "").unwrap();
        let file = std::fs::File::options().write(true).open(path).unwrap();
        file.set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(secs))
            .unwrap();
    }

    fn names(worlds: &[WorldFile]) -> Vec<String> {
        worlds.iter().map(|w| w.name.clone()).collect()
    }

    // The listing is newest-edited first, so the panel opens on what the user
    // was last working in.
    #[test]
    fn worlds_list_newest_edited_first() {
        let tree = concinnity_testing::TempTree::new();
        let dir = tree.path().join("worlds");
        std::fs::create_dir_all(&dir).unwrap();
        touch(&dir.join("old.jsonl"), 1_000);
        touch(&dir.join("newest.jsonl"), 3_000);
        touch(&dir.join("middle.jsonl"), 2_000);

        let listed = list(Some(&dir), None);
        assert_eq!(names(&listed), ["newest", "middle", "old"]);
        assert_eq!(listed[0].path, dir.join("newest.jsonl"));
    }

    // Only `.jsonl` files are worlds, and files that share an mtime still list
    // in a stable order.
    #[test]
    fn non_world_files_are_skipped_and_ties_break_by_name() {
        let tree = concinnity_testing::TempTree::new();
        let dir = tree.path().join("worlds");
        std::fs::create_dir_all(&dir).unwrap();
        touch(&dir.join("b.jsonl"), 5_000);
        touch(&dir.join("a.jsonl"), 5_000);
        touch(&dir.join("notes.txt"), 9_000);
        std::fs::create_dir_all(dir.join("nested.jsonl")).unwrap();

        assert_eq!(names(&list(Some(&dir), None)), ["a", "b"]);
    }

    // A project that still keeps its world at the root lists it alongside the
    // `worlds/` ones, so the panel can open it.
    #[test]
    fn a_legacy_root_world_is_listed_too() {
        let tree = concinnity_testing::TempTree::new();
        let dir = tree.path().join("worlds");
        std::fs::create_dir_all(&dir).unwrap();
        touch(&dir.join("arena.jsonl"), 1_000);
        touch(&tree.path().join(WORLD_JSONL), 2_000);

        let listed = list(Some(&dir), Some(tree.path()));
        assert_eq!(names(&listed), ["world", "arena"]);
        assert_eq!(listed[0].path, tree.path().join(WORLD_JSONL));
    }

    // Nothing to list is an empty listing, not a failure: a project with no
    // `worlds/` yet still opens the panel.
    #[test]
    fn missing_directories_list_nothing() {
        let tree = concinnity_testing::TempTree::new();
        assert!(list(Some(&tree.path().join("absent")), Some(tree.path())).is_empty());
        assert!(list(None, None).is_empty());
    }

    #[test]
    fn a_usable_name_is_accepted_and_trimmed() {
        assert_eq!(validate_name("  arena  ", &[]), Ok("arena".to_string()));
        assert_eq!(validate_name("level.02", &[]), Ok("level.02".to_string()));
    }

    // Every rejection says why, so the panel never fails silently.
    #[test]
    fn unusable_names_are_rejected_with_a_reason() {
        for bad in ["", "   ", ".", "..", ".hidden", "a/b", "a\\b", "a:b", "a?b"] {
            let Err(err) = validate_name(bad, &[]) else {
                panic!("'{bad}' was accepted as a world name");
            };
            assert!(!err.is_empty(), "'{bad}' was rejected without a reason");
        }
    }

    #[test]
    fn a_duplicate_name_is_rejected_whatever_its_case() {
        let existing = vec!["Arena".to_string()];
        let err = validate_name("arena", &existing).unwrap_err();
        assert!(err.contains("arena"), "message was: {err}");
        assert!(validate_name("arena2", &existing).is_ok());
    }

    // Creating makes the file (and the directory it needs) so the world lists
    // immediately, and never overwrites one that is already there.
    #[test]
    fn create_makes_an_empty_file_and_refuses_to_clobber() {
        let tree = concinnity_testing::TempTree::new();
        let dir = tree.path().join("worlds");
        let path = create(&dir, "arena").unwrap();
        assert_eq!(path, dir.join("arena.jsonl"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "");
        assert_eq!(names(&list(Some(&dir), None)), ["arena"]);

        let err = create(&dir, "arena").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
    }

    #[test]
    fn delete_removes_the_file_from_the_listing() {
        let tree = concinnity_testing::TempTree::new();
        let dir = tree.path().join("worlds");
        let path = create(&dir, "arena").unwrap();
        delete(&path).unwrap();
        assert!(list(Some(&dir), None).is_empty());
        assert_eq!(
            delete(&path).unwrap_err().kind(),
            std::io::ErrorKind::NotFound
        );
    }
}
