use std::path::{Path, PathBuf};

/// The world source file's conventional name.
pub const WORLD_JSONL: &str = "world.jsonl";

/// Locate the project's world: `world.jsonl` in the working directory, or in
/// the nearest parent directory holding one.
///
/// # Errors
///
/// Returns an error if the working directory cannot be read, or no world is
/// found by the search.
pub fn find_world_jsonl() -> std::io::Result<PathBuf> {
    find_world_from(&std::env::current_dir()?)
}

// The nearest `world.jsonl` at `dir` or above it. Split from the working
// directory lookup so the walk is testable without moving the process cwd,
// which every other test in this binary shares.
fn find_world_from(dir: &Path) -> std::io::Result<PathBuf> {
    dir.ancestors()
        .map(|d| d.join(WORLD_JSONL))
        .find(|p| p.exists())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no world found: run `concinnity init` or create `{WORLD_JSONL}`"),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_world_in_the_directory_itself_is_found() {
        let tree = concinnity_testing::TempTree::new();
        let world = tree.path().join(WORLD_JSONL);
        std::fs::write(&world, "").unwrap();

        let found = find_world_from(tree.path()).unwrap();
        assert_eq!(found, world);
    }

    // Running from inside a project's subdirectory still finds the project's
    // world, and the nearest one wins over any further up.
    #[test]
    fn the_nearest_world_above_the_directory_is_found() {
        let tree = concinnity_testing::TempTree::new();
        let project = tree.path().join("project");
        let nested = project.join("assets").join("ground");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(tree.path().join(WORLD_JSONL), "").unwrap();
        std::fs::write(project.join(WORLD_JSONL), "").unwrap();

        let found = find_world_from(&nested).unwrap();
        assert_eq!(found, project.join(WORLD_JSONL));
    }

    // Only `world.jsonl` is discovered: any other world file is named with
    // `-f`, so a lookup never picks one of several by guesswork.
    #[test]
    fn another_world_file_is_not_discovered() {
        let tree = concinnity_testing::TempTree::new();
        std::fs::write(tree.path().join("arena.jsonl"), "").unwrap();

        let err = find_world_from(tree.path()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(err.to_string().contains(WORLD_JSONL), "message was: {err}");
    }
}
