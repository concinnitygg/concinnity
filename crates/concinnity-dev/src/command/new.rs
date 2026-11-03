use concinnity_cook::authoring::world::WORLD_JSONL;
use std::path::{Path, PathBuf};

// Default starter world file. Everything else a running world needs (window,
// renderer, debug HUD) is injected at build time and recorded in
// world-lock.json; `cn list --expanded` shows the effective world.
//
// The label names no Font, so it draws with the engine's built-in face. It asks
// for `centered` itself rather than leaning on a default: unset, the greeting
// lands at the label's default x/y, under the HUD chips in the top-left corner.
const INIT_WORLD_JSONL: &str = r#"["TextLabel",{"$id":"hello_world","content":"Hello, world!","centered":true}]
"#;

/// Create a new project in a new directory at `path`.
pub(crate) fn new(path: &str) -> std::io::Result<()> {
    // A pre-existing empty directory is a valid target; one that already holds
    // a world is not.
    let world = world_in(Path::new(path));
    if world.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("'{}' already contains a {}", path, world.display()),
        ));
    }
    std::fs::create_dir_all(path)?;
    println!("Created directory '{path}'");
    init_in_dir(path)
}

/// Create a new project in the working directory.
pub(crate) fn init() -> std::io::Result<()> {
    init_in_dir(".")
}

// Write the starter world into `dir` and run an initial build
fn init_in_dir(dir: &str) -> std::io::Result<()> {
    let world_path = world_in(Path::new(dir));
    if world_path.exists() {
        println!("{} already exists, skipping init", world_path.display());
        return Ok(());
    }

    std::fs::write(&world_path, INIT_WORLD_JSONL)?;
    println!("Created {}", world_path.display());

    let world_path_str = world_path.to_str().unwrap_or(WORLD_JSONL);
    crate::authoring::build_world_file(world_path_str)
}

// The world a project rooted at `dir` scaffolds and discovers.
fn world_in(dir: &Path) -> PathBuf {
    dir.join(WORLD_JSONL)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Only the paths that stop before the initial build are exercised here;
    // a successful `cn new` runs the full compile pipeline.

    #[test]
    fn new_refuses_a_directory_that_already_has_a_world() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(world_in(dir.path()), "").unwrap();

        let err = new(dir.path().to_str().unwrap()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        assert!(err.to_string().contains(WORLD_JSONL), "got: {err}");
    }

    #[test]
    fn init_in_dir_skips_when_a_world_exists() {
        let dir = tempfile::tempdir().unwrap();
        let world = world_in(dir.path());
        std::fs::write(&world, "[\"Logger\",{\"$id\":\"keep\"}]\n").unwrap();

        init_in_dir(dir.path().to_str().unwrap()).unwrap();
        // The existing world is untouched, not overwritten by the starter.
        let content = std::fs::read_to_string(&world).unwrap();
        assert!(content.contains("\"keep\""), "got: {content}");
    }
}
