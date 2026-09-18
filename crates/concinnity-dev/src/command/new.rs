use concinnity_cook::authoring::world::WORLD_JSONL;
use concinnity_host::store::paths::StateTree;
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
pub fn new(path: &str) -> std::io::Result<()> {
    // A pre-existing empty directory is a valid target; one that already holds
    // a world is not.
    if let Some(world) = existing_world(Path::new(path)) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("'{}' already contains a {}", path, world.display()),
        ));
    }
    std::fs::create_dir_all(path)?;
    println!("Created directory '{}'", path);
    init_in_dir(path)
}

/// Create a new project in the working directory.
pub fn init() -> std::io::Result<()> {
    init_in_dir(".")
}

// Write the starter world into `dir/worlds/` and run an initial build
fn init_in_dir(dir: &str) -> std::io::Result<()> {
    let dir = Path::new(dir);
    if let Some(world) = existing_world(dir) {
        println!("{} already exists, skipping init", world.display());
        return Ok(());
    }

    let world_path = worlds_dir(dir).join(WORLD_JSONL);
    std::fs::create_dir_all(world_path.parent().expect("the world has a directory"))?;
    std::fs::write(&world_path, INIT_WORLD_JSONL)?;
    println!("Created {}", world_path.display());

    let world_path_str = world_path.to_str().unwrap_or(WORLD_JSONL);
    crate::authoring::build_world_file(world_path_str)
}

// The world already scaffolded in `dir`, if any: the one a new project writes,
// or the `world.jsonl` at the project root.
fn existing_world(dir: &Path) -> Option<PathBuf> {
    [worlds_dir(dir).join(WORLD_JSONL), dir.join(WORLD_JSONL)]
        .into_iter()
        .find(|p| p.exists())
}

// Where a project rooted at `dir` keeps its authored worlds.
fn worlds_dir(dir: &Path) -> PathBuf {
    StateTree::at(dir).worlds_dir()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Only the paths that stop before the initial build are exercised here;
    // a successful `cn new` runs the full compile pipeline.

    #[test]
    fn new_refuses_a_directory_that_already_has_a_world() {
        let dir = tempfile::tempdir().unwrap();
        let world = worlds_dir(dir.path()).join(WORLD_JSONL);
        std::fs::create_dir_all(world.parent().unwrap()).unwrap();
        std::fs::write(&world, "").unwrap();

        let err = new(dir.path().to_str().unwrap()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        assert!(err.to_string().contains(WORLD_JSONL), "got: {err}");
    }

    // A root `world.jsonl` counts too, so `cn new` refuses rather than
    // scaffolding a second world.
    #[test]
    fn new_refuses_a_directory_holding_only_a_root_world() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(WORLD_JSONL), "").unwrap();

        let err = new(dir.path().to_str().unwrap()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
    }

    #[test]
    fn init_in_dir_skips_when_a_world_exists() {
        let dir = tempfile::tempdir().unwrap();
        let world = worlds_dir(dir.path()).join(WORLD_JSONL);
        std::fs::create_dir_all(world.parent().unwrap()).unwrap();
        std::fs::write(&world, "[\"Logger\",{\"$id\":\"keep\"}]\n").unwrap();

        init_in_dir(dir.path().to_str().unwrap()).unwrap();
        // The existing world is untouched, not overwritten by the starter.
        let content = std::fs::read_to_string(&world).unwrap();
        assert!(content.contains("\"keep\""), "got: {content}");
    }

    #[test]
    fn init_in_dir_skips_a_root_world() {
        let dir = tempfile::tempdir().unwrap();
        let world = dir.path().join(WORLD_JSONL);
        std::fs::write(&world, "[\"Logger\",{\"$id\":\"keep\"}]\n").unwrap();

        init_in_dir(dir.path().to_str().unwrap()).unwrap();
        assert!(
            !worlds_dir(dir.path()).exists(),
            "a root world is left in place rather than duplicated into worlds/"
        );
    }
}
