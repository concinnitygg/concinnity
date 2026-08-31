// src/cli/new.rs

use concinnity_cook::authoring::world::WORLD_JSONL;
use concinnity_cook::build_from_path;

// Default starter world file. Everything else a running world needs (window,
// renderer, debug HUD) is injected at build time and recorded in
// world-lock.json; `cn list --expanded` shows the effective world.
//
// The label names no Font, so it draws with the engine's built-in face. It asks
// for `centered` itself rather than leaning on a default: unset, the greeting
// lands at the label's default x/y, under the HUD chips in the top-left corner.
const INIT_WORLD_JSONL: &str = r#"{"name":"hello_world","type":"TextLabel","args":{"content":"Hello, world!","centered":true}}
"#;

/// Create a new project in a new directory at `path`.
pub fn new(path: &str) -> std::io::Result<()> {
    if std::path::Path::new(path).exists() {
        // allow creating a project in a pre-existing empty directory,
        // but refuse if it already has a world.jsonl
        let world = std::path::Path::new(path).join(WORLD_JSONL);
        if world.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("'{}' already contains a {}", path, WORLD_JSONL),
            ));
        }
    }
    std::fs::create_dir_all(path)?;
    println!("Created directory '{}'", path);
    init_in_dir(path)
}

/// Create a new project in the working directory.
pub fn init() -> std::io::Result<()> {
    init_in_dir(".")
}

// Write the starter world.jsonl into `dir` and run an initial build
fn init_in_dir(dir: &str) -> std::io::Result<()> {
    let world_path = std::path::Path::new(dir).join(WORLD_JSONL);

    if world_path.exists() {
        println!("{} already exists, skipping init", world_path.display());
        return Ok(());
    }

    std::fs::write(&world_path, INIT_WORLD_JSONL)?;
    println!("Created {}", world_path.display());

    let world_path_str = world_path.to_str().unwrap_or(WORLD_JSONL);
    build_from_path(
        &crate::project::require()?,
        world_path_str,
        crate::cook_platform(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Only the paths that stop before the initial build are exercised here;
    // a successful `cn new` runs the full compile pipeline.

    #[test]
    fn new_refuses_a_directory_that_already_has_a_world() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(WORLD_JSONL), "").unwrap();

        let err = new(dir.path().to_str().unwrap()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        assert!(err.to_string().contains(WORLD_JSONL), "got: {err}");
    }

    #[test]
    fn init_in_dir_skips_when_a_world_exists() {
        let dir = tempfile::tempdir().unwrap();
        let world = dir.path().join(WORLD_JSONL);
        std::fs::write(
            &world,
            "{\"name\":\"keep\",\"type\":\"Logger\",\"args\":{}}\n",
        )
        .unwrap();

        init_in_dir(dir.path().to_str().unwrap()).unwrap();
        // The existing world is untouched, not overwritten by the starter.
        let content = std::fs::read_to_string(&world).unwrap();
        assert!(content.contains("\"keep\""), "got: {content}");
    }
}
