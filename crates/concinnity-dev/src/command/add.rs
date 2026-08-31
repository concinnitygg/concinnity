// src/cli/add.rs: discovery wrapper around crate::add_to_path
//
// The CLI binary handles world-path discovery: try the standard
// `.concinnity/worlds/` location first, then fall back to `world.jsonl` in
// cwd. When the fallback is hit and the target is a 3D scene (.glb),
// `add_to_path` scaffolds a fresh world at that location.

use crate::add_to_path;
use concinnity_cook::authoring::world::{WORLD_JSONL, find_world_jsonl};

/// Create an asset from `target` and apply it to the discovered world.
///
/// Tries `.concinnity/worlds/` first, then `world.jsonl` in the working
/// directory; when neither exists and the target is a 3D scene, the world is
/// scaffolded at the fallback location.
pub fn add(name: Option<&str>, target: &str, template: Option<&str>) -> std::io::Result<()> {
    let world_path = match find_world_jsonl(crate::project::worlds_dir().as_deref(), None) {
        Ok(p) => p,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => WORLD_JSONL.to_string(),
        Err(e) => return Err(e),
    };
    add_to_path(&world_path, name, target, template)
}
