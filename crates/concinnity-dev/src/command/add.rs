//! Discovery wrapper around `crate::authoring::add_to_path`.
//!
//! The CLI binary handles world-path discovery: the nearest `world.jsonl` at or
//! above cwd. When none exists and the target is a 3D scene (.glb),
//! `add_to_path` scaffolds a fresh world in cwd.

use concinnity_cook::authoring::world::WORLD_JSONL;

use crate::authoring::add_to_path;
use crate::command::discover_world_path;

/// Create an asset from `target` and apply it to the discovered world.
///
/// Applies to the discovered `world.jsonl`; when there is none and the target is
/// a 3D scene, the world is scaffolded in the working directory.
pub(crate) fn add(name: Option<&str>, target: &str, template: Option<&str>) -> std::io::Result<()> {
    let world_path = match discover_world_path() {
        Ok(p) => p,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => WORLD_JSONL.to_string(),
        Err(e) => return Err(e),
    };
    add_to_path(&world_path, name, target, template)
}
