// src/cli/rm.rs: discovery wrapper around crate::rm_at_path

use crate::rm_at_path;
use concinnity_cook::world::find_world_jsonl;

/// Delete the asset named `name` from the discovered world.
pub fn rm(name: &str) -> std::io::Result<()> {
    let world_path = find_world_jsonl(None)?;
    rm_at_path(&world_path, name)
}
