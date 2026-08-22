// src/cli/rm.rs: discovery wrapper around concinnity_editor::rm_at_path

use concinnity_cook::world::find_world_jsonl;
use concinnity_editor::rm_at_path;

pub(crate) fn rm(name: &str) -> std::io::Result<()> {
    let world_path = find_world_jsonl(None)?;
    rm_at_path(&world_path, name)
}
