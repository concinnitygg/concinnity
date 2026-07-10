// src/cli/rm.rs: discovery wrapper around crate::rm_at_path

use crate::rm_at_path;
use crate::world::find_world_jsonl;

pub fn rm(name: &str) -> std::io::Result<()> {
    let world_path = find_world_jsonl(None)?;
    rm_at_path(&world_path, name)
}
