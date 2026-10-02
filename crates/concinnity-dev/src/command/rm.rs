//! Discovery wrapper around `crate::authoring::rm_at_path`.

use crate::authoring::rm_at_path;
use crate::command::discover_world_path;

/// Delete the asset named `name` from the discovered world.
pub fn rm(name: &str) -> std::io::Result<()> {
    let world_path = discover_world_path()?;
    rm_at_path(&world_path, name)
}
