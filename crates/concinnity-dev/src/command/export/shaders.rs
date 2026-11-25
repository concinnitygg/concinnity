//! The engine shaders an export ships, compiled ahead of time for the player's
//! graphics APIs so the exported app runs with no shader compiler.

use std::fs;
use std::io;
use std::path::Path;

use concinnity_cook::paths::StateTree;
use concinnity_cook::platform::{Platform, PlatformSet};

/// Compile every engine program for `platforms`, reporting how many artifacts
/// the file holds.
pub(super) fn compile_shipped_shaders(platforms: PlatformSet) -> io::Result<Vec<u8>> {
    let platforms: Vec<Platform> = platforms.iter().collect();
    println!(
        "Compiling engine shaders for {}...",
        platforms
            .iter()
            .map(|p| p.key())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let (bytes, count) = concinnity_runtime::precompile_shipped_shaders(&platforms)
        .map_err(|e| io::Error::other(format!("could not compile the engine shaders: {e}")))?;
    println!("Compiled {count} shader artifacts");
    Ok(bytes)
}

/// Write the compiled shaders into the bundle whose state tree is rooted at
/// `root`.
pub(super) fn write_shipped_shaders(shaders: &[u8], root: &Path) -> io::Result<()> {
    let path = StateTree::at(root).shipped_shaders_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, shaders)
}
