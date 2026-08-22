//! Standalone host for the Amazon Lumberyard Bistro showcase. On first run it
//! fetches the Bistro asset pack into examples/bistro/assets/ (~833 MB), then
//! compiles the world in `world.rs` in memory and plays it through the runtime
//! renderer. Subsequent runs find the assets already present and skip the fetch.
//!
//! The fetch is a runtime preflight, not a build step: `cargo build` never
//! touches the network, and the download happens once, the first time someone
//! runs the example.
//!
//! The renderer is heavy here (2.8M triangles, ray-traced reflections, SSGI).
//! Run in release for full frame rate:
//! `cargo run --example bistro --release --features cook`.

use std::io;
use std::path::{Path, PathBuf};

use concinnity::App;
use concinnity::cook;
use concinnity_toolchain::fetch::{self, ZipFetch};

mod world;

concinnity::install_global_allocator!();

// The NVIDIA ORCA download. It redirects to the actual archive; ureq follows
// redirects. Override with BISTRO_URL, or point BISTRO_ARCHIVE at an
// already-downloaded archive to skip the download entirely.
const BISTRO_URL: &str = "https://developer.nvidia.com/bistro";

/// Files that must exist for the pack to count as present. Both ship inside the
/// same archive, and `world.rs` names the same paths as its asset sources, so
/// the content and this preflight cannot drift apart.
pub const FBX_REL: &str = "assets/Bistro_v5_2/BistroExterior.fbx";
/// Path to the pack's HDR environment map, relative to the world.
pub const HDR_REL: &str = "assets/Bistro_v5_2/san_giuseppe_bridge_4k.hdr";

fn main() -> io::Result<()> {
    // fbxcel logs a benign WARN about an "extra node end marker" near the end of
    // BistroExterior.fbx -- a known quirk of that file's binary node
    // terminators. The parse recovers and the whole scene imports, so silence
    // just that crate while leaving every other log at its normal level. The
    // runtime's default filter is info (debug) / warn (release); mirror it and
    // append the fbxcel directive, but only when the user hasn't set RUST_LOG.
    if std::env::var_os("RUST_LOG").is_none() {
        let base = if cfg!(debug_assertions) {
            "info"
        } else {
            "warn"
        };
        // SAFETY: single-threaded startup -- this runs before logging init
        // and before any thread that could read the environment is spawned.
        unsafe { std::env::set_var("RUST_LOG", format!("{base},fbxcel=error")) };
    }

    // Resolve every relative asset path in the world against this example's own
    // directory rather than wherever cargo was invoked from. The `.concinnity/`
    // state tree (payload cache, runtime config) lands here too, beside the
    // example's own content.
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/bistro");
    std::env::set_current_dir(&dir)
        .map_err(|e| io::Error::new(e.kind(), format!("could not enter {}: {e}", dir.display())))?;

    fetch::ensure(&ZipFetch {
        url: &std::env::var("BISTRO_URL").unwrap_or_else(|_| BISTRO_URL.to_string()),
        extract_to: Path::new("assets"),
        root: Path::new("."),
        sentinels: &[FBX_REL, HDR_REL],
        local_archive: std::env::var_os("BISTRO_ARCHIVE").map(PathBuf::from),
    })?;

    if cfg!(debug_assertions) {
        eprintln!(
            "note: debug build -- run `cargo run --example bistro --release` for full frame rate"
        );
    }

    let mut spec = cook::world();
    world::declare(&mut spec);
    App::from_world(spec.compile()?).run()
}

#[cfg(test)]
mod tests {
    use super::*;

    // As an example target of the root package, CARGO_MANIFEST_DIR is the repo
    // root rather than this directory, so the asset paths only resolve because
    // of the join below. Pin it: a wrong root sends the fetch and every
    // relative asset path somewhere silently empty.
    #[test]
    fn the_example_directory_resolves_from_the_manifest_root() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/bistro");
        assert!(dir.join("world.rs").is_file(), "{} is wrong", dir.display());
    }
}
