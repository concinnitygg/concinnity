//! Derives the hash of this crate's compile pipeline, folded into every payload
//! cache key. A cached payload is a function of the asset's args and source
//! files *and* of the code that compiled it; the args side is hashed into the
//! key directly, and this hash covers the compiler side, so a cook-logic change
//! misses instead of replaying stale bytes into a new blob. Over-sensitivity (a
//! comment edit invalidates the cache) is deliberate: it can only force a
//! recompile, never a stale replay.
//!
//! The other half of a payload's inputs lives in concinnity-core, which this
//! script cannot hash: a registry checkout has no sibling directory to read.
//! `cache.rs` folds in `concinnity_core::SCHEMA_VERSION` for it, which the
//! runtime bumps when a payload format or the asset schema changes.

use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let hash = concinnity_toolchain::hash_sources(&[manifest.join("src")]);
    std::fs::write(
        out.join("compile_source_hash.rs"),
        format!("const COMPILE_SOURCE_HASH: u32 = {hash:#010x};\n"),
    )
    .expect("write compile_source_hash.rs");
}
