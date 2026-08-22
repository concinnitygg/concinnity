//! Derives the hash of the compile sources folded into every payload cache key:
//! an FNV-1a over this crate's compile pipeline plus the payload format helpers
//! it shares with the runtime. A cached payload is a function of the asset's
//! args and source files *and* of the code that compiled it; the args side is
//! hashed into the key directly, and this hash covers the compiler side, so a
//! cook-logic change misses instead of replaying stale bytes into a new blob.
//! Over-sensitivity (a comment edit invalidates the cache) is deliberate: it can
//! only force a recompile, never a stale replay.
//!
//! The postcard-visible asset schema is the third thing a payload depends on. It
//! is already derived as `concinnity_blob::SCHEMA_HASH`, which `cache.rs` folds
//! in alongside this hash rather than duplicating that root list here.

use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let hash = concinnity_toolchain::hash_sources(&[
        manifest.join("src"),
        manifest.join("../concinnity-cpu/src/build"),
    ]);

    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("compile_source_hash.rs");
    std::fs::write(
        &out,
        format!("const COMPILE_SOURCE_HASH: u32 = {hash:#010x};\n"),
    )
    .expect("write compile_source_hash.rs");
}
