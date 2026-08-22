//! Derives the schema hash baked into every blob header: an FNV-1a hash over
//! the postcard-visible schema sources -- the asset schema crate, the divergent
//! runtime asset structs, the component registry (list order is the tag), and
//! the blob record shapes. Any change to those sources changes the hash, so a
//! stale blob fails the load check instead of mis-decoding, with no manually
//! maintained version. Over-sensitivity (a comment edit invalidates blobs) is
//! deliberate: it can only force a rebuild, never a mis-decode.

use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let hash = concinnity_toolchain::hash_sources(&[
        manifest.join("../concinnity-asset/src"),
        manifest.join("../concinnity-core/src/assets"),
        manifest.join("../concinnity-core/src/ecs/registry.rs"),
        manifest.join("src/schema.rs"),
    ]);

    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("schema_hash.rs");
    std::fs::write(
        &out,
        format!(
            "/// Hash of the postcard-visible schema sources this build was compiled\n\
             /// against, derived by build.rs and stamped into every blob header. A blob\n\
             /// whose stored hash differs was written by a different engine schema and\n\
             /// fails the load check instead of mis-decoding.\n\
             pub const SCHEMA_HASH: u32 = {hash:#010x};\n"
        ),
    )
    .expect("write schema_hash.rs");
}
