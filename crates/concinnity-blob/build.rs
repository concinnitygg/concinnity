//! Derives the hash of this crate's record shapes, one of the three parts
//! concinnity-core mixes into the `SCHEMA_HASH` stamped in every blob header.
//!
//! The container knows its own record layout and nothing about the asset types
//! travelling inside it, so the other two parts (the authored schema and the
//! runtime component definitions) are hashed by the crates that own them. Each
//! crate hashing its own directory is what lets the mix come out the same from
//! a registry checkout, where no sibling crate exists to read.

use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let hash = concinnity_toolchain::hash_sources(&[manifest.join("src/schema.rs")]);

    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR")).join("schema_hash.rs");
    std::fs::write(
        &out,
        format!(
            "/// Hash of the blob record shapes, mixed into\n\
             /// `concinnity_core::SCHEMA_HASH`.\n\
             pub const RECORD_SCHEMA_HASH: u32 = {hash:#010x};\n"
        ),
    )
    .expect("write schema_hash.rs");
}
