//! Derives the hash of this crate's postcard-visible schema sources. It is one
//! of the three parts concinnity-core mixes into the `SCHEMA_HASH` stamped in
//! every blob header, and the only part this crate owns. Each crate hashes its
//! own directory so the mix comes out the same whether the build runs from the
//! workspace or from a registry checkout, where no sibling crate exists to read.

use concinnity_toolchain::hash_sources;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let hash = hash_sources(&[manifest.join("src")]);

    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR")).join("source_hash.rs");
    std::fs::write(
        &out,
        format!(
            "/// Hash of the authored asset schema, mixed into\n\
             /// `concinnity_core::SCHEMA_HASH`.\n\
             pub const SOURCE_HASH: u32 = {hash:#010x};\n"
        ),
    )
    .expect("write source_hash.rs");
}
