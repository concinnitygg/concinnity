//! Derives two things from this crate's own sources, both of which a registry
//! checkout has to be able to produce with no sibling crate to read.
//!
//! The first is the hash of the postcard-visible schema. It is one of the three
//! parts concinnity-core mixes into the `SCHEMA_HASH` stamped in every blob
//! header, and the only part this crate owns.
//!
//! The second is `ASSET_DOCS`: the rustdoc, serde keys, and `Default` literals
//! of the authored schema, none of which survive compilation. concinnity-docs
//! assembles the asset reference from this table and the matching one
//! concinnity-core emits for the runtime half.

use concinnity_toolchain::doc_extract::{self, TableSpec};
use concinnity_toolchain::hash_sources;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let src = manifest.join("src");

    // hash_sources emits the rerun directive the extraction below relies on too.
    let hash = hash_sources(std::slice::from_ref(&src));
    std::fs::write(
        out.join("source_hash.rs"),
        format!(
            "/// Hash of the authored asset schema, mixed into\n\
             /// `concinnity_core::SCHEMA_HASH`.\n\
             pub const SOURCE_HASH: u32 = {hash:#010x};\n"
        ),
    )
    .expect("write source_hash.rs");

    // `doc_model` is the vocabulary the table is written in, not schema, so it
    // stays out of the table.
    let types = doc_extract::extract(std::slice::from_ref(&src), &[src.join("doc_model.rs")])
        .unwrap_or_else(|e| panic!("concinnity-asset: {e}"));
    std::fs::write(
        out.join("asset_docs.rs"),
        doc_extract::emit_table(
            &types,
            &TableSpec {
                const_name: "ASSET_DOCS",
                doc: "Every type declared by this crate's schema sources, sorted by name.\n\
                      \n\
                      The authored half of the asset reference: what a world.jsonl declares.\n\
                      The runtime half is `concinnity_core::RUNTIME_ASSET_DOCS`.",
                model_path: "doc_model",
            },
        ),
    )
    .expect("write asset_docs.rs");
}
