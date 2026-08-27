//! Resolves the rendering backend cfg the same way the client crate's build.rs
//! does. `Platform::current` is the only consumer: it reports which shader
//! source language the target backend consumes.
//!
//!   backend_metal  macOS, default
//!   backend_dx     Windows, default
//!   backend_vk     Linux (always), or macOS / Windows with the `vulkan` feature
//! The choice must stay in lockstep with concinnity-engine/build.rs.
//!
//! Also derives two of the three halves of the blob schema hash: the component
//! definitions and the component registry, whose list order is the tag, and the
//! blob container's record shapes. Each is hashed over its own
//! directory so an edit outside it cannot invalidate blobs already on disk. The
//! third half is hashed by the crate that owns it (`concinnity_asset::SOURCE_HASH`)
//! and mixed in by `lib.rs`, which is what lets the result come out the same
//! from a registry checkout where no sibling crate exists to read.
//!
//! And two more derived constants. `RUNTIME_ASSET_DOCS`: the rustdoc, serde
//! keys, and `Default` literals of the `impl Component` half of the asset
//! schema, which concinnity-cook joins with `concinnity_asset::ASSET_DOCS` to
//! build the reference. And `BUILD_SOURCE_HASH`: the payload format helpers
//! concinnity-cook folds into every payload cache key. Same reason both are
//! emitted here rather than read from a sibling directory.

use std::path::{Path, PathBuf};

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));

    emit_component_schema_hash(&manifest, &out);
    emit_record_schema_hash(&manifest, &out);
    emit_build_source_hash(&manifest, &out);
    emit_runtime_asset_docs(&manifest, &out);
    emit_backend_cfg();
}

// This crate's contribution to the blob schema hash. Over-sensitivity (a comment
// edit invalidates blobs) is deliberate: it can only force a rebuild, never a
// mis-decode.
fn emit_component_schema_hash(manifest: &Path, out: &Path) {
    let hash = concinnity_toolchain::hash_sources(&[
        manifest.join("src/components"),
        manifest.join("src/ecs/registry.rs"),
    ]);

    std::fs::write(
        out.join("component_schema_hash.rs"),
        format!("const COMPONENT_SCHEMA_HASH: u32 = {hash:#010x};\n"),
    )
    .expect("write component_schema_hash.rs");
}

// The blob container's contribution. Scoped to the one file that defines what a
// header's records decode as, so an edit elsewhere in the crate cannot force
// every cooked blob on disk to be rewritten.
fn emit_record_schema_hash(manifest: &Path, out: &Path) {
    let hash = concinnity_toolchain::hash_sources(&[manifest.join("src/blob/schema.rs")]);

    std::fs::write(
        out.join("schema_hash.rs"),
        format!(
            "/// Hash of the blob record shapes, mixed into\n\
             /// `crate::SCHEMA_HASH`.\n\
             pub const RECORD_SCHEMA_HASH: u32 = {hash:#010x};\n"
        ),
    )
    .expect("write schema_hash.rs");
}

// The payload format helpers concinnity-cook folds into every payload cache key.
// A cached payload is a function of the code that produced it, and that code is
// split across the two crates, so the half living here is published as a
// constant rather than read out of this directory by the consumer's build
// script: a registry checkout of concinnity-cook has no sibling copy of these
// files. Hashed over its own directory so an edit outside it cannot invalidate
// payloads already in the cache.
fn emit_build_source_hash(manifest: &Path, out: &Path) {
    let hash = concinnity_toolchain::hash_sources(&[manifest.join("src/build")]);

    std::fs::write(
        out.join("build_source_hash.rs"),
        format!(
            "/// Hash of the payload format helpers in `build`, folded into the\n\
             /// cook's payload cache key so a change here misses instead of\n\
             /// replaying bytes an older version of this code produced.\n\
             pub const BUILD_SOURCE_HASH: u32 = {hash:#010x};\n"
        ),
    )
    .expect("write build_source_hash.rs");
}

// The runtime half of the asset reference as source text: rustdoc, serde keys,
// and `Default` literals of the components an authored world declares, none of
// which a compiled dependency can hand to a consumer. The hash above already
// emits the rerun directive for the same directory.
fn emit_runtime_asset_docs(manifest: &Path, out: &Path) {
    use concinnity_toolchain::doc_extract::{self, TableSpec};

    let types = doc_extract::extract(&[manifest.join("src/components")], &[])
        .unwrap_or_else(|e| panic!("concinnity-core: {e}"));

    std::fs::write(
        out.join("runtime_asset_docs.rs"),
        doc_extract::emit_table(
            &types,
            &TableSpec {
                const_name: "RUNTIME_ASSET_DOCS",
                doc: "Every type declared by this crate's asset sources, sorted by name.\n\
                      \n\
                      The runtime half of the asset reference: the `impl Component` blocks and\n\
                      the structs that diverge from what a world.jsonl declares. The authored\n\
                      half is `concinnity_asset::ASSET_DOCS`.",
                model_path: "concinnity_asset::doc_model",
            },
        ),
    )
    .expect("write runtime_asset_docs.rs");
}

fn emit_backend_cfg() {
    println!("cargo::rustc-check-cfg=cfg(backend_metal)");
    println!("cargo::rustc-check-cfg=cfg(backend_dx)");
    println!("cargo::rustc-check-cfg=cfg(backend_vk)");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let vulkan = std::env::var("CARGO_FEATURE_VULKAN").is_ok();

    let backend = match (target_os.as_str(), vulkan) {
        ("macos", false) => "backend_metal",
        ("windows", false) => "backend_dx",
        _ => "backend_vk",
    };
    println!("cargo::rustc-cfg={backend}");
}
