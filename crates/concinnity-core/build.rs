//! Resolves the rendering backend cfg the same way the client crate's build.rs
//! does. `Platform::current` is the only consumer: it reports which shader
//! source language the target backend consumes.
//!
//!   backend_metal  macOS, default
//!   backend_dx     Windows, default
//!   backend_vk     Linux (always), or macOS / Windows with the `vulkan` feature
//! The choice must stay in lockstep with concinnity-engine/build.rs.
//!
//! Also derives this crate's half of the blob schema hash: the divergent runtime
//! asset structs and the component registry, whose list order is the tag. The
//! other two halves are hashed by the crates that own them
//! (`concinnity_asset::SOURCE_HASH`, `concinnity_blob::RECORD_SCHEMA_HASH`) and
//! mixed in by `lib.rs`, which is what lets the result come out the same from a
//! registry checkout where no sibling crate exists to read.
fn main() {
    emit_component_schema_hash();
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

// This crate's contribution to the blob schema hash. Over-sensitivity (a comment
// edit invalidates blobs) is deliberate: it can only force a rebuild, never a
// mis-decode.
fn emit_component_schema_hash() {
    let manifest =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let hash = concinnity_toolchain::hash_sources(&[
        manifest.join("src/assets"),
        manifest.join("src/ecs/registry.rs"),
    ]);

    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"))
        .join("component_schema_hash.rs");
    std::fs::write(
        &out,
        format!("const COMPONENT_SCHEMA_HASH: u32 = {hash:#010x};\n"),
    )
    .expect("write component_schema_hash.rs");
}
