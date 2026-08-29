//! Two jobs, both pure functions of this crate's own source.
//!
//! Resolves the rendering backend cfg the same way the client crate's build.rs
//! does. `Platform::current` is the only consumer: it reports which shader
//! source language the target backend consumes.
//!
//!   backend_metal  macOS, default
//!   backend_dx     Windows, default
//!   backend_vk     Linux (always), or macOS / Windows with the `vulkan` feature
//! The choice must stay in lockstep with concinnity-engine/build.rs.
//!
//! Generates the linearly-transformed-cosine lookup table the rectangular
//! area-light shading path samples, by running the fitter in
//! `src/render/ltc/fit.rs`. The table is pure offline data: it depends only on
//! the fitter, never on the world being built, so generating it here means zero
//! runtime cost and zero per-world build cost.
//!
//! The tables are written as raw little-endian f32, not as Rust source: a static
//! array of 24k float literals takes rustc tens of seconds to compile, while
//! `include_bytes!` of the same data is free.

// The fitter is written for the `no_std` lib that also compiles it, so it names
// `alloc` for its collections.
extern crate alloc;

include!("src/render/ltc/size.rs");
include!("src/render/ltc/fit.rs");

fn main() {
    emit_backend_cfg();
    generate_ltc_tables();
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

fn write_f32s(path: std::path::PathBuf, values: impl Iterator<Item = f32>) {
    let mut bytes = Vec::new();
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).expect("failed to write the generated LTC table");
}

fn generate_ltc_tables() {
    println!("cargo::rerun-if-changed=src/render/ltc/fit.rs");
    println!("cargo::rerun-if-changed=src/render/ltc/size.rs");

    let (matrix, magnitude) = fit_table(LTC_LUT_SIZE);
    let dir = std::path::PathBuf::from(
        std::env::var("OUT_DIR").expect("OUT_DIR is set for build scripts"),
    );
    write_f32s(dir.join("ltc_matrix.bin"), matrix.into_iter().flatten());
    write_f32s(
        dir.join("ltc_magnitude.bin"),
        magnitude.into_iter().flatten(),
    );
}
