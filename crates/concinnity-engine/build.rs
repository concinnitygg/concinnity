//! Generates the linearly-transformed-cosine lookup table the rectangular
//! area-light shading path samples, by running the fitter in
//! `src/render/light/ltc/fit.rs`. The table is pure offline data: it depends only on
//! the fitter, never on the world being built, so generating it here means zero
//! runtime cost and zero per-world build cost.
//!
//! The tables are written as raw little-endian f32, not as Rust source: a static
//! array of 24k float literals takes rustc tens of seconds to compile, while
//! `include_bytes!` of the same data is free.

// The fitter is written for the `no_std` lib that also compiles it, so it names
// `alloc` for its collections.
extern crate alloc;

include!("src/render/light/ltc/size.rs");
include!("src/render/light/ltc/fit.rs");

fn main() {
    generate_ltc_tables();
}

fn write_f32s(path: std::path::PathBuf, values: impl Iterator<Item = f32>) {
    let mut bytes = Vec::new();
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).expect("failed to write the generated LTC table");
}

#[expect(
    clippy::disallowed_methods,
    reason = "cargo passes build-script inputs through the environment"
)]
fn generate_ltc_tables() {
    println!("cargo::rerun-if-changed=src/render/light/ltc/fit.rs");
    println!("cargo::rerun-if-changed=src/render/light/ltc/size.rs");

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
