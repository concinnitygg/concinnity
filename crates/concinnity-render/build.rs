//! Generates the linearly-transformed-cosine lookup table the rectangular
//! area-light shading path samples, by running the fitter in `src/ltc/fit.rs`.
//!
//! The table is pure offline data: it depends only on the fitter, never on the
//! world being built, so generating it here means zero runtime cost and zero
//! per-world build cost. Cargo re-runs this only when the fitter changes.
//!
//! The tables are written as raw little-endian f32, not as Rust source: a static
//! array of 24k float literals takes rustc tens of seconds to compile, while
//! `include_bytes!` of the same data is free.
#![allow(dead_code)]

include!("src/ltc/fit.rs");

fn write_f32s(path: std::path::PathBuf, values: impl Iterator<Item = f32>) {
    let mut bytes = Vec::new();
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).expect("failed to write the generated LTC table");
}

fn main() {
    println!("cargo::rerun-if-changed=src/ltc/fit.rs");

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
