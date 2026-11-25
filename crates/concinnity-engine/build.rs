//! Generates the linearly-transformed-cosine lookup table the rectangular
//! area-light shading path samples, by running the fitter in
//! `src/render/light/ltc/fit.rs`. The table is pure offline data: it depends only on
//! the fitter, never on the world being built, so generating it here means zero
//! runtime cost and zero per-world build cost.
//!
//! The tables are written as raw little-endian f32, not as Rust source: a static
//! array of 24k float literals takes rustc tens of seconds to compile, while
//! `include_bytes!` of the same data is free.
//!
//! Also hashes the single-source shaders under `src/render/shader/hlsl/`, which
//! names the engine shader set an exported app's shipped artifacts were
//! compiled from.

// The fitter is written for the `no_std` lib that also compiles it, so it names
// `alloc` for its collections.
extern crate alloc;

include!("src/render/light/ltc/size.rs");
include!("src/render/light/ltc/fit.rs");

fn main() {
    generate_ltc_tables();
    emit_shader_sources_hash();
}

// FNV-1a over each shader's name and text, in name order, each length-prefixed.
// `embedded::tests` recomputes it from `SOURCES` with the same fold.
#[expect(
    clippy::disallowed_methods,
    reason = "cargo passes build-script inputs through the environment"
)]
fn emit_shader_sources_hash() {
    const DIR: &str = "src/render/shader/hlsl";
    println!("cargo::rerun-if-changed={DIR}");
    let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(DIR)
        .expect("the shader directory is readable")
        .map(|entry| {
            let path = entry.expect("a shader directory entry").path();
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .expect("a UTF-8 shader file name")
                .to_string();
            let text = std::fs::read(&path).expect("a readable shader");
            (name, text)
        })
        .filter(|(name, _)| {
            std::path::Path::new(name)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("hlsl"))
        })
        .collect();
    files.sort();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut write = |bytes: &[u8]| {
        for &b in bytes {
            hash = (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for (name, text) in &files {
        write(&(name.len() as u64).to_le_bytes());
        write(name.as_bytes());
        write(&(text.len() as u64).to_le_bytes());
        write(text);
    }
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is set"))
        .join("shader_sources_hash.rs");
    std::fs::write(
        out,
        format!(
            "/// Hash of every embedded shader's name and text.\npub const SOURCES_HASH: u64 = \
             0x{:04x}_{:04x}_{:04x}_{:04x};\n",
            hash >> 48,
            (hash >> 32) & 0xffff,
            (hash >> 16) & 0xffff,
            hash & 0xffff
        ),
    )
    .expect("write shader_sources_hash.rs");
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
