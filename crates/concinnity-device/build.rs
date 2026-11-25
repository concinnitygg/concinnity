//! The device-backend crate (rlib).
//!
//! 1. Resolve the rendering backends through the shared concinnity-toolchain
//!    helper's `setup_graphics_backends`, exposing one cfg per backend the crate
//!    gates on (`backend_metal` / `backend_dx` / `backend_vk`).
//!
//! 2. Derive the hash of the shader-compile sources that `shader_cache` folds
//!    into every artifact key (see `emit_shader_compile_source_hash`).
//!
//! No shader compiles here: the engine programs compile on first use into the
//! runtime cache, and an export compiles them ahead of time for the player.

use concinnity_toolchain::{hash_sources, setup_graphics_backends};
use std::path::PathBuf;

// The modules that decide how a shader artifact is produced: the cache itself
// (the key layout, and what an entry stores), each backend's compiler
// invocation, and the hand-written Metal kernel no engine shader hash covers. A
// cache key already covers the assembled shader source, the entry point, the
// target and the caller's option word, so what it cannot see is a change to the
// invocation around them -- a different optimization level, an added flag, a
// reworked entry format. Hashing these sources in closes that gap, so such a
// change misses instead of loading bytes the old invocation produced. Every
// backend's module participates on every build, which keeps the hash
// independent of the resolved backend; the key's `compiler` field is what keeps
// one toolchain's artifacts away from another's.
//
// A compiler upgrade changes no source here; the dxc and Metal toolchain ids in
// the cache stamp and keys cover it instead.
//
// The dxc invocation lives in another crate, so it arrives as
// `concinnity_shader::SOURCE_HASH` (folded in by `shader_cache`) rather than being
// read out of that crate's directory: a registry checkout of this crate has no
// sibling copy to read.
const SHADER_COMPILE_SOURCES: &[&str] = &[
    "src/shader/cache.rs",
    "src/shader/runtime_cache.rs",
    "src/shader/source.rs",
    "src/shader/compile.rs",
    "src/shader/stamp.rs",
    "src/shader/builtin.rs",
    "src/directx/builtin_shaders.rs",
    "src/metal/msl_cache.rs",
    "src/metal/builtin_shaders.rs",
    "src/metal/shaders/cull_encode.metal",
    "src/vulkan/builtin_shaders.rs",
];

fn main() {
    setup_graphics_backends();
    emit_shader_compile_source_hash();
}

#[expect(
    clippy::disallowed_methods,
    reason = "cargo passes build-script inputs through the environment"
)]
fn emit_shader_compile_source_hash() {
    let manifest =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let roots: Vec<PathBuf> = SHADER_COMPILE_SOURCES
        .iter()
        .map(|p| manifest.join(p))
        .collect();
    let hash = hash_sources(&roots);

    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"))
        .join("shader_compile_source_hash.rs");
    std::fs::write(
        &out,
        format!(
            "pub(crate) const SHADER_COMPILE_SOURCE_HASH: u32 = 0x{:04x}_{:04x};\n",
            hash >> 16,
            hash & 0xffff
        ),
    )
    .expect("write shader_compile_source_hash.rs");
}
