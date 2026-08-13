// build.rs
//
// The device-backend crate (rlib). Three jobs, all delegated to the shared
// concinnity-toolchain build helper:
//
// 1. Resolve the rendering backend once and expose it as a single cfg the crate
//    gates on (backend_metal / backend_dx / backend_vk).
//
// 2. Detect the optional upscaler SDKs and emit the cfgs the backends gate on.
//    This crate produces only an rlib (consumed by the client) plus its own test
//    binaries, so it does NOT bundle runtime DLLs next to a binary (that belongs
//    to whichever package owns the final artifact): SdkOptions { bundle_dlls: false }.
//
// 3. Derive the hash of the shader-compile sources that `shader_cache` folds
//    into every artifact key (see `emit_shader_compile_source_hash`).

use concinnity_toolchain::{
    Backend, SdkOptions, emit_backend_cfg, emit_check_cfgs, hash_sources, precompile_metal_shaders,
    setup_graphics_sdks,
};
use std::path::PathBuf;

// The raymarch SDF fragments are not standalone shaders: they are text
// templates assembled with the user's SdfVolume source at runtime (see
// src/metal/raymarch.rs), so they can only ever compile from source.
const SOURCE_ONLY_METAL_SHADERS: &[&str] = &[
    "raymarch_helpers.metal",
    "raymarch_shadow.metal",
    "raymarch_template.metal",
    "raymarch_volumetric_template.metal",
];

// Shared declarations spliced into the shaders that carry the marker, matching
// what `metal::pipeline::shader_source` substitutes when the same shader
// compiles from source. The `.msl` extension keeps a fragment out of the
// `.metal` precompile scan: it is not a standalone library.
const METAL_SHADER_FRAGMENTS: &[(&str, &str)] = &[("{OBJECT_DATA}", "object_common.msl")];

// The modules that decide how a shader artifact is produced: the cache itself
// (the key layout, and what an entry stores) and each backend's compiler
// invocation. A cache key already covers the assembled shader source, the entry
// point, the target and the caller's option word, so what it cannot see is a
// change to the invocation around them -- a different optimisation level, an
// added flag, a reworked entry format. Hashing these sources in closes that
// gap, so such a change misses instead of loading bytes the old invocation
// produced. Every backend's module participates on every build, which keeps the
// hash independent of the resolved backend; the key's `compiler` field is what
// keeps one toolchain's artifacts away from another's.
//
// A host toolchain upgrade (a new Xcode, a new Windows SDK) changes no source
// here and so is not covered: deleting the cache directory remains the way to
// force a full recompile.
const SHADER_COMPILE_SOURCES: &[&str] = &[
    "src/shader_cache.rs",
    "src/directx/dxc.rs",
    "src/directx/pipeline.rs",
    "src/metal/msl_cache.rs",
    "src/vulkan/pipeline.rs",
];

fn main() {
    emit_check_cfgs();
    let backend = emit_backend_cfg();
    setup_graphics_sdks(backend, SdkOptions { bundle_dlls: false });
    if backend == Backend::Metal {
        let shaders_dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/metal/shaders");
        precompile_metal_shaders(
            &shaders_dir,
            SOURCE_ONLY_METAL_SHADERS,
            METAL_SHADER_FRAGMENTS,
        );
    }
    emit_shader_compile_source_hash();
}

fn emit_shader_compile_source_hash() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let roots: Vec<PathBuf> = SHADER_COMPILE_SOURCES
        .iter()
        .map(|p| manifest.join(p))
        .collect();
    let hash = hash_sources(&roots);

    let out =
        PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("shader_compile_source_hash.rs");
    std::fs::write(
        &out,
        format!("const SHADER_COMPILE_SOURCE_HASH: u32 = {hash:#010x};\n"),
    )
    .expect("write shader_compile_source_hash.rs");
}
