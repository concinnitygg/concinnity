// build.rs
//
// The device-backend crate (rlib). Two jobs, both delegated to the shared
// concinnity-toolchain build helper:
//
// 1. Resolve the rendering backend once and expose it as a single cfg the crate
//    gates on (backend_metal / backend_dx / backend_vk).
//
// 2. Detect the optional upscaler SDKs and emit the cfgs the backends gate on.
//    This crate produces only an rlib (consumed by the client) plus its own test
//    binaries, so it does NOT bundle runtime DLLs next to a binary (that belongs
//    to whichever package owns the final artifact): SdkOptions { bundle_dlls: false }.

use concinnity_toolchain::{
    Backend, SdkOptions, emit_backend_cfg, emit_check_cfgs, precompile_metal_shaders,
    setup_graphics_sdks,
};

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
}
