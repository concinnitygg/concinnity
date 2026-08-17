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

use concinnity_slang as slang;
use concinnity_toolchain::{
    Backend, SdkOptions, SlangLibSpec, emit_backend_cfg, emit_check_cfgs, hash_sources,
    precompile_metal_shaders, setup_graphics_sdks,
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

// The Metal bindless texture-pool capacity and reflection-probe array length,
// baked into the single-source shaders at build time. Must match
// `metal::context::BINDLESS_TEXTURE_COUNT` and `metal::uniforms::MAX_PROBES`;
// the generated `slang_metal_defines.rs` consts let a unit test lock them.
const SLANG_METAL_POOL_SIZE: usize = 1024;
const SLANG_METAL_MAX_PROBES: usize = 8;

const SLANG_MAIN_DEFINES: &[(&str, &str)] = &[
    ("METAL_ABI", "1"),
    ("POOL_SIZE", "1024"),
    ("MAX_PROBES", "8"),
];

// The DirectX counterpart, matching `directx/slang_builtins.rs::MAIN_DEFINES`.
// No POOL_SIZE: the DXIL pool is an unbounded array.
const SLANG_DXIL_MAIN_DEFINES: &[(&str, &str)] = &[("DXIL_ABI", "1"), ("MAX_PROBES", "8")];

// Every register the bindless main root signature in
// `src/directx/init/pipelines.rs` declares, as (parameter, HLSL register). The
// vertex stage reads only the first three; the rest are the fragment's.
const DXIL_ABI_REGISTERS: &[(&str, &str)] = &[
    ("objid_cb", "b0"),
    ("view_cb", "b1"),
    ("lights_cb", "b2"),
    ("shadow_cb", "b3"),
    ("probe_set_cb", "b4"),
    ("cluster_cb", "b5"),
    ("shadow_map", "t0"),
    ("local_lights_sb", "t1"),
    ("cluster_list_sb", "t2"),
    ("objects_sb", "t3"),
    ("ssao_tex", "t4"),
    ("irradiance_cube", "t5"),
    ("prefilter_cube", "t6"),
    ("probe_cubes", "t7"),
    ("spot_shadows_sb", "t15"),
    ("spot_shadow_map", "t16"),
    ("area_lights_sb", "t17"),
    ("ltc_matrix", "t18"),
    ("ltc_magnitude", "t19"),
    ("tex_pool", "t0, space1"),
    ("shadow_sampler", "s0"),
    ("linear_sampler", "s1"),
    ("cube_sampler", "s2"),
];

// The single-source engine shaders the Metal backend precompiles to metallibs.
// One spec per variant library; `name` is the renderer's lookup key.
const SLANG_METAL_LIBS: &[SlangLibSpec] = &[
    SlangLibSpec {
        name: "main_bindless_vert.slang",
        file: "main_bindless.slang",
        entries: &["vertex_main_bindless"],
        defines: SLANG_MAIN_DEFINES,
    },
    SlangLibSpec {
        name: "main_bindless_frag.slang",
        file: "main_bindless.slang",
        entries: &["fragment_main_bindless"],
        defines: SLANG_MAIN_DEFINES,
    },
    SlangLibSpec {
        name: "light_cull.slang",
        file: "light_cull.slang",
        entries: &["light_cull_kernel"],
        defines: &[],
    },
    SlangLibSpec {
        name: "hiz_init_msaa.slang",
        file: "hiz_build.slang",
        entries: &["hiz_init_msaa"],
        defines: &[("HIZ_INIT_MSAA", "1")],
    },
    SlangLibSpec {
        name: "hiz_downsample.slang",
        file: "hiz_build.slang",
        entries: &["hiz_downsample"],
        defines: &[("HIZ_DOWNSAMPLE", "1")],
    },
];

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
    "src/slang_source.rs",
    "src/directx/dxc.rs",
    "src/directx/pipeline.rs",
    "src/directx/slang_builtins.rs",
    "src/metal/msl_cache.rs",
    "src/metal/slang_shaders.rs",
    "src/vulkan/pipeline.rs",
    "src/vulkan/slang_builtins.rs",
    "../concinnity-slang/src/lib.rs",
];

fn main() {
    emit_check_cfgs();
    let backend = emit_backend_cfg();
    setup_graphics_sdks(backend, SdkOptions { bundle_dlls: false });
    if backend == Backend::Metal {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let shaders_dir = manifest.join("src/metal/shaders");
        let slang_dir = manifest.join("src/shaders");
        precompile_metal_shaders(
            &shaders_dir,
            SOURCE_ONLY_METAL_SHADERS,
            METAL_SHADER_FRAGMENTS,
            &slang_dir,
            SLANG_METAL_LIBS,
        );
        assert_slang_metal_abi(&slang_dir);
        emit_slang_metal_defines();
    }
    if backend == Backend::Dx {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        assert_slang_dxil_abi(&manifest.join("src/shaders"));
    }
    emit_shader_compile_source_hash();
}

// The Metal main-pass binding layout is a stable contract: world-authored
// Shader assets hand-write MSL against it. The `.slang` source pins the loose
// buffers with register() numbers, but the two parameter blocks (the texture
// argument buffer and the sampler block) land on first-free slot assignment,
// which is compiler behaviour rather than an annotation. Assert the emitted
// MSL here so a slangc upgrade that moves a slot fails the build instead of
// binding garbage at draw time. Skipped when slangc is absent (the runtime
// compile path reports its own error then).
fn assert_slang_metal_abi(slang_dir: &std::path::Path) {
    if slang::slangc_path().is_none() {
        return;
    }
    let source = std::fs::read_to_string(slang_dir.join("main_bindless.slang"))
        .expect("read main_bindless.slang");
    let job = slang::SlangJob {
        source: &slang::inject_defines(&source, SLANG_MAIN_DEFINES),
        file_name: "main_bindless_abi_check.slang",
        entries: &["fragment_main_bindless"],
        target: slang::SlangTarget::Metal,
    };
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let msl = slang::compile(&job, &out_dir).expect("slang ABI check compile");
    let msl = String::from_utf8_lossy(&msl);
    for (param, index) in [
        ("view_cb", 0),
        ("lights_cb", 4),
        ("shadow_cb", 5),
        ("probe_set_cb", 6),
        ("tex", 7),
        ("local_lights_sb", 8),
        ("objects_sb", 9),
        ("samps", 10),
        ("cluster_cb", 11),
        ("cluster_list_sb", 12),
        ("spot_shadows_sb", 13),
        ("area_lights_sb", 14),
    ] {
        // Emitted parameter names carry a `_<n>` suffix (e.g. `tex_1`).
        let re = format!("{param}_1 [[buffer({index})]]");
        assert!(
            msl.contains(&re),
            "main_bindless.slang Metal ABI drifted: expected `{re}` in the emitted MSL. \
             The Metal binding layout is frozen (world shaders hand-write against it); \
             fix the .slang declarations or the slangc slot assignment before shipping."
        );
    }
}

// The DirectX bindless main-pass binding layout is a stable contract for the
// same reason Metal's is: a world Shader asset builds its own PSO against the
// bindless root signature (see directx/world_shaders.rs), so a register that
// moves silently misbinds every world shader. The `.slang` source pins each one
// with register(), but a Slang release that mis-lowered an annotation would
// bind garbage at draw time with no compile-time signal. Emit HLSL and assert
// the annotations survive. Skipped when slangc is absent (the runtime compile
// path reports its own error then).
fn assert_slang_dxil_abi(slang_dir: &std::path::Path) {
    // Re-run on any single-source edit: nothing else in this script reads the
    // `.slang` files on a DirectX build, so without this the assert would go
    // stale the moment a binding moved.
    println!("cargo:rerun-if-changed={}", slang_dir.display());
    if slang::slangc_path().is_none() {
        return;
    }
    let source = std::fs::read_to_string(slang_dir.join("main_bindless.slang"))
        .expect("read main_bindless.slang");
    let source = slang::inject_defines(&source, SLANG_DXIL_MAIN_DEFINES);
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    // The two stages compile separately and each drops the resources it does
    // not read, so a register only has to hold in the stage that declares it.
    let mut emitted = String::new();
    for (entry, profile) in [
        ("vertex_main_bindless", "vs_6_0"),
        ("fragment_main_bindless", "ps_6_0"),
    ] {
        let job = slang::SlangJob {
            source: &source,
            file_name: "main_bindless_abi_check.slang",
            entries: &[entry],
            target: slang::SlangTarget::Hlsl(profile),
        };
        let hlsl = slang::compile(&job, &out_dir)
            .unwrap_or_else(|e| panic!("slang DXIL ABI check ({entry}): {e}"));
        emitted.push_str(&String::from_utf8_lossy(&hlsl));
    }
    for (param, register) in DXIL_ABI_REGISTERS {
        // Emitted parameter names carry a `_<n>` suffix (e.g. `view_cb_0`).
        let expected = format!("{param}_0 : register({register})");
        // Resource arrays emit as `name_0[int(N)] : register(...)`.
        let expected_array = format!("{param}_0[");
        assert!(
            emitted.contains(&expected)
                || emitted
                    .lines()
                    .any(|l| l.contains(&expected_array)
                        && l.contains(&format!("register({register})"))),
            "main_bindless.slang DXIL ABI drifted: expected `{param}` at register({register}). \
             The DirectX bindless binding layout is frozen (world Shader assets build PSOs \
             against the same root signature); fix the .slang declarations or the root \
             signature in src/directx/init/pipelines.rs before shipping."
        );
    }
}

// Expose the define values baked into the precompiled slang metallibs so a
// unit test can lock them to the crate's own constants.
fn emit_slang_metal_defines() {
    let value = |key: &str| {
        SLANG_MAIN_DEFINES
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| *v)
            .unwrap_or_default()
    };
    assert_eq!(value("POOL_SIZE"), SLANG_METAL_POOL_SIZE.to_string());
    assert_eq!(value("MAX_PROBES"), SLANG_METAL_MAX_PROBES.to_string());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("slang_metal_defines.rs");
    std::fs::write(
        &out,
        format!(
            "pub(crate) const SLANG_METAL_POOL_SIZE: usize = {SLANG_METAL_POOL_SIZE};\n\
             pub(crate) const SLANG_METAL_MAX_PROBES: usize = {SLANG_METAL_MAX_PROBES};\n"
        ),
    )
    .expect("write slang_metal_defines.rs");
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
