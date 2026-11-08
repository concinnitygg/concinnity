//! The device-backend crate (rlib). The first two jobs run behind one
//! `setup_graphics_backends` call into the shared concinnity-toolchain helper:
//!
//! 1. Resolve the rendering backends and expose one cfg per backend the crate
//!    gates on (`backend_metal` / `backend_dx` / `backend_vk`).
//!
//! 2. Detect the optional upscaler SDKs and emit the cfgs the backends gate on.
//!    This crate produces only an rlib (consumed by the client) plus its own test
//!    binaries, so it does NOT bundle runtime DLLs next to a binary (that belongs
//!    to whichever package owns the final artifact).
//!
//! 3. Precompile every resolved backend's engine programs (metallibs, DXIL,
//!    SPIR-V) so the binary embeds them.
//!
//! 4. Check the Metal and DirectX bindings the shared sources declare against
//!    the slots and root signatures their hosts bind, as the engine's
//!    `programs::abi` tables state them (`assert_metal_abi`, `assert_dxil_abi`).
//!
//! 5. Derive the hash of the shader-compile sources that `shader_cache` folds
//!    into every artifact key (see `emit_shader_compile_source_hash`).

use concinnity_core::platform::Platform;
use concinnity_engine::render::shader::programs::abi::{
    DXIL_ENTRY_ABI, DxilAbi, METAL_ENTRY_ABI, MetalAbi,
};
use concinnity_engine::render::shader::programs::{ShaderProgram, Table, dx, metal};
use concinnity_shader::HlslTarget;
use concinnity_toolchain::{
    Backend, EmittedMsl, ShaderArtifact, hash_sources, msl_binds, msl_binds_a_resource,
    msl_entry_params, msl_param_name, parallel_map, precompile_metal_shaders,
    setup_graphics_backends,
};
use std::path::PathBuf;

// The modules that decide how a shader artifact is produced: the cache itself
// (the key layout, and what an entry stores) and each backend's compiler
// invocation. A cache key already covers the assembled shader source, the entry
// point, the target and the caller's option word, so what it cannot see is a
// change to the invocation around them -- a different optimization level, an
// added flag, a reworked entry format. Hashing these sources in closes that
// gap, so such a change misses instead of loading bytes the old invocation
// produced. Every backend's module participates on every build, which keeps the
// hash independent of the resolved backend; the key's `compiler` field is what
// keeps one toolchain's artifacts away from another's.
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
    "src/shader/builtin.rs",
    "src/directx/builtin_shaders.rs",
    "src/metal/msl_cache.rs",
    "src/metal/builtin_shaders.rs",
    "src/vulkan/builtin_shaders.rs",
];

// Compile every declared Vulkan program to SPIR-V now, for the same reason the
// DirectX leg does: so the binary carries its shaders rather than needing dxc on
// the host that runs it.
//
// Only `USE_MSAA` varies per program, and it does not depend on the world, which
// is what makes it enumerable here: it is the host's sample count, so a program
// that reads the main pass's depth gets both variants. Every array a program
// binds is unsized or a single resource, so no capacity reaches the source.
fn precompile_spirv() {
    use concinnity_engine::render::shader::programs::vk;

    concinnity_toolchain::precompile_shader_artifacts(
        &artifacts(&vk::TABLE, Platform::Vulkan, |_| HlslTarget::Spirv),
        "engine_spirv.rs",
        "embedded_spirv",
    );
}

// Compile every declared DirectX program to DXIL now, so the binary carries its
// shaders instead of needing dxc on whatever host runs it. The declarations
// and the assembly both come from `core::render`, the same ones the renderer
// uses, so the text compiled here is the text it would have compiled -- which is
// what lets `directx::builtin_shaders` serve these bytes without re-deriving the
// cache key.
//
// A DirectX build on any host but Windows is a cross-compile that never runs
// the binary, so it embeds nothing and the lookup answers `None` for everything.
fn precompile_dxil() {
    if !cfg!(windows) {
        concinnity_toolchain::precompile_shader_artifacts(&[], "engine_dxil.rs", "embedded_dxil");
        return;
    }
    let target = |p: &ShaderProgram| HlslTarget::Dxil {
        shader_model_6_5: dx::shader_model_6_5(p),
    };
    concinnity_toolchain::precompile_shader_artifacts(
        &artifacts(&dx::TABLE, Platform::DirectX, target),
        "engine_dxil.rs",
        "embedded_dxil",
    );
}

// Every variant `table` expands to, filed under the name the renderer looks it
// up by and assembled the way the renderer assembles it.
fn artifacts(
    table: &Table,
    platform: Platform,
    target: impl Fn(&ShaderProgram) -> HlslTarget,
) -> Vec<ShaderArtifact<'static>> {
    table
        .variants()
        .map(|v| ShaderArtifact {
            name: v.artifact_name(),
            source: v.assemble(platform),
            entry: v.program.entry,
            file_name: v.program.file,
            target: target(v.program),
        })
        .collect()
}

fn main() {
    let backends = setup_graphics_backends();
    if backends.contains(&Backend::Metal) {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let shaders_dir = manifest.join("src/metal/shaders");
        // A cross build with no Metal toolchain has already warned that it
        // embeds nothing, and it emitted no MSL to check.
        if let Some(msl) = precompile_metal_shaders(&shaders_dir, &metal::TABLE) {
            assert_metal_abi(&msl);
        }
    }
    if backends.contains(&Backend::Dx) {
        assert_dxil_abi();
        precompile_dxil();
    }
    if backends.contains(&Backend::Vk) {
        precompile_spirv();
    }
    emit_shader_compile_source_hash();
}

// The Metal main-pass binding layout is what the encoders in `metal/draw/`
// write, and a world Shader's stages compile from the same declarations. The
// sources pin every slot with a register(), and the table that turns those into
// Metal indices is generated (concinnity-shader's `metal_bindings`), so assert the
// MSL the precompile emitted: a slot that moves fails the build instead of
// binding garbage at draw time.
#[expect(clippy::panic, reason = "a build script reports failure by panicking")]
fn assert_metal_abi(emitted: &EmittedMsl) {
    for abi in METAL_ENTRY_ABI {
        let program = abi.program;
        let name = program.at(false).artifact_name();
        let msl = emitted
            .get(&name)
            .unwrap_or_else(|| panic!("{name}: the precompile emitted no MSL"));
        let params = msl_entry_params(msl, program.entry).unwrap_or_else(|e| panic!("{e}"));
        assert_every_param_is_attributed(abi, &params);
        assert_every_resource_is_listed(abi, &params);
        for (param, attribute) in abi.slots() {
            assert!(
                msl_binds(msl, param, attribute),
                "{}: Metal ABI drifted at `{}`: expected `{param}` on [[{attribute}]] in the \
                 emitted MSL. \
                 The Metal binding layout is what the encoder writes and what world shaders \
                 compile from; fix the shader's declarations or the slot assignment \
                 before shipping.",
                program.file,
                program.entry
            );
        }
        for (member, id) in abi.argument_ids {
            let found = concinnity_shader::msl_argument_id(msl, member)
                .and_then(|id| usize::try_from(id).ok());
            assert_eq!(
                found,
                Some(*id),
                "{}: argument-buffer member `{member}` is at id {found:?}, not {id}, which is \
                 where the host writes it. A member's register number is its [[id(n)]]; fix \
                 the declaration.",
                program.file,
            );
        }
    }
}

// An entry parameter with no binding attribute is placed by the Metal compiler
// at whatever slot happens to be unused -- a placement the emitted MSL cannot be
// read for, and one an unrelated declaration silently moves.
fn assert_every_param_is_attributed(abi: &MetalAbi, params: &[String]) {
    let unattributed: Vec<&str> = params
        .iter()
        .filter(|p| !p.contains("[["))
        .map(|p| msl_param_name(p))
        .collect();
    assert!(
        unattributed.is_empty(),
        "{} ({}): the emitted MSL has entry parameters with no binding attribute: \
         {unattributed:?}, so the Metal compiler places them at the next unused slot and no \
         host binding can be checked against them. Give each a register(), or a \
         [[cn::metal_argument_buffer(n)]] for a set that rides an argument buffer.",
        abi.program.file,
        abi.program.entry,
    );
}

// A resource parameter the row does not list is one its host encoder never
// binds, which Metal validation rejects and a release build reads as garbage.
fn assert_every_resource_is_listed(abi: &MetalAbi, params: &[String]) {
    let unlisted: Vec<&str> = params
        .iter()
        .filter(|p| msl_binds_a_resource(p))
        .map(|p| msl_param_name(p))
        .filter(|name| !abi.slots().any(|(param, _)| param == name))
        .collect();
    assert!(
        unlisted.is_empty(),
        "{} ({}): the emitted MSL binds {unlisted:?}, which the row does not list and so no \
         host encoder writes. Drop the read that keeps it, or bind it and list it.",
        abi.program.file,
        abi.program.entry,
    );
}

// Every DirectX program's registers against the root signature its host
// builds. Each variant compiles alone, so a register only has to hold in the
// entry that declares it. The rows preprocess side by side and are checked in
// table order.
#[expect(
    clippy::disallowed_methods,
    reason = "cargo passes build-script inputs through the environment"
)]
fn assert_dxil_abi() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    let declared = parallel_map(DXIL_ENTRY_ABI, |abi| {
        let program = abi.program;
        let source = program.at(false).assemble(Platform::DirectX);
        concinnity_shader::preprocessed_declarations(&source, program.file, &out_dir)
    });
    for (abi, declared) in DXIL_ENTRY_ABI.iter().zip(declared) {
        assert_hlsl_registers(abi, declared);
    }
}

// An `.hlsl` source states its registers outright, so the check is that the
// declarations the preprocessor leaves standing are the ones the root signature
// binds. Preprocessed rather than raw: a register behind an inactive `#if`, or
// spelled as a macro, is not the one the compile sees.
#[expect(clippy::panic, reason = "a build script reports failure by panicking")]
fn assert_hlsl_registers(
    abi: &DxilAbi,
    declared: Result<Vec<concinnity_shader::declarations::Declaration>, String>,
) {
    let program = abi.program;
    let declared = declared.unwrap_or_else(|e| panic!("DXIL ABI check ({}): {e}", program.entry));
    for (param, register) in abi.registers {
        let found = declared.iter().find(|d| d.name == *param);
        let actual = found.map(|d| dxil_register(d.register));
        assert_eq!(
            actual.as_deref(),
            Some(*register),
            "{}: `{param}` is declared at {actual:?}, not register({register}). Fix the \
             .hlsl declaration or the matching root signature under src/directx before \
             shipping.",
            program.file,
        );
    }
}

// A register annotation as the tables above spell it: `t0`, or `t0, space1`
// where the declaration names a space. The space is part of the D3D slot, so a
// row naming one asserts it.
fn dxil_register(register: concinnity_shader::declarations::Register) -> String {
    match register.space {
        0 => format!("{}{}", register.class, register.index),
        space => format!("{}{}, space{space}", register.class, register.index),
    }
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
            "const SHADER_COMPILE_SOURCE_HASH: u32 = 0x{:04x}_{:04x};\n",
            hash >> 16,
            hash & 0xffff
        ),
    )
    .expect("write shader_compile_source_hash.rs");
}
