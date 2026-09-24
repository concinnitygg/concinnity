//! The device-backend crate (rlib). The first two jobs run behind one
//! `setup_graphics_backend` call into the shared concinnity-toolchain helper:
//!
//! 1. Resolve the rendering backend once and expose it as a single cfg the crate
//!    gates on (backend_metal / backend_dx / backend_vk).
//!
//! 2. Detect the optional upscaler SDKs and emit the cfgs the backends gate on.
//!    This crate produces only an rlib (consumed by the client) plus its own test
//!    binaries, so it does NOT bundle runtime DLLs next to a binary (that belongs
//!    to whichever package owns the final artifact).
//!
//! 3. Precompile the resolved backend's engine programs (metallibs, DXIL or
//!    SPIR-V) so the binary embeds them.
//!
//! 4. Check the Metal and DirectX bindings the shared sources declare against
//!    the slots and root signatures their hosts bind (`assert_metal_abi`,
//!    `assert_dxil_abi`).
//!
//! 5. Derive the hash of the shader-compile sources that `shader_cache` folds
//!    into every artifact key (see `emit_shader_compile_source_hash`).

use concinnity_core::platform::Platform;
use concinnity_core::render::shader_programs::metal::bindless_textures;
use concinnity_core::render::shader_programs::{ShaderProgram, Table, dx, metal, shared, spd};
use concinnity_shader::HlslTarget;
use concinnity_toolchain::{
    Backend, EmittedMsl, ShaderArtifact, hash_sources, msl_binds, msl_binds_a_resource,
    msl_entry_params, msl_param_name, parallel_map, precompile_metal_shaders,
    setup_graphics_backend,
};
use std::path::PathBuf;

// Every register the bindless main root signature in
// `src/directx/init/pipelines.rs` declares, as (parameter, HLSL register).
const MAIN_BINDLESS_REGISTERS: &[(&str, &str)] = &[
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
    ("probe_records_sb", "t8"),
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

// Both view-cull phases, from `directx/cull.rs`.
const CULL_REGISTERS: &[(&str, &str)] = &[
    ("cull", "b0"),
    ("objects", "t0"),
    ("draw_args", "t1"),
    ("hiz_tex", "t2"),
    ("commands", "u0"),
    ("cull_status", "u1"),
];

// One program and the registers its DirectX root signature declares.
struct DxilAbi {
    program: &'static ShaderProgram,
    registers: &'static [(&'static str, &'static str)],
}

// The DirectX root signatures, from `src/directx/init/pipelines.rs`, `cull.rs`,
// `hiz.rs`, `probe_prefilter.rs`, `post/gbuffer.rs`, `post/rt_reflections.rs`,
// `transparent.rs` and the other modules the rows below name. They are pinned
// because the shader files are shared with Metal and Vulkan, whose hosts bind
// the same declarations at entirely different slots, so an edit made on either
// of those platforms cannot see a DirectX root signature at all.
// `dx_crosscheck.sh` runs this script's DirectX branch, which is where such an
// edit gets caught.
const DXIL_ENTRY_ABI: &[DxilAbi] = &[
    // The bindless main pair, whose layout is a contract with world Shaders as
    // well: each builds its own PSO against the same root signature (see
    // directx/world_shaders.rs), so a register that moves misbinds every one.
    // Both stages read the one declaration set.
    DxilAbi {
        program: &shared::MAIN_BINDLESS_FRAG,
        registers: MAIN_BINDLESS_REGISTERS,
    },
    // The draw cull. Its registers are pinned in the source because declaration
    // order would hand `cull_status` u0 and `commands` u1, the reverse of what
    // `directx/cull.rs` binds; the shadow variant declares no status or Hi-Z.
    DxilAbi {
        program: &shared::CULL_PHASE1,
        registers: CULL_REGISTERS,
    },
    DxilAbi {
        program: &shared::CULL_PHASE2,
        registers: CULL_REGISTERS,
    },
    DxilAbi {
        program: &shared::CULL_SHADOW,
        registers: &[
            ("cull", "b0"),
            ("objects", "t0"),
            ("draw_args", "t1"),
            ("commands", "u0"),
        ],
    },
    DxilAbi {
        program: &shared::MODEL_HISTORY,
        registers: &[("params", "b0"), ("objects", "t0"), ("history", "u0")],
    },
    // The Hi-Z pyramid's two SPD dispatches, from `directx/hiz.rs`: the params
    // as root constants at b0, the main depth at t0 (phase 1 only) and the
    // per-mip UAV table at u0.
    DxilAbi {
        program: &spd::HIZ_SPD_SINGLE,
        registers: &[("params", "b0"), ("src_depth", "t0"), ("spd_mips", "u0")],
    },
    DxilAbi {
        program: &spd::HIZ_SPD_MSAA,
        registers: &[("params", "b0"), ("src_depth", "t0"), ("spd_mips", "u0")],
    },
    DxilAbi {
        program: &spd::HIZ_SPD_TAIL,
        registers: &[("params", "b0"), ("spd_mips", "u0")],
    },
    // The reflection-probe prefilter, from `directx/probe_prefilter.rs`. The
    // mirror copy and the pyramid reduction share a signature whose UAV table
    // is u0 the source and u1 the destination; the GGX kernel binds one UAV, so
    // its destination is u0 and the capture pyramid takes t0 with the static
    // sampler at s0.
    DxilAbi {
        program: &shared::PROBE_MIP0,
        registers: &[("params", "b0"), ("src_mip", "u0"), ("dst_mip", "u1")],
    },
    DxilAbi {
        program: &shared::PROBE_DOWNSAMPLE,
        registers: &[("params", "b0"), ("src_mip", "u0"), ("dst_mip", "u1")],
    },
    DxilAbi {
        program: &shared::PROBE_GGX,
        registers: &[
            ("params", "b0"),
            ("src_cube", "t0"),
            ("src_sampler", "s0"),
            ("dst_mip", "u0"),
        ],
    },
    DxilAbi {
        program: &shared::GBUFFER_PREPASS_VERT_BINDLESS,
        registers: &[
            ("objid_cb", "b0"),
            ("gb_view", "b1"),
            ("objects", "t0"),
            ("prev_models", "t1"),
            ("draw_args", "t2"),
        ],
    },
    DxilAbi {
        program: &shared::GBUFFER_PREPASS_FRAG_BINDLESS,
        registers: &[],
    },
    DxilAbi {
        program: &shared::SHADOW_VERT,
        registers: &[("push", "b0"), ("shadow_cb", "b1")],
    },
    DxilAbi {
        program: &shared::SHADOW_VERT_SKINNED,
        registers: &[("push", "b0"), ("shadow_cb", "b1"), ("joints", "t0")],
    },
    DxilAbi {
        program: &shared::SHADOW_VERT_BINDLESS,
        registers: &[
            ("objid_cb", "b0"),
            ("shadow_cb", "b1"),
            ("push", "b2"),
            ("objects", "t0"),
        ],
    },
    DxilAbi {
        program: &shared::RT_REFLECTIONS_FRAG,
        registers: RT_REFLECTIONS_REGISTERS,
    },
    DxilAbi {
        program: &shared::RT_REFLECTIONS_FRAG_TEXTURED,
        registers: RT_REFLECTIONS_TEXTURED_REGISTERS,
    },
    DxilAbi {
        program: &shared::GLASS_VERT,
        registers: &[("view", "b0")],
    },
    DxilAbi {
        program: &shared::GLASS_FRAG,
        registers: GLASS_REGISTERS,
    },
    DxilAbi {
        program: &shared::GLASS_FRAG_RT,
        registers: GLASS_RT_REGISTERS,
    },
    DxilAbi {
        program: &shared::GLASS_FRAG_RT_TEXTURED,
        registers: GLASS_RT_TEXTURED_REGISTERS,
    },
    DxilAbi {
        program: &shared::GLASS_REFLECTION_FRAG,
        registers: GLASS_REFLECTION_REGISTERS,
    },
    DxilAbi {
        program: &shared::GLASS_REFLECTION_FRAG_TEXTURED,
        registers: GLASS_RT_TEXTURED_REGISTERS,
    },
    // The see-through glass MESH producer. Ray-traced only, so it declares the
    // RT register set unconditionally; its vertex stage reads the model matrix
    // out of b1, which is why b1 is visible to every stage in the root
    // signature.
    DxilAbi {
        program: &shared::GLASS_MESH_VERT,
        registers: &[("view", "b0"), ("params", "b1")],
    },
    DxilAbi {
        program: &shared::GLASS_MESH_FRAG_RT,
        registers: GLASS_MESH_RT_REGISTERS,
    },
    DxilAbi {
        program: &shared::GLASS_MESH_FRAG_RT_TEXTURED,
        registers: GLASS_RT_TEXTURED_REGISTERS,
    },
    DxilAbi {
        program: &shared::GLASS_MESH_REFLECTION_FRAG,
        registers: GLASS_REFLECTION_REGISTERS,
    },
    DxilAbi {
        program: &shared::GLASS_MESH_REFLECTION_FRAG_TEXTURED,
        registers: GLASS_RT_TEXTURED_REGISTERS,
    },
    // Water shares every register glass declares, which is what lets one root
    // signature per path serve both producers of the transparent pass. The RT
    // pair additionally reads the planar resolve at t3 (see
    // WATER_RT_REGISTERS), which the shared RT signature has a table for.
    DxilAbi {
        program: &shared::WATER_VERT,
        registers: &[("view", "b0"), ("params", "b1")],
    },
    DxilAbi {
        program: &shared::WATER_FRAG,
        registers: GLASS_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_FRAG_RT,
        registers: WATER_RT_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_FRAG_RT_TEXTURED,
        registers: WATER_RT_TEXTURED_REGISTERS,
    },
    // The compute families whose root signatures live in
    // `src/directx/{raytrace,light_cull,auto_exposure,particle}.rs`. Each
    // states its registers in a `CN_BACKEND_DIRECTX` branch, so these rows are the
    // check that the branch and the root signature still name the same slots.
    DxilAbi {
        program: &shared::RT_SKIN,
        registers: &[
            ("src", "t0"),
            ("palette", "t1"),
            ("dst", "u0"),
            ("morph_data", "t2"),
            ("morph_weights", "t3"),
            ("params", "b0"),
        ],
    },
    DxilAbi {
        program: &shared::LIGHT_CULL,
        registers: &[
            ("cluster", "b0"),
            ("lights", "t0"),
            ("cluster_list", "u0"),
            ("probe_records", "t1"),
        ],
    },
    DxilAbi {
        program: &shared::AUTO_EXPOSURE_BUILD,
        registers: &[("params", "b0"), ("hdr_texture", "t0"), ("histogram", "u0")],
    },
    DxilAbi {
        program: &shared::AUTO_EXPOSURE_AVERAGE,
        registers: &[("params", "b0"), ("histogram", "u0"), ("output_avg", "u1")],
    },
    DxilAbi {
        program: &shared::PARTICLE_SIMULATE,
        registers: &[("params", "b0"), ("pool", "u0"), ("spawn_counter", "u1")],
    },
    // The raster remainder, from `src/directx/{particle,decal,line}.rs` and
    // `pipeline.rs`. Only the particle pair has a `CN_BACKEND_DIRECTX` branch, and only
    // to swap the two constant buffers; the rest are here so an edit to their
    // declarations cannot move a register the root signature binds without
    // failing the build.
    DxilAbi {
        program: &shared::PARTICLE_VERT,
        registers: &[("view", "b0"), ("params", "b1"), ("pool", "t0")],
    },
    DxilAbi {
        program: &shared::PARTICLE_FRAG,
        registers: &[
            ("albedo", "t1"),
            ("albedo_sampler", "s0"),
            ("scene_depth", "t2"),
        ],
    },
    DxilAbi {
        program: &shared::DECAL_VERT,
        registers: &[("view", "b0"), ("params", "b1")],
    },
    DxilAbi {
        program: &shared::DECAL_FRAG,
        registers: &[
            ("view", "b0"),
            ("params", "b1"),
            ("scene_depth", "t0"),
            ("decal_tex", "t1"),
            ("decal_tex_sampler", "s0"),
        ],
    },
    DxilAbi {
        program: &shared::LINE_VERT,
        registers: &[("view", "b0")],
    },
    DxilAbi {
        program: &shared::LINE_FRAG,
        registers: &[("view", "b0"), ("scene_depth", "t0")],
    },
    DxilAbi {
        program: &shared::TEXT_VERT,
        registers: &[("uni", "b0")],
    },
    DxilAbi {
        program: &shared::TEXT_FRAG,
        registers: &[("atlas", "t0"), ("atlas_sampler", "s0")],
    },
];

// The ray-traced reflection resolve's root signature, from
// `src/directx/post/rt_reflections.rs`. The probe cube array, its records and
// the cluster lists binning them sit clear of the screen-space SRVs.
const RT_REFLECTIONS_REGISTERS: &[(&str, &str)] = &[
    ("rt_params", "b0"),
    ("probe_set", "b4"),
    ("cluster", "b5"),
    ("scene_tlas", "t0"),
    ("verts", "t1"),
    ("indices", "t2"),
    ("geom", "t3"),
    ("scene_tex", "t4"),
    ("gbuffer", "t5"),
    ("rough_tex", "t6"),
    ("prefilter", "t7"),
    ("sverts", "t8"),
    ("sidx", "t9"),
    ("probe_cubes", "t10"),
    ("probe_records", "t11"),
    ("cluster_list", "t12"),
    ("screen_sampler", "s0"),
    ("cube_sampler", "s1"),
    ("probe_cube_sampler", "s3"),
];

const RT_REFLECTIONS_TEXTURED_REGISTERS: &[(&str, &str)] =
    &[("tex_pool", "t0, space1"), ("pool_sampler", "s2")];

// The transparent pass's root signatures, from `src/directx/transparent.rs`,
// which glass and water both draw under. The probe cube array, its records and
// the cluster lists sit at t20 / t21 / t22 in every variant, clear of the
// trace's SRVs at t4..t12, with the cluster params at b6.
const GLASS_REGISTERS: &[(&str, &str)] = &[
    ("view", "b0"),
    ("params", "b1"),
    ("probe_set", "b4"),
    ("scene_color", "t0"),
    ("scene_depth", "t1"),
    ("prefilter_cube", "t2"),
    ("planar_reflection", "t3"),
    ("probe_cubes", "t20"),
    ("probe_records", "t21"),
    ("cluster", "b6"),
    ("cluster_list", "t22"),
    ("post_samp", "s0"),
    ("cube_sampler", "s2"),
];

const GLASS_RT_REGISTERS: &[(&str, &str)] = &[
    ("view", "b0"),
    ("params", "b1"),
    ("rt_params", "b5"),
    ("probe_set", "b4"),
    ("scene_color", "t0"),
    ("scene_depth", "t1"),
    ("prefilter_cube", "t2"),
    ("scene_tlas", "t4"),
    ("verts", "t5"),
    ("indices", "t6"),
    ("sverts", "t8"),
    ("sidx", "t9"),
    ("geom", "t10"),
    ("probe_cubes", "t20"),
    ("probe_records", "t21"),
    ("cluster", "b6"),
    ("cluster_list", "t22"),
    ("glass_reflection", "t11"),
    ("glass_reflection_back", "t12"),
];

// The reduced reflection pre-pass entries of both glass producers: the RT set,
// without the refraction snapshot or the second layer, reading the layer it
// peels behind at t11.
const GLASS_REFLECTION_REGISTERS: &[(&str, &str)] = &[
    ("view", "b0"),
    ("params", "b1"),
    ("rt_params", "b5"),
    ("probe_set", "b4"),
    ("scene_depth", "t1"),
    ("prefilter_cube", "t2"),
    ("scene_tlas", "t4"),
    ("verts", "t5"),
    ("indices", "t6"),
    ("sverts", "t8"),
    ("sidx", "t9"),
    ("geom", "t10"),
    ("glass_reflection", "t11"),
    ("probe_cubes", "t20"),
    ("probe_records", "t21"),
    ("cluster", "b6"),
    ("cluster_list", "t22"),
];

const GLASS_RT_TEXTURED_REGISTERS: &[(&str, &str)] =
    &[("tex_pool", "t0, space1"), ("pool_sampler", "s1")];

// Water's RT fragments, which are glass's set plus the planar resolve at t3: a
// water surface with a mirror plane samples it in place of tracing, so unlike
// glass the register survives into the RT variant and the RT root signature
// carries a table for it.
const WATER_RT_REGISTERS: &[(&str, &str)] = &[
    ("view", "b0"),
    ("params", "b1"),
    ("rt_params", "b5"),
    ("probe_set", "b4"),
    ("scene_color", "t0"),
    ("scene_depth", "t1"),
    ("prefilter_cube", "t2"),
    ("planar_reflection", "t3"),
    ("scene_tlas", "t4"),
    ("verts", "t5"),
    ("indices", "t6"),
    ("sverts", "t8"),
    ("sidx", "t9"),
    ("geom", "t10"),
    ("probe_cubes", "t20"),
    ("probe_records", "t21"),
    ("cluster", "b6"),
    ("cluster_list", "t22"),
];

const WATER_RT_TEXTURED_REGISTERS: &[(&str, &str)] = &[
    ("planar_reflection", "t3"),
    ("tex_pool", "t0, space1"),
    ("pool_sampler", "s1"),
];

// The see-through glass mesh fragment: the RT set above, minus the planar
// resolve at t3 (a curved mesh has no mirror plane) and with the same t20 probe
// cube base, so the pass's RT root signature covers it unchanged.
const GLASS_MESH_RT_REGISTERS: &[(&str, &str)] = &[
    ("view", "b0"),
    ("params", "b1"),
    ("rt_params", "b5"),
    ("probe_set", "b4"),
    ("scene_color", "t0"),
    ("scene_depth", "t1"),
    ("prefilter_cube", "t2"),
    ("scene_tlas", "t4"),
    ("verts", "t5"),
    ("indices", "t6"),
    ("sverts", "t8"),
    ("sidx", "t9"),
    ("geom", "t10"),
    ("probe_cubes", "t20"),
    ("probe_records", "t21"),
    ("cluster", "b6"),
    ("cluster_list", "t22"),
    ("post_samp", "s0"),
    ("cube_sampler", "s2"),
    ("glass_reflection", "t11"),
    ("glass_reflection_back", "t12"),
];

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
    use concinnity_core::render::shader_programs::vk;

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
// The DirectX branch runs on any host but Windows only under
// `dx_crosscheck.sh`, which type-checks the backend and never runs the binary,
// so that host embeds nothing and the lookup answers `None` for everything.
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
    let backend = setup_graphics_backend();
    if backend == Some(Backend::Metal) {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let shaders_dir = manifest.join("src/metal/shaders");
        // A cross build with no Metal toolchain has already warned that it
        // embeds nothing, and it emitted no MSL to check.
        if let Some(msl) = precompile_metal_shaders(&shaders_dir, &metal::TABLE) {
            assert_metal_abi(&msl);
        }
    }
    if backend == Some(Backend::Dx) {
        assert_dxil_abi();
        precompile_dxil();
    }
    if backend == Some(Backend::Vk) {
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

// One program and the Metal slots its host encoder writes, as groups of
// (parameter, attribute) pairs.
struct MetalAbi {
    program: &'static ShaderProgram,
    slots: &'static [&'static [(&'static str, &'static str)]],
    // Argument-buffer members and the `[[id(n)]]` the host encodes each at.
    argument_ids: &'static [(&'static str, usize)],
}

impl MetalAbi {
    // Every slot of the row, across its groups.
    fn slots(&self) -> impl Iterator<Item = &'static (&'static str, &'static str)> {
        self.slots.iter().flat_map(|group| group.iter())
    }
}

const METAL_ENTRY_ABI: &[MetalAbi] = &[
    // The draw cull's decision half. Its buffers are pinned by register() and
    // the Hi-Z texture takes texture(0) from declaration order; the slots are
    // what `metal/cull.rs` binds once for this dispatch and the ICB encode
    // dispatch that follows it on the same encoder.
    MetalAbi {
        program: &shared::CULL_PHASE1,
        slots: &[&[
            ("objects", "buffer(0)"),
            ("draw_args", "buffer(1)"),
            ("cull", "buffer(2)"),
            ("cull_status", "buffer(5)"),
            ("hiz_tex", "texture(0)"),
        ]],
        argument_ids: &[],
    },
    // Phase 2 re-tests candidates only, so it never reads `draw_args`; the
    // encode dispatch after it does, from the same binding.
    MetalAbi {
        program: &shared::CULL_PHASE2,
        slots: &[&[
            ("objects", "buffer(0)"),
            ("cull", "buffer(2)"),
            ("cull_status", "buffer(5)"),
            ("hiz_tex", "texture(0)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::CULL_SHADOW,
        slots: &[&[
            ("objects", "buffer(0)"),
            ("draw_args", "buffer(1)"),
            ("cull", "buffer(2)"),
            ("cull_status", "buffer(5)"),
        ]],
        argument_ids: &[],
    },
    // The model-history snapshot: params, the object buffer it reads and the
    // ring slot it writes take buffer(0..2) from declaration order, which is
    // what `metal/model_history.rs` binds.
    MetalAbi {
        program: &shared::MODEL_HISTORY,
        slots: &[&[
            ("params", "buffer(0)"),
            ("objects", "buffer(1)"),
            ("history", "buffer(2)"),
        ]],
        argument_ids: &[],
    },
    // The GPU-driven G-buffer pre-pass vertex. Its buffers are pinned by
    // register() numbers so they clear the vertex descriptor's streams at
    // buffer(1) and buffer(2); the slots are what `metal/post/gbuffer.rs`
    // binds once for every ICB-executed draw.
    MetalAbi {
        program: &shared::GBUFFER_PREPASS_VERT_BINDLESS,
        slots: &[&[
            ("gb_view", "buffer(0)"),
            ("objects", "buffer(9)"),
            ("prev_models", "buffer(10)"),
            ("draw_args", "buffer(11)"),
        ]],
        argument_ids: &[],
    },
    // The cascaded shadow pass, one row per variant: depth-only, so every
    // parameter here is a constant block or the object record, and the split
    // between buffer(2) and buffer(7) is the one the spot-shadow pass shares.
    MetalAbi {
        program: &shared::SHADOW_VERT,
        slots: &[&[
            ("shadow_cb", "buffer(0)"),
            ("model_cb", "buffer(2)"),
            ("cascade_cb", "buffer(7)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::SHADOW_VERT_SKINNED,
        slots: &[&[
            ("shadow_cb", "buffer(0)"),
            ("model_cb", "buffer(2)"),
            ("cascade_cb", "buffer(7)"),
            ("joints", "buffer(8)"),
        ]],
        argument_ids: &[],
    },
    // The GPU-driven variant takes its model out of the object record instead,
    // so it binds no model block and reads buffer(9) like the pre-pass.
    MetalAbi {
        program: &shared::SHADOW_VERT_BINDLESS,
        slots: &[&[
            ("shadow_cb", "buffer(0)"),
            ("cascade_cb", "buffer(7)"),
            ("objects", "buffer(9)"),
        ]],
        argument_ids: &[],
    },
    // The bindless main pass. Its two argument buffers are whole descriptor
    // sets, each at the index `[[cn::metal_argument_buffer(n)]]` names, and the
    // ids inside the texture one are where `metal/cull.rs` writes each member:
    // the fixed shadow / IBL / SSAO / probe / spot / LTC members first, then
    // the unsized pool. `EngineSamplers` is written 0..2 by
    // `build_bindless_sampler_args`.
    MetalAbi {
        program: &shared::MAIN_BINDLESS_FRAG,
        slots: &[&[
            ("view_cb", "buffer(0)"),
            ("lights_cb", "buffer(4)"),
            ("shadow_cb", "buffer(5)"),
            ("probe_set_cb", "buffer(6)"),
            ("probe_records_sb", "buffer(15)"),
            ("spvDescriptorSet1", "buffer(7)"),
            ("local_lights_sb", "buffer(8)"),
            ("objects_sb", "buffer(9)"),
            ("spvDescriptorSet2", "buffer(10)"),
            ("cluster_cb", "buffer(11)"),
            ("cluster_list_sb", "buffer(12)"),
            ("spot_shadows_sb", "buffer(13)"),
            ("area_lights_sb", "buffer(14)"),
        ]],
        argument_ids: &[
            ("shadow_map", bindless_textures::SHADOW_MAP),
            ("irradiance_cube", bindless_textures::IRRADIANCE_CUBE),
            ("prefilter_cube", bindless_textures::PREFILTER_CUBE),
            ("ssao_tex", bindless_textures::SSAO),
            ("probe_cubes", bindless_textures::PROBE_CUBES),
            ("spot_shadow_map", bindless_textures::SPOT_SHADOW_MAP),
            ("ltc_matrix", bindless_textures::LTC_MATRIX),
            ("ltc_magnitude", bindless_textures::LTC_MAGNITUDE),
            ("tex_pool", bindless_textures::pool(0)),
            ("tex_sampler", 0),
            ("shadow_sampler", 1),
            ("cube_sampler", 2),
        ],
    },
    MetalAbi {
        program: &shared::MAIN_BINDLESS_VERT,
        slots: &[&[("view_cb", "buffer(0)"), ("objects_sb", "buffer(9)")]],
        argument_ids: &[],
    },
    // The SSR resolve: the probe cube array, its records and the cluster grid
    // are discrete bindings beside the screen sources, which is what
    // `metal/post/post_device.rs` binds.
    MetalAbi {
        program: &shared::SSR_RESOLVE,
        slots: &[&[
            ("params", "buffer(0)"),
            ("probe_set", "buffer(1)"),
            ("scene", "texture(0)"),
            ("scene_samp", "sampler(0)"),
            ("gbuffer", "texture(1)"),
            ("gbuffer_samp", "sampler(1)"),
            ("rough_tex", "texture(2)"),
            ("rough_tex_samp", "sampler(2)"),
            ("prefilter", "texture(3)"),
            ("prefilter_samp", "sampler(3)"),
            ("probe_records", "buffer(5)"),
            ("probe_cubes", "texture(4)"),
            ("probe_cube_sampler", "sampler(4)"),
            ("cluster", "buffer(2)"),
            ("cluster_list", "buffer(6)"),
        ]],
        argument_ids: &[],
    },
    // The RT resolve. The textured variant's one argument buffer is the
    // bindless pool at buffer(7): the main pass's texture block, bound at the
    // pool's offset, so the unsized array's id 0 is the pool's first texture.
    // Everything else, the probe set included, is pinned by its register().
    MetalAbi {
        program: &shared::RT_REFLECTIONS_FRAG,
        slots: &[RT_REFLECTIONS_SLOTS],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::RT_REFLECTIONS_FRAG_TEXTURED,
        slots: &[RT_REFLECTIONS_SLOTS, RT_REFLECTIONS_POOL_SLOTS],
        argument_ids: &[("tex_pool", 0)],
    },
    MetalAbi {
        program: &shared::GLASS_FRAG,
        slots: GLASS_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GLASS_FRAG_RT,
        slots: GLASS_RT_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GLASS_REFLECTION_FRAG,
        slots: GLASS_REFLECTION_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GLASS_FRAG_RT_TEXTURED,
        slots: GLASS_RT_TEXTURED_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GLASS_REFLECTION_FRAG_TEXTURED,
        slots: GLASS_REFLECTION_TEXTURED_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GLASS_MESH_FRAG_RT,
        slots: GLASS_RT_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GLASS_MESH_REFLECTION_FRAG,
        slots: GLASS_REFLECTION_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GLASS_MESH_FRAG_RT_TEXTURED,
        slots: GLASS_RT_TEXTURED_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GLASS_MESH_REFLECTION_FRAG_TEXTURED,
        slots: GLASS_REFLECTION_TEXTURED_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_FRAG,
        slots: GLASS_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_FRAG_RT,
        slots: WATER_RT_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_FRAG_RT_TEXTURED,
        slots: WATER_RT_TEXTURED_SLOTS,
        argument_ids: &[],
    },
    // The HLSL families. Their Metal indices come from the `register()`
    // annotations in the source rather than from declaration order, so these
    // rows check the generated binding table against the slots the encoders in
    // `metal/{post/bloom,text_upload,decal,particle}.rs` write.
    MetalAbi {
        program: &shared::BLOOM_PREFILTER,
        slots: &[&[
            ("src", "texture(0)"),
            ("src_sampler", "sampler(0)"),
            ("post", "buffer(0)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::BLOOM_DOWNSAMPLE,
        slots: &[&[("src", "texture(0)"), ("src_sampler", "sampler(0)")]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::BLOOM_UPSAMPLE,
        slots: &[&[("src", "texture(0)"), ("src_sampler", "sampler(0)")]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::TEXT_VERT,
        slots: &[&[("uni", "buffer(0)")]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::TEXT_FRAG,
        slots: &[&[("atlas", "texture(0)"), ("atlas_sampler", "sampler(0)")]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::DECAL_VERT,
        slots: &[&[("view", "buffer(0)"), ("params", "buffer(1)")]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::DECAL_FRAG,
        slots: &[&[
            ("view", "buffer(0)"),
            ("params", "buffer(1)"),
            ("scene_depth", "texture(0)"),
            ("decal_tex", "texture(1)"),
            ("decal_tex_sampler", "sampler(0)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PARTICLE_VERT,
        slots: &[&[
            ("pool", "buffer(0)"),
            ("view", "buffer(1)"),
            ("params", "buffer(2)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PARTICLE_FRAG,
        slots: &[&[
            ("albedo", "texture(0)"),
            ("albedo_sampler", "sampler(0)"),
            ("scene_depth", "texture(2)"),
        ]],
        argument_ids: &[],
    },
    // The fullscreen post fragments. Each source is one texture/sampler pair at
    // the index its register names, and the constants ride buffer(0), which is
    // what `metal/post/{post_device,ssao}.rs` bind.
    MetalAbi {
        program: &shared::TAA_FRAG,
        slots: &[&[
            ("params", "buffer(0)"),
            ("scene_tex", "texture(0)"),
            ("scene_samp", "sampler(0)"),
            ("velocity_tex", "texture(1)"),
            ("velocity_samp", "sampler(1)"),
            ("history_tex", "texture(2)"),
            ("history_samp", "sampler(2)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::SSAO_KERNEL,
        slots: &[&[
            ("params", "buffer(0)"),
            ("gbuffer", "texture(0)"),
            ("gbuffer_samp", "sampler(0)"),
        ]],
        argument_ids: &[],
    },
    // The blur binds no constants at all, so its only slots are the two
    // sources.
    MetalAbi {
        program: &shared::SSAO_BLUR,
        slots: &[&[
            ("ao_raw", "texture(0)"),
            ("ao_raw_samp", "sampler(0)"),
            ("gbuffer", "texture(1)"),
            ("gbuffer_samp", "sampler(1)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::SSGI_GATHER,
        slots: &[&[
            ("params", "buffer(0)"),
            ("scene", "texture(0)"),
            ("scene_samp", "sampler(0)"),
            ("gbuffer", "texture(1)"),
            ("gbuffer_samp", "sampler(1)"),
        ]],
        argument_ids: &[],
    },
    // The volumetric-fog pair. The froxel kernel is the one place the two ABIs
    // disagree: its destination volume is a UAV, which D3D numbers in a space of
    // its own and Metal folds into the texture namespace behind the cascades, so
    // the non-DXIL leg declares it `u1` and lands on texture(1) -- which is what
    // `metal/fog.rs` binds, and the opposite of where the emitter would place it
    // unaided.
    MetalAbi {
        program: &shared::FOG_FROXEL,
        slots: &[&[
            ("fog", "buffer(0)"),
            ("froxel", "buffer(1)"),
            ("shadow_uni", "buffer(2)"),
            ("shadow_map", "texture(0)"),
            ("shadow_samp", "sampler(0)"),
            ("fog_volume", "texture(1)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::FOG_FRAG,
        slots: &[&[
            ("fog", "buffer(0)"),
            ("froxel", "buffer(1)"),
            ("scene_depth", "texture(0)"),
            ("fog_volume", "texture(1)"),
            ("fog_volume_samp", "sampler(0)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::SSGI_COMPOSITE,
        slots: &[&[
            ("params", "buffer(0)"),
            ("gi_tex", "texture(0)"),
            ("gi_samp", "sampler(0)"),
            ("gbuffer", "texture(1)"),
            ("gbuffer_samp", "sampler(1)"),
        ]],
        argument_ids: &[],
    },
    // The reflection pair and the final composite, the heaviest source counts
    // in the tree. `metal/post/ssr.rs` binds each source at its declaration
    // index and repeats one sampler across the same range, and
    // `metal/post/post_device.rs` puts the composite constants on buffer(0).
    MetalAbi {
        program: &shared::REFLECTION_BLUR,
        slots: &[&[
            ("reflection", "texture(0)"),
            ("reflection_samp", "sampler(0)"),
            ("rough_tex", "texture(1)"),
            ("rough_tex_samp", "sampler(1)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::REFLECTION_COMPOSITE,
        slots: &[&[
            ("reflection", "texture(0)"),
            ("reflection_samp", "sampler(0)"),
            ("scene", "texture(1)"),
            ("scene_samp", "sampler(1)"),
            ("gbuffer", "texture(2)"),
            ("gbuffer_samp", "sampler(2)"),
            ("rough_tex", "texture(3)"),
            ("rough_tex_samp", "sampler(3)"),
            ("blur_tex", "texture(4)"),
            ("blur_tex_samp", "sampler(4)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::COMPOSITE_FRAG,
        slots: &[&[
            ("post", "buffer(0)"),
            ("hdr_tex", "texture(0)"),
            ("hdr_tex_samp", "sampler(0)"),
            ("bloom_tex", "texture(1)"),
            ("bloom_tex_samp", "sampler(1)"),
            ("lut_tex", "texture(2)"),
            ("lut_tex_samp", "sampler(2)"),
            ("gbuf_nd_tex", "texture(3)"),
            ("gbuf_nd_tex_samp", "sampler(3)"),
            ("gbuf_rough_tex", "texture(4)"),
            ("gbuf_rough_tex_samp", "sampler(4)"),
            ("ao_tex", "texture(5)"),
            ("ao_tex_samp", "sampler(5)"),
        ]],
        argument_ids: &[],
    },
    // The compute families and the world-space line pair. Their Metal indices
    // come from the `register()` annotations too, and the slots below are what
    // `metal/{auto_exposure,light_cull,particle,line}.rs` and the two skin
    // dispatches in `metal/raytrace.rs` write.
    MetalAbi {
        program: &shared::AUTO_EXPOSURE_BUILD,
        slots: &[&[
            ("hdr_texture", "texture(0)"),
            ("histogram", "buffer(0)"),
            ("params", "buffer(1)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::AUTO_EXPOSURE_AVERAGE,
        slots: &[&[
            ("histogram", "buffer(0)"),
            ("output_avg", "buffer(1)"),
            ("params", "buffer(2)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::LIGHT_CULL,
        slots: &[&[
            ("cluster", "buffer(0)"),
            ("lights", "buffer(1)"),
            ("cluster_list", "buffer(2)"),
            ("probe_records", "buffer(3)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PARTICLE_SIMULATE,
        slots: &[&[
            ("pool", "buffer(0)"),
            ("spawn_counter", "buffer(1)"),
            ("params", "buffer(2)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::RT_SKIN,
        slots: &[&[
            ("src", "buffer(0)"),
            ("dst", "buffer(1)"),
            ("palette", "buffer(2)"),
            ("params", "buffer(3)"),
            ("morph_data", "buffer(4)"),
            ("morph_weights", "buffer(5)"),
        ]],
        argument_ids: &[],
    },
    // The Hi-Z pyramid's per-mip chain, which is Metal's half of the builder
    // (Vulkan and DirectX run the SPD trio instead). The slots are what
    // `metal/hiz.rs` writes for every dispatch of the chain.
    MetalAbi {
        program: &metal::HIZ_INIT_MSAA,
        slots: HIZ_INIT_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &metal::HIZ_INIT_SINGLE,
        slots: HIZ_INIT_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &metal::HIZ_DOWNSAMPLE,
        slots: &[&[
            ("params", "buffer(0)"),
            ("src_hiz", "texture(0)"),
            ("dst_mip", "texture(1)"),
        ]],
        argument_ids: &[],
    },
    // The reflection-probe prefilter, whose three kernels `metal/probe_prefilter.rs`
    // binds the same way: params, source, destination. The GGX kernel's
    // destination is the one slot D3D and Metal disagree on, so its `u`
    // register moves behind `CN_BACKEND_DIRECTX` and Metal keeps texture(1).
    MetalAbi {
        program: &shared::PROBE_MIP0,
        slots: &[&[
            ("params", "buffer(0)"),
            ("src_mip", "texture(0)"),
            ("dst_mip", "texture(1)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PROBE_DOWNSAMPLE,
        slots: &[&[
            ("params", "buffer(0)"),
            ("src_mip", "texture(0)"),
            ("dst_mip", "texture(1)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PROBE_GGX,
        slots: &[&[
            ("params", "buffer(0)"),
            ("src_cube", "texture(0)"),
            ("src_sampler", "sampler(0)"),
            ("dst_mip", "texture(1)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::LINE_VERT,
        slots: &[&[("view", "buffer(0)")]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::LINE_FRAG,
        slots: &[&[("view", "buffer(0)"), ("scene_depth", "texture(0)")]],
        argument_ids: &[],
    },
];

// Both Hi-Z init variants: the params, the depth they read and the first mip.
const HIZ_INIT_SLOTS: &[&[(&str, &str)]] = &[&[
    ("params", "buffer(0)"),
    ("src_depth", "texture(0)"),
    ("dst_mip", "texture(1)"),
]];

// What every transparent fragment binds for its own draw: the view and
// surface blocks, the scene depth and the sky prefilter cube.
const TRANSPARENT_SLOTS: &[(&str, &str)] = &[
    ("view", "buffer(5)"),
    ("params", "buffer(6)"),
    ("scene_depth", "texture(1)"),
    ("prefilter_cube", "texture(2)"),
    ("prefilter_cube_sampler", "sampler(1)"),
];

// The transparent pass's probe set and the cluster grid that bins it.
const TRANSPARENT_PROBE_SLOTS: &[(&str, &str)] = &[
    ("probe_set", "buffer(7)"),
    ("probe_records", "buffer(11)"),
    ("cluster", "buffer(12)"),
    ("cluster_list", "buffer(13)"),
    ("probe_cubes", "texture(6)"),
    ("probe_cube_sampler", "sampler(2)"),
];

// The refraction snapshot every see-through transparent fragment reads.
const SCENE_COLOR_SLOTS: &[(&str, &str)] = &[
    ("scene_color", "texture(0)"),
    ("scene_color_sampler", "sampler(0)"),
];

// The planar mirror resolve.
const PLANAR_SLOTS: &[(&str, &str)] = &[
    ("planar_reflection", "texture(3)"),
    ("planar_reflection_sampler", "sampler(3)"),
];

// The ray-traced transparent fragments' trace inputs.
const TRANSPARENT_TRACE_SLOTS: &[(&str, &str)] = &[
    ("rt_params", "buffer(0)"),
    ("verts", "buffer(1)"),
    ("indices", "buffer(2)"),
    ("geom", "buffer(3)"),
    ("scene_tlas", "buffer(4)"),
    ("sverts", "buffer(8)"),
    ("sidx", "buffer(9)"),
];

// Both reflection layers the reduced pre-pass wrote.
const REFLECTION_LAYER_SLOTS: &[(&str, &str)] = &[
    ("glass_reflection", "texture(4)"),
    ("glass_reflection_back", "texture(5)"),
];

// The reflection layer a pre-pass entry peels behind.
const REFLECTION_PEEL_SLOTS: &[(&str, &str)] = &[("glass_reflection", "texture(4)")];

// A textured transparent variant's bindless pool argument buffer and sampler.
const TRANSPARENT_POOL_SLOTS: &[(&str, &str)] = &[
    ("spvDescriptorSet6", "buffer(10)"),
    ("pool_sampler", "sampler(4)"),
];

// Every slot the RT resolve binds, the probe set included, pinned by its
// register().
const RT_REFLECTIONS_SLOTS: &[(&str, &str)] = &[
    ("rt_params", "buffer(0)"),
    ("verts", "buffer(1)"),
    ("indices", "buffer(2)"),
    ("geom", "buffer(3)"),
    ("scene_tlas", "buffer(4)"),
    ("sverts", "buffer(5)"),
    ("sidx", "buffer(6)"),
    ("probe_set", "buffer(8)"),
    ("scene_tex", "texture(0)"),
    ("scene_tex_sampler", "sampler(0)"),
    ("gbuffer", "texture(1)"),
    ("gbuffer_sampler", "sampler(1)"),
    ("rough_tex", "texture(2)"),
    ("rough_tex_sampler", "sampler(2)"),
    ("prefilter", "texture(3)"),
    ("prefilter_sampler", "sampler(3)"),
    ("probe_records", "buffer(11)"),
    ("probe_cubes", "texture(4)"),
    ("probe_cube_sampler", "sampler(4)"),
    ("cluster", "buffer(9)"),
    ("cluster_list", "buffer(10)"),
];

// The textured RT resolve's bindless pool: the main pass's texture block,
// bound at the pool's offset.
const RT_REFLECTIONS_POOL_SLOTS: &[(&str, &str)] = &[
    ("spvDescriptorSet3", "buffer(7)"),
    ("pool_sampler", "sampler(5)"),
];

const GLASS_SLOTS: &[&[(&str, &str)]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    SCENE_COLOR_SLOTS,
    PLANAR_SLOTS,
];
const GLASS_RT_SLOTS: &[&[(&str, &str)]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    SCENE_COLOR_SLOTS,
    TRANSPARENT_TRACE_SLOTS,
    REFLECTION_LAYER_SLOTS,
];
const GLASS_RT_TEXTURED_SLOTS: &[&[(&str, &str)]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    SCENE_COLOR_SLOTS,
    TRANSPARENT_TRACE_SLOTS,
    REFLECTION_LAYER_SLOTS,
    TRANSPARENT_POOL_SLOTS,
];
const GLASS_REFLECTION_SLOTS: &[&[(&str, &str)]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    TRANSPARENT_TRACE_SLOTS,
    REFLECTION_PEEL_SLOTS,
];
const GLASS_REFLECTION_TEXTURED_SLOTS: &[&[(&str, &str)]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    TRANSPARENT_TRACE_SLOTS,
    REFLECTION_PEEL_SLOTS,
    TRANSPARENT_POOL_SLOTS,
];
const WATER_RT_SLOTS: &[&[(&str, &str)]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    SCENE_COLOR_SLOTS,
    PLANAR_SLOTS,
    TRANSPARENT_TRACE_SLOTS,
];
const WATER_RT_TEXTURED_SLOTS: &[&[(&str, &str)]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    SCENE_COLOR_SLOTS,
    PLANAR_SLOTS,
    TRANSPARENT_TRACE_SLOTS,
    TRANSPARENT_POOL_SLOTS,
];

// Every DirectX program's registers against the root signature its host
// builds. Each variant compiles alone, so a register only has to hold in the
// entry that declares it. The rows preprocess side by side and are checked in
// table order.
fn assert_dxil_abi() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
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
