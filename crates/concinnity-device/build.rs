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
    Backend, SdkOptions, SlangLibSpec, SlangShaders, emit_backend_cfg, emit_check_cfgs,
    hash_sources, precompile_metal_shaders, setup_graphics_sdks,
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

// The same idea for the single-source shaders, matching what
// `crate::slang_source::assemble` splices when the renderer compiles one at
// runtime. The two tables must agree: a mismatch would have the build script
// and the renderer key different text for the same program.
const SLANG_SHADER_FRAGMENTS: &[(&str, &str)] = &[
    ("{POST_COMMON}", "post_common.slang"),
    ("{OBJECT_COMMON}", "object_common.slang"),
    ("{PROBE_TYPES}", "probe_types.slang"),
    ("{PROBE_COMMON}", "probe_common.slang"),
    ("{RT_TYPES}", "rt_types.slang"),
    ("{RT_TRACE}", "rt_trace.slang"),
    ("{PARTICLE_TYPES}", "particle_types.slang"),
];

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

// The SSR resolve is the one post pass that reads the reflection-probe array,
// so it needs the same probe count the main pass bakes in.
const SLANG_PROBE_DEFINES: &[(&str, &str)] = &[("MAX_PROBES", "8")];

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

// One shader variant and the registers its DirectX root signature declares.
struct DxilAbi {
    file: &'static str,
    // Variant gates, injected as `#define <gate> 1` alongside DXIL_ABI.
    gates: &'static [&'static str],
    entry: &'static str,
    profile: &'static str,
    registers: &'static [(&'static str, &'static str)],
}

// The G-buffer pre-pass, shadow, ray-traced reflection and glass root
// signatures, from `src/directx/post/gbuffer.rs`, `post/rt_reflections.rs`,
// `glass.rs`, `init/pipelines.rs` and `resources.rs`. These
// layouts are engine-internal rather than a world-shader contract, but they are
// pinned for a sharper reason: the `.slang` files are shared with Metal and
// Vulkan, whose hosts bind the same declarations at entirely different slots,
// so an edit made on either of those platforms cannot see a DirectX root
// signature at all. `dx_crosscheck.sh` runs this script's DirectX branch, which
// is where such an edit gets caught.
const SLANG_DXIL_ENTRY_ABI: &[DxilAbi] = &[
    DxilAbi {
        file: "gbuffer_prepass.slang",
        gates: &["GB_STATIC"],
        entry: "gbuffer_prepass_vertex",
        profile: "vs_6_0",
        registers: &[("gb_view", "b0"), ("gb_model", "b1")],
    },
    DxilAbi {
        file: "gbuffer_prepass.slang",
        gates: &["GB_INSTANCED"],
        entry: "gbuffer_prepass_vertex_instanced",
        profile: "vs_6_0",
        registers: &[("gb_view", "b0"), ("instances", "t0")],
    },
    DxilAbi {
        file: "gbuffer_prepass.slang",
        gates: &["GB_SKINNED"],
        entry: "gbuffer_prepass_vertex_skinned",
        profile: "vs_6_0",
        registers: &[
            ("gb_view", "b0"),
            ("gb_model", "b1"),
            ("cur_joints", "t0"),
            ("prev_joints", "t1"),
        ],
    },
    DxilAbi {
        file: "gbuffer_prepass.slang",
        gates: &["GB_BINDLESS"],
        entry: "gbuffer_prepass_vertex_bindless",
        profile: "vs_6_0",
        registers: &[
            ("objid_cb", "b0"),
            ("gb_view", "b1"),
            ("objects", "t0"),
            ("prev_models", "t1"),
        ],
    },
    DxilAbi {
        file: "gbuffer_prepass.slang",
        gates: &["GB_FRAGMENT"],
        entry: "gbuffer_prepass_fragment",
        profile: "ps_6_0",
        registers: &[("gb_mat", "b0")],
    },
    DxilAbi {
        file: "gbuffer_prepass.slang",
        gates: &["GB_FRAGMENT_BINDLESS"],
        entry: "gbuffer_prepass_fragment_bindless",
        profile: "ps_6_0",
        registers: &[],
    },
    DxilAbi {
        file: "shadow.slang",
        gates: &["SHADOW_STATIC"],
        entry: "shadow_vertex_main",
        profile: "vs_6_0",
        registers: &[("push", "b0"), ("shadow_cb", "b1")],
    },
    DxilAbi {
        file: "shadow.slang",
        gates: &["SHADOW_SKINNED"],
        entry: "shadow_vertex_main_skinned",
        profile: "vs_6_0",
        registers: &[("push", "b0"), ("shadow_cb", "b1"), ("joints", "t0")],
    },
    DxilAbi {
        file: "shadow.slang",
        gates: &["SHADOW_BINDLESS"],
        entry: "shadow_vertex_bindless",
        profile: "vs_6_0",
        registers: &[
            ("objid_cb", "b0"),
            ("shadow_cb", "b1"),
            ("push", "b2"),
            ("objects", "t0"),
        ],
    },
    // Ray query needs shader model 6.5, above the 6.0 floor the rest of the
    // single-source shaders compile at.
    DxilAbi {
        file: "rt_reflections.slang",
        gates: &[],
        entry: "rt_reflections_fragment",
        profile: "ps_6_5",
        registers: RT_REFLECTIONS_REGISTERS,
    },
    DxilAbi {
        file: "rt_reflections.slang",
        gates: &["RT_TEXTURED"],
        entry: "rt_reflections_fragment",
        profile: "ps_6_5",
        registers: RT_REFLECTIONS_TEXTURED_REGISTERS,
    },
    DxilAbi {
        file: "glass.slang",
        gates: &[],
        entry: "glass_vertex",
        profile: "vs_6_0",
        registers: &[("view", "b0")],
    },
    DxilAbi {
        file: "glass.slang",
        gates: &[],
        entry: "glass_fragment",
        profile: "ps_6_0",
        registers: GLASS_REGISTERS,
    },
    DxilAbi {
        file: "glass.slang",
        gates: &["GLASS_RT"],
        entry: "glass_rt_fragment",
        profile: "ps_6_5",
        registers: GLASS_RT_REGISTERS,
    },
    DxilAbi {
        file: "glass.slang",
        gates: &["GLASS_RT", "RT_TEXTURED"],
        entry: "glass_rt_fragment",
        profile: "ps_6_5",
        registers: GLASS_RT_TEXTURED_REGISTERS,
    },
    // The raster remainder, from `src/directx/{particle,decal,line}.rs` and
    // `pipeline.rs`. Only the particle pair carries a `DXIL_ABI` block, and only
    // to swap the two constant buffers; the rest are here because slangc
    // assigns their registers from declaration order, which is a weaker
    // guarantee than a `register()` annotation and nothing else would catch a
    // release that changed it.
    DxilAbi {
        file: "particle.slang",
        gates: &[],
        entry: "particle_vertex",
        profile: "vs_6_0",
        registers: &[("view", "b0"), ("params", "b1"), ("pool", "t0")],
    },
    DxilAbi {
        file: "particle.slang",
        gates: &[],
        entry: "particle_fragment",
        profile: "ps_6_0",
        registers: &[("albedo_texture", "t1"), ("albedo_sampler", "s0")],
    },
    DxilAbi {
        file: "decal.slang",
        gates: &[],
        entry: "decal_vertex",
        profile: "vs_6_0",
        registers: &[("view", "b0"), ("params", "b1")],
    },
    DxilAbi {
        file: "decal.slang",
        gates: &[],
        entry: "decal_fragment",
        profile: "ps_6_0",
        registers: &[
            ("view", "b0"),
            ("params", "b1"),
            ("scene_depth", "t0"),
            ("decal_tex_texture", "t1"),
            ("decal_tex_sampler", "s0"),
        ],
    },
    DxilAbi {
        file: "line.slang",
        gates: &[],
        entry: "line_vertex",
        profile: "vs_6_0",
        registers: &[("view", "b0")],
    },
    DxilAbi {
        file: "line.slang",
        gates: &[],
        entry: "line_fragment",
        profile: "ps_6_0",
        registers: &[("view", "b0"), ("scene_depth", "t0")],
    },
    DxilAbi {
        file: "text.slang",
        gates: &[],
        entry: "text_vertex_main",
        profile: "vs_6_0",
        registers: &[("uni", "b0")],
    },
    DxilAbi {
        file: "text.slang",
        gates: &[],
        entry: "text_fragment_main",
        profile: "ps_6_0",
        registers: &[("atlas_texture", "t0"), ("atlas_sampler", "s0")],
    },
];

// The ray-traced reflection resolve's root signature, from
// `src/directx/post/rt_reflections.rs`. The probe cube array is remapped clear
// of the screen-space SRVs, which is what the hand-written HLSL used its
// PROBE_CUBES_REGISTER define for.
const RT_REFLECTIONS_REGISTERS: &[(&str, &str)] = &[
    ("rt_params", "b0"),
    ("probe_set", "b4"),
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
    ("screen_sampler", "s0"),
    ("cube_sampler", "s1"),
    ("probe_cube_sampler", "s3"),
];

const RT_REFLECTIONS_TEXTURED_REGISTERS: &[(&str, &str)] =
    &[("tex_pool", "t0, space1"), ("pool_sampler", "s2")];

// The glass root signatures, from `src/directx/glass.rs`. The base pass leaves
// the probe cubes at their default t7; the RT variant moves them to t20 because
// the array spans MAX_PROBES registers and the trace's SRVs sit at t4..t10.
const GLASS_REGISTERS: &[(&str, &str)] = &[
    ("view", "b0"),
    ("params", "b1"),
    ("probe_set", "b4"),
    ("scene_color", "t0"),
    ("scene_depth", "t1"),
    ("prefilter_cube", "t2"),
    ("planar_reflection", "t3"),
    ("probe_cubes", "t7"),
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
];

const GLASS_RT_TEXTURED_REGISTERS: &[(&str, &str)] =
    &[("tex_pool", "t0, space1"), ("pool_sampler", "s1")];

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
    SlangLibSpec {
        name: "gbuffer_prepass_vert.slang",
        file: "gbuffer_prepass.slang",
        entries: &["gbuffer_prepass_vertex"],
        defines: &[("GB_STATIC", "1"), ("METAL_BINDINGS", "1")],
    },
    SlangLibSpec {
        name: "gbuffer_prepass_vert_instanced.slang",
        file: "gbuffer_prepass.slang",
        entries: &["gbuffer_prepass_vertex_instanced"],
        defines: &[("GB_INSTANCED", "1"), ("METAL_BINDINGS", "1")],
    },
    SlangLibSpec {
        name: "gbuffer_prepass_vert_skinned.slang",
        file: "gbuffer_prepass.slang",
        entries: &["gbuffer_prepass_vertex_skinned"],
        defines: &[("GB_SKINNED", "1"), ("METAL_BINDINGS", "1")],
    },
    SlangLibSpec {
        name: "gbuffer_prepass_vert_bindless.slang",
        file: "gbuffer_prepass.slang",
        entries: &["gbuffer_prepass_vertex_bindless"],
        defines: &[("GB_BINDLESS", "1"), ("METAL_BINDINGS", "1")],
    },
    SlangLibSpec {
        name: "gbuffer_prepass_frag.slang",
        file: "gbuffer_prepass.slang",
        entries: &["gbuffer_prepass_fragment"],
        defines: &[("GB_FRAGMENT", "1"), ("METAL_BINDINGS", "1")],
    },
    SlangLibSpec {
        name: "gbuffer_prepass_frag_bindless.slang",
        file: "gbuffer_prepass.slang",
        entries: &["gbuffer_prepass_fragment_bindless"],
        defines: &[("GB_FRAGMENT_BINDLESS", "1"), ("METAL_BINDINGS", "1")],
    },
    SlangLibSpec {
        name: "shadow_vert.slang",
        file: "shadow.slang",
        entries: &["shadow_vertex_main"],
        defines: &[("SHADOW_STATIC", "1"), ("METAL_BINDINGS", "1")],
    },
    SlangLibSpec {
        name: "shadow_vert_skinned.slang",
        file: "shadow.slang",
        entries: &["shadow_vertex_main_skinned"],
        defines: &[("SHADOW_SKINNED", "1"), ("METAL_BINDINGS", "1")],
    },
    SlangLibSpec {
        name: "shadow_vert_bindless.slang",
        file: "shadow.slang",
        entries: &["shadow_vertex_bindless"],
        defines: &[("SHADOW_BINDLESS", "1"), ("METAL_BINDINGS", "1")],
    },
    SlangLibSpec {
        name: "fullscreen_vert.slang",
        file: "fullscreen.slang",
        entries: &["fullscreen_vertex"],
        defines: &[],
    },
    SlangLibSpec {
        name: "taa_frag.slang",
        file: "taa.slang",
        entries: &["taa_fragment_main"],
        defines: &[],
    },
    SlangLibSpec {
        name: "bloom_prefilter.slang",
        file: "bloom.slang",
        entries: &["bloom_prefilter_fragment"],
        defines: &[("BLOOM_PREFILTER", "1")],
    },
    SlangLibSpec {
        name: "bloom_downsample.slang",
        file: "bloom.slang",
        entries: &["bloom_downsample_fragment"],
        defines: &[("BLOOM_DOWNSAMPLE", "1")],
    },
    SlangLibSpec {
        name: "bloom_upsample.slang",
        file: "bloom.slang",
        entries: &["bloom_upsample_fragment"],
        defines: &[("BLOOM_UPSAMPLE", "1")],
    },
    SlangLibSpec {
        name: "composite_frag.slang",
        file: "composite.slang",
        entries: &["composite_fragment"],
        defines: &[],
    },
    SlangLibSpec {
        name: "ssao_kernel.slang",
        file: "ssao.slang",
        entries: &["ssao_kernel_fragment"],
        defines: &[("SSAO_KERNEL", "1")],
    },
    SlangLibSpec {
        name: "ssao_blur.slang",
        file: "ssao.slang",
        entries: &["ssao_blur_fragment"],
        defines: &[("SSAO_BLUR", "1")],
    },
    SlangLibSpec {
        name: "ssr_resolve.slang",
        file: "ssr.slang",
        entries: &["ssr_resolve_fragment"],
        defines: SLANG_PROBE_DEFINES,
    },
    SlangLibSpec {
        name: "ssgi_gather.slang",
        file: "ssgi.slang",
        entries: &["ssgi_gather_fragment"],
        defines: &[("SSGI_GATHER", "1")],
    },
    SlangLibSpec {
        name: "ssgi_composite.slang",
        file: "ssgi.slang",
        entries: &["ssgi_composite_fragment"],
        defines: &[("SSGI_COMPOSITE", "1")],
    },
    SlangLibSpec {
        name: "reflection_blur.slang",
        file: "reflection.slang",
        entries: &["reflection_blur_fragment"],
        defines: &[("REFLECTION_BLUR", "1")],
    },
    SlangLibSpec {
        name: "reflection_composite.slang",
        file: "reflection.slang",
        entries: &["reflection_composite_fragment"],
        defines: &[("REFLECTION_COMPOSITE", "1")],
    },
    SlangLibSpec {
        name: "fog_froxel.slang",
        file: "fog.slang",
        entries: &["fog_froxel_kernel"],
        defines: &[("FOG_FROXEL", "1")],
    },
    SlangLibSpec {
        name: "fog_frag.slang",
        file: "fog.slang",
        entries: &["fog_fragment"],
        defines: &[("USE_MSAA", "0")],
    },
    SlangLibSpec {
        name: "auto_exposure_build.slang",
        file: "auto_exposure.slang",
        entries: &["histogram_build"],
        defines: &[("AE_BUILD", "1"), ("METAL_BINDINGS", "1")],
    },
    SlangLibSpec {
        name: "auto_exposure_average.slang",
        file: "auto_exposure.slang",
        entries: &["histogram_average"],
        defines: &[("AE_AVERAGE", "1"), ("METAL_BINDINGS", "1")],
    },
    SlangLibSpec {
        name: "particle_simulate.slang",
        file: "particle_simulate.slang",
        entries: &["particle_simulate"],
        defines: &[("METAL_BINDINGS", "1")],
    },
    SlangLibSpec {
        name: "particle_vert.slang",
        file: "particle.slang",
        entries: &["particle_vertex"],
        defines: &[("METAL_BINDINGS", "1")],
    },
    SlangLibSpec {
        name: "particle_frag.slang",
        file: "particle.slang",
        entries: &["particle_fragment"],
        defines: &[("METAL_BINDINGS", "1")],
    },
    SlangLibSpec {
        name: "decal_vert.slang",
        file: "decal.slang",
        entries: &["decal_vertex"],
        defines: &[],
    },
    SlangLibSpec {
        name: "decal_frag.slang",
        file: "decal.slang",
        entries: &["decal_fragment"],
        defines: &[("USE_MSAA", "0")],
    },
    SlangLibSpec {
        name: "line_vert.slang",
        file: "line.slang",
        entries: &["line_vertex"],
        defines: &[],
    },
    SlangLibSpec {
        name: "line_frag.slang",
        file: "line.slang",
        entries: &["line_fragment"],
        defines: &[("USE_MSAA", "0")],
    },
    SlangLibSpec {
        name: "text_vert.slang",
        file: "text.slang",
        entries: &["text_vertex_main"],
        defines: &[("METAL_BINDINGS", "1")],
    },
    SlangLibSpec {
        name: "text_frag.slang",
        file: "text.slang",
        entries: &["text_fragment_main"],
        defines: &[("METAL_BINDINGS", "1")],
    },
    SlangLibSpec {
        name: "rt_reflections_frag.slang",
        file: "rt_reflections.slang",
        entries: &["rt_reflections_fragment"],
        defines: SLANG_RT_DEFINES,
    },
    SlangLibSpec {
        name: "rt_reflections_frag_textured.slang",
        file: "rt_reflections.slang",
        entries: &["rt_reflections_fragment"],
        defines: SLANG_RT_TEXTURED_DEFINES,
    },
    SlangLibSpec {
        name: "glass_vert.slang",
        file: "glass.slang",
        entries: &["glass_vertex"],
        defines: SLANG_GLASS_DEFINES,
    },
    SlangLibSpec {
        name: "glass_frag.slang",
        file: "glass.slang",
        entries: &["glass_fragment"],
        defines: SLANG_GLASS_DEFINES,
    },
    SlangLibSpec {
        name: "glass_frag_rt.slang",
        file: "glass.slang",
        entries: &["glass_rt_fragment"],
        defines: SLANG_GLASS_RT_DEFINES,
    },
    SlangLibSpec {
        name: "glass_frag_rt_textured.slang",
        file: "glass.slang",
        entries: &["glass_rt_fragment"],
        defines: SLANG_GLASS_RT_TEXTURED_DEFINES,
    },
];

// The ray-traced families. Both bake the Metal binding layout in; the textured
// variants additionally read the bindless pool, so they take its capacity.
const SLANG_RT_DEFINES: &[(&str, &str)] = &[("METAL_ABI", "1"), ("MAX_PROBES", "8")];
const SLANG_RT_TEXTURED_DEFINES: &[(&str, &str)] = &[
    ("METAL_ABI", "1"),
    ("RT_TEXTURED", "1"),
    ("POOL_SIZE", "1024"),
    ("MAX_PROBES", "8"),
];
const SLANG_GLASS_DEFINES: &[(&str, &str)] = &[("METAL_ABI", "1"), ("MAX_PROBES", "8")];
const SLANG_GLASS_RT_DEFINES: &[(&str, &str)] =
    &[("METAL_ABI", "1"), ("GLASS_RT", "1"), ("MAX_PROBES", "8")];
const SLANG_GLASS_RT_TEXTURED_DEFINES: &[(&str, &str)] = &[
    ("METAL_ABI", "1"),
    ("GLASS_RT", "1"),
    ("RT_TEXTURED", "1"),
    ("POOL_SIZE", "1024"),
    ("MAX_PROBES", "8"),
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
            &SlangShaders {
                dir: &slang_dir,
                fragments: SLANG_SHADER_FRAGMENTS,
                specs: SLANG_METAL_LIBS,
            },
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
    let source = slang_source(slang_dir, "main_bindless.slang");
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
    let source = slang::inject_defines(
        &slang_source(slang_dir, "main_bindless.slang"),
        SLANG_DXIL_MAIN_DEFINES,
    );
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    // The two stages compile separately and each drops the resources it does
    // not read, so a register only has to hold in the stage that declares it.
    let mut emitted = String::new();
    for (entry, profile) in [
        ("vertex_main_bindless", "vs_6_0"),
        ("fragment_main_bindless", "ps_6_0"),
    ] {
        emitted.push_str(&emit_dxil_hlsl(
            &out_dir,
            "main_bindless_abi_check.slang",
            &source,
            entry,
            profile,
        ));
    }
    for (param, register) in DXIL_ABI_REGISTERS {
        assert_dxil_register(
            &emitted,
            param,
            register,
            "The DirectX bindless binding layout is frozen (world Shader assets build PSOs \
             against the same root signature); fix the .slang declarations or the root \
             signature in src/directx/init/pipelines.rs before shipping.",
        );
    }
    assert_slang_dxil_entry_abi(slang_dir, &out_dir);
}

// The same check over every other single-source family. Each variant compiles
// alone, so a register only has to hold in the entry that declares it. A file
// with no `DXIL_ABI` block is still worth a row: `DXIL_ABI` is simply inert
// there and the check reads back slangc's declaration-order assignment.
fn assert_slang_dxil_entry_abi(slang_dir: &std::path::Path, out_dir: &std::path::Path) {
    for abi in SLANG_DXIL_ENTRY_ABI {
        let mut defines: Vec<(&str, &str)> = vec![("DXIL_ABI", "1")];
        defines.extend(abi.gates.iter().map(|gate| (*gate, "1")));
        let source = slang::inject_defines(&slang_source(slang_dir, abi.file), &defines);
        let emitted = emit_dxil_hlsl(out_dir, abi.file, &source, abi.entry, abi.profile);
        for (param, register) in abi.registers {
            assert_dxil_register(
                &emitted,
                param,
                register,
                "Fix the .slang declaration or the matching root signature under \
                 src/directx before shipping.",
            );
        }
    }
}

// One entry point's emitted HLSL, for reading register annotations back out.
fn emit_dxil_hlsl(
    out_dir: &std::path::Path,
    file_name: &str,
    source: &str,
    entry: &str,
    profile: &'static str,
) -> String {
    let job = slang::SlangJob {
        source,
        file_name,
        entries: &[entry],
        target: slang::SlangTarget::Hlsl(profile),
    };
    let hlsl = slang::compile(&job, out_dir)
        .unwrap_or_else(|e| panic!("slang DXIL ABI check ({entry}): {e}"));
    String::from_utf8_lossy(&hlsl).into_owned()
}

fn assert_dxil_register(emitted: &str, param: &str, register: &str, remedy: &str) {
    // Emitted parameter names carry a `_<n>` suffix (e.g. `view_cb_0`).
    let expected = format!("{param}_0 : register({register})");
    // Resource arrays emit as `name_0[int(N)] : register(...)`.
    let expected_array = format!("{param}_0[");
    assert!(
        emitted.contains(&expected)
            || emitted.lines().any(
                |l| l.contains(&expected_array) && l.contains(&format!("register({register})"))
            ),
        "slang DXIL ABI drifted: expected `{param}` at register({register}). {remedy}"
    );
}

// One `.slang` file with its shared fragments spliced in, matching what
// `crate::slang_source::assemble` produces at run time. The ABI checks compile
// real source, so they need the same assembly the renderer gets.
fn slang_source(slang_dir: &std::path::Path, file: &str) -> String {
    let read = |name: &str| {
        std::fs::read_to_string(slang_dir.join(name)).unwrap_or_else(|e| panic!("read {name}: {e}"))
    };
    let mut source = read(file);
    for (marker, fragment) in SLANG_SHADER_FRAGMENTS {
        if source.contains(marker) {
            source = source.replace(marker, &read(fragment));
        }
    }
    source
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
