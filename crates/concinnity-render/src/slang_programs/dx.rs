// Single-source engine shader programs for the DirectX backend.
//
// Each program compiles a `.slang` file under `src/shaders/` (the
// backend-neutral single-source directory) to a signed DXIL container at
// renderer init by invoking slangc, cached in the content-addressed shader
// cache exactly like the HLSL programs in `builtins`. slangc resolves dxil.dll
// itself, so the containers it emits are already signed for D3D12.
//
// The bindless main pair compiles the source's `DXIL_ABI` block, whose
// register() annotations reproduce the bindless main root signature in
// `init/pipelines.rs` slot for slot: that layout is a contract, since a world
// Shader asset builds its own PSO against the same root signature (see
// world_shaders.rs). `assert_slang_dxil_abi` in build.rs locks it. The two
// compute kernels need no ABI block: slangc assigns b0/t0/u0 from declaration
// order, which is what their root signatures already bind.
//
// The G-buffer pre-pass and shadow families take the same `DXIL_ABI` block for
// a weaker reason: nothing outside the engine binds them, but their root
// signatures hand the same declarations entirely different slots than the Metal
// and Vulkan hosts do, so the shared source cannot carry one set of registers
// for all three. `assert_slang_dxil_abi` in build.rs locks every one of them,
// which is what a macOS edit to the shared file runs into under
// `dx_crosscheck.sh`.
//
// The fullscreen post passes need no ABI block either, and for a stronger
// reason than the compute kernels: nothing outside the engine binds them at
// all. slangc splits each top-level `Sampler2D` into a `Texture2D` + a
// `SamplerState` and numbers both from declaration order, so a pass with N
// sources lands on t0..tN-1 *and* s0..sN-1 -- where the hand HLSL declared one
// sampler for all of them. The root signatures name the samplers they hand out
// (a pass's static samplers are the same descriptor repeated).
//
// Declaration order is what the root signatures follow, and it is not always
// the order the hand HLSL used: the SSR resolve's probe cube array lands at t4
// (not the hand shader's t7) with its `ProbeSet` at b1 (not b4), and the
// reflection composite reads scene / G-buffer / roughness at t1 / t2 / t3 where
// the HLSL had roughness first.
//
// These are shader model 6.0 rather than the FXC path's 5.1: the bindless pool
// index is non-uniform across a fragment wave, and `NonUniformResourceIndex`
// is an SM 6.0 construct.

/// One DXIL program: which shader file, which entry point, at which
/// shader-model profile, under which variant defines.
pub struct SlangProgram {
    /// File name under this crate's `src/shaders/` for the hot-reload disk-first resolve;
    /// also the embedded fallback's origin and the name slangc diagnostics use.
    pub file: &'static str,
    /// Entry point compiled out of that file.
    pub entry: &'static str,
    /// Shader-model profile for the DXIL container (stage + feature floor).
    pub profile: &'static str,
    /// Diagnostic label (compile errors + cache miss logs + export report).
    pub label: &'static str,
    /// Variant defines injected as `#define` lines ahead of the source.
    pub defines: &'static [(&'static str, &'static str)],
}

// The bindless main pair's variant defines. `MAX_PROBES` sizes the probe cube
// array the root signature's descriptor table covers; `probe_cube_count_matches`
// locks it to the host constant. `POOL_SIZE` is deliberately absent: the DXIL
// pool is an unbounded array, so the shader never over-declares the per-frame
// descriptor region the host actually wrote.
const MAIN_DEFINES: &[(&str, &str)] = &[("DXIL_ABI", "1"), ("MAX_PROBES", "8")];

// The SSR resolve reads the probe array but none of the texture pool, so it
// takes the probe count alone. Same lock as `MAIN_DEFINES`.
// `SPLIT_PROBE_SAMPLER` declares that array as a texture array plus one
// sampler: D3D12 binds a shader sampler *array* only through a descriptor
// table, so the combined form Metal and Vulkan use cannot be covered by static
// samplers here (see the declaration in `ssr.slang`).
const SSR_DEFINES: &[(&str, &str)] = &[("MAX_PROBES", "8"), ("SPLIT_PROBE_SAMPLER", "1")];

/// `vertex_main_bindless` from `main_bindless.slang`.
pub static MAIN_BINDLESS_VERT: SlangProgram = SlangProgram {
    file: "main_bindless.slang",
    entry: "vertex_main_bindless",
    profile: "vs_6_0",
    label: "vert_bindless.slang",
    defines: MAIN_DEFINES,
};
/// `fragment_main_bindless` from `main_bindless.slang`.
pub static MAIN_BINDLESS_FRAG: SlangProgram = SlangProgram {
    file: "main_bindless.slang",
    entry: "fragment_main_bindless",
    profile: "ps_6_0",
    label: "frag_bindless.slang",
    defines: MAIN_DEFINES,
};
/// `light_cull_kernel` from `light_cull.slang`.
pub static LIGHT_CULL: SlangProgram = SlangProgram {
    file: "light_cull.slang",
    entry: "light_cull_kernel",
    profile: "cs_6_0",
    label: "light_cull.slang",
    defines: &[],
};
/// `hiz_init_single` from `hiz_build.slang`.
pub static HIZ_INIT_SINGLE: SlangProgram = SlangProgram {
    file: "hiz_build.slang",
    entry: "hiz_init_single",
    profile: "cs_6_0",
    label: "hiz_init_single.slang",
    defines: &[("HIZ_INIT_SINGLE", "1")],
};
/// `hiz_init_msaa` from `hiz_build.slang`.
pub static HIZ_INIT_MSAA: SlangProgram = SlangProgram {
    file: "hiz_build.slang",
    entry: "hiz_init_msaa",
    profile: "cs_6_0",
    label: "hiz_init_msaa.slang",
    defines: &[("HIZ_INIT_MSAA", "1")],
};
/// `hiz_downsample` from `hiz_build.slang`.
pub static HIZ_DOWNSAMPLE: SlangProgram = SlangProgram {
    file: "hiz_build.slang",
    entry: "hiz_downsample",
    profile: "cs_6_0",
    label: "hiz_downsample.slang",
    defines: &[("HIZ_DOWNSAMPLE", "1")],
};

// The G-buffer pre-pass and shadow families. Every entry is its own program so
// each variant declares exactly the resources its root signature binds; the
// `DXIL_ABI` gate pins those registers to the signatures in `post/gbuffer.rs`,
// `init/pipelines.rs` and `resources.rs`.
const GB_STATIC: &[(&str, &str)] = &[("GB_STATIC", "1"), ("DXIL_ABI", "1")];
const GB_INSTANCED: &[(&str, &str)] = &[("GB_INSTANCED", "1"), ("DXIL_ABI", "1")];
const GB_SKINNED: &[(&str, &str)] = &[("GB_SKINNED", "1"), ("DXIL_ABI", "1")];
const GB_BINDLESS: &[(&str, &str)] = &[("GB_BINDLESS", "1"), ("DXIL_ABI", "1")];
const GB_FRAGMENT: &[(&str, &str)] = &[("GB_FRAGMENT", "1"), ("DXIL_ABI", "1")];
const GB_FRAGMENT_BINDLESS: &[(&str, &str)] = &[("GB_FRAGMENT_BINDLESS", "1"), ("DXIL_ABI", "1")];
const SHADOW_STATIC: &[(&str, &str)] = &[("SHADOW_STATIC", "1"), ("DXIL_ABI", "1")];
const SHADOW_SKINNED: &[(&str, &str)] = &[("SHADOW_SKINNED", "1"), ("DXIL_ABI", "1")];
const SHADOW_BINDLESS: &[(&str, &str)] = &[("SHADOW_BINDLESS", "1"), ("DXIL_ABI", "1")];

/// `gbuffer_prepass_vertex` from `gbuffer_prepass.slang`.
pub static GBUFFER_PREPASS_VERT: SlangProgram = SlangProgram {
    file: "gbuffer_prepass.slang",
    entry: "gbuffer_prepass_vertex",
    profile: "vs_6_0",
    label: "gbuffer_prepass_vert.slang",
    defines: GB_STATIC,
};
/// `gbuffer_prepass_vertex_instanced` from `gbuffer_prepass.slang`.
pub static GBUFFER_PREPASS_VERT_INSTANCED: SlangProgram = SlangProgram {
    file: "gbuffer_prepass.slang",
    entry: "gbuffer_prepass_vertex_instanced",
    profile: "vs_6_0",
    label: "gbuffer_prepass_vert_instanced.slang",
    defines: GB_INSTANCED,
};
/// `gbuffer_prepass_vertex_skinned` from `gbuffer_prepass.slang`.
pub static GBUFFER_PREPASS_VERT_SKINNED: SlangProgram = SlangProgram {
    file: "gbuffer_prepass.slang",
    entry: "gbuffer_prepass_vertex_skinned",
    profile: "vs_6_0",
    label: "gbuffer_prepass_vert_skinned.slang",
    defines: GB_SKINNED,
};
/// `gbuffer_prepass_vertex_bindless` from `gbuffer_prepass.slang`.
pub static GBUFFER_BINDLESS_VERT: SlangProgram = SlangProgram {
    file: "gbuffer_prepass.slang",
    entry: "gbuffer_prepass_vertex_bindless",
    profile: "vs_6_0",
    label: "gbuffer_prepass_vert_bindless.slang",
    defines: GB_BINDLESS,
};
/// `gbuffer_prepass_fragment` from `gbuffer_prepass.slang`.
pub static GBUFFER_PREPASS_FRAG: SlangProgram = SlangProgram {
    file: "gbuffer_prepass.slang",
    entry: "gbuffer_prepass_fragment",
    profile: "ps_6_0",
    label: "gbuffer_prepass_frag.slang",
    defines: GB_FRAGMENT,
};
/// `gbuffer_prepass_fragment_bindless` from `gbuffer_prepass.slang`.
pub static GBUFFER_BINDLESS_FRAG: SlangProgram = SlangProgram {
    file: "gbuffer_prepass.slang",
    entry: "gbuffer_prepass_fragment_bindless",
    profile: "ps_6_0",
    label: "gbuffer_prepass_frag_bindless.slang",
    defines: GB_FRAGMENT_BINDLESS,
};
/// `shadow_vertex_main` from `shadow.slang`.
pub static SHADOW_VERT: SlangProgram = SlangProgram {
    file: "shadow.slang",
    entry: "shadow_vertex_main",
    profile: "vs_6_0",
    label: "shadow_vert.slang",
    defines: SHADOW_STATIC,
};
/// `shadow_vertex_main_skinned` from `shadow.slang`.
pub static SKINNED_SHADOW_VERT: SlangProgram = SlangProgram {
    file: "shadow.slang",
    entry: "shadow_vertex_main_skinned",
    profile: "vs_6_0",
    label: "shadow_vert_skinned.slang",
    defines: SHADOW_SKINNED,
};
/// `shadow_vertex_bindless` from `shadow.slang`.
pub static SHADOW_BINDLESS_VERT: SlangProgram = SlangProgram {
    file: "shadow.slang",
    entry: "shadow_vertex_bindless",
    profile: "vs_6_0",
    label: "shadow_vert_bindless.slang",
    defines: SHADOW_BINDLESS,
};

// The fog family, auto-exposure and the particle simulation kernel. None takes
// an ABI block: slangc assigns b/t/u/s from declaration order here, which is
// exactly what the root signatures in `fog.rs`, `auto_exposure.rs` and
// `particle.rs` already bind. The fog fragment's vertex half is
// `FULLSCREEN_VERT` below -- the last per-family fullscreen vertex to retire.
//
// Two DirectX-only gates ride these. `DXIL_SPLIT` declares the cascade array as
// a texture plus a comparison sampler rather than the combined
// `Sampler2DArray` the other two backends bind, because `SampleCmp` on a
// combined type mis-lowers on DXIL -- the same split `main_bindless.slang`
// carries for the same reason. `DXIL_STAGE_PACKING` makes the fog fragment
// declare the fullscreen varying it never reads, because D3D links the two
// stages by semantic *and* register and fog is the one post fragment that takes
// no varying; see the declaration in `fog.slang`.
const FOG_FROXEL_DEFINES: &[(&str, &str)] = &[("FOG_FROXEL", "1"), ("DXIL_SPLIT", "1")];

/// `fog_froxel_kernel` from `fog.slang`.
pub static FOG_FROXEL: SlangProgram = SlangProgram {
    file: "fog.slang",
    entry: "fog_froxel_kernel",
    profile: "cs_6_0",
    label: "fog_froxel.slang",
    defines: FOG_FROXEL_DEFINES,
};

// The fog fragment declares its depth source by the main pass's sample count,
// which is a host difference rather than a target one, so it takes two
// programs the way the Hi-Z init pair does: the caller picks by MSAA state and
// the export-time precompile leaves a bundle warm for either.
/// `fog_fragment` from `fog.slang`.
pub static FOG_FRAG: SlangProgram = SlangProgram {
    file: "fog.slang",
    entry: "fog_fragment",
    profile: "ps_6_0",
    label: "fog_frag.slang",
    defines: &[("USE_MSAA", "0"), ("DXIL_STAGE_PACKING", "1")],
};
/// `fog_fragment` from `fog.slang`.
pub static FOG_FRAG_MSAA: SlangProgram = SlangProgram {
    file: "fog.slang",
    entry: "fog_fragment",
    profile: "ps_6_0",
    label: "fog_frag_msaa.slang",
    defines: &[("USE_MSAA", "1"), ("DXIL_STAGE_PACKING", "1")],
};

/// `histogram_build` from `auto_exposure.slang`.
pub static AUTO_EXPOSURE_BUILD: SlangProgram = SlangProgram {
    file: "auto_exposure.slang",
    entry: "histogram_build",
    profile: "cs_6_0",
    label: "auto_exposure_build.slang",
    defines: &[("AE_BUILD", "1")],
};
/// `histogram_average` from `auto_exposure.slang`.
pub static AUTO_EXPOSURE_AVERAGE: SlangProgram = SlangProgram {
    file: "auto_exposure.slang",
    entry: "histogram_average",
    profile: "cs_6_0",
    label: "auto_exposure_average.slang",
    defines: &[("AE_AVERAGE", "1")],
};

// The RT skinning kernel takes the same shader model as the RT reflection
// resolve it feeds; its own body needs no SM 6.5 feature.
/// `rt_skin` from `rt_skin.slang`.
pub static RT_SKIN: SlangProgram = SlangProgram {
    file: "rt_skin.slang",
    entry: "rt_skin",
    profile: "cs_6_5",
    label: "rt_skin.slang",
    defines: &[],
};

/// `particle_simulate` from `particle_simulate.slang`.
pub static PARTICLE_SIMULATE: SlangProgram = SlangProgram {
    file: "particle_simulate.slang",
    entry: "particle_simulate",
    profile: "cs_6_0",
    label: "particle_simulate.slang",
    defines: &[],
};

// The fullscreen-triangle vertex stage every post pass pairs with; one module
// serves them all, retiring the four per-family `*_vert.hlsl` copies.
/// `fullscreen_vertex` from `fullscreen.slang`.
pub static FULLSCREEN_VERT: SlangProgram = SlangProgram {
    file: "fullscreen.slang",
    entry: "fullscreen_vertex",
    profile: "vs_6_0",
    label: "fullscreen_vert.slang",
    defines: &[],
};
/// `taa_fragment_main` from `taa.slang`.
pub static TAA_FRAG: SlangProgram = SlangProgram {
    file: "taa.slang",
    entry: "taa_fragment_main",
    profile: "ps_6_0",
    label: "taa_frag.slang",
    defines: &[],
};
/// `bloom_prefilter_fragment` from `bloom.slang`.
pub static BLOOM_PREFILTER: SlangProgram = SlangProgram {
    file: "bloom.slang",
    entry: "bloom_prefilter_fragment",
    profile: "ps_6_0",
    label: "bloom_prefilter.slang",
    defines: &[("BLOOM_PREFILTER", "1")],
};
/// `bloom_downsample_fragment` from `bloom.slang`.
pub static BLOOM_DOWNSAMPLE: SlangProgram = SlangProgram {
    file: "bloom.slang",
    entry: "bloom_downsample_fragment",
    profile: "ps_6_0",
    label: "bloom_downsample.slang",
    defines: &[("BLOOM_DOWNSAMPLE", "1")],
};
/// `bloom_upsample_fragment` from `bloom.slang`.
pub static BLOOM_UPSAMPLE: SlangProgram = SlangProgram {
    file: "bloom.slang",
    entry: "bloom_upsample_fragment",
    profile: "ps_6_0",
    label: "bloom_upsample.slang",
    defines: &[("BLOOM_UPSAMPLE", "1")],
};
/// `composite_fragment` from `composite.slang`.
pub static COMPOSITE_FRAG: SlangProgram = SlangProgram {
    file: "composite.slang",
    entry: "composite_fragment",
    profile: "ps_6_0",
    label: "composite_frag.slang",
    defines: &[],
};
/// `ssao_kernel_fragment` from `ssao.slang`.
pub static SSAO_KERNEL: SlangProgram = SlangProgram {
    file: "ssao.slang",
    entry: "ssao_kernel_fragment",
    profile: "ps_6_0",
    label: "ssao_kernel.slang",
    defines: &[("SSAO_KERNEL", "1")],
};
/// `ssao_blur_fragment` from `ssao.slang`.
pub static SSAO_BLUR: SlangProgram = SlangProgram {
    file: "ssao.slang",
    entry: "ssao_blur_fragment",
    profile: "ps_6_0",
    label: "ssao_blur.slang",
    defines: &[("SSAO_BLUR", "1")],
};
/// `ssr_resolve_fragment` from `ssr.slang`.
pub static SSR_RESOLVE: SlangProgram = SlangProgram {
    file: "ssr.slang",
    entry: "ssr_resolve_fragment",
    profile: "ps_6_0",
    label: "ssr_resolve.slang",
    defines: SSR_DEFINES,
};
/// `ssgi_gather_fragment` from `ssgi.slang`.
pub static SSGI_GATHER: SlangProgram = SlangProgram {
    file: "ssgi.slang",
    entry: "ssgi_gather_fragment",
    profile: "ps_6_0",
    label: "ssgi_gather.slang",
    defines: &[("SSGI_GATHER", "1")],
};
/// `ssgi_composite_fragment` from `ssgi.slang`.
pub static SSGI_COMPOSITE: SlangProgram = SlangProgram {
    file: "ssgi.slang",
    entry: "ssgi_composite_fragment",
    profile: "ps_6_0",
    label: "ssgi_composite.slang",
    defines: &[("SSGI_COMPOSITE", "1")],
};
/// `reflection_blur_fragment` from `reflection.slang`.
pub static REFLECTION_BLUR: SlangProgram = SlangProgram {
    file: "reflection.slang",
    entry: "reflection_blur_fragment",
    profile: "ps_6_0",
    label: "reflection_blur.slang",
    defines: &[("REFLECTION_BLUR", "1")],
};
/// `reflection_composite_fragment` from `reflection.slang`.
pub static REFLECTION_COMPOSITE: SlangProgram = SlangProgram {
    file: "reflection.slang",
    entry: "reflection_composite_fragment",
    profile: "ps_6_0",
    label: "reflection_composite.slang",
    defines: &[("REFLECTION_COMPOSITE", "1")],
};

// The ray-traced reflection resolve and the glass family. Both take the
// `DXIL_ABI` block for the reason the pre-pass family does: their root
// signatures in `post/rt_reflections.rs` and `glass.rs` hand the same
// declarations entirely different slots than the Metal and Vulkan hosts do, and
// `assert_slang_dxil_abi` in build.rs locks every one of them. No `POOL_SIZE`:
// the DXIL pool is an unbounded array, as it is for the bindless main pass.
//
// The three ray-query entries are `*_6_5`, above the 6.0 floor the rest of the
// single-source shaders compile at: `RayQuery` is an SM 6.5 construct. Nothing
// else here needs it, so the glass vertex and the base fragment stay at 6.0.
//
// `USE_MSAA` is a host difference rather than a target one -- the glass fragment
// declares its depth source by the main pass's sample count -- so each glass
// fragment comes as a pair the way the fog fragment does, and the caller picks
// by MSAA state. The vertex stage reads no depth and takes neither.
/// `rt_reflections_fragment` from `rt_reflections.slang`.
pub static RT_REFLECTIONS_FRAG: SlangProgram = SlangProgram {
    file: "rt_reflections.slang",
    entry: "rt_reflections_fragment",
    profile: "ps_6_5",
    label: "rt_reflections.slang",
    defines: &[("DXIL_ABI", "1"), ("MAX_PROBES", "8")],
};
/// `rt_reflections_fragment` from `rt_reflections.slang`.
pub static RT_REFLECTIONS_FRAG_TEXTURED: SlangProgram = SlangProgram {
    file: "rt_reflections.slang",
    entry: "rt_reflections_fragment",
    profile: "ps_6_5",
    label: "rt_reflections_textured.slang",
    defines: &[("DXIL_ABI", "1"), ("RT_TEXTURED", "1"), ("MAX_PROBES", "8")],
};

// One vertex stage for both glass pipelines: the base pass and the ray-traced
// one differ only in where the reflection comes from, and both root signatures
// put the transparent view CBV at b0.
/// `glass_vertex` from `glass.slang`.
pub static GLASS_VERT: SlangProgram = SlangProgram {
    file: "glass.slang",
    entry: "glass_vertex",
    profile: "vs_6_0",
    label: "glass_vert.slang",
    defines: &[("DXIL_ABI", "1"), ("MAX_PROBES", "8")],
};
/// `glass_fragment` from `glass.slang`.
pub static GLASS_FRAG: SlangProgram = SlangProgram {
    file: "glass.slang",
    entry: "glass_fragment",
    profile: "ps_6_0",
    label: "glass_frag.slang",
    defines: &[("DXIL_ABI", "1"), ("MAX_PROBES", "8"), ("USE_MSAA", "0")],
};
/// `glass_fragment` from `glass.slang`.
pub static GLASS_FRAG_MSAA: SlangProgram = SlangProgram {
    file: "glass.slang",
    entry: "glass_fragment",
    profile: "ps_6_0",
    label: "glass_frag_msaa.slang",
    defines: &[("DXIL_ABI", "1"), ("MAX_PROBES", "8"), ("USE_MSAA", "1")],
};
/// `glass_rt_fragment` from `glass.slang`.
pub static GLASS_RT_FRAG: SlangProgram = SlangProgram {
    file: "glass.slang",
    entry: "glass_rt_fragment",
    profile: "ps_6_5",
    label: "glass_frag_rt.slang",
    defines: &[
        ("DXIL_ABI", "1"),
        ("GLASS_RT", "1"),
        ("MAX_PROBES", "8"),
        ("USE_MSAA", "0"),
    ],
};
/// `glass_rt_fragment` from `glass.slang`.
pub static GLASS_RT_FRAG_MSAA: SlangProgram = SlangProgram {
    file: "glass.slang",
    entry: "glass_rt_fragment",
    profile: "ps_6_5",
    label: "glass_frag_rt_msaa.slang",
    defines: &[
        ("DXIL_ABI", "1"),
        ("GLASS_RT", "1"),
        ("MAX_PROBES", "8"),
        ("USE_MSAA", "1"),
    ],
};
/// `glass_rt_fragment` from `glass.slang`.
pub static GLASS_RT_FRAG_TEXTURED: SlangProgram = SlangProgram {
    file: "glass.slang",
    entry: "glass_rt_fragment",
    profile: "ps_6_5",
    label: "glass_frag_rt_textured.slang",
    defines: &[
        ("DXIL_ABI", "1"),
        ("GLASS_RT", "1"),
        ("RT_TEXTURED", "1"),
        ("MAX_PROBES", "8"),
        ("USE_MSAA", "0"),
    ],
};
/// `glass_rt_fragment` from `glass.slang`.
pub static GLASS_RT_FRAG_TEXTURED_MSAA: SlangProgram = SlangProgram {
    file: "glass.slang",
    entry: "glass_rt_fragment",
    profile: "ps_6_5",
    label: "glass_frag_rt_textured_msaa.slang",
    defines: &[
        ("DXIL_ABI", "1"),
        ("GLASS_RT", "1"),
        ("RT_TEXTURED", "1"),
        ("MAX_PROBES", "8"),
        ("USE_MSAA", "1"),
    ],
};

const GLASS_MESH_DEFINES: &[(&str, &str)] =
    &[("DXIL_ABI", "1"), ("MAX_PROBES", "8"), ("USE_MSAA", "0")];
const GLASS_MESH_MSAA_DEFINES: &[(&str, &str)] =
    &[("DXIL_ABI", "1"), ("MAX_PROBES", "8"), ("USE_MSAA", "1")];
const GLASS_MESH_TEXTURED_DEFINES: &[(&str, &str)] = &[
    ("DXIL_ABI", "1"),
    ("RT_TEXTURED", "1"),
    ("MAX_PROBES", "8"),
    ("USE_MSAA", "0"),
];
const GLASS_MESH_TEXTURED_MSAA_DEFINES: &[(&str, &str)] = &[
    ("DXIL_ABI", "1"),
    ("RT_TEXTURED", "1"),
    ("MAX_PROBES", "8"),
    ("USE_MSAA", "1"),
];

// The see-through glass MESH family, the transparent pass's third producer.
// Ray-traced only -- the per-pixel trace is what makes the mesh see-through
// rather than the opaque reflective glass the main pass draws -- so there is no
// base pair, only the MSAA x hit-shading matrix at SM 6.5. Its vertex stage is
// its own (it applies the per-draw model matrix) but shares b0/b1 with the rest
// of the pass.
/// `glass_mesh_vertex` from `glass_mesh.slang`.
pub static GLASS_MESH_VERT: SlangProgram = SlangProgram {
    file: "glass_mesh.slang",
    entry: "glass_mesh_vertex",
    profile: "vs_6_0",
    label: "glass_mesh_vert.slang",
    defines: GLASS_MESH_DEFINES,
};

/// `glass_mesh_rt_fragment` from `glass_mesh.slang`.
pub static GLASS_MESH_RT_FRAG: SlangProgram = SlangProgram {
    file: "glass_mesh.slang",
    entry: "glass_mesh_rt_fragment",
    profile: "ps_6_5",
    label: "glass_mesh_frag_rt.slang",
    defines: GLASS_MESH_DEFINES,
};

/// `glass_mesh_rt_fragment` from `glass_mesh.slang`.
pub static GLASS_MESH_RT_FRAG_MSAA: SlangProgram = SlangProgram {
    file: "glass_mesh.slang",
    entry: "glass_mesh_rt_fragment",
    profile: "ps_6_5",
    label: "glass_mesh_frag_rt_msaa.slang",
    defines: GLASS_MESH_MSAA_DEFINES,
};

/// `glass_mesh_rt_fragment` from `glass_mesh.slang`.
pub static GLASS_MESH_RT_FRAG_TEXTURED: SlangProgram = SlangProgram {
    file: "glass_mesh.slang",
    entry: "glass_mesh_rt_fragment",
    profile: "ps_6_5",
    label: "glass_mesh_frag_rt_textured.slang",
    defines: GLASS_MESH_TEXTURED_DEFINES,
};

/// `glass_mesh_rt_fragment` from `glass_mesh.slang`.
pub static GLASS_MESH_RT_FRAG_TEXTURED_MSAA: SlangProgram = SlangProgram {
    file: "glass_mesh.slang",
    entry: "glass_mesh_rt_fragment",
    profile: "ps_6_5",
    label: "glass_mesh_frag_rt_textured_msaa.slang",
    defines: GLASS_MESH_TEXTURED_MSAA_DEFINES,
};

// The water surface family, the transparent pass's other producer. Same shape as
// the glass table above -- one vertex stage for every pipeline, an MSAA pair of
// base fragments, and the two ray-traced fragments at shader model 6.5 -- and
// deliberately the same registers, so `transparent.rs` builds one root signature
// per path and both producers draw under it.
/// `water_vertex` from `water.slang`.
pub static WATER_VERT: SlangProgram = SlangProgram {
    file: "water.slang",
    entry: "water_vertex",
    profile: "vs_6_0",
    label: "water_vert.slang",
    defines: &[("DXIL_ABI", "1"), ("MAX_PROBES", "8")],
};
/// `water_fragment` from `water.slang`.
pub static WATER_FRAG: SlangProgram = SlangProgram {
    file: "water.slang",
    entry: "water_fragment",
    profile: "ps_6_0",
    label: "water_frag.slang",
    defines: &[("DXIL_ABI", "1"), ("MAX_PROBES", "8"), ("USE_MSAA", "0")],
};
/// `water_fragment` from `water.slang`.
pub static WATER_FRAG_MSAA: SlangProgram = SlangProgram {
    file: "water.slang",
    entry: "water_fragment",
    profile: "ps_6_0",
    label: "water_frag_msaa.slang",
    defines: &[("DXIL_ABI", "1"), ("MAX_PROBES", "8"), ("USE_MSAA", "1")],
};
/// `water_rt_fragment` from `water.slang`.
pub static WATER_RT_FRAG: SlangProgram = SlangProgram {
    file: "water.slang",
    entry: "water_rt_fragment",
    profile: "ps_6_5",
    label: "water_frag_rt.slang",
    defines: &[
        ("DXIL_ABI", "1"),
        ("WATER_RT", "1"),
        ("MAX_PROBES", "8"),
        ("USE_MSAA", "0"),
    ],
};
/// `water_rt_fragment` from `water.slang`.
pub static WATER_RT_FRAG_MSAA: SlangProgram = SlangProgram {
    file: "water.slang",
    entry: "water_rt_fragment",
    profile: "ps_6_5",
    label: "water_frag_rt_msaa.slang",
    defines: &[
        ("DXIL_ABI", "1"),
        ("WATER_RT", "1"),
        ("MAX_PROBES", "8"),
        ("USE_MSAA", "1"),
    ],
};
/// `water_rt_fragment` from `water.slang`.
pub static WATER_RT_FRAG_TEXTURED: SlangProgram = SlangProgram {
    file: "water.slang",
    entry: "water_rt_fragment",
    profile: "ps_6_5",
    label: "water_frag_rt_textured.slang",
    defines: &[
        ("DXIL_ABI", "1"),
        ("WATER_RT", "1"),
        ("RT_TEXTURED", "1"),
        ("MAX_PROBES", "8"),
        ("USE_MSAA", "0"),
    ],
};
/// `water_rt_fragment` from `water.slang`.
pub static WATER_RT_FRAG_TEXTURED_MSAA: SlangProgram = SlangProgram {
    file: "water.slang",
    entry: "water_rt_fragment",
    profile: "ps_6_5",
    label: "water_frag_rt_textured_msaa.slang",
    defines: &[
        ("DXIL_ABI", "1"),
        ("WATER_RT", "1"),
        ("RT_TEXTURED", "1"),
        ("MAX_PROBES", "8"),
        ("USE_MSAA", "1"),
    ],
};

// The remaining raster families: the particle billboard pair, the projected
// decal, world-space lines and the text / sprite overlay. Each has real vertex
// geometry, so unlike the post passes they keep their own vertex entry rather
// than pairing with `fullscreen.slang`.
//
// Only the particle pair takes an ABI block, and only to swap two constant
// buffers: its root signature in `particle.rs` binds the view at b0 and the
// per-emitter params at b1, where the Metal buffer indices are 1 and 2. Decal,
// line and text need none -- declaration order already lands each resource on
// the register its root signature declares, which the rows in build.rs's
// `SLANG_DXIL_ENTRY_ABI` pin so a slangc release cannot move one silently.
//
// The two depth-reading fragments come as MSAA pairs the way the fog and glass
// fragments do: the sample count is a host difference, so the caller picks by
// MSAA state and the export-time precompile leaves a bundle warm for either.
// Their vertex stages never name the depth source and take neither.
const PARTICLE_ABI: &[(&str, &str)] = &[("DXIL_ABI", "1")];

/// `particle_vertex` from `particle.slang`.
pub static PARTICLE_VERT: SlangProgram = SlangProgram {
    file: "particle.slang",
    entry: "particle_vertex",
    profile: "vs_6_0",
    label: "particle_vert.slang",
    defines: PARTICLE_ABI,
};
/// `particle_fragment` from `particle.slang`.
pub static PARTICLE_FRAG: SlangProgram = SlangProgram {
    file: "particle.slang",
    entry: "particle_fragment",
    profile: "ps_6_0",
    label: "particle_frag.slang",
    defines: PARTICLE_ABI,
};
/// `decal_vertex` from `decal.slang`.
pub static DECAL_VERT: SlangProgram = SlangProgram {
    file: "decal.slang",
    entry: "decal_vertex",
    profile: "vs_6_0",
    label: "decal_vert.slang",
    defines: &[],
};
/// `decal_fragment` from `decal.slang`.
pub static DECAL_FRAG: SlangProgram = SlangProgram {
    file: "decal.slang",
    entry: "decal_fragment",
    profile: "ps_6_0",
    label: "decal_frag.slang",
    defines: &[("USE_MSAA", "0")],
};
/// `decal_fragment` from `decal.slang`.
pub static DECAL_FRAG_MSAA: SlangProgram = SlangProgram {
    file: "decal.slang",
    entry: "decal_fragment",
    profile: "ps_6_0",
    label: "decal_frag_msaa.slang",
    defines: &[("USE_MSAA", "1")],
};
/// `line_vertex` from `line.slang`.
pub static LINE_VERT: SlangProgram = SlangProgram {
    file: "line.slang",
    entry: "line_vertex",
    profile: "vs_6_0",
    label: "line_vert.slang",
    defines: &[],
};
/// `line_fragment` from `line.slang`.
pub static LINE_FRAG: SlangProgram = SlangProgram {
    file: "line.slang",
    entry: "line_fragment",
    profile: "ps_6_0",
    label: "line_frag.slang",
    defines: &[("USE_MSAA", "0")],
};
/// `line_fragment` from `line.slang`.
pub static LINE_FRAG_MSAA: SlangProgram = SlangProgram {
    file: "line.slang",
    entry: "line_fragment",
    profile: "ps_6_0",
    label: "line_frag_msaa.slang",
    defines: &[("USE_MSAA", "1")],
};
/// `text_vertex_main` from `text.slang`.
pub static TEXT_VERT: SlangProgram = SlangProgram {
    file: "text.slang",
    entry: "text_vertex_main",
    profile: "vs_6_0",
    label: "text_vert.slang",
    defines: &[],
};
/// `text_fragment_main` from `text.slang`.
pub static TEXT_FRAG: SlangProgram = SlangProgram {
    file: "text.slang",
    entry: "text_fragment_main",
    profile: "ps_6_0",
    label: "text_frag.slang",
    defines: &[],
};

// Every declared program, iterated by the export-time precompile. Both Hi-Z
// init variants are enumerated, and both MSAA halves of every fragment that has
// them: which one a device runs depends on its MSAA mode, and a bundle should
// be warm for either.
/// Every declared program, which the renderer and the build script both
/// iterate: one compiles them at init, the other ahead of time.
pub static ALL: &[&SlangProgram] = &[
    &MAIN_BINDLESS_VERT,
    &MAIN_BINDLESS_FRAG,
    &LIGHT_CULL,
    &RT_SKIN,
    &HIZ_INIT_SINGLE,
    &HIZ_INIT_MSAA,
    &HIZ_DOWNSAMPLE,
    &GBUFFER_PREPASS_VERT,
    &GBUFFER_PREPASS_VERT_INSTANCED,
    &GBUFFER_PREPASS_VERT_SKINNED,
    &GBUFFER_BINDLESS_VERT,
    &GBUFFER_PREPASS_FRAG,
    &GBUFFER_BINDLESS_FRAG,
    &SHADOW_VERT,
    &SKINNED_SHADOW_VERT,
    &SHADOW_BINDLESS_VERT,
    &FOG_FROXEL,
    &FOG_FRAG,
    &FOG_FRAG_MSAA,
    &AUTO_EXPOSURE_BUILD,
    &AUTO_EXPOSURE_AVERAGE,
    &PARTICLE_SIMULATE,
    &FULLSCREEN_VERT,
    &TAA_FRAG,
    &BLOOM_PREFILTER,
    &BLOOM_DOWNSAMPLE,
    &BLOOM_UPSAMPLE,
    &COMPOSITE_FRAG,
    &SSAO_KERNEL,
    &SSAO_BLUR,
    &SSR_RESOLVE,
    &SSGI_GATHER,
    &SSGI_COMPOSITE,
    &REFLECTION_BLUR,
    &REFLECTION_COMPOSITE,
    &RT_REFLECTIONS_FRAG,
    &RT_REFLECTIONS_FRAG_TEXTURED,
    &GLASS_VERT,
    &GLASS_FRAG,
    &GLASS_FRAG_MSAA,
    &GLASS_RT_FRAG,
    &GLASS_RT_FRAG_MSAA,
    &GLASS_RT_FRAG_TEXTURED,
    &GLASS_RT_FRAG_TEXTURED_MSAA,
    &GLASS_MESH_VERT,
    &GLASS_MESH_RT_FRAG,
    &GLASS_MESH_RT_FRAG_MSAA,
    &GLASS_MESH_RT_FRAG_TEXTURED,
    &GLASS_MESH_RT_FRAG_TEXTURED_MSAA,
    &WATER_VERT,
    &WATER_FRAG,
    &WATER_FRAG_MSAA,
    &WATER_RT_FRAG,
    &WATER_RT_FRAG_MSAA,
    &WATER_RT_FRAG_TEXTURED,
    &WATER_RT_FRAG_TEXTURED_MSAA,
    &PARTICLE_VERT,
    &PARTICLE_FRAG,
    &DECAL_VERT,
    &DECAL_FRAG,
    &DECAL_FRAG_MSAA,
    &LINE_VERT,
    &LINE_FRAG,
    &LINE_FRAG_MSAA,
    &TEXT_VERT,
    &TEXT_FRAG,
];

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    // `label` is the key the build script files each precompiled DXIL artifact
    // under and the renderer looks it up by, so two programs sharing one would
    // silently hand one variant's bytes to the other. Nothing else is unique:
    // `fog_fragment` at `ps_6_0` is both the MSAA and the non-MSAA program, and
    // `glass_rt_fragment` at `ps_6_5` is four of them, separated only by their
    // defines.
    #[test]
    fn every_program_has_a_distinct_label() {
        let mut labels: Vec<&str> = ALL.iter().map(|p| p.label).collect();
        let declared = labels.len();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(
            labels.len(),
            declared,
            "two programs share a label, so they would share a precompiled artifact"
        );
    }

    // A program naming a file that is not embedded would assemble to nothing
    // and fail at slangc rather than here.
    #[test]
    fn every_program_names_an_embedded_shader() {
        for p in ALL {
            assert!(
                crate::shaders::embedded(p.file).is_some(),
                "{}: no embedded {}",
                p.label,
                p.file
            );
        }
    }

    // The table is the compile set: a declared program left out of it is never
    // precompiled and never reached.
    #[test]
    fn the_table_carries_every_declared_program() {
        for p in ALL {
            assert!(!p.entry.is_empty() && !p.profile.is_empty(), "{}", p.label);
        }
        assert!(
            ALL.len() >= 60,
            "the DirectX program set shrank unexpectedly"
        );
    }
}
