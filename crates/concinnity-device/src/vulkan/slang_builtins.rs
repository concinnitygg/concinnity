// Single-source engine shader programs for the Vulkan backend.
//
// Each program compiles a `.slang` file under `src/shaders/` (the
// backend-neutral single-source directory) to SPIR-V at renderer init by
// invoking slangc through `concinnity-slang`, cached in the content-addressed
// shader cache exactly like the GLSL programs in
// `builtins`. The runtime `POOL_SIZE` / `MAX_PROBES` values are injected as
// `#define` lines into the source text, so the cache keys them the same way
// the GLSL `{POOL_SIZE}` substitution always has.
//
// The `[[vk::binding]]` annotations (and ParameterBlock member order) in the
// sources reproduce the engine's existing descriptor-set layouts, so these
// programs are drop-in replacements for the GLSL they superseded. slangc
// names every single-entry SPIR-V entry point `main`, so pipeline stage
// creation is unchanged too.

use concinnity_slang as slang;

use super::builtins::Ctx;

// Which runtime capacities a program bakes in as `#define`s. They ride the
// source text rather than a command line so the shader cache keys them.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sizes {
    // No size defines.
    None,
    // The reflection-probe array length only: the SSR resolve binds the
    // forward global set's probe cubes but none of the texture pool.
    Probes,
    // The bindless texture-pool capacity and the probe array length.
    PoolAndProbes,
}

pub(crate) struct SlangProgram {
    // File name under `src/shaders/` for the `cn debug` disk-first resolve;
    // also the embedded fallback's origin.
    pub file: &'static str,
    pub embedded: &'static str,
    pub entry: &'static str,
    // Diagnostic label (compile errors + cache miss logs + export report).
    pub label: &'static str,
    // Fixed variant gates (e.g. HIZ_INIT_MSAA), each injected as
    // `#define <gate> 1`. More than one where a variant is the intersection of
    // two, like the textured ray-traced glass fragment.
    pub gates: &'static [&'static str],
    // Runtime capacities injected from the context.
    pub sizes: Sizes,
    // Inject `#define USE_MSAA {0|1}` from `Ctx::msaa`. A HOST difference
    // rather than a target one: only the fog fragment declares its depth source
    // by the main pass's sample count.
    pub msaa: bool,
}

const MAIN_BINDLESS_SLANG: &str = include_str!("../shaders/main_bindless.slang");
const LIGHT_CULL_SLANG: &str = include_str!("../shaders/light_cull.slang");
const HIZ_BUILD_SLANG: &str = include_str!("../shaders/hiz_build.slang");
const GBUFFER_PREPASS_SLANG: &str = include_str!("../shaders/gbuffer_prepass.slang");
const SHADOW_SLANG: &str = include_str!("../shaders/shadow.slang");
const FULLSCREEN_SLANG: &str = include_str!("../shaders/fullscreen.slang");
const TAA_SLANG: &str = include_str!("../shaders/taa.slang");
const BLOOM_SLANG: &str = include_str!("../shaders/bloom.slang");
const COMPOSITE_SLANG: &str = include_str!("../shaders/composite.slang");
const SSAO_SLANG: &str = include_str!("../shaders/ssao.slang");
const SSR_SLANG: &str = include_str!("../shaders/ssr.slang");
const SSGI_SLANG: &str = include_str!("../shaders/ssgi.slang");
const REFLECTION_SLANG: &str = include_str!("../shaders/reflection.slang");
const FOG_SLANG: &str = include_str!("../shaders/fog.slang");
const AUTO_EXPOSURE_SLANG: &str = include_str!("../shaders/auto_exposure.slang");
const PARTICLE_SIMULATE_SLANG: &str = include_str!("../shaders/particle_simulate.slang");
const RT_SKIN_SLANG: &str = include_str!("../shaders/rt_skin.slang");
const PARTICLE_SLANG: &str = include_str!("../shaders/particle.slang");
const DECAL_SLANG: &str = include_str!("../shaders/decal.slang");
const LINE_SLANG: &str = include_str!("../shaders/line.slang");
const TEXT_SLANG: &str = include_str!("../shaders/text.slang");
const RT_REFLECTIONS_SLANG: &str = include_str!("../shaders/rt_reflections.slang");
const GLASS_SLANG: &str = include_str!("../shaders/glass.slang");
const WATER_SLANG: &str = include_str!("../shaders/water.slang");
const GLASS_MESH_SLANG: &str = include_str!("../shaders/glass_mesh.slang");

pub(super) static MAIN_BINDLESS_VERT: SlangProgram = SlangProgram {
    file: "main_bindless.slang",
    embedded: MAIN_BINDLESS_SLANG,
    entry: "vertex_main_bindless",
    label: "vert_bindless.slang",
    gates: &[],
    sizes: Sizes::PoolAndProbes,
    msaa: false,
};
pub(super) static MAIN_BINDLESS_FRAG: SlangProgram = SlangProgram {
    file: "main_bindless.slang",
    embedded: MAIN_BINDLESS_SLANG,
    entry: "fragment_main_bindless",
    label: "frag_bindless.slang",
    gates: &[],
    sizes: Sizes::PoolAndProbes,
    msaa: false,
};
pub(super) static LIGHT_CULL: SlangProgram = SlangProgram {
    file: "light_cull.slang",
    embedded: LIGHT_CULL_SLANG,
    entry: "light_cull_kernel",
    label: "light_cull.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static HIZ_INIT_MSAA: SlangProgram = SlangProgram {
    file: "hiz_build.slang",
    embedded: HIZ_BUILD_SLANG,
    entry: "hiz_init_msaa",
    label: "hiz_init_msaa.slang",
    gates: &["HIZ_INIT_MSAA"],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static HIZ_INIT_SINGLE: SlangProgram = SlangProgram {
    file: "hiz_build.slang",
    embedded: HIZ_BUILD_SLANG,
    entry: "hiz_init_single",
    label: "hiz_init_single.slang",
    gates: &["HIZ_INIT_SINGLE"],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static HIZ_DOWNSAMPLE: SlangProgram = SlangProgram {
    file: "hiz_build.slang",
    embedded: HIZ_BUILD_SLANG,
    entry: "hiz_downsample",
    label: "hiz_downsample.slang",
    gates: &["HIZ_DOWNSAMPLE"],
    sizes: Sizes::None,
    msaa: false,
};

// The G-buffer pre-pass and shadow families. Every entry is its own program so
// it declares only the resources it binds; the `[[vk::binding]]` annotations
// reproduce the descriptor sets the GLSL declared, so the SPIR-V is a drop-in
// against the untouched pipeline layouts.
pub(super) static GBUFFER_PREPASS_VERT: SlangProgram = SlangProgram {
    file: "gbuffer_prepass.slang",
    embedded: GBUFFER_PREPASS_SLANG,
    entry: "gbuffer_prepass_vertex",
    label: "gbuffer_prepass_vert.slang",
    gates: &["GB_STATIC"],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static GBUFFER_PREPASS_VERT_INSTANCED: SlangProgram = SlangProgram {
    file: "gbuffer_prepass.slang",
    embedded: GBUFFER_PREPASS_SLANG,
    entry: "gbuffer_prepass_vertex_instanced",
    label: "gbuffer_prepass_vert_instanced.slang",
    gates: &["GB_INSTANCED"],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static GBUFFER_PREPASS_VERT_SKINNED: SlangProgram = SlangProgram {
    file: "gbuffer_prepass.slang",
    embedded: GBUFFER_PREPASS_SLANG,
    entry: "gbuffer_prepass_vertex_skinned",
    label: "gbuffer_prepass_vert_skinned.slang",
    gates: &["GB_SKINNED"],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static GBUFFER_BINDLESS_VERT: SlangProgram = SlangProgram {
    file: "gbuffer_prepass.slang",
    embedded: GBUFFER_PREPASS_SLANG,
    entry: "gbuffer_prepass_vertex_bindless",
    label: "gbuffer_prepass_vert_bindless.slang",
    gates: &["GB_BINDLESS"],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static GBUFFER_PREPASS_FRAG: SlangProgram = SlangProgram {
    file: "gbuffer_prepass.slang",
    embedded: GBUFFER_PREPASS_SLANG,
    entry: "gbuffer_prepass_fragment",
    label: "gbuffer_prepass_frag.slang",
    gates: &["GB_FRAGMENT"],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static GBUFFER_BINDLESS_FRAG: SlangProgram = SlangProgram {
    file: "gbuffer_prepass.slang",
    embedded: GBUFFER_PREPASS_SLANG,
    entry: "gbuffer_prepass_fragment_bindless",
    label: "gbuffer_prepass_frag_bindless.slang",
    gates: &["GB_FRAGMENT_BINDLESS"],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static SHADOW_VERT: SlangProgram = SlangProgram {
    file: "shadow.slang",
    embedded: SHADOW_SLANG,
    entry: "shadow_vertex_main",
    label: "shadow_vert.slang",
    gates: &["SHADOW_STATIC"],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static SKINNED_SHADOW_VERT: SlangProgram = SlangProgram {
    file: "shadow.slang",
    embedded: SHADOW_SLANG,
    entry: "shadow_vertex_main_skinned",
    label: "shadow_vert_skinned.slang",
    gates: &["SHADOW_SKINNED"],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static SHADOW_BINDLESS_VERT: SlangProgram = SlangProgram {
    file: "shadow.slang",
    embedded: SHADOW_SLANG,
    entry: "shadow_vertex_bindless",
    label: "shadow_vert_bindless.slang",
    gates: &["SHADOW_BINDLESS"],
    sizes: Sizes::None,
    msaa: false,
};

// The fullscreen-triangle vertex stage every ported post pass pairs with; one
// module serves them all, the way `composite.vert` served the GLSL ones.
pub(super) static FULLSCREEN_VERT: SlangProgram = SlangProgram {
    file: "fullscreen.slang",
    embedded: FULLSCREEN_SLANG,
    entry: "fullscreen_vertex",
    label: "fullscreen_vert.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static TAA_FRAG: SlangProgram = SlangProgram {
    file: "taa.slang",
    embedded: TAA_SLANG,
    entry: "taa_fragment_main",
    label: "taa_frag.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static BLOOM_PREFILTER: SlangProgram = SlangProgram {
    file: "bloom.slang",
    embedded: BLOOM_SLANG,
    entry: "bloom_prefilter_fragment",
    label: "bloom_prefilter.slang",
    gates: &["BLOOM_PREFILTER"],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static BLOOM_DOWNSAMPLE: SlangProgram = SlangProgram {
    file: "bloom.slang",
    embedded: BLOOM_SLANG,
    entry: "bloom_downsample_fragment",
    label: "bloom_downsample.slang",
    gates: &["BLOOM_DOWNSAMPLE"],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static BLOOM_UPSAMPLE: SlangProgram = SlangProgram {
    file: "bloom.slang",
    embedded: BLOOM_SLANG,
    entry: "bloom_upsample_fragment",
    label: "bloom_upsample.slang",
    gates: &["BLOOM_UPSAMPLE"],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static COMPOSITE_FRAG: SlangProgram = SlangProgram {
    file: "composite.slang",
    embedded: COMPOSITE_SLANG,
    entry: "composite_fragment",
    label: "composite_frag.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static SSAO_KERNEL: SlangProgram = SlangProgram {
    file: "ssao.slang",
    embedded: SSAO_SLANG,
    entry: "ssao_kernel_fragment",
    label: "ssao_kernel.slang",
    gates: &["SSAO_KERNEL"],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static SSAO_BLUR: SlangProgram = SlangProgram {
    file: "ssao.slang",
    embedded: SSAO_SLANG,
    entry: "ssao_blur_fragment",
    label: "ssao_blur.slang",
    gates: &["SSAO_BLUR"],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static SSR_RESOLVE: SlangProgram = SlangProgram {
    file: "ssr.slang",
    embedded: SSR_SLANG,
    entry: "ssr_resolve_fragment",
    label: "ssr_resolve.slang",
    gates: &[],
    sizes: Sizes::Probes,
    msaa: false,
};
pub(super) static SSGI_GATHER: SlangProgram = SlangProgram {
    file: "ssgi.slang",
    embedded: SSGI_SLANG,
    entry: "ssgi_gather_fragment",
    label: "ssgi_gather.slang",
    gates: &["SSGI_GATHER"],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static SSGI_COMPOSITE: SlangProgram = SlangProgram {
    file: "ssgi.slang",
    embedded: SSGI_SLANG,
    entry: "ssgi_composite_fragment",
    label: "ssgi_composite.slang",
    gates: &["SSGI_COMPOSITE"],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static REFLECTION_BLUR: SlangProgram = SlangProgram {
    file: "reflection.slang",
    embedded: REFLECTION_SLANG,
    entry: "reflection_blur_fragment",
    label: "reflection_blur.slang",
    gates: &["REFLECTION_BLUR"],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static REFLECTION_COMPOSITE: SlangProgram = SlangProgram {
    file: "reflection.slang",
    embedded: REFLECTION_SLANG,
    entry: "reflection_composite_fragment",
    label: "reflection_composite.slang",
    gates: &["REFLECTION_COMPOSITE"],
    sizes: Sizes::None,
    msaa: false,
};

// The compute kernels and the fog family. The fog fragment is the only program
// whose assembly depends on the host's MSAA mode.
pub(super) static FOG_FROXEL: SlangProgram = SlangProgram {
    file: "fog.slang",
    embedded: FOG_SLANG,
    entry: "fog_froxel_kernel",
    label: "fog_froxel.slang",
    gates: &["FOG_FROXEL"],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static FOG_FRAG: SlangProgram = SlangProgram {
    file: "fog.slang",
    embedded: FOG_SLANG,
    entry: "fog_fragment",
    label: "fog_frag.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: true,
};
pub(super) static AUTO_EXPOSURE_BUILD: SlangProgram = SlangProgram {
    file: "auto_exposure.slang",
    embedded: AUTO_EXPOSURE_SLANG,
    entry: "histogram_build",
    label: "auto_exposure_build.slang",
    gates: &["AE_BUILD"],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static AUTO_EXPOSURE_AVERAGE: SlangProgram = SlangProgram {
    file: "auto_exposure.slang",
    embedded: AUTO_EXPOSURE_SLANG,
    entry: "histogram_average",
    label: "auto_exposure_average.slang",
    gates: &["AE_AVERAGE"],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static RT_SKIN: SlangProgram = SlangProgram {
    file: "rt_skin.slang",
    embedded: RT_SKIN_SLANG,
    entry: "rt_skin",
    label: "rt_skin.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static PARTICLE_SIMULATE: SlangProgram = SlangProgram {
    file: "particle_simulate.slang",
    embedded: PARTICLE_SIMULATE_SLANG,
    entry: "particle_simulate",
    label: "particle_simulate.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: false,
};

// The remaining raster families: the particle billboard pair, the projected
// decal, world-space lines and the text / sprite overlay. Each has real vertex
// geometry, so unlike the post passes they keep their own vertex entry rather
// than pairing with `fullscreen.slang`. Only the two depth-reading fragments
// take the host's sample count; their vertex stages never name the depth source.
pub(super) static PARTICLE_VERT: SlangProgram = SlangProgram {
    file: "particle.slang",
    embedded: PARTICLE_SLANG,
    entry: "particle_vertex",
    label: "particle_vert.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static PARTICLE_FRAG: SlangProgram = SlangProgram {
    file: "particle.slang",
    embedded: PARTICLE_SLANG,
    entry: "particle_fragment",
    label: "particle_frag.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static DECAL_VERT: SlangProgram = SlangProgram {
    file: "decal.slang",
    embedded: DECAL_SLANG,
    entry: "decal_vertex",
    label: "decal_vert.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static DECAL_FRAG: SlangProgram = SlangProgram {
    file: "decal.slang",
    embedded: DECAL_SLANG,
    entry: "decal_fragment",
    label: "decal_frag.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: true,
};
pub(super) static LINE_VERT: SlangProgram = SlangProgram {
    file: "line.slang",
    embedded: LINE_SLANG,
    entry: "line_vertex",
    label: "line_vert.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static LINE_FRAG: SlangProgram = SlangProgram {
    file: "line.slang",
    embedded: LINE_SLANG,
    entry: "line_fragment",
    label: "line_frag.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: true,
};
pub(super) static TEXT_VERT: SlangProgram = SlangProgram {
    file: "text.slang",
    embedded: TEXT_SLANG,
    entry: "text_vertex_main",
    label: "text_vert.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: false,
};
pub(super) static TEXT_FRAG: SlangProgram = SlangProgram {
    file: "text.slang",
    embedded: TEXT_SLANG,
    entry: "text_fragment_main",
    label: "text_frag.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: false,
};

// Every declared program, iterated by the export-time precompile. Both Hi-Z
// init variants are enumerated: which one a device runs depends on its MSAA
// The ray-traced families, compiled only where `VK_KHR_ray_query` is present:
// slangc emits `SPV_KHR_ray_query` (SPIR-V 1.5), so these need the Vulkan 1.2
// device the extension already implies. The flat variants declare no bindless
// pool at all, which is what keeps a non-bindless world from binding one.
pub(super) static RT_REFLECTIONS_FRAG: SlangProgram = SlangProgram {
    file: "rt_reflections.slang",
    embedded: RT_REFLECTIONS_SLANG,
    entry: "rt_reflections_fragment",
    label: "rt_reflections.slang",
    gates: &[],
    sizes: Sizes::Probes,
    msaa: false,
};
pub(super) static RT_REFLECTIONS_FRAG_TEXTURED: SlangProgram = SlangProgram {
    file: "rt_reflections.slang",
    embedded: RT_REFLECTIONS_SLANG,
    entry: "rt_reflections_fragment",
    label: "rt_reflections_textured.slang",
    gates: &["RT_TEXTURED"],
    sizes: Sizes::PoolAndProbes,
    msaa: false,
};
pub(super) static GLASS_VERT: SlangProgram = SlangProgram {
    file: "glass.slang",
    embedded: GLASS_SLANG,
    entry: "glass_vertex",
    label: "glass_vert.slang",
    gates: &[],
    sizes: Sizes::Probes,
    msaa: true,
};
pub(super) static GLASS_FRAG: SlangProgram = SlangProgram {
    file: "glass.slang",
    embedded: GLASS_SLANG,
    entry: "glass_fragment",
    label: "glass_frag.slang",
    gates: &[],
    sizes: Sizes::Probes,
    msaa: true,
};
pub(super) static GLASS_FRAG_RT: SlangProgram = SlangProgram {
    file: "glass.slang",
    embedded: GLASS_SLANG,
    entry: "glass_rt_fragment",
    label: "glass_frag_rt.slang",
    gates: &["GLASS_RT"],
    sizes: Sizes::Probes,
    msaa: true,
};
pub(super) static GLASS_FRAG_RT_TEXTURED: SlangProgram = SlangProgram {
    file: "glass.slang",
    embedded: GLASS_SLANG,
    entry: "glass_rt_fragment",
    label: "glass_frag_rt_textured.slang",
    gates: &["GLASS_RT", "RT_TEXTURED"],
    sizes: Sizes::PoolAndProbes,
    msaa: true,
};

// The see-through glass MESH family, the transparent pass's third producer.
// Ray-traced only -- the per-pixel trace is what makes the mesh see-through
// rather than the opaque reflective glass the main pass draws -- so there is no
// base pair. Its vertex stage is its own (it applies the per-draw model matrix)
// but shares every descriptor binding with the rest of the pass.
pub(super) static GLASS_MESH_VERT: SlangProgram = SlangProgram {
    file: "glass_mesh.slang",
    embedded: GLASS_MESH_SLANG,
    entry: "glass_mesh_vertex",
    label: "glass_mesh_vert.slang",
    gates: &[],
    sizes: Sizes::Probes,
    msaa: true,
};
pub(super) static GLASS_MESH_FRAG_RT: SlangProgram = SlangProgram {
    file: "glass_mesh.slang",
    embedded: GLASS_MESH_SLANG,
    entry: "glass_mesh_rt_fragment",
    label: "glass_mesh_frag_rt.slang",
    gates: &[],
    sizes: Sizes::Probes,
    msaa: true,
};
pub(super) static GLASS_MESH_FRAG_RT_TEXTURED: SlangProgram = SlangProgram {
    file: "glass_mesh.slang",
    embedded: GLASS_MESH_SLANG,
    entry: "glass_mesh_rt_fragment",
    label: "glass_mesh_frag_rt_textured.slang",
    gates: &["RT_TEXTURED"],
    sizes: Sizes::PoolAndProbes,
    msaa: true,
};

// The water surface family, the transparent pass's other producer. Same shape as
// the glass table above, and deliberately the same descriptor bindings, so
// `transparent.rs` builds one set of set layouts and both producers draw under
// them.
pub(super) static WATER_VERT: SlangProgram = SlangProgram {
    file: "water.slang",
    embedded: WATER_SLANG,
    entry: "water_vertex",
    label: "water_vert.slang",
    gates: &[],
    sizes: Sizes::Probes,
    msaa: true,
};
pub(super) static WATER_FRAG: SlangProgram = SlangProgram {
    file: "water.slang",
    embedded: WATER_SLANG,
    entry: "water_fragment",
    label: "water_frag.slang",
    gates: &[],
    sizes: Sizes::Probes,
    msaa: true,
};
pub(super) static WATER_FRAG_RT: SlangProgram = SlangProgram {
    file: "water.slang",
    embedded: WATER_SLANG,
    entry: "water_rt_fragment",
    label: "water_frag_rt.slang",
    gates: &["WATER_RT"],
    sizes: Sizes::Probes,
    msaa: true,
};
pub(super) static WATER_FRAG_RT_TEXTURED: SlangProgram = SlangProgram {
    file: "water.slang",
    embedded: WATER_SLANG,
    entry: "water_rt_fragment",
    label: "water_frag_rt_textured.slang",
    gates: &["WATER_RT", "RT_TEXTURED"],
    sizes: Sizes::PoolAndProbes,
    msaa: true,
};

// mode, and a bundle should be warm for either.
pub(crate) static ALL: &[&SlangProgram] = &[
    &MAIN_BINDLESS_VERT,
    &MAIN_BINDLESS_FRAG,
    &LIGHT_CULL,
    &RT_SKIN,
    &HIZ_INIT_MSAA,
    &HIZ_INIT_SINGLE,
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
    &FOG_FROXEL,
    &FOG_FRAG,
    &AUTO_EXPOSURE_BUILD,
    &AUTO_EXPOSURE_AVERAGE,
    &PARTICLE_SIMULATE,
    &PARTICLE_VERT,
    &PARTICLE_FRAG,
    &DECAL_VERT,
    &DECAL_FRAG,
    &LINE_VERT,
    &LINE_FRAG,
    &TEXT_VERT,
    &TEXT_FRAG,
    &RT_REFLECTIONS_FRAG,
    &RT_REFLECTIONS_FRAG_TEXTURED,
    &GLASS_VERT,
    &GLASS_FRAG,
    &GLASS_FRAG_RT,
    &GLASS_FRAG_RT_TEXTURED,
    &GLASS_MESH_VERT,
    &GLASS_MESH_FRAG_RT,
    &GLASS_MESH_FRAG_RT_TEXTURED,
    &WATER_VERT,
    &WATER_FRAG,
    &WATER_FRAG_RT,
    &WATER_FRAG_RT_TEXTURED,
];

impl SlangProgram {
    // Assemble the exact source text this program compiles under `ctx`.
    pub(crate) fn source(&self, ctx: &Ctx) -> String {
        let pool = ctx.pool_size.to_string();
        let probes = ctx.probe_count.to_string();
        let mut defines: Vec<(&str, &str)> = self.gates.iter().map(|g| (*g, "1")).collect();
        if self.msaa {
            defines.push(("USE_MSAA", if ctx.msaa { "1" } else { "0" }));
        }
        if self.sizes == Sizes::PoolAndProbes {
            debug_assert!(
                ctx.pool_size > 0,
                "{}: sized program assembled with no pool count",
                self.label
            );
            defines.push(("POOL_SIZE", pool.as_str()));
        }
        if self.sizes != Sizes::None {
            debug_assert!(
                ctx.probe_count > 0,
                "{}: sized program assembled with no probe count",
                self.label
            );
            defines.push(("MAX_PROBES", probes.as_str()));
        }
        crate::slang_source::assemble(ctx.hot_reload, self.file, self.embedded, &defines)
    }

    // The shader-cache key for `source`. Shared by the runtime compile path
    // and the export-time precompile so the two can never key differently.
    pub(crate) fn cache_key<'a>(&self, source: &'a str) -> crate::shader_cache::Key<'a> {
        crate::shader_cache::Key {
            compiler: "slang",
            source,
            entry: self.entry,
            target: "spirv",
            options: 0,
        }
    }

    // Compile to SPIR-V, reusing a cached artifact when this exact assembled
    // source has been compiled before.
    pub(crate) fn compile(&self, ctx: &Ctx) -> Result<Vec<u8>, String> {
        let source = self.source(ctx);
        let key = self.cache_key(&source);
        crate::shader_cache::cached(&key, self.label, || compile_uncached(self, &source))
    }
}

pub(super) fn compile_uncached(program: &SlangProgram, source: &str) -> Result<Vec<u8>, String> {
    let job = slang::SlangJob {
        source,
        file_name: program.file,
        entries: &[program.entry],
        target: slang::SlangTarget::Spirv,
    };
    slang::compile(&job, &crate::shader_cache::slang_work_dir())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(pool_size: usize, probe_count: usize) -> Ctx {
        Ctx {
            hot_reload: false,
            msaa: false,
            pool_size,
            probe_count,
        }
    }

    #[test]
    fn sized_programs_inject_their_counts_and_gates_lead() {
        let src = MAIN_BINDLESS_FRAG.source(&ctx(17, 5));
        assert!(src.starts_with("#define POOL_SIZE 17\n#define MAX_PROBES 5\n"));
        // The SSR resolve reads the probe array but none of the texture pool,
        // so it takes the probe count alone.
        let src = SSR_RESOLVE.source(&ctx(17, 5));
        assert!(src.starts_with("#define MAX_PROBES 5\n"));
        let src = HIZ_INIT_MSAA.source(&ctx(0, 0));
        assert!(src.starts_with("#define HIZ_INIT_MSAA 1\n"));
        let src = LIGHT_CULL.source(&ctx(0, 0));
        assert!(!src.starts_with("#define"));
    }

    // An unreplaced `{...}` fragment marker would reach slangc as a syntax
    // error at renderer init; catch a missing splice here instead.
    #[test]
    fn every_program_assembles_with_its_fragments_spliced() {
        for p in ALL {
            let src = p.source(&ctx(4, 4));
            for marker in [
                "{POST_COMMON}",
                "{OBJECT_COMMON}",
                "{PROBE_TYPES}",
                "{PROBE_COMMON}",
                "{RT_TYPES}",
                "{RT_TRACE}",
                "{PARTICLE_TYPES}",
            ] {
                assert!(
                    !src.contains(marker),
                    "{}: unspliced fragment marker {marker}",
                    p.label
                );
            }
        }
    }

    // Reading the main pass's depth is what makes a program's assembly depend on
    // the host's sample count, and the export-time precompile enumerates both
    // variants for exactly those. A program that gained or lost the dependency
    // silently would leave a bundle cold for one MSAA mode.
    #[test]
    fn only_the_depth_reading_programs_take_the_sample_count() {
        let mut sampled: Vec<&str> = ALL.iter().filter(|p| p.msaa).map(|p| p.label).collect();
        sampled.sort_unstable();
        assert_eq!(
            sampled,
            [
                "decal_frag.slang",
                "fog_frag.slang",
                "glass_frag.slang",
                "glass_frag_rt.slang",
                "glass_frag_rt_textured.slang",
                "glass_mesh_frag_rt.slang",
                "glass_mesh_frag_rt_textured.slang",
                "glass_mesh_vert.slang",
                "glass_vert.slang",
                "line_frag.slang",
                "water_frag.slang",
                "water_frag_rt.slang",
                "water_frag_rt_textured.slang",
                "water_vert.slang",
            ]
        );
    }

    // Two programs collide when they would compile identical source with the
    // same entry; the table must not declare the same artifact twice.
    #[test]
    fn table_has_no_duplicate_programs() {
        let mut seen = std::collections::HashSet::new();
        for p in ALL {
            let src = p.source(&ctx(4, 4));
            assert!(
                seen.insert((src, p.entry)),
                "duplicate program: {}",
                p.label
            );
        }
    }

    #[test]
    fn every_key_field_tracks_the_program() {
        let a = MAIN_BINDLESS_VERT.source(&ctx(4, 4));
        let b = MAIN_BINDLESS_VERT.source(&ctx(5, 4));
        assert_ne!(a, b, "pool size must change the assembled source");
        let key = MAIN_BINDLESS_VERT.cache_key(&a);
        assert_eq!(key.compiler, "slang");
        assert_eq!(key.target, "spirv");
        assert_eq!(key.entry, "vertex_main_bindless");
    }

    // The single-source files carry cluster constants that must track the
    // Rust values the CPU sizes buffers with.
    #[test]
    fn cluster_constants_match_render_types() {
        use crate::gfx::render_types::{CLUSTER_LIGHT_LIST_STRIDE, MAX_LIGHTS_PER_CLUSTER};
        for src in [LIGHT_CULL_SLANG, MAIN_BINDLESS_SLANG] {
            assert!(src.contains(&format!(
                "CLUSTER_LIGHT_LIST_STRIDE = {CLUSTER_LIGHT_LIST_STRIDE}u"
            )));
        }
        assert!(LIGHT_CULL_SLANG.contains(&format!(
            "MAX_LIGHTS_PER_CLUSTER = {MAX_LIGHTS_PER_CLUSTER}u"
        )));
    }
}
