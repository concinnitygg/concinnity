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

// Which runtime capacities a program bakes in as `#define`s. They ride the
// source text rather than a command line so the shader cache keys them.
#[derive(Clone, Copy, PartialEq, Eq)]
/// Which runtime capacities a program bakes into its source.
pub enum Sizes {
    /// No size defines.
    None,
    /// The reflection-probe array length only: the SSR resolve binds the
    /// forward global set's probe cubes but none of the texture pool.
    Probes,
    /// The bindless texture-pool capacity and the probe array length.
    PoolAndProbes,
}

/// One SPIR-V program: which shader file, which entry point, under which
/// variant gates and baked capacities.
pub struct SlangProgram {
    /// File name under `src/shaders/` for the `cn debug` disk-first resolve;
    /// also the embedded fallback's origin.
    pub file: &'static str,
    /// Entry point compiled out of that file.
    pub entry: &'static str,
    /// Diagnostic label (compile errors + cache miss logs + export report).
    pub label: &'static str,
    /// Fixed variant gates (e.g. HIZ_INIT_MSAA), each injected as
    /// `#define <gate> 1`. More than one where a variant is the intersection of
    /// two, like the textured ray-traced glass fragment.
    pub gates: &'static [&'static str],
    /// Runtime capacities injected from the context.
    pub sizes: Sizes,
    /// Inject `#define USE_MSAA {0|1}` from `Ctx::msaa`. A HOST difference
    /// rather than a target one: only the fog fragment declares its depth source
    /// by the main pass's sample count.
    pub msaa: bool,
}

/// `vertex_main_bindless` from `main_bindless.slang`.
pub static MAIN_BINDLESS_VERT: SlangProgram = SlangProgram {
    file: "main_bindless.slang",
    entry: "vertex_main_bindless",
    label: "vert_bindless.slang",
    gates: &[],
    sizes: Sizes::PoolAndProbes,
    msaa: false,
};
/// `fragment_main_bindless` from `main_bindless.slang`.
pub static MAIN_BINDLESS_FRAG: SlangProgram = SlangProgram {
    file: "main_bindless.slang",
    entry: "fragment_main_bindless",
    label: "frag_bindless.slang",
    gates: &[],
    sizes: Sizes::PoolAndProbes,
    msaa: false,
};
/// `light_cull_kernel` from `light_cull.slang`.
pub static LIGHT_CULL: SlangProgram = SlangProgram {
    file: "light_cull.slang",
    entry: "light_cull_kernel",
    label: "light_cull.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: false,
};
/// `hiz_init_msaa` from `hiz_build.slang`.
pub static HIZ_INIT_MSAA: SlangProgram = SlangProgram {
    file: "hiz_build.slang",
    entry: "hiz_init_msaa",
    label: "hiz_init_msaa.slang",
    gates: &["HIZ_INIT_MSAA"],
    sizes: Sizes::None,
    msaa: false,
};
/// `hiz_init_single` from `hiz_build.slang`.
pub static HIZ_INIT_SINGLE: SlangProgram = SlangProgram {
    file: "hiz_build.slang",
    entry: "hiz_init_single",
    label: "hiz_init_single.slang",
    gates: &["HIZ_INIT_SINGLE"],
    sizes: Sizes::None,
    msaa: false,
};
/// `hiz_downsample` from `hiz_build.slang`.
pub static HIZ_DOWNSAMPLE: SlangProgram = SlangProgram {
    file: "hiz_build.slang",
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
/// `gbuffer_prepass_vertex` from `gbuffer_prepass.slang`.
pub static GBUFFER_PREPASS_VERT: SlangProgram = SlangProgram {
    file: "gbuffer_prepass.slang",
    entry: "gbuffer_prepass_vertex",
    label: "gbuffer_prepass_vert.slang",
    gates: &["GB_STATIC"],
    sizes: Sizes::None,
    msaa: false,
};
/// `gbuffer_prepass_vertex_instanced` from `gbuffer_prepass.slang`.
pub static GBUFFER_PREPASS_VERT_INSTANCED: SlangProgram = SlangProgram {
    file: "gbuffer_prepass.slang",
    entry: "gbuffer_prepass_vertex_instanced",
    label: "gbuffer_prepass_vert_instanced.slang",
    gates: &["GB_INSTANCED"],
    sizes: Sizes::None,
    msaa: false,
};
/// `gbuffer_prepass_vertex_skinned` from `gbuffer_prepass.slang`.
pub static GBUFFER_PREPASS_VERT_SKINNED: SlangProgram = SlangProgram {
    file: "gbuffer_prepass.slang",
    entry: "gbuffer_prepass_vertex_skinned",
    label: "gbuffer_prepass_vert_skinned.slang",
    gates: &["GB_SKINNED"],
    sizes: Sizes::None,
    msaa: false,
};
/// `gbuffer_prepass_vertex_bindless` from `gbuffer_prepass.slang`.
pub static GBUFFER_BINDLESS_VERT: SlangProgram = SlangProgram {
    file: "gbuffer_prepass.slang",
    entry: "gbuffer_prepass_vertex_bindless",
    label: "gbuffer_prepass_vert_bindless.slang",
    gates: &["GB_BINDLESS"],
    sizes: Sizes::None,
    msaa: false,
};
/// `gbuffer_prepass_fragment` from `gbuffer_prepass.slang`.
pub static GBUFFER_PREPASS_FRAG: SlangProgram = SlangProgram {
    file: "gbuffer_prepass.slang",
    entry: "gbuffer_prepass_fragment",
    label: "gbuffer_prepass_frag.slang",
    gates: &["GB_FRAGMENT"],
    sizes: Sizes::None,
    msaa: false,
};
/// `gbuffer_prepass_fragment_bindless` from `gbuffer_prepass.slang`.
pub static GBUFFER_BINDLESS_FRAG: SlangProgram = SlangProgram {
    file: "gbuffer_prepass.slang",
    entry: "gbuffer_prepass_fragment_bindless",
    label: "gbuffer_prepass_frag_bindless.slang",
    gates: &["GB_FRAGMENT_BINDLESS"],
    sizes: Sizes::None,
    msaa: false,
};
/// `shadow_vertex_main` from `shadow.slang`.
pub static SHADOW_VERT: SlangProgram = SlangProgram {
    file: "shadow.slang",
    entry: "shadow_vertex_main",
    label: "shadow_vert.slang",
    gates: &["SHADOW_STATIC"],
    sizes: Sizes::None,
    msaa: false,
};
/// `shadow_vertex_main_skinned` from `shadow.slang`.
pub static SKINNED_SHADOW_VERT: SlangProgram = SlangProgram {
    file: "shadow.slang",
    entry: "shadow_vertex_main_skinned",
    label: "shadow_vert_skinned.slang",
    gates: &["SHADOW_SKINNED"],
    sizes: Sizes::None,
    msaa: false,
};
/// `shadow_vertex_bindless` from `shadow.slang`.
pub static SHADOW_BINDLESS_VERT: SlangProgram = SlangProgram {
    file: "shadow.slang",
    entry: "shadow_vertex_bindless",
    label: "shadow_vert_bindless.slang",
    gates: &["SHADOW_BINDLESS"],
    sizes: Sizes::None,
    msaa: false,
};

// The fullscreen-triangle vertex stage every ported post pass pairs with; one
// module serves them all, the way `composite.vert` served the GLSL ones.
/// `fullscreen_vertex` from `fullscreen.slang`.
pub static FULLSCREEN_VERT: SlangProgram = SlangProgram {
    file: "fullscreen.slang",
    entry: "fullscreen_vertex",
    label: "fullscreen_vert.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: false,
};
/// `taa_fragment_main` from `taa.slang`.
pub static TAA_FRAG: SlangProgram = SlangProgram {
    file: "taa.slang",
    entry: "taa_fragment_main",
    label: "taa_frag.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: false,
};
/// `bloom_prefilter_fragment` from `bloom.slang`.
pub static BLOOM_PREFILTER: SlangProgram = SlangProgram {
    file: "bloom.slang",
    entry: "bloom_prefilter_fragment",
    label: "bloom_prefilter.slang",
    gates: &["BLOOM_PREFILTER"],
    sizes: Sizes::None,
    msaa: false,
};
/// `bloom_downsample_fragment` from `bloom.slang`.
pub static BLOOM_DOWNSAMPLE: SlangProgram = SlangProgram {
    file: "bloom.slang",
    entry: "bloom_downsample_fragment",
    label: "bloom_downsample.slang",
    gates: &["BLOOM_DOWNSAMPLE"],
    sizes: Sizes::None,
    msaa: false,
};
/// `bloom_upsample_fragment` from `bloom.slang`.
pub static BLOOM_UPSAMPLE: SlangProgram = SlangProgram {
    file: "bloom.slang",
    entry: "bloom_upsample_fragment",
    label: "bloom_upsample.slang",
    gates: &["BLOOM_UPSAMPLE"],
    sizes: Sizes::None,
    msaa: false,
};
/// `composite_fragment` from `composite.slang`.
pub static COMPOSITE_FRAG: SlangProgram = SlangProgram {
    file: "composite.slang",
    entry: "composite_fragment",
    label: "composite_frag.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: false,
};
/// `ssao_kernel_fragment` from `ssao.slang`.
pub static SSAO_KERNEL: SlangProgram = SlangProgram {
    file: "ssao.slang",
    entry: "ssao_kernel_fragment",
    label: "ssao_kernel.slang",
    gates: &["SSAO_KERNEL"],
    sizes: Sizes::None,
    msaa: false,
};
/// `ssao_blur_fragment` from `ssao.slang`.
pub static SSAO_BLUR: SlangProgram = SlangProgram {
    file: "ssao.slang",
    entry: "ssao_blur_fragment",
    label: "ssao_blur.slang",
    gates: &["SSAO_BLUR"],
    sizes: Sizes::None,
    msaa: false,
};
/// `ssr_resolve_fragment` from `ssr.slang`.
pub static SSR_RESOLVE: SlangProgram = SlangProgram {
    file: "ssr.slang",
    entry: "ssr_resolve_fragment",
    label: "ssr_resolve.slang",
    gates: &[],
    sizes: Sizes::Probes,
    msaa: false,
};
/// `ssgi_gather_fragment` from `ssgi.slang`.
pub static SSGI_GATHER: SlangProgram = SlangProgram {
    file: "ssgi.slang",
    entry: "ssgi_gather_fragment",
    label: "ssgi_gather.slang",
    gates: &["SSGI_GATHER"],
    sizes: Sizes::None,
    msaa: false,
};
/// `ssgi_composite_fragment` from `ssgi.slang`.
pub static SSGI_COMPOSITE: SlangProgram = SlangProgram {
    file: "ssgi.slang",
    entry: "ssgi_composite_fragment",
    label: "ssgi_composite.slang",
    gates: &["SSGI_COMPOSITE"],
    sizes: Sizes::None,
    msaa: false,
};
/// `reflection_blur_fragment` from `reflection.slang`.
pub static REFLECTION_BLUR: SlangProgram = SlangProgram {
    file: "reflection.slang",
    entry: "reflection_blur_fragment",
    label: "reflection_blur.slang",
    gates: &["REFLECTION_BLUR"],
    sizes: Sizes::None,
    msaa: false,
};
/// `reflection_composite_fragment` from `reflection.slang`.
pub static REFLECTION_COMPOSITE: SlangProgram = SlangProgram {
    file: "reflection.slang",
    entry: "reflection_composite_fragment",
    label: "reflection_composite.slang",
    gates: &["REFLECTION_COMPOSITE"],
    sizes: Sizes::None,
    msaa: false,
};

// The compute kernels and the fog family. The fog fragment is the only program
// whose assembly depends on the host's MSAA mode.
/// `fog_froxel_kernel` from `fog.slang`.
pub static FOG_FROXEL: SlangProgram = SlangProgram {
    file: "fog.slang",
    entry: "fog_froxel_kernel",
    label: "fog_froxel.slang",
    gates: &["FOG_FROXEL"],
    sizes: Sizes::None,
    msaa: false,
};
/// `fog_fragment` from `fog.slang`.
pub static FOG_FRAG: SlangProgram = SlangProgram {
    file: "fog.slang",
    entry: "fog_fragment",
    label: "fog_frag.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: true,
};
/// `histogram_build` from `auto_exposure.slang`.
pub static AUTO_EXPOSURE_BUILD: SlangProgram = SlangProgram {
    file: "auto_exposure.slang",
    entry: "histogram_build",
    label: "auto_exposure_build.slang",
    gates: &["AE_BUILD"],
    sizes: Sizes::None,
    msaa: false,
};
/// `histogram_average` from `auto_exposure.slang`.
pub static AUTO_EXPOSURE_AVERAGE: SlangProgram = SlangProgram {
    file: "auto_exposure.slang",
    entry: "histogram_average",
    label: "auto_exposure_average.slang",
    gates: &["AE_AVERAGE"],
    sizes: Sizes::None,
    msaa: false,
};
/// `rt_skin` from `rt_skin.slang`.
pub static RT_SKIN: SlangProgram = SlangProgram {
    file: "rt_skin.slang",
    entry: "rt_skin",
    label: "rt_skin.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: false,
};
/// `particle_simulate` from `particle_simulate.slang`.
pub static PARTICLE_SIMULATE: SlangProgram = SlangProgram {
    file: "particle_simulate.slang",
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
/// `particle_vertex` from `particle.slang`.
pub static PARTICLE_VERT: SlangProgram = SlangProgram {
    file: "particle.slang",
    entry: "particle_vertex",
    label: "particle_vert.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: false,
};
/// `particle_fragment` from `particle.slang`.
pub static PARTICLE_FRAG: SlangProgram = SlangProgram {
    file: "particle.slang",
    entry: "particle_fragment",
    label: "particle_frag.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: false,
};
/// `decal_vertex` from `decal.slang`.
pub static DECAL_VERT: SlangProgram = SlangProgram {
    file: "decal.slang",
    entry: "decal_vertex",
    label: "decal_vert.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: false,
};
/// `decal_fragment` from `decal.slang`.
pub static DECAL_FRAG: SlangProgram = SlangProgram {
    file: "decal.slang",
    entry: "decal_fragment",
    label: "decal_frag.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: true,
};
/// `line_vertex` from `line.slang`.
pub static LINE_VERT: SlangProgram = SlangProgram {
    file: "line.slang",
    entry: "line_vertex",
    label: "line_vert.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: false,
};
/// `line_fragment` from `line.slang`.
pub static LINE_FRAG: SlangProgram = SlangProgram {
    file: "line.slang",
    entry: "line_fragment",
    label: "line_frag.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: true,
};
/// `text_vertex_main` from `text.slang`.
pub static TEXT_VERT: SlangProgram = SlangProgram {
    file: "text.slang",
    entry: "text_vertex_main",
    label: "text_vert.slang",
    gates: &[],
    sizes: Sizes::None,
    msaa: false,
};
/// `text_fragment_main` from `text.slang`.
pub static TEXT_FRAG: SlangProgram = SlangProgram {
    file: "text.slang",
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
/// `rt_reflections_fragment` from `rt_reflections.slang`.
pub static RT_REFLECTIONS_FRAG: SlangProgram = SlangProgram {
    file: "rt_reflections.slang",
    entry: "rt_reflections_fragment",
    label: "rt_reflections.slang",
    gates: &[],
    sizes: Sizes::Probes,
    msaa: false,
};
/// `rt_reflections_fragment` from `rt_reflections.slang`.
pub static RT_REFLECTIONS_FRAG_TEXTURED: SlangProgram = SlangProgram {
    file: "rt_reflections.slang",
    entry: "rt_reflections_fragment",
    label: "rt_reflections_textured.slang",
    gates: &["RT_TEXTURED"],
    sizes: Sizes::PoolAndProbes,
    msaa: false,
};
/// `glass_vertex` from `glass.slang`.
pub static GLASS_VERT: SlangProgram = SlangProgram {
    file: "glass.slang",
    entry: "glass_vertex",
    label: "glass_vert.slang",
    gates: &[],
    sizes: Sizes::Probes,
    msaa: true,
};
/// `glass_fragment` from `glass.slang`.
pub static GLASS_FRAG: SlangProgram = SlangProgram {
    file: "glass.slang",
    entry: "glass_fragment",
    label: "glass_frag.slang",
    gates: &[],
    sizes: Sizes::Probes,
    msaa: true,
};
/// `glass_rt_fragment` from `glass.slang`.
pub static GLASS_FRAG_RT: SlangProgram = SlangProgram {
    file: "glass.slang",
    entry: "glass_rt_fragment",
    label: "glass_frag_rt.slang",
    gates: &["GLASS_RT"],
    sizes: Sizes::Probes,
    msaa: true,
};
/// `glass_rt_fragment` from `glass.slang`.
pub static GLASS_FRAG_RT_TEXTURED: SlangProgram = SlangProgram {
    file: "glass.slang",
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
/// `glass_mesh_vertex` from `glass_mesh.slang`.
pub static GLASS_MESH_VERT: SlangProgram = SlangProgram {
    file: "glass_mesh.slang",
    entry: "glass_mesh_vertex",
    label: "glass_mesh_vert.slang",
    gates: &[],
    sizes: Sizes::Probes,
    msaa: true,
};
/// `glass_mesh_rt_fragment` from `glass_mesh.slang`.
pub static GLASS_MESH_FRAG_RT: SlangProgram = SlangProgram {
    file: "glass_mesh.slang",
    entry: "glass_mesh_rt_fragment",
    label: "glass_mesh_frag_rt.slang",
    gates: &[],
    sizes: Sizes::Probes,
    msaa: true,
};
/// `glass_mesh_rt_fragment` from `glass_mesh.slang`.
pub static GLASS_MESH_FRAG_RT_TEXTURED: SlangProgram = SlangProgram {
    file: "glass_mesh.slang",
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
/// `water_vertex` from `water.slang`.
pub static WATER_VERT: SlangProgram = SlangProgram {
    file: "water.slang",
    entry: "water_vertex",
    label: "water_vert.slang",
    gates: &[],
    sizes: Sizes::Probes,
    msaa: true,
};
/// `water_fragment` from `water.slang`.
pub static WATER_FRAG: SlangProgram = SlangProgram {
    file: "water.slang",
    entry: "water_fragment",
    label: "water_frag.slang",
    gates: &[],
    sizes: Sizes::Probes,
    msaa: true,
};
/// `water_rt_fragment` from `water.slang`.
pub static WATER_FRAG_RT: SlangProgram = SlangProgram {
    file: "water.slang",
    entry: "water_rt_fragment",
    label: "water_frag_rt.slang",
    gates: &["WATER_RT"],
    sizes: Sizes::Probes,
    msaa: true,
};
/// `water_rt_fragment` from `water.slang`.
pub static WATER_FRAG_RT_TEXTURED: SlangProgram = SlangProgram {
    file: "water.slang",
    entry: "water_rt_fragment",
    label: "water_frag_rt_textured.slang",
    gates: &["WATER_RT", "RT_TEXTURED"],
    sizes: Sizes::PoolAndProbes,
    msaa: true,
};

// mode, and a bundle should be warm for either.
/// Every declared program, which the renderer and the build script both
/// iterate: one compiles them at init, the other ahead of time.
pub static ALL: &[&SlangProgram] = &[
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    // `label` keys each precompiled SPIR-V artifact, paired with the sample
    // count for the programs that read it, so two programs sharing a label
    // would hand one's bytes to the other.
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

    // Only the sample count varies per host among the enumerable dimensions;
    // the sizes are baked at their ceilings. A program that gained a dimension
    // the build script does not enumerate would leave that variant uncovered.
    #[test]
    fn the_only_per_host_variant_is_the_sample_count() {
        let sampled = ALL.iter().filter(|p| p.msaa).count();
        assert!(sampled > 0, "the MSAA dimension went missing");
        assert!(sampled < ALL.len(), "every program cannot read the depth");
        for p in ALL {
            assert!(!p.entry.is_empty(), "{}", p.label);
        }
    }
}
