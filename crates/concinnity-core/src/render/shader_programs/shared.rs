//! Every program here compiles on all three backends from the same file, entry
//! and gates. Where a backend binds a declaration at a different slot, the
//! source branches on the backend define the assembler adds, so the row stays
//! one row.
//!
//! A static's name is its label's stem, uppercased.

use super::ShaderProgram;

/// `vertex_main_bindless` from `main_bindless.hlsl`.
pub static MAIN_BINDLESS_VERT: ShaderProgram = ShaderProgram {
    file: "main_bindless.hlsl",
    entry: "vertex_main_bindless",
    label: "main_bindless_vert.hlsl",
    gates: &[],
    msaa: false,
};
/// `fragment_main_bindless` from `main_bindless.hlsl`.
pub static MAIN_BINDLESS_FRAG: ShaderProgram = ShaderProgram {
    file: "main_bindless.hlsl",
    entry: "fragment_main_bindless",
    label: "main_bindless_frag.hlsl",
    gates: &[],
    msaa: false,
};

// The draw cull's decision. Metal runs it as the first half of the cull and
// turns the `cull_status` it writes into the indirect command buffer with the
// hand-written `cull_encode.metal`.
/// `cull_kernel` from `cull.hlsl`: phase-1 draw cull.
pub static CULL_PHASE1: ShaderProgram = ShaderProgram {
    file: "cull.hlsl",
    entry: "cull_kernel",
    label: "cull_phase1.hlsl",
    gates: &[],
    msaa: false,
};
/// `cull_kernel` from `cull.hlsl` under `CULL_PHASE2`: the two-pass occlusion
/// re-test against the rebuilt Hi-Z pyramid.
pub static CULL_PHASE2: ShaderProgram = ShaderProgram {
    file: "cull.hlsl",
    entry: "cull_kernel",
    label: "cull_phase2.hlsl",
    gates: &["CULL_PHASE2"],
    msaa: false,
};
/// `cull_kernel` from `cull.hlsl` under `SHADOW_CULL`: light-frustum only.
pub static CULL_SHADOW: ShaderProgram = ShaderProgram {
    file: "cull.hlsl",
    entry: "cull_kernel",
    label: "cull_shadow.hlsl",
    gates: &["SHADOW_CULL"],
    msaa: false,
};
/// `model_history_kernel` from `model_history.hlsl`: this frame's model
/// snapshot for the next frame's motion vectors.
pub static MODEL_HISTORY: ShaderProgram = ShaderProgram {
    file: "model_history.hlsl",
    entry: "model_history_kernel",
    label: "model_history.hlsl",
    gates: &[],
    msaa: false,
};
/// `light_cull_kernel` from `light_cull.hlsl`.
pub static LIGHT_CULL: ShaderProgram = ShaderProgram {
    file: "light_cull.hlsl",
    entry: "light_cull_kernel",
    label: "light_cull.hlsl",
    gates: &[],
    msaa: false,
};
/// `rt_skin` from `rt_skin.hlsl`.
pub static RT_SKIN: ShaderProgram = ShaderProgram {
    file: "rt_skin.hlsl",
    entry: "rt_skin",
    label: "rt_skin.hlsl",
    gates: &[],
    msaa: false,
};

// The runtime reflection-probe prefilter. One variant per kernel, so each
// declares exactly the textures it binds.
/// `probe_mip0` from `probe_prefilter.hlsl`.
pub static PROBE_MIP0: ShaderProgram = ShaderProgram {
    file: "probe_prefilter.hlsl",
    entry: "probe_mip0",
    label: "probe_mip0.hlsl",
    gates: &["PROBE_MIP0"],
    msaa: false,
};
/// `probe_downsample` from `probe_prefilter.hlsl`.
pub static PROBE_DOWNSAMPLE: ShaderProgram = ShaderProgram {
    file: "probe_prefilter.hlsl",
    entry: "probe_downsample",
    label: "probe_downsample.hlsl",
    gates: &["PROBE_DOWNSAMPLE"],
    msaa: false,
};
/// `probe_ggx` from `probe_prefilter.hlsl`.
pub static PROBE_GGX: ShaderProgram = ShaderProgram {
    file: "probe_prefilter.hlsl",
    entry: "probe_ggx",
    label: "probe_ggx.hlsl",
    gates: &["PROBE_GGX"],
    msaa: false,
};

// The G-buffer pre-pass and shadow families. Every entry is its own program so
// it declares only the resources it binds.
/// `gbuffer_prepass_vertex_bindless` from `gbuffer_prepass.hlsl`.
pub static GBUFFER_PREPASS_VERT_BINDLESS: ShaderProgram = ShaderProgram {
    file: "gbuffer_prepass.hlsl",
    entry: "gbuffer_prepass_vertex_bindless",
    label: "gbuffer_prepass_vert_bindless.hlsl",
    gates: &["GB_BINDLESS"],
    msaa: false,
};
/// `gbuffer_prepass_fragment_bindless` from `gbuffer_prepass.hlsl`.
pub static GBUFFER_PREPASS_FRAG_BINDLESS: ShaderProgram = ShaderProgram {
    file: "gbuffer_prepass.hlsl",
    entry: "gbuffer_prepass_fragment_bindless",
    label: "gbuffer_prepass_frag_bindless.hlsl",
    gates: &["GB_FRAGMENT_BINDLESS"],
    msaa: false,
};
/// `shadow_vertex_main` from `shadow.hlsl`.
pub static SHADOW_VERT: ShaderProgram = ShaderProgram {
    file: "shadow.hlsl",
    entry: "shadow_vertex_main",
    label: "shadow_vert.hlsl",
    gates: &["SHADOW_STATIC"],
    msaa: false,
};
/// `shadow_vertex_main_skinned` from `shadow.hlsl`.
pub static SHADOW_VERT_SKINNED: ShaderProgram = ShaderProgram {
    file: "shadow.hlsl",
    entry: "shadow_vertex_main_skinned",
    label: "shadow_vert_skinned.hlsl",
    gates: &["SHADOW_SKINNED"],
    msaa: false,
};
/// `shadow_vertex_bindless` from `shadow.hlsl`.
pub static SHADOW_VERT_BINDLESS: ShaderProgram = ShaderProgram {
    file: "shadow.hlsl",
    entry: "shadow_vertex_bindless",
    label: "shadow_vert_bindless.hlsl",
    gates: &["SHADOW_BINDLESS"],
    msaa: false,
};

// The fullscreen-triangle vertex stage every post pass pairs with, and the post
// fragments.
/// `fullscreen_vertex` from `fullscreen.hlsl`.
pub static FULLSCREEN_VERT: ShaderProgram = ShaderProgram {
    file: "fullscreen.hlsl",
    entry: "fullscreen_vertex",
    label: "fullscreen_vert.hlsl",
    gates: &[],
    msaa: false,
};
/// `taa_fragment_main` from `taa.hlsl`.
pub static TAA_FRAG: ShaderProgram = ShaderProgram {
    file: "taa.hlsl",
    entry: "taa_fragment_main",
    label: "taa_frag.hlsl",
    gates: &[],
    msaa: false,
};
/// `bloom_prefilter_fragment` from `bloom.hlsl`.
pub static BLOOM_PREFILTER: ShaderProgram = ShaderProgram {
    file: "bloom.hlsl",
    entry: "bloom_prefilter_fragment",
    label: "bloom_prefilter.hlsl",
    gates: &["BLOOM_PREFILTER"],
    msaa: false,
};
/// `bloom_downsample_fragment` from `bloom.hlsl`.
pub static BLOOM_DOWNSAMPLE: ShaderProgram = ShaderProgram {
    file: "bloom.hlsl",
    entry: "bloom_downsample_fragment",
    label: "bloom_downsample.hlsl",
    gates: &["BLOOM_DOWNSAMPLE"],
    msaa: false,
};
/// `bloom_upsample_fragment` from `bloom.hlsl`.
pub static BLOOM_UPSAMPLE: ShaderProgram = ShaderProgram {
    file: "bloom.hlsl",
    entry: "bloom_upsample_fragment",
    label: "bloom_upsample.hlsl",
    gates: &["BLOOM_UPSAMPLE"],
    msaa: false,
};
/// `composite_fragment` from `composite.hlsl`.
pub static COMPOSITE_FRAG: ShaderProgram = ShaderProgram {
    file: "composite.hlsl",
    entry: "composite_fragment",
    label: "composite_frag.hlsl",
    gates: &[],
    msaa: false,
};
/// `ssao_kernel_fragment` from `ssao.hlsl`.
pub static SSAO_KERNEL: ShaderProgram = ShaderProgram {
    file: "ssao.hlsl",
    entry: "ssao_kernel_fragment",
    label: "ssao_kernel.hlsl",
    gates: &["SSAO_KERNEL"],
    msaa: false,
};
/// `ssao_blur_fragment` from `ssao.hlsl`.
pub static SSAO_BLUR: ShaderProgram = ShaderProgram {
    file: "ssao.hlsl",
    entry: "ssao_blur_fragment",
    label: "ssao_blur.hlsl",
    gates: &["SSAO_BLUR"],
    msaa: false,
};
/// `ssr_resolve_fragment` from `ssr.hlsl`.
pub static SSR_RESOLVE: ShaderProgram = ShaderProgram {
    file: "ssr.hlsl",
    entry: "ssr_resolve_fragment",
    label: "ssr_resolve.hlsl",
    gates: &[],
    msaa: false,
};
/// `ssgi_gather_fragment` from `ssgi.hlsl`.
pub static SSGI_GATHER: ShaderProgram = ShaderProgram {
    file: "ssgi.hlsl",
    entry: "ssgi_gather_fragment",
    label: "ssgi_gather.hlsl",
    gates: &["SSGI_GATHER"],
    msaa: false,
};
/// `ssgi_composite_fragment` from `ssgi.hlsl`.
pub static SSGI_COMPOSITE: ShaderProgram = ShaderProgram {
    file: "ssgi.hlsl",
    entry: "ssgi_composite_fragment",
    label: "ssgi_composite.hlsl",
    gates: &["SSGI_COMPOSITE"],
    msaa: false,
};
/// `reflection_blur_fragment` from `reflection.hlsl`.
pub static REFLECTION_BLUR: ShaderProgram = ShaderProgram {
    file: "reflection.hlsl",
    entry: "reflection_blur_fragment",
    label: "reflection_blur.hlsl",
    gates: &["REFLECTION_BLUR"],
    msaa: false,
};
/// `reflection_composite_fragment` from `reflection.hlsl`.
pub static REFLECTION_COMPOSITE: ShaderProgram = ShaderProgram {
    file: "reflection.hlsl",
    entry: "reflection_composite_fragment",
    label: "reflection_composite.hlsl",
    gates: &["REFLECTION_COMPOSITE"],
    msaa: false,
};

// The fog family, auto-exposure and the particle simulation kernel. The fog
// fragment's vertex half is `FULLSCREEN_VERT`.
/// `fog_froxel_kernel` from `fog.hlsl`.
pub static FOG_FROXEL: ShaderProgram = ShaderProgram {
    file: "fog.hlsl",
    entry: "fog_froxel_kernel",
    label: "fog_froxel.hlsl",
    gates: &["FOG_FROXEL"],
    msaa: false,
};
/// `fog_fragment` from `fog.hlsl`.
pub static FOG_FRAG: ShaderProgram = ShaderProgram {
    file: "fog.hlsl",
    entry: "fog_fragment",
    label: "fog_frag.hlsl",
    gates: &[],
    msaa: true,
};
/// `histogram_build` from `auto_exposure.hlsl`.
pub static AUTO_EXPOSURE_BUILD: ShaderProgram = ShaderProgram {
    file: "auto_exposure.hlsl",
    entry: "histogram_build",
    label: "auto_exposure_build.hlsl",
    gates: &["AE_BUILD"],
    msaa: false,
};
/// `histogram_average` from `auto_exposure.hlsl`.
pub static AUTO_EXPOSURE_AVERAGE: ShaderProgram = ShaderProgram {
    file: "auto_exposure.hlsl",
    entry: "histogram_average",
    label: "auto_exposure_average.hlsl",
    gates: &["AE_AVERAGE"],
    msaa: false,
};
/// `particle_simulate` from `particle_simulate.hlsl`.
pub static PARTICLE_SIMULATE: ShaderProgram = ShaderProgram {
    file: "particle_simulate.hlsl",
    entry: "particle_simulate",
    label: "particle_simulate.hlsl",
    gates: &[],
    msaa: false,
};

// The remaining raster families: the particle billboard pair, the projected
// decal, world-space lines and the text / sprite overlay. Each has real vertex
// geometry, so unlike the post passes they keep their own vertex entry rather
// than pairing with `fullscreen.hlsl`. Only the three depth-reading fragments
// take the sample count; their vertex stages never name the depth source.
/// `particle_vertex` from `particle.hlsl`.
pub static PARTICLE_VERT: ShaderProgram = ShaderProgram {
    file: "particle.hlsl",
    entry: "particle_vertex",
    label: "particle_vert.hlsl",
    gates: &[],
    msaa: false,
};
/// `particle_fragment` from `particle.hlsl`.
pub static PARTICLE_FRAG: ShaderProgram = ShaderProgram {
    file: "particle.hlsl",
    entry: "particle_fragment",
    label: "particle_frag.hlsl",
    gates: &[],
    msaa: true,
};
/// `decal_vertex` from `decal.hlsl`.
pub static DECAL_VERT: ShaderProgram = ShaderProgram {
    file: "decal.hlsl",
    entry: "decal_vertex",
    label: "decal_vert.hlsl",
    gates: &[],
    msaa: false,
};
/// `decal_fragment` from `decal.hlsl`.
pub static DECAL_FRAG: ShaderProgram = ShaderProgram {
    file: "decal.hlsl",
    entry: "decal_fragment",
    label: "decal_frag.hlsl",
    gates: &[],
    msaa: true,
};
/// `line_vertex` from `line.hlsl`.
pub static LINE_VERT: ShaderProgram = ShaderProgram {
    file: "line.hlsl",
    entry: "line_vertex",
    label: "line_vert.hlsl",
    gates: &[],
    msaa: false,
};
/// `line_fragment` from `line.hlsl`.
pub static LINE_FRAG: ShaderProgram = ShaderProgram {
    file: "line.hlsl",
    entry: "line_fragment",
    label: "line_frag.hlsl",
    gates: &[],
    msaa: true,
};
/// `text_vertex_main` from `text.hlsl`.
pub static TEXT_VERT: ShaderProgram = ShaderProgram {
    file: "text.hlsl",
    entry: "text_vertex_main",
    label: "text_vert.hlsl",
    gates: &[],
    msaa: false,
};
/// `text_fragment_main` from `text.hlsl`.
pub static TEXT_FRAG: ShaderProgram = ShaderProgram {
    file: "text.hlsl",
    entry: "text_fragment_main",
    label: "text_frag.hlsl",
    gates: &[],
    msaa: false,
};

// The ray-traced families. A host builds these pipelines only on a device that
// supports ray queries, once an acceleration structure exists. The textured
// variants shade a hit with the bindless pool; the flat ones do not declare it
// at all, which keeps a non-bindless world from binding one.
/// `rt_reflections_fragment` from `rt_reflections.hlsl`.
pub static RT_REFLECTIONS_FRAG: ShaderProgram = ShaderProgram {
    file: "rt_reflections.hlsl",
    entry: "rt_reflections_fragment",
    label: "rt_reflections_frag.hlsl",
    gates: &[],
    msaa: false,
};
/// `rt_reflections_fragment` from `rt_reflections.hlsl`.
pub static RT_REFLECTIONS_FRAG_TEXTURED: ShaderProgram = ShaderProgram {
    file: "rt_reflections.hlsl",
    entry: "rt_reflections_fragment",
    label: "rt_reflections_frag_textured.hlsl",
    gates: &["RT_TEXTURED"],
    msaa: false,
};

// One vertex stage for both glass pipelines: the base pass and the ray-traced
// one differ only in where the reflection comes from.
/// `glass_vertex` from `glass.hlsl`.
pub static GLASS_VERT: ShaderProgram = ShaderProgram {
    file: "glass.hlsl",
    entry: "glass_vertex",
    label: "glass_vert.hlsl",
    gates: &[],
    msaa: false,
};
/// `glass_fragment` from `glass.hlsl`.
pub static GLASS_FRAG: ShaderProgram = ShaderProgram {
    file: "glass.hlsl",
    entry: "glass_fragment",
    label: "glass_frag.hlsl",
    gates: &[],
    msaa: true,
};
/// `glass_rt_fragment` from `glass.hlsl`.
pub static GLASS_FRAG_RT: ShaderProgram = ShaderProgram {
    file: "glass.hlsl",
    entry: "glass_rt_fragment",
    label: "glass_frag_rt.hlsl",
    gates: &["GLASS_RT"],
    msaa: true,
};
/// `glass_rt_fragment` from `glass.hlsl`.
pub static GLASS_FRAG_RT_TEXTURED: ShaderProgram = ShaderProgram {
    file: "glass.hlsl",
    entry: "glass_rt_fragment",
    label: "glass_frag_rt_textured.hlsl",
    gates: &["GLASS_RT", "RT_TEXTURED"],
    msaa: true,
};
/// `glass_rt_reflection_fragment` from `glass.hlsl`.
pub static GLASS_REFLECTION_FRAG: ShaderProgram = ShaderProgram {
    file: "glass.hlsl",
    entry: "glass_rt_reflection_fragment",
    label: "glass_reflection_frag.hlsl",
    gates: &["GLASS_RT"],
    msaa: true,
};
/// `glass_rt_reflection_fragment` from `glass.hlsl`.
pub static GLASS_REFLECTION_FRAG_TEXTURED: ShaderProgram = ShaderProgram {
    file: "glass.hlsl",
    entry: "glass_rt_reflection_fragment",
    label: "glass_reflection_frag_textured.hlsl",
    gates: &["GLASS_RT", "RT_TEXTURED"],
    msaa: true,
};

// The see-through glass MESH family, the transparent pass's third producer.
// Ray-traced only -- the per-pixel trace is what makes the mesh see-through
// rather than the opaque reflective glass the main pass draws -- so there is no
// base pair. Its vertex stage is its own (it applies the per-draw model matrix)
// but shares every binding with the rest of the pass.
/// `glass_mesh_vertex` from `glass_mesh.hlsl`.
pub static GLASS_MESH_VERT: ShaderProgram = ShaderProgram {
    file: "glass_mesh.hlsl",
    entry: "glass_mesh_vertex",
    label: "glass_mesh_vert.hlsl",
    gates: &[],
    msaa: false,
};
/// `glass_mesh_rt_fragment` from `glass_mesh.hlsl`.
pub static GLASS_MESH_FRAG_RT: ShaderProgram = ShaderProgram {
    file: "glass_mesh.hlsl",
    entry: "glass_mesh_rt_fragment",
    label: "glass_mesh_frag_rt.hlsl",
    gates: &[],
    msaa: true,
};
/// `glass_mesh_rt_fragment` from `glass_mesh.hlsl`.
pub static GLASS_MESH_FRAG_RT_TEXTURED: ShaderProgram = ShaderProgram {
    file: "glass_mesh.hlsl",
    entry: "glass_mesh_rt_fragment",
    label: "glass_mesh_frag_rt_textured.hlsl",
    gates: &["RT_TEXTURED"],
    msaa: true,
};
/// `glass_mesh_reflection_fragment` from `glass_mesh.hlsl`.
pub static GLASS_MESH_REFLECTION_FRAG: ShaderProgram = ShaderProgram {
    file: "glass_mesh.hlsl",
    entry: "glass_mesh_reflection_fragment",
    label: "glass_mesh_reflection_frag.hlsl",
    gates: &[],
    msaa: true,
};
/// `glass_mesh_reflection_fragment` from `glass_mesh.hlsl`.
pub static GLASS_MESH_REFLECTION_FRAG_TEXTURED: ShaderProgram = ShaderProgram {
    file: "glass_mesh.hlsl",
    entry: "glass_mesh_reflection_fragment",
    label: "glass_mesh_reflection_frag_textured.hlsl",
    gates: &["RT_TEXTURED"],
    msaa: true,
};

// The water surface family, the transparent pass's other producer. Same shape as
// the glass table above, and deliberately the same bindings, so the transparent
// pass builds one layout per path and both producers draw under it.
/// `water_vertex` from `water.hlsl`.
pub static WATER_VERT: ShaderProgram = ShaderProgram {
    file: "water.hlsl",
    entry: "water_vertex",
    label: "water_vert.hlsl",
    gates: &[],
    msaa: false,
};
/// `water_fragment` from `water.hlsl`.
pub static WATER_FRAG: ShaderProgram = ShaderProgram {
    file: "water.hlsl",
    entry: "water_fragment",
    label: "water_frag.hlsl",
    gates: &[],
    msaa: true,
};
/// `water_rt_fragment` from `water.hlsl`.
pub static WATER_FRAG_RT: ShaderProgram = ShaderProgram {
    file: "water.hlsl",
    entry: "water_rt_fragment",
    label: "water_frag_rt.hlsl",
    gates: &["WATER_RT"],
    msaa: true,
};
/// `water_rt_fragment` from `water.hlsl`.
pub static WATER_FRAG_RT_TEXTURED: ShaderProgram = ShaderProgram {
    file: "water.hlsl",
    entry: "water_rt_fragment",
    label: "water_frag_rt_textured.hlsl",
    gates: &["WATER_RT", "RT_TEXTURED"],
    msaa: true,
};

/// Every program in this module.
pub static ALL: &[&ShaderProgram] = &[
    &MAIN_BINDLESS_VERT,
    &MAIN_BINDLESS_FRAG,
    &LIGHT_CULL,
    &MODEL_HISTORY,
    &CULL_PHASE1,
    &CULL_PHASE2,
    &CULL_SHADOW,
    &RT_SKIN,
    &PROBE_MIP0,
    &PROBE_DOWNSAMPLE,
    &PROBE_GGX,
    &GBUFFER_PREPASS_VERT_BINDLESS,
    &GBUFFER_PREPASS_FRAG_BINDLESS,
    &SHADOW_VERT,
    &SHADOW_VERT_SKINNED,
    &SHADOW_VERT_BINDLESS,
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
    &GLASS_REFLECTION_FRAG,
    &GLASS_REFLECTION_FRAG_TEXTURED,
    &GLASS_MESH_VERT,
    &GLASS_MESH_FRAG_RT,
    &GLASS_MESH_FRAG_RT_TEXTURED,
    &GLASS_MESH_REFLECTION_FRAG,
    &GLASS_MESH_REFLECTION_FRAG_TEXTURED,
    &WATER_VERT,
    &WATER_FRAG,
    &WATER_FRAG_RT,
    &WATER_FRAG_RT_TEXTURED,
];

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    // A declared program left off `ALL` compiles at renderer init on every host
    // instead of riding the binary, which is what a compiler-free player cannot
    // do. The declarations are read from this file's own text because nothing
    // else distinguishes the two cases.
    #[test]
    fn every_declared_program_is_in_the_table() {
        super::super::declared::assert_table_is_complete(include_str!("shared.rs"));
    }

    // Reading the main pass's depth is what makes a program's assembly depend on
    // the host's sample count, and the build script embeds a variant per sample
    // count for exactly those. A program that gained or lost the dependency
    // silently would leave one MSAA mode compiling at renderer init.
    #[test]
    fn only_the_depth_reading_programs_take_the_sample_count() {
        let mut sampled: Vec<&str> = ALL.iter().filter(|p| p.msaa).map(|p| p.label).collect();
        sampled.sort_unstable();
        assert_eq!(
            sampled,
            [
                "decal_frag.hlsl",
                "fog_frag.hlsl",
                "glass_frag.hlsl",
                "glass_frag_rt.hlsl",
                "glass_frag_rt_textured.hlsl",
                "glass_mesh_frag_rt.hlsl",
                "glass_mesh_frag_rt_textured.hlsl",
                "glass_mesh_reflection_frag.hlsl",
                "glass_mesh_reflection_frag_textured.hlsl",
                "glass_reflection_frag.hlsl",
                "glass_reflection_frag_textured.hlsl",
                "line_frag.hlsl",
                "particle_frag.hlsl",
                "water_frag.hlsl",
                "water_frag_rt.hlsl",
                "water_frag_rt_textured.hlsl",
            ]
        );
    }
}
