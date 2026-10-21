//! The single-source engine shaders, embedded once for every consumer.
//!
//! Both halves of the toolchain read these: the renderer compiles them at init
//! (or loads a precompiled artifact keyed by their text), and the device build
//! script compiles the same text ahead of time. They live here, below the
//! backends, so the two can never disagree about what a program's source is --
//! the content-addressed shader cache keys on exactly this text.
//!
//! Every one is HLSL: a program compiles through dxc, and through spirv-cross
//! as well on the Metal leg. A splice fragment is plain text to the assembler
//! and compiles only inside the program that splices it.

// Declares one `pub const` per embedded shader plus the `SOURCES` table, so a
// file is listed once.
macro_rules! embed_shaders {
    ($($name:ident => $file:literal,)*) => {
        $(
            #[doc = concat!("`", $file, "`.")]
            pub const $name: &str = include_str!(concat!("hlsl/", $file));
        )*

        /// Every embedded shader, as (file name, source). `embedded` looks up by
        /// name; a build script iterating the set walks this table.
        pub const SOURCES: &[(&str, &str)] = &[$(($file, $name)),*];
    };
}

embed_shaders! {
    ATMOSPHERE => "atmosphere.hlsl",
    ATMOSPHERE_COMMON => "atmosphere_common.hlsl",
    ATMOSPHERE_TYPES => "atmosphere_types.hlsl",
    AUTO_EXPOSURE => "auto_exposure.hlsl",
    BLOOM => "bloom.hlsl",
    CANOPY => "canopy.hlsl",
    CLUSTER_TYPES => "cluster_types.hlsl",
    COMPOSITE => "composite.hlsl",
    CULL => "cull.hlsl",
    DECAL => "decal.hlsl",
    DEPTH_CONVENTION => "depth_convention.hlsl",
    FOG => "fog.hlsl",
    FULLSCREEN => "fullscreen.hlsl",
    GBUFFER_COMMON => "gbuffer_common.hlsl",
    GBUFFER_SKY => "gbuffer_sky.hlsl",
    GLASS => "glass.hlsl",
    GLASS_MESH => "glass_mesh.hlsl",
    GLASS_REFLECTION => "glass_reflection.hlsl",
    GRASS => "grass.hlsl",
    GRASS_FIELD => "grass_field.hlsl",
    GROUND_RULE_TYPES => "ground_rule_types.hlsl",
    HDR_STORE => "hdr_store.hlsl",
    HIZ_BUILD => "hiz_build.hlsl",
    HIZ_TEST => "hiz_test.hlsl",
    LIGHT_CULL => "light_cull.hlsl",
    LIGHT_TYPES => "light_types.hlsl",
    LINE => "line.hlsl",
    MAIN_BINDLESS => "main_bindless.hlsl",
    MAIN_RESOURCES => "main_resources.hlsl",
    MAIN_SHADING => "main_shading.hlsl",
    MAIN_TYPES => "main_types.hlsl",
    MODEL_HISTORY => "model_history.hlsl",
    NIGHT_SKY => "night_sky.hlsl",
    OBJECT_COMMON => "object_common.hlsl",
    PARTICLE => "particle.hlsl",
    PARTICLE_SIMULATE => "particle_simulate.hlsl",
    PARTICLE_TYPES => "particle_types.hlsl",
    POST_COMMON => "post_common.hlsl",
    PROBE_COMMON => "probe_common.hlsl",
    PROBE_PREFILTER => "probe_prefilter.hlsl",
    PROBE_TYPES => "probe_types.hlsl",
    RAYMARCH => "raymarch.hlsl",
    RAYMARCH_COMMON => "raymarch_common.hlsl",
    RAYMARCH_TYPES => "raymarch_types.hlsl",
    REACTIVE_MASK => "reactive_mask.hlsl",
    REFLECTION => "reflection.hlsl",
    REFLECTION_CUT => "reflection_cut.hlsl",
    RT_REFLECTIONS => "rt_reflections.hlsl",
    RT_SKIN => "rt_skin.hlsl",
    RT_TRACE => "rt_trace.hlsl",
    RT_TYPES => "rt_types.hlsl",
    SHADOW => "shadow.hlsl",
    SHADOW_BIAS => "shadow_bias.hlsl",
    SHADOW_CASCADE => "shadow_cascade.hlsl",
    SKY => "sky.hlsl",
    SKY_RAY => "sky_ray.hlsl",
    SPECULAR_AA => "specular_aa.hlsl",
    SSAO => "ssao.hlsl",
    SSGI => "ssgi.hlsl",
    SSR => "ssr.hlsl",
    SURFACE_FRAGMENT_DEFAULT => "surface_fragment_default.hlsl",
    SURFACE_INPUTS => "surface_inputs.hlsl",
    SURFACE_VERTEX_DEFAULT => "surface_vertex_default.hlsl",
    TAA => "taa.hlsl",
    TEXT => "text.hlsl",
    TEXTURE_SIZE => "texture_size.hlsl",
    TILE_MORPH => "tile_morph.hlsl",
    TILE_MORPH_TYPES => "tile_morph_types.hlsl",
    TRANSPARENT_PROBES => "transparent_probes.hlsl",
    TRANSPARENT_RT => "transparent_rt.hlsl",
    TRANSPARENT_SCENE => "transparent_scene.hlsl",
    TRANSPARENT_TYPES => "transparent_types.hlsl",
    VIEW_UNIFORMS => "view_uniforms.hlsl",
    WATER => "water.hlsl",
    WATER_FFT => "water_fft.hlsl",
    WIND => "wind.hlsl",
}

/// One shader's embedded text, by file name.
pub fn embedded(file: &str) -> Option<&'static str> {
    SOURCES
        .iter()
        .find_map(|(name, text)| (*name == file).then_some(*text))
}
