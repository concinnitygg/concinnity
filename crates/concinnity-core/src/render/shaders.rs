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

/// Every embedded shader, as (file name, source). `embedded` looks up by name;
/// a build script iterating the set walks this table.
pub const SOURCES: &[(&str, &str)] = &[
    ("auto_exposure.hlsl", AUTO_EXPOSURE),
    ("bloom.hlsl", BLOOM),
    ("cluster_types.hlsl", CLUSTER_TYPES),
    ("composite.hlsl", COMPOSITE),
    ("cull.hlsl", CULL),
    ("decal.hlsl", DECAL),
    ("fog.hlsl", FOG),
    ("fullscreen.hlsl", FULLSCREEN),
    ("gbuffer_prepass.hlsl", GBUFFER_PREPASS),
    ("glass.hlsl", GLASS),
    ("glass_mesh.hlsl", GLASS_MESH),
    ("glass_reflection.hlsl", GLASS_REFLECTION),
    ("hiz_build.hlsl", HIZ_BUILD),
    ("light_cull.hlsl", LIGHT_CULL),
    ("light_types.hlsl", LIGHT_TYPES),
    ("line.hlsl", LINE),
    ("main_bindless.hlsl", MAIN_BINDLESS),
    ("main_shading.hlsl", MAIN_SHADING),
    ("main_types.hlsl", MAIN_TYPES),
    ("model_history.hlsl", MODEL_HISTORY),
    ("object_common.hlsl", OBJECT_COMMON),
    ("particle.hlsl", PARTICLE),
    ("particle_simulate.hlsl", PARTICLE_SIMULATE),
    ("particle_types.hlsl", PARTICLE_TYPES),
    ("post_common.hlsl", POST_COMMON),
    ("probe_common.hlsl", PROBE_COMMON),
    ("probe_prefilter.hlsl", PROBE_PREFILTER),
    ("probe_types.hlsl", PROBE_TYPES),
    ("raymarch.hlsl", RAYMARCH),
    ("raymarch_common.hlsl", RAYMARCH_COMMON),
    ("raymarch_types.hlsl", RAYMARCH_TYPES),
    ("reflection.hlsl", REFLECTION),
    ("rt_reflections.hlsl", RT_REFLECTIONS),
    ("rt_skin.hlsl", RT_SKIN),
    ("rt_trace.hlsl", RT_TRACE),
    ("rt_types.hlsl", RT_TYPES),
    ("shadow.hlsl", SHADOW),
    ("shadow_bias.hlsl", SHADOW_BIAS),
    ("ssao.hlsl", SSAO),
    ("ssgi.hlsl", SSGI),
    ("ssr.hlsl", SSR),
    ("surface_fragment_default.hlsl", SURFACE_FRAGMENT_DEFAULT),
    ("surface_vertex_default.hlsl", SURFACE_VERTEX_DEFAULT),
    ("taa.hlsl", TAA),
    ("text.hlsl", TEXT),
    ("texture_size.hlsl", TEXTURE_SIZE),
    ("transparent_probes.hlsl", TRANSPARENT_PROBES),
    ("transparent_rt.hlsl", TRANSPARENT_RT),
    ("transparent_scene.hlsl", TRANSPARENT_SCENE),
    ("transparent_types.hlsl", TRANSPARENT_TYPES),
    ("water.hlsl", WATER),
];

/// One shader's embedded text, by file name.
pub fn embedded(file: &str) -> Option<&'static str> {
    SOURCES
        .iter()
        .find_map(|(name, text)| (*name == file).then_some(*text))
}

/// `auto_exposure.hlsl`.
pub const AUTO_EXPOSURE: &str = include_str!("shaders/auto_exposure.hlsl");
/// `bloom.hlsl`.
pub const BLOOM: &str = include_str!("shaders/bloom.hlsl");
/// `cluster_types.hlsl`.
pub const CLUSTER_TYPES: &str = include_str!("shaders/cluster_types.hlsl");
/// `composite.hlsl`.
pub const COMPOSITE: &str = include_str!("shaders/composite.hlsl");
/// `cull.hlsl`.
pub const CULL: &str = include_str!("shaders/cull.hlsl");
/// `decal.hlsl`.
pub const DECAL: &str = include_str!("shaders/decal.hlsl");
/// `fog.hlsl`.
pub const FOG: &str = include_str!("shaders/fog.hlsl");
/// `fullscreen.hlsl`.
pub const FULLSCREEN: &str = include_str!("shaders/fullscreen.hlsl");
/// `gbuffer_prepass.hlsl`.
pub const GBUFFER_PREPASS: &str = include_str!("shaders/gbuffer_prepass.hlsl");
/// `glass.hlsl`.
pub const GLASS: &str = include_str!("shaders/glass.hlsl");
/// `glass_mesh.hlsl`.
pub const GLASS_MESH: &str = include_str!("shaders/glass_mesh.hlsl");
/// `glass_reflection.hlsl`.
pub const GLASS_REFLECTION: &str = include_str!("shaders/glass_reflection.hlsl");
/// `hiz_build.hlsl`.
pub const HIZ_BUILD: &str = include_str!("shaders/hiz_build.hlsl");
/// `light_cull.hlsl`.
pub const LIGHT_CULL: &str = include_str!("shaders/light_cull.hlsl");
/// `light_types.hlsl`.
pub const LIGHT_TYPES: &str = include_str!("shaders/light_types.hlsl");
/// `line.hlsl`.
pub const LINE: &str = include_str!("shaders/line.hlsl");
/// `main_bindless.hlsl`.
pub const MAIN_BINDLESS: &str = include_str!("shaders/main_bindless.hlsl");
/// `main_shading.hlsl`.
pub const MAIN_SHADING: &str = include_str!("shaders/main_shading.hlsl");
/// `main_types.hlsl`.
pub const MAIN_TYPES: &str = include_str!("shaders/main_types.hlsl");
/// `model_history.hlsl`.
pub const MODEL_HISTORY: &str = include_str!("shaders/model_history.hlsl");
/// `object_common.hlsl`.
pub const OBJECT_COMMON: &str = include_str!("shaders/object_common.hlsl");
/// `particle.hlsl`.
pub const PARTICLE: &str = include_str!("shaders/particle.hlsl");
/// `particle_simulate.hlsl`.
pub const PARTICLE_SIMULATE: &str = include_str!("shaders/particle_simulate.hlsl");
/// `particle_types.hlsl`.
pub const PARTICLE_TYPES: &str = include_str!("shaders/particle_types.hlsl");
/// `post_common.hlsl`.
pub const POST_COMMON: &str = include_str!("shaders/post_common.hlsl");
/// `probe_common.hlsl`.
pub const PROBE_COMMON: &str = include_str!("shaders/probe_common.hlsl");
/// `probe_prefilter.hlsl`.
pub const PROBE_PREFILTER: &str = include_str!("shaders/probe_prefilter.hlsl");
/// `probe_types.hlsl`.
pub const PROBE_TYPES: &str = include_str!("shaders/probe_types.hlsl");
/// `raymarch.hlsl`.
pub const RAYMARCH: &str = include_str!("shaders/raymarch.hlsl");
/// `raymarch_common.hlsl`.
pub const RAYMARCH_COMMON: &str = include_str!("shaders/raymarch_common.hlsl");
/// `raymarch_types.hlsl`.
pub const RAYMARCH_TYPES: &str = include_str!("shaders/raymarch_types.hlsl");
/// `reflection.hlsl`.
pub const REFLECTION: &str = include_str!("shaders/reflection.hlsl");
/// `rt_reflections.hlsl`.
pub const RT_REFLECTIONS: &str = include_str!("shaders/rt_reflections.hlsl");
/// `rt_skin.hlsl`.
pub const RT_SKIN: &str = include_str!("shaders/rt_skin.hlsl");
/// `rt_trace.hlsl`.
pub const RT_TRACE: &str = include_str!("shaders/rt_trace.hlsl");
/// `rt_types.hlsl`.
pub const RT_TYPES: &str = include_str!("shaders/rt_types.hlsl");
/// `shadow.hlsl`.
pub const SHADOW: &str = include_str!("shaders/shadow.hlsl");
/// `shadow_bias.hlsl`.
pub const SHADOW_BIAS: &str = include_str!("shaders/shadow_bias.hlsl");
/// `ssao.hlsl`.
pub const SSAO: &str = include_str!("shaders/ssao.hlsl");
/// `ssgi.hlsl`.
pub const SSGI: &str = include_str!("shaders/ssgi.hlsl");
/// `ssr.hlsl`.
pub const SSR: &str = include_str!("shaders/ssr.hlsl");
/// `surface_fragment_default.hlsl`.
pub const SURFACE_FRAGMENT_DEFAULT: &str = include_str!("shaders/surface_fragment_default.hlsl");
/// `surface_vertex_default.hlsl`.
pub const SURFACE_VERTEX_DEFAULT: &str = include_str!("shaders/surface_vertex_default.hlsl");
/// `taa.hlsl`.
pub const TAA: &str = include_str!("shaders/taa.hlsl");
/// `text.hlsl`.
pub const TEXT: &str = include_str!("shaders/text.hlsl");
/// `texture_size.hlsl`.
pub const TEXTURE_SIZE: &str = include_str!("shaders/texture_size.hlsl");
/// `transparent_probes.hlsl`.
pub const TRANSPARENT_PROBES: &str = include_str!("shaders/transparent_probes.hlsl");
/// `transparent_rt.hlsl`.
pub const TRANSPARENT_RT: &str = include_str!("shaders/transparent_rt.hlsl");
/// `transparent_scene.hlsl`.
pub const TRANSPARENT_SCENE: &str = include_str!("shaders/transparent_scene.hlsl");
/// `transparent_types.hlsl`.
pub const TRANSPARENT_TYPES: &str = include_str!("shaders/transparent_types.hlsl");
/// `water.hlsl`.
pub const WATER: &str = include_str!("shaders/water.hlsl");

#[cfg(test)]
mod tests {
    use super::*;

    // The sky shell's half-extent tracks the camera far plane, so its corners
    // always fall outside it: every vertex path must pin sky verts to the far
    // plane or those corners clip and the clear color shows through. One
    // `project_vertex` serves every main-pass entry, so one check covers them.
    #[test]
    fn the_vertex_path_pins_sky_to_the_far_plane() {
        assert!(MAIN_SHADING.contains("color.b > 1.5"));
        assert!(MAIN_SHADING.contains("o.position.z = o.position.w"));
    }

    // The pre-pass rasterizes the same visible set the main pass does, sky
    // shell included, so an unpinned sky vert clips and the G-buffer loses
    // coverage the main pass has. The bindless vertex entry is the only one
    // that carries skybox geometry, so the two matches are the pin's definition
    // and its one call.
    #[test]
    fn the_prepass_pins_sky_to_the_far_plane() {
        assert!(GBUFFER_PREPASS.contains("color.b > 1.5"));
        assert!(GBUFFER_PREPASS.contains("position.z = position.w"));
        assert_eq!(GBUFFER_PREPASS.matches("gb_sky_pin(").count(), 2);
    }
}
