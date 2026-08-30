//! The single-source `.slang` engine shaders, embedded once for every consumer.
//!
//! Both halves of the toolchain read these: the renderer compiles them at init
//! (or loads a precompiled artifact keyed by their text), and the device build
//! script compiles the same text ahead of time. They live here, below the
//! backends, so the two can never disagree about what a program's source is --
//! the content-addressed shader cache keys on exactly this text.

/// Every embedded shader, as (file name, source). `embedded` looks up by name;
/// a build script iterating the set walks this table.
pub const SOURCES: &[(&str, &str)] = &[
    ("auto_exposure.slang", AUTO_EXPOSURE),
    ("bloom.slang", BLOOM),
    ("composite.slang", COMPOSITE),
    ("decal.slang", DECAL),
    ("fog.slang", FOG),
    ("fullscreen.slang", FULLSCREEN),
    ("gbuffer_prepass.slang", GBUFFER_PREPASS),
    ("glass.slang", GLASS),
    ("glass_mesh.slang", GLASS_MESH),
    ("hiz_build.slang", HIZ_BUILD),
    ("light_cull.slang", LIGHT_CULL),
    ("line.slang", LINE),
    ("main_bindless.slang", MAIN_BINDLESS),
    ("object_common.slang", OBJECT_COMMON),
    ("particle.slang", PARTICLE),
    ("particle_simulate.slang", PARTICLE_SIMULATE),
    ("particle_types.slang", PARTICLE_TYPES),
    ("post_common.slang", POST_COMMON),
    ("probe_common.slang", PROBE_COMMON),
    ("probe_prefilter.slang", PROBE_PREFILTER),
    ("probe_types.slang", PROBE_TYPES),
    ("reflection.slang", REFLECTION),
    ("rt_reflections.slang", RT_REFLECTIONS),
    ("rt_skin.slang", RT_SKIN),
    ("rt_trace.slang", RT_TRACE),
    ("rt_types.slang", RT_TYPES),
    ("shadow.slang", SHADOW),
    ("ssao.slang", SSAO),
    ("ssgi.slang", SSGI),
    ("ssr.slang", SSR),
    ("taa.slang", TAA),
    ("text.slang", TEXT),
    ("water.slang", WATER),
];

/// One shader's embedded text, by file name.
pub fn embedded(file: &str) -> Option<&'static str> {
    SOURCES
        .iter()
        .find_map(|(name, text)| (*name == file).then_some(*text))
}

/// `auto_exposure.slang`.
pub const AUTO_EXPOSURE: &str = include_str!("shaders/auto_exposure.slang");
/// `bloom.slang`.
pub const BLOOM: &str = include_str!("shaders/bloom.slang");
/// `composite.slang`.
pub const COMPOSITE: &str = include_str!("shaders/composite.slang");
/// `decal.slang`.
pub const DECAL: &str = include_str!("shaders/decal.slang");
/// `fog.slang`.
pub const FOG: &str = include_str!("shaders/fog.slang");
/// `fullscreen.slang`.
pub const FULLSCREEN: &str = include_str!("shaders/fullscreen.slang");
/// `gbuffer_prepass.slang`.
pub const GBUFFER_PREPASS: &str = include_str!("shaders/gbuffer_prepass.slang");
/// `glass.slang`.
pub const GLASS: &str = include_str!("shaders/glass.slang");
/// `glass_mesh.slang`.
pub const GLASS_MESH: &str = include_str!("shaders/glass_mesh.slang");
/// `hiz_build.slang`.
pub const HIZ_BUILD: &str = include_str!("shaders/hiz_build.slang");
/// `light_cull.slang`.
pub const LIGHT_CULL: &str = include_str!("shaders/light_cull.slang");
/// `line.slang`.
pub const LINE: &str = include_str!("shaders/line.slang");
/// `main_bindless.slang`.
pub const MAIN_BINDLESS: &str = include_str!("shaders/main_bindless.slang");
/// `object_common.slang`.
pub const OBJECT_COMMON: &str = include_str!("shaders/object_common.slang");
/// `particle.slang`.
pub const PARTICLE: &str = include_str!("shaders/particle.slang");
/// `particle_simulate.slang`.
pub const PARTICLE_SIMULATE: &str = include_str!("shaders/particle_simulate.slang");
/// `particle_types.slang`.
pub const PARTICLE_TYPES: &str = include_str!("shaders/particle_types.slang");
/// `post_common.slang`.
pub const POST_COMMON: &str = include_str!("shaders/post_common.slang");
/// `probe_common.slang`.
pub const PROBE_COMMON: &str = include_str!("shaders/probe_common.slang");
/// `probe_prefilter.slang`.
pub const PROBE_PREFILTER: &str = include_str!("shaders/probe_prefilter.slang");
/// `probe_types.slang`.
pub const PROBE_TYPES: &str = include_str!("shaders/probe_types.slang");
/// `reflection.slang`.
pub const REFLECTION: &str = include_str!("shaders/reflection.slang");
/// `rt_reflections.slang`.
pub const RT_REFLECTIONS: &str = include_str!("shaders/rt_reflections.slang");
/// `rt_skin.slang`.
pub const RT_SKIN: &str = include_str!("shaders/rt_skin.slang");
/// `rt_trace.slang`.
pub const RT_TRACE: &str = include_str!("shaders/rt_trace.slang");
/// `rt_types.slang`.
pub const RT_TYPES: &str = include_str!("shaders/rt_types.slang");
/// `shadow.slang`.
pub const SHADOW: &str = include_str!("shaders/shadow.slang");
/// `ssao.slang`.
pub const SSAO: &str = include_str!("shaders/ssao.slang");
/// `ssgi.slang`.
pub const SSGI: &str = include_str!("shaders/ssgi.slang");
/// `ssr.slang`.
pub const SSR: &str = include_str!("shaders/ssr.slang");
/// `taa.slang`.
pub const TAA: &str = include_str!("shaders/taa.slang");
/// `text.slang`.
pub const TEXT: &str = include_str!("shaders/text.slang");
/// `water.slang`.
pub const WATER: &str = include_str!("shaders/water.slang");
