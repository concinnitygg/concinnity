//! What the DirectX backend compiles to DXIL: the shared engine programs and
//! the single-pass Hi-Z downsampler.
//!
//! Each program compiles a file under `src/render/shaders/` to a signed DXIL
//! container, at build time where the host can and at renderer init otherwise,
//! cached in the content-addressed shader cache. dxc resolves dxil.dll itself,
//! so the containers it emits are already signed for D3D12.
//!
//! The bindless main pair's `CN_BACKEND_DIRECTX` branch reproduces the bindless
//! main root signature in `init/pipelines.rs` slot for slot: that layout is a
//! contract, since a world Shader asset builds its own PSO against the same root
//! signature (see world_shaders.rs). `assert_dxil_abi` in the device build
//! script locks every program's registers.
//!
//! A program's stage comes from its entry's `[shader("...")]` attribute, so the
//! one part of a DXIL profile left to state is the shader model: 6.0, the floor
//! `NonUniformResourceIndex` needs for the bindless pool, or 6.5 for the
//! programs in [`SHADER_MODEL_6_5`], as every entry reaching an inline
//! `RayQuery` must.

use super::{ShaderProgram, Table, shared, spd};

/// Everything the DirectX backend compiles.
pub static TABLE: Table = Table {
    programs: &[shared::ALL, spd::ALL],
    msaa: &[false, true],
};

/// The programs compiled at shader model 6.5 rather than 6.0: the ray-traced
/// fragments. Only the ray-query entries are raised, so the glass and water
/// vertex stages and the base fragments stay at 6.0.
pub static SHADER_MODEL_6_5: &[&ShaderProgram] = &[
    &shared::RT_REFLECTIONS_FRAG,
    &shared::RT_REFLECTIONS_FRAG_TEXTURED,
    &shared::GLASS_FRAG_RT,
    &shared::GLASS_FRAG_RT_TEXTURED,
    &shared::GLASS_REFLECTION_FRAG,
    &shared::GLASS_REFLECTION_FRAG_TEXTURED,
    &shared::GLASS_MESH_FRAG_RT,
    &shared::GLASS_MESH_FRAG_RT_TEXTURED,
    &shared::GLASS_MESH_REFLECTION_FRAG,
    &shared::GLASS_MESH_REFLECTION_FRAG_TEXTURED,
    &shared::WATER_FRAG_RT,
    &shared::WATER_FRAG_RT_TEXTURED,
];

/// Whether `program` compiles at shader model 6.5.
pub fn shader_model_6_5(program: &ShaderProgram) -> bool {
    SHADER_MODEL_6_5.iter().any(|p| core::ptr::eq(*p, program))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::Platform;

    // SM 6.5 is a ray-tracing requirement, so a program asking for it must
    // reach a `RayQuery`. A 6.0 program reaching one is dxc's to catch, and
    // fails the build.
    #[test]
    fn only_the_ray_traced_path_asks_for_shader_model_6_5() {
        for p in SHADER_MODEL_6_5 {
            assert!(
                TABLE.programs().any(|q| core::ptr::eq(q, *p)),
                "{}: not a DirectX program",
                p.label
            );
            let src = p.at(false).assemble(Platform::DirectX);
            assert!(
                src.contains("RayQuery<"),
                "{}: at SM 6.5 but reaches no RayQuery",
                p.label
            );
        }
        assert!(shader_model_6_5(&shared::WATER_FRAG_RT));
        assert!(!shader_model_6_5(&shared::WATER_VERT));
    }
}
