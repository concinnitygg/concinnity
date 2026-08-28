// src/render_graph/passes.rs
//
// Stable identity for every render-graph pass. Used by:
//
//   - The graph itself, as the dispatch key the executor matches on.
//   - The per-pass GPU timer (`crate::pass_timing`), which keys its
//     sample-buffer slots off the same integer.
//
// The `pass_ids!` invocation below is the single registration point: one line
// per pass names the variant and its stable timing name, and the macro derives
// the enum, [`PASS_NAMES`], [`PASS_COUNT`], and [`PassId::ALL`] from it. A pass
// therefore cannot exist without a timing name (which would otherwise report
// zero GPU time), and the name table cannot drift out of index order.
//
// Variants are `#[repr(u32)]` so a `PassId` round-trips through `as usize` into
// [`PASS_NAMES`] and any `[T; PASS_COUNT]` companion array. The list is
// append-only: inserting in the middle renumbers later variants and silently
// shifts every timing slot.

/// Declare the pass vocabulary. Each entry is `Variant => "timing_name"`.
macro_rules! pass_ids {
    ($($(#[$doc:meta])* $variant:ident => $name:literal,)*) => {
        /// One per-pass identity. Cast to `usize` to index [`PASS_NAMES`] or any
        /// `[T; PASS_COUNT]` companion array.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(u32)]
        pub enum PassId {
            $($(#[$doc])* $variant,)*
        }

        /// Stable display name for each pass. Index = `PassId as usize`. Used by
        /// the WS `profile.passes` reply and the per-pass timing readback.
        pub const PASS_NAMES: [&str; PASS_COUNT] = [$($name,)*];

        /// Number of distinct passes the engine times. The per-pass timing array
        /// in [`crate::profile::RenderStats`] is sized to at least this many slots.
        pub const PASS_COUNT: usize = [$(PassId::$variant,)*].len();

        impl PassId {
            /// Every variant, in declaration (index) order.
            pub const ALL: [PassId; PASS_COUNT] = [$(PassId::$variant,)*];
        }
    };
}

pass_ids! {
    /// GPU visibility cull; produces the indirect draw arguments.
    Cull => "cull",
    /// Directional shadow cascades and spot shadow slices.
    Shadow => "shadow",
    /// Depth / normal prepass feeding screen-space reflections.
    SsrPrepass => "ssr_prepass",
    /// Depth / normal prepass feeding ambient occlusion.
    SsaoPrepass => "ssao_prepass",
    /// The ambient-occlusion gather.
    SsaoKernel => "ssao_kernel",
    /// Bilateral blur over the occlusion buffer.
    SsaoBlur => "ssao_blur",
    /// The lit forward pass.
    Main => "main",
    /// Luminance reduction driving auto-exposure.
    AutoExposure => "auto_exposure",
    /// Projected decals.
    Decals => "decals",
    /// Volumetric fog composite.
    Fog => "fog",
    /// Particle simulation compute.
    ParticlesSim => "particles_sim",
    /// Particle draw.
    ParticlesDraw => "particles_draw",
    /// Reflection trace and composite.
    SsrResolve => "ssr_resolve",
    /// Screen-space velocity, for TAA and motion blur.
    Velocity => "velocity",
    /// Temporal anti-aliasing resolve.
    TaaResolve => "taa_resolve",
    /// Bloom down/upsample chain.
    Bloom => "bloom",
    /// Tonemap, grade, and present.
    Composite => "composite",
    /// Volumetric-fog froxel-volume compute pass. Populates a 3D
    /// `(scattered, transmittance)` texture once per frame, sampled by the
    /// fullscreen `Fog` render pass instead of an inline ray-march. Every
    /// backend implements the path; the `Fog` pass trilinear-samples the
    /// volume by (screen_uv, view_z).
    FogFroxel => "fog_froxel",
    /// Temporal upscaling pass. When the world's `PostProcessConfig`
    /// enables `temporal_upscaling`, the renderer draws the 3D scene at a
    /// fraction of drawable size and inserts this pass between the post-SSR
    /// scene and the Bloom + Composite stack. The backend runs its
    /// platform-native temporal upscaler (MetalFX on macOS; FSR / DLSS /
    /// XeSS slots on the Windows backends are placeholders today) to
    /// reconstruct a drawable-resolution image. Replaces `TaaResolve`:
    /// the upscaler does temporal accumulation itself, so adding both
    /// would double-temporal the scene.
    Upscale => "upscale",
    /// Transparent / translucent geometry pass. Runs after `SsrResolve`
    /// (so water + glass see opaque reflections) and before
    /// `TaaResolve` / `Upscale` (so translucents pick up temporal
    /// accumulation). Reads the latest scene-pre-taa colour + main
    /// depth as sampled textures; writes scene-pre-taa blended
    /// (SRC_ALPHA / ONE_MINUS_SRC_ALPHA). Each transparent draw owns
    /// its own pipeline + descriptor set; the pass aggregates them as
    /// a back-to-front sorted list at encode time. Gated on
    /// `FrameGraphInputs::transparent_enabled`; when no consumer is
    /// in the world, the slot is omitted entirely.
    Transparent => "transparent",
    /// Raymarched SDF volume pass. Rasterises the back faces of each
    /// `SdfVolume`'s world-space bounding box and runs the user-authored
    /// fragment shader, which sphere-traces a signed distance field
    /// inside the box. Hit fragments write opaque colour into
    /// `hdr_resolve` (RMW between `AutoExposure` and `Decals`) and
    /// update the main depth attachment so the raymarched surface
    /// composites with rasterised geometry naturally: decals, fog,
    /// SSR-resolve, and TAA all consume the post-Raymarch depth and
    /// colour. Gated on `FrameGraphInputs::raymarch_enabled`; when no
    /// `SdfVolume` is in the world the slot is omitted entirely.
    Raymarch => "raymarch",
    /// Mid-frame Hi-Z (depth-mip pyramid) rebuild for two-pass occlusion
    /// culling. Inserted only when `FrameGraphInputs::two_pass_occlusion_enabled`
    /// is on: after `Main` (phase 1) has written this frame's depth, this
    /// compute pass reduces it into the Hi-Z pyramid so `Cull2` can re-test
    /// the objects phase 1 occluded against up-to-date depth. Distinct from
    /// the end-of-frame Hi-Z build (which feeds the *next* frame's phase-1
    /// cull and stays an inline action, not a graph node). Every backend
    /// implements the node; whether it appears is `two_pass_occlusion_enabled`,
    /// which each seeds from its own two-pass state.
    HizBuild => "hiz_build",
    /// Phase-2 GPU cull for two-pass occlusion. Re-tests the objects `Cull`
    /// (phase 1) marked Hi-Z-occluded against the freshly rebuilt pyramid
    /// (`HizBuild`) and encodes a draw for any that turn out visible into a
    /// second indirect command buffer `Main2` consumes. Reads the per-object
    /// status buffer phase-1 cull wrote + the `draw_args2` buffer it writes.
    /// Gated on `FrameGraphInputs::two_pass_occlusion_enabled`.
    Cull2 => "cull2",
    /// Phase-2 main pass for two-pass occlusion. Loads (does not clear) the
    /// HDR colour + depth `Main` wrote and re-runs only the bindless-static
    /// indirect draw through `Cull2`'s command buffer, depth-compositing the
    /// disoccluded geometry with phase 1. Instanced + skinned geometry is not
    /// Hi-Z-culled, so it is fully drawn in phase 1 and not repeated here.
    /// Becomes the new head of the hdr_resolve post-decoration chain (so
    /// AutoExposure / Decals / Fog / SSR see the combined result). Gated on
    /// `FrameGraphInputs::two_pass_occlusion_enabled`.
    Main2 => "main2",
    /// Screen-space global illumination. A refinement of SSR: it reuses the
    /// SSR depth + normal pre-pass G-buffer and screen-space ray-march, but
    /// integrates bounced radiance over a cosine-weighted hemisphere instead
    /// of along one reflection vector. Sits on the hdr_resolve RMW chain (after
    /// `Raymarch`, before `Decals`): it reads the lit scene as the bounce
    /// radiance source and additively composites the gathered + denoised
    /// indirect term back into it, so the near-field colour bleed layers on top
    /// of the IBL ambient. Gated on `FrameGraphInputs::ssgi_enabled`; when
    /// `indirect_lighting` is IBL-only the slot is omitted entirely.
    Ssgi => "ssgi",
    /// Hardware ray-traced reflections. Occupies the same scene-pre-taa slot as
    /// `SsrResolve` (reads the post-decoration `hdr_resolve`, writes
    /// `scene_pre_taa`) and takes precedence over it: when this pass is live the
    /// builder inserts it and omits `SsrResolve` (a world may author both; RT
    /// runs where available, SSR is the fallback). It still relies on the SSR
    /// depth + normal + roughness
    /// pre-pass (so `SsrPrepass` is forced on), but instead of a screen-space
    /// march it traces a world-space reflection ray against an acceleration
    /// structure built over the static scene geometry, so off-screen reflected
    /// geometry appears. Gated on `FrameGraphInputs::rt_reflections_enabled`,
    /// and so only on GPUs that report ray-tracing support.
    RtReflections => "rt_reflections",
    /// Unified geometry G-buffer pre-pass. One jittered traversal of the visible
    /// set writes view-space normal + linear depth, perceptual roughness, and
    /// screen-space motion into a single MRT (plus a sampleable depth), replacing
    /// the separate `SsrPrepass` + `Velocity` (and the SSAO-owned prepass): every
    /// consumer (SSR, SSAO, SSGI, RT, TAA, upscaler) reads this one output. Gated
    /// on `FrameGraphInputs::unified_gbuffer_prepass`.
    GBufferPrepass => "gbuffer_prepass",
    /// Roughness-aware reflection composite. Not a standalone graph node: it is
    /// encoded inline at the tail of the `SsrResolve` / `RtReflections` pass
    /// (both write a reflection target, then blur it by roughness and composite
    /// it over the scene). It carries its own timing slot so its cost is visible
    /// separately from the trace/march that precedes it. Inline on every
    /// backend, so no backend's graph executor ever dispatches this id: each
    /// treats it as a programming error the way it treats the bundled SSAO
    /// sub-passes.
    ReflectionComposite => "reflection_composite",
    /// Clustered light-binning compute pass. Once per frame, before Main: bins the
    /// scene's local lights (the GpuLight buffer) into a per-cluster index list
    /// over a screen-tiled, exponential-depth froxel grid, which the forward pass
    /// reads to shade each fragment from only its cluster's lights instead of
    /// iterating every light. Runs when the world has local lights. Writes a
    /// storage buffer Main reads (RAW edge).
    LightCull => "light_cull",
    /// Depth-only render of each shadowed spot light's cone into one slice of the
    /// spot shadow map array. Local lights are static, so the projections are
    /// built once; only the depth contents refresh, one slice per frame under
    /// `ShadowUpdate::Hybrid`. Runs when the world has a shadow-casting spot.
    SpotShadow => "spot_shadow",
    /// World-space line geometry (trajectories, tethers, path previews, the
    /// editor's origin axes). Blend-writes the resolved scene colour after the
    /// world decorations, sampling the resolved scene depth so a line behind
    /// geometry is occluded by it. Gated on `FrameGraphInputs::lines_enabled`:
    /// a frame that submits no lines omits the node entirely, so a frame that
    /// draws none never pays for it.
    Lines => "lines",
    /// Terminal Hi-Z (depth-mip pyramid) build. Reduces the frame's final main
    /// depth into the pyramid the *next* frame's phase-1 `Cull` tests against,
    /// so it is declared last and reads the depth every decoration pass has
    /// finished with. Distinct from `HizBuild`, which rebuilds the same pyramid
    /// mid-frame from phase-1 depth for `Cull2`; when two-pass occlusion is on
    /// both run and this one supersedes it for the next frame. Present whenever
    /// the GPU-cull path built a pyramid (`FrameGraphInputs::hiz_build_enabled`).
    HizFinal => "hiz_final",
}

impl PassId {
    /// Stable display name, looked up in [`PASS_NAMES`]. `'static` since
    /// the table is `const`.
    pub fn name(self) -> &'static str {
        PASS_NAMES[self as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_pass_id_indexes_its_own_name() {
        // The macro pairs each variant with its name, so this asserts the
        // derivation rather than a hand-maintained mirror: indices stay dense
        // and in declaration order, and no pass ships nameless.
        for (i, &pass) in PassId::ALL.iter().enumerate() {
            assert_eq!(pass as usize, i, "{pass:?} index out of order");
            assert_eq!(pass.name(), PASS_NAMES[i], "{pass:?} name table mismatch");
            assert!(!pass.name().is_empty(), "{pass:?} has an empty name");
        }
        assert_eq!(PASS_NAMES.len(), PASS_COUNT);
    }

    #[test]
    fn pass_names_are_unique() {
        // A copy-pasted name would silently merge two passes in the profiler.
        let mut seen = hashbrown::HashSet::new();
        for &name in PASS_NAMES.iter() {
            assert!(seen.insert(name), "duplicate pass name {name:?}");
        }
    }
}
