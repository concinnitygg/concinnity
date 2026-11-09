//! Stable identity for every render-graph pass, and for every GPU timing slot.
//!
//! [`PassId`] names a graph node: the key the compiled graph carries and each
//! backend's executor dispatches on. [`GpuSpan`] names work the GPU timer
//! measures on its own without being a graph node: a sub-pass bundled into
//! one node's encode, or a draw timed inside another pass. [`TimingId`] is
//! either one, and is what the per-pass GPU timer
//! (`crate::render::backend::pass_timing`) keys its sample slots off.
//!
//! The `pass_ids!` and `gpu_spans!` invocations below are the single
//! registration points: one line per entry names the variant and its stable
//! timing name, so nothing can be timed without a name and the name tables
//! cannot drift out of index order.

/// Declare the pass vocabulary. Each entry is `Variant => "timing_name"`.
macro_rules! pass_ids {
    ($($(#[$doc:meta])* $variant:ident => $name:literal,)*) => {
        /// One render-graph pass. Cast to `usize` to index [`PASS_NAMES`] or any
        /// `[T; PASS_COUNT]` companion array.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(u8)]
        pub enum PassId {
            $($(#[$doc])* $variant,)*
        }

        /// Stable display name for each pass. Index = `PassId as usize`.
        pub const PASS_NAMES: [&str; PASS_COUNT] = [$($name,)*];

        /// Number of distinct graph passes.
        pub const PASS_COUNT: usize = [$(PassId::$variant,)*].len();

        impl PassId {
            /// Every variant, in declaration (index) order.
            pub const ALL: [PassId; PASS_COUNT] = [$(PassId::$variant,)*];
        }
    };
}

/// Declare the timing-only spans. Each entry is
/// `Variant => "timing_name" in Enclosing` or `Variant => "timing_name"`.
macro_rules! gpu_spans {
    ($($(#[$doc:meta])* $variant:ident => $name:literal $(in $outer:ident)?,)*) => {
        /// GPU work timed on its own that is not a graph node.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(u8)]
        pub enum GpuSpan {
            $($(#[$doc])* $variant,)*
        }

        /// Stable display name for each span. Index = `GpuSpan as usize`.
        pub const SPAN_NAMES: [&str; SPAN_COUNT] = [$($name,)*];

        /// Number of distinct spans.
        pub const SPAN_COUNT: usize = [$(GpuSpan::$variant,)*].len();

        impl GpuSpan {
            /// Every variant, in declaration (index) order.
            pub const ALL: [GpuSpan; SPAN_COUNT] = [$(GpuSpan::$variant,)*];

            /// The pass whose timing span contains this one's. Its time is then
            /// already part of that pass's, so a breakdown of the frame lists it
            /// under that pass rather than beside it.
            pub const fn enclosing(self) -> Option<PassId> {
                match self {
                    $(Self::$variant => gpu_spans!(@outer $($outer)?),)*
                }
            }
        }
    };
    (@outer $outer:ident) => { Some(PassId::$outer) };
    (@outer) => { None };
}

pass_ids! {
    /// GPU visibility cull; produces the indirect draw arguments.
    Cull => "cull",
    /// Directional shadow cascades and spot shadow slices.
    Shadow => "shadow",
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
    /// Particle draw, under the see-through surfaces.
    ParticlesDraw => "particles_draw",
    /// Particle draw over the see-through surfaces.
    ParticlesLateDraw => "particles_late_draw",
    /// Reflection trace and composite.
    SsrResolve => "ssr_resolve",
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
    /// volume by (`screen_uv`, `view_z`).
    FogFroxel => "fog_froxel",
    /// Temporal upscaling pass. When the world's `AntiAliasing` enables
    /// `upscaling`, the renderer draws the 3D scene at a
    /// fraction of drawable size and inserts this pass between the post-SSR
    /// scene and the Bloom + Composite stack. The backend runs its
    /// platform-native temporal upscaler (`MetalFX` on macOS; FSR / DLSS /
    /// XeSS slots on the Windows backends are placeholders today) to
    /// reconstruct a drawable-resolution image. Replaces `TaaResolve`:
    /// the upscaler does temporal accumulation itself, so adding both
    /// would double-temporal the scene.
    Upscale => "upscale",
    /// Transparent / translucent geometry pass. Runs after `SsrResolve`
    /// (so water + glass see opaque reflections) and before
    /// `TaaResolve` / `Upscale` (so translucents pick up temporal
    /// accumulation). Reads the latest scene-pre-taa color + main
    /// depth as sampled textures; writes scene-pre-taa blended
    /// (`SRC_ALPHA` / `ONE_MINUS_SRC_ALPHA`). Each transparent draw owns
    /// its own pipeline + descriptor set; the pass aggregates them as
    /// a back-to-front sorted list at encode time. Gated on
    /// `FrameGraphInputs::transparent_enabled`; when no consumer is
    /// in the world, the slot is omitted entirely.
    Transparent => "transparent",
    /// Raymarched SDF volume pass. Rasterizes the back faces of each
    /// `SdfVolume`'s world-space bounding box and runs the user-authored
    /// fragment shader, which sphere-traces a signed distance field
    /// inside the box. Hit fragments write opaque color into
    /// `hdr_resolve` (RMW between `AutoExposure` and `Decals`) and
    /// update the main depth attachment so the raymarched surface
    /// composites with rasterized geometry naturally: decals, fog,
    /// SSR-resolve, and TAA all consume the post-Raymarch depth and
    /// color. Gated on `FrameGraphInputs::raymarch_enabled`; when no
    /// `SdfVolume` is in the world the slot is omitted entirely.
    Raymarch => "raymarch",
    /// Mid-frame Hi-Z (depth-mip pyramid) rebuild for two-pass occlusion
    /// culling. Inserted only when `FrameGraphInputs::two_pass_occlusion_enabled`
    /// is on: after `Main` (the early pass) has written this frame's depth, this
    /// compute pass reduces it into the Hi-Z pyramid so `CullLate` can re-test
    /// the objects the early pass occluded against up-to-date depth. Distinct from
    /// the end-of-frame Hi-Z build (which feeds the *next* frame's early
    /// cull and stays an inline action, not a graph node). Every backend
    /// implements the node; whether it appears is `two_pass_occlusion_enabled`,
    /// which each seeds from its own two-pass state.
    HizEarly => "hiz_early",
    /// Late GPU cull for two-pass occlusion. Re-tests the objects `Cull`
    /// (the early pass) marked Hi-Z-occluded against the freshly rebuilt pyramid
    /// (`HizEarly`) and encodes a draw for any that turn out visible into a
    /// second indirect command buffer `MainLate` consumes. Reads the per-object
    /// status buffer early cull wrote + the `draw_args_late` buffer it writes.
    /// Gated on `FrameGraphInputs::two_pass_occlusion_enabled`.
    CullLate => "cull_late",
    /// Late main pass for two-pass occlusion. Loads (does not clear) the
    /// HDR color + depth `Main` wrote and re-runs only the bindless-static
    /// indirect draw through `CullLate`'s command buffer, depth-compositing the
    /// disoccluded geometry with the early pass. Instanced + skinned geometry is not
    /// Hi-Z-culled, so it is fully drawn in the early pass and not repeated here.
    /// Becomes the new head of the `hdr_resolve` post-decoration chain (so
    /// `AutoExposure` / Decals / Fog / SSR see the combined result). Gated on
    /// `FrameGraphInputs::two_pass_occlusion_enabled`.
    MainLate => "main_late",
    /// Screen-space global illumination. A refinement of SSR: it reuses the
    /// SSR depth + normal pre-pass G-buffer and screen-space ray-march, but
    /// integrates bounced radiance over a cosine-weighted hemisphere instead
    /// of along one reflection vector. Reads the lit scene after `Raymarch`
    /// as the bounce radiance source and accumulates what it gathers in its
    /// own targets, which the next frame's `SsgiResolve` hands the main pass.
    /// Gated on `FrameGraphInputs::ssgi_enabled`; when `indirect_lighting` is
    /// IBL-only the slot is omitted entirely.
    Ssgi => "ssgi",
    /// The SSGI accumulation reprojected onto this frame's G-buffer and
    /// upsampled to full resolution, ahead of `Main`: the environment's
    /// visibility and the bounce the main pass shades as indirect diffuse
    /// light. Present with `Ssgi`.
    SsgiResolve => "ssgi_resolve",
    /// Hardware ray-traced reflections. Occupies the same scene-pre-taa slot as
    /// `SsrResolve` (reads the post-decoration `hdr_resolve`, writes
    /// `scene_pre_taa`) and takes precedence over it: when this pass is live the
    /// builder inserts it and omits `SsrResolve` (a world may author both; RT
    /// runs where available, SSR is the fallback). It still relies on the SSR
    /// depth + normal + roughness
    /// pre-pass (so `ssr_prepass_enabled` is forced on), but instead of a screen-space
    /// march it traces a world-space reflection ray against an acceleration
    /// structure built over the static scene geometry, so off-screen reflected
    /// geometry appears. Gated on `FrameGraphInputs::rt_reflections_enabled`,
    /// and so only on GPUs that report ray-tracing support.
    RtReflections => "rt_reflections",
    /// Geometry G-buffer pre-pass. One jittered traversal of the visible set
    /// writes view-space normal + linear depth, perceptual roughness, and
    /// screen-space motion into a single MRT (plus a sampleable depth): every
    /// consumer (SSR, SSAO, SSGI, RT, TAA, upscaler) reads this one output.
    /// Gated on `FrameGraphInputs::gbuffer_prepass_enabled` plus any consumer.
    GBufferPrepass => "gbuffer_prepass",
    /// Clustered light-binning compute pass. Once per frame, before Main: bins the
    /// scene's local lights (the `GpuLight` buffer) into a per-cluster index list
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
    /// editor's origin axes). Blend-writes the resolved scene color after the
    /// world decorations, sampling the resolved scene depth so a line behind
    /// geometry is occluded by it. Gated on `FrameGraphInputs::lines_enabled`:
    /// a frame that submits no lines omits the node entirely, so a frame that
    /// draws none never pays for it.
    Lines => "lines",
    /// Terminal Hi-Z (depth-mip pyramid) build. Reduces the frame's final main
    /// depth into the pyramid the *next* frame's early `Cull` tests against,
    /// so it is declared last and reads the depth every decoration pass has
    /// finished with. Distinct from `HizEarly`, which rebuilds the same pyramid
    /// mid-frame from early depth for `CullLate`; when two-pass occlusion is on
    /// both run and this one supersedes it for the next frame. Present whenever
    /// the GPU-cull path built a pyramid (`FrameGraphInputs::hiz_build_enabled`).
    HizFinal => "hiz_final",
    /// Mirror renders for flat reflectors (water surfaces, glass panes): the
    /// scene re-rendered from the camera reflected across each distinct
    /// reflector plane, cropped to the screen rectangle that plane's reflectors
    /// cover this frame. Writes the mirror targets `Transparent` samples. A plane
    /// whose reflectors are all off screen renders nothing, so the node costs
    /// nothing on a frame with no reflector in view. Gated on
    /// `FrameGraphInputs::planar_reflection_enabled`.
    PlanarReflection => "planar_reflection",
    /// Grass blade generation. A compute pass that places the grass field's
    /// blades in the tiles around the camera, culls them against the view
    /// frustum and the draw distance, and appends the survivors to a
    /// visible-blade buffer with the indirect draw arguments the grass draws in
    /// `GBufferPrepass` and `Main` consume. Gated on
    /// `FrameGraphInputs::grass_enabled`.
    Grass => "grass",
    /// The grass bend field: a compute pass that relaxes the camera-centered
    /// field the blades are pressed down by and stamps every nearby
    /// character's and body's footprint into it, ahead of the passes that
    /// place blades. Gated on `FrameGraphInputs::grass_enabled`.
    GrassBend => "grass_bend",
    /// Grass blade placement for the nearest shadow cascade: the grass kernel
    /// run again over the tiles near the camera, culled against the cascade's
    /// light frustum rather than the view, so blades off screen still cast.
    /// Gated on `FrameGraphInputs::grass_shadow_enabled`.
    GrassShadow => "grass_shadow",
    /// The atmosphere's lookup tables: a compute pass that builds the sky-view
    /// table the sky draws from, the aerial-perspective volume the haze pass
    /// reads and the irradiance cube the surfaces take their ambient light
    /// from (and, when the air changes, the transmittance and
    /// multiple-scattering tables those read). Gated on
    /// `FrameGraphInputs::atmosphere_enabled`.
    Atmosphere => "atmosphere",
    /// Aerial perspective: the air between the camera and each opaque surface,
    /// blended over the scene from the atmosphere's volume. Sits on the
    /// `hdr_resolve` RMW chain just before `Fog`, so a volumetric fog stays the
    /// nearer layer. Gated on `FrameGraphInputs::atmosphere_enabled`.
    AerialPerspective => "aerial_perspective",
    /// The simulated ocean: a compute pass that evolves every large water
    /// body's spectrum to this frame, runs its inverse FFTs and reduces the
    /// resulting displacement and slope arrays down their mips, which
    /// `Transparent` samples in both the water's vertex and fragment stages.
    /// It reads nothing else in the frame, so the schedule may run it on the
    /// async queue. Gated on `FrameGraphInputs::water_sim_enabled`.
    WaterSim => "water_sim",
    /// The reactive mask's disocclusion term: after `GBufferPrepass`, each
    /// pixel followed back along its motion vector into last frame's G-buffer
    /// snapshot and marked where no surface there matches it by depth and
    /// normal, so the upscaler drops the history an object leaves on what it
    /// uncovers. The mask's first writer. Gated on
    /// `FrameGraphInputs::disocclusion_enabled` and an upscaler that reads the
    /// mask.
    Disocclusion => "disocclusion",
    /// One cube face of the scene seen from the prop a prop reflection probe
    /// rides on, its own draws left out, into the probe's capture; on the
    /// cycle's sixth face the capture is convolved into the probe's cube of
    /// the probe array. After the shadow maps and the cull, ahead of `Main`.
    /// Gated on `FrameGraphInputs::prop_probe_enabled`.
    PropProbe => "prop_probe",
    /// The clouds marched at half resolution: a compute pass that walks each
    /// view ray through the atmosphere's cloud layers up to the scene's depth
    /// and writes their hazed light and transmittance to the cloud target.
    /// After the atmosphere's tables and the main depth, ahead of
    /// `CloudComposite`. Gated on `FrameGraphInputs::clouds_enabled`.
    Clouds => "clouds",
    /// The clouds blended over the scene: a fullscreen pass on the pre-TAA
    /// scene after `Transparent`, so the haze on the surfaces behind a cloud
    /// and the sea beneath it are dimmed by it, upsampling the cloud target by
    /// depth. Gated on `FrameGraphInputs::clouds_enabled`.
    CloudComposite => "cloud_composite",
}

gpu_spans! {
    /// The single-channel depth copy ambient occlusion samples, inside the
    /// `SsaoBlur` node's encode.
    SsaoDepth => "ssao_depth",
    /// The ambient-occlusion gather, inside the `SsaoBlur` node's encode.
    SsaoKernel => "ssao_kernel",
    /// The roughness-aware reflection blur and composite at the tail of the
    /// `SsrResolve` / `RtReflections` node.
    ReflectionComposite => "reflection_composite",
    /// The environment drawn as the background at the tail of each pass that
    /// renders the opaque scene. Metal samples only at encoder boundaries and
    /// counts it in `Main` alone.
    Sky => "sky" in Main,
    /// The grass blades drawn at the tail of `Main`'s opaque geometry.
    GrassDraw => "grass_draw" in Main,
    /// The grass blades drawn into the G-buffer pre-pass.
    GrassPrepass => "grass_prepass" in GBufferPrepass,
    /// The grass blades drawn into the nearest shadow cascade.
    GrassShadowDraw => "grass_shadow_draw" in Shadow,
}

/// Number of GPU timing slots: one per pass, then one per span.
pub const TIMING_COUNT: usize = PASS_COUNT + SPAN_COUNT;

/// Stable display name for each timing slot. Index = [`TimingId::index`].
pub const TIMING_NAMES: [&str; TIMING_COUNT] = {
    let mut out = [""; TIMING_COUNT];
    let mut i = 0;
    while i < PASS_COUNT {
        out[i] = PASS_NAMES[i];
        i += 1;
    }
    let mut j = 0;
    while j < SPAN_COUNT {
        out[PASS_COUNT + j] = SPAN_NAMES[j];
        j += 1;
    }
    out
};

/// One GPU timing slot: a graph pass, or a span timed apart from any node.
/// Passes take the first [`PASS_COUNT`] slots and spans the rest.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimingId(u8);

impl TimingId {
    /// Dense index into [`TIMING_NAMES`] or any `[T; TIMING_COUNT]` array.
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// The slot at `index`, if there is one.
    pub fn from_index(index: usize) -> Option<Self> {
        (index < TIMING_COUNT).then(|| Self(u8::try_from(index).unwrap_or(u8::MAX)))
    }

    /// Stable display name.
    pub fn name(self) -> &'static str {
        TIMING_NAMES[self.index()]
    }

    /// The slot with this stable name.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::all().find(|t| t.name() == name)
    }

    /// Every slot, in index order.
    pub fn all() -> impl Iterator<Item = Self> {
        PassId::ALL
            .into_iter()
            .map(PassId::timing)
            .chain(GpuSpan::ALL.into_iter().map(GpuSpan::timing))
    }

    /// The graph pass this slot times, `None` for a span.
    pub fn pass(self) -> Option<PassId> {
        PassId::ALL.get(self.index()).copied()
    }

    /// The pass whose timing contains this slot's (see [`GpuSpan::enclosing`]).
    pub fn enclosing(self) -> Option<PassId> {
        let span = self.index().checked_sub(PASS_COUNT)?;
        GpuSpan::ALL.get(span).and_then(|s| s.enclosing())
    }
}

impl core::fmt::Debug for TimingId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

const _: () = assert!(TIMING_COUNT <= u8::MAX as usize);

impl PassId {
    /// Stable display name, looked up in [`PASS_NAMES`].
    pub fn name(self) -> &'static str {
        PASS_NAMES[self as usize]
    }

    /// The pass with this stable name.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.name() == name)
    }

    /// This pass's GPU timing slot.
    pub const fn timing(self) -> TimingId {
        TimingId(self as u8)
    }
}

impl GpuSpan {
    /// Stable display name, looked up in [`SPAN_NAMES`].
    pub fn name(self) -> &'static str {
        SPAN_NAMES[self as usize]
    }

    /// This span's GPU timing slot.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "TIMING_COUNT fits a u8, asserted above"
    )]
    pub const fn timing(self) -> TimingId {
        TimingId((PASS_COUNT + self as usize) as u8)
    }
}

impl From<PassId> for TimingId {
    fn from(pass: PassId) -> Self {
        pass.timing()
    }
}

impl From<GpuSpan> for TimingId {
    fn from(span: GpuSpan) -> Self {
        span.timing()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_pass_id_indexes_its_own_name() {
        for (i, &pass) in PassId::ALL.iter().enumerate() {
            assert_eq!(pass as usize, i, "{pass:?} index out of order");
            assert_eq!(pass.name(), PASS_NAMES[i], "{pass:?} name table mismatch");
            assert!(!pass.name().is_empty(), "{pass:?} has an empty name");
        }
        assert_eq!(PASS_NAMES.len(), PASS_COUNT);
    }

    #[test]
    fn every_pass_is_found_by_its_name() {
        for pass in PassId::ALL {
            assert_eq!(PassId::from_name(pass.name()), Some(pass));
        }
        assert_eq!(PassId::from_name("not a pass"), None);
    }

    #[test]
    fn timing_slots_put_passes_first_then_spans() {
        let all: alloc::vec::Vec<TimingId> = TimingId::all().collect();
        assert_eq!(all.len(), TIMING_COUNT);
        for (i, t) in all.iter().enumerate() {
            assert_eq!(t.index(), i);
            assert_eq!(TimingId::from_index(i), Some(*t));
            assert_eq!(TimingId::from_name(t.name()), Some(*t));
        }
        assert_eq!(TimingId::from_index(TIMING_COUNT), None);
        assert_eq!(PassId::Cull.timing().pass(), Some(PassId::Cull));
        assert_eq!(GpuSpan::Sky.timing().pass(), None);
        assert_eq!(GpuSpan::SsaoDepth.timing().index(), PASS_COUNT);
        assert_eq!(GpuSpan::Sky.timing().name(), "sky");
    }

    #[test]
    fn spans_nest_inside_a_pass_and_passes_nest_in_nothing() {
        for pass in PassId::ALL {
            assert_eq!(pass.timing().enclosing(), None, "{pass:?}");
        }
        assert_eq!(GpuSpan::Sky.enclosing(), Some(PassId::Main));
        assert_eq!(GpuSpan::GrassDraw.enclosing(), Some(PassId::Main));
        assert_eq!(
            GpuSpan::GrassPrepass.enclosing(),
            Some(PassId::GBufferPrepass)
        );
        assert_eq!(GpuSpan::GrassShadowDraw.enclosing(), Some(PassId::Shadow));
        assert_eq!(GpuSpan::SsaoKernel.enclosing(), None);
        assert_eq!(GpuSpan::GrassDraw.timing().enclosing(), Some(PassId::Main));
    }

    #[test]
    fn timing_names_are_unique() {
        // A copy-pasted name would silently merge two slots in the profiler.
        let mut seen = hashbrown::HashSet::new();
        for &name in &TIMING_NAMES {
            assert!(!name.is_empty());
            assert!(seen.insert(name), "duplicate timing name {name:?}");
        }
    }
}
