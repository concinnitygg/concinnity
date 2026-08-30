// src/components/post_process_config.rs
//
// The PostProcessConfig asset: the authored schema (the struct, its enums and
// their `Default`), the `Component` impl, and the `PostProcessResolve` extension
// trait that resolves the authored tunables into the renderer's clamped `gfx`
// settings.

use crate::ecs::Component;
use crate::gfx::render_types::PostProcessTunables;
use crate::math::exp2;

/// Tunables for the post-process stack. One per world; the first declared
/// instance wins. With no `PostProcessConfig` present, the defaults below are
/// used.
///
/// The defaults describe the look on capable hardware: temporal
/// anti-aliasing, ambient occlusion, reflections and a screen-space indirect
/// bounce are all on. They are not what every GPU runs. The `Auto` graphics
/// quality preset resolves the detected GPU into a performance ceiling that
/// forces the expensive effects off tier by tier, so a world that authors
/// nothing still runs well on a laptop and still looks its best on a
/// workstation. A world that wants a cheaper look regardless of hardware turns
/// the effects off here; a ceiling only ever reduces, so it cannot undo that.
///
/// Colour-LUT grading is a separate [ColorLut](#colorlut) asset; `lut_strength`
/// here is the blend amount applied to whichever [ColorLut](#colorlut) the world
/// declares.
///
/// When `auto_exposure` is on, the scene's average brightness is measured each
/// frame and exposure adapts toward a balanced mid-tone. The authored
/// `exposure_ev` then acts as an additive bias (in stops) on top of the adapted
/// value.
///
/// ```rust
/// # use concinnity_core::components::PostProcessConfig;
/// PostProcessConfig {
///     bloom_intensity: 0.8,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PostProcessConfig {
    /// Additive bloom contribution. 0 skips bloom entirely.
    pub bloom_intensity: f32,
    /// Brightness threshold for bloom. Pixels brighter than this contribute
    /// fully; pixels within `bloom_knee` below it ramp in softly.
    pub bloom_threshold: f32,
    /// Width of the soft knee just below `bloom_threshold`.
    pub bloom_knee: f32,
    /// Exposure offset in photographic stops. Each +1 doubles scene
    /// brightness before bloom and tonemapping; 0 is neutral.
    pub exposure_ev: f32,
    /// Vignette strength in `[0, 1]`. 0 disables the corner darkening.
    pub vignette_strength: f32,
    /// Colour-LUT blend in `[0, 1]`. Mixes the graded colour over the ungraded
    /// one by this amount. Only matters when the world declares a
    /// [ColorLut](#colorlut); with none, grading is a no-op at any strength.
    pub lut_strength: f32,
    /// Anti-aliasing mode. `fxaa` applies a cheap composite-pass edge filter;
    /// `taa` (default) adds a temporal pass that jitters the projection and
    /// accumulates detail across frames for the cleanest edges, at the cost of a
    /// velocity pre-pass and a history buffer; `off` disables edge smoothing.
    /// Clamped to `fxaa` below the mid quality tier.
    pub aa_mode: AaMode,
    /// Screen-space ambient occlusion toggle. Darkens creases and contact areas
    /// where ambient light is occluded. On by default, forced off on the lowest
    /// quality tier.
    pub ssao: bool,
    /// How far the ambient-occlusion search reaches for occluders, in world
    /// units. Larger values pick up broader, softer occlusion.
    pub ssao_radius: f32,
    /// Ambient-occlusion strength, clamped to `[0, 4]`. 1.0 is the natural
    /// amount; higher values exaggerate the contact darkening.
    pub ssao_intensity: f32,
    /// Screen-space reflection toggle. Mixes reflected scene colour over glossy
    /// surfaces (water, polished floors). On by default, forced off below the
    /// high quality tier.
    pub ssr: bool,
    /// Reflection blend strength, clamped to `[0, 1]`. Scales the
    /// Fresnel-weighted reflection mixed over the base shading.
    pub ssr_intensity: f32,
    /// How far a reflection reaches, in world units. Longer reaches catch more
    /// distant reflections, more coarsely.
    pub ssr_max_distance: f32,
    /// Hardware ray-traced reflection toggle. When the GPU supports ray tracing,
    /// traces real reflection rays so off-screen geometry still appears, instead
    /// of the screen-space method. Reuses the `ssr_intensity` /
    /// `ssr_max_distance` tunables and takes precedence over `ssr`, falling back
    /// to it where ray tracing isn't available. On by default; only the top
    /// quality tier permits it, so everything below falls back to `ssr`.
    pub ray_traced_reflections: bool,
    /// Internal resolution of the roughness-aware reflection blur the SSR /
    /// ray-traced reflection composite runs. `half` (default) blurs at a
    /// quarter of the pixels for a large saving and bilinearly upsamples;
    /// `full` blurs at native resolution; `quarter` is the cheapest. Smooth
    /// mirror surfaces stay sharp at any setting (the composite keeps the sharp
    /// reflection for low roughness). Only matters when `ssr` or
    /// `ray_traced_reflections` is on.
    pub reflection_blur_resolution: ReflectionBlurResolution,
    /// Indirect-diffuse lighting source. `ibl` uses the environment map's
    /// ambient alone. `ssgi` (default) adds a screen-space global-illumination
    /// pass on top, so nearby lit surfaces bleed colour onto one another; the
    /// environment ambient still covers the off-screen / sky fallback. Clamped
    /// back to `ibl` below the high quality tier.
    pub indirect_lighting: IndirectLighting,
    /// Multiplier on the indirect (ambient / IBL) lighting term, clamped to
    /// `[0, 16]`. 1.0 (default) leaves the environment-derived ambient at its
    /// physical level. Raising it lifts fill light in areas the directional
    /// light cannot reach (shadowed facades, alleys) without brightening
    /// directly lit surfaces, which the sun already dominates. Scales the
    /// diffuse and specular IBL together, so reflections stay consistent with
    /// the brighter ambient. Useful for high-contrast exterior scenes where a
    /// strong sun would otherwise crush shadows to black.
    pub ambient_intensity: f32,
    /// Indirect-bounce strength, clamped to `[0, 4]`. Scales the gathered
    /// indirect light added on top of the existing shading; 0 makes it a no-op.
    /// Only matters when `indirect_lighting` is `ssgi`.
    pub ssgi_intensity: f32,
    /// How far the indirect-light gather reaches, in world units. A near-field
    /// effect, so it defaults well below `ssr_max_distance`. Only matters when
    /// `indirect_lighting` is `ssgi`.
    pub ssgi_max_distance: f32,
    /// Internal resolution of the SSGI gather. `half` (default) trades a little
    /// sharpness for a large performance saving; `full` is native; `quarter` is
    /// the cheapest. Only matters when `indirect_lighting` is `ssgi`.
    pub ssgi_resolution: SsgiResolution,
    /// Hemisphere rays cast per pixel by the SSGI gather, clamped to `[1, 32]`.
    /// More rays reduce noise at a higher cost. Only matters when
    /// `indirect_lighting` is `ssgi`.
    pub ssgi_rays: u32,
    /// Ray-march samples per SSGI ray, clamped to `[1, 64]`. More samples catch
    /// finer occlusion at a higher cost. Only matters when `indirect_lighting`
    /// is `ssgi`.
    pub ssgi_steps: u32,
    /// Auto-exposure toggle. Adapts exposure each frame toward a balanced
    /// mid-tone. The authored `exposure_ev` then acts as an additive bias in
    /// stops on top of the adapted value.
    pub auto_exposure: bool,
    /// Lower bound on the adapted exposure (EV). The `exposure_ev` bias is
    /// applied before this clamp.
    pub auto_exposure_min_ev: f32,
    /// Upper bound on the adapted exposure (EV).
    pub auto_exposure_max_ev: f32,
    /// How quickly exposure chases a new target (per second). Higher converges
    /// faster but can pump under flickering content; 1-3 is comfortable.
    pub auto_exposure_speed: f32,
    /// HDR display output toggle. On a capable display, emits extended-range
    /// HDR instead of the standard tonemapped output. Falls back to standard
    /// output when the display or platform doesn't support HDR.
    pub hdr_display: bool,
    /// PQ (HDR10) output mode. When true, and `hdr_display` is on, and the
    /// display has HDR headroom, output is PQ-encoded for HDR10 panels. No
    /// effect when `hdr_display` is off.
    pub hdr_pq: bool,
    /// Temporal upscaling toggle. Renders the 3D scene at a lower resolution
    /// (set by `upscale_quality`) and reconstructs a full-resolution image,
    /// trading some sharpness for performance. Replaces TAA while on (the `taa`
    /// flag is ignored).
    pub temporal_upscaling: bool,
    /// Render-scale preset for `temporal_upscaling`; each step progressively
    /// lowers the internal resolution. No effect when `temporal_upscaling` is
    /// off.
    pub upscale_quality: UpscaleQuality,
    /// Which upscaler backend `temporal_upscaling` uses. `auto` (default) picks
    /// the best available at runtime (DLSS on NVIDIA RTX, else XeSS, else FSR3);
    /// `fsr3` / `dlss` / `xess` request a specific one and fall back when it is
    /// unavailable on the current GPU or build. No effect when
    /// `temporal_upscaling` is off. DLSS and XeSS are DirectX-only.
    pub upscale_backend: UpscalerBackend,
    /// Two-pass occlusion culling toggle. Reduces objects popping in a frame
    /// late when they're revealed by camera or occluder motion, at the cost of
    /// extra culling work each frame. Needs the bindless GPU-cull path.
    pub occlusion_two_pass: bool,
}

/// Render-scale preset for `PostProcessConfig.temporal_upscaling`. The ratio
/// applies to both axes (input pixel count = output * ratio per axis), so
/// `Quality` renders at 4/9 of the output pixel count, `Performance` at 1/4,
/// and `UltraPerformance` at 1/9.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum UpscaleQuality {
    /// 4/9 of the output pixel count.
    #[default]
    Quality,
    /// Roughly a third of the output pixel count.
    Balanced,
    /// A quarter of the output pixel count.
    Performance,
    /// A ninth of the output pixel count.
    UltraPerformance,
}

impl UpscaleQuality {
    /// Per-axis input-to-output ratio. The render target's width/height are
    /// `(output_w * scale(), output_h * scale())`.
    pub fn scale(self) -> f32 {
        match self {
            UpscaleQuality::Quality => 2.0 / 3.0,
            UpscaleQuality::Balanced => 0.587,
            UpscaleQuality::Performance => 0.5,
            UpscaleQuality::UltraPerformance => 1.0 / 3.0,
        }
    }
}

/// Upscaler backend selector for `PostProcessConfig.temporal_upscaling`.
/// `Auto` resolves at runtime to the best available (DLSS, then XeSS, then
/// FSR3); the explicit variants request a specific backend and fall back when
/// it is unavailable. DLSS (NVIDIA NGX) and XeSS (Intel) are DirectX-only;
/// Metal uses MetalFX and Vulkan has no upscaler yet, so both treat any value
/// as their native path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum UpscalerBackend {
    /// Pick the best backend the device offers.
    #[default]
    Auto,
    /// AMD FidelityFX Super Resolution 3.
    Fsr3,
    /// NVIDIA DLSS, through NGX.
    Dlss,
    /// Intel XeSS.
    Xess,
}

/// Anti-aliasing mode for `PostProcessConfig.aa_mode`. `Off` runs no edge
/// smoothing; `Fxaa` (default) applies the composite's single-frame edge
/// filter, which is nearly free; `Taa` adds a temporal pass that jitters the
/// projection and reprojects detail across frames for the cleanest edges, at
/// the cost of a velocity pre-pass and a per-frame history buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum AaMode {
    /// No edge smoothing.
    Off,
    /// Single-frame edge filter in the composite.
    #[default]
    Fxaa,
    /// Temporal anti-aliasing: jittered projection plus a reprojected history.
    Taa,
}

impl AaMode {
    /// Whether the temporal anti-aliasing pass runs. Only the `Taa` mode does;
    /// it needs the velocity pre-pass and the history buffer the other modes
    /// skip.
    pub fn taa_enabled(self) -> bool {
        matches!(self, AaMode::Taa)
    }

    // Whether the composite's FXAA edge filter runs. Every mode except `Off`
    // does (so `Taa` keeps FXAA as a cheap spatial cleanup on top of the
    // temporal resolve).
    fn fxaa_enabled(self) -> bool {
        !matches!(self, AaMode::Off)
    }

    /// The composite's FXAA gate as the `0.0` / `1.0` flag `PostProcessParams`
    /// carries to the shader.
    pub fn fxaa_flag(self) -> f32 {
        if self.fxaa_enabled() { 1.0 } else { 0.0 }
    }
}

/// Indirect-diffuse lighting source for `PostProcessConfig.indirect_lighting`.
/// `Ibl` is the image-based-lighting-only ambient term the renderer has always
/// used; `Ssgi` layers a screen-space global-illumination bounce on top.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum IndirectLighting {
    /// Image-based lighting only.
    #[default]
    Ibl,
    /// Image-based lighting plus a screen-space bounce.
    Ssgi,
}

/// Internal render resolution of the SSGI gather pass (only meaningful when
/// `indirect_lighting` is `ssgi`). The gather is the expensive part (a
/// hemisphere ray-march per pixel), and its composite is a depth-aware
/// bilateral filter that upsamples a lower-resolution gather back to full
/// resolution at little visible cost. `half` (the default) gathers at a quarter
/// of the pixels for a large saving; `full` keeps the gather at native
/// resolution; `quarter` is the cheapest, for low-end GPUs or debugging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum SsgiResolution {
    /// Gather at native resolution.
    Full,
    /// Gather at half resolution per axis.
    #[default]
    Half,
    /// Gather at quarter resolution per axis.
    Quarter,
}

impl SsgiResolution {
    /// Per-axis render-resolution divisor the gather target is scaled by.
    pub fn scale_divisor(self) -> u32 {
        match self {
            SsgiResolution::Full => 1,
            SsgiResolution::Half => 2,
            SsgiResolution::Quarter => 4,
        }
    }
}

/// Internal render resolution of the roughness-aware reflection blur (only
/// meaningful when `ssr` or `ray_traced_reflections` is on). The blur is the
/// expensive multi-tap part of the reflection composite and is low-frequency
/// (a widening glossy cone), so running it at a fraction of the pixels and
/// bilinearly upsampling is visually free. `half` (the default) blurs at a
/// quarter of the pixels; `full` keeps it at native resolution; `quarter` is
/// the cheapest. Mirrors stay sharp regardless: the composite lerps in the
/// full-resolution reflection for low roughness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum ReflectionBlurResolution {
    /// Blur at native resolution.
    Full,
    /// Blur at half resolution per axis.
    #[default]
    Half,
    /// Blur at quarter resolution per axis.
    Quarter,
}

impl ReflectionBlurResolution {
    /// Per-axis render-resolution divisor the reflection blur target is scaled
    /// by.
    pub fn scale_divisor(self) -> u32 {
        match self {
            ReflectionBlurResolution::Full => 1,
            ReflectionBlurResolution::Half => 2,
            ReflectionBlurResolution::Quarter => 4,
        }
    }
}

/// Default SSGI hemisphere-ray and ray-march-step counts for the authored
/// `ssgi_rays` / `ssgi_steps` fields. Defined here (the schema default) and
/// re-exported by `concinnity-core`' `gfx::ssgi` for its runtime clamp path, so
/// the authored default and the runtime code stay a single source of truth.
pub const DEFAULT_SSGI_RAYS: u32 = 8;
/// Default ray-march steps per SSGI ray. See [`DEFAULT_SSGI_RAYS`].
pub const DEFAULT_SSGI_STEPS: u32 = 12;

impl Default for PostProcessConfig {
    fn default() -> Self {
        Self {
            bloom_intensity: 0.6,
            bloom_threshold: 1.0,
            bloom_knee: 0.5,
            exposure_ev: 0.0,
            vignette_strength: 0.0,
            lut_strength: 1.0,
            aa_mode: AaMode::Taa,
            ssao: true,
            ssao_radius: 0.5,
            ssao_intensity: 1.0,
            ssr: true,
            ssr_intensity: 0.7,
            ssr_max_distance: 40.0,
            ray_traced_reflections: true,
            reflection_blur_resolution: ReflectionBlurResolution::default(),
            indirect_lighting: IndirectLighting::Ssgi,
            ambient_intensity: 1.0,
            ssgi_intensity: 0.5,
            ssgi_max_distance: 8.0,
            ssgi_resolution: SsgiResolution::default(),
            ssgi_rays: DEFAULT_SSGI_RAYS,
            ssgi_steps: DEFAULT_SSGI_STEPS,
            auto_exposure: false,
            auto_exposure_min_ev: -8.0,
            auto_exposure_max_ev: 8.0,
            auto_exposure_speed: 1.5,
            hdr_display: false,
            hdr_pq: false,
            temporal_upscaling: false,
            upscale_quality: UpscaleQuality::default(),
            upscale_backend: UpscalerBackend::default(),
            occlusion_two_pass: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_author_the_capable_hardware_look() {
        // The renderer's optional work is on by default; the quality preset's
        // ceiling is what takes it back off tier by tier, so a world that
        // authors nothing gets the best look its GPU can carry.
        let c = PostProcessConfig::default();
        assert_eq!(c.aa_mode, AaMode::Taa);
        assert_eq!(c.bloom_intensity, 0.6);
        assert!(c.ssao);
        assert!(c.ssr);
        assert!(c.ray_traced_reflections);
        assert!(c.occlusion_two_pass);
        assert_eq!(c.indirect_lighting, IndirectLighting::Ssgi);
        assert_eq!(c.ssgi_rays, DEFAULT_SSGI_RAYS);
        assert_eq!(c.ssgi_steps, DEFAULT_SSGI_STEPS);
    }

    #[test]
    fn look_and_display_choices_stay_off_by_default() {
        // No quality tier turns these on, so they are authoring decisions, not
        // hardware ones: auto-exposure meters a scene the author framed, HDR
        // output and temporal upscaling trade fidelity the author chose.
        let c = PostProcessConfig::default();
        assert!(!c.auto_exposure);
        assert!(!c.temporal_upscaling);
        assert!(!c.hdr_display);
        assert!(!c.hdr_pq);
        assert_eq!(c.vignette_strength, 0.0);
    }

    #[test]
    fn enum_defaults_are_the_cheap_variants() {
        let c = PostProcessConfig::default();
        assert_eq!(c.upscale_quality, UpscaleQuality::Quality);
        assert_eq!(c.upscale_backend, UpscalerBackend::Auto);
        assert_eq!(c.ssgi_resolution, SsgiResolution::Half);
        assert_eq!(c.reflection_blur_resolution, ReflectionBlurResolution::Half);
        assert_eq!(AaMode::default(), AaMode::Fxaa);
        assert_eq!(IndirectLighting::default(), IndirectLighting::Ibl);
        // The two enum `Default`s the config deliberately does not use: the
        // cheap variant is the right fallback for a bare `AaMode` /
        // `IndirectLighting`, while the config defaults to the richer one.
        assert_ne!(c.aa_mode, AaMode::default());
        assert_ne!(c.indirect_lighting, IndirectLighting::default());
    }

    #[test]
    fn upscale_quality_scales_the_render_resolution_down() {
        // Ordered coarsest-last: each tier renders strictly fewer pixels.
        assert_eq!(UpscaleQuality::Quality.scale(), 2.0 / 3.0);
        assert_eq!(UpscaleQuality::Balanced.scale(), 0.587);
        assert_eq!(UpscaleQuality::Performance.scale(), 0.5);
        assert_eq!(UpscaleQuality::UltraPerformance.scale(), 1.0 / 3.0);
        let tiers = [
            UpscaleQuality::Quality,
            UpscaleQuality::Balanced,
            UpscaleQuality::Performance,
            UpscaleQuality::UltraPerformance,
        ];
        assert!(tiers.windows(2).all(|w| w[0].scale() > w[1].scale()));
    }

    #[test]
    fn fxaa_runs_for_every_mode_but_off_and_taa_only_for_taa() {
        // Taa keeps the FXAA pass: the temporal resolve does not replace it.

        assert!(!AaMode::Off.taa_enabled());
        assert!(!AaMode::Fxaa.taa_enabled());
        assert!(AaMode::Taa.taa_enabled());

        // The shader-side flag is the enabled bit as a float.
        assert_eq!(AaMode::Off.fxaa_flag(), 0.0);
        assert_eq!(AaMode::Fxaa.fxaa_flag(), 1.0);
        assert_eq!(AaMode::Taa.fxaa_flag(), 1.0);
    }

    #[test]
    fn half_and_quarter_resolutions_divide_the_target() {
        assert_eq!(SsgiResolution::Full.scale_divisor(), 1);
        assert_eq!(SsgiResolution::Half.scale_divisor(), 2);
        assert_eq!(SsgiResolution::Quarter.scale_divisor(), 4);
        assert_eq!(ReflectionBlurResolution::Full.scale_divisor(), 1);
        assert_eq!(ReflectionBlurResolution::Half.scale_divisor(), 2);
        assert_eq!(ReflectionBlurResolution::Quarter.scale_divisor(), 4);
    }

    #[test]
    fn enum_names_parse_in_snake_case() {
        let aa = |s: &str| serde_json::from_str::<AaMode>(s).unwrap();
        assert_eq!(aa(r#""off""#), AaMode::Off);
        assert_eq!(aa(r#""fxaa""#), AaMode::Fxaa);
        assert_eq!(aa(r#""taa""#), AaMode::Taa);

        let q = |s: &str| serde_json::from_str::<UpscaleQuality>(s).unwrap();
        assert_eq!(q(r#""balanced""#), UpscaleQuality::Balanced);
        assert_eq!(
            q(r#""ultra_performance""#),
            UpscaleQuality::UltraPerformance
        );
        assert_eq!(
            serde_json::to_string(&UpscaleQuality::UltraPerformance).unwrap(),
            r#""ultra_performance""#
        );

        let b = |s: &str| serde_json::from_str::<UpscalerBackend>(s).unwrap();
        assert_eq!(b(r#""auto""#), UpscalerBackend::Auto);
        assert_eq!(b(r#""fsr3""#), UpscalerBackend::Fsr3);
        assert_eq!(b(r#""dlss""#), UpscalerBackend::Dlss);
        assert_eq!(b(r#""xess""#), UpscalerBackend::Xess);

        assert_eq!(
            serde_json::from_str::<IndirectLighting>(r#""ssgi""#).unwrap(),
            IndirectLighting::Ssgi
        );
        assert_eq!(
            serde_json::from_str::<SsgiResolution>(r#""quarter""#).unwrap(),
            SsgiResolution::Quarter
        );
        assert_eq!(
            serde_json::from_str::<ReflectionBlurResolution>(r#""full""#).unwrap(),
            ReflectionBlurResolution::Full
        );
    }

    #[test]
    fn an_authored_stack_round_trips_through_postcard() {
        let c: PostProcessConfig = serde_json::from_str(
            r#"{"aa_mode":"taa","ssao":true,"ssr":true,"indirect_lighting":"ssgi",
                "ssgi_resolution":"quarter","temporal_upscaling":true,
                "upscale_quality":"performance","upscale_backend":"dlss",
                "auto_exposure":true,"hdr_display":true,"hdr_pq":true}"#,
        )
        .unwrap();
        assert!(c.aa_mode.taa_enabled());
        assert_eq!(c.ssgi_resolution.scale_divisor(), 4);
        // Fields the args did not mention keep the schema defaults.
        assert_eq!(c.bloom_intensity, 0.6);

        let bytes = postcard::to_allocvec(&c).unwrap();
        let back: PostProcessConfig = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.aa_mode, AaMode::Taa);
        assert_eq!(back.upscale_backend, UpscalerBackend::Dlss);
        assert_eq!(back.upscale_quality, UpscaleQuality::Performance);
        assert_eq!(back.indirect_lighting, IndirectLighting::Ssgi);
        assert!(back.hdr_pq);
    }
}

// `exposure_ev` is clamped to this range before resolving to a multiplier so a
// stray value cannot push the scene to `inf` / `0`.
const EXPOSURE_EV_LIMIT: f32 = 16.0;

/// Resolves a `PostProcessConfig`'s authored tunables into the clamped,
/// GPU-facing settings the renderer consumes. Kept in `gfx` (not the schema)
/// because every return type is a `crate::gfx` settings struct.
pub trait PostProcessResolve {
    /// Resolve the authored fields into the GPU-facing `PostProcessTunables`:
    /// clamps each tunable and converts `exposure_ev` (stops) into the linear
    /// multiplier the shaders expect. The composite's display-output flags are
    /// not authored, so they are absent here: the backend adds them to the full
    /// `PostProcessParams` once it has negotiated EDR support with the display.
    fn resolve(&self) -> PostProcessTunables;

    /// Clamp the authored `ambient_intensity` to a safe `[0, 16]` multiplier the
    /// backend folds into `LightUniforms` to scale the indirect (ambient / IBL)
    /// term.
    fn ambient_intensity(&self) -> f32;

    /// Per-axis divisor for the roughness-aware reflection blur target, resolved
    /// from `reflection_blur_resolution`. Always at least 1.
    fn reflection_blur_divisor(&self) -> u32;

    /// Resolve the SSAO tunables into clamped `SsaoSettings`, or `None` when the
    /// `ssao` toggle is off so the backend can skip the SSAO passes entirely.
    fn ssao_settings(&self) -> Option<crate::gfx::ssao::SsaoSettings>;

    /// Resolve the SSR tunables into clamped `SsrSettings`, or `None` when the
    /// `ssr` toggle is off.
    fn ssr_settings(&self) -> Option<crate::gfx::ssr::SsrSettings>;

    /// Resolve the ray-traced-reflection tunables into clamped
    /// `RtReflectionSettings`, or `None` when `ray_traced_reflections` is off.
    /// Reuses the SSR intensity / distance fields; the backend additionally gates
    /// on GPU ray-tracing support.
    fn rt_reflection_settings(&self) -> Option<crate::gfx::rt_reflections::RtReflectionSettings>;

    /// Resolve the SSGI tunables into clamped `SsgiSettings`, or `None` when
    /// `indirect_lighting` is not `Ssgi` so the backend can skip the SSGI passes.
    fn ssgi_settings(&self) -> Option<crate::gfx::ssgi::SsgiSettings>;

    /// Resolve the auto-exposure tunables into clamped `AutoExposureSettings`, or
    /// `None` when the toggle is off so the backend can skip the histogram passes.
    fn auto_exposure_settings(&self) -> Option<crate::gfx::auto_exposure::AutoExposureSettings>;
}

impl PostProcessResolve for PostProcessConfig {
    fn resolve(&self) -> PostProcessTunables {
        let ev = self
            .exposure_ev
            .clamp(-EXPOSURE_EV_LIMIT, EXPOSURE_EV_LIMIT);
        PostProcessTunables {
            bloom_intensity: self.bloom_intensity.max(0.0),
            bloom_threshold: self.bloom_threshold.max(0.0),
            bloom_knee: self.bloom_knee.max(0.0),
            exposure: exp2(ev),
            vignette: self.vignette_strength.clamp(0.0, 1.0),
            lut_strength: self.lut_strength.clamp(0.0, 1.0),
            fxaa: self.aa_mode.fxaa_flag(),
        }
    }

    fn ambient_intensity(&self) -> f32 {
        self.ambient_intensity.clamp(0.0, 16.0)
    }

    fn reflection_blur_divisor(&self) -> u32 {
        self.reflection_blur_resolution.scale_divisor()
    }

    fn ssao_settings(&self) -> Option<crate::gfx::ssao::SsaoSettings> {
        self.ssao
            .then(|| crate::gfx::ssao::SsaoSettings::resolve(self.ssao_radius, self.ssao_intensity))
    }

    fn ssr_settings(&self) -> Option<crate::gfx::ssr::SsrSettings> {
        self.ssr.then(|| {
            crate::gfx::ssr::SsrSettings::resolve(self.ssr_intensity, self.ssr_max_distance)
        })
    }

    fn rt_reflection_settings(&self) -> Option<crate::gfx::rt_reflections::RtReflectionSettings> {
        self.ray_traced_reflections.then(|| {
            crate::gfx::rt_reflections::RtReflectionSettings::resolve(
                self.ssr_intensity,
                self.ssr_max_distance,
            )
        })
    }

    fn ssgi_settings(&self) -> Option<crate::gfx::ssgi::SsgiSettings> {
        (self.indirect_lighting == IndirectLighting::Ssgi).then(|| {
            crate::gfx::ssgi::SsgiSettings::resolve(
                self.ssgi_intensity,
                self.ssgi_max_distance,
                self.ssgi_rays,
                self.ssgi_steps,
                self.ssgi_resolution.scale_divisor(),
            )
        })
    }

    fn auto_exposure_settings(&self) -> Option<crate::gfx::auto_exposure::AutoExposureSettings> {
        self.auto_exposure.then(|| {
            // `hdr_display = true` shifts AE's pivot from scene-white
            // (legacy SDR + ACES) to perceptual middle-grey, so the average
            // pixel reads as a comfortable mid-tone on a panel that does no
            // implicit tonemap. Falls back gracefully: even if the platform
            // rejects the HDR request at swapchain time, SDR + ACES still
            // produces a sensible (slightly darker) result.
            crate::gfx::auto_exposure::AutoExposureSettings::resolve(
                self.auto_exposure_min_ev,
                self.auto_exposure_max_ev,
                self.auto_exposure_speed,
                self.hdr_display,
            )
        })
    }
}

impl Component for PostProcessConfig {
    const NAME: &'static str = "PostProcessConfig";

    fn from_baked(bytes: &[u8]) -> Result<Self, crate::result::CnResult> {
        Ok(crate::blob::decode_exact(bytes)?)
    }
}

#[cfg(test)]
mod runtime_tests {
    use super::*;
    use crate::components::{
        AaMode, ReflectionBlurResolution, SsgiResolution, UpscaleQuality, UpscalerBackend,
    };
    use alloc::format;

    #[test]
    fn default_resolves_to_neutral_params() {
        let p = PostProcessConfig::default().resolve();
        assert_eq!(p.bloom_intensity, 0.6);
        assert_eq!(p.bloom_threshold, 1.0);
        assert_eq!(p.bloom_knee, 0.5);
        // No exposure offset and no vignette out of the box.
        assert_eq!(p.exposure, 1.0);
        assert_eq!(p.vignette, 0.0);
        // Full LUT blend by default: a no-op until a ColorLut is declared.
        assert_eq!(p.lut_strength, 1.0);
        // The renderer's no-asset fallback has to resolve to the same thing.
        assert_eq!(p, PostProcessTunables::DEFAULT);
    }

    #[test]
    fn exposure_ev_resolves_to_power_of_two_multiplier() {
        let cfg = PostProcessConfig {
            exposure_ev: 2.0,
            ..Default::default()
        };
        assert_eq!(cfg.resolve().exposure, 4.0);

        let cfg = PostProcessConfig {
            exposure_ev: -1.0,
            ..Default::default()
        };
        assert_eq!(cfg.resolve().exposure, 0.5);
    }

    #[test]
    fn exposure_ev_is_clamped_to_a_finite_multiplier() {
        let cfg = PostProcessConfig {
            exposure_ev: 1.0e9,
            ..Default::default()
        };
        let exposure = cfg.resolve().exposure;
        assert!(exposure.is_finite());
        assert_eq!(exposure, EXPOSURE_EV_LIMIT.exp2());
    }

    #[test]
    fn negative_and_overrange_inputs_are_clamped() {
        let cfg = PostProcessConfig {
            bloom_intensity: -3.0,
            bloom_threshold: -1.0,
            bloom_knee: -0.2,
            vignette_strength: 5.0,
            lut_strength: -2.0,
            ..Default::default()
        };
        let p = cfg.resolve();
        assert_eq!(p.bloom_intensity, 0.0);
        assert_eq!(p.bloom_threshold, 0.0);
        assert_eq!(p.bloom_knee, 0.0);
        assert_eq!(p.vignette, 1.0);
        assert_eq!(p.lut_strength, 0.0);
    }

    #[test]
    fn lut_strength_is_clamped_to_unit_range() {
        let cfg = PostProcessConfig {
            lut_strength: 3.0,
            ..Default::default()
        };
        assert_eq!(cfg.resolve().lut_strength, 1.0);
    }

    #[test]
    fn aa_mode_defaults_to_taa_and_round_trips_through_args() {
        assert_eq!(PostProcessConfig::default().aa_mode, AaMode::Taa);
        let cfg = PostProcessConfig {
            aa_mode: AaMode::Fxaa,
            ..Default::default()
        };
        assert_eq!(cfg.clone().aa_mode, AaMode::Fxaa);
    }

    #[test]
    fn aa_mode_gates_taa_and_fxaa() {
        assert!(!AaMode::Off.taa_enabled());
        assert!(!AaMode::Fxaa.taa_enabled());
        assert!(AaMode::Taa.taa_enabled());
        // resolve() carries the FXAA gate into the composite uniform.
        let off = PostProcessConfig {
            aa_mode: AaMode::Off,
            ..Default::default()
        };
        assert_eq!(off.resolve().fxaa, 0.0);
        assert_eq!(PostProcessConfig::default().resolve().fxaa, 1.0);
    }

    #[test]
    fn ssao_defaults_on_with_neutral_tunables() {
        let cfg = PostProcessConfig::default();
        assert!(cfg.ssao);
        assert_eq!(cfg.ssao_radius, 0.5);
        assert_eq!(cfg.ssao_intensity, 1.0);
        assert!(cfg.ssao_settings().is_some());
        // No SsaoSettings once the toggle is off.
        let off = PostProcessConfig {
            ssao: false,
            ..Default::default()
        };
        assert!(off.ssao_settings().is_none());
    }

    #[test]
    fn ssao_settings_resolve_and_clamp_when_enabled() {
        let cfg = PostProcessConfig {
            ssao: true,
            ssao_radius: -1.0,
            ssao_intensity: 99.0,
            ..Default::default()
        };
        let s = cfg.ssao_settings().expect("ssao on");
        assert!(s.radius > 0.0);
        assert_eq!(s.intensity, 4.0);
    }

    #[test]
    fn ssao_deserialises_from_jsonl_args() {
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"ssao":true,"ssao_radius":0.6}"#).expect("parse");
        assert!(cfg.ssao);
        assert_eq!(cfg.ssao_radius, 0.6);
        // Omitted intensity falls back to the default.
        assert_eq!(cfg.ssao_intensity, 1.0);
    }

    #[test]
    fn ssr_defaults_on_with_neutral_tunables() {
        let cfg = PostProcessConfig::default();
        assert!(cfg.ssr);
        assert_eq!(cfg.ssr_intensity, 0.7);
        assert_eq!(cfg.ssr_max_distance, 40.0);
        assert!(cfg.ssr_settings().is_some());
        // No SsrSettings once the toggle is off.
        let off = PostProcessConfig {
            ssr: false,
            ..Default::default()
        };
        assert!(off.ssr_settings().is_none());
    }

    #[test]
    fn ssr_settings_resolve_and_clamp_when_enabled() {
        let cfg = PostProcessConfig {
            ssr: true,
            ssr_intensity: 9.0,
            ssr_max_distance: 1.0e6,
            ..Default::default()
        };
        let s = cfg.ssr_settings().expect("ssr on");
        assert_eq!(s.intensity, 1.0);
        assert!(s.max_distance > 0.0 && s.max_distance.is_finite());
    }

    #[test]
    fn ssr_deserialises_from_jsonl_args() {
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"ssr":true,"ssr_intensity":0.5}"#).expect("parse");
        assert!(cfg.ssr);
        assert_eq!(cfg.ssr_intensity, 0.5);
        // Omitted distance falls back to the default.
        assert_eq!(cfg.ssr_max_distance, 40.0);
    }

    #[test]
    fn rt_reflections_default_on_and_resolve_to_settings() {
        let cfg = PostProcessConfig::default();
        assert!(cfg.ray_traced_reflections);
        assert!(cfg.rt_reflection_settings().is_some());
        // No RtReflectionSettings once the toggle is off.
        let off = PostProcessConfig {
            ray_traced_reflections: false,
            ..Default::default()
        };
        assert!(off.rt_reflection_settings().is_none());
    }

    #[test]
    fn rt_reflection_settings_reuse_ssr_tunables_when_enabled() {
        let cfg = PostProcessConfig {
            ray_traced_reflections: true,
            ssr_intensity: 9.0,
            ssr_max_distance: 1.0e6,
            ..Default::default()
        };
        let s = cfg.rt_reflection_settings().expect("rt on");
        // Reuses the SSR intensity / distance fields, clamped by the RT resolve.
        assert_eq!(s.intensity, 1.0);
        assert!(s.max_distance > 0.0 && s.max_distance.is_finite());
    }

    #[test]
    fn rt_reflections_deserialise_from_jsonl_args() {
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"ray_traced_reflections":true,"ssr_intensity":0.5}"#)
                .expect("parse");
        assert!(cfg.ray_traced_reflections);
        assert!(cfg.rt_reflection_settings().is_some());
        // An explicit false is what turns ray tracing off; omitting the field
        // keeps the default on.
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"ray_traced_reflections":false}"#).expect("parse");
        assert!(!cfg.ray_traced_reflections);
        assert!(cfg.rt_reflection_settings().is_none());
    }

    #[test]
    fn ambient_intensity_defaults_neutral_and_clamps() {
        // Default is a no-op multiplier.
        assert_eq!(PostProcessConfig::default().ambient_intensity(), 1.0);
        // Authored values clamp into [0, 16].
        let hot = PostProcessConfig {
            ambient_intensity: 100.0,
            ..Default::default()
        };
        assert_eq!(hot.ambient_intensity(), 16.0);
        let neg = PostProcessConfig {
            ambient_intensity: -2.0,
            ..Default::default()
        };
        assert_eq!(neg.ambient_intensity(), 0.0);
        // Round-trips through JSONL like any other tunable.
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"ambient_intensity":3.5}"#).expect("parse");
        assert_eq!(cfg.ambient_intensity(), 3.5);
    }

    #[test]
    fn ssgi_defaults_on_with_neutral_tunables() {
        let cfg = PostProcessConfig::default();
        assert_eq!(cfg.indirect_lighting, IndirectLighting::Ssgi);
        assert_eq!(cfg.ssgi_intensity, 0.5);
        assert_eq!(cfg.ssgi_max_distance, 8.0);
        // The gather defaults to half resolution with the historical 8x12
        // ray/step counts.
        assert_eq!(cfg.ssgi_resolution, SsgiResolution::Half);
        assert_eq!(cfg.ssgi_rays, 8);
        assert_eq!(cfg.ssgi_steps, 12);
        assert!(cfg.ssgi_settings().is_some());
        // No SsgiSettings once indirect lighting is IBL-only.
        let ibl = PostProcessConfig {
            indirect_lighting: IndirectLighting::Ibl,
            ..Default::default()
        };
        assert!(ibl.ssgi_settings().is_none());
    }

    #[test]
    fn ssgi_resolution_maps_to_a_per_axis_divisor() {
        assert_eq!(SsgiResolution::Full.scale_divisor(), 1);
        assert_eq!(SsgiResolution::Half.scale_divisor(), 2);
        assert_eq!(SsgiResolution::Quarter.scale_divisor(), 4);
        assert_eq!(SsgiResolution::default(), SsgiResolution::Half);
    }

    #[test]
    fn ssgi_resolution_and_counts_flow_into_settings() {
        let cfg = PostProcessConfig {
            indirect_lighting: IndirectLighting::Ssgi,
            ssgi_resolution: SsgiResolution::Quarter,
            ssgi_rays: 4,
            ssgi_steps: 20,
            ..Default::default()
        };
        let s = cfg.ssgi_settings().expect("ssgi on");
        assert_eq!(s.rays, 4);
        assert_eq!(s.steps, 20);
        assert_eq!(s.gi_scale, 4);
    }

    #[test]
    fn ssgi_resolution_and_counts_deserialise_from_jsonl_args() {
        let cfg: PostProcessConfig = serde_json::from_str(
            r#"{"indirect_lighting":"ssgi","ssgi_resolution":"full","ssgi_rays":16,"ssgi_steps":8}"#,
        )
        .expect("parse");
        assert_eq!(cfg.ssgi_resolution, SsgiResolution::Full);
        assert_eq!(cfg.ssgi_rays, 16);
        assert_eq!(cfg.ssgi_steps, 8);
        // Omitting them falls back to the half-resolution 8x12 defaults.
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"indirect_lighting":"ssgi"}"#).expect("parse");
        assert_eq!(cfg.ssgi_resolution, SsgiResolution::Half);
        assert_eq!(cfg.ssgi_rays, 8);
        assert_eq!(cfg.ssgi_steps, 12);
    }

    #[test]
    fn reflection_blur_resolution_defaults_to_half() {
        let cfg = PostProcessConfig::default();
        assert_eq!(
            cfg.reflection_blur_resolution,
            ReflectionBlurResolution::Half
        );
        assert_eq!(cfg.reflection_blur_divisor(), 2);
    }

    #[test]
    fn reflection_blur_resolution_maps_to_a_per_axis_divisor() {
        assert_eq!(ReflectionBlurResolution::Full.scale_divisor(), 1);
        assert_eq!(ReflectionBlurResolution::Half.scale_divisor(), 2);
        assert_eq!(ReflectionBlurResolution::Quarter.scale_divisor(), 4);
        assert_eq!(
            ReflectionBlurResolution::default(),
            ReflectionBlurResolution::Half
        );
    }

    #[test]
    fn reflection_blur_resolution_deserialises_from_jsonl_args() {
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"ssr":true,"reflection_blur_resolution":"quarter"}"#)
                .expect("parse");
        assert_eq!(
            cfg.reflection_blur_resolution,
            ReflectionBlurResolution::Quarter
        );
        assert_eq!(cfg.reflection_blur_divisor(), 4);
        // Omitting the field falls back to the half-resolution default.
        let cfg: PostProcessConfig = serde_json::from_str(r#"{"ssr":true}"#).expect("parse");
        assert_eq!(
            cfg.reflection_blur_resolution,
            ReflectionBlurResolution::Half
        );
        assert_eq!(cfg.reflection_blur_divisor(), 2);
    }

    #[test]
    fn ssgi_settings_resolve_and_clamp_when_enabled() {
        let cfg = PostProcessConfig {
            indirect_lighting: IndirectLighting::Ssgi,
            ssgi_intensity: 99.0,
            ssgi_max_distance: 1.0e6,
            ..Default::default()
        };
        let s = cfg.ssgi_settings().expect("ssgi on");
        assert_eq!(s.intensity, 4.0);
        assert!(s.max_distance > 0.0 && s.max_distance.is_finite());
    }

    #[test]
    fn ssgi_deserialises_from_jsonl_args() {
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"indirect_lighting":"ssgi","ssgi_intensity":0.8}"#)
                .expect("parse");
        assert_eq!(cfg.indirect_lighting, IndirectLighting::Ssgi);
        assert_eq!(cfg.ssgi_intensity, 0.8);
        // Omitted distance falls back to the default.
        assert_eq!(cfg.ssgi_max_distance, 8.0);
        // An explicit "ibl" is what drops the screen-space bounce; omitting the
        // field keeps the default on.
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"indirect_lighting":"ibl"}"#).expect("parse");
        assert_eq!(cfg.indirect_lighting, IndirectLighting::Ibl);
        assert!(cfg.ssgi_settings().is_none());
    }

    #[test]
    fn auto_exposure_defaults_off_with_neutral_tunables() {
        let cfg = PostProcessConfig::default();
        assert!(!cfg.auto_exposure);
        assert_eq!(cfg.auto_exposure_min_ev, -8.0);
        assert_eq!(cfg.auto_exposure_max_ev, 8.0);
        assert_eq!(cfg.auto_exposure_speed, 1.5);
        assert!(cfg.auto_exposure_settings().is_none());
    }

    #[test]
    fn auto_exposure_settings_resolve_when_enabled() {
        let cfg = PostProcessConfig {
            auto_exposure: true,
            auto_exposure_min_ev: -4.0,
            auto_exposure_max_ev: 6.0,
            auto_exposure_speed: 2.0,
            ..Default::default()
        };
        let s = cfg.auto_exposure_settings().expect("auto-exposure on");
        assert_eq!(s.min_ev, -4.0);
        assert_eq!(s.max_ev, 6.0);
        assert_eq!(s.speed, 2.0);
    }

    #[test]
    fn auto_exposure_deserialises_from_jsonl_args() {
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"auto_exposure":true,"auto_exposure_speed":3.0}"#)
                .expect("parse");
        assert!(cfg.auto_exposure);
        assert_eq!(cfg.auto_exposure_speed, 3.0);
        // Omitted bounds fall back to the defaults.
        assert_eq!(cfg.auto_exposure_min_ev, -8.0);
        assert_eq!(cfg.auto_exposure_max_ev, 8.0);
    }

    #[test]
    fn aa_mode_deserialises_from_jsonl_args() {
        let cfg: PostProcessConfig = serde_json::from_str(r#"{"aa_mode":"taa"}"#).expect("parse");
        assert_eq!(cfg.aa_mode, AaMode::Taa);
        // Omitting the field falls back to the TAA default.
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"bloom_intensity":0.5}"#).expect("parse");
        assert_eq!(cfg.aa_mode, AaMode::Taa);
        // "off" disables edge smoothing entirely.
        let cfg: PostProcessConfig = serde_json::from_str(r#"{"aa_mode":"off"}"#).expect("parse");
        assert_eq!(cfg.aa_mode, AaMode::Off);
    }

    #[test]
    fn hdr_display_defaults_off() {
        assert!(!PostProcessConfig::default().hdr_display);
    }

    #[test]
    fn hdr_display_round_trips_through_args_and_jsonl() {
        let cfg = PostProcessConfig {
            hdr_display: true,
            ..Default::default()
        };
        assert!(cfg.clone().hdr_display);

        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"hdr_display":true}"#).expect("parse");
        assert!(cfg.hdr_display);
    }

    #[test]
    fn temporal_upscaling_defaults_off_with_quality_preset() {
        let cfg = PostProcessConfig::default();
        assert!(!cfg.temporal_upscaling);
        assert_eq!(cfg.upscale_quality, UpscaleQuality::Quality);
    }

    #[test]
    fn upscale_quality_scales_are_monotonic() {
        // Each step down in quality must reduce the per-axis ratio so render
        // cost drops monotonically as users dial quality lower.
        let q = UpscaleQuality::Quality.scale();
        let b = UpscaleQuality::Balanced.scale();
        let p = UpscaleQuality::Performance.scale();
        let u = UpscaleQuality::UltraPerformance.scale();
        assert!(q > b && b > p && p > u);
        assert!(u > 0.0);
    }

    #[test]
    fn occlusion_two_pass_defaults_on_and_round_trips() {
        assert!(PostProcessConfig::default().occlusion_two_pass);
        let cfg = PostProcessConfig {
            occlusion_two_pass: false,
            ..Default::default()
        };
        assert!(!cfg.clone().occlusion_two_pass);
        // Deserialises from jsonl args; omitting it leaves the feature on.
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"occlusion_two_pass":false}"#).expect("parse");
        assert!(!cfg.occlusion_two_pass);
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"bloom_intensity":0.5}"#).expect("parse");
        assert!(cfg.occlusion_two_pass);
    }

    #[test]
    fn upscale_backend_defaults_to_auto() {
        assert_eq!(
            PostProcessConfig::default().upscale_backend,
            UpscalerBackend::Auto
        );
        assert_eq!(UpscalerBackend::default(), UpscalerBackend::Auto);
    }

    #[test]
    fn upscale_backend_round_trips_via_snake_case_json() {
        for (s, want) in [
            ("auto", UpscalerBackend::Auto),
            ("fsr3", UpscalerBackend::Fsr3),
            ("dlss", UpscalerBackend::Dlss),
            ("xess", UpscalerBackend::Xess),
        ] {
            let json = format!(r#"{{"temporal_upscaling":true,"upscale_backend":"{s}"}}"#);
            let cfg: PostProcessConfig = serde_json::from_str(&json).expect("parse");
            assert_eq!(cfg.upscale_backend, want, "for {s}");
        }
        // Omitting the field falls back to Auto.
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"temporal_upscaling":true}"#).expect("parse");
        assert_eq!(cfg.upscale_backend, UpscalerBackend::Auto);
    }

    #[test]
    fn upscale_backend_round_trips_through_args() {
        let cfg = PostProcessConfig {
            upscale_backend: UpscalerBackend::Xess,
            ..Default::default()
        };
        assert_eq!(cfg.clone().upscale_backend, UpscalerBackend::Xess);
    }

    #[test]
    fn upscale_quality_round_trips_via_snake_case_json() {
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"temporal_upscaling":true,"upscale_quality":"performance"}"#)
                .expect("parse");
        assert!(cfg.temporal_upscaling);
        assert_eq!(cfg.upscale_quality, UpscaleQuality::Performance);
        // Omitting the preset falls back to the default.
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"temporal_upscaling":true}"#).expect("parse");
        assert_eq!(cfg.upscale_quality, UpscaleQuality::Quality);
    }
}
