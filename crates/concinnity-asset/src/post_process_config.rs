// Post-process stack schema.

/// Tunables for the post-process stack. One per world; the first declared
/// instance wins. With no `PostProcessConfig` present, the defaults below are
/// used (bloom on at a moderate intensity).
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
/// # use concinnity_asset::PostProcessConfig;
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
    /// Anti-aliasing mode. `fxaa` (default) applies a cheap composite-pass edge
    /// filter; `taa` adds a temporal pass that jitters the projection and
    /// accumulates detail across frames for the cleanest edges, at the cost of a
    /// velocity pre-pass and a history buffer; `off` disables edge smoothing.
    pub aa_mode: AaMode,
    /// Screen-space ambient occlusion toggle. Darkens creases and contact areas
    /// where ambient light is occluded.
    pub ssao: bool,
    /// How far the ambient-occlusion search reaches for occluders, in world
    /// units. Larger values pick up broader, softer occlusion.
    pub ssao_radius: f32,
    /// Ambient-occlusion strength, clamped to `[0, 4]`. 1.0 is the natural
    /// amount; higher values exaggerate the contact darkening.
    pub ssao_intensity: f32,
    /// Screen-space reflection toggle. Mixes reflected scene colour over glossy
    /// surfaces (water, polished floors).
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
    /// to it where ray tracing isn't available.
    pub ray_traced_reflections: bool,
    /// Internal resolution of the roughness-aware reflection blur the SSR /
    /// ray-traced reflection composite runs. `half` (default) blurs at a
    /// quarter of the pixels for a large saving and bilinearly upsamples;
    /// `full` blurs at native resolution; `quarter` is the cheapest. Smooth
    /// mirror surfaces stay sharp at any setting (the composite keeps the sharp
    /// reflection for low roughness). Only matters when `ssr` or
    /// `ray_traced_reflections` is on.
    pub reflection_blur_resolution: ReflectionBlurResolution,
    /// Indirect-diffuse lighting source. `ibl` (default) uses the environment
    /// map's ambient alone. `ssgi` adds a screen-space global-illumination pass
    /// on top, so nearby lit surfaces bleed colour onto one another; the
    /// environment ambient still covers the off-screen / sky fallback.
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
    /// extra culling work each frame.
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

    /// Whether the composite's FXAA edge filter runs. Every mode except `Off`
    /// does (so `Taa` keeps FXAA as a cheap spatial cleanup on top of the
    /// temporal resolve).
    pub fn fxaa_enabled(self) -> bool {
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
            aa_mode: AaMode::Fxaa,
            ssao: false,
            ssao_radius: 0.5,
            ssao_intensity: 1.0,
            ssr: false,
            ssr_intensity: 0.7,
            ssr_max_distance: 40.0,
            ray_traced_reflections: false,
            reflection_blur_resolution: ReflectionBlurResolution::default(),
            indirect_lighting: IndirectLighting::Ibl,
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
            occlusion_two_pass: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_leave_the_expensive_effects_off() {
        // Bloom and FXAA are cheap enough to ship on; everything that costs a
        // full-screen pass is opt-in so a blank world runs on any hardware.
        let c = PostProcessConfig::default();
        assert_eq!(c.aa_mode, AaMode::Fxaa);
        assert_eq!(c.bloom_intensity, 0.6);
        assert!(!c.ssao);
        assert!(!c.ssr);
        assert!(!c.ray_traced_reflections);
        assert!(!c.auto_exposure);
        assert!(!c.temporal_upscaling);
        assert!(!c.hdr_display);
        assert!(!c.occlusion_two_pass);
        assert_eq!(c.indirect_lighting, IndirectLighting::Ibl);
        assert_eq!(c.ssgi_rays, DEFAULT_SSGI_RAYS);
        assert_eq!(c.ssgi_steps, DEFAULT_SSGI_STEPS);
    }

    #[test]
    fn every_enum_default_matches_the_config_default() {
        let c = PostProcessConfig::default();
        assert_eq!(c.upscale_quality, UpscaleQuality::Quality);
        assert_eq!(c.upscale_backend, UpscalerBackend::Auto);
        assert_eq!(c.ssgi_resolution, SsgiResolution::Half);
        assert_eq!(c.reflection_blur_resolution, ReflectionBlurResolution::Half);
        assert_eq!(AaMode::default(), AaMode::Fxaa);
        assert_eq!(IndirectLighting::default(), IndirectLighting::Ibl);
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
        assert!(!AaMode::Off.fxaa_enabled());
        assert!(AaMode::Fxaa.fxaa_enabled());
        // Taa keeps the FXAA pass: the temporal resolve does not replace it.
        assert!(AaMode::Taa.fxaa_enabled());

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
