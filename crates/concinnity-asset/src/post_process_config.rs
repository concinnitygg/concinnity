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
/// ```jsonl
/// {"name":"post","type":"PostProcessConfig","args":{"bloom_intensity":0.8}}
/// {"name":"post_dim","type":"PostProcessConfig","args":{"exposure_ev":-1.0,"vignette_strength":0.4}}
/// {"name":"post_taa","type":"PostProcessConfig","args":{"aa_mode":"taa"}}
/// {"name":"post_ssao","type":"PostProcessConfig","args":{"ssao":true,"ssao_radius":0.6}}
/// {"name":"post_ssr","type":"PostProcessConfig","args":{"ssr":true,"ssr_intensity":0.8}}
/// {"name":"post_rt","type":"PostProcessConfig","args":{"ray_traced_reflections":true,"ssr_intensity":0.8}}
/// {"name":"post_refl_blur","type":"PostProcessConfig","args":{"ssr":true,"reflection_blur_resolution":"quarter"}}
/// {"name":"post_ssgi","type":"PostProcessConfig","args":{"indirect_lighting":"ssgi","ssgi_intensity":0.6}}
/// {"name":"post_auto_ev","type":"PostProcessConfig","args":{"auto_exposure":true}}
/// {"name":"post_hdr","type":"PostProcessConfig","args":{"hdr_display":true}}
/// {"name":"post_upscale","type":"PostProcessConfig","args":{"temporal_upscaling":true,"upscale_quality":"balanced"}}
/// {"name":"post_dlss","type":"PostProcessConfig","args":{"temporal_upscaling":true,"upscale_backend":"dlss"}}
/// {"name":"post_occ2","type":"PostProcessConfig","args":{"occlusion_two_pass":true}}
/// {"name":"post_off","type":"PostProcessConfig","args":{"bloom_intensity":0.0}}
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
    #[default]
    Quality,
    Balanced,
    Performance,
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
    #[default]
    Auto,
    Fsr3,
    Dlss,
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
    Off,
    #[default]
    Fxaa,
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
}

/// Indirect-diffuse lighting source for `PostProcessConfig.indirect_lighting`.
/// `Ibl` is the image-based-lighting-only ambient term the renderer has always
/// used; `Ssgi` layers a screen-space global-illumination bounce on top.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum IndirectLighting {
    #[default]
    Ibl,
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
    Full,
    #[default]
    Half,
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
    Full,
    #[default]
    Half,
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

// Default SSGI hemisphere-ray and ray-march-step counts for the authored
// `ssgi_rays` / `ssgi_steps` fields. Defined here (the schema default) and
// re-exported by `concinnity-core`'s `gfx::ssgi` for its runtime clamp path, so
// the authored default and the runtime code stay a single source of truth.
pub const DEFAULT_SSGI_RAYS: u32 = 8;
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
