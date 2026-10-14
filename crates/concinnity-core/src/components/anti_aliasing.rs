// Anti-aliasing and temporal-upscaling schema.

use crate::components::Vocabulary;

/// Anti-aliasing and temporal upscaling: how edges are smoothed, and whether
/// the scene renders below native resolution and is reconstructed to full. The
/// `anti_aliasing` section of [GraphicsConfig](#graphicsconfig); a field left
/// unset takes the default below.
///
/// Temporal anti-aliasing is the default, which is the look on capable
/// hardware: the `Auto` graphics quality preset clamps it to `fxaa` below the
/// mid tier.
///
/// ```rust
/// # use concinnity_core::components::{AaMode, AntiAliasingOverrides, GraphicsConfig};
/// GraphicsConfig {
///     anti_aliasing: AntiAliasingOverrides {
///         mode: Some(AaMode::Fxaa),
///         ..Default::default()
///     },
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[asset(section)]
#[serde(default)]
pub struct AntiAliasing {
    /// Anti-aliasing mode. `fxaa` applies a cheap composite-pass edge filter;
    /// `taa` (default) adds a temporal pass that jitters the projection and
    /// accumulates detail across frames for the cleanest edges, at the cost of a
    /// velocity pre-pass and a history buffer; `off` disables edge smoothing.
    #[asset(default = AaMode::Taa, apply = rebuild)]
    pub mode: AaMode,
    /// Temporal upscaling. Renders the 3D scene at a lower resolution (set by
    /// `upscale_quality`) and reconstructs a full-resolution image, trading
    /// some sharpness for performance. Replaces TAA while on.
    #[asset(apply = restart)]
    pub upscaling: bool,
    /// Render-scale preset for `upscaling`; each step progressively lowers the
    /// internal resolution. No effect when `upscaling` is off.
    #[asset(apply = restart)]
    pub upscale_quality: UpscaleQuality,
    /// Which upscaler `upscaling` uses. `auto` (default) picks the best
    /// available at runtime (DLSS on NVIDIA RTX, else XeSS, else FSR3); `fsr3` /
    /// `dlss` / `xess` request a specific one and fall back when it is
    /// unavailable on the current GPU or build. No effect when `upscaling` is
    /// off. Metal treats every value as MetalFX.
    #[asset(apply = restart)]
    pub upscaler: UpscalerBackend,
}

/// Render-scale preset for `GraphicsConfig.anti_aliasing.upscaling`. The ratio applies to both
/// axes (input pixel count = output * ratio per axis), so `Quality` renders at
/// 4/9 of the output pixel count, `Performance` at 1/4, and `UltraPerformance`
/// at 1/9.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default, Vocabulary)]
pub enum UpscaleQuality {
    /// 4/9 of the output pixel count.
    #[default]
    #[vocab("quality")]
    Quality,
    /// Roughly a third of the output pixel count.
    #[vocab("balanced")]
    Balanced,
    /// A quarter of the output pixel count.
    #[vocab("performance")]
    Performance,
    /// A ninth of the output pixel count.
    #[vocab("ultra_performance")]
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

    /// The preset nearest a per-axis render-to-output ratio, or `None` for a
    /// native-resolution ratio, where an upscaler only anti-aliases.
    pub fn nearest(scale: f32) -> Option<Self> {
        if scale >= 0.99 {
            None
        } else if scale >= 0.62 {
            Some(UpscaleQuality::Quality)
        } else if scale >= 0.55 {
            Some(UpscaleQuality::Balanced)
        } else if scale >= 0.42 {
            Some(UpscaleQuality::Performance)
        } else {
            Some(UpscaleQuality::UltraPerformance)
        }
    }
}

/// Upscaler selector for `GraphicsConfig.anti_aliasing.upscaler`. `Auto` resolves at runtime
/// to the best available (DLSS, then XeSS, then FSR3); the explicit variants
/// request a specific upscaler and fall back when it is unavailable. DirectX
/// and Vulkan both resolve all three, each from the vendor SDK staged beside
/// the binary; Metal uses MetalFX and treats any value as its native path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default, Vocabulary)]
pub enum UpscalerBackend {
    /// Pick the best upscaler the device offers.
    #[default]
    #[vocab("auto")]
    Auto,
    /// AMD FidelityFX Super Resolution 3.
    #[vocab("fsr3")]
    Fsr3,
    /// NVIDIA DLSS, through NGX.
    #[vocab("dlss")]
    Dlss,
    /// Intel XeSS.
    #[vocab("xess")]
    Xess,
}

/// Anti-aliasing mode for `GraphicsConfig.anti_aliasing.mode`. `Off` runs no edge smoothing;
/// `Fxaa` applies the composite's single-frame edge filter, which is nearly
/// free; `Taa` adds a temporal pass that jitters the projection and reprojects
/// detail across frames for the cleanest edges, at the cost of a velocity
/// pre-pass and a per-frame history buffer.
///
/// The mode also decides whether the scene renders multisampled: `Off` and
/// `Fxaa` keep the multisampled target, `Taa` renders single-sampled because
/// the temporal filter reconstructs the same edges. Temporal upscaling does
/// the same. See `hdr_sample_count`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default, Vocabulary)]
pub enum AaMode {
    /// No edge smoothing.
    #[vocab("off")]
    Off,
    /// Single-frame edge filter in the composite.
    #[default]
    #[vocab("fxaa")]
    Fxaa,
    /// Temporal anti-aliasing: jittered projection plus a reprojected history.
    #[vocab("taa")]
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

/// Sample count the off-screen HDR render target uses when no temporal
/// technique is active. The pre-post-process color and depth attachments, the
/// pipelines that write them, and the planar-mirror and probe-bake faces that
/// reuse those pipelines all carry this count.
pub const HDR_MULTISAMPLE_COUNT: u32 = 4;

/// Sample count for the off-screen HDR render target, resolved from the
/// anti-aliasing mode and whether temporal upscaling runs.
///
/// Temporal anti-aliasing and temporal upscaling both jitter the projection and
/// accumulate sub-pixel coverage across frames, so multisampling the pass they
/// consume pays sample-rate depth, sample-rate coverage and a full resolve for
/// edge quality the temporal filter reconstructs anyway. Both therefore render
/// single-sampled; every other mode keeps the multisampled target.
///
/// This is the *requested* count: a backend clamps it to what the device
/// reports for the HDR format, and a backend whose temporal upscaler fails to
/// initialize keeps the single-sample target rather than rebuilding every
/// pipeline that baked the count.
pub fn hdr_sample_count(aa_mode: AaMode, temporal_upscaling: bool) -> u32 {
    if aa_mode.taa_enabled() || temporal_upscaling {
        1
    } else {
        HDR_MULTISAMPLE_COUNT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_inverts_each_preset_scale() {
        for q in [
            UpscaleQuality::Quality,
            UpscaleQuality::Balanced,
            UpscaleQuality::Performance,
            UpscaleQuality::UltraPerformance,
        ] {
            assert_eq!(UpscaleQuality::nearest(q.scale()), Some(q));
        }
        assert_eq!(UpscaleQuality::nearest(1.0), None);
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
    fn temporal_modes_resolve_to_a_single_sample_hdr_target() {
        // Both temporal techniques reconstruct edges from a jittered history,
        // so neither pays for multisampling. Upscaling wins even when the AA
        // mode is not TAA: the scaler jitters the projection either way.
        assert_eq!(hdr_sample_count(AaMode::Taa, false), 1);
        assert_eq!(hdr_sample_count(AaMode::Off, true), 1);
        assert_eq!(hdr_sample_count(AaMode::Fxaa, true), 1);
        assert_eq!(hdr_sample_count(AaMode::Taa, true), 1);
    }

    #[test]
    fn non_temporal_modes_keep_the_multisampled_hdr_target() {
        assert_eq!(hdr_sample_count(AaMode::Off, false), HDR_MULTISAMPLE_COUNT);
        assert_eq!(hdr_sample_count(AaMode::Fxaa, false), HDR_MULTISAMPLE_COUNT);
        const { assert!(HDR_MULTISAMPLE_COUNT > 1) };
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
    }

    #[test]
    fn taa_is_the_authored_default() {
        let parsed: AntiAliasing = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed.mode, AaMode::Taa);
        assert!(!parsed.upscaling);
    }
}
