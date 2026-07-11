// src/assets/post_process_config.rs
//
// Runtime behavior for the PostProcessConfig asset. The authored schema (the
// struct, its enums, and their Default) lives in concinnity-asset; this file
// keeps the `Component` impl plus the `PostProcessResolve` extension trait that
// resolves the authored tunables into the renderer's clamped `gfx` settings.

use crate::assets::{IndirectLighting, PostProcessConfig};
use crate::ecs::{AssetOrigin, Component};
use crate::gfx::render_types::PostProcessParams;

// `exposure_ev` is clamped to this range before resolving to a multiplier so a
// stray value cannot push the scene to `inf` / `0`.
const EXPOSURE_EV_LIMIT: f32 = 16.0;

// Resolves a `PostProcessConfig`'s authored tunables into the clamped,
// GPU-facing settings the renderer consumes. Kept in core (not concinnity-asset)
// because every return type is a `crate::gfx` settings struct.
pub trait PostProcessResolve {
    // Resolve the authored fields into the GPU-facing `PostProcessParams`:
    // clamps each tunable and converts `exposure_ev` (stops) into the linear
    // multiplier the shaders expect. `hdr_output` is left at 0.0 (SDR path);
    // the backend overwrites it to 1.0 only after confirming EDR support, so the
    // asset-side resolve stays pure.
    fn resolve(&self) -> PostProcessParams;

    // Clamp the authored `ambient_intensity` to a safe `[0, 16]` multiplier the
    // backend folds into `LightUniforms` to scale the indirect (ambient / IBL)
    // term.
    fn ambient_intensity(&self) -> f32;

    // Per-axis divisor for the roughness-aware reflection blur target, resolved
    // from `reflection_blur_resolution`. Always at least 1.
    fn reflection_blur_divisor(&self) -> u32;

    // Resolve the SSAO tunables into clamped `SsaoSettings`, or `None` when the
    // `ssao` toggle is off so the backend can skip the SSAO passes entirely.
    fn ssao_settings(&self) -> Option<crate::gfx::ssao::SsaoSettings>;

    // Resolve the SSR tunables into clamped `SsrSettings`, or `None` when the
    // `ssr` toggle is off.
    fn ssr_settings(&self) -> Option<crate::gfx::ssr::SsrSettings>;

    // Resolve the ray-traced-reflection tunables into clamped
    // `RtReflectionSettings`, or `None` when `ray_traced_reflections` is off.
    // Reuses the SSR intensity / distance fields; the backend additionally gates
    // on GPU ray-tracing support.
    fn rt_reflection_settings(&self) -> Option<crate::gfx::rt_reflections::RtReflectionSettings>;

    // Resolve the SSGI tunables into clamped `SsgiSettings`, or `None` when
    // `indirect_lighting` is not `Ssgi` so the backend can skip the SSGI passes.
    fn ssgi_settings(&self) -> Option<crate::gfx::ssgi::SsgiSettings>;

    // Resolve the auto-exposure tunables into clamped `AutoExposureSettings`, or
    // `None` when the toggle is off so the backend can skip the histogram passes.
    fn auto_exposure_settings(&self) -> Option<crate::gfx::auto_exposure::AutoExposureSettings>;
}

impl PostProcessResolve for PostProcessConfig {
    fn resolve(&self) -> PostProcessParams {
        let ev = self
            .exposure_ev
            .clamp(-EXPOSURE_EV_LIMIT, EXPOSURE_EV_LIMIT);
        PostProcessParams {
            bloom_intensity: self.bloom_intensity.max(0.0),
            bloom_threshold: self.bloom_threshold.max(0.0),
            bloom_knee: self.bloom_knee.max(0.0),
            exposure: ev.exp2(),
            vignette: self.vignette_strength.clamp(0.0, 1.0),
            lut_strength: self.lut_strength.clamp(0.0, 1.0),
            hdr_output: 0.0,
            pq_output: 0.0,
            fxaa: if self.aa_mode.fxaa_enabled() {
                1.0
            } else {
                0.0
            },
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
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    // Pass-through leaf: baked component is its authored args (see PointLight).
    const BAKED: bool = true;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }

    fn from_args(args: Self) -> Self {
        args
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{
        AaMode, ReflectionBlurResolution, SsgiResolution, UpscaleQuality, UpscalerBackend,
    };

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
    fn aa_mode_defaults_to_fxaa_and_round_trips_through_args() {
        assert_eq!(PostProcessConfig::default().aa_mode, AaMode::Fxaa);
        let cfg = PostProcessConfig {
            aa_mode: AaMode::Taa,
            ..Default::default()
        };
        assert_eq!(
            PostProcessConfig::from_args(cfg.to_args()).aa_mode,
            AaMode::Taa
        );
    }

    #[test]
    fn aa_mode_gates_taa_and_fxaa() {
        assert!(!AaMode::Off.taa_enabled());
        assert!(!AaMode::Off.fxaa_enabled());
        assert!(!AaMode::Fxaa.taa_enabled());
        assert!(AaMode::Fxaa.fxaa_enabled());
        assert!(AaMode::Taa.taa_enabled());
        assert!(AaMode::Taa.fxaa_enabled());
        // resolve() carries the FXAA gate into the composite uniform.
        let off = PostProcessConfig {
            aa_mode: AaMode::Off,
            ..Default::default()
        };
        assert_eq!(off.resolve().fxaa, 0.0);
        assert_eq!(PostProcessConfig::default().resolve().fxaa, 1.0);
    }

    #[test]
    fn ssao_defaults_off_with_neutral_tunables() {
        let cfg = PostProcessConfig::default();
        assert!(!cfg.ssao);
        assert_eq!(cfg.ssao_radius, 0.5);
        assert_eq!(cfg.ssao_intensity, 1.0);
        // No SsaoSettings produced while the toggle is off.
        assert!(cfg.ssao_settings().is_none());
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
    fn ssr_defaults_off_with_neutral_tunables() {
        let cfg = PostProcessConfig::default();
        assert!(!cfg.ssr);
        assert_eq!(cfg.ssr_intensity, 0.7);
        assert_eq!(cfg.ssr_max_distance, 40.0);
        // No SsrSettings produced while the toggle is off.
        assert!(cfg.ssr_settings().is_none());
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
    fn rt_reflections_default_off_and_resolve_to_none() {
        let cfg = PostProcessConfig::default();
        assert!(!cfg.ray_traced_reflections);
        // No RtReflectionSettings produced while the toggle is off.
        assert!(cfg.rt_reflection_settings().is_none());
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
        // Omitting the field leaves ray tracing off.
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"bloom_intensity":0.5}"#).expect("parse");
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
    fn ssgi_defaults_to_ibl_with_neutral_tunables() {
        let cfg = PostProcessConfig::default();
        assert_eq!(cfg.indirect_lighting, IndirectLighting::Ibl);
        assert_eq!(cfg.ssgi_intensity, 0.5);
        assert_eq!(cfg.ssgi_max_distance, 8.0);
        // The gather defaults to half resolution with the historical 8x12
        // ray/step counts.
        assert_eq!(cfg.ssgi_resolution, SsgiResolution::Half);
        assert_eq!(cfg.ssgi_rays, 8);
        assert_eq!(cfg.ssgi_steps, 12);
        // No SsgiSettings produced while indirect lighting is IBL-only.
        assert!(cfg.ssgi_settings().is_none());
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
        // Omitting the field leaves indirect lighting on IBL.
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"bloom_intensity":0.5}"#).expect("parse");
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
        // Omitting the field falls back to the FXAA default.
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"bloom_intensity":0.5}"#).expect("parse");
        assert_eq!(cfg.aa_mode, AaMode::Fxaa);
        // "off" disables edge smoothing entirely.
        let cfg: PostProcessConfig = serde_json::from_str(r#"{"aa_mode":"off"}"#).expect("parse");
        assert_eq!(cfg.aa_mode, AaMode::Off);
    }

    #[test]
    fn hdr_display_defaults_off_and_resolve_leaves_flag_zero() {
        let cfg = PostProcessConfig::default();
        assert!(!cfg.hdr_display);
        // The asset-side resolve always emits the SDR path; the backend is the
        // one that promotes hdr_output to 1.0 after confirming EDR.
        assert_eq!(cfg.resolve().hdr_output, 0.0);
    }

    #[test]
    fn hdr_display_round_trips_through_args_and_jsonl() {
        let cfg = PostProcessConfig {
            hdr_display: true,
            ..Default::default()
        };
        assert!(PostProcessConfig::from_args(cfg.to_args()).hdr_display);

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
    fn occlusion_two_pass_defaults_off_and_round_trips() {
        assert!(!PostProcessConfig::default().occlusion_two_pass);
        let cfg = PostProcessConfig {
            occlusion_two_pass: true,
            ..Default::default()
        };
        assert!(PostProcessConfig::from_args(cfg.to_args()).occlusion_two_pass);
        // Deserialises from jsonl args; omitting it leaves the feature off.
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"occlusion_two_pass":true}"#).expect("parse");
        assert!(cfg.occlusion_two_pass);
        let cfg: PostProcessConfig =
            serde_json::from_str(r#"{"bloom_intensity":0.5}"#).expect("parse");
        assert!(!cfg.occlusion_two_pass);
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
        assert_eq!(
            PostProcessConfig::from_args(cfg.to_args()).upscale_backend,
            UpscalerBackend::Xess
        );
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
