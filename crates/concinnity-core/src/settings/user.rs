//! The user's persisted settings-menu choices, the user layer of settings
//! resolution. Every field is an override: `None` leaves the world's value (or
//! the engine default) in place, so an unchanged setting never overrides what
//! the world authors. Reading and writing them to disk is the engine's.

use serde::de::value::StrDeserializer;
use serde::de::{DeserializeOwned, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};

use alloc::string::{String, ToString};
use core::fmt;

use crate::components::{
    AaMode, PassResolution, ShadowUpdate, UpscaleQuality, UpscalerBackend, WindowMode,
};
use crate::platform::Platform;
use crate::render::dlss::DlssPreset;
use crate::settings::quality::QualityPreset;

/// Persisted overrides for graphics settings. Missing fields stay `None` and
/// fall back to the world's settings assets and `Window`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UserGraphicsSettings {
    /// Master graphics-quality preset. `None` means never configured: the first
    /// launch seeds `Auto` (detect the GPU tier and clamp quality under the
    /// world's authored look) and saves once. `Auto` re-resolves from the
    /// detected tier each launch; a named tier is a fixed ceiling; `Custom`
    /// imposes no ceiling (only the per-field overrides below apply).
    #[serde(default, deserialize_with = "lenient")]
    pub quality_preset: Option<QualityPreset>,
    /// Display sync (vsync). `None` uses the world's `FramePacing.vsync`.
    #[serde(default)]
    pub vsync: Option<bool>,
    /// Frame-rate cap in FPS (0 = unlimited). `None` uses the world's
    /// `FramePacing.fps_cap`. Applied live (the render loop's frame pacer reads
    /// it each frame); independent of the quality preset.
    #[serde(default)]
    pub fps_cap: Option<u32>,
    /// Stats-HUD display toggles. `perf_stats` is the master "Display performance
    /// stats" switch; `show_fps` / `show_vram` gate the individual readouts under
    /// it. `None` means shown (the engine default), so an existing settings file
    /// keeps the frame-rate / GPU-memory chips visible. Applied live; with the
    /// master off the per-readout rows stay visible in the menu but grayed out.
    #[serde(default)]
    pub perf_stats: Option<bool>,
    /// The frame-rate readout under `perf_stats`.
    #[serde(default)]
    pub show_fps: Option<bool>,
    /// The GPU-memory readout under `perf_stats`.
    #[serde(default)]
    pub show_vram: Option<bool>,
    /// Window mode (windowed / borderless / fullscreen). `None` uses the world's
    /// `Window.mode`. Applied live.
    #[serde(default, deserialize_with = "lenient")]
    pub window_mode: Option<WindowMode>,
    /// Chosen fullscreen display mode [width, height, refresh_hz] in pixels
    /// (refresh_hz 0 = unknown / keep the display's rate). `None` means never
    /// chosen: the display keeps its own mode and the Resolution row shows it.
    /// Fullscreen-only (the row is grayed in windowed / borderless, where the
    /// window itself defines the size); applied live while fullscreen.
    #[serde(default)]
    pub resolution: Option<[u32; 3]>,
    /// The graphics API to render with. `None` uses the build's native one.
    /// Applied at next launch (one backend is built per process); a choice the
    /// running build did not compile in falls back to the native API.
    #[serde(default, deserialize_with = "lenient")]
    pub graphics_api: Option<Platform>,
    /// Where a switch to `graphics_api` stands, so a launch that never reaches
    /// its first frame on it is not repeated. `None` once a launch has drawn
    /// on the choice (or nothing is on trial).
    #[serde(default, deserialize_with = "lenient")]
    pub graphics_api_trial: Option<GraphicsApiTrial>,
    /// Render-scale preset (upscaling quality). `None` uses the world's
    /// `AntiAliasing.upscale_quality`. Applied at next launch (the upscaler
    /// and render targets are sized once at init).
    #[serde(default, deserialize_with = "lenient")]
    pub render_scale: Option<UpscaleQuality>,
    /// Upscaler backend (`AntiAliasing.upscaler`: Auto / FSR3 / DLSS /
    /// XeSS). `None` uses the world's value. Applied at next launch (the upscaler
    /// is selected + built once at init); DirectX / Vulkan only (Metal uses
    /// MetalFX). A user/hardware preference, independent of the quality preset.
    #[serde(default, deserialize_with = "lenient")]
    pub upscale_backend: Option<UpscalerBackend>,
    /// The DLSS render preset. `None` leaves DLSS's own default. Applied at next
    /// launch, like the backend.
    #[serde(default, deserialize_with = "lenient")]
    pub dlss_preset: Option<DlssPreset>,
    /// Exposure offset in photographic stops. `None` uses the world's
    /// `Exposure.ev`. Applied live (a pure post-process
    /// uniform), and re-applied at init for a persisted choice.
    #[serde(default)]
    pub exposure_ev: Option<f32>,
    /// Bloom additive strength. `None` uses the world's
    /// `Bloom.intensity`. Applied live.
    #[serde(default)]
    pub bloom_intensity: Option<f32>,
    /// Bloom luminance threshold. `None` uses the world's
    /// `Bloom.threshold`. Applied live.
    #[serde(default)]
    pub bloom_threshold: Option<f32>,
    /// Bloom soft-knee width. `None` uses the world's `Bloom.knee`.
    /// Applied live (a `PostProcessParams` field, like the other bloom sliders).
    #[serde(default)]
    pub bloom_knee: Option<f32>,
    /// Vignette strength in [0, 1]. `None` uses the world's
    /// `Vignette.strength`. Applied live.
    #[serde(default)]
    pub vignette: Option<f32>,
    /// Color-LUT blend in [0, 1]. `None` uses the world's
    /// `ColorLut.strength`. Applied live.
    #[serde(default)]
    pub lut_strength: Option<f32>,
    /// Ambient (IBL) light scale. `None` uses the world's
    /// `IndirectLighting.ambient_intensity`. Applied live on Metal (it rides
    /// `LightUniforms`, not `PostProcessParams`); re-applied at init.
    #[serde(default)]
    pub ambient_intensity: Option<f32>,
    /// Camera vertical field of view in degrees. `None` uses the world's authored
    /// `Camera3D.fov_y_degrees`. Applied live (Camera3DSystem updates the camera
    /// from a ControlsCommand; the projection rebuilds from it each frame) and
    /// re-applied at init. A user preference, independent of the quality preset.
    #[serde(default)]
    pub fov: Option<f32>,
    /// Anti-aliasing mode (`AntiAliasing.mode`: off / FXAA / TAA). `None`
    /// uses the world's value. Applied live on Metal (the TAA pass rebuilds and
    /// the composite FXAA flag updates in place) and governed by the quality
    /// preset ceiling like the toggles below.
    #[serde(default, deserialize_with = "lenient")]
    pub aa_mode: Option<AaMode>,
    /// Quality-feature toggles. Each `None` uses the world's authored
    /// value. They gate render passes whose GPU resources
    /// (pipelines, targets, acceleration structures) are built at init, so a
    /// change rebuilds those resources: applied live on Metal (the backend
    /// rebuilds the affected effects in place); on backends without a live path
    /// the choice persists and applies at the next launch.
    #[serde(default)]
    pub ssao: Option<bool>,
    /// Screen-space reflections (`Reflections.screen_space`).
    #[serde(default)]
    pub ssr: Option<bool>,
    /// Hardware ray-traced reflections (`Reflections.ray_traced`).
    #[serde(default)]
    pub ray_traced_reflections: Option<bool>,
    /// Sun shadows inside ray-traced reflections
    /// (`Reflections.ray_traced_shadows`).
    #[serde(default)]
    pub rt_reflection_shadows: Option<bool>,
    /// Screen-space global illumination (`IndirectLighting.technique ==
    /// ssgi`).
    #[serde(default)]
    pub ssgi: Option<bool>,
    /// Auto-exposure (`Exposure.auto`).
    #[serde(default)]
    pub auto_exposure: Option<bool>,
    /// SSGI sub-quality: trace resolution and hemisphere rays per pixel per frame
    /// (`IndirectLighting.ssgi_resolution` / `ssgi_rays`). Each `None` uses the
    /// world's value. Applied live on Metal (the
    /// backend rebuilds the SSGI pass in place); persisted + applied at the next
    /// launch on backends without a live path. Governed by the quality preset
    /// ceiling like the toggles above.
    #[serde(default, deserialize_with = "lenient")]
    pub ssgi_resolution: Option<PassResolution>,
    /// SSGI rays per pixel per frame (`IndirectLighting.ssgi_rays`).
    #[serde(default)]
    pub ssgi_rays: Option<u32>,
    /// Roughness-aware reflection blur resolution
    /// (`Reflections.blur_resolution`). `None` uses the world's
    /// value. Applied live on Metal; governed by the quality preset ceiling like
    /// the SSGI sub-quality above (only bites when a reflection feature is on).
    #[serde(default, deserialize_with = "lenient")]
    pub reflection_blur_resolution: Option<PassResolution>,
    /// Ray-traced reflection trace resolution
    /// (`Reflections.ray_traced_resolution`). `None` uses the world's
    /// value; governed by the quality preset ceiling.
    #[serde(default, deserialize_with = "lenient")]
    pub rt_reflection_resolution: Option<PassResolution>,
    /// Per-feature sub-quality tunables (SSAO radius / intensity, SSR intensity /
    /// distance, SSGI intensity / distance, auto-exposure EV bounds + speed). Each
    /// `None` uses the world's authored value. Applied live: the backend
    /// re-reads them into a per-frame uniform with no pass rebuild. Look-tuning
    /// knobs, independent of the quality preset.
    #[serde(default)]
    pub ssao_radius: Option<f32>,
    /// `AmbientOcclusion.intensity`.
    #[serde(default)]
    pub ssao_intensity: Option<f32>,
    /// `Reflections.intensity`.
    #[serde(default)]
    pub ssr_intensity: Option<f32>,
    /// `Reflections.max_distance`.
    #[serde(default)]
    pub ssr_max_distance: Option<f32>,
    /// `IndirectLighting.ssgi_intensity`.
    #[serde(default)]
    pub ssgi_intensity: Option<f32>,
    /// `IndirectLighting.ssgi_max_distance`.
    #[serde(default)]
    pub ssgi_max_distance: Option<f32>,
    /// `Exposure.auto_min_ev`.
    #[serde(default)]
    pub auto_exposure_min_ev: Option<f32>,
    /// `Exposure.auto_max_ev`.
    #[serde(default)]
    pub auto_exposure_max_ev: Option<f32>,
    /// `Exposure.auto_speed`.
    #[serde(default)]
    pub auto_exposure_speed: Option<f32>,
    /// Shadow quality: cascade map resolution in texels (0 disables shadows) and
    /// re-render cadence (`Shadows.map_size` / `shadow_update`).
    /// `None` uses the world's value. Resolution is restart-required (the shadow
    /// map array is sized once at backend init); cadence is applied live on Metal.
    /// Both are governed by the quality preset ceiling like the toggles above.
    #[serde(default)]
    pub shadow_map_size: Option<u32>,
    /// Shadow re-render cadence (`Shadows.update`).
    #[serde(default, deserialize_with = "lenient")]
    pub shadow_update: Option<ShadowUpdate>,
    /// Shadow distance in world units (`Shadows.distance`). `None`
    /// uses the world's value. Applied live on Metal (the cascade-split math reads
    /// it each frame) and governed by the quality preset ceiling like the shadow
    /// knobs above.
    #[serde(default)]
    pub shadow_distance: Option<u32>,
    /// Shadow cascade count, 1..4 (`Shadows.cascades`). `None` uses
    /// the world's value. Applied live on Metal (the per-frame split + schedule
    /// read it) and governed by the quality preset ceiling like the shadow knobs
    /// above.
    #[serde(default)]
    pub shadow_cascades: Option<u32>,
    /// Anisotropic-filtering degree for the scene sampler
    /// (`TextureFiltering.anisotropy`). `None` uses the world's value. Restart-
    /// required (the sampler is built once at backend init) and governed by the
    /// quality preset ceiling like the shadow knobs above.
    #[serde(default)]
    pub anisotropy: Option<u32>,
    /// Display-output / upscaling preferences. Unlike the quality knobs above,
    /// these are independent of the master preset (a user choice, not a tier), and
    /// each is restart-required: the swapchain format / render targets are sized
    /// once at backend init, so a change persists and applies at the next launch.
    /// `None` uses the world's `Display` value.
    #[serde(default)]
    pub temporal_upscaling: Option<bool>,
    /// HDR display output (`Display.hdr`).
    #[serde(default)]
    pub hdr_display: Option<bool>,
    /// PQ (HDR10) output (`Display.pq`).
    #[serde(default)]
    pub hdr_pq: Option<bool>,
    /// System / streaming restart preferences, independent of the master preset
    /// (like the display rows above) and each restart-required: ring-buffer depth
    /// (`FramePacing.frames_in_flight`), two-pass occlusion culling
    /// (`Culling.two_pass_occlusion`), and the texture-streaming pool /
    /// per-frame upload budget (`StreamingConfig.texture_cap` / `texture_budget`,
    /// driven together by one "Texture Quality" row). `None` uses the world's value.
    #[serde(default)]
    pub frames_in_flight: Option<u32>,
    /// Two-pass occlusion culling (`Culling.two_pass_occlusion`).
    #[serde(default)]
    pub occlusion_two_pass: Option<bool>,
    /// Texture-streaming pool cap (`StreamingConfig.texture_cap`).
    #[serde(default)]
    pub texture_cap: Option<u32>,
    /// Texture-streaming per-frame upload budget (`StreamingConfig.texture_budget`).
    #[serde(default)]
    pub texture_budget: Option<u32>,
}

/// How far a launch has gotten on a newly chosen graphics API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphicsApiTrial {
    /// Chosen in the settings menu; no launch has tried it yet.
    Untried,
    /// A launch is rendering with it and has not drawn a frame yet. A launch
    /// that finds this left over starts on the native API instead.
    Trying,
}

impl UserGraphicsSettings {
    /// These overrides with each one `top` sets in place of this one's: how a
    /// session override layers over the user's persisted choices.
    pub fn overlaid_by(&self, top: &Self) -> Self {
        let Self {
            quality_preset,
            vsync,
            fps_cap,
            perf_stats,
            show_fps,
            show_vram,
            window_mode,
            resolution,
            graphics_api,
            graphics_api_trial,
            render_scale,
            upscale_backend,
            dlss_preset,
            exposure_ev,
            bloom_intensity,
            bloom_threshold,
            bloom_knee,
            vignette,
            lut_strength,
            ambient_intensity,
            fov,
            aa_mode,
            ssao,
            ssr,
            ray_traced_reflections,
            rt_reflection_shadows,
            ssgi,
            auto_exposure,
            ssgi_resolution,
            ssgi_rays,
            reflection_blur_resolution,
            rt_reflection_resolution,
            ssao_radius,
            ssao_intensity,
            ssr_intensity,
            ssr_max_distance,
            ssgi_intensity,
            ssgi_max_distance,
            auto_exposure_min_ev,
            auto_exposure_max_ev,
            auto_exposure_speed,
            shadow_map_size,
            shadow_update,
            shadow_distance,
            shadow_cascades,
            anisotropy,
            temporal_upscaling,
            hdr_display,
            hdr_pq,
            frames_in_flight,
            occlusion_two_pass,
            texture_cap,
            texture_budget,
        } = *top;
        Self {
            quality_preset: quality_preset.or(self.quality_preset),
            vsync: vsync.or(self.vsync),
            fps_cap: fps_cap.or(self.fps_cap),
            perf_stats: perf_stats.or(self.perf_stats),
            show_fps: show_fps.or(self.show_fps),
            show_vram: show_vram.or(self.show_vram),
            window_mode: window_mode.or(self.window_mode),
            resolution: resolution.or(self.resolution),
            graphics_api: graphics_api.or(self.graphics_api),
            graphics_api_trial: graphics_api_trial.or(self.graphics_api_trial),
            render_scale: render_scale.or(self.render_scale),
            upscale_backend: upscale_backend.or(self.upscale_backend),
            dlss_preset: dlss_preset.or(self.dlss_preset),
            exposure_ev: exposure_ev.or(self.exposure_ev),
            bloom_intensity: bloom_intensity.or(self.bloom_intensity),
            bloom_threshold: bloom_threshold.or(self.bloom_threshold),
            bloom_knee: bloom_knee.or(self.bloom_knee),
            vignette: vignette.or(self.vignette),
            lut_strength: lut_strength.or(self.lut_strength),
            ambient_intensity: ambient_intensity.or(self.ambient_intensity),
            fov: fov.or(self.fov),
            aa_mode: aa_mode.or(self.aa_mode),
            ssao: ssao.or(self.ssao),
            ssr: ssr.or(self.ssr),
            ray_traced_reflections: ray_traced_reflections.or(self.ray_traced_reflections),
            rt_reflection_shadows: rt_reflection_shadows.or(self.rt_reflection_shadows),
            ssgi: ssgi.or(self.ssgi),
            auto_exposure: auto_exposure.or(self.auto_exposure),
            ssgi_resolution: ssgi_resolution.or(self.ssgi_resolution),
            ssgi_rays: ssgi_rays.or(self.ssgi_rays),
            reflection_blur_resolution: reflection_blur_resolution
                .or(self.reflection_blur_resolution),
            rt_reflection_resolution: rt_reflection_resolution.or(self.rt_reflection_resolution),
            ssao_radius: ssao_radius.or(self.ssao_radius),
            ssao_intensity: ssao_intensity.or(self.ssao_intensity),
            ssr_intensity: ssr_intensity.or(self.ssr_intensity),
            ssr_max_distance: ssr_max_distance.or(self.ssr_max_distance),
            ssgi_intensity: ssgi_intensity.or(self.ssgi_intensity),
            ssgi_max_distance: ssgi_max_distance.or(self.ssgi_max_distance),
            auto_exposure_min_ev: auto_exposure_min_ev.or(self.auto_exposure_min_ev),
            auto_exposure_max_ev: auto_exposure_max_ev.or(self.auto_exposure_max_ev),
            auto_exposure_speed: auto_exposure_speed.or(self.auto_exposure_speed),
            shadow_map_size: shadow_map_size.or(self.shadow_map_size),
            shadow_update: shadow_update.or(self.shadow_update),
            shadow_distance: shadow_distance.or(self.shadow_distance),
            shadow_cascades: shadow_cascades.or(self.shadow_cascades),
            anisotropy: anisotropy.or(self.anisotropy),
            temporal_upscaling: temporal_upscaling.or(self.temporal_upscaling),
            hdr_display: hdr_display.or(self.hdr_display),
            hdr_pq: hdr_pq.or(self.hdr_pq),
            frames_in_flight: frames_in_flight.or(self.frames_in_flight),
            occlusion_two_pass: occlusion_two_pass.or(self.occlusion_two_pass),
            texture_cap: texture_cap.or(self.texture_cap),
            texture_budget: texture_budget.or(self.texture_budget),
        }
    }
}

/// Read an `Option<enum>` override, taking a value this build does not know (a
/// variant since removed, or one a newer build wrote) as unset. One stale field
/// then falls back to the world's value instead of failing the whole store,
/// which would drop every other saved choice with it.
fn lenient<'de, D, T>(d: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: DeserializeOwned,
{
    let name = d.deserialize_any(VariantName)?;
    Ok(name.and_then(|name| {
        T::deserialize(StrDeserializer::<serde::de::value::Error>::new(&name)).ok()
    }))
}

/// The name of a unit variant, or `None` for any other value, which is read
/// through and dropped.
struct VariantName;

impl<'de> Visitor<'de> for VariantName {
    type Value = Option<String>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a variant name")
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
        Ok(Some(v.to_string()))
    }

    fn visit_bool<E>(self, _: bool) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_i64<E>(self, _: i64) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_i128<E>(self, _: i128) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_u64<E>(self, _: u64) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_u128<E>(self, _: u128) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_f64<E>(self, _: f64) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_bytes<E>(self, _: &[u8]) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        d.deserialize_any(self)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        while seq.next_element::<IgnoredAny>()?.is_some() {}
        Ok(None)
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_session_override_replaces_only_what_it_sets() {
        let user = UserGraphicsSettings {
            vsync: Some(true),
            dlss_preset: Some(DlssPreset::K),
            ssao: Some(false),
            ..Default::default()
        };
        let session = UserGraphicsSettings {
            dlss_preset: Some(DlssPreset::M),
            fps_cap: Some(30),
            ..Default::default()
        };
        let merged = user.overlaid_by(&session);
        assert_eq!(merged.dlss_preset, Some(DlssPreset::M));
        assert_eq!(merged.fps_cap, Some(30));
        assert_eq!(merged.vsync, Some(true));
        assert_eq!(merged.ssao, Some(false));
        assert_eq!(user.overlaid_by(&UserGraphicsSettings::default()), user);
    }

    // A value this build does not know unsets that one override and leaves the
    // rest of the store intact.
    #[test]
    fn an_unknown_enum_value_unsets_only_its_own_override() {
        let json = r#"{
            "vsync": true,
            "dlss_preset": "preset_z",
            "aa_mode": "taa",
            "upscale_backend": 7,
            "shadow_update": null,
            "window_mode": {"Tiled": 2},
            "render_scale": ["quality"],
            "ssgi_resolution": "quarter",
            "quality_preset": "High"
        }"#;
        let g: UserGraphicsSettings = serde_json::from_str(json).unwrap();
        assert_eq!(g.vsync, Some(true));
        assert_eq!(g.dlss_preset, None);
        assert_eq!(g.aa_mode, Some(AaMode::Taa));
        assert_eq!(g.upscale_backend, None);
        assert_eq!(g.shadow_update, None);
        assert_eq!(g.window_mode, None);
        assert_eq!(g.render_scale, None);
        assert_eq!(g.ssgi_resolution, Some(PassResolution::Quarter));
        assert_eq!(g.quality_preset, Some(QualityPreset::High));
    }
}
