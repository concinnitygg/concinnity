//! The resolved graphics settings: `GraphicsConfig`'s sections with every field
//! set, which the settings menu shows and the renderer is built from.

use crate::components::{
    AaMode, AmbientOcclusion, AntiAliasing, Background, Bloom, Culling, Display, Exposure,
    FramePacing, GraphicsConfig, IndirectLighting, IndirectTechnique, PlanetDetail, Reflections,
    Shadows, TextureFiltering, Vignette, hdr_sample_count,
};
use crate::gfx::auto_exposure::AutoExposureSettings;
use crate::gfx::render_types::PostProcessTunables;
use crate::render::backend_init::{ShadowCadence, ShadowParams};
use crate::render::dlss::DlssPreset;
use crate::render::post::rt_reflections::RtReflectionSettings;
use crate::render::post::ssao::settings::SsaoSettings;
use crate::render::post::ssgi::settings::SsgiSettings;
use crate::render::post::ssr::settings::SsrSettings;

use super::post_process_tunables;

/// Every graphics setting, resolved: one section per `GraphicsConfig` section
/// with each field the world, the quality ceiling, the user and the device
/// settled on, plus the two values that live outside `GraphicsConfig`.
#[derive(Clone, Debug)]
pub struct ResolvedGraphics {
    /// Frame pacing.
    pub frame_pacing: FramePacing,
    /// Cascaded shadows.
    pub shadows: Shadows,
    /// Texture filtering.
    pub texture_filtering: TextureFiltering,
    /// The color shown where nothing draws.
    pub background: Background,
    /// Screen-space ambient occlusion.
    pub ambient_occlusion: AmbientOcclusion,
    /// Screen-space and ray-traced reflections.
    pub reflections: Reflections,
    /// The ambient term and the screen-space bounce.
    pub indirect_lighting: IndirectLighting,
    /// Anti-aliasing and temporal upscaling.
    pub anti_aliasing: AntiAliasing,
    /// Fixed and automatic exposure.
    pub exposure: Exposure,
    /// Bloom.
    pub bloom: Bloom,
    /// Vignette.
    pub vignette: Vignette,
    /// Display output.
    pub display: Display,
    /// Culling.
    pub culling: Culling,
    /// Planet terrain tile detail.
    pub planet_detail: PlanetDetail,
    /// The `ColorLut` blend; 1.0 without one.
    pub lut_strength: f32,
    /// The DLSS render preset, chosen by the user only.
    pub dlss_preset: DlssPreset,
}

impl Default for ResolvedGraphics {
    fn default() -> Self {
        Self {
            frame_pacing: FramePacing::default(),
            shadows: Shadows::default(),
            texture_filtering: TextureFiltering::default(),
            background: Background::default(),
            ambient_occlusion: AmbientOcclusion::default(),
            reflections: Reflections::default(),
            indirect_lighting: IndirectLighting::default(),
            anti_aliasing: AntiAliasing::default(),
            exposure: Exposure::default(),
            bloom: Bloom::default(),
            vignette: Vignette::default(),
            display: Display::default(),
            culling: Culling::default(),
            planet_detail: PlanetDetail::default(),
            lut_strength: 1.0,
            dlss_preset: DlssPreset::default(),
        }
    }
}

impl ResolvedGraphics {
    /// Replace every field `graphics` sets.
    pub fn apply(&mut self, graphics: &GraphicsConfig) {
        graphics.frame_pacing.apply_to(&mut self.frame_pacing);
        graphics.shadows.apply_to(&mut self.shadows);
        graphics
            .texture_filtering
            .apply_to(&mut self.texture_filtering);
        graphics.background.apply_to(&mut self.background);
        graphics
            .ambient_occlusion
            .apply_to(&mut self.ambient_occlusion);
        graphics.reflections.apply_to(&mut self.reflections);
        graphics
            .indirect_lighting
            .apply_to(&mut self.indirect_lighting);
        graphics.anti_aliasing.apply_to(&mut self.anti_aliasing);
        graphics.exposure.apply_to(&mut self.exposure);
        graphics.bloom.apply_to(&mut self.bloom);
        graphics.vignette.apply_to(&mut self.vignette);
        graphics.display.apply_to(&mut self.display);
        graphics.culling.apply_to(&mut self.culling);
        graphics.planet_detail.apply_to(&mut self.planet_detail);
    }

    /// Take the settings the quality preset governs from `other`: the five
    /// effect sections, the render scale, the shadow map and schedule, the
    /// texture filtering and the planet terrain detail.
    pub fn adopt_quality(&mut self, other: &Self) {
        self.ambient_occlusion = other.ambient_occlusion.clone();
        self.reflections = other.reflections.clone();
        self.indirect_lighting = other.indirect_lighting.clone();
        self.anti_aliasing.mode = other.anti_aliasing.mode;
        self.anti_aliasing.upscale_quality = other.anti_aliasing.upscale_quality;
        self.exposure = other.exposure.clone();
        self.shadows = other.shadows.clone();
        self.texture_filtering = other.texture_filtering.clone();
        self.planet_detail = other.planet_detail.clone();
    }

    /// Frames in flight, at least one.
    pub fn frames_in_flight(&self) -> usize {
        (self.frame_pacing.frames_in_flight as usize).max(1)
    }

    /// The cascade schedule, read each frame.
    pub fn shadow_cadence(&self) -> ShadowCadence {
        ShadowCadence {
            update: self.shadows.update,
            distance: self.shadows.distance,
            cascades: self.shadows.cascades,
        }
    }

    /// The shadow map resolution and schedule.
    pub fn shadow_params(&self) -> ShadowParams {
        ShadowParams {
            map_size: self.shadows.map_size,
            cadence: self.shadow_cadence(),
        }
    }

    /// The ambient (IBL) scale, which rides the light uniforms.
    pub fn ambient_intensity(&self) -> f32 {
        self.indirect_lighting.resolved_ambient()
    }

    /// The live composite tunables; `fxaa` follows the AA mode.
    pub fn post_process(&self) -> PostProcessTunables {
        post_process_tunables(
            &self.bloom,
            &self.exposure,
            &self.vignette,
            self.lut_strength,
            self.anti_aliasing.mode,
        )
    }

    /// Whether the temporal anti-aliasing pass runs.
    pub fn taa(&self) -> bool {
        self.anti_aliasing.mode.taa_enabled()
    }

    /// The requested HDR sample count: one under any temporal mode, else the
    /// multisampled count.
    pub fn hdr_samples(&self) -> u32 {
        hdr_sample_count(self.anti_aliasing.mode, self.anti_aliasing.upscaling)
    }

    /// The upscaler's per-axis input-to-output ratio.
    pub fn upscale_scale(&self) -> f32 {
        self.anti_aliasing.upscale_quality.scale()
    }

    /// Screen-space ambient occlusion, or `None` when off or inert.
    pub fn ssao(&self) -> Option<SsaoSettings> {
        SsaoSettings::from_config(&self.ambient_occlusion)
    }

    /// Screen-space reflections, or `None` when off.
    pub fn ssr(&self) -> Option<SsrSettings> {
        SsrSettings::from_config(&self.reflections)
    }

    /// Ray-traced reflections, or `None` when off.
    pub fn rt_reflections(&self) -> Option<RtReflectionSettings> {
        RtReflectionSettings::from_config(&self.reflections)
    }

    /// Screen-space global illumination, or `None` when off or inert.
    pub fn ssgi(&self) -> Option<SsgiSettings> {
        SsgiSettings::from_config(&self.indirect_lighting)
    }

    /// Per-axis divisor for the roughness-aware reflection blur target.
    pub fn reflection_blur_scale(&self) -> u32 {
        self.reflections.blur_divisor()
    }

    /// Auto-exposure, metered for the display output, or `None` when off.
    pub fn auto_exposure(&self) -> Option<AutoExposureSettings> {
        self.exposure.auto_settings(self.display.hdr)
    }

    /// The exposure offset auto-exposure adds to the adapted EV.
    pub fn auto_exposure_bias_ev(&self) -> f32 {
        self.exposure.ev
    }

    /// Turn off every feature that only decorates a 3D scene, for a world that
    /// draws none. TAA gives way to FXAA, so the composite filters edges as it
    /// did; the HDR sample count is the caller's to keep from before the trim.
    pub fn trim_scene_features(&mut self) {
        self.shadows.map_size = 0;
        self.ambient_occlusion.enabled = false;
        self.reflections.screen_space = false;
        self.reflections.ray_traced = false;
        self.indirect_lighting.technique = IndirectTechnique::Ibl;
        self.exposure.auto = false;
        self.culling.two_pass_occlusion = false;
        self.anti_aliasing.upscaling = false;
        if self.anti_aliasing.mode == AaMode::Taa {
            self.anti_aliasing.mode = AaMode::Fxaa;
        }
    }
}
