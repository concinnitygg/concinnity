//! The render settings a running world can change.
//!
//! The quality presets the settings menu resolves, the shadow schedule, the
//! post-process block, and the lighting scalars the editor previews live.
//! These all push a value the backend already reads each frame, so none of
//! them rebuilds a pipeline or a target.

use crate::gfx::auto_exposure::AutoExposureSettings;
use crate::gfx::render_types::PostProcessTunables;
use crate::render::backend_init::ShadowCadence;
use crate::render::error::{RenderError, RenderResult};
use crate::render::post::rt_reflections::RtReflectionSettings;
use crate::render::post::ssao::settings::SsaoSettings;
use crate::render::post::ssgi::settings::SsgiSettings;
use crate::render::post::ssr::settings::SsrSettings;
use crate::render::volumetric_fog::FogSettings;

/// The resolved per-feature quality settings for [`RenderTuning::apply_quality_settings`].
/// `GraphicsSystem` derives these from its stored `PostProcessConfig` (with the
/// user's persisted toggle overrides applied) whenever a Quality-group toggle
/// changes, so the backend receives ready-to-use settings rather than re-deriving
/// from the asset. Each `Option` mirrors the init-time gate: `None` means the
/// feature is off and its passes / resources should be torn down; `Some` means it
/// is on and its resources should exist.
pub struct QualitySettings {
    /// Temporal anti-aliasing on/off (the `Taa` anti-aliasing mode). The backend
    /// additionally suppresses TAA while temporal upscaling is active (the scaler
    /// does its own accumulation). The other anti-aliasing modes are the composite
    /// FXAA edge filter, which rides `PostProcessTunables.fxaa` (pushed via
    /// `update_post_process`), not this pass-rebuild payload.
    pub taa: bool,
    /// Screen-space ambient occlusion, or `None` when off.
    pub ssao: Option<SsaoSettings>,
    /// Screen-space reflections, or `None` when off.
    pub ssr: Option<SsrSettings>,
    /// Hardware ray-traced reflections. The backend further gates this on GPU
    /// ray-tracing support, falling back to leaving it off when unsupported.
    pub rt_reflections: Option<RtReflectionSettings>,
    /// Screen-space global illumination, or `None` when off.
    pub ssgi: Option<SsgiSettings>,
    /// Per-axis divisor for the roughness-aware reflection blur target (the
    /// reduced-resolution first pass of the SSR / RT reflection composite),
    /// resolved from `PostProcessConfig.reflection_blur_resolution`. Every backend
    /// sizes its blur target at render / this on a live reflection rebuild.
    pub reflection_blur_scale: u32,
    /// Auto-exposure, or `None` when off.
    pub auto_exposure: Option<AutoExposureSettings>,
    /// The authored exposure bias (stops) auto-exposure applies on top of its
    /// adapted value; carried so a live auto-exposure enable matches init.
    pub auto_exposure_bias_ev: f32,
}

/// Live render settings: quality, shadows, post-process and lighting.
///
/// All defaulted. The infallible setters default to a no-op, so a backend that
/// reads a setting only at init keeps its init-time value. The one fallible
/// method, [`apply_quality_settings`](Self::apply_quality_settings), defaults to
/// [`RenderError::Unsupported`] so the caller learns the rebuild did not happen.
pub trait RenderTuning {
    /// Supply the reflection-probe placements (from declared `ReflectionProbe`
    /// assets, or empty to auto-seed from the scene bounds). The backend bakes a
    /// cube per placement and samples the nearest for the specular reflection.
    /// Pushed once after construction. Default no-op: a backend without probe
    /// support keeps the sky reflection.
    fn set_reflection_probes(
        &mut self,
        probes: &[crate::render::reflection_probe::ProbePlacement],
    ) {
        let _ = probes;
    }

    /// Replace the live post-process tunables (bloom / exposure / vignette /
    /// LUT blend / FXAA). These are pushed to the bloom + composite shaders each
    /// frame, so a change takes effect on the next draw with no allocation or
    /// pipeline rebuild. Only the authored half travels here: the composite's
    /// display-output flags belong to the display the backend negotiated with
    /// at init, so a push cannot disturb them. Default no-op: a backend that
    /// only reads the tunables at init ignores runtime changes.
    fn update_post_process(&mut self, tunables: PostProcessTunables) {
        let _ = tunables;
    }

    /// Set the live ambient (IBL) light scale. Unlike the post-process params
    /// above, `ambient_intensity` lives in the shared `LightUniforms` (uploaded
    /// each frame by the main lighting pass), so it takes its own setter rather
    /// than `update_post_process`. Default no-op: a backend that reads the scale
    /// only at init keeps the init-time value.
    fn set_ambient_intensity(&mut self, value: f32) {
        let _ = value;
    }

    /// Replace the live directional-light set (the sun). Unlike the local
    /// lights, which ride a per-scene storage buffer sized once at init, the
    /// directional slots are a fixed-size array in the shared `LightUniforms`,
    /// so a new set is written in place: the backend re-packs the array and
    /// re-caches whatever it derived from the first light at init (the cascade
    /// shadow direction, the fog sun). Default no-op: a backend that only reads
    /// the lights at init keeps the init-time sun.
    fn update_directional_lights(&mut self, lights: &[crate::components::DirectionalLight]) {
        let _ = lights;
    }

    /// Move the local lights (point, spot and rect area) in place: `lights` is
    /// `lights::build_light_data` over the same lights in the same order, so
    /// the count and every slot assignment are unchanged and only positions
    /// and directions differ. `uniforms` supplies the point array world shader
    /// hooks read; its other fields are ignored. This is how a world that
    /// moves to another simulated frame carries its lights. The spot shadow
    /// slices re-render from their first refresh. Default:
    /// [`RenderError::Unsupported`]; an `Err` also reports a count that does
    /// not match the lights built at init.
    fn move_local_lights(
        &mut self,
        lights: &crate::render::lights::LightData,
        uniforms: &crate::gfx::render_types::LightUniforms,
    ) -> RenderResult<()> {
        let _ = (lights, uniforms);
        Err(RenderError::Unsupported {
            op: "move_local_lights",
        })
    }

    /// Apply a change to the quality-feature toggles (TAA / SSAO / SSR / RT
    /// reflections / SSGI / auto-exposure) live. Unlike the post-process params,
    /// these gate render passes whose GPU resources (pipelines, render targets,
    /// ray-tracing acceleration structures) are built once at init, so applying a
    /// change rebuilds the affected resources in place rather than flipping a
    /// uniform. Default: [`RenderError::Unsupported`], for a backend with no live
    /// rebuild path.
    fn apply_quality_settings(&mut self, settings: QualitySettings) -> RenderResult<()> {
        let _ = settings;
        Err(RenderError::Unsupported {
            op: "apply_quality_settings",
        })
    }

    /// Replace the live cascade-shadow schedule. The shadow pass reads it each
    /// frame, so a change lands on the next draw with no rebuild; the shadow map
    /// resolution is init-only. Default no-op: the init-time cadence stays.
    fn set_shadow_cadence(&mut self, cadence: ShadowCadence) {
        let _ = cadence;
    }

    /// Update the live scalar sub-tunables of the SSAO / SSR / SSGI / auto-exposure
    /// passes (radius, intensity, distance, EV bounds, adaptation speed). Unlike
    /// `apply_quality_settings`, this rebuilds nothing: each backend re-reads these
    /// values from its stored `*Settings` structs into a per-frame uniform every
    /// draw, so mutating them takes effect on the next frame with no pipeline /
    /// target rebuild and no TAA-history reset. Only the fields of a feature that is
    /// currently on are honored (its settings are present); a value for an off
    /// feature is ignored here and applies when the feature next turns on. The
    /// structural sub-knobs (gather resolution, ray / step counts) are NOT live and
    /// still ride `apply_quality_settings`. Default no-op: a backend that reads
    /// these only at init keeps the init-time values, so the choice takes effect
    /// at the next launch there.
    fn update_quality_params(&mut self, settings: QualitySettings) {
        let _ = settings;
    }

    /// Replace the live volumetric-fog settings, or disable the fog pass when
    /// `None`. Default no-op: the fog pass keeps its init-time settings.
    ///
    /// A backend that built its fog pipeline lazily based on the world's
    /// init-time `VolumetricFog` cannot enable the pass via this call when
    /// the world started with no fog declared; re-enabling fog on a world
    /// that did not declare it at startup requires a relaunch.
    fn update_fog_settings(&mut self, settings: Option<FogSettings>) {
        let _ = settings;
    }
}
