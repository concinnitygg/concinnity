//! Settings resolution: one function from ordered layers to the
//! [`ResolvedSettings`] snapshot the renderer is built with, and the
//! [`SettingsDiff`] between two snapshots, grouped by when each change takes
//! effect.
//!
//! The layers, lowest first:
//!
//! 1. Each `GraphicsConfig` section's defaults, the look the top quality tier
//!    runs.
//! 2. The world's `GraphicsConfig` ([`WorldSettings`]): each field it sets
//!    replaces the default.
//! 3. A scene's override ([`SceneOverride`]), the same sparse shape, replacing
//!    the fields it sets.
//! 4. The quality ceiling ([`QualityCeiling`]), which caps the quality-governed
//!    fields of the result, set or not, and only ever reduces.
//! 5. The user's persisted choices ([`UserGraphicsSettings`]), each of which
//!    beats the ceiling for its own setting.
//! 6. Session overrides, the same shape as the user's, beating them.
//!
//! Device capabilities clamp after every layer: no setting resolves on that the
//! device cannot run.

mod diff;
#[cfg(test)]
mod tests;
mod tunables;

pub use diff::SettingsDiff;
pub use tunables::post_process_tunables;

use alloc::borrow::Cow;

use crate::components::{
    AmbientOcclusion, AntiAliasing, Background, Bloom, Culling, Display, Exposure, FramePacing,
    GraphicsConfig, IndirectLighting, Reflections, Shadows, StreamingConfig, TextureFiltering,
    UpscaleQuality, UpscalerBackend, Vignette,
};
use crate::gfx::render_types::PostProcessTunables;
use crate::render::backend::{DeviceCapabilities, QualitySettings};
use crate::render::backend_init::ShadowCadence;
use crate::render::dlss::DlssPreset;
use crate::render::post::rt_reflections::RtReflectionSettings;
use crate::render::post::ssao::settings::SsaoSettings;
use crate::render::post::ssgi::settings::SsgiSettings;
use crate::render::post::ssr::settings::SsrSettings;
use crate::settings::quality::{QualityCeiling, clamp_shadow_update, more_aggressive_upscale};
use crate::settings::rows::{
    GRAPHICS_SLIDERS, GraphicsSliderTarget, QUALITY_CYCLES, QUALITY_TOGGLES,
};
use crate::settings::user::UserGraphicsSettings;

/// The settings assets a world declares, each `None` where it declares none.
#[derive(Clone, Copy, Default)]
pub struct WorldAssets<'a> {
    /// The world's `GraphicsConfig`; `None` sets nothing.
    pub graphics: Option<&'a GraphicsConfig>,
    /// The `strength` of the world's `ColorLut`.
    pub lut_strength: Option<f32>,
    /// The world's `StreamingConfig`.
    pub streaming: Option<&'a StreamingConfig>,
}

/// The settings the quality preset governs: the features its toggles and
/// cycle knobs switch, and the look sliders tuning them.
#[derive(Clone, Debug, Default)]
pub struct QualityEffects {
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
}

/// The world's authored graphics values, before any override or ceiling:
/// each section's defaults with the fields the world sets in their place.
#[derive(Clone, Debug)]
pub struct WorldSettings {
    /// Frame pacing.
    pub frame_pacing: FramePacing,
    /// Cascaded shadows.
    pub shadows: Shadows,
    /// Texture filtering.
    pub texture_filtering: TextureFiltering,
    /// The color shown where nothing draws.
    pub background: Background,
    /// The quality-governed sections.
    pub effects: QualityEffects,
    /// Bloom.
    pub bloom: Bloom,
    /// Vignette.
    pub vignette: Vignette,
    /// Display output.
    pub display: Display,
    /// Culling.
    pub culling: Culling,
    /// The `ColorLut` blend; 1.0 without one.
    pub lut_strength: f32,
    /// `StreamingConfig.texture_cap`; `None` without a `StreamingConfig`.
    pub texture_cap: Option<u32>,
    /// `StreamingConfig.texture_budget`; `None` without a `StreamingConfig`.
    pub texture_budget: Option<u32>,
}

impl WorldSettings {
    /// The settings a world's declared assets author.
    pub fn new(assets: WorldAssets<'_>) -> Self {
        let mut world = Self {
            frame_pacing: FramePacing::default(),
            shadows: Shadows::default(),
            texture_filtering: TextureFiltering::default(),
            background: Background::default(),
            effects: QualityEffects::default(),
            bloom: Bloom::default(),
            vignette: Vignette::default(),
            display: Display::default(),
            culling: Culling::default(),
            lut_strength: assets.lut_strength.unwrap_or(1.0),
            texture_cap: assets.streaming.map(|s| s.texture_cap),
            texture_budget: assets.streaming.map(|s| s.texture_budget),
        };
        if let Some(graphics) = assets.graphics {
            world.apply(graphics);
        }
        world
    }

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
            .apply_to(&mut self.effects.ambient_occlusion);
        graphics.reflections.apply_to(&mut self.effects.reflections);
        graphics
            .indirect_lighting
            .apply_to(&mut self.effects.indirect_lighting);
        graphics
            .anti_aliasing
            .apply_to(&mut self.effects.anti_aliasing);
        graphics.exposure.apply_to(&mut self.effects.exposure);
        graphics.bloom.apply_to(&mut self.bloom);
        graphics.vignette.apply_to(&mut self.vignette);
        graphics.display.apply_to(&mut self.display);
        graphics.culling.apply_to(&mut self.culling);
    }

    /// The cascade schedule the world's `Shadows` authors.
    pub fn shadow_cadence(&self) -> ShadowCadence {
        ShadowCadence {
            update: self.shadows.update,
            distance: self.shadows.distance,
            cascades: self.shadows.cascades,
        }
    }

    /// The ambient (IBL) scale with no user override, the value the static
    /// light uniforms are built with.
    pub fn world_ambient(&self) -> f32 {
        self.effects.indirect_lighting.resolved_ambient()
    }

    /// The composite tunables the world authors, with no user override.
    pub fn post_process(&self) -> PostProcessTunables {
        post_process_tunables(
            &self.bloom,
            &self.effects.exposure,
            &self.vignette,
            self.lut_strength,
            self.effects.anti_aliasing.mode,
        )
    }
}

impl Default for WorldSettings {
    fn default() -> Self {
        Self::new(WorldAssets::default())
    }
}

/// Settings a scene authors over the world's while it is loaded: each field
/// its `GraphicsConfig` sets replaces the world's.
#[derive(Clone, Debug, Default)]
pub struct SceneOverride {
    /// The scene's sparse graphics settings.
    pub graphics: GraphicsConfig,
}

impl SceneOverride {
    /// The world's settings with this scene's set fields in place of the
    /// world's.
    pub fn over(&self, world: &WorldSettings) -> WorldSettings {
        let mut out = world.clone();
        out.apply(&self.graphics);
        out
    }
}

/// Every layer one resolution reads.
#[derive(Clone, Copy)]
pub struct SettingsLayers<'a> {
    /// The world's authored settings.
    pub world: &'a WorldSettings,
    /// The loaded scene's override, if it authors one.
    pub scene: Option<&'a SceneOverride>,
    /// The active quality ceiling.
    pub ceiling: &'a QualityCeiling,
    /// The user's persisted choices.
    pub user: &'a UserGraphicsSettings,
    /// This session's overrides.
    pub session: Option<&'a UserGraphicsSettings>,
    /// What the device can run.
    pub capabilities: &'a DeviceCapabilities,
}

impl<'a> SettingsLayers<'a> {
    /// The world, the ceiling and the user's choices, with no scene or session
    /// override, on a device that can run everything.
    pub fn new(
        world: &'a WorldSettings,
        ceiling: &'a QualityCeiling,
        user: &'a UserGraphicsSettings,
    ) -> Self {
        Self {
            world,
            scene: None,
            ceiling,
            user,
            session: None,
            capabilities: &DeviceCapabilities::ALL,
        }
    }

    /// Every setting, resolved through the layers.
    pub fn resolve(&self) -> ResolvedSettings {
        let world = self.world_layer();
        let user = self.user_layer();
        let mut resolved = resolve_graphics(&world, &user, self.ceiling);
        resolved.clamp_to(self.capabilities);
        resolved
    }

    /// The settings the quality preset governs, resolved through the layers.
    pub fn resolve_quality(&self) -> ResolvedQuality {
        let mut quality = resolve_quality(&self.world_layer(), &self.user_layer(), self.ceiling);
        quality.clamp_to(self.capabilities);
        quality
    }

    fn world_layer(&self) -> Cow<'a, WorldSettings> {
        match self.scene {
            Some(scene) => Cow::Owned(scene.over(self.world)),
            None => Cow::Borrowed(self.world),
        }
    }

    fn user_layer(&self) -> Cow<'a, UserGraphicsSettings> {
        match self.session {
            Some(session) => Cow::Owned(self.user.overlaid_by(session)),
            None => Cow::Borrowed(self.user),
        }
    }
}

/// The settings the quality preset governs, re-resolved together when it
/// changes.
#[derive(Clone, Debug)]
pub struct ResolvedQuality {
    /// The world's quality-governed assets with the user's quality choices
    /// overlaid, then clamped under the ceiling. The source of truth for the
    /// quality rows.
    pub effects: QualityEffects,
    /// The world's `Display.hdr`, which pivots auto-exposure metering toward
    /// middle gray.
    pub hdr_metering: bool,
    /// The upscaler's render scale. Sized at launch.
    pub render_scale: UpscaleQuality,
    /// Shadow map resolution. Sized at launch.
    pub shadow_map_size: u32,
    /// The cascade schedule, read each frame.
    pub shadow_cadence: ShadowCadence,
    /// Scene-sampler anisotropy. Built at launch.
    pub anisotropy: u32,
}

impl ResolvedQuality {
    fn clamp_to(&mut self, capabilities: &DeviceCapabilities) {
        if !capabilities.ray_tracing {
            self.effects.reflections.ray_traced = false;
        }
    }

    /// The per-feature settings the backend builds its quality passes from.
    pub fn quality_settings(&self) -> QualitySettings {
        quality_settings(&self.effects, self.hdr_metering)
    }
}

/// Every graphics value the settings menu shows and the renderer is built with.
#[derive(Clone, Debug)]
pub struct ResolvedSettings {
    /// The preset-governed group.
    pub quality: ResolvedQuality,
    /// The live composite tunables; `fxaa` follows the resolved AA mode.
    pub post_process: PostProcessTunables,
    /// The ambient (IBL) scale, which rides the light uniforms.
    pub ambient_intensity: f32,
    /// Display sync.
    pub vsync: bool,
    /// The frame-rate cap; 0 is uncapped.
    pub fps_cap: u32,
    /// The performance-stats overlay.
    pub perf_stats: bool,
    /// The frame-rate readout.
    pub show_fps: bool,
    /// The GPU-memory readout.
    pub show_vram: bool,
    /// Frames in flight.
    pub frames_in_flight: usize,
    /// HDR display output.
    pub hdr_display: bool,
    /// PQ (HDR10) output.
    pub hdr_pq: bool,
    /// Temporal upscaling.
    pub temporal_upscaling: bool,
    /// The upscaler backend.
    pub upscale_backend: UpscalerBackend,
    /// The DLSS render preset.
    pub dlss_preset: DlssPreset,
    /// Two-pass occlusion culling.
    pub occlusion_two_pass: bool,
    /// The texture-streaming pool cap.
    pub texture_cap: u32,
    /// The texture-streaming per-frame upload budget.
    pub texture_budget: u32,
}

impl ResolvedSettings {
    /// Adopt a re-resolved quality group, refreshing the composite FXAA flag
    /// from its AA mode.
    pub fn set_quality(&mut self, quality: ResolvedQuality) {
        self.post_process.fxaa = quality.effects.anti_aliasing.mode.fxaa_flag();
        self.quality = quality;
    }

    /// The per-feature settings the backend builds its quality passes from.
    pub fn quality_settings(&self) -> QualitySettings {
        self.quality.quality_settings()
    }

    /// What changed since `previous`, grouped by when each change applies.
    pub fn diff(&self, previous: &Self) -> SettingsDiff {
        SettingsDiff::between(previous, self)
    }

    fn clamp_to(&mut self, capabilities: &DeviceCapabilities) {
        self.quality.clamp_to(capabilities);
        if !capabilities.selectable_upscaler {
            self.upscale_backend = UpscalerBackend::Auto;
        }
    }
}

/// An unconfigured world with no user overrides and no ceiling.
impl Default for ResolvedSettings {
    fn default() -> Self {
        SettingsLayers::new(
            &WorldSettings::default(),
            &QualityCeiling::NONE,
            &UserGraphicsSettings::default(),
        )
        .resolve()
    }
}

/// The per-feature settings resolved effects build their quality passes from.
/// `hdr_metering` is the world's `Display.hdr`.
pub fn quality_settings(effects: &QualityEffects, hdr_metering: bool) -> QualitySettings {
    QualitySettings {
        taa: effects.anti_aliasing.mode.taa_enabled(),
        ssao: SsaoSettings::from_config(&effects.ambient_occlusion),
        ssr: SsrSettings::from_config(&effects.reflections),
        rt_reflections: RtReflectionSettings::from_config(&effects.reflections),
        ssgi: SsgiSettings::from_config(&effects.indirect_lighting),
        reflection_blur_scale: effects.reflections.blur_divisor(),
        auto_exposure: effects.exposure.auto_settings(hdr_metering),
        auto_exposure_bias_ev: effects.exposure.ev,
    }
}

// Each value is the user's choice where they made one, otherwise the world's
// (the section default where the world sets none), clamped under the ceiling
// if the preset governs it.
fn resolve_graphics(
    world: &WorldSettings,
    user: &UserGraphicsSettings,
    ceiling: &QualityCeiling,
) -> ResolvedSettings {
    let streaming = StreamingConfig::default();
    let aa = &world.effects.anti_aliasing;
    let mut resolved = ResolvedSettings {
        quality: resolve_quality(world, user, ceiling),
        post_process: post_process_params(world, user),
        ambient_intensity: ambient_intensity(world, user),
        vsync: user.vsync.unwrap_or(world.frame_pacing.vsync),
        fps_cap: user.fps_cap.unwrap_or(world.frame_pacing.fps_cap),
        perf_stats: user.perf_stats.unwrap_or(true),
        show_fps: user.show_fps.unwrap_or(true),
        show_vram: user.show_vram.unwrap_or(true),
        frames_in_flight: user
            .frames_in_flight
            .map_or(world.frame_pacing.frames_in_flight as usize, |v| {
                (v as usize).clamp(1, 3)
            }),
        hdr_display: user.hdr_display.unwrap_or(world.display.hdr),
        hdr_pq: user.hdr_pq.unwrap_or(world.display.pq),
        temporal_upscaling: user.temporal_upscaling.unwrap_or(aa.upscaling),
        upscale_backend: user.upscale_backend.unwrap_or(aa.upscaler),
        dlss_preset: user.dlss_preset.unwrap_or_default(),
        occlusion_two_pass: user
            .occlusion_two_pass
            .unwrap_or(world.culling.two_pass_occlusion),
        // The user's streaming choices only apply where the world streams.
        texture_cap: world
            .texture_cap
            .map_or(streaming.texture_cap, |v| user.texture_cap.unwrap_or(v)),
        texture_budget: world.texture_budget.map_or(streaming.texture_budget, |v| {
            user.texture_budget.unwrap_or(v)
        }),
    };
    resolved.post_process.fxaa = resolved.quality.effects.anti_aliasing.mode.fxaa_flag();
    resolved
}

fn resolve_quality(
    world: &WorldSettings,
    user: &UserGraphicsSettings,
    ceiling: &QualityCeiling,
) -> ResolvedQuality {
    let mut effects = world.effects.clone();
    overlay_quality_overrides(&mut effects, user);
    clamp_quality_under_ceiling(&mut effects, user, ceiling);
    let authored = world.shadow_cadence();
    ResolvedQuality {
        effects,
        hdr_metering: world.display.hdr,
        render_scale: user.render_scale.unwrap_or_else(|| {
            more_aggressive_upscale(
                world.effects.anti_aliasing.upscale_quality,
                ceiling.min_upscale,
            )
        }),
        shadow_map_size: user
            .shadow_map_size
            .unwrap_or(world.shadows.map_size.min(ceiling.shadow_map_size)),
        shadow_cadence: ShadowCadence {
            update: user
                .shadow_update
                .unwrap_or_else(|| clamp_shadow_update(authored.update, ceiling)),
            distance: user
                .shadow_distance
                .unwrap_or(authored.distance.min(ceiling.shadow_distance)),
            cascades: user
                .shadow_cascades
                .unwrap_or(authored.cascades.min(ceiling.shadow_cascades)),
        },
        anisotropy: user
            .anisotropy
            .unwrap_or(world.texture_filtering.anisotropy.min(ceiling.anisotropy)),
    }
}

// The composite tunables: the world's assets resolved, then each slider the
// user has moved. `fxaa` is left seeded from the authored AA mode; the caller
// refreshes it once the override and ceiling settle the mode.
fn post_process_params(world: &WorldSettings, user: &UserGraphicsSettings) -> PostProcessTunables {
    let mut params = world.post_process();
    for s in &GRAPHICS_SLIDERS {
        if let GraphicsSliderTarget::PostProcess { field, persisted } = &s.target
            && let Some(v) = *(persisted.get)(user)
        {
            *(field.get_mut)(&mut params) = (s.apply)(v);
        }
    }
    params
}

fn ambient_intensity(world: &WorldSettings, user: &UserGraphicsSettings) -> f32 {
    let world = world.world_ambient();
    GRAPHICS_SLIDERS
        .iter()
        .find_map(|s| match &s.target {
            GraphicsSliderTarget::Ambient { persisted } => {
                Some((s.apply)((persisted.get)(user).unwrap_or(world)))
            }
            _ => None,
        })
        .unwrap_or(world)
}

/// Overlay the user's sub-quality sliders (look tuning, not preset-governed)
/// onto effects already carrying the world's values.
pub fn overlay_quality_scalars(cfg: &mut QualityEffects, user: &UserGraphicsSettings) {
    for s in &GRAPHICS_SLIDERS {
        if let GraphicsSliderTarget::Effect { field, persisted } = &s.target
            && let Some(v) = *(persisted.get)(user)
        {
            *(field.get_mut)(cfg) = (s.apply)(v);
        }
    }
}

/// Overlay every quality choice the user made onto effects already carrying
/// the world's values: the feature toggles, the cycle knobs, and the look
/// sliders. The schema defaults a world falls back to without an asset are a
/// real authored look, so they take the overlay too.
pub fn overlay_quality_overrides(cfg: &mut QualityEffects, user: &UserGraphicsSettings) {
    for row in &QUALITY_TOGGLES {
        if let Some(on) = *(row.persisted.get)(user) {
            (row.set)(cfg, on);
        }
    }
    for row in &QUALITY_CYCLES {
        (row.overlay)(cfg, user);
    }
    overlay_quality_scalars(cfg, user);
}

/// Clamp the preset-governed settings under the ceiling: a feature the tier
/// forbids turns off and a cycle knob lowers, except where the user chose that
/// row. Only ever reduces, so a config within the ceiling passes untouched.
pub fn clamp_quality_under_ceiling(
    cfg: &mut QualityEffects,
    user: &UserGraphicsSettings,
    ceiling: &QualityCeiling,
) {
    for row in &QUALITY_TOGGLES {
        if (row.persisted.get)(user).is_none() && !(row.allowed)(ceiling) {
            (row.set)(cfg, false);
        }
    }
    for row in &QUALITY_CYCLES {
        if !(row.overridden)(user) {
            (row.clamp)(cfg, ceiling);
        }
    }
}
