//! Settings resolution: one function from ordered layers to the
//! [`ResolvedSettings`] snapshot the renderer is built with, and the
//! [`SettingsDiff`] between two snapshots, grouped by when each change takes
//! effect.
//!
//! The layers, lowest first:
//!
//! 1. The world's authored settings ([`WorldSettings`]), defaults where absent.
//! 2. A scene's override ([`SceneOverride`]), replacing the world's assets.
//! 3. The quality ceiling ([`QualityCeiling`]), which only ever reduces.
//! 4. The user's persisted choices ([`UserGraphicsSettings`]), each of which
//!    beats the ceiling for its own setting.
//! 5. Session overrides, the same shape as the user's, beating them.
//!
//! Device capabilities clamp after every layer: no setting resolves on that the
//! device cannot run.

mod diff;
#[cfg(test)]
mod tests;

pub use diff::SettingsDiff;

use alloc::borrow::Cow;

use crate::components::{
    GraphicsConfig, PostProcessConfig, StreamingConfig, UpscaleQuality, UpscalerBackend,
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

/// The world's authored graphics values, before any override or ceiling.
#[derive(Clone, Debug)]
pub struct WorldSettings {
    /// `GraphicsConfig.shadow_map_size`.
    pub shadow_map_size: u32,
    /// `GraphicsConfig`'s cascade schedule.
    pub shadow_cadence: ShadowCadence,
    /// `GraphicsConfig.anisotropy`.
    pub anisotropy: u32,
    /// `GraphicsConfig.frames_in_flight`.
    pub frames_in_flight: u32,
    /// `GraphicsConfig.vsync`.
    pub vsync: bool,
    /// `GraphicsConfig.fps_cap`.
    pub fps_cap: u32,
    /// The world's `PostProcessConfig`, or the schema defaults without one.
    pub post_config: PostProcessConfig,
    /// Whether the world declares a `PostProcessConfig`.
    pub post_declared: bool,
    /// `StreamingConfig.texture_cap`; `None` without a `StreamingConfig`.
    pub texture_cap: Option<u32>,
    /// `StreamingConfig.texture_budget`; `None` without a `StreamingConfig`.
    pub texture_budget: Option<u32>,
}

impl WorldSettings {
    /// The settings a world's declared assets author.
    pub fn new(
        graphics: Option<&GraphicsConfig>,
        post: Option<&PostProcessConfig>,
        streaming: Option<&StreamingConfig>,
    ) -> Self {
        let mut world = Self {
            shadow_map_size: 0,
            shadow_cadence: ShadowCadence {
                update: Default::default(),
                distance: 0,
                cascades: 0,
            },
            anisotropy: 0,
            frames_in_flight: 0,
            vsync: false,
            fps_cap: 0,
            post_config: post.cloned().unwrap_or_default(),
            post_declared: post.is_some(),
            texture_cap: streaming.map(|s| s.texture_cap),
            texture_budget: streaming.map(|s| s.texture_budget),
        };
        world.set_graphics(graphics.unwrap_or(&GraphicsConfig::default()));
        world
    }

    /// Take every value a `GraphicsConfig` authors.
    pub fn set_graphics(&mut self, g: &GraphicsConfig) {
        self.shadow_map_size = g.shadow_map_size;
        self.shadow_cadence = ShadowCadence {
            update: g.shadow_update,
            distance: g.shadow_distance,
            cascades: g.shadow_cascades,
        };
        self.anisotropy = g.anisotropy;
        self.frames_in_flight = g.frames_in_flight;
        self.vsync = g.vsync;
        self.fps_cap = g.fps_cap;
    }

    fn declared_post(&self) -> Option<&PostProcessConfig> {
        self.post_declared.then_some(&self.post_config)
    }

    /// The ambient (IBL) scale with no user override, the value the static
    /// light uniforms are built with.
    pub fn world_ambient(&self) -> f32 {
        self.declared_post()
            .map(|c| c.ambient_intensity())
            .unwrap_or(1.0)
    }
}

impl Default for WorldSettings {
    fn default() -> Self {
        Self::new(None, None, None)
    }
}

/// Settings a scene authors over the world's while it is loaded. Each asset
/// present replaces the world's copy whole.
#[derive(Clone, Debug, Default)]
pub struct SceneOverride {
    /// The scene's `GraphicsConfig`.
    pub graphics: Option<GraphicsConfig>,
    /// The scene's `PostProcessConfig`.
    pub post_process: Option<PostProcessConfig>,
}

impl SceneOverride {
    /// The world's settings with this scene's assets in place of the world's.
    pub fn over(&self, world: &WorldSettings) -> WorldSettings {
        let mut out = world.clone();
        if let Some(graphics) = &self.graphics {
            out.set_graphics(graphics);
        }
        if let Some(post) = &self.post_process {
            out.post_config = post.clone();
            out.post_declared = true;
        }
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
    /// The world's config with the user's quality choices overlaid, then
    /// clamped under the ceiling. The source of truth for the quality rows.
    pub post_config: PostProcessConfig,
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
            self.post_config.ray_traced_reflections = false;
        }
    }

    /// The per-feature settings the backend builds its quality passes from.
    pub fn quality_settings(&self) -> QualitySettings {
        quality_settings(&self.post_config)
    }
}

/// Every graphics value the settings menu shows and the renderer is built with.
#[derive(Clone, Debug)]
pub struct ResolvedSettings {
    /// The preset-governed group.
    pub quality: ResolvedQuality,
    /// The live composite tunables; `fxaa` follows `quality.post_config.aa_mode`.
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
        self.post_process.fxaa = quality.post_config.aa_mode.fxaa_flag();
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

/// The per-feature settings a resolved config builds its quality passes from.
pub fn quality_settings(cfg: &PostProcessConfig) -> QualitySettings {
    QualitySettings {
        taa: cfg.aa_mode.taa_enabled(),
        ssao: SsaoSettings::from_config(cfg),
        ssr: SsrSettings::from_config(cfg),
        rt_reflections: RtReflectionSettings::from_config(cfg),
        ssgi: SsgiSettings::from_config(cfg),
        reflection_blur_scale: cfg.reflection_blur_divisor(),
        auto_exposure: cfg.auto_exposure_settings(),
        auto_exposure_bias_ev: cfg.exposure_ev,
    }
}

// Each value is the user's choice where they made one, otherwise the world's,
// clamped under the ceiling if the preset governs it. Settings the world does
// not author fall back to the engine defaults, and occlusion two-pass is off
// without a `PostProcessConfig`.
fn resolve_graphics(
    world: &WorldSettings,
    user: &UserGraphicsSettings,
    ceiling: &QualityCeiling,
) -> ResolvedSettings {
    let post = &world.post_config;
    let streaming = StreamingConfig::default();
    let mut resolved = ResolvedSettings {
        quality: resolve_quality(world, user, ceiling),
        post_process: post_process_params(world.declared_post(), user),
        ambient_intensity: ambient_intensity(world.declared_post(), user),
        vsync: user.vsync.unwrap_or(world.vsync),
        fps_cap: user.fps_cap.unwrap_or(world.fps_cap),
        perf_stats: user.perf_stats.unwrap_or(true),
        show_fps: user.show_fps.unwrap_or(true),
        show_vram: user.show_vram.unwrap_or(true),
        frames_in_flight: user
            .frames_in_flight
            .map_or(world.frames_in_flight as usize, |v| {
                (v as usize).clamp(1, 3)
            }),
        hdr_display: user.hdr_display.unwrap_or(post.hdr_display),
        hdr_pq: user.hdr_pq.unwrap_or(post.hdr_pq),
        temporal_upscaling: user.temporal_upscaling.unwrap_or(post.temporal_upscaling),
        upscale_backend: user.upscale_backend.unwrap_or(post.upscale_backend),
        dlss_preset: user.dlss_preset.unwrap_or_default(),
        occlusion_two_pass: user
            .occlusion_two_pass
            .unwrap_or(world.declared_post().is_some_and(|c| c.occlusion_two_pass)),
        // The user's streaming choices only apply where the world streams.
        texture_cap: world
            .texture_cap
            .map_or(streaming.texture_cap, |v| user.texture_cap.unwrap_or(v)),
        texture_budget: world.texture_budget.map_or(streaming.texture_budget, |v| {
            user.texture_budget.unwrap_or(v)
        }),
    };
    resolved.post_process.fxaa = resolved.quality.post_config.aa_mode.fxaa_flag();
    resolved
}

fn resolve_quality(
    world: &WorldSettings,
    user: &UserGraphicsSettings,
    ceiling: &QualityCeiling,
) -> ResolvedQuality {
    let mut post_config = world.post_config.clone();
    overlay_quality_overrides(&mut post_config, user);
    clamp_quality_under_ceiling(&mut post_config, user, ceiling);
    let authored = world.shadow_cadence;
    ResolvedQuality {
        post_config,
        render_scale: user.render_scale.unwrap_or_else(|| {
            more_aggressive_upscale(world.post_config.upscale_quality, ceiling.min_upscale)
        }),
        shadow_map_size: user
            .shadow_map_size
            .unwrap_or(world.shadow_map_size.min(ceiling.shadow_map_size)),
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
            .unwrap_or(world.anisotropy.min(ceiling.anisotropy)),
    }
}

// The composite tunables: the world's config resolved, then each slider the
// user has moved. `fxaa` is left as `resolve` seeded it, from the authored AA
// mode; the caller refreshes it once the override and ceiling settle the mode.
fn post_process_params(
    config: Option<&PostProcessConfig>,
    user: &UserGraphicsSettings,
) -> PostProcessTunables {
    let mut params = config
        .map(|c| c.resolve())
        .unwrap_or(PostProcessTunables::DEFAULT);
    for s in &GRAPHICS_SLIDERS {
        if let GraphicsSliderTarget::PostProcess { field, persisted } = &s.target
            && let Some(v) = *(persisted.get)(user)
        {
            *(field.get_mut)(&mut params) = (s.apply)(v);
        }
    }
    params
}

fn ambient_intensity(config: Option<&PostProcessConfig>, user: &UserGraphicsSettings) -> f32 {
    let world = config.map(|c| c.ambient_intensity()).unwrap_or(1.0);
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
/// onto a config already carrying the world's values.
pub fn overlay_quality_scalars(cfg: &mut PostProcessConfig, user: &UserGraphicsSettings) {
    for s in &GRAPHICS_SLIDERS {
        if let GraphicsSliderTarget::PostConfig { field, persisted } = &s.target
            && let Some(v) = *(persisted.get)(user)
        {
            *(field.get_mut)(cfg) = (s.apply)(v);
        }
    }
}

/// Overlay every quality choice the user made onto a config already carrying
/// the world's values: the feature toggles, the cycle knobs, and the look
/// sliders. The schema defaults a world without a `PostProcessConfig` falls
/// back to are a real authored look, so they take the overlay too.
pub fn overlay_quality_overrides(cfg: &mut PostProcessConfig, user: &UserGraphicsSettings) {
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
    cfg: &mut PostProcessConfig,
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
