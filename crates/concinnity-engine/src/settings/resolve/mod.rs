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
//!    fields of the result, set or not, and only ever reduces. The preset
//!    alone may turn temporal upscaling on, where no layer chose it.
//! 5. The user's persisted choices ([`UserGraphicsSettings`]), each of which
//!    beats the ceiling for its own setting.
//! 6. Session overrides, the same shape as the user's, beating them.
//!
//! Device capabilities clamp after every layer: no setting resolves on that the
//! device cannot run.

mod diff;
mod graphics;
#[cfg(test)]
mod tests;
mod tunables;

pub use diff::SettingsDiff;
pub use graphics::ResolvedGraphics;
pub use tunables::post_process_tunables;

use alloc::borrow::Cow;

use crate::components::{GraphicsConfig, StreamingConfig, UpscalerBackend};
use crate::gfx::render_types::PostProcessTunables;
use crate::render::backend::DeviceCapabilities;
use crate::render::backend::init::ShadowCadence;
use crate::render::output::upscaler::{VendorUpscaler, preferred_upscaler};
use crate::settings::quality::{QualityCeiling, clamp_shadow_update, more_aggressive_upscale};
use crate::settings::rows::{GRAPHICS_SLIDERS, QUALITY_CYCLES, QUALITY_TOGGLES};
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

/// The world's authored graphics values, before any override or ceiling:
/// each section's defaults with the fields the world sets in their place.
#[derive(Clone, Debug, Default)]
pub struct WorldSettings {
    /// The authored sections.
    pub graphics: ResolvedGraphics,
    /// `StreamingConfig.texture_cap`; `None` without a `StreamingConfig`.
    pub texture_cap: Option<u32>,
    /// `StreamingConfig.texture_budget`; `None` without a `StreamingConfig`.
    pub texture_budget: Option<u32>,
    /// `AntiAliasing.upscaling` where the world or a scene sets it; `None`
    /// leaves it to the quality preset.
    pub upscaling: Option<bool>,
}

impl WorldSettings {
    /// The settings a world's declared assets author.
    pub fn new(assets: WorldAssets<'_>) -> Self {
        let mut world = Self {
            graphics: ResolvedGraphics {
                lut_strength: assets.lut_strength.unwrap_or(0.0),
                ..ResolvedGraphics::default()
            },
            texture_cap: assets.streaming.map(|s| s.texture_cap),
            texture_budget: assets.streaming.map(|s| s.texture_budget),
            upscaling: None,
        };
        if let Some(authored) = assets.graphics {
            world.apply(authored);
        }
        world
    }

    /// Replace every field `graphics` sets.
    pub fn apply(&mut self, graphics: &GraphicsConfig) {
        self.graphics.apply(graphics);
        if let Some(upscaling) = graphics.anti_aliasing.upscaling {
            self.upscaling = Some(upscaling);
        }
    }

    /// The cascade schedule the world's `Shadows` authors.
    pub fn shadow_cadence(&self) -> ShadowCadence {
        self.graphics.shadow_cadence()
    }

    /// The ambient (IBL) scale with no user override, the value the static
    /// light uniforms are built with.
    pub fn world_ambient(&self) -> f32 {
        self.graphics.ambient_intensity()
    }

    /// The composite tunables the world authors, with no user override.
    pub fn post_process(&self) -> PostProcessTunables {
        self.graphics.post_process()
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
    /// Whether the quality preset upscales where neither the world nor the
    /// user chose.
    pub auto_upscaling: bool,
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
            auto_upscaling: false,
            user,
            session: None,
            capabilities: &DeviceCapabilities::ALL,
        }
    }

    /// Every setting, resolved through the layers.
    pub fn resolve(&self) -> ResolvedSettings {
        let world = self.world_layer();
        let user = self.user_layer();
        let mut resolved = resolve_settings(&world, &user, self.ceiling, self.auto_upscaling);
        clamp_to(&mut resolved.graphics, *self.capabilities);
        resolved
    }

    /// The settings the quality preset governs, resolved through the layers.
    /// Only those fields are meaningful; [`ResolvedSettings::set_quality`]
    /// takes them.
    pub fn resolve_quality(&self) -> ResolvedGraphics {
        let mut quality = resolve_quality(&self.world_layer(), &self.user_layer(), self.ceiling);
        clamp_to(&mut quality, *self.capabilities);
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

// No setting resolves on that the device cannot run: ray tracing on a device
// without it, or beside XeSS where the two cannot share the device, and an
// upscaler choice where there is none to make.
fn clamp_to(graphics: &mut ResolvedGraphics, capabilities: DeviceCapabilities) {
    if !capabilities.selectable_upscaler {
        graphics.anti_aliasing.upscaler = UpscalerBackend::Auto;
    }
    let aa = &graphics.anti_aliasing;
    let xess = aa.upscaling
        && preferred_upscaler(aa.upscaler, capabilities.upscalers) == Some(VendorUpscaler::Xess);
    if !capabilities.ray_tracing || (xess && !capabilities.ray_tracing_with_xess) {
        graphics.reflections.ray_traced = false;
    }
}

/// Every value the settings menu shows: the resolved graphics the renderer is
/// built with, and the menu's own values that never reach it.
#[derive(Clone, Debug)]
pub struct ResolvedSettings {
    /// The resolved graphics settings.
    pub graphics: ResolvedGraphics,
    /// The performance-stats overlay.
    pub perf_stats: bool,
    /// The frame-rate readout.
    pub show_fps: bool,
    /// The GPU-memory readout.
    pub show_vram: bool,
    /// The texture-streaming pool cap.
    pub texture_cap: u32,
    /// The texture-streaming per-frame upload budget.
    pub texture_budget: u32,
}

impl ResolvedSettings {
    /// Adopt a re-resolved quality group.
    pub fn set_quality(&mut self, quality: &ResolvedGraphics) {
        self.graphics.adopt_quality(quality);
    }

    /// What changed since `previous`, grouped by when each change applies.
    pub fn diff(&self, previous: &Self) -> SettingsDiff {
        SettingsDiff::between(previous, self)
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

// Each value is the user's choice where they made one, otherwise the world's
// (the section default where the world sets none), clamped under the ceiling
// if the preset governs it. Upscaling unset by both follows `auto_upscaling`.
fn resolve_settings(
    world: &WorldSettings,
    user: &UserGraphicsSettings,
    ceiling: &QualityCeiling,
    auto_upscaling: bool,
) -> ResolvedSettings {
    let streaming = StreamingConfig::default();
    let mut graphics = resolve_quality(world, user, ceiling);
    let pacing = &mut graphics.frame_pacing;
    pacing.vsync = user.vsync.unwrap_or(pacing.vsync);
    pacing.fps_cap = user.fps_cap.unwrap_or(pacing.fps_cap);
    if let Some(frames) = user.frames_in_flight {
        pacing.frames_in_flight = frames.clamp(1, 3);
    }
    let display = &mut graphics.display;
    display.hdr = user.hdr_display.unwrap_or(display.hdr);
    display.pq = user.hdr_pq.unwrap_or(display.pq);
    let aa = &mut graphics.anti_aliasing;
    aa.upscaling = user
        .temporal_upscaling
        .or(world.upscaling)
        .unwrap_or(auto_upscaling);
    aa.upscaler = user.upscale_backend.unwrap_or(aa.upscaler);
    let culling = &mut graphics.culling;
    culling.two_pass_occlusion = user
        .occlusion_two_pass
        .unwrap_or(culling.two_pass_occlusion);
    graphics.dlss_preset = user.dlss_preset.unwrap_or_default();
    ResolvedSettings {
        graphics,
        perf_stats: user.perf_stats.unwrap_or(true),
        show_fps: user.show_fps.unwrap_or(true),
        show_vram: user.show_vram.unwrap_or(true),
        // The user's streaming choices only apply where the world streams.
        texture_cap: world
            .texture_cap
            .map_or(streaming.texture_cap, |v| user.texture_cap.unwrap_or(v)),
        texture_budget: world.texture_budget.map_or(streaming.texture_budget, |v| {
            user.texture_budget.unwrap_or(v)
        }),
    }
}

// The world's sections with the user's quality choices and sliders overlaid,
// the preset-governed fields clamped under the ceiling. The fields outside the
// quality group keep the world's values.
fn resolve_quality(
    world: &WorldSettings,
    user: &UserGraphicsSettings,
    ceiling: &QualityCeiling,
) -> ResolvedGraphics {
    let authored = &world.graphics;
    let mut g = authored.clone();
    overlay_quality_overrides(&mut g, user);
    clamp_quality_under_ceiling(&mut g, user, ceiling);
    g.anti_aliasing.upscale_quality = user.render_scale.unwrap_or_else(|| {
        more_aggressive_upscale(authored.anti_aliasing.upscale_quality, ceiling.min_upscale)
    });
    let shadows = &mut g.shadows;
    shadows.map_size = user
        .shadow_map_size
        .unwrap_or_else(|| authored.shadows.map_size.min(ceiling.shadow_map_size));
    shadows.update = user
        .shadow_update
        .unwrap_or_else(|| clamp_shadow_update(authored.shadows.update, ceiling));
    shadows.distance = user
        .shadow_distance
        .unwrap_or_else(|| authored.shadows.distance.min(ceiling.shadow_distance));
    shadows.cascades = user
        .shadow_cascades
        .unwrap_or_else(|| authored.shadows.cascades.min(ceiling.shadow_cascades));
    g.texture_filtering.anisotropy = user.anisotropy.unwrap_or_else(|| {
        authored
            .texture_filtering
            .anisotropy
            .min(ceiling.anisotropy)
    });
    g.planet_detail.finest_cell_size = authored
        .planet_detail
        .finest_cell_size
        .max(ceiling.min_planet_cell_size);
    g.water_trace = ceiling.water_trace();
    g
}

/// Overlay the user's sliders (look tuning, not preset-governed) onto settings
/// already carrying the world's values.
fn overlay_sliders(graphics: &mut ResolvedGraphics, user: &UserGraphicsSettings) {
    for s in &GRAPHICS_SLIDERS {
        if let Some(v) = *(s.persisted.get)(user) {
            *(s.field.get_mut)(graphics) = (s.apply)(v);
        }
    }
}

/// Overlay every quality choice the user made onto settings already carrying
/// the world's values: the feature toggles, the cycle knobs, and the sliders.
/// The schema defaults a world falls back to without an asset are a real
/// authored look, so they take the overlay too.
pub fn overlay_quality_overrides(graphics: &mut ResolvedGraphics, user: &UserGraphicsSettings) {
    for row in &QUALITY_TOGGLES {
        if let Some(on) = *(row.persisted.get)(user) {
            (row.set)(graphics, on);
        }
    }
    for row in &QUALITY_CYCLES {
        (row.overlay)(graphics, user);
    }
    overlay_sliders(graphics, user);
}

/// Clamp the preset-governed settings under the ceiling: a feature the tier
/// forbids turns off and a cycle knob lowers, except where the user chose that
/// row. Only ever reduces, so a config within the ceiling passes untouched.
fn clamp_quality_under_ceiling(
    graphics: &mut ResolvedGraphics,
    user: &UserGraphicsSettings,
    ceiling: &QualityCeiling,
) {
    for row in &QUALITY_TOGGLES {
        if (row.persisted.get)(user).is_none() && !(row.allowed)(ceiling) {
            (row.set)(graphics, false);
        }
    }
    for row in &QUALITY_CYCLES {
        if !(row.overridden)(user) {
            (row.clamp)(graphics, ceiling);
        }
    }
}
