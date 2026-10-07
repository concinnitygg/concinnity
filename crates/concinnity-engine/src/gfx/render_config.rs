//! One expression per render setting the world authors and the settings menu can
//! override. A setting resolves to the user's persisted choice where they made
//! one, otherwise to the world's value under the active quality preset's
//! ceiling. Init resolves every setting through [`resolve_graphics`] at launch,
//! a preset change re-resolves the [`QualityGraphics`] group, and the
//! live-lighting seam re-resolves the ones an authoring edit can reach, so a
//! change applied to a running world shows what relaunching that world would.

use concinnity_core::components::{
    GraphicsConfig, PostProcessConfig, StreamingConfig, UpscaleQuality, UpscalerBackend,
};
use concinnity_core::gfx::render_types::PostProcessTunables;
use concinnity_core::render::backend_init::ShadowCadence;
use concinnity_core::render::dlss::DlssPreset;

use crate::config::GraphicsSettings;
use crate::gfx::quality_preset::{QualityCeiling, clamp_shadow_update, more_aggressive_upscale};
use crate::settings::quality_rows::{QUALITY_CYCLES, QUALITY_TOGGLES};
use crate::settings::{SLIDERS, SliderTarget};

/// The world's authored graphics values, before any user override or ceiling.
/// A preset change and an authoring edit re-resolve from it.
#[derive(Clone, Debug)]
pub(crate) struct GraphicsBaseline {
    pub(crate) shadow_map_size: u32,
    pub(crate) shadow_cadence: ShadowCadence,
    pub(crate) anisotropy: u32,
    pub(crate) frames_in_flight: u32,
    pub(crate) vsync: bool,
    pub(crate) fps_cap: u32,
    /// The world's `PostProcessConfig`, or the schema defaults without one.
    pub(crate) post_config: PostProcessConfig,
    pub(crate) post_declared: bool,
    /// The world's streaming caps; `None` without a `StreamingConfig`.
    pub(crate) texture_cap: Option<u32>,
    pub(crate) texture_budget: Option<u32>,
}

impl GraphicsBaseline {
    pub(crate) fn new(
        graphics: Option<&GraphicsConfig>,
        post: Option<&PostProcessConfig>,
        streaming: Option<&StreamingConfig>,
    ) -> Self {
        let default_graphics = GraphicsConfig::default();
        let g = graphics.unwrap_or(&default_graphics);
        Self {
            shadow_map_size: g.shadow_map_size,
            shadow_cadence: ShadowCadence {
                update: g.shadow_update,
                distance: g.shadow_distance,
                cascades: g.shadow_cascades,
            },
            anisotropy: g.anisotropy,
            frames_in_flight: g.frames_in_flight,
            vsync: g.vsync,
            fps_cap: g.fps_cap,
            post_config: post.cloned().unwrap_or_default(),
            post_declared: post.is_some(),
            texture_cap: streaming.map(|s| s.texture_cap),
            texture_budget: streaming.map(|s| s.texture_budget),
        }
    }

    fn declared_post(&self) -> Option<&PostProcessConfig> {
        self.post_declared.then_some(&self.post_config)
    }

    /// The ambient (IBL) scale with no user override, the value the static
    /// light uniforms are built with.
    pub(crate) fn world_ambient(&self) -> f32 {
        self.declared_post()
            .map(|c| c.ambient_intensity())
            .unwrap_or(1.0)
    }
}

impl Default for GraphicsBaseline {
    fn default() -> Self {
        Self::new(None, None, None)
    }
}

/// The settings the quality preset governs, re-derived together when it
/// changes.
#[derive(Clone, Debug)]
pub(crate) struct QualityGraphics {
    /// The world's config with the user's Quality-group choices overlaid, then
    /// clamped under the ceiling. The source of truth for the Quality rows.
    pub(crate) post_config: PostProcessConfig,
    /// Restart-required: the upscaler and render targets are sized at init.
    pub(crate) render_scale: UpscaleQuality,
    pub(crate) shadow_map_size: u32,
    pub(crate) shadow_cadence: ShadowCadence,
    pub(crate) anisotropy: u32,
}

/// Every graphics value the settings menu shows and the backend is built with.
#[derive(Clone, Debug)]
pub(crate) struct ResolvedGraphics {
    pub(crate) quality: QualityGraphics,
    /// Live composite params; `fxaa` follows `quality.post_config.aa_mode`.
    pub(crate) post_process: PostProcessTunables,
    /// Rides `LightUniforms` rather than `PostProcessParams`.
    pub(crate) ambient_intensity: f32,
    pub(crate) vsync: bool,
    pub(crate) fps_cap: u32,
    pub(crate) perf_stats: bool,
    pub(crate) show_fps: bool,
    pub(crate) show_vram: bool,
    pub(crate) frames_in_flight: usize,
    pub(crate) hdr_display: bool,
    pub(crate) hdr_pq: bool,
    pub(crate) temporal_upscaling: bool,
    pub(crate) upscale_backend: UpscalerBackend,
    pub(crate) dlss_preset: DlssPreset,
    pub(crate) occlusion_two_pass: bool,
    pub(crate) texture_cap: u32,
    pub(crate) texture_budget: u32,
}

impl ResolvedGraphics {
    /// Adopt a re-derived quality group, refreshing the composite FXAA flag
    /// from its AA mode.
    pub(crate) fn set_quality(&mut self, quality: QualityGraphics) {
        self.post_process.fxaa = quality.post_config.aa_mode.fxaa_flag();
        self.quality = quality;
    }
}

/// An unconfigured world with no user overrides and no ceiling.
impl Default for ResolvedGraphics {
    fn default() -> Self {
        let no_ceiling = crate::gfx::quality_preset::resolve_ceiling(
            crate::gfx::quality_preset::QualityPreset::Custom,
            &concinnity_core::render::backend::GpuProfile::UNKNOWN,
        );
        resolve_graphics(
            &GraphicsBaseline::default(),
            &GraphicsSettings::default(),
            &no_ceiling,
        )
    }
}

/// Resolve every graphics setting. Each value is the user's persisted choice
/// where they made one, otherwise the world's value, clamped under the ceiling
/// if the preset governs it. Settings the world does not author fall back to
/// the engine defaults, and occlusion two-pass is off without a
/// `PostProcessConfig`.
pub(crate) fn resolve_graphics(
    world: &GraphicsBaseline,
    user: &GraphicsSettings,
    ceiling: &QualityCeiling,
) -> ResolvedGraphics {
    let post = &world.post_config;
    let streaming = StreamingConfig::default();
    let mut resolved = ResolvedGraphics {
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

/// Resolve the preset-governed group: see [`QualityGraphics`].
pub(crate) fn resolve_quality(
    world: &GraphicsBaseline,
    user: &GraphicsSettings,
    ceiling: &QualityCeiling,
) -> QualityGraphics {
    let mut post_config = world.post_config.clone();
    overlay_quality_overrides(&mut post_config, user);
    clamp_quality_under_ceiling(&mut post_config, user, ceiling);
    QualityGraphics {
        post_config,
        render_scale: user.render_scale.unwrap_or_else(|| {
            more_aggressive_upscale(world.post_config.upscale_quality, ceiling.min_upscale)
        }),
        shadow_map_size: shadow_map_size(world.shadow_map_size, user, ceiling),
        shadow_cadence: shadow_cadence(world.shadow_cadence, user, ceiling),
        anisotropy: anisotropy(world.anisotropy, user, ceiling),
    }
}

/// Shadow map resolution. Restart-required: the cascade array is sized once at
/// backend init.
pub(crate) fn shadow_map_size(
    authored: u32,
    user: &GraphicsSettings,
    ceiling: &QualityCeiling,
) -> u32 {
    user.shadow_map_size
        .unwrap_or(authored.min(ceiling.shadow_map_size))
}

/// The cascade schedule. Live: the scheduler and the per-frame cascade split
/// read it each frame.
pub(crate) fn shadow_cadence(
    authored: ShadowCadence,
    user: &GraphicsSettings,
    ceiling: &QualityCeiling,
) -> ShadowCadence {
    ShadowCadence {
        update: user
            .shadow_update
            .unwrap_or_else(|| clamp_shadow_update(authored.update, ceiling)),
        distance: user
            .shadow_distance
            .unwrap_or(authored.distance.min(ceiling.shadow_distance)),
        cascades: user
            .shadow_cascades
            .unwrap_or(authored.cascades.min(ceiling.shadow_cascades)),
    }
}

/// Scene-sampler max anisotropy. Restart-required: the sampler is built at
/// backend init.
pub(crate) fn anisotropy(authored: u32, user: &GraphicsSettings, ceiling: &QualityCeiling) -> u32 {
    user.anisotropy.unwrap_or(authored.min(ceiling.anisotropy))
}

/// The post-process tunables: the world's config resolved, then each slider the
/// user has moved, through the same clamp the live drag applies. `fxaa` is left
/// as `resolve` seeded it, from the authored AA mode; the caller refreshes it
/// once the override + ceiling have settled the final mode.
pub(crate) fn post_process_params(
    config: Option<&PostProcessConfig>,
    user: &GraphicsSettings,
) -> PostProcessTunables {
    let mut params = config
        .map(|c| c.resolve())
        .unwrap_or(PostProcessTunables::DEFAULT);
    for s in &SLIDERS {
        if let SliderTarget::PostProcess { field, persisted } = &s.target
            && let Some(v) = *(persisted.get)(user)
        {
            *(field.get_mut)(&mut params) = (s.apply)(v);
        }
    }
    params
}

/// The ambient (IBL) scale. It rides `LightUniforms` rather than
/// `PostProcessParams`, so it resolves on its own.
pub(crate) fn ambient_intensity(
    config: Option<&PostProcessConfig>,
    user: &GraphicsSettings,
) -> f32 {
    let world = config.map(|c| c.ambient_intensity()).unwrap_or(1.0);
    SLIDERS
        .iter()
        .find_map(|s| match &s.target {
            SliderTarget::Ambient { persisted } => {
                Some((s.apply)((persisted.get)(user).unwrap_or(world)))
            }
            _ => None,
        })
        .unwrap_or(world)
}

/// Overlay the per-feature sub-quality sliders (look tuning, applied live
/// through `update_quality_params`) onto a config already carrying the world's
/// values. Not preset-governed, so no ceiling clamp.
pub(crate) fn overlay_quality_scalars(cfg: &mut PostProcessConfig, user: &GraphicsSettings) {
    for s in &SLIDERS {
        if let SliderTarget::PostConfig { field, persisted } = &s.target
            && let Some(v) = *(persisted.get)(user)
        {
            *(field.get_mut)(cfg) = (s.apply)(v);
        }
    }
}

/// Overlay the user's persisted Quality-group choices onto a config already
/// carrying the world's values: the feature toggles, the cycle (dropdown)
/// knobs, and the look-tuning sliders. Applied whether or not the world
/// declared a `PostProcessConfig` -- the schema defaults it falls back to are a
/// real authored look, not a placeholder.
pub(crate) fn overlay_quality_overrides(cfg: &mut PostProcessConfig, user: &GraphicsSettings) {
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

/// Clamp the preset-governed settings under the active ceiling: a feature the
/// tier disallows is forced off and a cycle knob is clamped coarser, except
/// where the user explicitly overrode that row. Only ever reduces, so a config
/// already within the ceiling passes through untouched.
pub(crate) fn clamp_quality_under_ceiling(
    cfg: &mut PostProcessConfig,
    user: &GraphicsSettings,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::quality_preset::{QualityPreset, resolve_ceiling};
    use concinnity_core::components::{AaMode, IndirectLighting, ShadowUpdate};
    use concinnity_core::render::backend::{GpuProfile, GpuTier};

    fn ceiling_for(preset: QualityPreset, tier: GpuTier) -> QualityCeiling {
        resolve_ceiling(
            preset,
            &GpuProfile {
                tier,
                ..GpuProfile::UNKNOWN
            },
        )
    }

    // The schema defaults author the top-tier look, so the ceiling is what
    // settles a world that declares no PostProcessConfig.
    #[test]
    fn the_low_ceiling_clamps_the_schema_defaults_off() {
        let mut cfg = PostProcessConfig::default();
        clamp_quality_under_ceiling(
            &mut cfg,
            &GraphicsSettings::default(),
            &ceiling_for(QualityPreset::Low, GpuTier::Integrated),
        );
        assert!(!cfg.ssao);
        assert!(!cfg.ssr);
        assert!(!cfg.ray_traced_reflections);
        assert_eq!(cfg.indirect_lighting, IndirectLighting::Ibl);
        assert_eq!(cfg.aa_mode, AaMode::Fxaa);
    }

    #[test]
    fn the_top_ceiling_leaves_the_schema_defaults_alone() {
        let mut cfg = PostProcessConfig::default();
        clamp_quality_under_ceiling(
            &mut cfg,
            &GraphicsSettings::default(),
            &ceiling_for(QualityPreset::Ultra, GpuTier::HighDiscrete),
        );
        assert!(cfg.ssao);
        assert!(cfg.ssr);
        assert!(cfg.ray_traced_reflections);
        assert_eq!(cfg.indirect_lighting, IndirectLighting::Ssgi);
        assert_eq!(cfg.aa_mode, AaMode::Taa);
    }

    // A ceiling only reduces: what the world turned off stays off at any tier.
    #[test]
    fn a_ceiling_never_turns_a_feature_back_on() {
        let mut cfg = PostProcessConfig {
            ssao: false,
            ssr: false,
            ray_traced_reflections: false,
            indirect_lighting: IndirectLighting::Ibl,
            aa_mode: AaMode::Off,
            ..Default::default()
        };
        clamp_quality_under_ceiling(
            &mut cfg,
            &GraphicsSettings::default(),
            &ceiling_for(QualityPreset::Ultra, GpuTier::HighDiscrete),
        );
        assert!(!cfg.ssao);
        assert!(!cfg.ssr);
        assert!(!cfg.ray_traced_reflections);
        assert_eq!(cfg.indirect_lighting, IndirectLighting::Ibl);
        assert_eq!(cfg.aa_mode, AaMode::Off);
    }

    // An explicit per-row choice survives a ceiling that would have cleared it.
    #[test]
    fn a_user_override_wins_over_the_ceiling() {
        let user = GraphicsSettings {
            ssao: Some(true),
            aa_mode: Some(AaMode::Taa),
            ..GraphicsSettings::default()
        };
        let mut cfg = PostProcessConfig {
            ssao: false,
            aa_mode: AaMode::Off,
            ..Default::default()
        };
        overlay_quality_overrides(&mut cfg, &user);
        assert!(cfg.ssao, "the override applies over the world's value");
        clamp_quality_under_ceiling(
            &mut cfg,
            &user,
            &ceiling_for(QualityPreset::Low, GpuTier::Integrated),
        );
        assert!(
            cfg.ssao,
            "the Low ceiling does not clear an explicit choice"
        );
        assert_eq!(cfg.aa_mode, AaMode::Taa);
        // A row the user left alone still clamps.
        assert!(!cfg.ssr);
    }

    fn low() -> QualityCeiling {
        ceiling_for(QualityPreset::Low, GpuTier::Integrated)
    }

    fn post_world(post: PostProcessConfig) -> GraphicsBaseline {
        GraphicsBaseline::new(None, Some(&post), None)
    }

    #[test]
    fn a_user_override_beats_the_world_value() {
        let world = GraphicsBaseline::new(
            Some(&GraphicsConfig {
                vsync: true,
                fps_cap: 60,
                ..Default::default()
            }),
            Some(&PostProcessConfig {
                hdr_display: true,
                upscale_backend: UpscalerBackend::Fsr3,
                ..Default::default()
            }),
            None,
        );
        let user = GraphicsSettings {
            vsync: Some(false),
            fps_cap: Some(30),
            hdr_display: Some(false),
            upscale_backend: Some(UpscalerBackend::Auto),
            dlss_preset: Some(DlssPreset::L),
            perf_stats: Some(false),
            ..GraphicsSettings::default()
        };
        let r = resolve_graphics(&world, &user, &low());
        assert!(!r.vsync);
        assert_eq!(r.fps_cap, 30);
        assert!(!r.hdr_display);
        assert_eq!(r.upscale_backend, UpscalerBackend::Auto);
        assert_eq!(r.dlss_preset, DlssPreset::L);
        assert!(!r.perf_stats);
        assert!(r.show_fps, "an untouched stats toggle defaults on");

        let r = resolve_graphics(&world, &GraphicsSettings::default(), &low());
        assert!(r.vsync);
        assert_eq!(r.fps_cap, 60);
        assert!(r.hdr_display);
        assert_eq!(r.upscale_backend, UpscalerBackend::Fsr3);
        assert_eq!(r.dlss_preset, DlssPreset::Default);
    }

    #[test]
    fn the_ceiling_clamps_only_a_knob_the_user_left_alone() {
        let world = GraphicsBaseline::new(
            Some(&GraphicsConfig {
                shadow_map_size: 4096,
                shadow_update: ShadowUpdate::EveryFrame,
                shadow_distance: 200,
                shadow_cascades: 4,
                anisotropy: 16,
                ..Default::default()
            }),
            None,
            None,
        );
        let q = resolve_graphics(&world, &GraphicsSettings::default(), &low()).quality;
        assert_eq!(q.shadow_map_size, 1024);
        assert_eq!(
            q.shadow_cadence,
            ShadowCadence {
                update: ShadowUpdate::Hybrid,
                distance: 40,
                cascades: 2,
            }
        );
        assert_eq!(q.anisotropy, 4);

        let user = GraphicsSettings {
            shadow_map_size: Some(4096),
            shadow_distance: Some(200),
            ..GraphicsSettings::default()
        };
        let q = resolve_graphics(&world, &user, &low()).quality;
        assert_eq!(q.shadow_map_size, 4096);
        assert_eq!(q.shadow_cadence.distance, 200);
        assert_eq!(q.shadow_cadence.cascades, 2);
    }

    #[test]
    fn a_frames_in_flight_override_clamps_to_three() {
        let world = GraphicsBaseline::new(
            Some(&GraphicsConfig {
                frames_in_flight: 7,
                ..Default::default()
            }),
            None,
            None,
        );
        let r = resolve_graphics(&world, &GraphicsSettings::default(), &low());
        assert_eq!(r.frames_in_flight, 7, "the world value is not clamped");
        let user = GraphicsSettings {
            frames_in_flight: Some(7),
            ..GraphicsSettings::default()
        };
        assert_eq!(resolve_graphics(&world, &user, &low()).frames_in_flight, 3);
    }

    #[test]
    fn render_scale_takes_the_more_aggressive_of_world_and_ceiling() {
        let scale = |world: UpscaleQuality, user: Option<UpscaleQuality>| {
            let world = post_world(PostProcessConfig {
                upscale_quality: world,
                ..Default::default()
            });
            let user = GraphicsSettings {
                render_scale: user,
                ..GraphicsSettings::default()
            };
            resolve_graphics(&world, &user, &low()).quality.render_scale
        };
        assert_eq!(
            scale(UpscaleQuality::Quality, None),
            UpscaleQuality::Performance
        );
        assert_eq!(
            scale(UpscaleQuality::UltraPerformance, None),
            UpscaleQuality::UltraPerformance
        );
        assert_eq!(
            scale(UpscaleQuality::Performance, Some(UpscaleQuality::Quality)),
            UpscaleQuality::Quality
        );
    }

    #[test]
    fn occlusion_two_pass_is_off_without_a_post_process_config() {
        let none = GraphicsSettings::default();
        let r = resolve_graphics(&GraphicsBaseline::default(), &none, &low());
        assert!(!r.occlusion_two_pass);
        let declared = post_world(PostProcessConfig::default());
        assert!(resolve_graphics(&declared, &none, &low()).occlusion_two_pass);
    }

    #[test]
    fn texture_overrides_apply_only_where_the_world_streams() {
        let user = GraphicsSettings {
            texture_cap: Some(10),
            texture_budget: Some(1),
            ..GraphicsSettings::default()
        };
        let defaults = StreamingConfig::default();
        let r = resolve_graphics(&GraphicsBaseline::default(), &user, &low());
        assert_eq!(
            (r.texture_cap, r.texture_budget),
            (defaults.texture_cap, defaults.texture_budget)
        );
        let streaming = GraphicsBaseline::new(None, None, Some(&defaults));
        let r = resolve_graphics(&streaming, &user, &low());
        assert_eq!((r.texture_cap, r.texture_budget), (10, 1));
    }

    #[test]
    fn the_fxaa_flag_follows_the_clamped_aa_mode() {
        let world = post_world(PostProcessConfig {
            aa_mode: AaMode::Taa,
            ..Default::default()
        });
        let r = resolve_graphics(&world, &GraphicsSettings::default(), &low());
        assert_eq!(r.quality.post_config.aa_mode, AaMode::Fxaa);
        assert_eq!(r.post_process.fxaa, AaMode::Fxaa.fxaa_flag());
    }

    // Look tuning is not preset-governed, so a cleared set of Quality-row
    // overrides still carries the user's sub-quality sliders.
    #[test]
    fn the_quality_group_keeps_the_look_tuning_sliders() {
        let world = post_world(PostProcessConfig {
            ssao_radius: 0.5,
            ..Default::default()
        });
        let user = GraphicsSettings {
            ssao_radius: Some(1.5),
            ..GraphicsSettings::default()
        };
        let q = resolve_quality(&world, &user, &low());
        assert_eq!(q.post_config.ssao_radius, 1.5);
    }
}
