//! One expression per render setting the world authors and the settings menu can
//! override. A setting resolves to the user's persisted choice where they made
//! one, otherwise to the world's value under the active quality preset's
//! ceiling. Init resolves every setting through these at launch and the
//! live-lighting seam re-resolves the ones an authoring edit can reach, so an
//! edit applied to a running world shows what relaunching that world would.

use concinnity_core::components::{PostProcessConfig, ShadowUpdate};
use concinnity_core::gfx::render_types::PostProcessTunables;

use crate::config::GraphicsSettings;
use crate::gfx::quality_preset::{QualityCeiling, clamp_shadow_update};
use crate::settings::quality_rows::{QUALITY_CYCLES, QUALITY_TOGGLES};
use crate::settings::{SLIDERS, SliderTarget};

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

/// Cascade re-render cadence. Live: the scheduler reads it each shadow pass.
pub(crate) fn shadow_update(
    authored: ShadowUpdate,
    user: &GraphicsSettings,
    ceiling: &QualityCeiling,
) -> ShadowUpdate {
    match user.shadow_update {
        Some(v) => v,
        None => clamp_shadow_update(authored, ceiling),
    }
}

/// Shadow distance in world units. Live: the per-frame cascade split reads it.
pub(crate) fn shadow_distance(
    authored: u32,
    user: &GraphicsSettings,
    ceiling: &QualityCeiling,
) -> u32 {
    user.shadow_distance
        .unwrap_or(authored.min(ceiling.shadow_distance))
}

/// Cascade count. Live: the per-frame split + schedule read it.
pub(crate) fn shadow_cascades(
    authored: u32,
    user: &GraphicsSettings,
    ceiling: &QualityCeiling,
) -> u32 {
    user.shadow_cascades
        .unwrap_or(authored.min(ceiling.shadow_cascades))
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
    use concinnity_core::components::{AaMode, IndirectLighting};
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
}
