// src/gfx/render_config.rs
//
// One expression per render setting the world authors and the settings menu can
// override. A setting resolves to the user's persisted choice where they made
// one, otherwise to the world's value under the active quality preset's
// ceiling. Init resolves every setting through these at launch and the
// live-lighting seam re-resolves the ones an authoring edit can reach, so an
// edit applied to a running world shows what relaunching that world would.

use crate::components::{PostProcessConfig, PostProcessResolve, ShadowUpdate};
use crate::config::GraphicsSettings;
use crate::gfx::quality_preset::{QualityCeiling, clamp_shadow_update};
use crate::gfx::render_types::PostProcessTunables;
use crate::gfx::settings::slider_apply_value;

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
    if let Some(v) = user.exposure_ev {
        params.exposure = slider_apply_value("exposure", v);
    }
    if let Some(v) = user.bloom_intensity {
        params.bloom_intensity = slider_apply_value("bloom_intensity", v);
    }
    if let Some(v) = user.bloom_threshold {
        params.bloom_threshold = slider_apply_value("bloom_threshold", v);
    }
    if let Some(v) = user.bloom_knee {
        params.bloom_knee = slider_apply_value("bloom_knee", v);
    }
    if let Some(v) = user.vignette {
        params.vignette = slider_apply_value("vignette", v);
    }
    if let Some(v) = user.lut_strength {
        params.lut_strength = slider_apply_value("lut_strength", v);
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
    slider_apply_value("ambient_intensity", user.ambient_intensity.unwrap_or(world))
}

/// Overlay the per-feature sub-quality sliders (look tuning, applied live
/// through `update_quality_params`) onto a config already carrying the world's
/// values. Not preset-governed, so no ceiling clamp.
pub(crate) fn overlay_quality_scalars(cfg: &mut PostProcessConfig, user: &GraphicsSettings) {
    if let Some(v) = user.ssao_radius {
        cfg.ssao_radius = slider_apply_value("ssao_radius", v);
    }
    if let Some(v) = user.ssao_intensity {
        cfg.ssao_intensity = slider_apply_value("ssao_intensity", v);
    }
    if let Some(v) = user.ssr_intensity {
        cfg.ssr_intensity = slider_apply_value("ssr_intensity", v);
    }
    if let Some(v) = user.ssr_max_distance {
        cfg.ssr_max_distance = slider_apply_value("ssr_max_distance", v);
    }
    if let Some(v) = user.ssgi_intensity {
        cfg.ssgi_intensity = slider_apply_value("ssgi_intensity", v);
    }
    if let Some(v) = user.ssgi_max_distance {
        cfg.ssgi_max_distance = slider_apply_value("ssgi_max_distance", v);
    }
    if let Some(v) = user.auto_exposure_min_ev {
        cfg.auto_exposure_min_ev = slider_apply_value("auto_exposure_min_ev", v);
    }
    if let Some(v) = user.auto_exposure_max_ev {
        cfg.auto_exposure_max_ev = slider_apply_value("auto_exposure_max_ev", v);
    }
    if let Some(v) = user.auto_exposure_speed {
        cfg.auto_exposure_speed = slider_apply_value("auto_exposure_speed", v);
    }
}
