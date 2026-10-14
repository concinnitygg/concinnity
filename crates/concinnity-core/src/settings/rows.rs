//! The settings rows the resolver reads: the quality feature toggles and cycle
//! knobs the preset governs, and the graphics sliders. Each row states how it
//! reads and writes its `PostProcessConfig` field, where the user's choice for
//! it persists, and how a ceiling clamps it. The settings menu drives the same
//! rows, so what the menu shows is what resolution produces.

use crate::components::{IndirectLighting, PostProcessConfig};
use crate::gfx::auto_exposure::EV_LIMIT;
use crate::gfx::render_types::PostProcessTunables;
use crate::math::exp2;
use crate::settings::SettingKey;
use crate::settings::levels::{
    aa_mode_at, aa_mode_index, pass_resolution_at, pass_resolution_index, ssgi_rays_at,
    ssgi_rays_index,
};
use crate::settings::quality::{QualityCeiling, clamp_aa_mode};
use crate::settings::user::UserGraphicsSettings;

/// A shared and a mutable accessor for the same field of `T`.
pub struct Lens<T, V> {
    /// Read the field.
    pub get: fn(&T) -> &V,
    /// Write the field.
    pub get_mut: fn(&mut T) -> &mut V,
}

macro_rules! lens {
    ($($field:ident).+) => {
        Lens {
            get: |t| &t.$($field).+,
            get_mut: |t| &mut t.$($field).+,
        }
    };
}

/// An off/on quality feature.
pub struct QualityToggle {
    /// The settings row.
    pub key: SettingKey,
    /// Whether the feature is on in a config.
    pub get: fn(&PostProcessConfig) -> bool,
    /// Turn the feature on or off in a config.
    pub set: fn(&mut PostProcessConfig, bool),
    /// Where the user's choice persists.
    pub persisted: Lens<UserGraphicsSettings, Option<bool>>,
    /// Whether the ceiling permits the feature on.
    pub allowed: fn(&QualityCeiling) -> bool,
}

/// A cycle (dropdown) quality knob. `clamp` lowers the value under the ceiling
/// and never raises it.
pub struct QualityCycle {
    /// The settings row.
    pub key: SettingKey,
    /// The option index a config's value shows as.
    pub index: fn(&PostProcessConfig) -> usize,
    /// Set a config's value to the one at an option index.
    pub set: fn(&mut PostProcessConfig, usize),
    /// Write the user's choice, where they made one, over a config.
    pub overlay: fn(&mut PostProcessConfig, &UserGraphicsSettings),
    /// Persist a config's value as the user's choice.
    pub persist: fn(&PostProcessConfig, &mut UserGraphicsSettings),
    /// Whether the user made a choice.
    pub overridden: fn(&UserGraphicsSettings) -> bool,
    /// Forget the user's choice.
    pub clear: fn(&mut UserGraphicsSettings),
    /// Lower a config's value under the ceiling.
    pub clamp: fn(&mut PostProcessConfig, &QualityCeiling),
}

// A toggle whose config, persisted, and ceiling fields share one name.
macro_rules! toggle {
    ($key:ident, $field:ident, $get:expr, $set:expr) => {
        QualityToggle {
            key: SettingKey::$key,
            get: $get,
            set: $set,
            persisted: lens!($field),
            allowed: |ceiling| ceiling.$field,
        }
    };
    ($key:ident, $field:ident) => {
        toggle!($key, $field, |cfg| cfg.$field, |cfg, on| cfg.$field = on)
    };
}

// A cycle knob whose config and persisted fields share one name.
macro_rules! cycle {
    ($key:ident, $field:ident, $index:path, $at:path, $clamp:expr) => {
        QualityCycle {
            key: SettingKey::$key,
            index: |cfg| $index(cfg.$field),
            set: |cfg, index| cfg.$field = $at(index),
            overlay: |cfg, user| {
                if let Some(v) = user.$field {
                    cfg.$field = v;
                }
            },
            persist: |cfg, user| user.$field = Some(cfg.$field),
            overridden: |user| user.$field.is_some(),
            clear: |user| user.$field = None,
            clamp: $clamp,
        }
    };
}

/// The quality toggles, in `SettingKey::QUALITY_TOGGLES` order.
pub static QUALITY_TOGGLES: [QualityToggle; 6] = [
    toggle!(Ssao, ssao),
    toggle!(Ssr, ssr),
    toggle!(RayTracedReflections, ray_traced_reflections),
    toggle!(RtReflectionShadows, rt_reflection_shadows),
    toggle!(
        Ssgi,
        ssgi,
        |cfg| cfg.indirect_lighting == IndirectLighting::Ssgi,
        |cfg, on| {
            cfg.indirect_lighting = if on {
                IndirectLighting::Ssgi
            } else {
                IndirectLighting::Ibl
            }
        }
    ),
    toggle!(AutoExposure, auto_exposure),
];

/// The quality cycle knobs, in `SettingKey::QUALITY_CYCLES` order.
pub static QUALITY_CYCLES: [QualityCycle; 5] = [
    cycle!(
        AaMode,
        aa_mode,
        aa_mode_index,
        aa_mode_at,
        |cfg, ceiling| cfg.aa_mode = clamp_aa_mode(cfg.aa_mode, ceiling.aa_mode)
    ),
    cycle!(
        SsgiResolution,
        ssgi_resolution,
        pass_resolution_index,
        pass_resolution_at,
        |cfg, ceiling| cfg.ssgi_resolution = cfg.ssgi_resolution.max(ceiling.ssgi_resolution)
    ),
    cycle!(
        SsgiRays,
        ssgi_rays,
        ssgi_rays_index,
        ssgi_rays_at,
        |cfg, ceiling| cfg.ssgi_rays = cfg.ssgi_rays.min(ceiling.ssgi_rays)
    ),
    cycle!(
        RtReflectionResolution,
        rt_reflection_resolution,
        pass_resolution_index,
        pass_resolution_at,
        |cfg, ceiling| {
            cfg.rt_reflection_resolution = cfg
                .rt_reflection_resolution
                .max(ceiling.rt_reflection_resolution)
        }
    ),
    cycle!(
        ReflectionBlurResolution,
        reflection_blur_resolution,
        pass_resolution_index,
        pass_resolution_at,
        |cfg, ceiling| {
            cfg.reflection_blur_resolution = cfg
                .reflection_blur_resolution
                .max(ceiling.reflection_blur_resolution)
        }
    ),
];

/// The toggle row for `key`, or `None` if it is not a quality toggle.
pub fn quality_toggle(key: SettingKey) -> Option<&'static QualityToggle> {
    QUALITY_TOGGLES.iter().find(|row| row.key == key)
}

/// The cycle row for `key`, or `None` if it is not a quality cycle knob.
pub fn quality_cycle(key: SettingKey) -> Option<&'static QualityCycle> {
    QUALITY_CYCLES.iter().find(|row| row.key == key)
}

/// The resolved state a graphics slider drives.
pub enum GraphicsSliderTarget {
    /// A composite tunable, pushed live through `update_post_process`.
    PostProcess {
        /// The tunable.
        field: Lens<PostProcessTunables, f32>,
        /// Where the user's choice persists.
        persisted: Lens<UserGraphicsSettings, Option<f32>>,
    },
    /// A per-feature sub-quality `PostProcessConfig` field, pushed live through
    /// `update_quality_params`. Look tuning, so no ceiling clamps it.
    PostConfig {
        /// The field.
        field: Lens<PostProcessConfig, f32>,
        /// Where the user's choice persists.
        persisted: Lens<UserGraphicsSettings, Option<f32>>,
    },
    /// The ambient (IBL) scale, which rides the light uniforms.
    Ambient {
        /// Where the user's choice persists.
        persisted: Lens<UserGraphicsSettings, Option<f32>>,
    },
}

impl GraphicsSliderTarget {
    /// Where the user's choice persists.
    pub fn persisted(&self) -> &Lens<UserGraphicsSettings, Option<f32>> {
        match self {
            Self::PostProcess { persisted, .. }
            | Self::PostConfig { persisted, .. }
            | Self::Ambient { persisted } => persisted,
        }
    }
}

/// A graphics slider: the clamp from the value the user picks to the value
/// applied, and the state it drives. The persisted choice is the user-facing
/// value.
pub struct GraphicsSlider {
    /// The settings row.
    pub key: SettingKey,
    /// The user-facing value to the applied value, clamped to the engine's
    /// domain. The clamps mirror `PostProcessConfig::resolve` and each
    /// feature's `*Settings::resolve`.
    pub apply: fn(f32) -> f32,
    /// What the applied value drives.
    pub target: GraphicsSliderTarget,
}

/// Every graphics slider.
pub static GRAPHICS_SLIDERS: [GraphicsSlider; 16] = [
    // Authored in EV (centered on neutral), applied as the multiplier 2^ev.
    GraphicsSlider {
        key: SettingKey::Exposure,
        apply: |v| exp2(v.clamp(-EV_LIMIT, EV_LIMIT)),
        target: GraphicsSliderTarget::PostProcess {
            field: lens!(exposure),
            persisted: lens!(exposure_ev),
        },
    },
    GraphicsSlider {
        key: SettingKey::BloomIntensity,
        apply: |v| v.max(0.0),
        target: GraphicsSliderTarget::PostProcess {
            field: lens!(bloom_intensity),
            persisted: lens!(bloom_intensity),
        },
    },
    GraphicsSlider {
        key: SettingKey::BloomThreshold,
        apply: |v| v.max(0.0),
        target: GraphicsSliderTarget::PostProcess {
            field: lens!(bloom_threshold),
            persisted: lens!(bloom_threshold),
        },
    },
    GraphicsSlider {
        key: SettingKey::Vignette,
        apply: |v| v.clamp(0.0, 1.0),
        target: GraphicsSliderTarget::PostProcess {
            field: lens!(vignette),
            persisted: lens!(vignette),
        },
    },
    GraphicsSlider {
        key: SettingKey::LutStrength,
        apply: |v| v.clamp(0.0, 1.0),
        target: GraphicsSliderTarget::PostProcess {
            field: lens!(lut_strength),
            persisted: lens!(lut_strength),
        },
    },
    GraphicsSlider {
        key: SettingKey::AmbientIntensity,
        apply: |v| v.clamp(0.0, 16.0),
        target: GraphicsSliderTarget::Ambient {
            persisted: lens!(ambient_intensity),
        },
    },
    GraphicsSlider {
        key: SettingKey::BloomKnee,
        apply: |v| v.max(0.0),
        target: GraphicsSliderTarget::PostProcess {
            field: lens!(bloom_knee),
            persisted: lens!(bloom_knee),
        },
    },
    GraphicsSlider {
        key: SettingKey::SsaoRadius,
        apply: |v| v.max(1.0e-3),
        target: GraphicsSliderTarget::PostConfig {
            field: lens!(ssao_radius),
            persisted: lens!(ssao_radius),
        },
    },
    GraphicsSlider {
        key: SettingKey::SsaoIntensity,
        apply: |v| v.clamp(0.0, 4.0),
        target: GraphicsSliderTarget::PostConfig {
            field: lens!(ssao_intensity),
            persisted: lens!(ssao_intensity),
        },
    },
    GraphicsSlider {
        key: SettingKey::SsrIntensity,
        apply: |v| v.clamp(0.0, 1.0),
        target: GraphicsSliderTarget::PostConfig {
            field: lens!(ssr_intensity),
            persisted: lens!(ssr_intensity),
        },
    },
    GraphicsSlider {
        key: SettingKey::SsrMaxDistance,
        apply: |v| v.clamp(1.0, 200.0),
        target: GraphicsSliderTarget::PostConfig {
            field: lens!(ssr_max_distance),
            persisted: lens!(ssr_max_distance),
        },
    },
    GraphicsSlider {
        key: SettingKey::SsgiIntensity,
        apply: |v| v.clamp(0.0, 4.0),
        target: GraphicsSliderTarget::PostConfig {
            field: lens!(ssgi_intensity),
            persisted: lens!(ssgi_intensity),
        },
    },
    GraphicsSlider {
        key: SettingKey::SsgiMaxDistance,
        apply: |v| v.clamp(0.5, 100.0),
        target: GraphicsSliderTarget::PostConfig {
            field: lens!(ssgi_max_distance),
            persisted: lens!(ssgi_max_distance),
        },
    },
    // The resolve also orders the EV bounds (min <= max).
    GraphicsSlider {
        key: SettingKey::AutoExposureMinEv,
        apply: |v| v.clamp(-EV_LIMIT, EV_LIMIT),
        target: GraphicsSliderTarget::PostConfig {
            field: lens!(auto_exposure_min_ev),
            persisted: lens!(auto_exposure_min_ev),
        },
    },
    GraphicsSlider {
        key: SettingKey::AutoExposureMaxEv,
        apply: |v| v.clamp(-EV_LIMIT, EV_LIMIT),
        target: GraphicsSliderTarget::PostConfig {
            field: lens!(auto_exposure_max_ev),
            persisted: lens!(auto_exposure_max_ev),
        },
    },
    GraphicsSlider {
        key: SettingKey::AutoExposureSpeed,
        apply: |v| v.clamp(1.0e-3, 20.0),
        target: GraphicsSliderTarget::PostConfig {
            field: lens!(auto_exposure_speed),
            persisted: lens!(auto_exposure_speed),
        },
    },
];

/// The graphics slider for `key`, or `None` if it is not one.
pub fn graphics_slider(key: SettingKey) -> Option<&'static GraphicsSlider> {
    GRAPHICS_SLIDERS.iter().find(|s| s.key == key)
}

#[cfg(test)]
mod tests;
