//! The settings rows the resolver reads: the quality feature toggles and cycle
//! knobs the preset governs, and the graphics sliders. Each row states how it
//! reads and writes its field of the quality-governed assets, where the user's
//! choice for it persists, and how a ceiling clamps it. The settings menu drives the same
//! rows, so what the menu shows is what resolution produces.

use crate::components::IndirectTechnique;
use crate::gfx::auto_exposure::EV_LIMIT;
use crate::gfx::render_types::PostProcessTunables;
use crate::math::exp2;
use crate::settings::SettingKey;
use crate::settings::levels::{
    aa_mode_at, aa_mode_index, pass_resolution_at, pass_resolution_index, ssgi_rays_at,
    ssgi_rays_index,
};
use crate::settings::quality::{QualityCeiling, clamp_aa_mode};
use crate::settings::resolve::QualityEffects;
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
    /// Whether the feature is on.
    pub get: fn(&QualityEffects) -> bool,
    /// Turn the feature on or off.
    pub set: fn(&mut QualityEffects, bool),
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
    /// The option index the value shows as.
    pub index: fn(&QualityEffects) -> usize,
    /// Set the value to the one at an option index.
    pub set: fn(&mut QualityEffects, usize),
    /// Write the user's choice, where they made one, over the value.
    pub overlay: fn(&mut QualityEffects, &UserGraphicsSettings),
    /// Persist the value as the user's choice.
    pub persist: fn(&QualityEffects, &mut UserGraphicsSettings),
    /// Whether the user made a choice.
    pub overridden: fn(&UserGraphicsSettings) -> bool,
    /// Forget the user's choice.
    pub clear: fn(&mut UserGraphicsSettings),
    /// Lower the value under the ceiling.
    pub clamp: fn(&mut QualityEffects, &QualityCeiling),
}

// A toggle whose persisted and ceiling fields share one name.
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
    ($key:ident, $field:ident, $($path:ident).+) => {
        toggle!($key, $field, |cfg| cfg.$($path).+, |cfg, on| cfg.$($path).+ = on)
    };
}

// A cycle knob over the effects field at `$path`, persisted as `$field`.
macro_rules! cycle {
    ($key:ident, $field:ident, $($path:ident).+; $index:path, $at:path, $clamp:expr) => {
        QualityCycle {
            key: SettingKey::$key,
            index: |cfg| $index(cfg.$($path).+),
            set: |cfg, index| cfg.$($path).+ = $at(index),
            overlay: |cfg, user| {
                if let Some(v) = user.$field {
                    cfg.$($path).+ = v;
                }
            },
            persist: |cfg, user| user.$field = Some(cfg.$($path).+),
            overridden: |user| user.$field.is_some(),
            clear: |user| user.$field = None,
            clamp: $clamp,
        }
    };
}

/// The quality toggles, in `SettingKey::QUALITY_TOGGLES` order.
pub static QUALITY_TOGGLES: [QualityToggle; 6] = [
    toggle!(Ssao, ssao, ambient_occlusion.enabled),
    toggle!(Ssr, ssr, reflections.screen_space),
    toggle!(
        RayTracedReflections,
        ray_traced_reflections,
        reflections.ray_traced
    ),
    toggle!(
        RtReflectionShadows,
        rt_reflection_shadows,
        reflections.ray_traced_shadows
    ),
    toggle!(
        Ssgi,
        ssgi,
        |cfg| cfg.indirect_lighting.technique == IndirectTechnique::Ssgi,
        |cfg, on| {
            cfg.indirect_lighting.technique = if on {
                IndirectTechnique::Ssgi
            } else {
                IndirectTechnique::Ibl
            }
        }
    ),
    toggle!(AutoExposure, auto_exposure, exposure.auto),
];

/// The quality cycle knobs, in `SettingKey::QUALITY_CYCLES` order.
pub static QUALITY_CYCLES: [QualityCycle; 5] = [
    cycle!(
        AaMode,
        aa_mode,
        anti_aliasing.mode;
        aa_mode_index,
        aa_mode_at,
        |cfg, ceiling| {
            let aa = &mut cfg.anti_aliasing;
            aa.mode = clamp_aa_mode(aa.mode, ceiling.aa_mode)
        }
    ),
    cycle!(
        SsgiResolution,
        ssgi_resolution,
        indirect_lighting.ssgi_resolution;
        pass_resolution_index,
        pass_resolution_at,
        |cfg, ceiling| {
            let gi = &mut cfg.indirect_lighting;
            gi.ssgi_resolution = gi.ssgi_resolution.max(ceiling.ssgi_resolution)
        }
    ),
    cycle!(
        SsgiRays,
        ssgi_rays,
        indirect_lighting.ssgi_rays;
        ssgi_rays_index,
        ssgi_rays_at,
        |cfg, ceiling| {
            let gi = &mut cfg.indirect_lighting;
            gi.ssgi_rays = gi.ssgi_rays.min(ceiling.ssgi_rays)
        }
    ),
    cycle!(
        RtReflectionResolution,
        rt_reflection_resolution,
        reflections.ray_traced_resolution;
        pass_resolution_index,
        pass_resolution_at,
        |cfg, ceiling| {
            let r = &mut cfg.reflections;
            r.ray_traced_resolution = r
                .ray_traced_resolution
                .max(ceiling.rt_reflection_resolution)
        }
    ),
    cycle!(
        ReflectionBlurResolution,
        reflection_blur_resolution,
        reflections.blur_resolution;
        pass_resolution_index,
        pass_resolution_at,
        |cfg, ceiling| {
            let r = &mut cfg.reflections;
            r.blur_resolution = r.blur_resolution.max(ceiling.reflection_blur_resolution)
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
    /// A per-feature sub-quality field of the quality-governed assets, pushed
    /// live through `update_quality_params`. Look tuning, so no ceiling clamps
    /// it.
    Effect {
        /// The field.
        field: Lens<QualityEffects, f32>,
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
            | Self::Effect { persisted, .. }
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
    /// domain. The clamps mirror `post_process_tunables` and each feature's
    /// `*Settings::resolve`.
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
        target: GraphicsSliderTarget::Effect {
            field: lens!(ambient_occlusion.radius),
            persisted: lens!(ssao_radius),
        },
    },
    GraphicsSlider {
        key: SettingKey::SsaoIntensity,
        apply: |v| v.clamp(0.0, 4.0),
        target: GraphicsSliderTarget::Effect {
            field: lens!(ambient_occlusion.intensity),
            persisted: lens!(ssao_intensity),
        },
    },
    GraphicsSlider {
        key: SettingKey::SsrIntensity,
        apply: |v| v.clamp(0.0, 1.0),
        target: GraphicsSliderTarget::Effect {
            field: lens!(reflections.intensity),
            persisted: lens!(ssr_intensity),
        },
    },
    GraphicsSlider {
        key: SettingKey::SsrMaxDistance,
        apply: |v| v.clamp(1.0, 200.0),
        target: GraphicsSliderTarget::Effect {
            field: lens!(reflections.max_distance),
            persisted: lens!(ssr_max_distance),
        },
    },
    GraphicsSlider {
        key: SettingKey::SsgiIntensity,
        apply: |v| v.clamp(0.0, 4.0),
        target: GraphicsSliderTarget::Effect {
            field: lens!(indirect_lighting.ssgi_intensity),
            persisted: lens!(ssgi_intensity),
        },
    },
    GraphicsSlider {
        key: SettingKey::SsgiMaxDistance,
        apply: |v| v.clamp(0.5, 100.0),
        target: GraphicsSliderTarget::Effect {
            field: lens!(indirect_lighting.ssgi_max_distance),
            persisted: lens!(ssgi_max_distance),
        },
    },
    // The resolve also orders the EV bounds (min <= max).
    GraphicsSlider {
        key: SettingKey::AutoExposureMinEv,
        apply: |v| v.clamp(-EV_LIMIT, EV_LIMIT),
        target: GraphicsSliderTarget::Effect {
            field: lens!(exposure.auto_min_ev),
            persisted: lens!(auto_exposure_min_ev),
        },
    },
    GraphicsSlider {
        key: SettingKey::AutoExposureMaxEv,
        apply: |v| v.clamp(-EV_LIMIT, EV_LIMIT),
        target: GraphicsSliderTarget::Effect {
            field: lens!(exposure.auto_max_ev),
            persisted: lens!(auto_exposure_max_ev),
        },
    },
    GraphicsSlider {
        key: SettingKey::AutoExposureSpeed,
        apply: |v| v.clamp(1.0e-3, 20.0),
        target: GraphicsSliderTarget::Effect {
            field: lens!(exposure.auto_speed),
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
