//! The settings rows the resolver reads: the quality feature toggles and cycle
//! knobs the preset governs, and the graphics sliders. Each row states how it
//! reads and writes its field of the resolved graphics settings, where the user's
//! choice for it persists, and how a ceiling clamps it. The settings menu drives the same
//! rows, so what the menu shows is what resolution produces.

use crate::components::IndirectTechnique;
use crate::gfx::auto_exposure::EV_LIMIT;
use crate::settings::SettingKey;
use crate::settings::levels::{AA_MODE, PASS_RESOLUTION, SSGI_RAYS};
use crate::settings::quality::{QualityCeiling, clamp_aa_mode};
use crate::settings::resolve::ResolvedGraphics;
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
    pub get: fn(&ResolvedGraphics) -> bool,
    /// Turn the feature on or off.
    pub set: fn(&mut ResolvedGraphics, bool),
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
    pub index: fn(&ResolvedGraphics) -> usize,
    /// Set the value to the one at an option index.
    pub set: fn(&mut ResolvedGraphics, usize),
    /// Write the user's choice, where they made one, over the value.
    pub overlay: fn(&mut ResolvedGraphics, &UserGraphicsSettings),
    /// Persist the value as the user's choice.
    pub persist: fn(&ResolvedGraphics, &mut UserGraphicsSettings),
    /// Whether the user made a choice.
    pub overridden: fn(&UserGraphicsSettings) -> bool,
    /// Forget the user's choice.
    pub clear: fn(&mut UserGraphicsSettings),
    /// Lower the value under the ceiling.
    pub clamp: fn(&mut ResolvedGraphics, &QualityCeiling),
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

// A cycle knob over the effects field at `$path` with options `$levels`,
// persisted as `$field`.
macro_rules! cycle {
    ($key:ident, $field:ident, $($path:ident).+; $levels:expr, $clamp:expr) => {
        QualityCycle {
            key: SettingKey::$key,
            index: |cfg| $levels.index_of(cfg.$($path).+),
            set: |cfg, index| cfg.$($path).+ = $levels.at(index),
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
        AA_MODE,
        |cfg, ceiling| {
            let aa = &mut cfg.anti_aliasing;
            aa.mode = clamp_aa_mode(aa.mode, ceiling.aa_mode)
        }
    ),
    cycle!(
        SsgiResolution,
        ssgi_resolution,
        indirect_lighting.ssgi_resolution;
        PASS_RESOLUTION,
        |cfg, ceiling| {
            let gi = &mut cfg.indirect_lighting;
            gi.ssgi_resolution = gi.ssgi_resolution.max(ceiling.ssgi_resolution)
        }
    ),
    cycle!(
        SsgiRays,
        ssgi_rays,
        indirect_lighting.ssgi_rays;
        SSGI_RAYS,
        |cfg, ceiling| {
            let gi = &mut cfg.indirect_lighting;
            gi.ssgi_rays = gi.ssgi_rays.min(ceiling.ssgi_rays)
        }
    ),
    cycle!(
        RtReflectionResolution,
        rt_reflection_resolution,
        reflections.ray_traced_resolution;
        PASS_RESOLUTION,
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
        PASS_RESOLUTION,
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

/// A graphics slider: the clamp from the value the user picks to the value
/// applied, and the resolved field it sets. The persisted choice is the
/// user-facing value.
pub struct GraphicsSlider {
    /// The settings row.
    pub key: SettingKey,
    /// The user-facing value to the applied value, clamped to the engine's
    /// domain. The clamps mirror `post_process_tunables` and each feature's
    /// `*Settings::resolve`.
    pub apply: fn(f32) -> f32,
    /// The resolved field the applied value sets.
    pub field: Lens<ResolvedGraphics, f32>,
    /// Where the user's choice persists.
    pub persisted: Lens<UserGraphicsSettings, Option<f32>>,
}

// A slider over the resolved field at `$path`, persisted as `$field`.
macro_rules! slider {
    ($key:ident, $field:ident, $($path:ident).+, $apply:expr) => {
        GraphicsSlider {
            key: SettingKey::$key,
            apply: $apply,
            field: lens!($($path).+),
            persisted: lens!($field),
        }
    };
}

/// Every graphics slider.
pub static GRAPHICS_SLIDERS: [GraphicsSlider; 16] = [
    // Authored in EV (centered on neutral); the composite applies 2^ev.
    slider!(Exposure, exposure_ev, exposure.ev, |v| v
        .clamp(-EV_LIMIT, EV_LIMIT)),
    slider!(BloomIntensity, bloom_intensity, bloom.intensity, |v| v
        .max(0.0)),
    slider!(BloomThreshold, bloom_threshold, bloom.threshold, |v| v
        .max(0.0)),
    slider!(Vignette, vignette, vignette.strength, |v| v.clamp(0.0, 1.0)),
    slider!(LutStrength, lut_strength, lut_strength, |v| v
        .clamp(0.0, 1.0)),
    slider!(
        AmbientIntensity,
        ambient_intensity,
        indirect_lighting.ambient_intensity,
        |v| v.clamp(0.0, 16.0)
    ),
    slider!(BloomKnee, bloom_knee, bloom.knee, |v| v.max(0.0)),
    slider!(SsaoRadius, ssao_radius, ambient_occlusion.radius, |v| v
        .max(1.0e-3)),
    slider!(
        SsaoIntensity,
        ssao_intensity,
        ambient_occlusion.intensity,
        |v| v.clamp(0.0, 4.0)
    ),
    slider!(SsrIntensity, ssr_intensity, reflections.intensity, |v| v
        .clamp(0.0, 1.0)),
    slider!(
        SsrMaxDistance,
        ssr_max_distance,
        reflections.max_distance,
        |v| v.clamp(1.0, 200.0)
    ),
    slider!(
        SsgiIntensity,
        ssgi_intensity,
        indirect_lighting.ssgi_intensity,
        |v| v.clamp(0.0, 4.0)
    ),
    slider!(
        SsgiMaxDistance,
        ssgi_max_distance,
        indirect_lighting.ssgi_max_distance,
        |v| v.clamp(0.5, 100.0)
    ),
    // The resolve also orders the EV bounds (min <= max).
    slider!(
        AutoExposureMinEv,
        auto_exposure_min_ev,
        exposure.auto_min_ev,
        |v| v.clamp(-EV_LIMIT, EV_LIMIT)
    ),
    slider!(
        AutoExposureMaxEv,
        auto_exposure_max_ev,
        exposure.auto_max_ev,
        |v| v.clamp(-EV_LIMIT, EV_LIMIT)
    ),
    slider!(
        AutoExposureSpeed,
        auto_exposure_speed,
        exposure.auto_speed,
        |v| v.clamp(1.0e-3, 20.0)
    ),
];

/// The graphics slider for `key`, or `None` if it is not one.
pub fn graphics_slider(key: SettingKey) -> Option<&'static GraphicsSlider> {
    GRAPHICS_SLIDERS.iter().find(|s| s.key == key)
}

#[cfg(test)]
mod tests;
