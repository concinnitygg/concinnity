// The settings rows the quality preset governs: the Off/On feature toggles and
// the cycle knobs, each with how it reads, writes, persists, and clamps its
// `PostProcessConfig` field. Every quality mapping walks these two tables.

use concinnity_core::components::{IndirectLighting, PostProcessConfig};

use super::{Lens, SettingKey};
use crate::config::GraphicsSettings;
use crate::gfx::quality_preset::{
    QualityCeiling, clamp_aa_mode, coarser_reflection_blur, coarser_ssgi_resolution,
};

// An Off/On quality feature. `allowed` is whether the ceiling permits it on.
pub(crate) struct QualityToggle {
    pub(crate) key: SettingKey,
    pub(crate) get: fn(&PostProcessConfig) -> bool,
    pub(crate) set: fn(&mut PostProcessConfig, bool),
    pub(crate) persisted: Lens<GraphicsSettings, Option<bool>>,
    pub(crate) allowed: fn(&QualityCeiling) -> bool,
}

// A cycle (dropdown) quality knob. `clamp` lowers the value under the ceiling
// and never raises it.
pub(crate) struct QualityCycle {
    pub(crate) key: SettingKey,
    pub(crate) index: fn(&PostProcessConfig) -> usize,
    pub(crate) set: fn(&mut PostProcessConfig, usize),
    pub(crate) overlay: fn(&mut PostProcessConfig, &GraphicsSettings),
    pub(crate) persist: fn(&PostProcessConfig, &mut GraphicsSettings),
    pub(crate) overridden: fn(&GraphicsSettings) -> bool,
    pub(crate) clear: fn(&mut GraphicsSettings),
    pub(crate) clamp: fn(&mut PostProcessConfig, &QualityCeiling),
}

// A toggle whose config, persisted, and ceiling fields share one name.
macro_rules! toggle {
    ($key:ident, $field:ident, $get:expr, $set:expr) => {
        QualityToggle {
            key: SettingKey::$key,
            get: $get,
            set: $set,
            persisted: Lens {
                get: |g| &g.$field,
                get_mut: |g| &mut g.$field,
            },
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

// In `SettingKey::QUALITY_TOGGLES` order.
pub(crate) static QUALITY_TOGGLES: [QualityToggle; 5] = [
    toggle!(Ssao, ssao),
    toggle!(Ssr, ssr),
    toggle!(RayTracedReflections, ray_traced_reflections),
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

pub(crate) static QUALITY_CYCLES: [QualityCycle; 5] = [
    cycle!(
        AaMode,
        aa_mode,
        super::aa_mode_index,
        super::aa_mode_at,
        |cfg, ceiling| cfg.aa_mode = clamp_aa_mode(cfg.aa_mode, ceiling.aa_mode)
    ),
    cycle!(
        SsgiResolution,
        ssgi_resolution,
        super::ssgi_resolution_index,
        super::ssgi_resolution_at,
        |cfg, ceiling| {
            cfg.ssgi_resolution =
                coarser_ssgi_resolution(cfg.ssgi_resolution, ceiling.ssgi_resolution)
        }
    ),
    cycle!(
        SsgiRays,
        ssgi_rays,
        super::ssgi_rays_index,
        super::ssgi_rays_at,
        |cfg, ceiling| cfg.ssgi_rays = cfg.ssgi_rays.min(ceiling.ssgi_rays)
    ),
    cycle!(
        SsgiSteps,
        ssgi_steps,
        super::ssgi_steps_index,
        super::ssgi_steps_at,
        |cfg, ceiling| cfg.ssgi_steps = cfg.ssgi_steps.min(ceiling.ssgi_steps)
    ),
    cycle!(
        ReflectionBlurResolution,
        reflection_blur_resolution,
        super::reflection_blur_index,
        super::reflection_blur_at,
        |cfg, ceiling| {
            cfg.reflection_blur_resolution = coarser_reflection_blur(
                cfg.reflection_blur_resolution,
                ceiling.reflection_blur_resolution,
            )
        }
    ),
];

// The toggle row for `key`, or `None` if it is not a quality toggle.
pub(crate) fn quality_toggle(key: SettingKey) -> Option<&'static QualityToggle> {
    QUALITY_TOGGLES.iter().find(|row| row.key == key)
}

// The cycle row for `key`, or `None` if it is not a quality cycle knob.
pub(crate) fn quality_cycle(key: SettingKey) -> Option<&'static QualityCycle> {
    QUALITY_CYCLES.iter().find(|row| row.key == key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::quality_preset::{QualityPreset, resolve_ceiling};
    use crate::gfx::render_config::overlay_quality_overrides;
    use concinnity_core::components::{AaMode, ReflectionBlurResolution, SsgiResolution};
    use concinnity_core::render::backend::{GpuProfile, GpuTier};

    fn low_ceiling() -> QualityCeiling {
        resolve_ceiling(
            QualityPreset::Low,
            &GpuProfile {
                tier: GpuTier::Integrated,
                ..GpuProfile::UNKNOWN
            },
        )
    }

    #[test]
    fn toggle_rows_follow_the_core_toggle_order() {
        let keys: Vec<SettingKey> = QUALITY_TOGGLES.iter().map(|row| row.key).collect();
        assert_eq!(keys, SettingKey::QUALITY_TOGGLES);
    }

    #[test]
    fn cycle_rows_are_unique_static_option_rows() {
        let mut seen = std::collections::HashSet::new();
        for row in &QUALITY_CYCLES {
            assert!(seen.insert(row.key), "{:?} is listed twice", row.key);
            assert!(!row.key.is_quality_toggle(), "{:?}", row.key);
            assert!(super::super::options(row.key).is_some(), "{:?}", row.key);
        }
    }

    #[test]
    fn lookups_find_their_own_rows_only() {
        for row in &QUALITY_TOGGLES {
            assert!(std::ptr::eq(quality_toggle(row.key).unwrap(), row));
            assert!(quality_cycle(row.key).is_none());
        }
        for row in &QUALITY_CYCLES {
            assert!(std::ptr::eq(quality_cycle(row.key).unwrap(), row));
            assert!(quality_toggle(row.key).is_none());
        }
        assert!(quality_toggle(SettingKey::Vsync).is_none());
        assert!(quality_cycle(SettingKey::Vsync).is_none());
    }

    #[test]
    fn toggle_rows_round_trip_through_set_and_persist() {
        for row in &QUALITY_TOGGLES {
            for on in [true, false] {
                let mut cfg = PostProcessConfig::default();
                (row.set)(&mut cfg, on);
                assert_eq!((row.get)(&cfg), on, "{:?} get after set", row.key);

                let mut user = GraphicsSettings::default();
                *(row.persisted.get_mut)(&mut user) = Some(on);
                let mut overlaid = PostProcessConfig::default();
                (row.set)(&mut overlaid, !on);
                overlay_quality_overrides(&mut overlaid, &user);
                assert_eq!((row.get)(&overlaid), on, "{:?} overlay", row.key);
            }
        }
    }

    #[test]
    fn cycle_rows_round_trip_through_set_and_persist() {
        for row in &QUALITY_CYCLES {
            let count = super::super::options(row.key).unwrap().len();
            for index in 0..count {
                let mut cfg = PostProcessConfig::default();
                (row.set)(&mut cfg, index);
                assert_eq!((row.index)(&cfg), index, "{:?} get after set", row.key);

                let mut user = GraphicsSettings::default();
                assert!(!(row.overridden)(&user), "{:?}", row.key);
                (row.persist)(&cfg, &mut user);
                assert!((row.overridden)(&user), "{:?} persisted", row.key);
                let mut cleared = user.clone();
                (row.clear)(&mut cleared);
                assert!(!(row.overridden)(&cleared), "{:?} cleared", row.key);
                let mut overlaid = PostProcessConfig::default();
                (row.set)(&mut overlaid, (index + 1) % count);
                (row.overlay)(&mut overlaid, &user);
                assert_eq!((row.index)(&overlaid), index, "{:?} overlay", row.key);
            }
        }
    }

    // A top-quality config clamps to exactly the ceiling's caps.
    #[test]
    fn cycle_clamp_lowers_to_the_ceiling() {
        let ceiling = low_ceiling();
        let mut cfg = PostProcessConfig {
            aa_mode: AaMode::Taa,
            ssgi_resolution: SsgiResolution::Full,
            ssgi_rays: 32,
            ssgi_steps: 48,
            reflection_blur_resolution: ReflectionBlurResolution::Full,
            ..Default::default()
        };
        for row in &QUALITY_CYCLES {
            (row.clamp)(&mut cfg, &ceiling);
        }
        assert_eq!(cfg.aa_mode, ceiling.aa_mode);
        assert_eq!(cfg.ssgi_resolution, ceiling.ssgi_resolution);
        assert_eq!(cfg.ssgi_rays, ceiling.ssgi_rays);
        assert_eq!(cfg.ssgi_steps, ceiling.ssgi_steps);
        assert_eq!(
            cfg.reflection_blur_resolution,
            ceiling.reflection_blur_resolution
        );
    }

    // A clamp is idempotent, and a looser ceiling never raises a value a tighter
    // one lowered.
    #[test]
    fn cycle_clamp_never_raises() {
        let low = low_ceiling();
        let ultra = resolve_ceiling(QualityPreset::Ultra, &GpuProfile::UNKNOWN);
        for row in &QUALITY_CYCLES {
            let count = super::super::options(row.key).unwrap().len();
            for index in 0..count {
                let mut cfg = PostProcessConfig::default();
                (row.set)(&mut cfg, index);
                (row.clamp)(&mut cfg, &low);
                let lowered = (row.index)(&cfg);
                let mut again = cfg.clone();
                (row.clamp)(&mut again, &low);
                assert_eq!((row.index)(&again), lowered, "{:?} {index}", row.key);
                (row.clamp)(&mut cfg, &ultra);
                assert_eq!((row.index)(&cfg), lowered, "{:?} {index} raised", row.key);
            }
        }
    }
}
