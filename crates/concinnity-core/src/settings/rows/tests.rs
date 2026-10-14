use super::*;
use crate::components::{AaMode, PassResolution};
use crate::render::backend::{GpuProfile, GpuTier};
use crate::settings::options;
use crate::settings::quality::{QualityPreset, resolve_ceiling};
use crate::settings::resolve::{QualityEffects, overlay_quality_overrides};
use alloc::vec::Vec;

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
fn cycle_rows_follow_the_core_cycle_order() {
    let keys: Vec<SettingKey> = QUALITY_CYCLES.iter().map(|row| row.key).collect();
    assert_eq!(keys, SettingKey::QUALITY_CYCLES);
}

#[test]
fn cycle_rows_are_unique_static_option_rows() {
    let mut seen = Vec::new();
    for row in &QUALITY_CYCLES {
        assert!(!seen.contains(&row.key), "{:?} is listed twice", row.key);
        seen.push(row.key);
        assert!(!row.key.is_quality_toggle(), "{:?}", row.key);
        assert!(options(row.key).is_some(), "{:?}", row.key);
    }
}

#[test]
fn lookups_find_their_own_rows_only() {
    for row in &QUALITY_TOGGLES {
        assert!(core::ptr::eq(quality_toggle(row.key).unwrap(), row));
        assert!(quality_cycle(row.key).is_none());
    }
    for row in &QUALITY_CYCLES {
        assert!(core::ptr::eq(quality_cycle(row.key).unwrap(), row));
        assert!(quality_toggle(row.key).is_none());
    }
    assert!(quality_toggle(SettingKey::Vsync).is_none());
    assert!(quality_cycle(SettingKey::Vsync).is_none());
}

#[test]
fn toggle_rows_round_trip_through_set_and_persist() {
    for row in &QUALITY_TOGGLES {
        for on in [true, false] {
            let mut cfg = QualityEffects::default();
            (row.set)(&mut cfg, on);
            assert_eq!((row.get)(&cfg), on, "{:?} get after set", row.key);

            let mut user = UserGraphicsSettings::default();
            *(row.persisted.get_mut)(&mut user) = Some(on);
            let mut overlaid = QualityEffects::default();
            (row.set)(&mut overlaid, !on);
            overlay_quality_overrides(&mut overlaid, &user);
            assert_eq!((row.get)(&overlaid), on, "{:?} overlay", row.key);
        }
    }
}

#[test]
fn cycle_rows_round_trip_through_set_and_persist() {
    for row in &QUALITY_CYCLES {
        let count = options(row.key).unwrap().len();
        for index in 0..count {
            let mut cfg = QualityEffects::default();
            (row.set)(&mut cfg, index);
            assert_eq!((row.index)(&cfg), index, "{:?} get after set", row.key);

            let mut user = UserGraphicsSettings::default();
            assert!(!(row.overridden)(&user), "{:?}", row.key);
            (row.persist)(&cfg, &mut user);
            assert!((row.overridden)(&user), "{:?} persisted", row.key);
            let mut cleared = user.clone();
            (row.clear)(&mut cleared);
            assert!(!(row.overridden)(&cleared), "{:?} cleared", row.key);
            let mut overlaid = QualityEffects::default();
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
    let mut cfg = QualityEffects::default();
    cfg.anti_aliasing.mode = AaMode::Taa;
    cfg.indirect_lighting.ssgi_resolution = PassResolution::Full;
    cfg.indirect_lighting.ssgi_rays = 4;
    cfg.reflections.blur_resolution = PassResolution::Full;
    for row in &QUALITY_CYCLES {
        (row.clamp)(&mut cfg, &ceiling);
    }
    assert_eq!(cfg.anti_aliasing.mode, ceiling.aa_mode);
    let gi = &cfg.indirect_lighting;
    assert_eq!(gi.ssgi_resolution, ceiling.ssgi_resolution);
    assert_eq!(gi.ssgi_rays, ceiling.ssgi_rays);
    assert_eq!(
        cfg.reflections.blur_resolution,
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
        let count = options(row.key).unwrap().len();
        for index in 0..count {
            let mut cfg = QualityEffects::default();
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

#[test]
fn every_graphics_slider_is_a_distinct_slider_key() {
    let mut seen = Vec::new();
    for s in &GRAPHICS_SLIDERS {
        assert!(!seen.contains(&s.key), "{:?} is listed twice", s.key);
        seen.push(s.key);
        assert_eq!(s.key.kind(), crate::settings::SettingKind::Slider);
        assert!(core::ptr::eq(graphics_slider(s.key).unwrap(), s));
    }
    assert!(graphics_slider(SettingKey::Vsync).is_none());
}

// A slider's persisted choice is the user-facing value; its applied value is
// what the resolved state holds.
#[test]
fn the_exposure_slider_applies_stops_as_a_multiplier() {
    let exposure = graphics_slider(SettingKey::Exposure).unwrap();
    assert_eq!((exposure.apply)(1.0), 2.0);
    assert_eq!((exposure.apply)(-100.0), exp2(-EV_LIMIT));
    let mut user = UserGraphicsSettings::default();
    *(exposure.target.persisted().get_mut)(&mut user) = Some(0.5);
    assert_eq!(user.exposure_ev, Some(0.5));
}
