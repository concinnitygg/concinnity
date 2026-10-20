use super::*;
use crate::components::PassResolution;
use crate::render::output::dlss::DlssPreset;
use crate::settings::levels::{SSGI_RAYS, TEXTURE_QUALITY};
use crate::settings::rows::{quality_cycle, quality_toggle};
use crate::settings::{SettingKind, options};
use alloc::vec::Vec;

const BOTH: [Platform; 2] = [Platform::DirectX, Platform::Vulkan];

fn available(key: SettingKey, caps: DeviceCapabilities, vendor: GpuVendor) -> bool {
    setting_available(key, &caps, vendor, &BOTH)
}

#[test]
fn stats_hud_toggles_are_off_then_on_and_never_gated() {
    let caps = DeviceCapabilities {
        ray_tracing: false,
        ..DeviceCapabilities::ALL
    };
    for key in [
        SettingKey::PerfStats,
        SettingKey::ShowFps,
        SettingKey::ShowVram,
        SettingKey::Vsync,
    ] {
        assert_eq!(options(key), Some(&["Off", "On"][..]), "{key:?}");
        assert!(available(key, caps, GpuVendor::Nvidia), "{key:?}");
    }
}

#[test]
fn graphics_quality_options_match_preset_order() {
    use crate::settings::quality::QualityPreset;
    let labels = options(SettingKey::GraphicsQuality).expect("graphics_quality options");
    assert_eq!(labels.len(), QualityPreset::ALL.len());
    for (i, p) in QualityPreset::ALL.iter().enumerate() {
        assert_eq!(labels[i], p.name(), "label {i}");
    }
}

#[test]
fn quality_toggles_are_off_then_on_and_classified() {
    for key in SettingKey::QUALITY_TOGGLES {
        assert!(
            key.is_quality_toggle(),
            "{key:?} should classify as a toggle"
        );
        assert_eq!(options(key), Some(&["Off", "On"][..]), "{key:?} options");
        assert!(slider(key).is_none(), "{key:?} should not be a slider");
    }
    assert!(!SettingKey::Vsync.is_quality_toggle());
    assert!(!SettingKey::Exposure.is_quality_toggle());
}

#[test]
fn rebind_keys_are_a_distinct_category() {
    use crate::input::keymap::Bindable;
    for b in Bindable::ALL {
        let key = SettingKey::KeyRebind(b);
        assert!(options(key).is_none(), "{key:?} should not be a cycle row");
        assert!(slider(key).is_none(), "{key:?} should not be a slider");
    }
}

#[test]
fn rt_rows_gated_on_ray_tracing_capability() {
    let capable = DeviceCapabilities {
        ray_tracing: true,
        ..DeviceCapabilities::ALL
    };
    let incapable = DeviceCapabilities {
        ray_tracing: false,
        ..DeviceCapabilities::ALL
    };
    for key in [
        SettingKey::RayTracedReflections,
        SettingKey::RtReflectionResolution,
        SettingKey::RtReflectionShadows,
    ] {
        assert!(available(key, capable, GpuVendor::Nvidia));
        assert!(!available(key, incapable, GpuVendor::Nvidia));
    }
    for key in [
        SettingKey::AaMode,
        SettingKey::Ssao,
        SettingKey::Ssr,
        SettingKey::Ssgi,
        SettingKey::AutoExposure,
    ] {
        assert!(available(key, incapable, GpuVendor::Nvidia), "{key:?}");
    }
    assert!(available(
        SettingKey::RayTracedReflections,
        DeviceCapabilities::default(),
        GpuVendor::Nvidia
    ));
}

#[test]
fn cycle_wraps_and_jumps() {
    assert_eq!(cycle(0, 2, SettingOp::Next), 1);
    assert_eq!(cycle(1, 2, SettingOp::Next), 0);
    assert_eq!(cycle(0, 2, SettingOp::Prev), 1);
    assert_eq!(cycle(2, 3, SettingOp::Next), 0);
    assert_eq!(cycle(0, 3, SettingOp::Prev), 2);
    // A dropdown pick jumps straight to the chosen index, clamped to the last.
    assert_eq!(cycle(0, 4, SettingOp::SetIndex(2)), 2);
    assert_eq!(cycle(1, 4, SettingOp::SetIndex(9)), 3);
    // A row with no options stays put.
    assert_eq!(cycle(0, 0, SettingOp::Next), 0);
}

#[test]
fn graphics_api_choice_moves_among_compiled_apis() {
    assert_eq!(
        graphics_api_choice(&BOTH, Platform::DirectX, SettingOp::Next),
        Some(GraphicsApiChoice {
            platform: Platform::Vulkan,
            trial: Some(GraphicsApiTrial::Untried),
        })
    );
    // Back to native needs no trial.
    assert_eq!(
        graphics_api_choice(&BOTH, Platform::Vulkan, SettingOp::SetIndex(0)),
        Some(GraphicsApiChoice {
            platform: Platform::DirectX,
            trial: None,
        })
    );
    // Picking what is already chosen changes nothing, and so does a build with
    // one API or none.
    assert_eq!(
        graphics_api_choice(&BOTH, Platform::Vulkan, SettingOp::SetIndex(1)),
        None
    );
    assert_eq!(
        graphics_api_choice(&[Platform::Vulkan], Platform::Vulkan, SettingOp::Next),
        None
    );
    assert_eq!(
        graphics_api_choice(&[], Platform::Vulkan, SettingOp::Next),
        None
    );
}

#[test]
fn the_graphics_api_label_names_a_failed_switch() {
    assert_eq!(graphics_api_label(Platform::Metal, None), "Metal");
    assert_eq!(
        graphics_api_label(Platform::DirectX, Some(Platform::Vulkan)),
        "DirectX 12 (Vulkan failed)"
    );
}

#[test]
fn the_graphics_api_row_is_live_only_with_a_choice() {
    let caps = DeviceCapabilities::ALL;
    assert!(setting_available(
        SettingKey::GraphicsApi,
        &caps,
        GpuVendor::Other,
        &BOTH
    ));
    assert!(!setting_available(
        SettingKey::GraphicsApi,
        &caps,
        GpuVendor::Other,
        &[Platform::Metal]
    ));
}

#[test]
fn known_settings_have_options() {
    assert_eq!(options(SettingKey::WindowMode).unwrap().len(), 3);
    assert_eq!(options(SettingKey::RenderScale).unwrap().len(), 4);
    assert!(options(SettingKey::Resolution).is_none());
    assert!(crate::settings::is_dynamic_dropdown(SettingKey::Resolution));
    assert_eq!(options(SettingKey::MasterVolume).unwrap().len(), 5);
    assert!(options(SettingKey::MouseSensitivity).is_none());
    assert!(slider(SettingKey::MouseSensitivity).is_some());
    for key in [
        SettingKey::SsgiResolution,
        SettingKey::RtReflectionResolution,
        SettingKey::ReflectionBlurResolution,
    ] {
        assert_eq!(
            options(key).map(<[&str]>::len),
            Some(PassResolution::ALL.len())
        );
    }
    assert_eq!(
        options(SettingKey::SsgiRays).map(<[&str]>::len),
        Some(SSGI_RAYS.values().len())
    );
    assert_eq!(
        options(SettingKey::TextureQuality).map(<[&str]>::len),
        Some(TEXTURE_QUALITY.values().len())
    );
    assert_eq!(
        options(SettingKey::DlssPreset).unwrap(),
        DlssPreset::LABELS.as_slice()
    );
}

// Every cycle row with levels is a cycle row and never a slider.
#[test]
fn level_rows_are_cycle_rows() {
    for key in [
        SettingKey::SsgiResolution,
        SettingKey::SsgiRays,
        SettingKey::ShadowMapSize,
        SettingKey::Anisotropy,
        SettingKey::ShadowDistance,
        SettingKey::ShadowCascades,
        SettingKey::ShadowUpdate,
        SettingKey::FramesInFlight,
        SettingKey::UpscaleBackend,
        SettingKey::DlssPreset,
    ] {
        assert!(options(key).is_some(), "{key:?} should be a cycle row");
        assert!(slider(key).is_none(), "{key:?} should not be a slider");
    }
    assert!(quality_cycle(SettingKey::RtReflectionResolution).is_some());
    assert!(quality_cycle(SettingKey::ReflectionBlurResolution).is_some());
    assert!(quality_toggle(SettingKey::RtReflectionShadows).is_some());
}

#[test]
fn display_toggles_are_off_on_rows_outside_the_preset() {
    for key in [
        SettingKey::TemporalUpscaling,
        SettingKey::HdrDisplay,
        SettingKey::HdrPq,
        SettingKey::OcclusionTwoPass,
    ] {
        assert_eq!(options(key), Some(&["Off", "On"][..]), "{key:?} options");
        assert!(slider(key).is_none(), "{key:?} should not be a slider");
        assert!(!key.is_quality_toggle(), "{key:?} is not a quality toggle");
        assert!(
            quality_cycle(key).is_none(),
            "{key:?} is not a quality knob"
        );
    }
}

#[test]
fn upscalers_are_vendor_gated_and_the_row_capability_gated() {
    for vendor in [
        GpuVendor::Apple,
        GpuVendor::Nvidia,
        GpuVendor::Amd,
        GpuVendor::Intel,
        GpuVendor::Other,
    ] {
        assert!(upscale_backend_available(UpscalerBackend::Auto, vendor));
        assert!(upscale_backend_available(UpscalerBackend::Fsr3, vendor));
    }
    assert!(upscale_backend_available(
        UpscalerBackend::Dlss,
        GpuVendor::Nvidia
    ));
    assert!(!upscale_backend_available(
        UpscalerBackend::Dlss,
        GpuVendor::Amd
    ));
    assert!(upscale_backend_available(
        UpscalerBackend::Xess,
        GpuVendor::Intel
    ));
    assert!(!upscale_backend_available(
        UpscalerBackend::Xess,
        GpuVendor::Nvidia
    ));
    let fixed = DeviceCapabilities {
        selectable_upscaler: false,
        ..DeviceCapabilities::ALL
    };
    assert!(available(
        SettingKey::UpscaleBackend,
        DeviceCapabilities::ALL,
        GpuVendor::Nvidia
    ));
    assert!(!available(
        SettingKey::UpscaleBackend,
        fixed,
        GpuVendor::Nvidia
    ));
    // The DLSS preset is live wherever the upscaler row can pick DLSS.
    let caps = DeviceCapabilities::ALL;
    assert!(available(SettingKey::DlssPreset, caps, GpuVendor::Nvidia));
    assert!(!available(SettingKey::DlssPreset, caps, GpuVendor::Amd));
    assert!(!available(SettingKey::DlssPreset, fixed, GpuVendor::Nvidia));
}

#[test]
fn live_quality_changes_reset_the_temporal_history_and_restart_rows_do_not() {
    assert!(resets_temporal_history(SettingKey::GraphicsQuality));
    assert!(resets_temporal_history(SettingKey::AaMode));
    assert!(resets_temporal_history(SettingKey::Ssr));
    for key in [
        SettingKey::DlssPreset,
        SettingKey::UpscaleBackend,
        SettingKey::RenderScale,
        SettingKey::Vsync,
        SettingKey::Exposure,
    ] {
        assert!(!resets_temporal_history(key), "{key:?}");
    }
}

#[test]
fn mouse_sensitivity_is_a_slider_1_to_100() {
    let s = slider(SettingKey::MouseSensitivity).expect("a slider");
    assert_eq!(s.range, (1.0, 100.0));
    for &ui in &[1.0_f32, 25.0, 50.0, 100.0] {
        let stored = s.apply(ui);
        let back = (s.recover)(stored);
        assert!((back - ui).abs() < 1.0e-2, "ui={ui} -> {stored} -> {back}");
    }
    assert!((s.apply(1.0) - MOUSE_SENS_MIN).abs() < 1.0e-9);
    assert!((s.apply(100.0) - MOUSE_SENS_MAX).abs() < 1.0e-9);
    assert!(s.apply(10.0) < s.apply(90.0));
    assert_eq!((s.format)(26.3), "26");
    let def = (s.recover)(DEFAULT_MOUSE_SENSITIVITY);
    assert!(
        (1.0..=100.0).contains(&def),
        "default UI value {def} in range"
    );
}

#[test]
fn fov_is_a_degrees_slider() {
    let s = slider(SettingKey::Fov).expect("a slider");
    assert_eq!(s.range, (50.0, 100.0));
    assert!(options(SettingKey::Fov).is_none());
    for &deg in &[50.0_f32, 75.0, 100.0] {
        let stored = s.apply(deg);
        assert!((stored - deg).abs() < 1.0e-6);
        assert!(((s.recover)(stored) - deg).abs() < 1.0e-6);
    }
    assert_eq!(s.apply(10.0), 50.0);
    assert_eq!(s.apply(200.0), 100.0);
    assert_eq!((s.format)(74.6), "75\u{00b0}");
    assert!((50.0..=100.0).contains(&DEFAULT_FOV));
}

#[test]
fn slider_value_and_fraction_round_trip_and_clamp() {
    let exposure = slider(SettingKey::Exposure).unwrap();
    assert_eq!(exposure.value_at(0.0), -3.0);
    assert_eq!(exposure.value_at(1.0), 3.0);
    assert_eq!(exposure.value_at(0.5), 0.0);
    assert_eq!(exposure.fraction(-100.0), 0.0);
    assert_eq!(exposure.fraction(100.0), 1.0);
    assert_eq!(exposure.fraction(0.0), 0.5);
    let format = exposure.format;
    assert_eq!(format(0.0), "+0.0 EV");
    assert_eq!(format(1.5), "+1.5 EV");
    assert_eq!(format(-2.0), "-2.0 EV");
}

// Every slider key is listed once, resolves to its own entry, spans a non-empty
// range its fraction mapping round-trips, and is not a cycle row; every
// slider-kind setting has an entry.
#[test]
fn sliders_have_unique_keys_and_valid_ranges() {
    for key in SettingKey::ALL {
        let entries = SLIDERS.iter().filter(|s| s.key == key).count();
        let expected = usize::from(key.kind() == SettingKind::Slider);
        assert_eq!(entries, expected, "{key:?} SLIDERS entries");
    }
    let mut seen = hashbrown::HashSet::new();
    for s in &SLIDERS {
        assert!(seen.insert(s.key), "{:?} is listed twice", s.key);
        assert!(core::ptr::eq(slider(s.key).unwrap(), s), "{:?}", s.key);
        assert!(s.range.0 < s.range.1, "{:?} range must be non-empty", s.key);
        assert!(
            options(s.key).is_none(),
            "{:?} should not be a cycle row",
            s.key
        );
        for &f in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let back = s.fraction(s.value_at(f));
            assert!((back - f).abs() < 1.0e-5, "{:?}: f={f} -> {back}", s.key);
        }
    }
}

// Applying a slider value then recovering it returns the same value, so the
// handle never jumps when a persisted choice is re-applied at the next launch.
#[test]
fn slider_apply_and_recover_round_trip() {
    for s in &SLIDERS {
        let (lo, hi) = s.range;
        for v in [lo, (lo + hi) * 0.5, hi] {
            let recovered = (s.recover)(s.apply(v));
            assert!(
                (recovered - v).abs() < 1.0e-3,
                "{:?}: v={v} recovered={recovered}",
                s.key
            );
        }
    }
}

// A distinct value written through every entry's accessors reads back
// unchanged, so no two keys share a live or persisted field.
#[test]
fn no_two_sliders_alias_one_field() {
    let mut graphics = ResolvedGraphics::default();
    let mut cfg = StoredSettings::default();
    let marker = |i: usize| 1000.0 + i as f32;
    for (i, s) in SLIDERS.iter().enumerate() {
        match &s.target {
            SliderTarget::Graphics(row) => {
                *(row.persisted.get_mut)(&mut cfg.graphics) = Some(marker(i));
                *(row.field.get_mut)(&mut graphics) = marker(i);
            }
            SliderTarget::Controls { persisted, .. } => {
                *(persisted.get_mut)(&mut cfg) = Some(marker(i));
            }
        }
    }
    for (i, s) in SLIDERS.iter().enumerate() {
        let (live, persisted) = match &s.target {
            SliderTarget::Graphics(row) => (
                Some(*(row.field.get)(&graphics)),
                *(row.persisted.get)(&cfg.graphics),
            ),
            SliderTarget::Controls { persisted, .. } => (None, *(persisted.get)(&cfg)),
        };
        if let Some(v) = live {
            assert_eq!(v, marker(i), "{:?} shares its live field", s.key);
        }
        assert_eq!(
            persisted,
            Some(marker(i)),
            "{:?} shares its persisted field",
            s.key
        );
    }
}

// Render sliders persist the user-facing value; controls sliders persist the
// applied value the camera and input sampling read.
#[test]
fn persist_keeps_ui_values_for_render_sliders_and_applied_values_for_controls() {
    let mut cfg = StoredSettings::default();
    slider(SettingKey::Exposure).unwrap().persist(&mut cfg, 2.0);
    assert_eq!(cfg.graphics.exposure_ev, Some(2.0));
    let mouse = slider(SettingKey::MouseSensitivity).unwrap();
    mouse.persist(&mut cfg, 100.0);
    assert_eq!(cfg.controls.mouse_sensitivity, Some(mouse.apply(100.0)));
    slider(SettingKey::Fov).unwrap().persist(&mut cfg, 200.0);
    assert_eq!(cfg.graphics.fov, Some(100.0));
}

#[test]
fn current_value_reads_live_state_or_the_persisted_controls() {
    let mut graphics = ResolvedGraphics::default();
    graphics.exposure.ev = 2.0;
    graphics.indirect_lighting.ambient_intensity = 1.5;
    let mut cfg = StoredSettings::default();
    cfg.controls.gamepad_deadzone = Some(0.25);
    let read = |key: SettingKey| slider(key).unwrap().current_value(&graphics, &cfg);
    assert_eq!(read(SettingKey::Exposure), 2.0);
    assert_eq!(read(SettingKey::AmbientIntensity), 1.5);
    assert_eq!(read(SettingKey::GamepadDeadzone), 25.0);
    assert_eq!(read(SettingKey::Fov), DEFAULT_FOV);
}

#[test]
fn slider_apply_clamps_match_resolve() {
    let apply = |key: SettingKey, v: f32| slider(key).unwrap().apply(v);
    assert_eq!(apply(SettingKey::BloomIntensity, -5.0), 0.0);
    assert_eq!(apply(SettingKey::Vignette, 2.0), 1.0);
    assert_eq!(apply(SettingKey::LutStrength, -1.0), 0.0);
    assert_eq!(apply(SettingKey::AmbientIntensity, 100.0), 16.0);
    assert_eq!(apply(SettingKey::Exposure, 2.0), 2.0);
    assert_eq!(apply(SettingKey::Exposure, -100.0), -EV_LIMIT);
    assert_eq!(apply(SettingKey::BloomKnee, -1.0), 0.0);
    assert_eq!(apply(SettingKey::SsaoIntensity, 100.0), 4.0);
    assert_eq!(apply(SettingKey::SsrIntensity, 9.0), 1.0);
    assert_eq!(apply(SettingKey::SsrMaxDistance, 1.0e6), 200.0);
    assert_eq!(apply(SettingKey::SsgiIntensity, 99.0), 4.0);
    assert_eq!(apply(SettingKey::SsgiMaxDistance, 1.0e6), 100.0);
    assert_eq!(apply(SettingKey::AutoExposureMinEv, -100.0), -EV_LIMIT);
    assert_eq!(apply(SettingKey::AutoExposureMaxEv, 100.0), EV_LIMIT);
    assert_eq!(apply(SettingKey::AutoExposureSpeed, 100.0), 20.0);
}

#[test]
fn graphics_sliders_point_at_their_own_resolver_rows_outside_the_preset() {
    let mut rows = Vec::new();
    for s in &SLIDERS {
        if let SliderTarget::Graphics(row) = &s.target {
            assert_eq!(row.key, s.key);
            assert!(quality_cycle(row.key).is_none(), "{:?}", row.key);
            rows.push(row.key);
        }
    }
    let resolver: Vec<SettingKey> = GRAPHICS_SLIDERS.iter().map(|r| r.key).collect();
    assert_eq!(rows, resolver);
}

#[test]
fn strength_sliders_format_as_percent() {
    let format = |key: SettingKey, v: f32| (slider(key).unwrap().format)(v);
    assert_eq!(format(SettingKey::Vignette, 0.0), "0%");
    assert_eq!(format(SettingKey::Vignette, 0.5), "50%");
    assert_eq!(format(SettingKey::LutStrength, 1.0), "100%");
    assert_eq!(format(SettingKey::BloomIntensity, 0.6), "0.60");
    assert_eq!(format(SettingKey::AmbientIntensity, 1.25), "1.25");
}
