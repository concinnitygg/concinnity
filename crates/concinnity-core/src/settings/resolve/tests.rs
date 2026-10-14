use super::*;
use crate::components::{AaMode, IndirectTechnique, ShadowUpdate};
use crate::ecs::asset_fields::ApplyClass;
use crate::render::backend::{GpuProfile, GpuTier};
use crate::settings::quality::{QualityPreset, resolve_ceiling};
use alloc::vec::Vec;

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
// settles a world that declares none of the settings assets.
#[test]
fn the_low_ceiling_clamps_the_schema_defaults_off() {
    let mut cfg = QualityEffects::default();
    clamp_quality_under_ceiling(
        &mut cfg,
        &UserGraphicsSettings::default(),
        &ceiling_for(QualityPreset::Low, GpuTier::Integrated),
    );
    assert!(!cfg.ambient_occlusion.enabled);
    assert!(!cfg.reflections.screen_space);
    assert!(!cfg.reflections.ray_traced);
    assert_eq!(cfg.indirect_lighting.technique, IndirectTechnique::Ibl);
    assert_eq!(cfg.anti_aliasing.mode, AaMode::Fxaa);
}

#[test]
fn the_top_ceiling_leaves_the_schema_defaults_alone() {
    let mut cfg = QualityEffects::default();
    clamp_quality_under_ceiling(
        &mut cfg,
        &UserGraphicsSettings::default(),
        &ceiling_for(QualityPreset::Ultra, GpuTier::HighDiscrete),
    );
    assert!(cfg.ambient_occlusion.enabled);
    assert!(cfg.reflections.screen_space);
    assert!(cfg.reflections.ray_traced);
    assert_eq!(cfg.indirect_lighting.technique, IndirectTechnique::Ssgi);
    assert_eq!(cfg.anti_aliasing.mode, AaMode::Taa);
}

// Effects with every quality-governed feature authored off.
fn all_off() -> QualityEffects {
    QualityEffects {
        ambient_occlusion: AmbientOcclusion {
            enabled: false,
            ..Default::default()
        },
        reflections: Reflections {
            screen_space: false,
            ray_traced: false,
            ..Default::default()
        },
        indirect_lighting: IndirectLighting {
            technique: IndirectTechnique::Ibl,
            ..Default::default()
        },
        anti_aliasing: AntiAliasing {
            mode: AaMode::Off,
            ..Default::default()
        },
        exposure: Exposure::default(),
    }
}

// A ceiling only reduces: what the world turned off stays off at any tier.
#[test]
fn a_ceiling_never_turns_a_feature_back_on() {
    let mut cfg = all_off();
    clamp_quality_under_ceiling(
        &mut cfg,
        &UserGraphicsSettings::default(),
        &ceiling_for(QualityPreset::Ultra, GpuTier::HighDiscrete),
    );
    assert!(!cfg.ambient_occlusion.enabled);
    assert!(!cfg.reflections.screen_space);
    assert!(!cfg.reflections.ray_traced);
    assert_eq!(cfg.indirect_lighting.technique, IndirectTechnique::Ibl);
    assert_eq!(cfg.anti_aliasing.mode, AaMode::Off);
}

// An explicit per-row choice survives a ceiling that would have cleared it.
#[test]
fn a_user_override_wins_over_the_ceiling() {
    let user = UserGraphicsSettings {
        ssao: Some(true),
        aa_mode: Some(AaMode::Taa),
        ..UserGraphicsSettings::default()
    };
    let mut cfg = QualityEffects::default();
    cfg.ambient_occlusion.enabled = false;
    cfg.anti_aliasing.mode = AaMode::Off;
    overlay_quality_overrides(&mut cfg, &user);
    assert!(
        cfg.ambient_occlusion.enabled,
        "the override applies over the world's value"
    );
    clamp_quality_under_ceiling(
        &mut cfg,
        &user,
        &ceiling_for(QualityPreset::Low, GpuTier::Integrated),
    );
    assert!(
        cfg.ambient_occlusion.enabled,
        "the Low ceiling does not clear an explicit choice"
    );
    assert_eq!(cfg.anti_aliasing.mode, AaMode::Taa);
    // A row the user left alone still clamps.
    assert!(!cfg.reflections.screen_space);
}

fn low() -> QualityCeiling {
    ceiling_for(QualityPreset::Low, GpuTier::Integrated)
}

#[test]
fn a_user_override_beats_the_world_value() {
    let world = WorldSettings::new(WorldAssets {
        frame_pacing: Some(&FramePacing {
            vsync: true,
            fps_cap: 60,
            ..Default::default()
        }),
        display: Some(&Display {
            hdr: true,
            ..Default::default()
        }),
        anti_aliasing: Some(&AntiAliasing {
            upscaler: UpscalerBackend::Fsr3,
            ..Default::default()
        }),
        ..Default::default()
    });
    let user = UserGraphicsSettings {
        vsync: Some(false),
        fps_cap: Some(30),
        hdr_display: Some(false),
        upscale_backend: Some(UpscalerBackend::Auto),
        dlss_preset: Some(DlssPreset::L),
        perf_stats: Some(false),
        ..UserGraphicsSettings::default()
    };
    let r = resolve_graphics(&world, &user, &low());
    assert!(!r.vsync);
    assert_eq!(r.fps_cap, 30);
    assert!(!r.hdr_display);
    assert_eq!(r.upscale_backend, UpscalerBackend::Auto);
    assert_eq!(r.dlss_preset, DlssPreset::L);
    assert!(!r.perf_stats);
    assert!(r.show_fps, "an untouched stats toggle defaults on");

    let r = resolve_graphics(&world, &UserGraphicsSettings::default(), &low());
    assert!(r.vsync);
    assert_eq!(r.fps_cap, 60);
    assert!(r.hdr_display);
    assert_eq!(r.upscale_backend, UpscalerBackend::Fsr3);
    assert_eq!(r.dlss_preset, DlssPreset::Default);
}

#[test]
fn the_ceiling_clamps_only_a_knob_the_user_left_alone() {
    let world = WorldSettings::new(WorldAssets {
        shadows: Some(&Shadows {
            map_size: 4096,
            update: ShadowUpdate::EveryFrame,
            distance: 200,
            cascades: 4,
        }),
        texture_filtering: Some(&TextureFiltering { anisotropy: 16 }),
        ..Default::default()
    });
    let q = resolve_graphics(&world, &UserGraphicsSettings::default(), &low()).quality;
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

    let user = UserGraphicsSettings {
        shadow_map_size: Some(4096),
        shadow_distance: Some(200),
        ..UserGraphicsSettings::default()
    };
    let q = resolve_graphics(&world, &user, &low()).quality;
    assert_eq!(q.shadow_map_size, 4096);
    assert_eq!(q.shadow_cadence.distance, 200);
    assert_eq!(q.shadow_cadence.cascades, 2);
}

#[test]
fn a_frames_in_flight_override_clamps_to_three() {
    let world = WorldSettings::new(WorldAssets {
        frame_pacing: Some(&FramePacing {
            frames_in_flight: 7,
            ..Default::default()
        }),
        ..Default::default()
    });
    let r = resolve_graphics(&world, &UserGraphicsSettings::default(), &low());
    assert_eq!(r.frames_in_flight, 7, "the world value is not clamped");
    let user = UserGraphicsSettings {
        frames_in_flight: Some(7),
        ..UserGraphicsSettings::default()
    };
    assert_eq!(resolve_graphics(&world, &user, &low()).frames_in_flight, 3);
}

#[test]
fn render_scale_takes_the_more_aggressive_of_world_and_ceiling() {
    let scale = |world: UpscaleQuality, user: Option<UpscaleQuality>| {
        let world = WorldSettings::new(WorldAssets {
            anti_aliasing: Some(&AntiAliasing {
                upscale_quality: world,
                ..Default::default()
            }),
            ..Default::default()
        });
        let user = UserGraphicsSettings {
            render_scale: user,
            ..UserGraphicsSettings::default()
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
fn occlusion_two_pass_is_off_without_a_culling() {
    let none = UserGraphicsSettings::default();
    let r = resolve_graphics(&WorldSettings::default(), &none, &low());
    assert!(!r.occlusion_two_pass);
    let declared = WorldSettings::new(WorldAssets {
        culling: Some(&Culling::default()),
        ..Default::default()
    });
    assert!(resolve_graphics(&declared, &none, &low()).occlusion_two_pass);
}

#[test]
fn texture_overrides_apply_only_where_the_world_streams() {
    let user = UserGraphicsSettings {
        texture_cap: Some(10),
        texture_budget: Some(1),
        ..UserGraphicsSettings::default()
    };
    let defaults = StreamingConfig::default();
    let r = resolve_graphics(&WorldSettings::default(), &user, &low());
    assert_eq!(
        (r.texture_cap, r.texture_budget),
        (defaults.texture_cap, defaults.texture_budget)
    );
    let streaming = WorldSettings::new(WorldAssets {
        streaming: Some(&defaults),
        ..Default::default()
    });
    let r = resolve_graphics(&streaming, &user, &low());
    assert_eq!((r.texture_cap, r.texture_budget), (10, 1));
}

#[test]
fn the_fxaa_flag_follows_the_clamped_aa_mode() {
    let world = WorldSettings::default();
    assert_eq!(world.effects.anti_aliasing.mode, AaMode::Taa);
    let r = resolve_graphics(&world, &UserGraphicsSettings::default(), &low());
    assert_eq!(r.quality.effects.anti_aliasing.mode, AaMode::Fxaa);
    assert_eq!(r.post_process.fxaa, AaMode::Fxaa.fxaa_flag());
}

// Look tuning is not preset-governed, so a cleared set of Quality-row
// overrides still carries the user's sub-quality sliders.
#[test]
fn the_quality_group_keeps_the_look_tuning_sliders() {
    let world = WorldSettings::new(WorldAssets {
        ambient_occlusion: Some(&AmbientOcclusion {
            radius: 0.5,
            ..Default::default()
        }),
        ..Default::default()
    });
    let user = UserGraphicsSettings {
        ssao_radius: Some(1.5),
        ..UserGraphicsSettings::default()
    };
    let q = resolve_quality(&world, &user, &low());
    assert_eq!(q.effects.ambient_occlusion.radius, 1.5);
}

// Each layer beats the one below it: a scene beats the world, the ceiling
// clamps what the scene authored, the user's choice beats the ceiling, and a
// session override beats the user.
#[test]
fn each_layer_beats_the_one_below_it() {
    let world = WorldSettings::new(WorldAssets {
        shadows: Some(&Shadows {
            distance: 60,
            ..Default::default()
        }),
        frame_pacing: Some(&FramePacing {
            fps_cap: 120,
            ..Default::default()
        }),
        ..Default::default()
    });
    let scene = SceneOverride {
        shadows: Some(Shadows {
            distance: 300,
            ..Default::default()
        }),
        frame_pacing: Some(FramePacing {
            fps_cap: 90,
            ..Default::default()
        }),
        ..Default::default()
    };
    let none = UserGraphicsSettings::default();
    let mut layers = SettingsLayers::new(&world, &QualityCeiling::NONE, &none);
    let base = layers.resolve();
    assert_eq!(base.quality.shadow_cadence.distance, 60);
    assert_eq!(base.fps_cap, 120);

    layers.scene = Some(&scene);
    let scened = layers.resolve();
    assert_eq!(scened.quality.shadow_cadence.distance, 300);
    assert_eq!(scened.fps_cap, 90);

    let low = low();
    layers.ceiling = &low;
    assert_eq!(layers.resolve().quality.shadow_cadence.distance, 40);

    let user = UserGraphicsSettings {
        shadow_distance: Some(200),
        fps_cap: Some(60),
        ..Default::default()
    };
    layers.user = &user;
    let chosen = layers.resolve();
    assert_eq!(chosen.quality.shadow_cadence.distance, 200);
    assert_eq!(chosen.fps_cap, 60);

    let session = UserGraphicsSettings {
        fps_cap: Some(0),
        ..Default::default()
    };
    layers.session = Some(&session);
    let sessioned = layers.resolve();
    assert_eq!(sessioned.fps_cap, 0);
    assert_eq!(sessioned.quality.shadow_cadence.distance, 200);
}

// A scene's asset replaces the world's whole, and counts as declared.
#[test]
fn a_scene_asset_counts_as_declared() {
    let world = WorldSettings::default();
    let none = UserGraphicsSettings::default();
    let mut layers = SettingsLayers::new(&world, &QualityCeiling::NONE, &none);
    assert!(!layers.resolve().occlusion_two_pass);
    let scene = SceneOverride {
        culling: Some(Culling::default()),
        ambient_occlusion: Some(AmbientOcclusion {
            enabled: false,
            ..Default::default()
        }),
        ..Default::default()
    };
    layers.scene = Some(&scene);
    let r = layers.resolve();
    assert!(r.occlusion_two_pass);
    assert!(!r.quality.effects.ambient_occlusion.enabled);
    assert!(!layers.resolve_quality().effects.ambient_occlusion.enabled);
}

// The device clamps after every layer: an explicit user choice cannot turn on
// what the hardware cannot run.
#[test]
fn capabilities_clamp_after_every_layer() {
    let world = WorldSettings::default();
    let user = UserGraphicsSettings {
        ray_traced_reflections: Some(true),
        upscale_backend: Some(UpscalerBackend::Dlss),
        ..Default::default()
    };
    let mut layers = SettingsLayers::new(&world, &QualityCeiling::NONE, &user);
    let capable = layers.resolve();
    assert!(capable.quality.effects.reflections.ray_traced);
    assert_eq!(capable.upscale_backend, UpscalerBackend::Dlss);

    let caps = DeviceCapabilities {
        ray_tracing: false,
        selectable_upscaler: false,
        ..DeviceCapabilities::ALL
    };
    layers.capabilities = &caps;
    let r = layers.resolve();
    assert!(!r.quality.effects.reflections.ray_traced);
    assert!(r.quality_settings().rt_reflections.is_none());
    assert_eq!(r.upscale_backend, UpscalerBackend::Auto);
    assert!(
        r.quality.effects.reflections.screen_space,
        "screen-space reflections stay"
    );
    assert!(!layers.resolve_quality().effects.reflections.ray_traced);
}

#[test]
fn an_unchanged_snapshot_has_an_empty_diff() {
    let r = ResolvedSettings::default();
    let diff = r.diff(&r.clone());
    assert!(diff.is_empty());
    assert_eq!(diff.costliest(), None);
    assert_eq!(diff.changes().count(), 0);
}

// Each change lands under the class its authored field declares, once, even
// when two resolved values name the same field.
#[test]
fn the_diff_groups_changes_by_apply_class() {
    let world = WorldSettings::default();
    let none = UserGraphicsSettings::default();
    let before = SettingsLayers::new(&world, &QualityCeiling::NONE, &none).resolve();
    let user = UserGraphicsSettings {
        bloom_intensity: Some(1.5),
        ssao_radius: Some(1.25),
        ssao: Some(false),
        aa_mode: Some(AaMode::Fxaa),
        shadow_distance: Some(20),
        shadow_map_size: Some(1024),
        hdr_display: Some(true),
        dlss_preset: Some(DlssPreset::M),
        show_fps: Some(false),
        ..Default::default()
    };
    let after = SettingsLayers::new(&world, &QualityCeiling::NONE, &user).resolve();
    let diff = after.diff(&before);
    let keys = |class| -> Vec<(&str, &str)> {
        diff.class(class).iter().map(|c| (c.asset, c.key)).collect()
    };
    assert_eq!(
        keys(ApplyClass::Live),
        [
            ("AmbientOcclusion", "radius"),
            ("Bloom", "intensity"),
            ("Shadows", "distance"),
            ("UserGraphicsSettings", "show_fps"),
        ]
    );
    assert_eq!(
        keys(ApplyClass::Rebuild),
        [("AmbientOcclusion", "enabled"), ("AntiAliasing", "mode")]
    );
    assert!(keys(ApplyClass::Variant).is_empty());
    assert_eq!(
        keys(ApplyClass::Restart),
        [
            ("Display", "hdr"),
            ("Shadows", "map_size"),
            ("UserGraphicsSettings", "dlss_preset"),
        ]
    );
    assert_eq!(diff.costliest(), Some(ApplyClass::Restart));
}

// Every resolved value the diff names is a field that declares its class, so
// none falls back to the restart default by omission.
#[test]
fn every_diffed_field_declares_its_apply_class() {
    use crate::components::ColorLut;
    use crate::ecs::asset_fields::AssetFields;
    fn declares<A: AssetFields>(keys: &[&str]) {
        for key in keys {
            assert!(A::param(key).and_then(|m| m.apply).is_some(), "{key}");
        }
    }
    declares::<AntiAliasing>(&["upscale_quality", "mode", "upscaling", "upscaler"]);
    declares::<Exposure>(&["ev"]);
    declares::<Bloom>(&["intensity", "threshold", "knee"]);
    declares::<Vignette>(&["strength"]);
    declares::<ColorLut>(&["strength"]);
    declares::<IndirectLighting>(&["ambient_intensity"]);
    declares::<Display>(&["hdr", "pq"]);
    declares::<Culling>(&["two_pass_occlusion"]);
    declares::<Shadows>(&["map_size", "update", "distance", "cascades"]);
    declares::<TextureFiltering>(&["anisotropy"]);
    declares::<FramePacing>(&["vsync", "fps_cap", "frames_in_flight"]);
    declares::<StreamingConfig>(&["texture_cap", "texture_budget"]);
}

#[test]
fn quality_settings_follow_the_resolved_config() {
    let world = WorldSettings::default();
    let none = UserGraphicsSettings::default();
    let r = SettingsLayers::new(&world, &low(), &none).resolve();
    let q = r.quality_settings();
    assert!(!q.taa, "the Low ceiling caps TAA to FXAA");
    assert!(q.ssao.is_none() && q.ssr.is_none() && q.ssgi.is_none());
    let r = SettingsLayers::new(&world, &QualityCeiling::NONE, &none).resolve();
    let q = r.quality_settings();
    assert!(q.taa);
    assert!(q.ssao.is_some() && q.ssr.is_some() && q.rt_reflections.is_some());
}
