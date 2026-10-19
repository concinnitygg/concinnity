use super::*;
use crate::components::{
    AaMode, AmbientOcclusion, AmbientOcclusionOverrides, AntiAliasing, AntiAliasingOverrides,
    Bloom, Culling, Display, DisplayOverrides, Exposure, FramePacing, FramePacingOverrides,
    IndirectLighting, IndirectTechnique, PassResolution, Reflections, ShadowUpdate, Shadows,
    ShadowsOverrides, TextureFiltering, TextureFilteringOverrides, UpscaleQuality, Vignette,
};
use crate::ecs::asset_fields::ApplyClass;
use crate::render::backend::{GpuProfile, GpuTier};
use crate::render::dlss::DlssPreset;
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
    let mut cfg = ResolvedGraphics::default();
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
    let mut cfg = ResolvedGraphics::default();
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
fn all_off() -> ResolvedGraphics {
    ResolvedGraphics {
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
        ..Default::default()
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
    let mut cfg = ResolvedGraphics::default();
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

// The world a sparse `GraphicsConfig` authors.
fn authored(graphics: GraphicsConfig) -> WorldSettings {
    WorldSettings::new(WorldAssets {
        graphics: Some(&graphics),
        ..Default::default()
    })
}

#[test]
fn a_user_override_beats_the_world_value() {
    let world = authored(GraphicsConfig {
        frame_pacing: FramePacingOverrides {
            vsync: Some(true),
            fps_cap: Some(60),
            ..Default::default()
        },
        display: DisplayOverrides {
            hdr: Some(true),
            ..Default::default()
        },
        anti_aliasing: AntiAliasingOverrides {
            upscaler: Some(UpscalerBackend::Fsr3),
            ..Default::default()
        },
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
    let r = resolve_settings(&world, &user, &low());
    assert!(!r.graphics.frame_pacing.vsync);
    assert_eq!(r.graphics.frame_pacing.fps_cap, 30);
    assert!(!r.graphics.display.hdr);
    assert_eq!(r.graphics.anti_aliasing.upscaler, UpscalerBackend::Auto);
    assert_eq!(r.graphics.dlss_preset, DlssPreset::L);
    assert!(!r.perf_stats);
    assert!(r.show_fps, "an untouched stats toggle defaults on");

    let r = resolve_settings(&world, &UserGraphicsSettings::default(), &low());
    assert!(r.graphics.frame_pacing.vsync);
    assert_eq!(r.graphics.frame_pacing.fps_cap, 60);
    assert!(r.graphics.display.hdr);
    assert_eq!(r.graphics.anti_aliasing.upscaler, UpscalerBackend::Fsr3);
    assert_eq!(r.graphics.dlss_preset, DlssPreset::Default);
}

#[test]
fn the_ceiling_clamps_only_a_knob_the_user_left_alone() {
    let world = authored(GraphicsConfig {
        shadows: ShadowsOverrides::from(Shadows {
            map_size: 4096,
            update: ShadowUpdate::EveryFrame,
            distance: 200,
            cascades: 4,
        }),
        texture_filtering: TextureFilteringOverrides {
            anisotropy: Some(16),
        },
        ..Default::default()
    });
    let q = resolve_settings(&world, &UserGraphicsSettings::default(), &low()).graphics;
    assert_eq!(q.shadows.map_size, 1024);
    assert_eq!(
        q.shadow_cadence(),
        ShadowCadence {
            update: ShadowUpdate::Hybrid,
            distance: 40,
            cascades: 2,
        }
    );
    assert_eq!(q.texture_filtering.anisotropy, 4);

    let user = UserGraphicsSettings {
        shadow_map_size: Some(4096),
        shadow_distance: Some(200),
        ..UserGraphicsSettings::default()
    };
    let q = resolve_settings(&world, &user, &low()).graphics;
    assert_eq!(q.shadows.map_size, 4096);
    assert_eq!(q.shadows.distance, 200);
    assert_eq!(q.shadows.cascades, 2);
}

#[test]
fn a_frames_in_flight_override_clamps_to_three() {
    let world = authored(GraphicsConfig {
        frame_pacing: FramePacingOverrides {
            frames_in_flight: Some(7),
            ..Default::default()
        },
        ..Default::default()
    });
    let r = resolve_settings(&world, &UserGraphicsSettings::default(), &low());
    assert_eq!(
        r.graphics.frame_pacing.frames_in_flight, 7,
        "the world value is not clamped"
    );
    let user = UserGraphicsSettings {
        frames_in_flight: Some(7),
        ..UserGraphicsSettings::default()
    };
    assert_eq!(
        resolve_settings(&world, &user, &low())
            .graphics
            .frame_pacing
            .frames_in_flight,
        3
    );
}

#[test]
fn render_scale_takes_the_more_aggressive_of_world_and_ceiling() {
    let scale = |world: UpscaleQuality, user: Option<UpscaleQuality>| {
        let world = authored(GraphicsConfig {
            anti_aliasing: AntiAliasingOverrides {
                upscale_quality: Some(world),
                ..Default::default()
            },
            ..Default::default()
        });
        let user = UserGraphicsSettings {
            render_scale: user,
            ..UserGraphicsSettings::default()
        };
        resolve_settings(&world, &user, &low())
            .graphics
            .anti_aliasing
            .upscale_quality
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

// Absence is the section default: an empty `GraphicsConfig`, none at all, and
// every field set to its default resolve alike.
#[test]
fn an_unset_field_resolves_like_its_default_set_explicitly() {
    let none = UserGraphicsSettings::default();
    let every = GraphicsConfig {
        culling: Culling::default().into(),
        anti_aliasing: AntiAliasing::default().into(),
        shadows: Shadows::default().into(),
        ..Default::default()
    };
    for ceiling in [low(), QualityCeiling::NONE] {
        let bare = resolve_settings(&WorldSettings::default(), &none, &ceiling);
        let empty = resolve_settings(&authored(GraphicsConfig::default()), &none, &ceiling);
        let set = resolve_settings(&authored(every.clone()), &none, &ceiling);
        assert!(bare.graphics.culling.two_pass_occlusion);
        for r in [&empty, &set] {
            assert_eq!(
                r.graphics.culling.two_pass_occlusion,
                bare.graphics.culling.two_pass_occlusion
            );
            assert_eq!(
                r.graphics.anti_aliasing.upscale_quality,
                bare.graphics.anti_aliasing.upscale_quality
            );
            assert_eq!(r.graphics.shadows.map_size, bare.graphics.shadows.map_size);
            assert_eq!(r.graphics.shadow_cadence(), bare.graphics.shadow_cadence());
        }
    }
}

#[test]
fn texture_overrides_apply_only_where_the_world_streams() {
    let user = UserGraphicsSettings {
        texture_cap: Some(10),
        texture_budget: Some(1),
        ..UserGraphicsSettings::default()
    };
    let defaults = StreamingConfig::default();
    let r = resolve_settings(&WorldSettings::default(), &user, &low());
    assert_eq!(
        (r.texture_cap, r.texture_budget),
        (defaults.texture_cap, defaults.texture_budget)
    );
    let streaming = WorldSettings::new(WorldAssets {
        streaming: Some(&defaults),
        ..Default::default()
    });
    let r = resolve_settings(&streaming, &user, &low());
    assert_eq!((r.texture_cap, r.texture_budget), (10, 1));
}

// Without a `ColorLut` there is nothing to grade, so the composite skips the
// LUT; with one, its own strength blends it in.
#[test]
fn the_lut_strength_comes_from_the_color_lut() {
    let none = WorldSettings::new(WorldAssets::default());
    assert_eq!(none.post_process().lut_strength, 0.0);
    let graded = WorldSettings::new(WorldAssets {
        lut_strength: Some(0.6),
        ..Default::default()
    });
    assert_eq!(graded.post_process().lut_strength, 0.6);
}

#[test]
fn the_fxaa_flag_follows_the_clamped_aa_mode() {
    let world = WorldSettings::default();
    assert_eq!(world.graphics.anti_aliasing.mode, AaMode::Taa);
    let r = resolve_settings(&world, &UserGraphicsSettings::default(), &low());
    assert_eq!(r.graphics.anti_aliasing.mode, AaMode::Fxaa);
    assert_eq!(r.graphics.post_process().fxaa, AaMode::Fxaa.fxaa_flag());
}

// Look tuning is not preset-governed, so a cleared set of Quality-row
// overrides still carries the user's sub-quality sliders.
#[test]
fn the_quality_group_keeps_the_look_tuning_sliders() {
    let world = authored(GraphicsConfig {
        ambient_occlusion: AmbientOcclusionOverrides {
            radius: Some(0.5),
            ..Default::default()
        },
        ..Default::default()
    });
    let user = UserGraphicsSettings {
        ssao_radius: Some(1.5),
        ..UserGraphicsSettings::default()
    };
    let q = resolve_quality(&world, &user, &low());
    assert_eq!(q.ambient_occlusion.radius, 1.5);
}

// Each layer beats the one below it: a scene beats the world, the ceiling
// clamps what the scene authored, the user's choice beats the ceiling, and a
// session override beats the user.
#[test]
fn each_layer_beats_the_one_below_it() {
    let world = authored(GraphicsConfig {
        shadows: ShadowsOverrides {
            distance: Some(60),
            ..Default::default()
        },
        frame_pacing: FramePacingOverrides {
            fps_cap: Some(120),
            ..Default::default()
        },
        ..Default::default()
    });
    let scene = SceneOverride {
        graphics: GraphicsConfig {
            shadows: ShadowsOverrides {
                distance: Some(300),
                ..Default::default()
            },
            frame_pacing: FramePacingOverrides {
                fps_cap: Some(90),
                ..Default::default()
            },
            ..Default::default()
        },
    };
    let none = UserGraphicsSettings::default();
    let mut layers = SettingsLayers::new(&world, &QualityCeiling::NONE, &none);
    let base = layers.resolve();
    assert_eq!(base.graphics.shadows.distance, 60);
    assert_eq!(base.graphics.frame_pacing.fps_cap, 120);

    layers.scene = Some(&scene);
    let scened = layers.resolve();
    assert_eq!(scened.graphics.shadows.distance, 300);
    assert_eq!(scened.graphics.frame_pacing.fps_cap, 90);

    let low = low();
    layers.ceiling = &low;
    assert_eq!(layers.resolve().graphics.shadows.distance, 40);

    let user = UserGraphicsSettings {
        shadow_distance: Some(200),
        fps_cap: Some(60),
        ..Default::default()
    };
    layers.user = &user;
    let chosen = layers.resolve();
    assert_eq!(chosen.graphics.shadows.distance, 200);
    assert_eq!(chosen.graphics.frame_pacing.fps_cap, 60);

    let session = UserGraphicsSettings {
        fps_cap: Some(0),
        ..Default::default()
    };
    layers.session = Some(&session);
    let sessioned = layers.resolve();
    assert_eq!(sessioned.graphics.frame_pacing.fps_cap, 0);
    assert_eq!(sessioned.graphics.shadows.distance, 200);
}

// A scene replaces only the fields it sets; the world's other fields, set or
// not, show through.
#[test]
fn a_scene_replaces_only_the_fields_it_sets() {
    let world = authored(GraphicsConfig {
        ambient_occlusion: AmbientOcclusionOverrides {
            radius: Some(2.0),
            ..Default::default()
        },
        ..Default::default()
    });
    let none = UserGraphicsSettings::default();
    let mut layers = SettingsLayers::new(&world, &QualityCeiling::NONE, &none);
    let scene = SceneOverride {
        graphics: GraphicsConfig {
            ambient_occlusion: AmbientOcclusionOverrides {
                enabled: Some(false),
                ..Default::default()
            },
            ..Default::default()
        },
    };
    layers.scene = Some(&scene);
    let r = layers.resolve();
    assert!(r.graphics.culling.two_pass_occlusion);
    let ao = &r.graphics.ambient_occlusion;
    assert!(!ao.enabled);
    assert_eq!(ao.radius, 2.0);
    assert!(!layers.resolve_quality().ambient_occlusion.enabled);
}

// The sparse authored layer sits between the section defaults and the
// ceiling: a field the world sets replaces the default, the ceiling caps set
// and unset fields alike, the user beats the ceiling, and the session beats
// the user.
#[test]
fn a_sparse_world_layers_between_the_defaults_and_the_ceiling() {
    let world = authored(GraphicsConfig {
        shadows: ShadowsOverrides {
            map_size: Some(8192),
            ..Default::default()
        },
        ..Default::default()
    });
    let none = UserGraphicsSettings::default();
    let mut layers = SettingsLayers::new(&world, &QualityCeiling::NONE, &none);
    let top = layers.resolve().graphics;
    assert_eq!(top.shadows.map_size, 8192, "set replaces the default");
    assert_eq!(top.shadows.distance, 80, "unset takes the default");

    let low = low();
    layers.ceiling = &low;
    let capped = layers.resolve().graphics;
    assert_eq!(
        capped.shadows.map_size, 1024,
        "the ceiling caps a set field"
    );
    assert_eq!(capped.shadows.distance, 40, "and an unset one");

    let user = UserGraphicsSettings {
        shadow_map_size: Some(2048),
        ..Default::default()
    };
    layers.user = &user;
    assert_eq!(layers.resolve().graphics.shadows.map_size, 2048);
    let session = UserGraphicsSettings {
        shadow_map_size: Some(512),
        ..Default::default()
    };
    layers.session = Some(&session);
    assert_eq!(layers.resolve().graphics.shadows.map_size, 512);
}

// With nothing authored, each tier's resolution of a quality-governed field
// is the section default capped by that tier.
#[test]
fn an_unset_world_resolves_to_the_tier_capped_defaults() {
    let none = UserGraphicsSettings::default();
    let tiers = [
        (GpuTier::Integrated, QualityCeiling::LOW),
        (GpuTier::Unknown, QualityCeiling::MEDIUM),
        (GpuTier::EntryDiscrete, QualityCeiling::MEDIUM),
        (GpuTier::MidDiscrete, QualityCeiling::HIGH),
        (GpuTier::HighDiscrete, QualityCeiling::ULTRA),
    ];
    for (tier, cap) in tiers {
        assert_eq!(ceiling_for(QualityPreset::Auto, tier), cap);
        let r = resolve_settings(&WorldSettings::default(), &none, &cap);
        let q = &r.graphics;
        let fx = q;
        assert_eq!(q.shadows.map_size, 4096.min(cap.shadow_map_size));
        let update = match cap.allow_every_frame_shadows {
            true => ShadowUpdate::EveryFrame,
            false => ShadowUpdate::Hybrid,
        };
        assert_eq!(q.shadows.update, update);
        assert_eq!(q.shadows.distance, 80.min(cap.shadow_distance));
        assert_eq!(q.shadows.cascades, 4.min(cap.shadow_cascades));
        assert_eq!(q.texture_filtering.anisotropy, 16.min(cap.anisotropy));
        assert_eq!(fx.ambient_occlusion.enabled, cap.ssao);
        assert_eq!(fx.reflections.screen_space, cap.ssr);
        assert_eq!(fx.reflections.ray_traced, cap.ray_traced_reflections);
        assert_eq!(fx.reflections.ray_traced_shadows, cap.rt_reflection_shadows);
        assert_eq!(
            fx.reflections.ray_traced_resolution,
            PassResolution::Half.max(cap.rt_reflection_resolution)
        );
        assert_eq!(
            fx.reflections.blur_resolution,
            PassResolution::Half.max(cap.reflection_blur_resolution)
        );
        let technique = match cap.ssgi {
            true => IndirectTechnique::Ssgi,
            false => IndirectTechnique::Ibl,
        };
        assert_eq!(fx.indirect_lighting.technique, technique);
        assert_eq!(
            fx.indirect_lighting.ssgi_resolution,
            PassResolution::Half.max(cap.ssgi_resolution)
        );
        assert_eq!(
            fx.indirect_lighting.ssgi_rays,
            crate::components::DEFAULT_SSGI_RAYS.min(cap.ssgi_rays)
        );
        assert_eq!(fx.anti_aliasing.mode, cap.aa_mode);
        assert_eq!(
            q.anti_aliasing.upscale_quality,
            more_aggressive_upscale(UpscaleQuality::Quality, cap.min_upscale)
        );
        // Fields no tier governs resolve to their section defaults.
        assert_eq!(
            r.graphics.frame_pacing.fps_cap,
            FramePacing::default().fps_cap
        );
        assert_eq!(r.graphics.frame_pacing.frames_in_flight, 2);
        assert!(
            !r.graphics.frame_pacing.vsync
                && !r.graphics.display.hdr
                && !r.graphics.anti_aliasing.upscaling
        );
        assert!(r.graphics.culling.two_pass_occlusion);
        assert_eq!(fx.ambient_occlusion.radius, 0.5);
        assert!(!fx.exposure.auto);
    }
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
    assert!(capable.graphics.reflections.ray_traced);
    assert_eq!(
        capable.graphics.anti_aliasing.upscaler,
        UpscalerBackend::Dlss
    );

    let caps = DeviceCapabilities {
        ray_tracing: false,
        selectable_upscaler: false,
        ..DeviceCapabilities::ALL
    };
    layers.capabilities = &caps;
    let r = layers.resolve();
    assert!(!r.graphics.reflections.ray_traced);
    assert!(r.graphics.rt_reflections().is_none());
    assert_eq!(r.graphics.anti_aliasing.upscaler, UpscalerBackend::Auto);
    assert!(
        r.graphics.reflections.screen_space,
        "screen-space reflections stay"
    );
    assert!(!layers.resolve_quality().reflections.ray_traced);
}

// Ray tracing resolves off where the upscaler the request resolves to is XeSS
// and the device cannot run the two together, and stays on wherever either
// half differs.
#[test]
fn ray_tracing_gives_way_to_xess_where_they_cannot_share_the_device() {
    use crate::render::upscaler::UpscalerAvailability;
    let world = WorldSettings::default();
    let rt_with = |upscaling: bool, upscaler: UpscalerBackend, caps: DeviceCapabilities| {
        let user = UserGraphicsSettings {
            ray_traced_reflections: Some(true),
            temporal_upscaling: Some(upscaling),
            upscale_backend: Some(upscaler),
            ..Default::default()
        };
        let mut layers = SettingsLayers::new(&world, &QualityCeiling::NONE, &user);
        layers.capabilities = &caps;
        layers.resolve().graphics.reflections.ray_traced
    };
    let vulkan = DeviceCapabilities {
        ray_tracing_with_xess: false,
        ..DeviceCapabilities::ALL
    };
    assert!(!rt_with(true, UpscalerBackend::Xess, vulkan));
    assert!(
        rt_with(false, UpscalerBackend::Xess, vulkan),
        "no upscaling"
    );
    assert!(rt_with(true, UpscalerBackend::Dlss, vulkan), "DLSS leads");
    let only_xess = DeviceCapabilities {
        upscalers: UpscalerAvailability {
            dlss: false,
            xess: true,
            fsr: true,
        },
        ..vulkan
    };
    assert!(
        !rt_with(true, UpscalerBackend::Auto, only_xess),
        "Auto picks XeSS"
    );
    assert!(
        rt_with(true, UpscalerBackend::Fsr3, only_xess),
        "FSR asked for"
    );
    assert!(
        rt_with(true, UpscalerBackend::Xess, DeviceCapabilities::ALL),
        "a device that can run both"
    );
}

// A clamp shapes the resolution, never the user's stored choice, so the same
// settings file turns the feature on where the device can run it.
#[test]
fn a_capability_clamp_leaves_the_stored_choice_alone() {
    let world = WorldSettings::default();
    let user = UserGraphicsSettings {
        ray_traced_reflections: Some(true),
        ..Default::default()
    };
    let before = user.clone();
    let caps = DeviceCapabilities {
        ray_tracing: false,
        ..DeviceCapabilities::ALL
    };
    let mut layers = SettingsLayers::new(&world, &QualityCeiling::NONE, &user);
    layers.capabilities = &caps;
    assert!(!layers.resolve().graphics.reflections.ray_traced);
    assert_eq!(user, before);
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

// The planet's finest cell size is the world's, coarsened to the tier's floor
// and never refined by it, and a change to it applies live.
#[test]
fn the_planet_cell_size_resolves_under_the_ceiling() {
    use crate::components::PlanetDetailOverrides;
    let none = UserGraphicsSettings::default();
    let cells = |world: &WorldSettings, ceiling: &QualityCeiling| {
        resolve_settings(world, &none, ceiling)
            .graphics
            .planet_detail
            .finest_cell_size
    };
    let bare = WorldSettings::default();
    assert_eq!(cells(&bare, &QualityCeiling::NONE), 1);
    assert_eq!(cells(&bare, &low()), 4);
    let medium = authored(GraphicsConfig {
        planet_detail: PlanetDetailOverrides {
            finest_cell_size: Some(3),
        },
        ..Default::default()
    });
    assert_eq!(cells(&medium, &QualityCeiling::NONE), 3);
    assert_eq!(cells(&medium, &QualityCeiling::HIGH), 3);
    assert_eq!(cells(&medium, &low()), 4);
    let coarse = authored(GraphicsConfig {
        planet_detail: PlanetDetailOverrides {
            finest_cell_size: Some(8),
        },
        ..Default::default()
    });
    assert_eq!(cells(&coarse, &QualityCeiling::ULTRA), 8);

    let before = resolve_settings(&bare, &none, &QualityCeiling::NONE);
    let after = resolve_settings(&bare, &none, &low());
    let live = after.diff(&before);
    assert!(
        live.class(ApplyClass::Live)
            .iter()
            .any(|c| c.asset == "PlanetDetail" && c.key == "finest_cell_size"),
        "{live:?}"
    );
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
    declares::<crate::components::PlanetDetail>(&["finest_cell_size"]);
    declares::<Shadows>(&["map_size", "update", "distance", "cascades"]);
    declares::<TextureFiltering>(&["anisotropy"]);
    declares::<FramePacing>(&["vsync", "fps_cap", "frames_in_flight"]);
    declares::<StreamingConfig>(&["texture_cap", "texture_budget"]);
}

#[test]
fn derived_feature_settings_follow_the_resolved_sections() {
    let world = WorldSettings::default();
    let none = UserGraphicsSettings::default();
    let g = SettingsLayers::new(&world, &low(), &none)
        .resolve()
        .graphics;
    assert!(!g.taa(), "the Low ceiling caps TAA to FXAA");
    assert!(g.ssao().is_none() && g.ssr().is_none() && g.ssgi().is_none());
    let g = SettingsLayers::new(&world, &QualityCeiling::NONE, &none)
        .resolve()
        .graphics;
    assert!(g.taa());
    assert!(g.ssao().is_some() && g.ssr().is_some() && g.rt_reflections().is_some());
    assert_eq!(g.reflection_blur_scale(), g.reflections.blur_divisor());
    assert_eq!(g.hdr_samples(), 1, "TAA renders single-sampled");
}

// The look sliders write their section fields, so the composite tunables and
// the auto-exposure bias both follow a moved exposure slider.
#[test]
fn a_slider_sets_its_section_field() {
    let world = WorldSettings::default();
    let user = UserGraphicsSettings {
        exposure_ev: Some(1.5),
        bloom_intensity: Some(-2.0),
        lut_strength: Some(0.25),
        ambient_intensity: Some(3.0),
        ..Default::default()
    };
    let g = SettingsLayers::new(&world, &QualityCeiling::NONE, &user)
        .resolve()
        .graphics;
    assert_eq!(g.exposure.ev, 1.5);
    assert_eq!(g.auto_exposure_bias_ev(), 1.5);
    let tunables = g.post_process();
    assert_eq!(tunables.exposure, crate::math::exp2(1.5));
    assert_eq!(tunables.bloom_intensity, 0.0);
    assert_eq!(tunables.lut_strength, 0.25);
    assert_eq!(g.ambient_intensity(), 3.0);
}

// Auto-exposure meters for the display output the user resolved, not only
// the one the world authored.
#[test]
fn auto_exposure_meters_for_the_resolved_display() {
    let world = authored(GraphicsConfig {
        display: DisplayOverrides {
            hdr: Some(true),
            ..Default::default()
        },
        exposure: crate::components::ExposureOverrides {
            auto: Some(true),
            ..Default::default()
        },
        ..Default::default()
    });
    let user = UserGraphicsSettings {
        hdr_display: Some(false),
        ..Default::default()
    };
    let g = SettingsLayers::new(&world, &QualityCeiling::NONE, &user)
        .resolve()
        .graphics;
    assert_eq!(g.auto_exposure(), g.exposure.auto_settings(false));
}

// The authored night vision reaches the auto-exposure settings the backends
// adapt with, and applies live.
#[test]
fn night_vision_resolves_into_the_auto_exposure() {
    let world = authored(GraphicsConfig {
        exposure: crate::components::ExposureOverrides {
            auto: Some(true),
            auto_night_ev: Some(-3.0),
            night_vision: Some(0.75),
            ..Default::default()
        },
        ..Default::default()
    });
    let none = UserGraphicsSettings::default();
    let on = SettingsLayers::new(&world, &QualityCeiling::NONE, &none).resolve();
    let ae = on.graphics.auto_exposure().expect("auto-exposure on");
    assert_eq!(ae.night_vision, 0.75);
    assert_eq!(ae.night_bias_ev, -3.0);
    let world_off = authored(GraphicsConfig {
        exposure: crate::components::ExposureOverrides {
            auto: Some(true),
            auto_night_ev: Some(-3.0),
            ..Default::default()
        },
        ..Default::default()
    });
    let off = SettingsLayers::new(&world_off, &QualityCeiling::NONE, &none).resolve();
    assert_eq!(off.graphics.exposure.night_vision, 0.0);
    let diff = on.diff(&off);
    let live: Vec<(&str, &str)> = diff
        .class(ApplyClass::Live)
        .iter()
        .map(|c| (c.asset, c.key))
        .collect();
    assert_eq!(live, [("Exposure", "night_vision")]);
    assert_eq!(diff.costliest(), Some(ApplyClass::Live));
}

// A preset change adopts the re-resolved quality group and leaves every other
// resolved value as it was.
#[test]
fn adopting_a_quality_group_keeps_the_rest() {
    let world = WorldSettings::default();
    let user = UserGraphicsSettings {
        fps_cap: Some(30),
        temporal_upscaling: Some(true),
        ..Default::default()
    };
    let mut r = SettingsLayers::new(&world, &QualityCeiling::NONE, &user).resolve();
    let quality =
        SettingsLayers::new(&world, &low(), &UserGraphicsSettings::default()).resolve_quality();
    r.set_quality(quality.clone());
    assert_eq!(r.graphics.anti_aliasing.mode, quality.anti_aliasing.mode);
    assert_eq!(r.graphics.shadows.map_size, quality.shadows.map_size);
    assert_eq!(
        r.graphics.anti_aliasing.upscale_quality,
        quality.anti_aliasing.upscale_quality
    );
    assert_eq!(r.graphics.frame_pacing.fps_cap, 30);
    assert!(r.graphics.anti_aliasing.upscaling);
}
