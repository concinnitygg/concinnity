use super::*;
use concinnity_core::benchmark::BenchmarkRun;
use concinnity_core::components::{
    AaMode, BenchmarkConfig, FrameCapture, IndirectTechnique, KeyBinding, PassResolution,
    SettingCommand, SettingOp, UiAction, UpscaleQuality, UpscalerBackend, WindowMode,
};
use concinnity_core::render::backend_init::EmbeddedSurface;
use concinnity_core::render::dlss::DlssPreset;
use concinnity_core::render::rt_geom::RtDynamicMode;
use concinnity_core::settings::system::SettingsSystem;
use concinnity_core::settings::{SettingsRevision, StoredSettings};
use concinnity_core::window::display_mode::DisplayMode;

// Every persisted post-process choice overrides the world's authored value and
// reaches both the backend and the live settings snapshot. Under `Custom` there
// is no ceiling, so each override stands exactly as stored.
#[test]
fn persisted_post_process_overrides_win_over_authored_config() {
    let mut settings = concinnity_core::settings::StoredSettings::default();
    settings.graphics.quality_preset = Some(QualityPreset::Custom);
    // Sliders (stored as user-facing values; init runs them through the same
    // clamp/transform the live drag uses).
    settings.graphics.exposure_ev = Some(2.0);
    settings.graphics.bloom_intensity = Some(0.25);
    settings.graphics.bloom_threshold = Some(1.5);
    settings.graphics.bloom_knee = Some(0.2);
    settings.graphics.vignette = Some(0.4);
    settings.graphics.lut_strength = Some(0.3);
    settings.graphics.ambient_intensity = Some(2.0);
    // Quality toggles, each authored off in the world below.
    settings.graphics.ssao = Some(true);
    settings.graphics.ssr = Some(true);
    settings.graphics.ray_traced_reflections = Some(true);
    settings.graphics.ssgi = Some(true);
    settings.graphics.auto_exposure = Some(true);
    // Cycle dropdowns.
    settings.graphics.aa_mode = Some(AaMode::Taa);
    settings.graphics.ssgi_resolution = Some(PassResolution::Full);
    settings.graphics.ssgi_rays = Some(2);
    settings.graphics.reflection_blur_resolution = Some(PassResolution::Full);
    // Per-feature sub-quality sliders.
    settings.graphics.ssao_radius = Some(0.9);
    settings.graphics.ssao_intensity = Some(2.0);
    settings.graphics.ssr_intensity = Some(0.4);
    settings.graphics.ssr_max_distance = Some(20.0);
    settings.graphics.ssgi_intensity = Some(1.5);
    settings.graphics.ssgi_max_distance = Some(4.0);
    settings.graphics.auto_exposure_min_ev = Some(-4.0);
    settings.graphics.auto_exposure_max_ev = Some(4.0);
    settings.graphics.auto_exposure_speed = Some(3.0);

    let (state, hooks) = recording_hooks();
    let mut world = effects_scene(effects(|e| {
        // Deliberately the opposite of every override above, so a value that
        // survives to the backend can only have come from the store.
        e.anti_aliasing.mode = AaMode::Off;
        e.ambient_occlusion.enabled = false;
        e.exposure.ev = -3.0;
    }))
    .build();
    world.insert_resource(settings);
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    {
        let s = lock(&state);
        let init = s.init.as_ref().unwrap();
        assert!(
            init.taa_enabled,
            "the persisted TAA mode reaches the backend"
        );
        assert!(
            init.ssao_enabled,
            "the persisted SSAO toggle reaches the backend"
        );
    }

    let live = settings_state(&world);
    // Toggles.
    assert!(live.resolved().graphics.ambient_occlusion.enabled);
    assert!(live.resolved().graphics.reflections.screen_space);
    assert!(live.resolved().graphics.reflections.ray_traced);
    assert_eq!(
        live.resolved().graphics.indirect_lighting.technique,
        IndirectTechnique::Ssgi
    );
    assert!(live.resolved().graphics.exposure.auto);
    // Cycles.
    assert_eq!(live.resolved().graphics.anti_aliasing.mode, AaMode::Taa);
    assert_eq!(
        live.resolved().graphics.indirect_lighting.ssgi_resolution,
        PassResolution::Full
    );
    assert_eq!(live.resolved().graphics.indirect_lighting.ssgi_rays, 2);
    assert_eq!(
        live.resolved().graphics.reflections.blur_resolution,
        PassResolution::Full
    );
    // Sub-quality sliders.
    assert_eq!(live.resolved().graphics.ambient_occlusion.radius, 0.9);
    assert_eq!(live.resolved().graphics.ambient_occlusion.intensity, 2.0);
    assert_eq!(live.resolved().graphics.reflections.intensity, 0.4);
    assert_eq!(live.resolved().graphics.reflections.max_distance, 20.0);
    assert_eq!(
        live.resolved().graphics.indirect_lighting.ssgi_intensity,
        1.5
    );
    assert_eq!(
        live.resolved().graphics.indirect_lighting.ssgi_max_distance,
        4.0
    );
    assert_eq!(live.resolved().graphics.exposure.auto_min_ev, -4.0);
    assert_eq!(live.resolved().graphics.exposure.auto_max_ev, 4.0);
    assert_eq!(live.resolved().graphics.exposure.auto_speed, 3.0);
    // Post-process params: exposure is stored as EV and applied as 2^EV.
    assert_eq!(
        live.resolved().graphics.post_process().exposure,
        4.0,
        "2^2 EV, not the authored -3"
    );
    assert_eq!(
        live.resolved().graphics.post_process().bloom_intensity,
        0.25
    );
    assert_eq!(live.resolved().graphics.post_process().bloom_threshold, 1.5);
    assert_eq!(live.resolved().graphics.post_process().bloom_knee, 0.2);
    assert_eq!(live.resolved().graphics.post_process().vignette, 0.4);
    assert_eq!(live.resolved().graphics.post_process().lut_strength, 0.3);
    assert_eq!(live.resolved().graphics.ambient_intensity(), 2.0);
    // The composite FXAA flag follows the final AA mode, not the authored one.
    assert_eq!(
        live.resolved().graphics.post_process().fxaa,
        1.0,
        "TAA keeps the FXAA cleanup"
    );
    // The world's pristine config is kept as the baseline a live preset change
    // re-clamps from, untouched by the overrides.
    assert_eq!(live.authored().graphics.anti_aliasing.mode, AaMode::Off);
    assert!(!live.authored().graphics.ambient_occlusion.enabled);
}

// A weak-tier ceiling forces the world's authored effects off -- but only where
// the user expressed no preference. The pristine authored config is kept so a
// later preset up-shift can restore exactly what the world asked for.
#[test]
fn low_preset_forces_authored_effects_off_but_keeps_the_baseline() {
    let mut settings = concinnity_core::settings::StoredSettings::default();
    settings.graphics.quality_preset = Some(QualityPreset::Low);
    let (state, hooks) = recording_hooks_with(profile_at(GpuTier::Integrated));
    let mut world = effects_scene(effects(|e| {
        e.ambient_occlusion.enabled = true;
        e.reflections.screen_space = true;
        e.reflections.ray_traced = true;
        e.indirect_lighting.technique = IndirectTechnique::Ssgi;
        e.exposure.auto = true;
        e.anti_aliasing.mode = AaMode::Taa;
    }))
    .build();
    world.insert_resource(settings);
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    {
        let s = lock(&state);
        let init = s.init.as_ref().unwrap();
        assert!(
            !init.ssao_enabled,
            "the ceiling trims SSAO before the backend sizes it"
        );
        assert!(!init.taa_enabled, "TAA clamps down to a cheaper AA mode");
    }

    let live = settings_state(&world);
    assert!(!live.resolved().graphics.ambient_occlusion.enabled);
    assert!(!live.resolved().graphics.reflections.screen_space);
    assert!(!live.resolved().graphics.reflections.ray_traced);
    assert_eq!(
        live.resolved().graphics.indirect_lighting.technique,
        IndirectTechnique::Ibl
    );
    assert_ne!(live.resolved().graphics.anti_aliasing.mode, AaMode::Taa);
    // Auto-exposure is cheap enough that even the weakest ceiling permits it, so
    // the world's authored choice stands.
    assert!(live.resolved().graphics.exposure.auto);
    // The authored baseline survives the clamp, so an up-shift restores it.
    assert!(live.authored().graphics.ambient_occlusion.enabled);
    assert!(live.authored().graphics.reflections.screen_space);
    assert_eq!(live.authored().graphics.anti_aliasing.mode, AaMode::Taa);
}

// An explicit per-row override beats the preset ceiling: the user asked for this
// feature on a tier that would otherwise force it off, and that choice wins.
// Rows the user never touched still clamp.
#[test]
fn an_explicit_override_survives_the_preset_ceiling() {
    let mut settings = concinnity_core::settings::StoredSettings::default();
    settings.graphics.quality_preset = Some(QualityPreset::Low);
    settings.graphics.ssao = Some(true);
    settings.graphics.aa_mode = Some(AaMode::Taa);
    let (_state, hooks) = recording_hooks_with(profile_at(GpuTier::Integrated));
    let mut world = effects_scene(effects(|e| {
        e.ambient_occlusion.enabled = true;
        e.reflections.screen_space = true;
        e.anti_aliasing.mode = AaMode::Taa;
    }))
    .build();
    world.insert_resource(settings);
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    let live = settings_state(&world);
    assert!(
        live.resolved().graphics.ambient_occlusion.enabled,
        "the explicit SSAO choice wins"
    );
    assert_eq!(
        live.resolved().graphics.anti_aliasing.mode,
        AaMode::Taa,
        "explicit AA mode wins"
    );
    assert!(
        !live.resolved().graphics.reflections.screen_space,
        "a row the user never touched still clamps under the ceiling"
    );
}

// The `--quality-preset` launch flag has to reach the CEILING, not merely the
// menu label. On a mid-tier GPU `Auto` resolves to the `High` ceiling, which
// clamps ray-traced reflections off and logs nothing about it, so a probe that
// forces `ultra` and never checks would measure a non-RT frame and read it as a
// result. Both halves are asserted on the same GPU and the same authored world.
#[test]
fn the_quality_preset_flag_reaches_the_ray_tracing_ceiling() {
    let authored = effects(|e| {
        e.reflections.ray_traced = true;
    });
    let mut settings = concinnity_core::settings::StoredSettings::default();
    settings.graphics.quality_preset = Some(QualityPreset::Auto);

    let (clamped, hooks) = recording_hooks_with(profile_at(GpuTier::MidDiscrete));
    let mut world = effects_scene(authored.clone()).build();
    world.insert_resource(settings.clone());
    init_graphics(&mut world, hooks);
    assert_eq!(settings_state(&world).quality_preset(), QualityPreset::Auto);
    assert!(
        !lock(&clamped).init.as_ref().unwrap().rt_reflections_on,
        "the Auto -> High ceiling clamps RT off, silently"
    );

    let (forced, hooks) = recording_hooks_with(profile_at(GpuTier::MidDiscrete));
    let mut world = effects_scene(authored).build();
    world.insert_resource(settings);
    world.insert_resource(LaunchRequest {
        args: LaunchArgs {
            quality_preset: Some(QualityPreset::Ultra),
            ..LaunchArgs::NONE
        },
        ..Default::default()
    });
    init_graphics(&mut world, hooks);
    assert_eq!(
        settings_state(&world).quality_preset(),
        QualityPreset::Ultra,
        "the flag outranks the persisted Auto"
    );
    assert!(
        lock(&forced).init.as_ref().unwrap().rt_reflections_on,
        "the Ultra ceiling the flag forced permits RT on the same GPU"
    );
}

// The two ray-tracing launch flags reach the assembled `BackendInit` the
// backends destructure, and an absent flag lands on the shipping value.
#[test]
fn the_ray_tracing_flags_reach_the_backend() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    init_graphics(&mut world, hooks);
    {
        let s = lock(&state);
        let init = s.init.as_ref().unwrap();
        assert_eq!(init.rt_dynamic, RtDynamicMode::Auto);
        assert!(init.rt_skinned_geometry);
    }

    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    world.insert_resource(LaunchRequest {
        rt_dynamic: Some(RtDynamicMode::Rebuild),
        rt_skinned_geometry: Some(false),
        ..Default::default()
    });
    init_graphics(&mut world, hooks);
    {
        let s = lock(&state);
        let init = s.init.as_ref().unwrap();
        assert_eq!(init.rt_dynamic, RtDynamicMode::Rebuild);
        assert!(!init.rt_skinned_geometry);
    }
}

// `--dlss-preset` outranks the persisted preset for the launch: the backend is
// built with it and the menu row shows it, while the settings store keeps the
// saved choice until the row itself is cycled.
#[test]
fn a_forced_dlss_preset_reaches_the_backend_and_the_menu_unpersisted() {
    const ROW_LABEL: AssetId = AssetId(700);
    let mut saved_settings = concinnity_core::settings::StoredSettings::default();
    saved_settings.graphics.dlss_preset = Some(DlssPreset::K);
    let launch = |forced: Option<DlssPreset>| {
        // An NVIDIA GPU, where the DLSS preset row is live.
        let nvidia = GpuProfile {
            vendor: GpuVendor::Nvidia,
            ..GpuProfile::UNKNOWN
        };
        let (state, hooks) = recording_hooks_with(nvidia);
        let mut b = effects_scene(Default::default());
        push_settings_row(&mut b, "dlss_preset", "next", ROW_LABEL);
        let mut world = b.build();
        world.insert_resource(saved_settings.clone());
        world.insert_resource(LaunchRequest {
            dlss_preset: forced,
            ..Default::default()
        });
        let gs = init_graphics(&mut world, hooks);
        assert!(!gs.failed);
        let built = lock(&state).init.as_ref().unwrap().dlss_preset;
        (world, built)
    };
    let menu = |world: &mut World| {
        let live = settings_state(world);
        let shown = (
            live.resolved().graphics.dlss_preset,
            live.persisted_graphics().dlss_preset,
        );
        (shown, label_text(world, ROW_LABEL))
    };

    let (mut world, built) = launch(None);
    assert_eq!(built, DlssPreset::K);
    assert_eq!(
        menu(&mut world),
        ((DlssPreset::K, Some(DlssPreset::K)), "K".to_string())
    );

    let (mut world, built) = launch(Some(DlssPreset::M));
    assert_eq!(built, DlssPreset::M);
    assert_eq!(
        menu(&mut world),
        ((DlssPreset::M, Some(DlssPreset::K)), "M".to_string())
    );

    // Cycling the row steps from the forced value and saves the preset alone.
    world.insert_resource(saved_settings.clone());
    world.events_mut::<SettingCommand>().send(SettingCommand {
        setting: SettingKey::DlssPreset,
        op: SettingOp::Next,
        value_label: Some(ROW_LABEL),
        persist: true,
    });
    {
        use concinnity_core::ecs::System;
        SettingsSystem::new().step(&mut world.context());
    }
    assert_eq!(
        settings_state(&world).resolved().graphics.dlss_preset,
        DlssPreset::Default
    );
    assert_eq!(label_text(&mut world, ROW_LABEL), "Default");
    let mut expected = saved_settings;
    expected.graphics.dlss_preset = Some(DlssPreset::Default);
    assert_eq!(world.resource::<StoredSettings>(), Some(&expected));
    assert!(
        world
            .resource::<SettingsRevision>()
            .is_some_and(|r| r.0 > 0)
    );
}

// A world launched without a screenshot request keeps the presented frame
// unreadable, so production pays nothing for capture.
#[test]
fn capture_stays_off_without_a_launch_request() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    init_graphics(&mut world, hooks);
    assert!(!lock(&state).init.as_ref().unwrap().capture);
}

// The launch's screenshot request reaches the backend through the world's
// `LaunchRequest`.
#[test]
fn a_capture_launch_request_reaches_the_backend() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    world.insert_resource(LaunchRequest {
        capture: true,
        ..Default::default()
    });
    init_graphics(&mut world, hooks);
    assert!(lock(&state).init.as_ref().unwrap().capture);
}

// A benchmark that saves frames along the way keeps the presented frame
// readable, with no launch request asking for it.
#[test]
fn a_benchmark_with_captures_arms_frame_capture() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    world.insert_resource(BenchmarkRun::new(BenchmarkConfig {
        captures: vec![FrameCapture {
            seconds: 1.0,
            path: "shot.png".to_string(),
        }],
        ..Default::default()
    }));
    init_graphics(&mut world, hooks);
    assert!(lock(&state).init.as_ref().unwrap().capture);
}

// A key bound to a screenshot keeps the presented frame readable, with no
// launch request asking for it.
#[test]
fn a_screenshot_binding_arms_frame_capture() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(KeyBinding {
        key: "P".to_string(),
        action: Some(UiAction::Screenshot(1)),
        ..Default::default()
    });
    let mut world = b.build();
    init_graphics(&mut world, hooks);
    assert!(lock(&state).init.as_ref().unwrap().capture);
}

// KeyBindings alone need no free cursor, so a camera world that only binds
// keys keeps its first-person capture.
#[test]
fn key_bindings_alone_leave_a_camera_world_out_of_menu_mode() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(KeyBinding {
        key: "Q".to_string(),
        action: Some(UiAction::Quit),
        ..Default::default()
    });
    let mut world = b.build();
    init_graphics(&mut world, hooks);
    let s = lock(&state);
    assert!(s.saw(&Call::SetMenuMode(false)));
    assert!(
        !s.init.as_ref().unwrap().capture,
        "no screenshot, no capture"
    );
}

// A development session keeps frame capture available without a screenshot.
#[test]
fn a_dev_loop_launch_arms_frame_capture() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    world.insert_resource(LaunchRequest {
        dev_loop: true,
        ..Default::default()
    });
    init_graphics(&mut world, hooks);
    assert!(lock(&state).init.as_ref().unwrap().capture);
}

// A host-owned view inserted before start reaches the backend; without one the
// backend opens its own window.
#[test]
fn an_embedded_surface_resource_reaches_the_backend() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    init_graphics(&mut world, hooks);
    assert!(!lock(&state).init.as_ref().unwrap().embedded_surface);

    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    world.insert_resource(EmbeddedSurface {
        view: std::ptr::NonNull::dangling(),
        pump_events: false,
    });
    init_graphics(&mut world, hooks);
    assert!(lock(&state).init.as_ref().unwrap().embedded_surface);
}

// A ceiling only ever reduces: a world that turned the expensive effects off is
// not "upgraded" by a high tier, and the resolved preset is held for the master
// menu row.
#[test]
fn a_high_ceiling_never_enables_what_the_world_turned_off() {
    let mut settings = concinnity_core::settings::StoredSettings::default();
    settings.graphics.quality_preset = Some(QualityPreset::Ultra);
    let (_state, hooks) = recording_hooks_with(profile_at(GpuTier::HighDiscrete));
    let mut world = effects_scene(effects(|e| {
        e.ambient_occlusion.enabled = false;
        e.reflections.screen_space = false;
        e.reflections.ray_traced = false;
    }))
    .build();
    world.insert_resource(settings);
    init_graphics(&mut world, hooks);

    let live = settings_state(&world);
    assert_eq!(live.quality_preset(), QualityPreset::Ultra);
    assert!(
        !live.resolved().graphics.ambient_occlusion.enabled,
        "a feature turned off stays off"
    );
    assert!(!live.resolved().graphics.reflections.screen_space);
    assert!(!live.resolved().graphics.reflections.ray_traced);
}

// The other direction: a world that authors nothing takes the schema defaults,
// so the top tier runs the whole stack without the world asking for it.
#[test]
fn the_top_tier_runs_the_default_stack_for_a_world_that_authors_nothing() {
    let mut settings = concinnity_core::settings::StoredSettings::default();
    settings.graphics.quality_preset = Some(QualityPreset::Ultra);
    let (_state, hooks) = recording_hooks_with(profile_at(GpuTier::HighDiscrete));
    let mut world = scene_builder().build();
    world.insert_resource(settings);
    init_graphics(&mut world, hooks);

    let live = settings_state(&world);
    assert!(live.resolved().graphics.ambient_occlusion.enabled);
    assert!(live.resolved().graphics.reflections.screen_space);
    assert!(live.resolved().graphics.reflections.ray_traced);
    assert_eq!(live.resolved().graphics.anti_aliasing.mode, AaMode::Taa);
}

// And the tier below it clamps that same default stack down, which is how a
// world that authors nothing ends up matching its GPU.
#[test]
fn a_low_tier_clamps_the_default_stack_off() {
    let mut settings = concinnity_core::settings::StoredSettings::default();
    settings.graphics.quality_preset = Some(QualityPreset::Low);
    let (_state, hooks) = recording_hooks_with(profile_at(GpuTier::Integrated));
    let mut world = scene_builder().build();
    world.insert_resource(settings);
    init_graphics(&mut world, hooks);

    let live = settings_state(&world);
    assert!(!live.resolved().graphics.ambient_occlusion.enabled);
    assert!(!live.resolved().graphics.reflections.screen_space);
    assert!(!live.resolved().graphics.reflections.ray_traced);
    assert_eq!(
        live.resolved().graphics.indirect_lighting.technique,
        IndirectTechnique::Ibl
    );
    assert_eq!(live.resolved().graphics.anti_aliasing.mode, AaMode::Fxaa);
}

// Under `Auto` the ceiling re-resolves from the detected tier each launch: a
// weaker GPU shadows a smaller map than a stronger one, and the strongest tier
// leaves the world's authored size alone.
#[test]
fn auto_preset_shadow_ceiling_tracks_the_detected_tier() {
    let sizes: Vec<u32> = [
        GpuTier::Integrated,
        GpuTier::EntryDiscrete,
        GpuTier::MidDiscrete,
        GpuTier::HighDiscrete,
    ]
    .into_iter()
    .map(|tier| {
        let (state, hooks) = recording_hooks_with(profile_at(tier));
        let mut world = scene_builder().build();
        init_graphics(&mut world, hooks);
        assert_eq!(settings_state(&world).quality_preset(), QualityPreset::Auto);
        lock(&state).init.as_ref().unwrap().shadows.map_size
    })
    .collect();

    assert!(
        sizes.windows(2).all(|w| w[0] <= w[1]),
        "a stronger tier never shadows at a smaller map: {sizes:?}"
    );
    assert_eq!(
        sizes.last().copied(),
        Some(4096),
        "the top tier leaves the world's authored 4096 unclamped"
    );
}

// Display-output, upscaling, and system/streaming preferences are restart-class:
// resolved once at init from the world's config overridden by the persisted
// choice, passed to the backend, and held for the settings rows. The window mode
// and display mode are applied to the backend after construction, since the
// window is always created windowed.
#[test]
fn persisted_display_and_system_overrides_reach_the_backend() {
    let mut settings = concinnity_core::settings::StoredSettings::default();
    settings.graphics.quality_preset = Some(QualityPreset::Custom);
    settings.graphics.window_mode = Some(WindowMode::Fullscreen);
    settings.graphics.resolution = Some([1920, 1080, 60]);
    settings.graphics.render_scale = Some(UpscaleQuality::Performance);
    settings.graphics.upscale_backend = Some(UpscalerBackend::Fsr3);
    settings.graphics.temporal_upscaling = Some(true);
    settings.graphics.hdr_display = Some(true);
    settings.graphics.hdr_pq = Some(true);
    settings.graphics.occlusion_two_pass = Some(true);
    settings.graphics.texture_cap = Some(192);
    settings.graphics.texture_budget = Some(8);
    settings.graphics.perf_stats = Some(false);
    settings.graphics.show_fps = Some(false);
    settings.graphics.show_vram = Some(false);

    let (state, hooks) = recording_hooks();
    let mut b = effects_scene(Default::default());
    b.push(StreamingConfig::default());
    let mut world = b.build();
    world.insert_resource(settings);
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    {
        let s = lock(&state);
        assert!(
            s.saw(&Call::SetWindowMode(WindowMode::Fullscreen)),
            "a persisted non-windowed mode is applied after construction"
        );
        assert!(
            s.saw(&Call::SetDisplayMode(DisplayMode {
                width: 1920,
                height: 1080,
                refresh_hz: 60,
            })),
            "the chosen fullscreen mode is handed to the backend"
        );
    }

    let live = settings_state(&world);
    assert_eq!(
        live.resolved().graphics.anti_aliasing.upscale_quality,
        UpscaleQuality::Performance
    );
    assert_eq!(
        live.resolved().graphics.anti_aliasing.upscaler,
        UpscalerBackend::Fsr3
    );
    assert!(live.resolved().graphics.anti_aliasing.upscaling);
    assert!(live.resolved().graphics.display.hdr);
    assert!(live.resolved().graphics.display.pq);
    assert!(live.resolved().graphics.culling.two_pass_occlusion);
    assert_eq!(live.resolved().texture_cap, 192);
    assert_eq!(live.resolved().texture_budget, 8);
    assert!(!live.resolved().perf_stats);
    assert!(!live.resolved().show_fps);
    assert!(!live.resolved().show_vram);
    assert_eq!(live.window().mode, WindowMode::Fullscreen);
    assert_eq!(
        live.resolution(),
        Some(DisplayMode {
            width: 1920,
            height: 1080,
            refresh_hz: 60,
        })
    );
    // A backend that cannot enumerate modes falls back to the static preset list
    // rather than publishing an empty dropdown.
    assert!(
        !world
            .resource::<concinnity_core::ecs::DisplayModes>()
            .expect("mode list published for the dropdown")
            .0
            .is_empty()
    );
    // The resolved cap reaches the runtime-level pacer through its own resource.
    assert!(world.resource::<FrameRateCap>().is_some());
}
