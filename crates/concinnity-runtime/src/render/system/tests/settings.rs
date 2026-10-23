use super::*;
use concinnity_engine::components::{KeyBinding, Screenshot, UiAction, WindowMode};
use concinnity_engine::launch::LaunchArgs;
use concinnity_engine::render::backend::init::EmbeddedSurface;
use concinnity_engine::render::rt_accel::geom::RtDynamicMode;
use concinnity_engine::window::display_mode::DisplayMode;

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
    let mut settings = concinnity_engine::settings::StoredSettings::default();
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

// A run that takes screenshots keeps the presented frame readable, with no
// capture request asking for it; the same world played normally does not.
#[test]
fn a_run_taking_screenshots_arms_frame_capture() {
    let armed = |args: LaunchArgs| {
        let (state, hooks) = recording_hooks();
        let mut b = scene_builder();
        b.push(Screenshot {
            seconds: 1.0,
            file: "shot.png".to_string(),
            hud: true,
        });
        let mut world = b.build();
        world.insert_resource(LaunchRequest {
            args,
            ..Default::default()
        });
        init_graphics(&mut world, hooks);
        lock(&state).init.as_ref().unwrap().capture
    };
    assert!(armed(LaunchArgs {
        screenshots: true,
        ..LaunchArgs::NONE
    }));
    assert!(!armed(LaunchArgs::NONE));
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

// The persisted window mode and display mode are applied to the backend after
// construction, since the window is always created windowed, and init
// publishes the mode list and the frame-rate cap.
#[test]
fn persisted_display_and_system_overrides_reach_the_backend() {
    let mut settings = concinnity_engine::settings::StoredSettings::default();
    settings.graphics.quality_preset = Some(QualityPreset::Custom);
    settings.graphics.window_mode = Some(WindowMode::Fullscreen);
    settings.graphics.resolution = Some([1920, 1080, 60]);

    let (state, hooks) = recording_hooks();
    let mut world = effects_scene(ResolvedGraphics::default()).build();
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

    // A backend that cannot enumerate modes falls back to the static preset list
    // rather than publishing an empty dropdown.
    assert!(
        !world
            .resource::<concinnity_engine::ecs::DisplayModes>()
            .expect("mode list published for the dropdown")
            .0
            .is_empty()
    );
    // The resolved cap reaches the runtime-level pacer through its own resource.
    assert!(world.resource::<FrameRateCap>().is_some());
}
