use super::*;
use concinnity_host::store::paths::StateTree;

// A first launch (no persisted preset, no launch request) seeds `Auto`, and the
// settings system saves it into the world's own state tree.
#[test]
fn a_first_launch_persists_the_auto_preset_into_the_state_tree() {
    use concinnity_engine::ecs::System;
    let dir = concinnity_testing::TempTree::new();
    let tree = StateTree::at(dir.path());
    let (_state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    world.insert_resource(tree.clone());
    let mut gs = GraphicsSystem::new(Some(&tree));
    gs.test_hooks = Some(hooks);
    let mut settings = concinnity_engine::settings::system::SettingsSystem::new()
        .with_store(Box::new(crate::persist::settings::FileStore::new(&tree)));
    {
        let mut ctx = world.context();
        concinnity_engine::ecs::decompose::run(&mut ctx);
        System::init(&mut gs, &mut ctx);
        settings.step(&mut ctx);
    }
    // Dropping the store's writer waits for its queued save.
    drop(settings);
    assert!(!gs.failed);
    assert_eq!(
        crate::persist::settings::load(Some(&tree))
            .graphics
            .quality_preset,
        Some(QualityPreset::Auto)
    );
}

#[test]
fn init_builds_draw_list_and_render_handles() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed, "init must succeed");
    assert!(backend_parked(&world), "mock backend parked in the slot");

    let s = lock(&state);
    let init = s.init.as_ref().expect("factory captured the BackendInit");
    assert_eq!(init.window_width, 640);
    assert_eq!(init.window_height, 360);
    assert_eq!(init.window_title, "mock world");
    assert_eq!(init.clear_color, [0.1, 0.2, 0.3, 1.0]);
    assert_eq!(init.frames_in_flight, 2);
    // One prop placing one mesh = one draw object over the quad geometry.
    assert_eq!(init.draw_objects.len(), 1);
    assert_eq!(init.vertex_count, 4);
    assert_eq!(init.index_count, 6);
    let draw = &init.draw_objects[0];
    assert_eq!(draw.index_count, 6);
    assert_eq!(draw.texture_slot, 0, "material albedo resolves to slot 0");
    assert!(draw.visible);
    // The prop's translation is baked into the draw's model matrix.
    assert_eq!(draw.model[3][0], 1.0);
    assert_eq!(draw.model[3][1], 2.0);
    assert_eq!(draw.model[3][2], 3.0);
    assert_eq!(init.texture_count, 1);
    assert_eq!(init.text_atlas_count, 0, "no fonts or sprite textures");
    assert_eq!(init.n_skinned, 0);
    assert_eq!(init.instanced_cluster_count, 0);
    assert!(init.scene_required, "a mesh world requires the scene chain");
    assert!(!init.fog, "no VolumetricFog declared");
    drop(s);

    // The prop entity received its GPU handle + init world matrix.
    let ctx = world.context();
    let handles: Vec<Vec<DrawIndex>> = ctx
        .query::<RenderHandle>()
        .map(|h| h.draws.to_vec())
        .collect();
    assert_eq!(handles, vec![vec![DrawIndex(0)]]);
    let globals: Vec<[[f32; 4]; 4]> = ctx.query::<GlobalTransform>().map(|g| g.0).collect();
    assert_eq!(globals.len(), 1);
    assert_eq!(globals[0][3][0], 1.0);
}

// Regression: init parks exactly one OverlayAssets carrying the captured HUD
// chip ids. A duplicated park block once re-took the already-moved fields
// (`std::mem::take`), so the second park overwrote the resource with empty
// defaults and the whole overlay -- HUD chips and menu alike -- silently drew
// nothing. The chips surviving to the parked resource guards that.
#[test]
fn init_parks_overlay_assets_with_the_hud_chips() {
    use concinnity_engine::components::{DebugHud, StatHud};
    use concinnity_engine::render::overlay::OverlayAssets;

    let (_state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(StatHud {
        fps_label: Some(Ref::new(AssetId(10))),
        vram_label: Some(Ref::new(AssetId(11))),
        ..Default::default()
    });
    b.push(DebugHud {
        mouse_label: Some(Ref::new(AssetId(20))),
        passes_label: Some(Ref::new(AssetId(21))),
        ..Default::default()
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed, "init must succeed");

    let overlay = world
        .resource::<OverlayAssets>()
        .expect("OverlayAssets parked at init");
    assert_eq!(
        overlay.stat_hud_chips,
        vec![AssetId(10), AssetId(11)],
        "StatHud chip ids survive to the parked OverlayAssets"
    );
    // DebugHud strip order is mouse, camera, sys, passes; only mouse + passes
    // were set here.
    assert_eq!(
        overlay.debug_hud_chips,
        vec![AssetId(20), AssetId(21)],
        "DebugHud chip ids survive to the parked OverlayAssets"
    );
}

// The built backend can be taken exactly once (the `cn editor` transplant).
#[test]
fn take_backend_yields_the_backend_once() {
    let (_state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let _gs = init_graphics(&mut world, hooks);
    assert!(
        concinnity_engine::ecs::ActiveRenderBackend::take(world.context().resources).is_some(),
        "the built backend is taken"
    );
    assert!(
        concinnity_engine::ecs::ActiveRenderBackend::take(world.context().resources).is_none(),
        "and only once"
    );
}

// A world carrying a transplanted backend whose swapchain config matches reuses
// it (calls reload_world) instead of building a fresh one, and the reused
// backend ends up carrying the new world's content -- proving the same instance
// was swapped over (the `cn editor` live SAVE hot path).
#[test]
fn pending_backend_reuses_instance_and_ends_with_new_world() {
    // World A builds its own backend (the factory records into state_a).
    let (state_a, hooks_a) = recording_hooks();
    let mut world_a = titled_scene("world A").build();
    let _gs_a = init_graphics(&mut world_a, hooks_a);
    let backend_a = concinnity_engine::ecs::ActiveRenderBackend::take(world_a.context().resources)
        .expect("world A built a backend");

    // Transplant it into world B (a different world, same swapchain config).
    let (state_b, hooks_b) = recording_hooks();
    let mut world_b = titled_scene("world B").build();
    world_b.insert_resource(crate::live_edit::PendingBackend(backend_a));
    let gs_b = init_graphics(&mut world_b, hooks_b);

    assert!(!gs_b.failed);
    assert!(backend_parked(&world_b), "the reused backend is installed");
    let a = lock(&state_a);
    assert!(
        a.saw(&Call::ReloadWorld),
        "the transplanted backend was reloaded, not rebuilt"
    );
    assert_eq!(
        a.init.as_ref().unwrap().window_title,
        "world B",
        "the reused backend now carries the new world's content"
    );
    drop(a);
    assert!(
        state_b.lock().init.is_none(),
        "world B never built a fresh backend"
    );
}

// A transplanted backend whose swapchain config differs (frames-in-flight, HDR)
// cannot reuse the window: it is idled + dropped and a fresh backend is built.
#[test]
fn pending_backend_swapchain_change_forces_full_rebuild() {
    let (state_t, _unused) = recording_hooks();
    let transplant = RecordingBackend::transplant(
        Arc::clone(&state_t),
        Some(SwapchainConfig {
            // The scene world uses the default 2 frames-in-flight; 3 mismatches.
            frames_in_flight: 3,
            hdr_display: false,
            hdr_pq: false,
        }),
    );
    let (state_b, hooks_b) = recording_hooks();
    let mut world = scene_builder().build();
    world.insert_resource(crate::live_edit::PendingBackend(Box::new(transplant)));
    let gs = init_graphics(&mut world, hooks_b);

    assert!(!gs.failed);
    assert!(backend_parked(&world));
    let t = lock(&state_t);
    assert!(
        t.saw(&Call::WaitIdle),
        "the mismatched backend is idled before it is dropped"
    );
    assert!(
        !t.saw(&Call::ReloadWorld),
        "no reload is attempted across a swapchain change"
    );
    drop(t);
    assert!(
        state_b.lock().init.is_some(),
        "a fresh backend is built for the changed swapchain"
    );
}

// The HDR axis of the swapchain identity is honored too: a transplant whose
// hdr_display request differs from the new world's forces a full rebuild rather
// than an in-place reload (the pixel format would otherwise diverge).
#[test]
fn pending_backend_hdr_change_forces_full_rebuild() {
    let (state_t, _unused) = recording_hooks();
    let transplant = RecordingBackend::transplant(
        Arc::clone(&state_t),
        Some(SwapchainConfig {
            // The scene world requests no HDR (the Display default); an
            // HDR-requesting transplant is a swapchain change.
            frames_in_flight: 2,
            hdr_display: true,
            hdr_pq: false,
        }),
    );
    let (state_b, hooks_b) = recording_hooks();
    let mut world = scene_builder().build();
    world.insert_resource(crate::live_edit::PendingBackend(Box::new(transplant)));
    let gs = init_graphics(&mut world, hooks_b);

    assert!(!gs.failed);
    let t = lock(&state_t);
    assert!(
        t.saw(&Call::WaitIdle),
        "the HDR-mismatched backend is idled"
    );
    assert!(!t.saw(&Call::ReloadWorld), "no reload across an HDR change");
    drop(t);
    assert!(
        state_b.lock().init.is_some(),
        "a fresh backend is built for the changed HDR output"
    );
}

// A transplanted backend that reports hot-swap-capable but fails the reload
// leaves GraphicsSystem failed (no silent fallback that loses the edit).
#[test]
fn reload_world_failure_marks_graphics_failed() {
    let (state_a, hooks_a) = recording_hooks();
    let mut world_a = scene_builder().build();
    let _gs_a = init_graphics(&mut world_a, hooks_a);
    let backend_a =
        concinnity_engine::ecs::ActiveRenderBackend::take(world_a.context().resources).unwrap();
    state_a.lock().fail_reload = Some(error::RenderError::DeviceLost {
        reason: error::DeviceLostReason::Removed,
        detail: "boom".to_string(),
    });

    let (_state_b, hooks_b) = recording_hooks();
    let mut world_b = scene_builder().build();
    world_b.insert_resource(crate::live_edit::PendingBackend(backend_a));
    let gs_b = init_graphics(&mut world_b, hooks_b);

    assert!(gs_b.failed, "a failed reload marks the system failed");
    assert!(!backend_parked(&world_b));
    assert!(state_a.lock().saw(&Call::ReloadWorld));
}

#[test]
fn init_pushes_startup_backend_state() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let _gs = init_graphics(&mut world, hooks);

    let s = lock(&state);
    // A camera with no UI: plain first-person capture, menu mode off.
    assert!(s.saw(&Call::SetMenuMode(false)));
    assert!(s.saw(&Call::RequestCursorCapture));
    // The ambient scale + keymap are pushed once after construction.
    assert!(s.saw(&Call::SetAmbientIntensity(1.0)));
    assert!(s.saw(&Call::SetKeymap));
    // Reflection-probe placements are pushed once (auto-seed or empty).
    assert!(
        s.calls
            .iter()
            .any(|c| matches!(c, Call::SetReflectionProbes(..)))
    );
    // Windowed start: no window-mode override is applied.
    assert!(!s.calls.iter().any(|c| matches!(c, Call::SetWindowMode(_))));
}

// A menu / editor driver present at init (the editor's live-preview rebuild seeds
// a `MenuOverride` before re-running init) suppresses the first-person startup
// grab: the per-frame drive owns capture, and re-grabbing on every rebuild would
// re-hide and decouple the OS cursor, desyncing the free-cursor handoff.
#[test]
fn menu_driven_init_skips_the_first_person_cursor_grab() {
    let (state, hooks) = recording_hooks();
    // A plain first-person world (Camera3D, no HitRegion / KeyBinding) -- the same
    // world that grabs at startup above -- but with a driver already in control.
    let mut world = scene_builder().build();
    world.insert_resource(MenuOverride(Some(true)));
    let _gs = init_graphics(&mut world, hooks);

    let s = lock(&state);
    assert!(
        !s.saw(&Call::RequestCursorCapture),
        "a menu/editor driver owns capture; init must not auto-grab"
    );
    // Menu mode still reflects the world's own UI (none here), unchanged.
    assert!(s.saw(&Call::SetMenuMode(false)));
}

#[test]
fn ui_only_world_trims_scene_features() {
    let (state, hooks) = recording_hooks();
    let mut b = WorldBuilder::new();
    b.push(Window::default());
    b.push_shaders();
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed);
    let s = lock(&state);
    let init = s.init.as_ref().unwrap();
    assert!(init.draw_objects.is_empty());
    assert!(!init.scene_required, "no 3D content: scene chain trimmed");
    assert_eq!(init.shadows.map_size, 0, "shadows trimmed with the scene");
    assert!(!init.taa_enabled);
    assert!(
        !init.ssao_enabled,
        "screen-space effects trimmed with the scene"
    );
    // No camera: the cursor is never captured.
    assert!(!s.saw(&Call::RequestCursorCapture));
}

// The launch's session overrides reach the backend: `--vsync` wins over the
// saved choice and the world's.
#[test]
fn launch_vsync_overrides_persisted_and_authored_config() {
    let mut settings = concinnity_engine::settings::StoredSettings::default();
    settings.graphics.vsync = Some(false);
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    world.insert_resource(settings);
    world.insert_resource(LaunchRequest {
        args: LaunchArgs {
            vsync: true,
            ..LaunchArgs::NONE
        },
        ..Default::default()
    });
    init_graphics(&mut world, hooks);
    assert!(lock(&state).init.as_ref().unwrap().vsync);
}

// The stored choices reach the backend over the world's authored values.
#[test]
fn persisted_settings_override_authored_config() {
    let mut settings = concinnity_engine::settings::StoredSettings::default();
    settings.graphics.quality_preset = Some(QualityPreset::Custom);
    settings.graphics.vsync = Some(true);
    settings.graphics.fps_cap = Some(30);
    settings.graphics.frames_in_flight = Some(3);
    settings.graphics.shadow_map_size = Some(1024);
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    world.insert_resource(settings);
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed);
    let s = lock(&state);
    let init = s.init.as_ref().unwrap();
    assert!(init.vsync, "persisted vsync overrides the authored false");
    assert_eq!(init.frames_in_flight, 3);
    assert_eq!(init.shadows.map_size, 1024, "explicit override wins");
}

#[test]
fn low_preset_ceiling_clamps_quality_knobs() {
    let mut settings = concinnity_engine::settings::StoredSettings::default();
    settings.graphics.quality_preset = Some(QualityPreset::Low);
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    world.insert_resource(settings);
    init_graphics(&mut world, hooks);

    // The stored preset's ceiling sizes the backend's shadows and filtering.
    let s = lock(&state);
    let init = s.init.as_ref().unwrap();
    assert_eq!(init.shadows.map_size, 1024);
    assert_eq!(init.shadows.cadence.distance, 40);
    assert_eq!(init.shadows.cadence.cascades, 2);
    assert_eq!(init.anisotropy, 4);
}

#[test]
fn a_world_without_a_shader_renders_with_the_engine_program() {
    let (state, hooks) = recording_hooks();
    let mut b = WorldBuilder::new();
    b.push(Window::default());
    b.push_textured_quad(MESH, TEX, MAT, PROP);
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed, "no Shader is the common case, not a failure");
    let s = lock(&state);
    let init = s.init.as_ref().expect("backend constructed");
    // One bucket carrying no programs: every backend reads that as "use the
    // engine's own main-pass program".
    assert_eq!(init.shader_program_counts, vec![0]);
}

// A world with a Shader hands every program the cook emitted to the backend.
#[test]
fn a_world_with_a_shader_hands_its_programs_to_the_backend() {
    let (state, hooks) = recording_hooks();
    let mut b = WorldBuilder::new();
    b.push(Window::default());
    b.push_shaders();
    b.push_textured_quad(MESH, TEX, MAT, PROP);
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed);
    let s = lock(&state);
    let init = s.init.as_ref().expect("backend constructed");
    assert_eq!(init.shader_program_counts, vec![2]);
}

#[test]
fn first_declared_scene_applies_start_visibility() {
    let scene_a = AssetId(20);
    let scene_b = AssetId(21);
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    // A second prop assigned to scene B; the first prop joins scene A.
    b.push_textured_quad(AssetId(5), AssetId(6), AssetId(7), AssetId(8));
    let mut world = b.build();
    {
        // Assign scenes on the two props before decomposition maps them to
        // SceneMember components.
        let mut ctx = world.context();
        let first = ctx.entity_of(PROP).expect("the first prop is identified");
        for (entity, prop) in ctx.query_mut_with_entity::<Prop>() {
            prop.scene = Some(if entity == first {
                Ref::new(scene_a)
            } else {
                Ref::new(scene_b)
            });
        }
        ctx.push_identified(scene_a, Scene { camera_shot: None });
        ctx.push_identified(scene_b, Scene { camera_shot: None });
    }
    let mut gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    {
        let s = lock(&state);
        assert_eq!(
            s.visibility.get(&DrawIndex(0)),
            Some(&true),
            "scene A prop visible"
        );
        assert_eq!(
            s.visibility.get(&DrawIndex(1)),
            Some(&false),
            "scene B prop hidden"
        );
    }

    // An imperative jump to scene B flips both.
    {
        let mut ctx = world.context();
        ctx.events_mut::<SceneCommand>().send(SceneCommand {
            scene: scene_b,
            transition: concinnity_engine::components::SceneTransition::Cut,
        });
    }
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    let s = lock(&state);
    assert_eq!(s.visibility.get(&DrawIndex(0)), Some(&false));
    assert_eq!(s.visibility.get(&DrawIndex(1)), Some(&true));
}
