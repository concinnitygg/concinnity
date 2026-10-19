use super::*;

// A log writer the tests read back.
#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Captured {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Captured {
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
    }
}

#[test]
fn jump_to_undeclared_scene_warns_and_changes_nothing() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    world
        .context()
        .push_identified(AssetId(20), Scene { camera_shot: None });
    let mut gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);
    lock(&state).visibility.clear();

    world
        .context()
        .events_mut::<SceneCommand>()
        .send(SceneCommand {
            scene: AssetId(99),
            transition: concinnity_core::components::SceneTransition::Cut,
        });
    let captured = Captured::default();
    let writer = captured.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_max_level(tracing::Level::WARN)
        .with_writer(move || writer.clone())
        .finish();
    let result = tracing::subscriber::with_default(subscriber, || step(&mut gs, &mut world));

    assert_eq!(result, StepResult::Continue);
    let log = captured.text();
    assert!(
        log.contains("scene jump to 99 rejected: UnknownScene"),
        "{log}"
    );
    assert!(lock(&state).visibility.is_empty());
}

// Each frame's extract adopts the overlay draw list wholesale and hands the
// spent one back through `OverlayRecycle`, which the next overlay build
// consumes -- the loop that keeps the overlay's buffers recycling.
#[test]
fn extract_hands_the_spent_overlay_list_back() {
    let (_state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    for _ in 0..2 {
        assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
        assert!(
            world
                .resource::<concinnity_core::render::overlay::OverlayRecycle>()
                .is_some(),
            "extraction returned the spent draw list"
        );
    }
}

#[test]
fn frame_steps_draw_and_publish_input() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    for _ in 0..2 {
        assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    }
    // take_input drains the snapshot per poll, so seed the third frame's
    // input just before the step that should publish it.
    lock(&state).next_input.mouse_x = 33.0;
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    assert_eq!(gs.frames.frames(), 3);

    let s = lock(&state);
    assert_eq!(s.draw_frames(), 3);
    assert_eq!(
        s.calls
            .iter()
            .filter(|c| matches!(c, Call::UpdateView(_)))
            .count(),
        3
    );
    assert_eq!(s.calls.iter().filter(|c| **c == Call::TakeInput).count(), 3);
    drop(s);

    // The polled input snapshot is republished as the FrameInput component +
    // resource for the camera / UI systems.
    let ctx = world.context();
    let inputs: Vec<&FrameInput> = ctx.query::<FrameInput>().collect();
    assert_eq!(inputs.len(), 1, "previous snapshot drained, one deposited");
    let res = ctx.resource::<FrameInput>().expect("resource published");
    assert_eq!(res.mouse_x, 33.0);
    assert_eq!(res.viewport, [1280.0, 720.0]);
    assert!(
        ctx.resource::<MenuActive>().is_some(),
        "menu state published every frame"
    );
}

// A serial frame submits on the stepping thread, and the whole submission is
// the render side's time rather than the graphics system's own.
#[test]
fn a_serial_frame_records_its_submission_as_the_render_handoff() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);
    lock(&state).on_draw = Some(Box::new(|| {
        std::thread::sleep(std::time::Duration::from_millis(3))
    }));
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);

    let ctx = world.context();
    ctx.profile.record_system("GraphicsSystem", 0);
    ctx.profile.begin_frame();
    let handoffs = ctx.profile.render_handoffs();
    assert_eq!(handoffs.len(), 1);
    assert_eq!(handoffs[0].0, "GraphicsSystem");
    assert!(handoffs[0].1 >= 3_000, "{} us", handoffs[0].1);
}

#[test]
fn camera_state_reaches_the_backend_each_frame() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    step(&mut gs, &mut world);
    // Move the camera the way Camera3DSystem would (write position + view).
    let moved_view = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-7.0, -8.0, -9.0, 1.0],
    ];
    {
        let mut ctx = world.context();
        let cam = ctx.query_mut::<Camera3D>().next().unwrap();
        cam.position = [7.0, 8.0, 9.0];
        cam.view_matrix = moved_view;
    }
    step(&mut gs, &mut world);

    let s = lock(&state);
    match s.last_draw_frame() {
        Some(Call::DrawFrame { cam_pos, .. }) => assert_eq!(cam_pos, [7.0, 8.0, 9.0]),
        other => panic!("expected a DrawFrame, got {other:?}"),
    }
    assert!(
        s.saw(&Call::UpdateView(moved_view)),
        "the moved view matrix reaches update_view"
    );
}

#[test]
fn transform_edit_pushes_new_model_matrix() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    {
        let mut ctx = world.context();
        let t = ctx.query_mut::<Transform>().next().unwrap();
        t.position = [10.0, 20.0, 30.0];
    }
    step(&mut gs, &mut world);

    let s = lock(&state);
    assert!(s.saw(&Call::UpdateModel(DrawIndex(0))));
    let model = s.models.get(&DrawIndex(0)).expect("slot 0 model pushed");
    assert_eq!(model[3][0], 10.0);
    assert_eq!(model[3][1], 20.0);
    assert_eq!(model[3][2], 30.0);
}

// The model-push change gate: a slot is sent when its matrix differs from the
// last push, so a static frame sends nothing and a move sends exactly one
// update for the moved slot.
#[test]
fn model_matrices_push_only_on_change() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    let model_pushes = |s: &RecordingState| {
        s.calls
            .iter()
            .filter(|c| matches!(c, Call::UpdateModel(_)))
            .count()
    };

    // First frame: nothing was pushed before, so the quad's slot is sent.
    step(&mut gs, &mut world);
    assert_eq!(model_pushes(&lock(&state)), 1, "first frame seeds the slot");

    // Static frames send nothing.
    for _ in 0..2 {
        lock(&state).calls.clear();
        step(&mut gs, &mut world);
        assert_eq!(
            model_pushes(&lock(&state)),
            0,
            "static frame pushed a model"
        );
    }

    // A move sends exactly one update carrying the new matrix.
    {
        let mut ctx = world.context();
        let t = ctx.query_mut::<Transform>().next().unwrap();
        t.position = [4.0, 5.0, 6.0];
    }
    lock(&state).calls.clear();
    step(&mut gs, &mut world);
    {
        let s = lock(&state);
        assert_eq!(model_pushes(&s), 1, "one move, one update");
        assert_eq!(s.models.get(&DrawIndex(0)).expect("slot 0")[3][0], 4.0);
    }

    // And the frame after the move is quiet again.
    lock(&state).calls.clear();
    step(&mut gs, &mut world);
    assert_eq!(
        model_pushes(&lock(&state)),
        0,
        "settled frame pushed a model"
    );
}

#[test]
fn opaque_menu_backdrop_hides_world_and_freezes_gameplay_input() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    // UI presence (a HitRegion) flips the world into menu mode at init.
    b.push(HitRegion::default());
    // A visible, opaque, view-owned sprite spanning the whole 1280x720
    // reference canvas: a menu dim at full alpha.
    b.push_identified(
        AssetId(40),
        Sprite {
            x: 0.0,
            y: 0.0,
            width: 1280.0,
            height: 720.0,
            tint: [0.0, 0.0, 0.0, 1.0],
            visible: true,
            screen: Some(Ref::new(AssetId(41))),
            ..Default::default()
        },
    );
    let mut world = b.build();
    // The active-screen state UiInputSystem publishes when a world-pausing
    // screen (id 41) is open; the overlay derives menu_active from it.
    world.context().insert_resource(ScreenStack {
        layers: [(AssetId(41), 1)].into_iter().collect(),
        pauses_world: true,
        captures_input: true,
    });
    let mut gs = init_graphics(&mut world, hooks);
    assert!(lock(&state).saw(&Call::SetMenuMode(true)));

    lock(&state).next_input.forward = true;
    step(&mut gs, &mut world);

    {
        let s = lock(&state);
        match s.last_draw_frame() {
            Some(Call::DrawFrame { world_hidden, .. }) => {
                assert!(world_hidden, "opaque full-canvas backdrop skips the world");
            }
            other => panic!("expected a DrawFrame, got {other:?}"),
        }
        // Menu open: the camera capture is released each frame.
        assert!(s.saw(&Call::SetCameraCapture(false)));
    }
    {
        let ctx = world.context();
        // The runtime-level pacer clamps from this same resource next step.
        assert!(ctx.resource::<MenuActive>().unwrap().0);
        let input = ctx.resource::<FrameInput>().unwrap();
        assert!(!input.forward, "gameplay input frozen behind the menu");
    }

    // Dimming the backdrop below full alpha keeps the menu active but the
    // world visible again.
    {
        let mut ctx = world.context();
        let sprite = ctx.query_mut::<Sprite>().next().unwrap();
        sprite.tint[3] = 0.5;
    }
    step(&mut gs, &mut world);
    let s = lock(&state);
    match s.last_draw_frame() {
        Some(Call::DrawFrame { world_hidden, .. }) => {
            assert!(!world_hidden, "translucent dim keeps the world render");
        }
        other => panic!("expected a DrawFrame, got {other:?}"),
    }
}

// A `MenuOverride(Some(true))` forces edit mode on a plain first-person world
// (no menu UI): the backend is put into menu mode, the cursor is released, the
// freeze resource is set, and gameplay input is frozen -- everything a menu open
// would do, without the world declaring one.
#[test]
fn menu_override_true_forces_cursor_free_and_freezes_input() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);
    // A plain camera world starts in first-person capture, menu mode off.
    assert!(lock(&state).saw(&Call::SetMenuMode(false)));

    world.insert_resource(MenuOverride(Some(true)));
    lock(&state).next_input.forward = true;
    step(&mut gs, &mut world);

    let s = lock(&state);
    assert!(
        s.saw(&Call::SetMenuMode(true)),
        "override forces backend menu mode"
    );
    assert!(
        s.saw(&Call::SetCameraCapture(false)),
        "override releases the cursor"
    );
    drop(s);
    let ctx = world.context();
    assert!(
        ctx.resource::<MenuActive>().unwrap().0,
        "freeze resource set"
    );
    assert!(
        !ctx.resource::<FrameInput>().unwrap().forward,
        "gameplay input frozen in edit mode"
    );
}

// A `MenuOverride(Some(false))` forces play mode: the backend stays in menu mode
// (so an editor session's clicks still route sanely), but the cursor is captured
// and gameplay input runs -- the world plays behind the editor.
#[test]
fn menu_override_false_captures_cursor_and_runs_input() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    world.insert_resource(MenuOverride(Some(false)));
    lock(&state).next_input.forward = true;
    step(&mut gs, &mut world);

    let s = lock(&state);
    assert!(
        s.saw(&Call::SetCameraCapture(true)),
        "override captures the cursor"
    );
    drop(s);
    let ctx = world.context();
    assert!(!ctx.resource::<MenuActive>().unwrap().0, "not frozen");
    assert!(
        ctx.resource::<FrameInput>().unwrap().forward,
        "gameplay input runs in play mode"
    );
}

#[test]
fn window_close_stops_after_wait_idle() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    lock(&state).window_closed = true;
    assert_eq!(step(&mut gs, &mut world), StepResult::Stop);
    let s = lock(&state);
    assert!(s.saw(&Call::WaitIdle));
    assert_eq!(s.draw_frames(), 0, "no frame drawn after the close");
}

// An unclassified frame error skips frames up to its bound, then stops after
// draining the (still healthy) device.
#[test]
fn unclassified_draw_error_skips_bounded_then_stops() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    lock(&state).fail_draw = Some(error::RenderError::Other("boom".to_string()));
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    assert_eq!(step(&mut gs, &mut world), StepResult::Stop);
    assert!(lock(&state).saw(&Call::WaitIdle));
}

// A frame drawing successfully resets the failure streak, so intermittent
// errors never accumulate to a stop.
#[test]
fn successful_frame_resets_the_failure_streak() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    for _ in 0..3 {
        lock(&state).fail_draw = Some(error::RenderError::Other("boom".to_string()));
        assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
        assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
        lock(&state).fail_draw = None;
        assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    }
}

// Device loss stops immediately, and must NOT drain the GPU: waiting on a
// lost device blocks on a queue that can never signal.
#[test]
fn device_lost_stops_immediately_without_gpu_drain() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    lock(&state).fail_draw = Some(error::RenderError::DeviceLost {
        reason: error::DeviceLostReason::Removed,
        detail: "test".to_string(),
    });
    assert_eq!(step(&mut gs, &mut world), StepResult::Stop);
    assert!(
        !lock(&state).saw(&Call::WaitIdle),
        "must not wait on a lost device"
    );
}

// A swapchain mismatch is transient: the frame is skipped and the loop keeps
// running while the backend recreates the swapchain.
#[test]
fn swapchain_out_of_date_skips_the_frame() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    lock(&state).fail_draw = Some(error::RenderError::SwapchainOutOfDate);
    for _ in 0..5 {
        assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    }
}

// A frame the policy dropped never drew the history reset it decided, so the
// next frame draws it instead, and only the drawn one is logged.
#[test]
fn a_dropped_frame_hands_its_history_reset_to_the_next() {
    use concinnity_core::render::history_reset::{HistoryResetCauses, PendingHistoryReset};

    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);

    let captured = Captured::default();
    let writer = captured.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_max_level(tracing::Level::INFO)
        .with_writer(move || writer.clone())
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        PendingHistoryReset::raise(&mut world.context(), HistoryResetCauses::SETTINGS_CHANGE);
        lock(&state).fail_draw = Some(error::RenderError::SwapchainOutOfDate);
        assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
        lock(&state).fail_draw = None;
        lock(&state).calls.clear();
        for _ in 0..2 {
            assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
        }
    });
    let log = captured.text();
    assert_eq!(log.matches("temporal history reset: ").count(), 1, "{log}");
    let resets: Vec<bool> = lock(&state)
        .calls
        .iter()
        .filter_map(|c| match c {
            Call::DrawFrame { history_reset, .. } => Some(*history_reset),
            _ => None,
        })
        .collect();
    assert_eq!(resets, [true, false]);
}

// Device-memory exhaustion publishes the pressure resource the streaming
// valve consumes, and keeps running within its bound.
#[test]
fn out_of_device_memory_publishes_pressure_and_continues() {
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);

    lock(&state).fail_draw = Some(error::RenderError::OutOfDeviceMemory("test".to_string()));
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    let pressure = *world
        .context()
        .resource::<GpuMemoryPressure>()
        .expect("pressure resource published");
    assert_eq!(pressure.events, 2);
}

// Reaching the cap must Stop rather than report Done: `World::step` treats Done
// as "retire this system and carry on with the others", which would leave a real
// world running headlessly forever once the renderer removed itself.
#[test]
fn max_frames_stops_the_run() {
    let (_state, hooks) = recording_hooks();
    let mut b = WorldBuilder::new();
    b.push(Window::default());
    b.push_shaders();
    b.push(Camera3D::bake(Default::default()));
    b.push_textured_quad(MESH, TEX, MAT, PROP);
    let mut world = b.build();
    world.insert_resource(LaunchRequest {
        args: LaunchArgs {
            frames: Some(2),
            ..LaunchArgs::NONE
        },
        ..Default::default()
    });
    let mut gs = init_graphics(&mut world, hooks);

    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    assert_eq!(step(&mut gs, &mut world), StepResult::Stop);
}

// The pipelined cap counts frames handed to the render half, not feedback
// drained back from it. The drain is a non-blocking `try_recv`, so a
// feedback-driven count lags by however many frames the render half has not
// reported yet, and a capped run renders `max` or `max + 1` frames depending on
// which side wins the race -- which on a temporal world is a different image.
// The render half here never answers, so a lagging count would never reach the
// cap at all.
#[test]
fn pipelined_max_frames_counts_frames_sent() {
    let (_state, hooks) = recording_hooks();
    let mut b = WorldBuilder::new();
    b.push(Window::default());
    b.push_shaders();
    b.push(Camera3D::bake(Default::default()));
    b.push_textured_quad(MESH, TEX, MAT, PROP);
    let mut world = b.build();
    world.insert_resource(LaunchRequest {
        args: LaunchArgs {
            frames: Some(3),
            ..LaunchArgs::NONE
        },
        ..Default::default()
    });
    let mut gs = init_graphics(&mut world, hooks);

    let (snapshot_tx, snapshot_rx) = std::sync::mpsc::sync_channel(0);
    let (_feedback_tx, feedback_rx) = std::sync::mpsc::channel();
    world
        .context()
        .insert_resource(crate::ecs::PipelinedFrames::new(
            crate::ecs::PipelineChannels {
                snapshot_tx,
                feedback_rx,
            },
        ));
    // Stand in for the render half: receive every snapshot (the send is a
    // rendezvous, so an unread channel would deadlock the step) and report
    // nothing back.
    let drain = std::thread::spawn(move || snapshot_rx.into_iter().count());

    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    assert_eq!(step(&mut gs, &mut world), StepResult::Stop);

    world
        .context()
        .remove_resource::<crate::ecs::PipelinedFrames>();
    assert_eq!(drain.join().expect("drain thread"), 3);
}

// The System trait's init and step are the entry points `World::start` and
// the schedule use.
#[test]
fn system_trait_delegates_to_init_and_step() {
    use concinnity_core::ecs::System;
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = GraphicsSystem {
        test_hooks: Some(hooks),
        ..GraphicsSystem::new(None)
    };
    {
        let mut ctx = world.context();
        concinnity_core::ecs::decompose::run(&mut ctx);
        gs.init(&mut ctx);
    }
    assert!(
        lock(&state).init.is_some(),
        "System::init built the backend"
    );

    let result = {
        let mut ctx = world.context();
        gs.step(&mut ctx)
    };
    assert_eq!(result, StepResult::Continue);
    assert_eq!(lock(&state).draw_frames(), 1, "System::step drew a frame");
    // The Debug impl surfaces the two fields a failure report needs.
    let shown = format!("{gs:?}");
    assert!(shown.contains("frame_count: 1"), "got {shown}");
    assert!(shown.contains("failed: false"), "got {shown}");
}

// A world declaring no Window or settings assets still builds against the system's
// defaults: the components are optional, and their absence is not an error.
#[test]
fn a_world_without_a_window_or_config_builds_on_the_defaults() {
    let (state, hooks) = recording_hooks();
    let mut b = WorldBuilder::new();
    b.push_shaders();
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed);
    let s = lock(&state);
    let init = s.init.as_ref().unwrap();
    assert_eq!(init.frames_in_flight, 2, "the system's own default");
    assert_eq!(init.window_width, Window::default().width);
}

// The step exits Done when the backend has been taken out of the world (the
// editor's live transplant), rather than drawing against nothing.
#[test]
fn a_step_without_a_parked_backend_finishes_the_system() {
    let (_state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);
    concinnity_core::ecs::ActiveRenderBackend::take(world.context().resources)
        .expect("backend was parked");

    assert_eq!(step(&mut gs, &mut world), StepResult::Done);
}

// Viewport picking is opt-in: with a PickIndex resource present at init (the
// editor injects one), init captures a candidate per prop and the step
// publishes the index with world-space bounds; without it, nothing is captured
// and nothing is published.
#[test]
fn pick_index_publishes_world_bounds_only_when_opted_in() {
    // Opted in: the quad prop at [1,2,3] indexes with its translated bounds.
    let (_state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    world.insert_resource(PickIndex::default());
    let mut gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);
    assert_eq!(
        gs.extractor.pick_candidates().len(),
        1,
        "one candidate per prop"
    );
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);

    let index = world
        .resource::<PickIndex>()
        .expect("step publishes the index");
    assert_eq!(index.entries.len(), 1);
    let e = &index.entries[0];
    assert_eq!(e.asset_id, PROP);
    // Unit quad local bounds x/z in [0,1], y = 0, translated by [1,2,3].
    assert_eq!(e.bb_min, [1.0, 2.0, 3.0]);
    assert_eq!(e.bb_max, [2.0, 2.0, 4.0]);

    // Not opted in: no candidates, and the resource never appears.
    let (_state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);
    assert!(gs.extractor.pick_candidates().is_empty());
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    assert!(world.resource::<PickIndex>().is_none());
}

// The editor's per-frame HiddenAssets set collapses a hidden prop's draw slots
// to a degenerate model matrix (nothing rasterizes) and drops it from the
// published PickIndex; clearing the set restores both the next step.
#[test]
fn editor_hidden_collapses_draws_and_skips_the_pick_index() {
    const COLLAPSED: [[f32; 4]; 4] = [[0.0; 4], [0.0; 4], [0.0; 4], [0.0, 0.0, 0.0, 1.0]];
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    world.insert_resource(PickIndex::default());
    let mut gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    world.insert_resource(HiddenAssets([PROP].into_iter().collect()));
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    {
        let index = world.resource::<PickIndex>().unwrap();
        assert!(index.entries.is_empty(), "a hidden prop is not pickable");
        let s = lock(&state);
        let model = s.models.get(&DrawIndex(0)).expect("slot 0 model pushed");
        assert_eq!(*model, COLLAPSED, "the hidden prop's slot is degenerate");
    }

    world.insert_resource(HiddenAssets::default());
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    let index = world.resource::<PickIndex>().unwrap();
    assert_eq!(index.entries.len(), 1, "clearing the set restores the prop");
    let s = lock(&state);
    let model = s.models.get(&DrawIndex(0)).unwrap();
    assert_ne!(*model, COLLAPSED, "the real transform is pushed again");
}

// Under the editor a skinned mesh template is pickable and movable like a
// prop: its pose entity carries the authored Transform, joins the pick index
// with its world-space bounds, and a Transform edit reaches the backend as a
// skinned model update the next step.
#[test]
fn skinned_mesh_joins_the_pick_index_when_opted_in() {
    use concinnity_core::components::SkinnedMesh;

    const BODY: AssetId = AssetId(842);
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    push_skinned_mesh(
        &mut b,
        BODY,
        SkinnedMesh {
            material: Some(MaterialHandle::new(0)),
            position: [5.0, 0.0, 0.0],
            ..Default::default()
        },
        4,
    );
    let mut world = b.build();
    world.insert_resource(PickIndex::default());
    let mut gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);
    assert_eq!(
        gs.extractor.pick_candidates().len(),
        2,
        "the prop and the skinned mesh"
    );
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    let index = world
        .resource::<PickIndex>()
        .expect("step publishes the index");
    let body = index
        .entries
        .iter()
        .find(|e| e.asset_id == BODY)
        .expect("the skinned mesh indexes");
    assert!(body.bb_min[0] >= 5.0, "bounds follow the authored position");

    // Move the template: the live Transform drives the skinned model push.
    let entity = world
        .context()
        .join2::<SkeletonPose, Transform>()
        .map(|(e, _, _)| e)
        .next()
        .expect("the template pose carries a Transform");
    world
        .context()
        .get_mut::<Transform>(entity)
        .unwrap()
        .position = [7.0, 0.0, 0.0];
    assert_eq!(step(&mut gs, &mut world), StepResult::Continue);
    assert!(
        lock(&state)
            .calls
            .iter()
            .any(|c| matches!(c, Call::UpdateSkinnedModel(SkinnedIndex(0)))),
        "the moved template reaches the backend"
    );
    let index = world.resource::<PickIndex>().unwrap();
    let body = index.entries.iter().find(|e| e.asset_id == BODY).unwrap();
    assert!(body.bb_min[0] >= 7.0, "the index follows the move");

    // Not opted in: the template stays a bare pose.
    let (_state, hooks) = recording_hooks();
    let mut b = scene_builder();
    push_skinned_mesh(&mut b, BODY, SkinnedMesh::default(), 4);
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);
    assert!(gs.extractor.pick_candidates().is_empty());
    assert_eq!(
        world.context().join2::<SkeletonPose, Transform>().count(),
        0
    );
}
