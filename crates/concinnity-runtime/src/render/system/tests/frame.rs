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
            transition: concinnity_engine::components::SceneTransition::Cut,
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

    assert_eq!(
        ctx.query::<FrameInput>().count(),
        1,
        "previous snapshot drained, one deposited"
    );
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
        std::thread::sleep(std::time::Duration::from_millis(3));
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

// Init puts the backend in the menu mode the world's cursor policy resolves:
// UI presence (a HitRegion) beside a camera makes a menu-mode world, and a
// plain camera world starts in first-person capture.
#[test]
fn init_puts_the_backend_in_the_worlds_menu_mode() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(HitRegion::default());
    let mut world = b.build();
    let _gs = init_graphics(&mut world, hooks);
    assert!(lock(&state).saw(&Call::SetMenuMode(true)));

    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let _gs = init_graphics(&mut world, hooks);
    assert!(lock(&state).saw(&Call::SetMenuMode(false)));
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

// A frame the policy dropped never drew the history reset it decided, so the
// next frame draws it instead, and only the drawn one is logged.
#[test]
fn a_dropped_frame_hands_its_history_reset_to_the_next() {
    use concinnity_engine::render::temporal::history_reset::{
        HistoryResetCauses, PendingHistoryReset,
    };

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
        .insert_resource(crate::app::pipeline::PipelinedFrames::new(
            crate::app::pipeline::PipelineChannels {
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
        .remove_resource::<crate::app::pipeline::PipelinedFrames>();
    assert_eq!(drain.join().expect("drain thread"), 3);
}

// The System trait's init and step are the entry points `World::start` and
// the schedule use.
#[test]
fn system_trait_delegates_to_init_and_step() {
    use concinnity_engine::ecs::System;
    let (state, hooks) = recording_hooks();
    let mut world = scene_builder().build();
    let mut gs = GraphicsSystem {
        test_hooks: Some(hooks),
        ..GraphicsSystem::new(None)
    };
    {
        let mut ctx = world.context();
        concinnity_engine::ecs::decompose::run(&mut ctx);
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
    concinnity_engine::ecs::ActiveRenderBackend::take(world.context().resources)
        .expect("backend was parked");

    assert_eq!(step(&mut gs, &mut world), StepResult::Done);
}
