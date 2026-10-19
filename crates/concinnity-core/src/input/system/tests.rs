use super::*;
use crate::input::pad::{PadAxis, PadEvent};
use crate::input::snapshot::{InputPacket, InputSnapshot};

fn started(system: &mut InputSystem, world: &mut World) {
    system.init(&mut world.context());
}

fn deposit(world: &mut World, packet: InputPacket) {
    InputMailbox::deposit_in(world.context().resources, packet, 42);
}

fn walking() -> InputPacket {
    InputPacket {
        raw: InputSnapshot {
            forward: true,
            mouse_x: 10.0,
            mouse_y: 20.0,
            ..Default::default()
        },
        cursor_outside_window: true,
        viewport: (800.0, 600.0),
        top_inset: 28.0,
        window_minimized: true,
    }
}

fn published(world: &World) -> FrameInput {
    world
        .resource::<FrameInput>()
        .cloned()
        .expect("FrameInput published")
}

// A deposited packet becomes the tick's FrameInput, in both forms, with the
// cursor, the minimized flag and the sample time beside it.
#[test]
fn a_packet_publishes_the_frame_input_cursor_and_window_state() {
    let mut world = World::new();
    let mut system = InputSystem::new();
    started(&mut system, &mut world);
    deposit(&mut world, walking());
    system.step(&mut world.context());

    let input = published(&world);
    assert!(input.forward);
    assert_eq!(input.viewport, [800.0, 600.0]);
    assert_eq!(input.top_inset, 28.0);
    assert_eq!(
        world.query::<FrameInput>().next().map(|f| f.forward),
        Some(true)
    );
    let cursor = world.resource::<CursorState>().unwrap();
    assert_eq!((cursor.pos, cursor.outside_window), ((10.0, 20.0), true));
    assert_eq!(world.resource::<WindowMinimized>().map(|m| m.0), Some(true));
    assert_eq!(world.resource::<InputSampledAt>().map(|s| s.0), Some(42));
}

// No packet leaves the last snapshot in place.
#[test]
fn a_tick_without_a_packet_keeps_the_last_snapshot() {
    let mut world = World::new();
    let mut system = InputSystem::new();
    started(&mut system, &mut world);
    deposit(&mut world, walking());
    system.step(&mut world.context());
    system.step(&mut world.context());
    assert!(published(&world).forward);
}

// An open pausing menu freezes gameplay keys; the editor's fly camera keeps
// them live.
#[test]
fn a_pausing_menu_freezes_gameplay_input_unless_the_fly_camera_runs() {
    for (fly, live) in [(false, false), (true, true)] {
        let mut world = World::new();
        world.insert_resource(MenuActive(true));
        world.insert_resource(FlyCam(fly));
        let mut system = InputSystem::new();
        started(&mut system, &mut world);
        deposit(&mut world, walking());
        system.step(&mut world.context());
        assert_eq!(published(&world).forward, live, "fly camera {fly}");
    }
}

// A pad source that connects one pad and pushes its left stick half forward.
#[derive(Debug)]
struct Stick;

impl PadSource for Stick {
    fn poll(&mut self, state: &mut PadState) {
        state.ingest(PadEvent::Connected { pad: 0 });
        state.ingest(PadEvent::Axis {
            pad: 0,
            axis: PadAxis::LeftY,
            value: 0.5,
        });
    }
}

// The lent pads reach the snapshot's move axis, and a live deadzone change
// wider than the deflection silences them.
#[test]
fn lent_pads_reach_the_move_axis_through_the_live_deadzone() {
    let mut world = World::new();
    let mut system = InputSystem::new().with_pads(Box::new(Stick));
    started(&mut system, &mut world);
    deposit(&mut world, InputPacket::default());
    system.step(&mut world.context());
    assert!(published(&world).move_axis[1].abs() > 0.1);

    world.events_mut::<ControlsCommand>().send(ControlsCommand {
        gamepad_deadzone: Some(0.9),
        ..Default::default()
    });
    deposit(&mut world, InputPacket::default());
    system.step(&mut world.context());
    assert_eq!(published(&world).move_axis, [0.0, 0.0]);
}

// Stored controls replace the defaults at init.
#[test]
fn stored_controls_seed_the_map_and_deadzone() {
    let mut stored = StoredSettings::default();
    stored.controls.gamepad_deadzone = Some(0.4);
    let mut world = World::new();
    world.insert_resource(stored);
    let mut system = InputSystem::new();
    started(&mut system, &mut world);
    assert_eq!(system.deadzone, 0.4);
    assert_eq!(system.map, GamepadMap::default());
}

// A sampled packet carries whatever chrome the backend reports over the top of
// its frame; a backend reporting none leaves it at zero.
#[test]
fn a_sampled_packet_carries_the_backends_top_content_inset() {
    use crate::render::backend::recording::recording_backend;
    let (state, mut backend) = recording_backend();
    assert_eq!(InputPacket::sample(&mut backend).top_inset, 0.0);
    state.lock().top_inset = 28.0;
    assert_eq!(InputPacket::sample(&mut backend).top_inset, 28.0);
}
