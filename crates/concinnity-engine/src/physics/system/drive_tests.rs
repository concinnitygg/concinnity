use super::*;
use alloc::string::String;

use crate::components::{
    CameraTrackArgs, CameraTravel, PropCollider, PropColliderShape, RollingDrive,
};
use crate::ecs::World;
use crate::planet::{
    FrameRebases, LocalFrame, PlanetGravity, PlanetGround, PlanetShape, Rebase, ground_patch,
};
use concinnity_core::math::narrow_f32;
use concinnity_core::math::vec3::{dot, length, sub};

const RADIUS: f32 = 0.5;

// A driven half-meter ball standing on the ground at `position`.
fn driven_ball(world: &mut World, position: [f32; 3]) -> Entity {
    let entity = world.push(Transform {
        position,
        ..Default::default()
    });
    world.insert(
        entity,
        Collider(PropCollider {
            shape: PropColliderShape::Ball,
            half_extents: [RADIUS; 3],
            radius: RADIUS,
            half_height: 0.0,
            layer: String::new(),
        }),
    );
    world.insert(
        entity,
        BodyDynamics {
            mass: 10.0,
            friction: 0.9,
            ..Default::default()
        },
    );
    world.insert(
        entity,
        RollingControl::new(&RollingDrive {
            acceleration: 6.0,
            max_speed: 4.0,
            jump_speed: 5.0,
            ..RollingDrive::default()
        }),
    );
    world.identify(entity, AssetId(7));
    entity
}

fn started(world: &mut World) -> PhysicsSystem {
    world.insert_resource(FrameRebases::default());
    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());
    physics
}

fn steps(physics: &mut PhysicsSystem, world: &mut World, count: usize) {
    for _ in 0..count {
        physics.step(&mut world.context());
    }
}

fn control(world: &World, entity: Entity) -> RollingControl {
    *world.get::<RollingControl>(entity).unwrap()
}

fn steer(world: &mut World, entity: Entity, heading: [f32; 3]) {
    world
        .context()
        .get_mut::<RollingControl>(entity)
        .unwrap()
        .heading = heading;
}

fn velocity(physics: &PhysicsSystem, entity: Entity) -> [f32; 3] {
    let handle = physics.props.handle_of(entity).unwrap();
    let world = physics.world.as_ref().unwrap();
    world.body_state(handle).unwrap().linear_velocity
}

// Steered along +X on level ground the ball rolls there, grounded, picking
// up speed toward the drive's top speed and not past it.
#[test]
fn a_driven_ball_rolls_along_its_heading_to_its_top_speed() {
    let mut world = World::new();
    let ball = driven_ball(&mut world, [0.0, RADIUS, 0.0]);
    let mut physics = started(&mut world);
    steps(&mut physics, &mut world, 30);
    steer(&mut world, ball, [1.0, 0.0, 0.0]);
    steps(&mut physics, &mut world, 30);
    let v = velocity(&physics, ball);
    assert!(v[0] > 1.5, "picking up speed: {v:?}");
    assert!(v[2].abs() < 1e-3, "straight along the heading: {v:?}");
    assert!(control(&world, ball).grounded);
    steps(&mut physics, &mut world, 600);
    let top = velocity(&physics, ball)[0];
    assert!((3.0..4.05).contains(&top), "held at the top speed: {top}");
    let at = world.get::<Transform>(ball).unwrap().position;
    assert!(
        (at[1] - RADIUS).abs() < 0.03,
        "rolling, not hopping: {at:?}"
    );
}

// A jump leaves the ground only from it. One asked for in the air waits a
// moment for the landing and is dropped past it, and every ask is taken from
// the control as it is read.
#[test]
fn a_jump_leaves_the_ground_only_from_it() {
    let mut world = World::new();
    let ball = driven_ball(&mut world, [0.0, RADIUS, 0.0]);
    let mut physics = started(&mut world);
    steps(&mut physics, &mut world, 30);
    world
        .context()
        .get_mut::<RollingControl>(ball)
        .unwrap()
        .jump = true;
    steps(&mut physics, &mut world, 1);
    assert!(!control(&world, ball).jump, "taken");
    assert!(
        velocity(&physics, ball)[1] > 4.0,
        "{:?}",
        velocity(&physics, ball)
    );
    // At the top of the jump, far longer than the window from the ground.
    steps(&mut physics, &mut world, 10);
    assert!(!control(&world, ball).grounded);
    world
        .context()
        .get_mut::<RollingControl>(ball)
        .unwrap()
        .jump = true;
    steps(&mut physics, &mut world, 120);
    let y = world.get::<Transform>(ball).unwrap().position[1];
    assert!(
        (y - RADIUS).abs() < 0.03,
        "dropped, not jumped on landing: {y}"
    );
    assert!(velocity(&physics, ball)[1].abs() < 0.1);
}

// Frames that run no tick keep a jump for the next tick to take; frames
// that run several take it once.
#[test]
fn a_jump_is_taken_once_however_frames_group_their_ticks() {
    let mut world = World::new();
    let ball = driven_ball(&mut world, [0.0, RADIUS, 0.0]);
    let mut physics = started(&mut world);
    steps(&mut physics, &mut world, 30);
    world
        .context()
        .get_mut::<RollingControl>(ball)
        .unwrap()
        .jump = true;
    let frame = |ticks| SimTiming {
        ticks,
        tick_dt: SimTiming::TICK_DT,
        alpha: 1.0,
        ..Default::default()
    };
    world.insert_resource(frame(0));
    steps(&mut physics, &mut world, 2);
    assert!(velocity(&physics, ball)[1].abs() < 0.1, "no tick, no jump");
    world.insert_resource(frame(5));
    steps(&mut physics, &mut world, 1);
    let rising = velocity(&physics, ball)[1];
    assert!(rising > 2.0 && rising < 5.0, "jumped once: {rising}");
}

// A planet small enough that its ground curves within a few hundred meters.
const SHAPE: PlanetShape = PlanetShape {
    center: [0.0, -5_000.0, 0.0],
    radius: 5_000.0,
    amplitude: 0.0,
    feature_size: 100.0,
    octaves: 1,
    seed: 0,
    continents: crate::planet::Continents::NONE,
};

fn publish_planet(world: &mut World, frame: &LocalFrame, around: [f32; 3], revision: u64) {
    let patch = ground_patch(&SHAPE, frame, around, 64, 32.0).unwrap();
    world.insert_resource(PlanetGround { revision, patch });
    world.insert_resource(PlanetGravity {
        center: frame.to_local(SHAPE.center),
        strength: 20.0,
    });
}

fn up_of(p: [f32; 3]) -> [f32; 3] {
    let d: [f32; 3] = core::array::from_fn(|i| p[i] - narrow_f32(SHAPE.center[i]));
    let n = length(d);
    d.map(|c| c / n)
}

// On a planet's side, where up leans well off +Y, the drive rolls the ball
// along the ground toward its heading: it keeps to the surface and goes the
// way it was steered.
#[test]
fn on_a_sphere_the_drive_rolls_along_the_ground() {
    let mut world = World::new();
    // 600 m from the pole, where up leans about 7 degrees off +Y.
    let start = [600.0, 0.0, 0.0];
    let up = up_of([600.0, 0.0, 0.0]);
    let surface: [f32; 3] =
        core::array::from_fn(|i| narrow_f32(SHAPE.center[i]) + up[i] * (5_000.0 + RADIUS));
    let ball = driven_ball(&mut world, surface);
    publish_planet(&mut world, &LocalFrame::AUTHORED, start, 0);
    let mut physics = started(&mut world);
    steps(&mut physics, &mut world, 30);
    // Steer along the ground away from the pole.
    let heading = [up[1], -up[0], 0.0];
    steer(&mut world, ball, heading);
    steps(&mut physics, &mut world, 90);
    let at = world.get::<Transform>(ball).unwrap().position;
    let moved: [f32; 3] = core::array::from_fn(|i| at[i] - surface[i]);
    assert!(
        dot(moved, heading) > 2.0,
        "rolled the way it was steered: {moved:?}"
    );
    assert!(moved[2].abs() < 0.05, "and not off it: {moved:?}");
    let height = length(core::array::from_fn(|i| {
        at[i] - narrow_f32(SHAPE.center[i])
    })) - 5_000.0;
    assert!((height - RADIUS).abs() < 0.05, "on the ground: {height}");
    assert!(control(&world, ball).grounded);
}

// A frame move turns a heading set before it with the frame, so the ball
// keeps rolling the same way over the ground.
#[test]
fn a_frame_move_turns_the_heading_with_the_frame() {
    let mut world = World::new();
    let ball = driven_ball(&mut world, [0.0, RADIUS, 0.0]);
    let mut physics = started(&mut world);
    steer(&mut world, ball, [1.0, 0.0, 0.0]);
    let half = 0.3_f32.to_radians() * 0.5;
    let rebase = Rebase {
        rotation: [0.0, 0.0, half.sin(), half.cos()],
        translation: [0.0; 3],
    };
    let mut rebases = FrameRebases::default();
    rebases.push(rebase);
    world.insert_resource(rebases);
    steps(&mut physics, &mut world, 1);
    let heading = control(&world, ball).heading;
    let expected = rebase.apply_vector([1.0, 0.0, 0.0]);
    assert!(
        (0..3).all(|i| (heading[i] - expected[i]).abs() < 1e-6),
        "{heading:?}"
    );
}

// A body a camera track carries follows the track kinematically: gravity
// does not pull it off the path, no drive steers it, and it pushes what it
// runs into. Released, it is simulated again at the speed it was carried.
#[test]
fn a_tracked_subject_moves_kinematically_with_the_track() {
    use crate::camera::track::CameraTrackSystem;
    let mut world = World::new();
    world.push(Camera3D::bake(crate::components::cook::Camera3D {
        position: [0.0, 5.0, 6.0],
        controller: None,
        ..Default::default()
    }));
    let ball = driven_ball(&mut world, [0.0, 3.0, 0.0]);
    steer(&mut world, ball, [0.0, 0.0, 1.0]);
    let track = CameraTrack::bake(CameraTrackArgs {
        travel: alloc::vec![CameraTravel {
            direction: [1.0, 0.0, 0.0],
            distance: 6.0,
            speed: 3.0,
            ..Default::default()
        }],
        subject: Some(crate::ecs::Ref::new(AssetId(7))),
        ..Default::default()
    });
    let mut tracking = CameraTrackSystem::new(&track);
    let track_entity = world.push(track);
    let mut physics = started(&mut world);
    tracking.init(&mut world.context());
    for _ in 0..60 {
        physics.step(&mut world.context());
        tracking.step(&mut world.context());
    }
    let handle = physics.props.handle_of(ball).unwrap();
    let sim = physics.world.as_ref().unwrap();
    assert_eq!(sim.is_kinematic(handle), Some(true));
    let (at, _) = sim.body_pose(handle).unwrap();
    assert!(
        (at[0] - 3.0).abs() < 0.1,
        "a second along the track: {at:?}"
    );
    assert!((at[1] - 3.0).abs() < 1e-4 && at[2].abs() < 1e-4, "{at:?}");

    world.despawn(track_entity);
    physics.step(&mut world.context());
    let sim = physics.world.as_ref().unwrap();
    assert_eq!(sim.is_kinematic(handle), Some(false));
    let v = velocity(&physics, ball);
    assert!(
        (v[0] - 3.0).abs() < 0.3,
        "launched at the track's speed: {v:?}"
    );
}

// Held through a hop at the top speed, the drive winds up no spin in the
// air, so the ball lands no faster than the top speed.
#[test]
fn spin_wound_up_in_the_air_does_not_land_past_the_top_speed() {
    let mut world = World::new();
    let ball = driven_ball(&mut world, [0.0, RADIUS, 0.0]);
    let mut physics = started(&mut world);
    steps(&mut physics, &mut world, 30);
    steer(&mut world, ball, [1.0, 0.0, 0.0]);
    steps(&mut physics, &mut world, 600);
    world
        .context()
        .get_mut::<RollingControl>(ball)
        .unwrap()
        .jump = true;
    let mut fastest = 0.0_f32;
    for _ in 0..240 {
        steps(&mut physics, &mut world, 1);
        let v = velocity(&physics, ball);
        fastest = fastest.max(length([v[0], 0.0, v[2]]));
    }
    assert!(!control(&world, ball).jump);
    assert!(
        fastest < 4.0 * 1.03,
        "landed at {fastest} past the top speed"
    );
}

// A second press just after leaving the ground cannot double the jump: the
// contact it left no longer holds it up, and a jump only ever brings the
// speed away from the ground up to the jump speed.
#[test]
fn a_second_press_cannot_double_a_jump() {
    let mut world = World::new();
    let ball = driven_ball(&mut world, [0.0, RADIUS, 0.0]);
    let mut physics = started(&mut world);
    steps(&mut physics, &mut world, 30);
    for _ in 0..3 {
        world
            .context()
            .get_mut::<RollingControl>(ball)
            .unwrap()
            .jump = true;
        steps(&mut physics, &mut world, 1);
    }
    let up = velocity(&physics, ball)[1];
    assert!(up <= 5.0 + 1e-3, "rising at {up}, past the jump speed");
    // One jump, then three ticks of gravity.
    assert!(up > 3.9, "{up}");
}

// A world whose camera track carries the driven ball, run with its track
// system; `parented` hangs the ball under another entity, which the track
// refuses.
fn tracked_world(
    parented: bool,
) -> (
    World,
    PhysicsSystem,
    crate::camera::track::CameraTrackSystem,
    Entity,
    Entity,
) {
    use crate::camera::track::CameraTrackSystem;
    let mut world = World::new();
    world.push(Camera3D::bake(crate::components::cook::Camera3D {
        position: [0.0, 5.0, 6.0],
        controller: None,
        ..Default::default()
    }));
    let ball = driven_ball(&mut world, [0.0, 3.0, 0.0]);
    if parented {
        let parent = world.push(Transform::default());
        world.insert(ball, crate::components::Parent(parent));
    }
    let track = CameraTrack::bake(CameraTrackArgs {
        travel: alloc::vec![CameraTravel {
            direction: [1.0, 0.0, 0.0],
            distance: 6.0,
            speed: 3.0,
            ..Default::default()
        }],
        subject: Some(crate::ecs::Ref::new(AssetId(7))),
        ..Default::default()
    });
    let mut tracking = CameraTrackSystem::new(&track);
    let track_entity = world.push(track);
    let physics = started(&mut world);
    tracking.init(&mut world.context());
    (world, physics, tracking, ball, track_entity)
}

// Frames that run no tick leave a posed body where it was; the next frame
// moves it on from there, never back, and it reaches each pose its owner
// set by the frame's last tick.
#[test]
fn a_posed_body_moves_on_only_from_where_it_got_to() {
    let (mut world, mut physics, mut tracking, ball, _) = tracked_world(false);
    let handle = physics.props.handle_of(ball).unwrap();
    let frame = |ticks| SimTiming {
        ticks,
        tick_dt: SimTiming::TICK_DT,
        alpha: 1.0,
        ..Default::default()
    };
    let mut last_x = 0.0_f32;
    for ticks in [1, 0, 0, 3, 1, 0, 2, 1, 1, 0, 4]
        .into_iter()
        .cycle()
        .take(60)
    {
        world.insert_resource(frame(ticks));
        physics.step(&mut world.context());
        let x = physics.world.as_ref().unwrap().body_pose(handle).unwrap().0[0];
        assert!(x >= last_x - 1e-5, "moved back from {last_x} to {x}");
        if ticks > 0 {
            let owned = world.get::<Transform>(ball).unwrap().position[0];
            assert!((x - owned).abs() < 1e-4, "{x} short of {owned}");
        }
        last_x = x;
        tracking.step(&mut world.context());
    }
}

// A jump asked of a body the track holds is dropped, not fired when the
// track lets go.
#[test]
fn a_jump_asked_while_posed_is_dropped() {
    let (mut world, mut physics, mut tracking, ball, track) = tracked_world(false);
    for _ in 0..10 {
        physics.step(&mut world.context());
        tracking.step(&mut world.context());
    }
    world
        .context()
        .get_mut::<RollingControl>(ball)
        .unwrap()
        .jump = true;
    physics.step(&mut world.context());
    assert!(!control(&world, ball).jump);
    world.despawn(track);
    steps(&mut physics, &mut world, 2);
    assert!(velocity(&physics, ball)[1] < 0.0, "falling, not jumping");
}

// A subject the track refuses keeps its own body: it is never posed, so it
// is not left kinematic with nothing posing it.
#[test]
fn a_subject_the_track_refuses_stays_dynamic() {
    let (mut world, mut physics, mut tracking, ball, _) = tracked_world(true);
    for _ in 0..10 {
        physics.step(&mut world.context());
        tracking.step(&mut world.context());
    }
    let handle = physics.props.handle_of(ball).unwrap();
    let sim = physics.world.as_ref().unwrap();
    assert_eq!(sim.is_kinematic(handle), Some(false));
    assert!(velocity(&physics, ball)[1] < -1.0, "it falls");
}

// A scripted roll over the planet's ground: west, then a jump turning north,
// then half throttle south.
fn scripted_roll(world: &mut World) -> Entity {
    let ball = driven_ball(world, [0.0, RADIUS, 0.0]);
    let key = |seconds, heading_deg, throttle, jump| crate::components::DriveKey {
        seconds,
        heading_deg,
        throttle,
        sprint: true,
        jump,
    };
    world.insert(
        ball,
        crate::components::DriveScript::new(
            &[
                key(0.5, 270.0, 1.0, false),
                key(2.0, 0.0, 1.0, true),
                key(4.0, 180.0, 0.5, false),
            ],
            0.0,
        ),
    );
    ball
}

// The scripted ball's body after `frames`, each running that many ticks, on a
// planet patch, and the sim clock the frames reached.
fn roll_with(frames: &[u32]) -> ([f32; 3], [f32; 4], [f32; 3]) {
    let mut world = World::new();
    publish_planet(&mut world, &LocalFrame::AUTHORED, [0.0; 3], 0);
    let ball = scripted_roll(&mut world);
    let mut physics = started(&mut world);
    let mut time = 0.0_f64;
    for &ticks in frames {
        time += f64::from(ticks) * f64::from(SimTiming::TICK_DT);
        world.insert_resource(SimTiming {
            ticks,
            tick_dt: SimTiming::TICK_DT,
            alpha: 0.5,
            time,
        });
        physics.step(&mut world.context());
    }
    let handle = physics.props.handle_of(ball).unwrap();
    let state = physics.world.as_ref().unwrap().body_state(handle).unwrap();
    (state.position, state.rotation, state.linear_velocity)
}

// The script is read at each tick's own moment, so the roll lands on exactly
// the same state however frames group the ticks, and run after run.
#[test]
fn a_scripted_roll_repeats_exactly_however_ticks_are_grouped() {
    let even = [1; 360];
    let mut lumpy = Vec::new();
    let mut sum = 0;
    for ticks in [0, 2, 3, 1, 0, 4, 2].into_iter().cycle() {
        if sum == 360 {
            break;
        }
        let ticks = ticks.min(360 - sum);
        sum += ticks;
        lumpy.push(ticks);
    }
    assert_eq!(lumpy.iter().sum::<u32>(), 360);
    let a = roll_with(&even);
    assert_eq!(a, roll_with(&even), "the same run twice");
    assert_eq!(a, roll_with(&lumpy), "ticks grouped differently");
    // And it went somewhere: west, a hop, north, then south.
    assert!(length([a.0[0], 0.0, a.0[2]]) > 3.0, "{:?}", a.0);
}

// A body a track picks up mid-roll stays simulated until then and becomes
// kinematic from then, without jumping.
#[test]
fn a_rolling_body_turns_kinematic_where_the_track_picks_it_up() {
    use crate::camera::track::CameraTrackSystem;
    let mut world = World::new();
    world.push(Camera3D::bake(crate::components::cook::Camera3D {
        position: [0.0, 3.0, 6.0],
        controller: None,
        ..Default::default()
    }));
    let ball = driven_ball(&mut world, [0.0, RADIUS, 0.0]);
    world.insert(
        ball,
        crate::components::DriveScript::new(
            &[crate::components::DriveKey {
                heading_deg: 90.0,
                ..Default::default()
            }],
            0.0,
        ),
    );
    let track = CameraTrack::bake(CameraTrackArgs {
        travel: alloc::vec![CameraTravel {
            direction: [1.0, 0.0, 0.0],
            distance: 60.0,
            speed: 6.0,
            ..Default::default()
        }],
        subject: Some(crate::ecs::Ref::new(AssetId(7))),
        carry_from_seconds: 2.0,
        ..Default::default()
    });
    let mut tracking = CameraTrackSystem::new(&track);
    world.push(track);
    let mut physics = started(&mut world);
    tracking.init(&mut world.context());
    let handle = physics.props.handle_of(ball).unwrap();
    let mut last = world.get::<Transform>(ball).unwrap().position;
    let mut kinematic_from = None;
    for frame in 0..240 {
        world.insert_resource(SimTiming {
            ticks: 1,
            tick_dt: SimTiming::TICK_DT,
            alpha: 1.0,
            time: f64::from(frame + 1) * f64::from(SimTiming::TICK_DT),
        });
        physics.step(&mut world.context());
        tracking.step(&mut world.context());
        let sim = physics.world.as_ref().unwrap();
        if kinematic_from.is_none() && sim.is_kinematic(handle) == Some(true) {
            kinematic_from = Some(frame);
        }
        let at = world.get::<Transform>(ball).unwrap().position;
        assert!(length(sub(at, last)) < 0.25, "jumped at frame {frame}");
        last = at;
    }
    let from = kinematic_from.expect("picked up");
    assert!((119..=122).contains(&from), "kinematic from frame {from}");
}
