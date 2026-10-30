// A track with a subject: the prop travels, the camera frames it from the boom.

use super::*;
use crate::components::cook::Camera3D as Camera3DArgs;
use crate::components::{CameraBoom, CameraTrackArgs, CameraTravel, CameraTurn, Ease};
use crate::ecs::Ref;
use crate::planet::Rebase;
use alloc::vec;
use concinnity_core::approx::assert_close;
use concinnity_core::math::{SaturatingFrom, asin, atan2, quat_from_axis_angle};

const SUBJECT: AssetId = AssetId(7);

fn travel(direction: [f32; 3], distance: f32, speed: f32) -> CameraTravel {
    CameraTravel {
        direction,
        distance,
        speed,
        ..Default::default()
    }
}

fn boom(distance: Option<f32>, offset_deg: Option<[f32; 2]>, seconds: f32) -> CameraBoom {
    CameraBoom {
        distance,
        offset_deg,
        seconds,
        ..Default::default()
    }
}

fn track(travel: Vec<CameraTravel>, turn: Vec<CameraTurn>, boom: Vec<CameraBoom>) -> CameraTrack {
    CameraTrack::bake(CameraTrackArgs {
        travel,
        turn,
        subject: Some(Ref::new(SUBJECT)),
        boom,
        ..Default::default()
    })
}

// A world with the camera at `camera`, looking down -Z, and the subject prop
// at `subject`.
fn world_with(
    track: CameraTrack,
    camera: [f32; 3],
    subject: [f32; 3],
) -> (World, CameraTrackSystem) {
    let system = CameraTrackSystem::new(&track);
    let mut world = World::new();
    world.add_component(Camera3D::bake(Camera3DArgs {
        position: camera,
        controller: None,
        ..Default::default()
    }));
    world.add_component(track);
    world.push_identified(
        SUBJECT,
        Transform {
            position: subject,
            ..Default::default()
        },
    );
    (world, system)
}

fn frame(world: &mut World, system: &mut CameraTrackSystem, ticks: u32, alpha: f32) {
    world.context().insert_resource(SimTiming {
        ticks,
        tick_dt: SimTiming::TICK_DT,
        alpha,
        ..Default::default()
    });
    system.step(&mut world.context());
}

fn seconds(world: &mut World, system: &mut CameraTrackSystem, seconds: u32) {
    for _ in 0..seconds * 60 {
        frame(world, system, 1, 1.0);
    }
}

struct Pose {
    position: [f32; 3],
    yaw: f32,
    pitch: f32,
    up: [f32; 3],
}

fn camera(world: &World) -> Pose {
    let cam = world.query::<Camera3D>().next().expect("camera");
    Pose {
        position: cam.position,
        yaw: cam.yaw,
        pitch: cam.pitch,
        up: cam.up,
    }
}

fn subject(world: &World) -> [f32; 3] {
    world
        .get_by_id::<Transform>(SUBJECT)
        .expect("subject")
        .position
}

// The subject as the camera sees it: its distance, and how far right of and
// above the center of view it sits, in degrees.
fn framing(world: &World) -> [f32; 3] {
    let cam = camera(world);
    let to = vec3::sub(subject(world), cam.position);
    let distance = vec3::length(to);
    let d = vec3::scale(to, 1.0 / distance);
    let [right, up, back] = view_axes(cam.yaw, cam.pitch, cam.up);
    let (x, y, z) = (vec3::dot(d, right), vec3::dot(d, up), -vec3::dot(d, back));
    [
        distance,
        atan2(x, z).to_degrees(),
        asin(y.clamp(-1.0, 1.0)).to_degrees(),
    ]
}

fn assert_framing(world: &World, expected: [f32; 3]) {
    let got = framing(world);
    assert!(
        (got[0] - expected[0]).abs() < 1e-3
            && (got[1] - expected[1]).abs() < 0.01
            && (got[2] - expected[2]).abs() < 0.01,
        "{got:?} vs {expected:?}"
    );
}

#[test]
fn travel_carries_the_subject_and_the_camera_frames_it_from_the_boom() {
    let (mut world, mut system) = world_with(
        track(
            vec![travel([1.0, 0.0, 0.0], 10.0, 5.0)],
            vec![CameraTurn {
                yaw_deg: Some(30.0),
                pitch_deg: Some(-10.0),
                seconds: 2.0,
                ..Default::default()
            }],
            vec![boom(Some(4.0), Some([3.0, -6.0]), 0.0)],
        ),
        [0.0, 1.0, 5.0],
        [0.0, 1.0, 0.0],
    );
    system.init(&mut world.context());
    assert_eq!(subject(&world), [0.0, 1.0, 0.0]);
    assert_framing(&world, [4.0, 3.0, -6.0]);
    seconds(&mut world, &mut system, 1);
    assert_close(subject(&world), [5.0, 1.0, 0.0], 1e-4);
    assert!((camera(&world).yaw.to_degrees() - 15.0).abs() < 1e-3);
    assert_framing(&world, [4.0, 3.0, -6.0]);
}

#[test]
fn without_boom_legs_the_camera_keeps_its_authored_distance_looking_straight_at_the_subject() {
    let (mut world, mut system) = world_with(
        track(vec![travel([0.0, 0.0, -1.0], 6.0, 3.0)], vec![], vec![]),
        [0.0, 2.0, 5.0],
        [0.0, 2.0, 0.0],
    );
    system.init(&mut world.context());
    assert_close(camera(&world).position, [0.0, 2.0, 5.0], 1e-5);
    seconds(&mut world, &mut system, 1);
    assert_close(camera(&world).position, [0.0, 2.0, 2.0], 1e-4);
    assert_framing(&world, [5.0, 0.0, 0.0]);
}

#[test]
fn boom_legs_blend_hold_and_cut() {
    let (mut world, mut system) = world_with(
        track(
            vec![],
            vec![],
            vec![
                boom(Some(2.0), Some([0.0, 0.0]), 1.0),
                CameraBoom {
                    distance: Some(6.0),
                    seconds: 2.0,
                    ease: Ease::Linear,
                    ..Default::default()
                },
                boom(None, Some([0.0, 5.0]), 0.0),
                boom(None, None, 0.5),
                CameraBoom {
                    distance: Some(10.0),
                    cut: true,
                    ..Default::default()
                },
                boom(None, None, 1.0),
            ],
        ),
        [0.0, 0.0, 9.0],
        [0.0; 3],
    );
    system.init(&mut world.context());
    // The boom opens on its first leg's distance, not the authored nine.
    assert_framing(&world, [2.0, 0.0, 0.0]);
    seconds(&mut world, &mut system, 1);
    assert_framing(&world, [2.0, 0.0, 0.0]);
    seconds(&mut world, &mut system, 1);
    assert_framing(&world, [4.0, 0.0, 0.0]);
    let mut cuts = 0;
    for _ in 0..90 {
        frame(&mut world, &mut system, 1, 0.5);
        if PendingHistoryReset::take(&mut world.context()).contains(HistoryResetCauses::CAMERA_CUT)
        {
            cuts += 1;
        }
    }
    assert_eq!(cuts, 2, "the instant offset change and the cut leg");
    // Opens on the first value a leg sets: a later offset holds until then.
    let (mut late, mut system) = world_with(
        track(
            vec![],
            vec![],
            vec![boom(None, None, 1.0), boom(None, Some([1.0, 2.0]), 1.0)],
        ),
        [0.0, 0.0, 9.0],
        [0.0; 3],
    );
    system.init(&mut late.context());
    assert_framing(&late, [9.0, 1.0, 2.0]);
    assert_framing(&world, [10.0, 0.0, 5.0]);
    assert!((status_duration(&world) - 4.5).abs() < 1e-5);
}

fn status_duration(world: &World) -> f32 {
    world
        .resource::<CameraTrackStatus>()
        .expect("published")
        .duration_seconds
}

#[test]
fn an_orbit_sweep_circles_the_camera_round_the_subject_and_keeps_it_framed() {
    let quarter = |yaw: f32| CameraTurn {
        yaw_deg: Some(yaw),
        pitch_deg: Some(-20.0),
        seconds: 1.0,
        ..Default::default()
    };
    let (mut world, mut system) = world_with(
        track(
            vec![travel([1.0, 0.0, 0.0], 40.0, 10.0)],
            vec![quarter(90.0), quarter(180.0), quarter(270.0), quarter(0.0)],
            vec![boom(Some(3.0), Some([-4.0, -8.0]), 0.0)],
        ),
        [0.0, 0.0, 3.0],
        [0.0; 3],
    );
    system.init(&mut world.context());
    let mut swept = 0.0;
    let mut last = camera(&world).yaw;
    for _ in 0..250 {
        frame(&mut world, &mut system, 1, 0.4);
        assert_framing(&world, [3.0, -4.0, -8.0]);
        let yaw = camera(&world).yaw;
        swept += (yaw - last).abs();
        last = yaw;
    }
    assert!(
        (swept.to_degrees() - 360.0).abs() < 0.5,
        "{}",
        swept.to_degrees()
    );
}

#[test]
fn a_missing_subject_leaves_the_track_driving_the_camera() {
    let track = track(vec![travel([1.0, 0.0, 0.0], 4.0, 2.0)], vec![], vec![]);
    let mut system = CameraTrackSystem::new(&track);
    let mut world = World::new();
    world.add_component(Camera3D::bake(Camera3DArgs {
        position: [0.0, 1.0, 0.0],
        controller: None,
        ..Default::default()
    }));
    world.add_component(track);
    system.init(&mut world.context());
    seconds(&mut world, &mut system, 1);
    assert_close(camera(&world).position, [2.0, 1.0, 0.0], 1e-4);
}

#[test]
fn a_subject_removed_mid_track_hands_travel_to_the_camera_where_it_stands() {
    let (mut world, mut system) = world_with(
        track(
            vec![travel([1.0, 0.0, 0.0], 10.0, 5.0)],
            vec![],
            vec![boom(Some(3.0), None, 0.0)],
        ),
        [0.0, 0.0, 3.0],
        [0.0; 3],
    );
    system.init(&mut world.context());
    seconds(&mut world, &mut system, 1);
    let before = camera(&world).position;
    let entity = world.entity_of(SUBJECT).expect("subject");
    world.despawn(entity);
    seconds(&mut world, &mut system, 1);
    // The camera carries on along the leg from where it stood, not from where
    // the subject was.
    assert_close(
        camera(&world).position,
        vec3::add(before, [5.0, 0.0, 0.0]),
        1e-3,
    );
}

// A subject flown far through frame moves that turn as well as shift the
// frame: it reaches where the legs add up to, and the camera holds the
// boom's framing on every frame, across every move.
#[test]
fn a_tracked_subject_reaches_the_end_of_its_track_through_frame_moves() {
    const LEGS: usize = 40;
    const LEG: f32 = 1_000.0;
    let (mut world, mut system) = world_with(
        track(
            (0..LEGS)
                .map(|_| travel([1.0, 0.0, 0.0], LEG, 3_000.0))
                .collect(),
            vec![CameraTurn {
                yaw_deg: Some(-60.0),
                pitch_deg: Some(-15.0),
                seconds: 5.0,
                ..Default::default()
            }],
            vec![boom(Some(3.5), Some([2.0, -9.0]), 0.0)],
        ),
        [0.0, 50.0, 3.5],
        [0.0, 50.0, 0.0],
    );
    world.insert_resource(FrameRebases::default());
    system.init(&mut world.context());

    // Every move turns the frame a little about +Z, as a planet's does.
    const TURN: f32 = 0.004;
    const FRAMES: [(u32, f32); 5] = [(1, 0.3), (2, 0.9), (0, 0.95), (5, 0.1), (1, 0.5)];
    let mut rebases = FrameRebases::default();
    // The world frame the run started in, mapped into the current one.
    let mut to_current = Rebase {
        rotation: [0.0, 0.0, 0.0, 1.0],
        translation: [0.0; 3],
    };
    let mut moves = Vec::new();
    for (ticks, alpha) in FRAMES.into_iter().cycle() {
        let at = subject(&world);
        if crate::planet::needs_rebase(at, LEG) {
            let rebase = Rebase {
                rotation: quat_from_axis_angle([0.0, 0.0, 1.0], TURN),
                translation: quat_rotate(
                    quat_from_axis_angle([0.0, 0.0, 1.0], TURN),
                    vec3::scale(at, -1.0),
                ),
            };
            if let Some(t) = world.get_mut_by_id::<Transform>(SUBJECT) {
                t.position = rebase.apply_point(t.position);
            }
            for camera in world.context().query_mut::<Camera3D>() {
                camera.position = rebase.apply_point(camera.position);
            }
            moves.push(rebase);
            rebases.push(rebase);
            world.insert_resource(rebases.clone());
        }
        frame(&mut world, &mut system, ticks, alpha);
        let [distance, right, up] = framing(&world);
        assert!((distance - 3.5).abs() < 1e-3, "{distance}");
        if status(&world).pose_seconds > 5.0 {
            assert!(
                (right - 2.0).abs() < 0.02 && (up + 9.0).abs() < 0.02,
                "{right} {up}"
            );
        }
        if status(&world).pose_seconds >= status(&world).duration_seconds {
            break;
        }
    }
    assert!(moves.len() >= 30, "{}", moves.len());
    // Where the legs end, carried through every move into the current frame.
    for rebase in &moves {
        to_current = Rebase {
            rotation: quat_normalize(quat_mul(rebase.rotation, to_current.rotation)),
            translation: rebase.apply_point(to_current.translation),
        };
    }
    let expected = to_current.apply_point([LEG * LEGS as f32, 50.0, 0.0]);
    assert_close(subject(&world), expected, 0.05);
}

fn status(world: &World) -> CameraTrackStatus {
    world
        .resource::<CameraTrackStatus>()
        .copied()
        .expect("published")
}

// The track runs as long as its lists were laid out at init, so losing the
// subject does not cut the boom's time from it and end the track early.
#[test]
fn losing_the_subject_does_not_shorten_the_track() {
    let (mut world, mut system) = world_with(
        track(
            vec![travel([1.0, 0.0, 0.0], 4.0, 2.0)],
            vec![],
            vec![boom(Some(3.0), None, 1.0), boom(Some(5.0), None, 3.0)],
        ),
        [0.0, 0.0, 3.0],
        [0.0; 3],
    );
    system.init(&mut world.context());
    assert!((status_duration(&world) - 4.0).abs() < 1e-5);
    seconds(&mut world, &mut system, 1);
    let entity = world.entity_of(SUBJECT).expect("subject");
    world.despawn(entity);
    seconds(&mut world, &mut system, 2);
    assert!((status_duration(&world) - 4.0).abs() < 1e-5);
    assert!(!status(&world).finished, "3 s into a 4 s track");
    seconds(&mut world, &mut system, 1);
    assert!(status(&world).finished);
}

// A subject whose Transform is local to a parent cannot be carried in world
// space: the track drives the camera instead.
#[test]
fn a_parented_subject_leaves_the_track_driving_the_camera() {
    let (mut world, mut system) = world_with(
        track(vec![travel([1.0, 0.0, 0.0], 4.0, 2.0)], vec![], vec![]),
        [0.0, 1.0, 0.0],
        [0.0, 1.0, -3.0],
    );
    let parent = world.push(Transform::default());
    let entity = world.entity_of(SUBJECT).expect("subject");
    world.insert(entity, Parent(parent));
    system.init(&mut world.context());
    seconds(&mut world, &mut system, 1);
    assert_close(camera(&world).position, [2.0, 1.0, 0.0], 1e-4);
    assert_eq!(subject(&world), [0.0, 1.0, -3.0], "left where it was");
}

// A subject the physics also simulates is carried.
#[test]
fn a_subject_with_a_body_is_carried() {
    let (mut world, mut system) = world_with(
        track(
            vec![travel([1.0, 0.0, 0.0], 4.0, 2.0)],
            vec![],
            vec![boom(Some(3.0), None, 0.0)],
        ),
        [0.0, 0.0, 3.0],
        [0.0; 3],
    );
    let entity = world.entity_of(SUBJECT).expect("subject");
    world.insert(entity, crate::components::BodyDynamics::default());
    system.init(&mut world.context());
    seconds(&mut world, &mut system, 1);
    assert_close(subject(&world), [2.0, 0.0, 0.0], 1e-4);
    assert_framing(&world, [3.0, 0.0, 0.0]);
}

// Boom legs on a track with no subject are not used, and do not lengthen it.
#[test]
fn boom_legs_without_a_subject_are_ignored() {
    let track = CameraTrack::bake(CameraTrackArgs {
        travel: vec![travel([1.0, 0.0, 0.0], 2.0, 2.0)],
        boom: vec![boom(Some(3.0), None, 5.0)],
        ..Default::default()
    });
    let mut system = CameraTrackSystem::new(&track);
    let mut world = World::new();
    world.add_component(Camera3D::bake(Camera3DArgs {
        position: [0.0, 1.0, 0.0],
        controller: None,
        ..Default::default()
    }));
    world.add_component(track);
    system.init(&mut world.context());
    assert!((status_duration(&world) - 1.0).abs() < 1e-5);
    seconds(&mut world, &mut system, 1);
    assert_close(camera(&world).position, [2.0, 1.0, 0.0], 1e-4);
}

// A subject carried from a moment of the track is only filmed until then,
// moving by itself, and is picked up where it is and as fast as it goes:
// neither its position nor its speed jumps, and it eases onto travel's path.
#[test]
fn a_subject_picked_up_mid_motion_carries_on_without_a_jump() {
    use crate::camera::track::carry::CARRY_BLEND_SECONDS;
    let mut track = track(vec![travel([1.0, 0.0, 0.0], 100.0, 10.0)], vec![], vec![]);
    track.carry_from_seconds = 2.0;
    let (mut world, mut system) = world_with(track, [0.0, 2.0, 5.0], [0.0; 3]);
    system.init(&mut world.context());
    let status = |world: &World| world.resource::<CameraTrackStatus>().unwrap().subject;
    // It rolls along +Z at 3 m/s by itself, 3 m off the path's start.
    let mut at = [3.0, 0.0, 0.0];
    let mut trail = Vec::new();
    // Up to the frame it is picked up on, at 2 s: whatever moves it has
    // moved it there by then.
    for i in 0..120 {
        at[2] += 3.0 * SimTiming::TICK_DT;
        if let Some(t) = world.context().get_mut_by_id::<Transform>(SUBJECT) {
            t.position = at;
        }
        frame(&mut world, &mut system, 1, 1.0);
        trail.push(subject(&world));
        if i < 119 {
            assert_eq!(status(&world), None, "only filmed");
            assert_eq!(subject(&world), at, "the track left it where it went");
        }
    }
    // Picked up: the track moves it from here on.
    for _ in 0..u32::saturating_from((CARRY_BLEND_SECONDS + 1.0) * 60.0) {
        frame(&mut world, &mut system, 1, 1.0);
        trail.push(subject(&world));
    }
    assert_eq!(status(&world), Some(SUBJECT));
    let step = |i: usize| vec3::sub(trail[i], trail[i - 1]);
    let before = step(118);
    let after = step(120);
    assert!(
        vec3::length(vec3::sub(after, before)) < 0.01 * vec3::length(before) + 1e-3,
        "{before:?} then {after:?}"
    );
    for i in 119..trail.len() {
        let jump = vec3::length(vec3::sub(step(i), step(i - 1)));
        assert!(jump < 0.02, "step {i} jumps by {jump}");
    }
    // Eased onto the path, which travel has run along +X at 10 m/s since
    // the track started, and off it by none.
    let end = subject(&world);
    let elapsed = 2.0 + CARRY_BLEND_SECONDS + 1.0;
    assert!((end[0] - 10.0 * elapsed).abs() < 0.2, "{end:?}");
    assert!(end[2].abs() < 1e-3 && end[1].abs() < 1e-3, "{end:?}");
}

// A frame move while the subject is only filmed or still easing on carries
// the path along: the subject ends where it would have with no move.
#[test]
fn a_frame_move_during_the_pick_up_carries_the_path() {
    let run = |moved: bool| {
        let mut track = track(vec![travel([1.0, 0.0, 0.0], 100.0, 10.0)], vec![], vec![]);
        track.carry_from_seconds = 1.0;
        let (mut world, mut system) = world_with(track, [0.0, 2.0, 5.0], [0.0; 3]);
        world.insert_resource(FrameRebases::default());
        system.init(&mut world.context());
        let rebase = Rebase {
            rotation: quat_from_axis_angle([0.0, 1.0, 0.0], 0.2),
            translation: [-5.0, 0.0, 3.0],
        };
        let mut origin = Rebase {
            rotation: [0.0, 0.0, 0.0, 1.0],
            translation: [0.0; 3],
        };
        for i in 0..300 {
            if moved && (i == 30 || i == 90) {
                let mut ctx = world.context();
                for t in ctx.query_mut::<Transform>() {
                    t.position = rebase.apply_point(t.position);
                }
                for c in ctx.query_mut::<Camera3D>() {
                    c.position = rebase.apply_point(c.position);
                }
                let mut rebases = world
                    .resource::<FrameRebases>()
                    .cloned()
                    .unwrap_or_default();
                rebases.push(rebase);
                world.insert_resource(rebases);
                origin = Rebase {
                    rotation: quat_mul(rebase.rotation, origin.rotation),
                    translation: rebase.apply_point(origin.translation),
                };
            }
            if i < 60 {
                let back = origin.apply_point([2.0, 0.0, 0.05 * i as f32]);
                if let Some(t) = world.context().get_mut_by_id::<Transform>(SUBJECT) {
                    t.position = back;
                }
            }
            frame(&mut world, &mut system, 1, 1.0);
        }
        // Back into the frame the run started in.
        let at = subject(&world);
        let r = origin.rotation;
        let inverse = [-r[0], -r[1], -r[2], r[3]];
        quat_rotate(inverse, vec3::sub(at, origin.translation))
    };
    let still = run(false);
    let moved = run(true);
    assert!(
        vec3::length(vec3::sub(still, moved)) < 0.01,
        "{still:?} vs {moved:?}"
    );
}

// A pick-up on the first instant of a short leg takes the pace that leg sets
// off at: a subject already running along the path stays on it.
#[test]
fn a_pick_up_on_a_leg_boundary_keeps_the_new_legs_pace() {
    let legs = (0..12)
        .map(|_| travel([1.0, 0.0, 0.0], 2.5, 10.0))
        .collect();
    let mut track = track(legs, vec![], vec![]);
    track.carry_from_seconds = 1.0;
    let (mut world, mut system) = world_with(track, [0.0, 2.0, 5.0], [0.0; 3]);
    system.init(&mut world.context());
    let mut worst = 0.0_f32;
    for i in 1..=150_u16 {
        let seconds = f32::from(i) * SimTiming::TICK_DT;
        if i <= 60
            && let Some(t) = world.context().get_mut_by_id::<Transform>(SUBJECT)
        {
            t.position = [10.0 * seconds, 0.0, 0.0];
        }
        frame(&mut world, &mut system, 1, 1.0);
        let off = vec3::length(vec3::sub(subject(&world), [10.0 * seconds, 0.0, 0.0]));
        worst = worst.max(off);
    }
    assert!(worst < 0.01, "strays {worst} m off the path");
}
