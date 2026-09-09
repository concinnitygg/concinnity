use alloc::vec::Vec;

use crate::camera_track::CameraTrackStatus;
use crate::camera_track::timeline::{self, Key};
use crate::components::{Camera3D, CameraTrack, CameraTurnKey};
use crate::ecs::{Entity, MenuActive, PipelineContext, SimTiming, StepResult, System};
use crate::math::rem_euclid;

const HALF_TURN_DEG: f32 = 180.0;
const FULL_TURN_DEG: f32 = 360.0;

/// Drives the world's [`Camera3D`] along its declared
/// [`CameraTrack`](crate::components::CameraTrack).
///
/// Runs on the fixed simulation clock rather than the frame delta, so the pose
/// at a given track time is the same on every machine and two runs of one
/// track are comparable. Frozen while a world-pausing screen is open.
#[derive(Debug)]
pub struct CameraTrackSystem {
    travel: Vec<Key<3>>,
    // Parallel to `travel`: the segment each key runs inside.
    segments: Vec<Option<u32>>,
    // Held until `init`, which is the first point the heading the first leg
    // turns away from is known.
    turn_legs: Vec<CameraTurnKey>,
    turn: Vec<Key<2>>,
    start_position: [f32; 3],
    start_angles_deg: [f32; 2],
    // Ticks are counted and multiplied rather than accumulated: a long run
    // stays free of float drift, and two hosts running the same track at
    // different frame rates land on exactly the same pose.
    ticks: u64,
    tick_dt: f32,
    camera: Option<Entity>,
}

impl CameraTrackSystem {
    /// The system for a world's baked track. The travel list is ready here;
    /// the turn list waits for `init` to supply the camera's starting heading.
    pub fn new(track: &CameraTrack) -> Self {
        Self {
            travel: track
                .travel
                .iter()
                .map(|key| Key {
                    value: key.offset,
                    end_seconds: key.end_seconds,
                    ease: key.ease,
                })
                .collect(),
            segments: track.travel.iter().map(|key| key.segment).collect(),
            turn_legs: track.turn.clone(),
            turn: Vec::new(),
            start_position: [0.0; 3],
            start_angles_deg: [0.0; 2],
            ticks: 0,
            tick_dt: SimTiming::TICK_DT,
            camera: None,
        }
    }

    /// How long the whole track runs: whichever of its two lists lasts longer.
    pub fn duration(&self) -> f32 {
        timeline::duration(&self.travel).max(timeline::duration(&self.turn))
    }

    /// Seconds of simulated time the track has run for.
    pub fn elapsed(&self) -> f32 {
        self.ticks as f32 * self.tick_dt
    }

    // Lay the turn legs out on the clock, starting from the heading the camera
    // was authored at. Yaw is carried forward as a continuous angle so a leg
    // takes the shorter way round and the interpolation never wraps.
    fn resolve_turn(&mut self) {
        let [mut yaw, mut pitch] = self.start_angles_deg;
        let mut end_seconds = 0.0;
        self.turn.clear();
        self.turn.reserve(self.turn_legs.len());
        for leg in &self.turn_legs {
            let target_yaw = leg
                .yaw_deg
                .map_or(yaw, |to| yaw + shortest_turn_deg(yaw, to));
            let target_pitch = leg.pitch_deg.unwrap_or(pitch);
            let sweep = (target_yaw - yaw).abs().max((target_pitch - pitch).abs());
            end_seconds += turn_seconds(leg, sweep);
            self.turn.push(Key {
                value: [target_yaw, target_pitch],
                end_seconds,
                ease: leg.ease,
            });
            yaw = target_yaw;
            pitch = target_pitch;
        }
    }

    // Write the pose the track holds at this instant onto the camera, and
    // publish where the track has reached.
    fn apply(&self, ctx: &mut PipelineContext) {
        let elapsed = self.elapsed();
        let offset = timeline::sample([0.0; 3], &self.travel, elapsed);
        let angles = timeline::sample(self.start_angles_deg, &self.turn, elapsed);
        if let Some(camera) = self.camera
            && let Some(cam) = ctx.get_mut::<Camera3D>(camera)
        {
            cam.position = [
                self.start_position[0] + offset[0],
                self.start_position[1] + offset[1],
                self.start_position[2] + offset[2],
            ];
            cam.yaw = angles[0].to_radians();
            cam.pitch = angles[1].to_radians();
            cam.view_matrix = crate::gfx::camera::view_matrix(cam.position, cam.yaw, cam.pitch);
        }
        let duration = self.duration();
        ctx.insert_resource(CameraTrackStatus {
            elapsed_seconds: elapsed,
            duration_seconds: duration,
            segment: timeline::active(&self.travel, elapsed).and_then(|at| self.segments[at]),
            finished: elapsed >= duration,
        });
    }
}

impl System for CameraTrackSystem {
    fn init(&mut self, ctx: &mut PipelineContext) {
        if let Some((entity, camera)) = ctx.query_with_entity::<Camera3D>().next() {
            self.camera = Some(entity);
            self.start_position = camera.position;
            self.start_angles_deg = [camera.yaw.to_degrees(), camera.pitch.to_degrees()];
        }
        self.resolve_turn();
        self.apply(ctx);
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        if ctx.resource::<MenuActive>().is_some_and(|m| m.0) {
            return StepResult::Continue;
        }
        let timing = ctx.resource::<SimTiming>().copied().unwrap_or_default();
        self.ticks += u64::from(timing.ticks);
        self.tick_dt = timing.tick_dt;
        self.apply(ctx);
        StepResult::Continue
    }
}

// How long a turn leg runs: its authored duration, or the one its rate implies
// over the wider of the yaw and pitch sweeps.
fn turn_seconds(leg: &CameraTurnKey, sweep_deg: f32) -> f32 {
    if leg.seconds > 0.0 {
        leg.seconds
    } else if leg.degrees_per_second > 0.0 {
        sweep_deg / leg.degrees_per_second
    } else {
        0.0
    }
}

// The signed turn from `from` to `to` the short way round, in -180..=180.
fn shortest_turn_deg(from: f32, to: f32) -> f32 {
    rem_euclid(to - from + HALF_TURN_DEG, FULL_TURN_DEG) - HALF_TURN_DEG
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::cook::Camera3D as Camera3DArgs;
    use crate::components::{CameraTravel, CameraTurn, Ease};
    use crate::ecs::World;
    use alloc::string::ToString;
    use alloc::vec;

    const TICKS_PER_SECOND: usize = 60;

    fn track(travel: Vec<CameraTravel>, turn: Vec<CameraTurn>) -> CameraTrack {
        CameraTrack::bake(crate::components::CameraTrackArgs { travel, turn })
    }

    // A world holding one camera at `pose` (position, yaw degrees, pitch
    // degrees) and the track that drives it.
    fn world_with(
        track: CameraTrack,
        position: [f32; 3],
        yaw_deg: f32,
        pitch_deg: f32,
    ) -> (World, CameraTrackSystem) {
        let system = CameraTrackSystem::new(&track);
        let mut world = World::new();
        world.add_component(Camera3D::bake(Camera3DArgs {
            position,
            yaw: yaw_deg.to_radians(),
            pitch: pitch_deg.to_radians(),
            controller: None,
            ..Default::default()
        }));
        world.add_component(track);
        (world, system)
    }

    fn run(world: &mut World, system: &mut CameraTrackSystem, seconds: f32) {
        let ticks = (seconds * TICKS_PER_SECOND as f32).round() as usize;
        for _ in 0..ticks {
            system.step(&mut world.context());
        }
    }

    fn camera_position(world: &World) -> [f32; 3] {
        world.query::<Camera3D>().next().expect("camera").position
    }

    fn camera_yaw_deg(world: &World) -> f32 {
        world
            .query::<Camera3D>()
            .next()
            .expect("camera")
            .yaw
            .to_degrees()
    }

    fn status(world: &World) -> CameraTrackStatus {
        world
            .resource::<CameraTrackStatus>()
            .copied()
            .expect("published")
    }

    fn travel(direction: [f32; 3], distance: f32, speed: f32) -> CameraTravel {
        CameraTravel {
            direction,
            distance,
            speed,
            ..Default::default()
        }
    }

    #[test]
    fn the_camera_starts_where_it_was_authored() {
        let (mut world, mut system) = world_with(
            track(vec![travel([1.0, 0.0, 0.0], 8.0, 2.0)], vec![]),
            [3.0, 1.0, -2.0],
            45.0,
            0.0,
        );
        system.init(&mut world.context());
        assert_eq!(camera_position(&world), [3.0, 1.0, -2.0]);
        assert!((camera_yaw_deg(&world) - 45.0).abs() < 1e-3);
        assert_eq!(status(&world).elapsed_seconds, 0.0);
        assert!(!status(&world).finished);
    }

    #[test]
    fn travel_offsets_ride_on_the_authored_position() {
        // The track's offsets are relative, so a camera authored away from the
        // origin follows the same shape from where it stands.
        let (mut world, mut system) = world_with(
            track(vec![travel([1.0, 0.0, 0.0], 8.0, 2.0)], vec![]),
            [10.0, 0.0, 0.0],
            0.0,
            0.0,
        );
        system.init(&mut world.context());
        run(&mut world, &mut system, 2.0);
        assert!((camera_position(&world)[0] - 14.0).abs() < 1e-3);
        run(&mut world, &mut system, 2.0);
        assert!((camera_position(&world)[0] - 18.0).abs() < 1e-3);
    }

    #[test]
    fn the_clock_is_the_fixed_step_so_two_tick_rates_reach_the_same_pose() {
        // The whole point of the asset: what the camera is looking at is a
        // function of track time, never of how fast the host renders.
        let legs = || vec![travel([0.0, 0.0, -1.0], 12.0, 3.0)];
        let (mut a, mut sa) = world_with(track(legs(), vec![]), [0.0; 3], 0.0, 0.0);
        sa.init(&mut a.context());
        for _ in 0..120 {
            sa.step(&mut a.context());
        }

        let (mut b, mut sb) = world_with(track(legs(), vec![]), [0.0; 3], 0.0, 0.0);
        sb.init(&mut b.context());
        // Half the frames, each carrying twice the simulation steps: the same
        // two seconds of track time.
        for _ in 0..60 {
            b.context().insert_resource(SimTiming {
                ticks: 2,
                ..Default::default()
            });
            sb.step(&mut b.context());
        }
        assert_eq!(camera_position(&a), camera_position(&b));
    }

    #[test]
    fn a_finished_track_holds_its_last_pose() {
        let (mut world, mut system) = world_with(
            track(vec![travel([1.0, 0.0, 0.0], 4.0, 2.0)], vec![]),
            [0.0; 3],
            0.0,
            0.0,
        );
        system.init(&mut world.context());
        run(&mut world, &mut system, 10.0);
        assert!((camera_position(&world)[0] - 4.0).abs() < 1e-3);
        let s = status(&world);
        assert!(s.finished);
        assert!((s.duration_seconds - 2.0).abs() < 1e-6);
    }

    #[test]
    fn a_turn_takes_its_sweep_divided_by_its_rate_from_the_authored_heading() {
        // 90 degrees at 45 per second is two seconds, and the sweep is measured
        // from where the camera already looks rather than from zero.
        let (mut world, mut system) = world_with(
            track(
                vec![],
                vec![CameraTurn {
                    yaw_deg: Some(90.0),
                    degrees_per_second: 45.0,
                    ..Default::default()
                }],
            ),
            [0.0; 3],
            30.0,
            0.0,
        );
        system.init(&mut world.context());
        assert!((system.duration() - 60.0 / 45.0).abs() < 1e-5);
        run(&mut world, &mut system, 60.0 / 45.0);
        assert!((camera_yaw_deg(&world) - 90.0).abs() < 1e-2);
    }

    #[test]
    fn a_turn_takes_the_shorter_way_round() {
        // From 350 to 10 degrees is 20 degrees forward, not 340 back.
        let (mut world, mut system) = world_with(
            track(
                vec![],
                vec![CameraTurn {
                    yaw_deg: Some(10.0),
                    degrees_per_second: 20.0,
                    ..Default::default()
                }],
            ),
            [0.0; 3],
            350.0,
            0.0,
        );
        system.init(&mut world.context());
        assert!((system.duration() - 1.0).abs() < 1e-5);
        run(&mut world, &mut system, 0.5);
        // Half way is 360, which is the same heading as 0 and never 180.
        let half = camera_yaw_deg(&world);
        assert!((half - 360.0).abs() < 1e-2, "{half}");
    }

    #[test]
    fn a_turn_leg_with_no_target_holds_the_heading_it_started_from() {
        // An omitted angle has to mean "leave it alone": zero is a heading
        // like any other, so a hold leg would otherwise swing the camera to it.
        let (mut world, mut system) = world_with(
            track(
                vec![],
                vec![
                    CameraTurn {
                        seconds: 2.0,
                        ..Default::default()
                    },
                    CameraTurn {
                        yaw_deg: Some(60.0),
                        degrees_per_second: 30.0,
                        ..Default::default()
                    },
                ],
            ),
            [0.0; 3],
            25.0,
            -8.0,
        );
        system.init(&mut world.context());
        run(&mut world, &mut system, 2.0);
        assert!((camera_yaw_deg(&world) - 25.0).abs() < 1e-3);
        // The turn that follows starts from the held heading, so its sweep is
        // 35 degrees rather than 60.
        assert!((system.duration() - (2.0 + 35.0 / 30.0)).abs() < 1e-4);
    }

    #[test]
    fn a_turn_leg_moves_only_the_angle_it_names() {
        let (mut world, mut system) = world_with(
            track(
                vec![],
                vec![CameraTurn {
                    yaw_deg: Some(90.0),
                    degrees_per_second: 45.0,
                    ..Default::default()
                }],
            ),
            [0.0; 3],
            0.0,
            -30.0,
        );
        system.init(&mut world.context());
        run(&mut world, &mut system, 2.0);
        assert!((camera_yaw_deg(&world) - 90.0).abs() < 1e-2);
        let pitch = world.query::<Camera3D>().next().expect("camera").pitch;
        assert!((pitch.to_degrees() + 30.0).abs() < 1e-2);
    }

    #[test]
    fn yaw_and_pitch_arrive_together_on_the_wider_sweep() {
        let (mut world, mut system) = world_with(
            track(
                vec![],
                vec![CameraTurn {
                    yaw_deg: Some(20.0),
                    pitch_deg: Some(-60.0),
                    degrees_per_second: 30.0,
                    ..Default::default()
                }],
            ),
            [0.0; 3],
            0.0,
            0.0,
        );
        system.init(&mut world.context());
        // Pitch sweeps 60 degrees and yaw only 20, so the leg is two seconds.
        assert!((system.duration() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn the_two_tracks_run_against_one_clock_without_bending_each_other() {
        // The path is world space, so a camera turning through 90 degrees while
        // it travels still ends up where the travel leg alone would put it.
        let (mut world, mut system) = world_with(
            track(
                vec![travel([0.0, 0.0, -1.0], 10.0, 5.0)],
                vec![CameraTurn {
                    yaw_deg: Some(90.0),
                    degrees_per_second: 45.0,
                    ..Default::default()
                }],
            ),
            [0.0; 3],
            0.0,
            0.0,
        );
        system.init(&mut world.context());
        run(&mut world, &mut system, 2.0);
        let p = camera_position(&world);
        assert!((p[2] + 10.0).abs() < 1e-3, "{p:?}");
        assert!(p[0].abs() < 1e-4, "{p:?}");
        assert!((camera_yaw_deg(&world) - 90.0).abs() < 1e-2);
    }

    #[test]
    fn the_longer_list_fixes_the_duration_and_the_shorter_one_holds() {
        let (mut world, mut system) = world_with(
            track(
                vec![travel([1.0, 0.0, 0.0], 2.0, 2.0)],
                vec![CameraTurn {
                    yaw_deg: Some(90.0),
                    degrees_per_second: 15.0,
                    ..Default::default()
                }],
            ),
            [0.0; 3],
            0.0,
            0.0,
        );
        system.init(&mut world.context());
        // Travel runs 1s, the turn 6s.
        assert!((system.duration() - 6.0).abs() < 1e-5);
        run(&mut world, &mut system, 3.0);
        assert!(!status(&world).finished);
        // Travel finished at 1s and has been holding its end ever since.
        assert!((camera_position(&world)[0] - 2.0).abs() < 1e-3);
        run(&mut world, &mut system, 3.0);
        assert!(status(&world).finished);
    }

    #[test]
    fn the_published_segment_follows_the_travel_legs() {
        let named = |name: &str, distance: f32| CameraTravel {
            segment: name.to_string(),
            ..travel([1.0, 0.0, 0.0], distance, 1.0)
        };
        let (mut world, mut system) = world_with(
            track(vec![named("shadows", 2.0), named("rays", 3.0)], vec![]),
            [0.0; 3],
            0.0,
            0.0,
        );
        system.init(&mut world.context());
        assert_eq!(status(&world).segment, Some(0));
        run(&mut world, &mut system, 3.0);
        assert_eq!(status(&world).segment, Some(1));
        // The tail segment keeps answering once the track has run out, so a
        // reader never loses the name of what it was looking at.
        run(&mut world, &mut system, 10.0);
        assert!(status(&world).finished);
        assert_eq!(status(&world).segment, Some(1));
    }

    #[test]
    fn an_open_menu_freezes_the_track() {
        let (mut world, mut system) = world_with(
            track(vec![travel([1.0, 0.0, 0.0], 8.0, 2.0)], vec![]),
            [0.0; 3],
            0.0,
            0.0,
        );
        system.init(&mut world.context());
        run(&mut world, &mut system, 1.0);
        let held = camera_position(&world);
        world.context().insert_resource(MenuActive(true));
        run(&mut world, &mut system, 2.0);
        assert_eq!(camera_position(&world), held);
        world.context().insert_resource(MenuActive(false));
        run(&mut world, &mut system, 1.0);
        assert!(camera_position(&world)[0] > held[0]);
    }

    #[test]
    fn easing_reaches_the_same_end_as_a_linear_leg() {
        let eased = CameraTravel {
            ease: Ease::InOut,
            ..travel([1.0, 0.0, 0.0], 10.0, 5.0)
        };
        let (mut world, mut system) = world_with(track(vec![eased], vec![]), [0.0; 3], 0.0, 0.0);
        system.init(&mut world.context());
        // A quarter of the way in the camera is still behind where a constant
        // rate would have put it, and three quarters in it is ahead.
        run(&mut world, &mut system, 0.5);
        assert!(camera_position(&world)[0] < 2.5);
        run(&mut world, &mut system, 1.0);
        assert!(camera_position(&world)[0] > 7.5);
        // The endpoint is the leg's whole distance either way.
        run(&mut world, &mut system, 0.5);
        assert!((camera_position(&world)[0] - 10.0).abs() < 1e-3);
    }

    #[test]
    fn a_world_with_no_camera_still_publishes_where_the_track_reached() {
        let t = track(vec![travel([1.0, 0.0, 0.0], 4.0, 2.0)], vec![]);
        let mut system = CameraTrackSystem::new(&t);
        let mut world = World::new();
        world.add_component(t);
        system.init(&mut world.context());
        run(&mut world, &mut system, 3.0);
        assert!(status(&world).finished);
    }
}
