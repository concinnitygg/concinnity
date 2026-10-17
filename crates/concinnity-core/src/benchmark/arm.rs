// Arming a world for a benchmark: the camera's start pose, the track that
// drives it, and the run the measurement reads.

use alloc::vec::Vec;

use crate::benchmark::BenchmarkRun;
use crate::components::cook::CameraTrack as CameraTrackArgs;
use crate::components::{BenchmarkConfig, Camera3D, CameraTrack, CameraTravel, CameraTurn};
use crate::ecs::World;

/// Arm `world` for a benchmark, before it starts: pose its camera where the
/// [`BenchmarkConfig`] starts it, replace any track it declares with the
/// benchmark's own, and publish the [`BenchmarkRun`]. A world that declares no
/// config is armed with the default one. Arming twice changes nothing.
pub fn arm(world: &mut World) {
    if world.resource::<BenchmarkRun>().is_some() {
        return;
    }
    let config = world
        .query::<BenchmarkConfig>()
        .next()
        .cloned()
        .unwrap_or_default();
    if let Some(pose) = config.start
        && let Some(camera) = world.query_mut::<Camera3D>().next()
    {
        camera.set_pose(
            pose.position,
            pose.yaw_deg.to_radians(),
            pose.pitch_deg.to_radians(),
        );
    }
    let track = CameraTrack::bake(track_args(&config));
    match world.query_mut::<CameraTrack>().next() {
        Some(declared) => *declared = track,
        None => world.add_component(track),
    }
    world.insert_resource(BenchmarkRun::new(config));
}

// The config's legs behind a hold on each list that lasts the warm-up, so the
// camera keeps still while the run settles and moves off as the first phase
// opens.
fn track_args(config: &BenchmarkConfig) -> CameraTrackArgs {
    let warmup = config.warmup_seconds.max(0.0);
    let mut travel = Vec::with_capacity(config.travel.len() + 1);
    let mut turn = Vec::with_capacity(config.turn.len() + 1);
    if warmup > 0.0 {
        travel.push(CameraTravel {
            seconds: warmup,
            ..Default::default()
        });
        turn.push(CameraTurn {
            seconds: warmup,
            ..Default::default()
        });
    }
    travel.extend(config.travel.iter().cloned());
    turn.extend(config.turn.iter().cloned());
    CameraTrackArgs { travel, turn }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::CameraPose;
    use crate::components::cook::Camera3D as Camera3DArgs;

    fn world_with_camera() -> World {
        let mut world = World::new();
        world.add_component(Camera3D::bake(Camera3DArgs {
            position: [1.0, 2.0, 3.0],
            controller: None,
            ..Default::default()
        }));
        world
    }

    fn leg(distance: f32) -> CameraTravel {
        CameraTravel {
            direction: [1.0, 0.0, 0.0],
            distance,
            speed: 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn a_world_without_a_config_is_armed_with_the_default_one() {
        let mut world = world_with_camera();
        arm(&mut world);
        let run = world.resource::<BenchmarkRun>().expect("armed");
        assert_eq!(run.config, BenchmarkConfig::default());
        let track = world.query::<CameraTrack>().next().expect("a track");
        // Only the warm-up hold: the camera stays where the world put it.
        assert_eq!(track.travel.len(), 1);
        assert_eq!(track.travel[0].delta, [0.0; 3]);
        assert_eq!(track.travel[0].end_seconds, 2.0);
        let camera = world.query::<Camera3D>().next().expect("camera");
        assert_eq!(camera.position, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn the_legs_start_when_the_warm_up_ends() {
        let mut world = world_with_camera();
        world.add_component(BenchmarkConfig {
            warmup_seconds: 3.0,
            travel: alloc::vec![leg(4.0)],
            ..Default::default()
        });
        arm(&mut world);
        let track = world.query::<CameraTrack>().next().expect("a track");
        assert_eq!(track.travel[0].end_seconds, 3.0);
        assert_eq!(track.travel[1].end_seconds, 7.0);
        assert_eq!(track.travel[1].delta, [4.0, 0.0, 0.0]);
        assert_eq!(track.turn[0].seconds, 3.0);
    }

    #[test]
    fn no_warm_up_plays_the_legs_from_the_start() {
        let config = BenchmarkConfig {
            warmup_seconds: 0.0,
            travel: alloc::vec![leg(4.0)],
            ..Default::default()
        };
        let args = track_args(&config);
        assert_eq!(args.travel, config.travel);
        assert!(args.turn.is_empty());
    }

    #[test]
    fn the_benchmark_track_replaces_a_declared_one() {
        let mut world = world_with_camera();
        world.add_component(CameraTrack::bake(CameraTrackArgs {
            travel: alloc::vec![leg(50.0)],
            ..Default::default()
        }));
        arm(&mut world);
        let tracks: Vec<_> = world.query::<CameraTrack>().collect();
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].travel.len(), 1, "only the warm-up hold");
    }

    #[test]
    fn a_start_pose_moves_the_camera() {
        let mut world = world_with_camera();
        world.add_component(BenchmarkConfig {
            start: Some(CameraPose {
                position: [9.0, 8.0, 7.0],
                yaw_deg: 90.0,
                pitch_deg: -10.0,
            }),
            ..Default::default()
        });
        arm(&mut world);
        let camera = world.query::<Camera3D>().next().expect("camera");
        assert_eq!(camera.position, [9.0, 8.0, 7.0]);
        assert!((camera.yaw - 90.0_f32.to_radians()).abs() < 1e-6);
        assert!((camera.pitch + 10.0_f32.to_radians()).abs() < 1e-6);
    }

    #[test]
    fn arming_twice_keeps_the_first_run() {
        let mut world = world_with_camera();
        arm(&mut world);
        world.add_component(BenchmarkConfig {
            warmup_seconds: 9.0,
            ..Default::default()
        });
        arm(&mut world);
        let run = world.resource::<BenchmarkRun>().expect("armed");
        assert_eq!(run.config.warmup_seconds, 2.0);
    }
}
