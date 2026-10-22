// Arming a world for a benchmark: the camera's start pose, the track that
// drives it, and the run the measurement reads.

use alloc::vec::Vec;

use crate::benchmark::BenchmarkRun;
use crate::components::cook::CameraTrack as CameraTrackArgs;
use crate::components::{
    BenchmarkConfig, Camera3D, CameraBoom, CameraTrack, CameraTravel, CameraTurn, Prop, Transform,
};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;
use concinnity_core::math::vec3;

/// Arm `world` for a benchmark, before it starts: pose its camera (or the
/// subject it films, with the camera kept at its authored offset) where the
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
    if let Some(pose) = config.start {
        let subject = config
            .subject
            .and_then(|s| place_subject(world, s.id(), pose.position));
        if let Some(camera) = world.query_mut::<Camera3D>().next() {
            // A filmed subject moves with its camera's authored offset.
            let position = subject.map_or(pose.position, |authored| {
                vec3::add(pose.position, vec3::sub(camera.position, authored))
            });
            camera.set_pose(
                position,
                pose.yaw_deg.to_radians(),
                pose.pitch_deg.to_radians(),
            );
        }
    }
    let track = CameraTrack::bake(track_args(&config));
    match world.query_mut::<CameraTrack>().next() {
        Some(declared) => *declared = track,
        None => world.add_component(track),
    }
    world.insert_resource(BenchmarkRun::new(config));
}

// Move the subject `id` to `at`, returning where it was authored: on its
// Transform once the world's props are decomposed, else on the Prop itself.
fn place_subject(world: &mut World, id: AssetId, at: [f32; 3]) -> Option<[f32; 3]> {
    let entity = world.entity_of(id)?;
    if let Some(t) = world.get_mut::<Transform>(entity) {
        return Some(core::mem::replace(&mut t.position, at));
    }
    world
        .get_mut::<Prop>(entity)
        .map(|p| core::mem::replace(&mut p.position, at))
}

// The config's legs behind a hold on each list that lasts the warm-up, so the
// camera keeps still while the run settles and moves off as the first phase
// opens.
fn track_args(config: &BenchmarkConfig) -> CameraTrackArgs {
    let warmup = config.warmup_seconds.max(0.0);
    let mut travel = Vec::with_capacity(config.travel.len() + 1);
    let mut turn = Vec::with_capacity(config.turn.len() + 1);
    let mut boom = Vec::with_capacity(config.boom.len() + 1);
    if warmup > 0.0 {
        travel.push(CameraTravel {
            seconds: warmup,
            ..Default::default()
        });
        turn.push(CameraTurn {
            seconds: warmup,
            ..Default::default()
        });
        if !config.boom.is_empty() {
            boom.push(CameraBoom {
                seconds: warmup,
                ..Default::default()
            });
        }
    }
    travel.extend(config.travel.iter().cloned());
    turn.extend(config.turn.iter().cloned());
    boom.extend(config.boom.iter().cloned());
    CameraTrackArgs {
        travel,
        turn,
        subject: config.subject,
        boom,
    }
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
        assert!(
            tracks[0].boom.is_empty(),
            "no boom legs without any in the config"
        );
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

    // A filmed subject starts where the config starts the run, the camera at
    // the offset from it the world authored, and the track carries it.
    #[test]
    fn a_config_with_a_subject_starts_the_subject_and_films_it() {
        use crate::components::CameraPose;
        use crate::ecs::Ref;
        let mut world = world_with_camera();
        let id = AssetId(4);
        world.push_identified(
            id,
            Transform {
                position: [1.0, 2.0, 0.0],
                ..Default::default()
            },
        );
        world.add_component(BenchmarkConfig {
            start: Some(CameraPose {
                position: [10.0, 0.0, 0.0],
                yaw_deg: 0.0,
                pitch_deg: 0.0,
            }),
            subject: Some(Ref::<Prop>::new(id)),
            boom: alloc::vec![CameraBoom {
                distance: Some(5.0),
                ..Default::default()
            }],
            ..Default::default()
        });
        arm(&mut world);
        let subject = world.get_by_id::<Transform>(id).unwrap().position;
        assert_eq!(subject, [10.0, 0.0, 0.0]);
        let camera = world.query::<Camera3D>().next().unwrap().position;
        assert_eq!(camera, [10.0, 0.0, 3.0], "the authored offset rides along");
        let track = world.query::<CameraTrack>().next().unwrap();
        assert_eq!(track.subject, Some(Ref::new(id)));
        assert_eq!(track.boom.len(), 2, "the warm-up hold, then the config's");
        assert_eq!(track.boom[1].distance, Some(5.0));
    }

    // Armed before the world's props are decomposed, the subject is moved on
    // its Prop, which its Transform is then made from.
    #[test]
    fn a_subject_not_yet_decomposed_starts_on_its_prop() {
        use crate::components::CameraPose;
        use crate::ecs::Ref;
        let mut world = world_with_camera();
        let id = AssetId(4);
        world.push_identified(
            id,
            Prop {
                position: [1.0, 2.0, 0.0],
                ..Default::default()
            },
        );
        world.add_component(BenchmarkConfig {
            start: Some(CameraPose {
                position: [10.0, 0.0, 0.0],
                yaw_deg: 0.0,
                pitch_deg: 0.0,
            }),
            subject: Some(Ref::<Prop>::new(id)),
            ..Default::default()
        });
        arm(&mut world);
        assert_eq!(
            world.get_by_id::<Prop>(id).unwrap().position,
            [10.0, 0.0, 0.0]
        );
        let camera = world.query::<Camera3D>().next().unwrap().position;
        assert_eq!(camera, [10.0, 0.0, 3.0]);
    }
}
