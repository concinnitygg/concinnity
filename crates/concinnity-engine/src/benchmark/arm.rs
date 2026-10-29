// Arming a world for a benchmark: the camera's start pose, the track that
// drives it, and the run the measurement reads.

use alloc::vec::Vec;

use crate::benchmark::BenchmarkRun;
use crate::benchmark::planet_tour;
use crate::components::cook::CameraTrack as CameraTrackArgs;
use crate::components::{
    BenchmarkConfig, BenchmarkRoute, BodyDynamics, Camera3D, CameraBoom, CameraTrack, CameraTravel,
    CameraTurn, DriveScript, Planet, Prop, PropBody, RollingControl, RollingDrive, Transform,
};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;
use concinnity_core::math::vec3;

/// Arm `world` for a benchmark, before it starts: pose its camera (or the
/// subject it films, with the camera kept at its authored offset) where the
/// [`BenchmarkConfig`] starts it, replace any track it declares with the
/// benchmark's own, and publish the [`BenchmarkRun`]. A world that declares no
/// config is armed with the default one, stopped `stop_at` seconds into the run
/// when that comes first. Arming twice changes nothing.
pub fn arm(world: &mut World, stop_at: Option<f32>) {
    if world.resource::<BenchmarkRun>().is_some() {
        return;
    }
    let mut config = world
        .query::<BenchmarkConfig>()
        .next()
        .cloned()
        .unwrap_or_default();
    resolve_route(world, &mut config);
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
    if !config.drive.is_empty()
        && let Some(entity) = config.subject.and_then(|s| world.entity_of(s.id()))
    {
        let warmup = config.warmup_seconds.max(0.0);
        world.insert(entity, DriveScript::new(&config.drive, warmup));
    }
    let track = CameraTrack::bake(track_args(&config));
    match world.query_mut::<CameraTrack>().next() {
        Some(declared) => *declared = track,
        None => world.add_component(track),
    }
    world.insert_resource(BenchmarkRun::new(config).with_stop_at(stop_at));
}

// Lay `config`'s route out over `world`: a planet tour fills the start, the
// legs and the phases from the world's planet and sun. Says which route was
// taken, since a world that chooses none can be toured without asking.
fn resolve_route(world: &World, config: &mut BenchmarkConfig) {
    let planet = world.query::<Planet>().next().is_some();
    let tours = match config.route {
        BenchmarkRoute::Legs => false,
        BenchmarkRoute::PlanetTour if !planet => {
            tracing::warn!(
                "benchmark: the run asks to tour a planet, but the world has none, so it follows its {} authored leg(s)",
                config.travel.len()
            );
            return;
        }
        BenchmarkRoute::PlanetTour => true,
        BenchmarkRoute::Auto => planet && !config.authors_route(),
    };
    if !tours {
        tracing::info!(
            "benchmark: the run follows its {} authored leg(s)",
            config.travel.len()
        );
        return;
    }
    let rolls = rolls(world, config);
    let Some(plan) = planet_tour::plan(world, config.tour, rolls) else {
        return;
    };
    tracing::info!(
        "benchmark: the run tours the planet, setting out {:.0} degrees from north, {}",
        config.tour.heading_deg,
        if rolls {
            "the subject rolling itself into the sea before it flies"
        } else {
            "skimming the ground"
        }
    );
    config.start = Some(plan.start);
    config.travel = plan.travel;
    config.turn = plan.turn;
    config.boom = plan.boom;
    config.phases = plan.phases;
    config.drive = plan.drive;
    config.carry_from_seconds = plan.carry_from_seconds;
}

// Whether the run's subject rolls itself: it has a drive and a body to push.
fn rolls(world: &World, config: &BenchmarkConfig) -> bool {
    let Some(subject) = config.subject.map(crate::ecs::Ref::id) else {
        return false;
    };
    let entity = world.entity_of(subject);
    let driven = world
        .query::<RollingDrive>()
        .any(|d| d.prop_name.is_some_and(|p| p.id() == subject))
        || entity.is_some_and(|e| world.get::<RollingControl>(e).is_some());
    let body = world
        .query::<PropBody>()
        .any(|b| b.prop_name.is_some_and(|p| p.id() == subject))
        || entity.is_some_and(|e| world.get::<BodyDynamics>(e).is_some());
    driven && body
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
    // A subject carried from a moment of the run is carried from that moment
    // after the warm-up; one carried from the start, through the warm-up too.
    let carry_from_seconds = if config.carry_from_seconds > 0.0 {
        warmup + config.carry_from_seconds
    } else {
        0.0
    };
    CameraTrackArgs {
        travel,
        turn,
        subject: config.subject,
        boom,
        carry_from_seconds,
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

    // A planet world whose config authors none of its route tours its planet;
    // one that authors any of it, or asks for its legs, keeps it; a tour asked
    // of a world with no planet keeps the authored legs.
    #[test]
    fn the_route_is_a_tour_for_a_planet_without_legs() {
        let planet: Planet = crate::test_support::from_json(
            r#"{"center":[0.0,-5000.0,0.0],"radius":5000.0,"amplitude":0.0}"#,
        );
        let armed = |config: BenchmarkConfig, with_planet: bool| {
            let mut world = world_with_camera();
            if with_planet {
                world.push(planet.clone());
            }
            world.add_component(config);
            arm(&mut world, None);
            world.resource::<BenchmarkRun>().unwrap().config.clone()
        };
        let toured = armed(BenchmarkConfig::default(), true);
        assert!(toured.travel.len() > 100 && toured.start.is_some());
        assert_eq!(toured.phases[0].name, "settle");
        let legs = BenchmarkConfig {
            travel: alloc::vec![leg(4.0)],
            ..Default::default()
        };
        assert_eq!(armed(legs.clone(), true).travel, legs.travel);
        // Any authored part of the route keeps the route authored.
        let phased = BenchmarkConfig {
            phases: alloc::vec![crate::components::BenchmarkPhase {
                name: "still".into(),
                seconds: 5.0,
            }],
            ..Default::default()
        };
        let kept = armed(phased.clone(), true);
        assert!(kept.travel.is_empty() && kept.start.is_none());
        assert_eq!(kept.phases, phased.phases);
        let started = BenchmarkConfig {
            start: Some(CameraPose::default()),
            ..Default::default()
        };
        assert!(armed(started, true).travel.is_empty());
        let held = BenchmarkConfig {
            route: BenchmarkRoute::Legs,
            ..Default::default()
        };
        assert!(armed(held, true).travel.is_empty());
        let forced = BenchmarkConfig {
            route: BenchmarkRoute::PlanetTour,
            ..Default::default()
        };
        assert!(armed(forced.clone(), false).travel.is_empty());
        let forced_legs = BenchmarkConfig {
            travel: alloc::vec![leg(4.0)],
            ..forced.clone()
        };
        assert_eq!(armed(forced_legs.clone(), false).travel, forced_legs.travel);
        assert!(!armed(forced, true).travel.is_empty());
        assert!(armed(BenchmarkConfig::default(), false).travel.is_empty());
    }

    #[test]
    fn a_world_without_a_config_is_armed_with_the_default_one() {
        let mut world = world_with_camera();
        arm(&mut world, None);
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
        arm(&mut world, None);
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
        arm(&mut world, None);
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
        arm(&mut world, None);
        let camera = world.query::<Camera3D>().next().expect("camera");
        assert_eq!(camera.position, [9.0, 8.0, 7.0]);
        assert!((camera.yaw - 90.0_f32.to_radians()).abs() < 1e-6);
        assert!((camera.pitch + 10.0_f32.to_radians()).abs() < 1e-6);
    }

    #[test]
    fn the_launch_stop_reaches_the_armed_run() {
        let mut world = world_with_camera();
        arm(&mut world, Some(1.5));
        let run = world.resource::<BenchmarkRun>().expect("armed");
        assert_eq!(run.stop_seconds(), 1.5);
    }

    #[test]
    fn arming_twice_keeps_the_first_run() {
        let mut world = world_with_camera();
        arm(&mut world, None);
        world.add_component(BenchmarkConfig {
            warmup_seconds: 9.0,
            ..Default::default()
        });
        arm(&mut world, None);
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
        arm(&mut world, None);
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
        arm(&mut world, None);
        assert_eq!(
            world.get_by_id::<Prop>(id).unwrap().position,
            [10.0, 0.0, 0.0]
        );
        let camera = world.query::<Camera3D>().next().unwrap().position;
        assert_eq!(camera, [10.0, 0.0, 3.0]);
    }
}
