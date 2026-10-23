use super::*;
use crate::test_support::from_json;
use alloc::string::String;
use concinnity_core::math::vec3;

// A 50 km planet with hills and a shore a couple of hundred meters east of
// the start, a sea, and a sun the sky turns.
fn world(sea: bool) -> World {
    let mut world = World::new();
    world.push(from_json::<Planet>(
        r#"{"center":[0.0,-50000.0,0.0],"radius":50000.0,"amplitude":40.0,"feature_size":2000.0,"octaves":4,"seed":95,"land_share":0.4,"continent_size":20000.0,"ocean_depth":60.0,"beach_width":60.0}"#,
    ));
    if sea {
        world.push(from_json::<WaterSurface>(
            r#"{"shape":"planet_sea","sea_level":0.0}"#,
        ));
    }
    world.push(from_json::<SkyRotation>(
        r#"{"axis":[0.0,0.0,-1.0],"degrees_per_second":1.5}"#,
    ));
    world.push(DirectionalLight {
        direction: [1.0, 0.0, 0.35],
        ..Default::default()
    });
    world
}

fn height(p: [f32; 3]) -> f32 {
    vec3::length(vec3::sub(p, [0.0, -50_000.0, 0.0])) - 50_000.0
}

#[test]
fn a_world_without_a_planet_has_no_tour() {
    assert!(plan(&World::new(), PlanetTour::default(), 2.0, false).is_none());
}

// The tour starts `start_back` behind where the world starts along its
// heading, just over the ground, and its legs, turns and framing run back to
// back for every phase.
#[test]
fn the_tour_sets_out_behind_the_start_along_its_heading() {
    let tour = PlanetTour::default();
    let plan = plan(&world(true), tour, 2.0, false).expect("a planet");
    let start = plan.start.position;
    let back = vec3::length([start[0], 0.0, start[2]]);
    assert!((back - tour.start_back).abs() < 1.0, "{start:?}");
    // Heading 96 degrees is a little south of east, so the start lies west.
    assert!(start[0] < -140.0 && start[2] < 0.0, "{start:?}");
    assert!(
        (0.0..60.0).contains(&height(start)),
        "near the ground: {start:?}"
    );
    assert_eq!(plan.travel.len(), plan.turn.len());
    assert_eq!(plan.boom.len(), plan.travel.len() + 1);
    let legs: f32 = plan.travel.iter().map(|t| t.seconds).sum();
    let phases: f32 = plan.phases.iter().map(|p| p.seconds).sum();
    assert!((legs - phases).abs() < 1e-2, "{legs} vs {phases}");
    let names: Vec<&str> = plan.phases.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names[0], "settle");
    assert!(names.contains(&"orbit"));
}

// The heading turns the whole tour about the start's up.
#[test]
fn the_heading_turns_the_tour() {
    let north = PlanetTour {
        heading_deg: 0.0,
        ..PlanetTour::default()
    };
    let start = plan(&world(true), north, 2.0, false).unwrap().start;
    // Setting out north (-Z), it starts to the south (+Z).
    assert!(
        start.position[2] > 140.0 && start.position[0].abs() < 1.0,
        "{start:?}"
    );
    assert!(start.yaw_deg.abs() < 1e-3, "{start:?}");
}

// Over a sea the path rides the water, not the seabed under it; without one
// it follows the ground down.
#[test]
fn the_sea_floors_the_path() {
    let lowest = |sea: bool| {
        let plan = plan(&world(sea), PlanetTour::default(), 2.0, false).unwrap();
        let mut at = plan.start.position;
        let mut lowest = f32::MAX;
        for leg in plan.travel.iter().take(200) {
            // A leg's direction is its whole step, which the bake normalizes.
            let along = vec3::normalize_or(leg.direction, 1e-9, [0.0; 3]);
            at = vec3::add(at, vec3::scale(along, leg.distance));
            lowest = lowest.min(height(at));
        }
        lowest
    };
    assert!(lowest(true) > 0.0, "{}", lowest(true));
    assert!(
        lowest(false) < lowest(true),
        "{} {}",
        lowest(false),
        lowest(true)
    );
}

// A rolling subject rolls itself through a settle, a roll, a float and a
// wake before the flight: its drive holds still, sprints along the tour's
// heading, then is towed at the wake's pace, and the tour picks it up as the
// sea phase opens. A skimming one has no
// drive and is carried from the start.
#[test]
fn a_rolling_subject_rolls_before_it_flies() {
    let tour = PlanetTour::default();
    let rolled = plan(&world(true), tour, 2.0, true).unwrap();
    let names: Vec<&str> = rolled.phases.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(&names[..5], ["settle", "roll", "float", "wake", "sea"]);
    let sea: f32 = rolled.phases[..4].iter().map(|p| p.seconds).sum();
    assert_eq!(rolled.carry_from_seconds, sea);
    assert_eq!(rolled.drive.len(), 3);
    assert_eq!(
        (rolled.drive[0].seconds, rolled.drive[0].throttle),
        (0.0, 0.0)
    );
    let roll = &rolled.drive[1];
    assert_eq!(roll.seconds, rolled.phases[0].seconds);
    assert!(roll.sprint && roll.throttle == 1.0 && roll.heading_deg == tour.heading_deg);
    assert_eq!(roll.pace, 0.0);
    let wake = &rolled.drive[2];
    let wake_start: f32 = rolled.phases[..3].iter().map(|p| p.seconds).sum();
    assert_eq!(wake.seconds, wake_start);
    assert_eq!(f64::from(wake.pace), timetable::WAKE_SPEED);
    assert_eq!(wake.heading_deg, tour.heading_deg);
    let skimmed = plan(&world(true), tour, 2.0, false).unwrap();
    assert!(skimmed.drive.is_empty() && skimmed.carry_from_seconds == 0.0);
    assert_eq!(skimmed.phases[1].name, "meadow");
    // The flight after the ground stretch is the same either way.
    let flight = |p: &TourPlan, from: usize| -> Vec<(String, f32)> {
        p.phases[from..]
            .iter()
            .map(|x| (x.name.clone(), x.seconds))
            .collect()
    };
    assert_eq!(flight(&rolled, 4), flight(&skimmed, 3));
}

// With no sea to float on, a rolling subject is picked up as its roll ends,
// and the path never sinks into the ground under it.
#[test]
fn a_rolling_subject_on_a_dry_planet_skips_the_float() {
    let dry = plan(&world(false), PlanetTour::default(), 2.0, true).unwrap();
    let names: Vec<&str> = dry.phases.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(&names[..3], ["settle", "roll", "sea"]);
    let roll: f32 = dry.phases[..2].iter().map(|p| p.seconds).sum();
    assert_eq!(dry.carry_from_seconds, roll);
    let mut at = dry.start.position;
    let shape = world(false).query::<Planet>().next().unwrap().shape();
    for leg in &dry.travel {
        at = vec3::add(
            at,
            vec3::scale(vec3::normalize(leg.direction), leg.distance),
        );
        let up = dvec3::normalize_or(
            dvec3::from_f32(vec3::sub(at, [0.0, -50_000.0, 0.0])),
            [0.0, 1.0, 0.0],
        );
        let over = f64::from(height(at)) - shape.height(up);
        assert!(over > 0.9, "{over} m over the ground");
    }
}
