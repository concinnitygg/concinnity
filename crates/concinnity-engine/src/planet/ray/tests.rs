use super::*;
use crate::planet::coast::fibonacci_direction;
use crate::planet::{Continents, LocalFrame};

const CENTER: DVec3 = [0.0, -50_000.0, 0.0];

// A smooth ball, whose crossings have a closed form.
fn ball() -> PlanetShape {
    PlanetShape {
        center: CENTER,
        radius: 50_000.0,
        amplitude: 0.0,
        feature_size: 2_000.0,
        octaves: 1,
        seed: 1,
        continents: Continents::NONE,
        edits: None,
    }
}

fn hills() -> PlanetShape {
    PlanetShape {
        amplitude: 60.0,
        octaves: 8,
        seed: 7,
        ..ball()
    }
}

// Hills on continents, with seas between.
fn coasts() -> PlanetShape {
    PlanetShape {
        continents: Continents::new(50_000.0, 7, 20_000.0, 0.4, 60.0, 60.0),
        ..hills()
    }
}

fn above(s: &PlanetShape, dir: DVec3, height: f64) -> DVec3 {
    dvec3::add(
        s.center,
        dvec3::scale(dir, s.radius + s.height(dir) + height),
    )
}

fn directions(n: usize) -> impl Iterator<Item = DVec3> {
    (0..n).map(move |i| fibonacci_direction(i, n))
}

// The ground point a hit names lies on the height function, along the ray.
fn assert_on_ground(s: &PlanetShape, origin: DVec3, dir: DVec3, hit: &GroundHit) {
    let off = s.clearance(dvec3::sub(hit.point, s.center));
    assert!(off.abs() < 1e-3, "{off}");
    let along = dvec3::add(origin, dvec3::scale(dir, hit.distance));
    assert!(dvec3::length(dvec3::sub(along, hit.point)) < 1e-6);
    assert!((dvec3::length(hit.normal) - 1.0).abs() < 1e-9);
}

#[test]
fn a_ray_down_onto_a_ball_lands_where_the_sphere_says() {
    let s = ball();
    for dir in directions(50) {
        let origin = above(&s, dir, 300.0);
        let hit = s
            .ray_hit(origin, dvec3::scale(dir, -1.0), f64::INFINITY)
            .unwrap();
        assert!((hit.distance - 300.0).abs() < 1e-3, "{}", hit.distance);
        assert!(dvec3::dot(hit.normal, dir) > 1.0 - 1e-9);
    }
}

// A slanted ray onto the ball meets it at the near root of the sphere
// equation, normal radial.
#[test]
fn a_slanted_ray_onto_a_ball_meets_the_near_root() {
    let s = ball();
    let origin = [0.0, 40.0, 0.0];
    let dir = dvec3::normalize_or([0.6, -0.3, 0.2], [0.0; 3]);
    let hit = s.ray_hit(origin, dir, f64::INFINITY).unwrap();
    let rel = dvec3::sub(origin, CENTER);
    let b = dvec3::dot(rel, dir);
    let c = dvec3::dot(rel, rel) - s.radius * s.radius;
    let want = -b - libm::sqrt(b * b - c);
    assert!(
        (hit.distance - want).abs() < 1e-3,
        "{} {want}",
        hit.distance
    );
    let radial = s.up_at(hit.point);
    assert!(dvec3::dot(hit.normal, radial) > 1.0 - 1e-9);
}

#[test]
fn rays_onto_hills_and_coasts_land_on_the_ground() {
    for s in [hills(), coasts()] {
        for (i, dir) in directions(120).enumerate() {
            let origin = above(&s, dir, 25.0 + i as f64);
            let [east, north, _] = crate::planet::geo_axes(dir);
            // Down and a little sideways, so the ray crosses slopes.
            let ray = dvec3::normalize_or(
                dvec3::add(
                    dvec3::scale(dir, -1.0),
                    dvec3::add(dvec3::scale(east, 0.4), dvec3::scale(north, -0.2)),
                ),
                [0.0; 3],
            );
            let hit = s.ray_hit(origin, ray, f64::INFINITY).unwrap();
            assert_on_ground(&s, origin, ray, &hit);
            // Nothing nearer along the ray is below ground.
            let samples = 400;
            for k in 0..samples {
                let t = hit.distance * f64::from(k) / f64::from(samples);
                let p = dvec3::add(origin, dvec3::scale(ray, t));
                assert!(s.clearance(dvec3::sub(p, s.center)) > -1e-3);
            }
        }
    }
}

// The normal is the shape's own, which tilts with the hills.
#[test]
fn the_normal_is_the_shapes_normal_there() {
    let s = hills();
    let dir = fibonacci_direction(17, 64);
    let origin = above(&s, dir, 40.0);
    let hit = s
        .ray_hit(origin, dvec3::scale(dir, -1.0), f64::INFINITY)
        .unwrap();
    let want = s.normal(s.up_at(hit.point));
    assert!(dvec3::dot(hit.normal, want) > 1.0 - 1e-12);
}

#[test]
fn rays_away_from_the_ground_or_past_the_planet_miss() {
    let s = hills();
    let up = [0.0, 1.0, 0.0];
    let origin = above(&s, up, 10.0);
    assert!(s.ray_hit(origin, up, f64::INFINITY).is_none());
    // Level off the top of the planet, well above every hill.
    let high = dvec3::add(CENTER, [0.0, s.radius + 500.0, 0.0]);
    assert!(s.ray_hit(high, [1.0, 0.0, 0.0], f64::INFINITY).is_none());
    // Far out in space and pointed beside the planet.
    let far = [0.0, 0.0, 400_000.0];
    let beside = dvec3::normalize_or(dvec3::sub([60_000.0, -50_000.0, 0.0], far), [0.0; 3]);
    assert!(s.ray_hit(far, beside, f64::INFINITY).is_none());
}

// From far out in space a ray at the center finds the ground nearest it.
#[test]
fn a_ray_from_far_away_finds_the_near_side() {
    let s = hills();
    let far = [0.0, 0.0, 2_000_000.0];
    let dir = dvec3::normalize_or(dvec3::sub(CENTER, far), [0.0; 3]);
    let hit = s.ray_hit(far, dir, f64::INFINITY).unwrap();
    assert_on_ground(&s, far, dir, &hit);
    assert!(dvec3::dot(s.up_at(hit.point), [0.0, 0.0, 1.0]) > 0.999);
}

#[test]
fn the_ground_beyond_the_reach_is_a_miss() {
    let s = ball();
    let origin = above(&s, [0.0, 1.0, 0.0], 300.0);
    let down = [0.0, -1.0, 0.0];
    assert!(s.ray_hit(origin, down, 299.0).is_none());
    assert!(s.ray_hit(origin, down, 301.0).is_some());
}

// A level ray whose closest approach passes just over the ball misses; one
// passing just under it dips in where the sphere equation says.
#[test]
fn grazing_rays_split_at_the_surface() {
    let s = ball();
    let start = |h: f64| dvec3::add(CENTER, [-800.0, s.radius + h, 0.0]);
    let level = [1.0, 0.0, 0.0];
    assert!(s.ray_hit(start(0.05), level, f64::INFINITY).is_none());
    let hit = s.ray_hit(start(-0.05), level, f64::INFINITY).unwrap();
    let half_chord = libm::sqrt(s.radius * s.radius - (s.radius - 0.05) * (s.radius - 0.05));
    assert!(
        (hit.distance - (800.0 - half_chord)).abs() < 1e-2,
        "{}",
        hit.distance
    );
    assert_on_ground(&s, start(-0.05), level, &hit);
}

// Grazing rays over hills agree with a dense scan of the same ray.
#[test]
fn grazing_rays_over_hills_agree_with_a_dense_scan() {
    let s = hills();
    for dir in directions(40) {
        let origin = above(&s, dir, 3.0);
        let [east, _, _] = crate::planet::geo_axes(dir);
        let ray = dvec3::normalize_or(dvec3::add(east, dvec3::scale(dir, -0.002)), [0.0; 3]);
        let hit = s.ray_hit(origin, ray, 3_000.0);
        let scanned = (1..30_000).map(|k| f64::from(k) * 0.1).find(|t| {
            s.clearance(dvec3::sub(
                dvec3::add(origin, dvec3::scale(ray, *t)),
                s.center,
            )) <= 0.0
        });
        match (hit, scanned) {
            (Some(h), Some(t)) => assert!((h.distance - t).abs() < 0.11, "{} {t}", h.distance),
            (None, None) => {}
            other => panic!("march and scan disagree: {other:?}"),
        }
    }
}

#[test]
fn a_ray_starting_below_the_ground_hits_nothing() {
    let s = hills();
    for dir in directions(20) {
        let under = above(&s, dir, -2.0);
        assert!(s.ray_hit(under, [0.0, 1.0, 0.0], f64::INFINITY).is_none());
        assert!(
            s.ray_hit(under, dvec3::scale(dir, -1.0), f64::INFINITY)
                .is_none()
        );
    }
}

// A frame moved tens of kilometers around the planet picks the same ground
// an authored ray does, to the millimeter, on a planet the size of Earth.
#[test]
fn a_pick_in_a_far_frame_stays_precise() {
    let radius = 6_360_000.0;
    let s = PlanetShape {
        center: [0.0, -radius, 0.0],
        radius,
        amplitude: 300.0,
        feature_size: 4_000.0,
        octaves: 10,
        seed: 3,
        continents: Continents::NONE,
        edits: None,
    };
    let mut frame = LocalFrame::AUTHORED;
    for _ in 0..40 {
        let at = [900.0, 0.0, -700.0];
        frame = frame.recentered(at, s.up_at(frame.to_world(at))).0;
    }
    let pf = PlanetFrame {
        shape: s.clone(),
        frame,
    };
    let here = frame.to_world([0.0; 3]);
    let ground = s.surface(s.up_at(here));
    let origin = frame.to_local(dvec3::add(ground, dvec3::scale(s.up_at(here), 50.0)));
    let dir = [0.3, -1.0, 0.1];
    let pick = pf.pick_ground(origin, dir, f64::INFINITY).unwrap();
    let want = s
        .ray_hit(
            frame.to_world(origin),
            dvec3::normalize_or(frame.dir_to_world(dir), [0.0; 3]),
            f64::INFINITY,
        )
        .unwrap();
    assert!(pick.geo.height_above_ground.abs() < 1e-3, "{:?}", pick.geo);
    assert_eq!(pick.distance, want.distance);
    let back = frame.to_world(pick.point);
    assert!(dvec3::length(dvec3::sub(back, want.point)) < 0.01);
    // The local normal leans the way the local ground does.
    let up = pf.up_at(pick.point);
    let dot: f32 = (0..3).map(|i| pick.normal[i] * up[i]).sum();
    assert!(dot > 0.5, "{dot}");
    assert!(pf.pick_ground(origin, [0.0; 3], f64::INFINITY).is_none());
}
