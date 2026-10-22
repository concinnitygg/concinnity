use super::*;
use crate::components::{Planet, WaterShape, Wind};
use crate::render::environment::water::sea::shore::ShoreWaves;
use concinnity_core::math::{dvec3, narrow_f32};

// The planet example's world: a coast east of the start, its sea in a
// 4 m/s breeze.
fn planet() -> PlanetShape {
    Planet {
        center: [0.0, -50_000.0, 0.0],
        radius: 50_000.0,
        amplitude: 40.0,
        feature_size: 2000.0,
        octaves: 8,
        seed: 95,
        land_share: 0.4,
        continent_size: 20_000.0,
        ocean_depth: 60.0,
        beach_width: 60.0,
        ..Planet::default()
    }
    .shape()
}

fn breeze() -> WindField {
    WindField::new(Some(&Wind {
        direction: [1.0, 0.35],
        strength: 4.0,
        ..Wind::default()
    }))
}

fn sea_query() -> WaterQuery {
    let sea = WaterSurface {
        shape: WaterShape::PlanetSea,
        fetch_meters: 5.0e4,
        wind_response: 1.75,
        ..Default::default()
    };
    WaterQuery::new([&sea], &breeze(), Some(&planet()))
}

fn shore(q: &WaterQuery) -> &ShoreWaves {
    q.bodies[0]
        .body
        .sea
        .as_ref()
        .unwrap()
        .shore
        .as_ref()
        .unwrap()
}

fn read(q: &WaterQuery, frame: &LocalFrame, at: [f32; 3], time: f64) -> WaterSample {
    let spot = q.spot(frame, at);
    q.sample(&spot, frame, at, time).expect("over the sea")
}

// The authored direction of local point `at` from the planet's center.
fn dir_of(frame: &LocalFrame, at: [f32; 3]) -> DVec3 {
    dvec3::normalize_or(
        dvec3::sub(frame.to_world(at), planet().center),
        [0.0, 1.0, 0.0],
    )
}

// A point on the sea's rest east of the start where the water is `depth`
// meters deep, give or take a meter.
fn east_where(q: &WaterQuery, depth: f64) -> [f32; 3] {
    let p = planet();
    (0..2000)
        .map(|i| [i as f32 * 0.5, 0.0, 28.0])
        .find(|&at| {
            let (inland, _) = p.inland(dir_of(&LocalFrame::AUTHORED, at)).unwrap();
            (shore(q).coast().depth(inland) - depth).abs() < 0.5
        })
        .expect("a coast to the east")
}

fn close(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() < tol)
}

// Still water lies at its rest: a point above it is out of the water by its
// height, the surface is level and the water still. Outside the rectangle
// there is none.
#[test]
fn still_water_lies_at_rest() {
    let pond = WaterSurface {
        center: [2.0, 1.5, -3.0],
        extent: [4.0, 4.0],
        ..Default::default()
    };
    let q = WaterQuery::new([&pond], &WindField::STILL, None);
    let s = read(&q, &LocalFrame::AUTHORED, [3.0, 1.8, -2.0], 9.0);
    assert!((s.depth + 0.3).abs() < 1e-6, "{}", s.depth);
    assert_eq!(s.normal, [0.0, 1.0, 0.0]);
    assert_eq!(s.velocity, [0.0; 3]);
    assert_eq!(s.density, 1025.0);
    let spot = q.spot(&LocalFrame::AUTHORED, [9.0, 1.0, 0.0]);
    assert!(
        q.sample(&spot, &LocalFrame::AUTHORED, [9.0, 1.0, 0.0], 0.0)
            .is_none()
    );
    // Its water goes 10 m down by default; under that, a cave stays dry.
    let deep = read(&q, &LocalFrame::AUTHORED, [3.0, -8.4, -2.0], 9.0);
    assert!((deep.depth - 9.9).abs() < 1e-5, "{}", deep.depth);
    let cave = [3.0, -8.6, -2.0];
    assert!(q.sample(&spot, &LocalFrame::AUTHORED, cave, 9.0).is_none());
}

// The summed surface's drawn point, as `water_displace` moves rest point
// `xz`: the query finds the surface there.
#[test]
fn a_summed_surface_reads_its_drawn_waves() {
    let lake = WaterSurface {
        extent: [60.0, 60.0],
        subdivisions: 200,
        ..Default::default()
    };
    let q = WaterQuery::new([&lake], &breeze(), None);
    let body = &q.bodies[0].body;
    assert!(body.displacing > 0 && body.ocean.is_none());
    let time = 31.7;
    let mut steepest = 0.0f32;
    for i in 0..40 {
        let xz = [f64::from(i) * 1.37 - 20.0, 13.0 - f64::from(i) * 0.71];
        let mut p = [xz[0], 0.0, xz[1]];
        for w in &body.waves[..body.displacing] {
            let k = f64::from(w.wave.wavenumber());
            let d = w.wave.direction.map(f64::from);
            let phase = k * (d[0] * xz[0] + d[1] * xz[1]) + f64::from(w.wave.phase)
                - f64::from(w.wave.angular_speed()) * time;
            let a = f64::from(w.wave.amplitude);
            p[0] += d[0] * f64::from(w.pinch) * a * libm::cos(phase);
            p[1] += a * libm::sin(phase);
            p[2] += d[1] * f64::from(w.pinch) * a * libm::cos(phase);
        }
        let at = [narrow_f32(p[0]), narrow_f32(p[1]) - 0.25, narrow_f32(p[2])];
        let s = read(&q, &LocalFrame::AUTHORED, at, time);
        assert!((s.depth - 0.25).abs() < 5e-3, "{i}: {}", s.depth);
        steepest = steepest.max(1.0 - s.normal[1]);
    }
    assert!(steepest > 1e-4, "the waves tilt the surface");
}

// A large rectangle's simulated sea: the rest point the query finds under a
// drawn point moved there.
#[test]
fn a_simulated_rectangle_reads_its_band() {
    let sea = WaterSurface {
        extent: [3000.0, 3000.0],
        fetch_meters: 1.0e5,
        ..Default::default()
    };
    let wind = WindField::new(Some(&Wind {
        strength: 9.0,
        ..Wind::default()
    }));
    let mut q = WaterQuery::new([&sea], &wind, None);
    assert!(q.bodies[0].band.is_some());
    let time = 12.34;
    q.prepare(time);
    let band = q.bands[0].as_ref().unwrap();
    let mut moving = 0.0f32;
    for i in 0..30 {
        let rest = [f64::from(i) * 7.3 - 90.0, 40.0 - f64::from(i) * 3.1];
        let b = band.sample(time, rest);
        let at = [
            narrow_f32(rest[0] + f64::from(b.shift[0])),
            b.height - 0.4,
            narrow_f32(rest[1] + f64::from(b.shift[1])),
        ];
        let s = read(&q, &LocalFrame::AUTHORED, at, time);
        assert!((s.depth - 0.4).abs() < 0.02, "{i}: {}", s.depth);
        moving = moving.max(s.velocity[1].abs());
    }
    assert!(moving > 0.1, "{moving}");
}

// In the surf the open sea is damped away and the shore's trains are the
// surface: the query reads `ShoreWaves::at` where the waves drew it.
#[test]
fn the_surf_reads_the_shore_waves() {
    let q = sea_query();
    let frame = LocalFrame::AUTHORED;
    let rest = east_where(&q, 1.5);
    let time = 41.0;
    let p = planet();
    let r = q.bodies[0].body.sea.as_ref().unwrap().sphere.radius;
    let mut seen = 0.0f64;
    for i in 0..12 {
        let at = [rest[0], 0.0, rest[2] + i as f32 * 3.0];
        let d = dir_of(&frame, at);
        let w = shore(&q).at(&p, d, time);
        let depth = shore(&q).coast().depth(p.inland(d).unwrap().0);
        assert_eq!(shore(&q).cascade_damping(depth)[0], 0.0);
        let drawn = dvec3::add(
            dvec3::add(p.center, dvec3::scale(d, r + w.height - 0.3)),
            w.shift,
        );
        let s = read(&q, &frame, frame.to_local(drawn), time);
        assert!(
            (f64::from(s.depth) - 0.3).abs() < 0.02,
            "{i}: {} vs {}",
            s.depth,
            w.height
        );
        seen = seen.max(w.height.abs());
    }
    assert!(seen > 0.05, "{seen}");
}

// A body's samples share the spot taken at its center: a few meters off it
// they read the surf almost as a spot of their own would.
#[test]
fn a_nearby_spot_reads_the_same_surf() {
    let q = sea_query();
    let frame = LocalFrame::AUTHORED;
    let center = east_where(&q, 2.0);
    let spot = q.spot(&frame, center);
    for offset in [[2.0, 0.0, 0.0], [0.0, 0.0, -3.0], [-2.5, 0.3, 2.5]] {
        let at = [center[0] + offset[0], offset[1], center[2] + offset[2]];
        let shared = q.sample(&spot, &frame, at, 7.0).unwrap();
        let own = read(&q, &frame, at, 7.0);
        assert!((shared.depth - own.depth).abs() < 0.02, "{offset:?}");
    }
}

// A move of the simulated frame changes nothing the sea reports: the same
// place on the planet reads the same depth, and the same normal and
// velocity turned into the new frame.
#[test]
fn a_frame_move_leaves_the_water_level_unchanged() {
    let mut q = sea_query();
    let frame = LocalFrame::AUTHORED;
    let time = 63.3;
    q.prepare(time);
    let up = dir_of(&frame, [900.0, 0.0, -400.0]);
    let (moved, _) = frame.recentered([900.0, 30.0, -400.0], up);
    for at in [
        east_where(&q, 1.0),
        east_where(&q, 6.0),
        [700.0, 0.4, 40.0],
        [1200.0, -0.3, -900.0],
    ] {
        let a = read(&q, &frame, at, time);
        let there = moved.to_local(frame.to_world(at));
        let b = read(&q, &moved, there, time);
        assert!(
            (a.depth - b.depth).abs() < 2e-3,
            "{at:?}: {} vs {}",
            a.depth,
            b.depth
        );
        let turn = |v: [f32; 3]| moved.dir_to_local(frame.dir_to_world(v));
        assert!(close(turn(a.normal), b.normal, 1e-4), "{at:?}");
        assert!(close(turn(a.velocity), b.velocity, 1e-3), "{at:?}");
    }
}

// The open sea rises and falls: away from the shore the band moves the
// surface, and its up is the sea's radial up.
#[test]
fn the_open_sea_moves() {
    let mut q = sea_query();
    let frame = LocalFrame::AUTHORED;
    let at = [1500.0, 0.0, 40.0];
    let mut heights = alloc::vec![];
    for i in 0..20 {
        let time = 10.0 + f64::from(i) * 0.25;
        q.prepare(time);
        let s = read(&q, &frame, at, time);
        let radial = frame.dir_to_local(dir_of(&frame, at));
        assert!(close(s.up, radial, 1e-5));
        heights.push(s.depth);
    }
    let lo = heights.iter().copied().fold(f32::MAX, f32::min);
    let hi = heights.iter().copied().fold(f32::MIN, f32::max);
    assert!(hi - lo > 0.1, "{lo} {hi}");
}

// The same water read twice reads the same.
#[test]
fn the_query_is_deterministic() {
    let mut a = sea_query();
    let mut b = sea_query();
    a.prepare(5.5);
    b.prepare(5.5);
    for at in [[300.0, 0.0, 10.0], [1500.0, 0.2, 40.0]] {
        let frame = LocalFrame::AUTHORED;
        assert_eq!(read(&a, &frame, at, 5.5), read(&b, &frame, at, 5.5));
    }
}

// One clock for what floats and what it floats on. Through uneven frames, a
// menu pause and a hitch longer than the tick clock keeps, the water is drawn
// at the instant the frame's blended poses show: between the state before the
// last tick and the state after it, by the frame's alpha. It never runs back,
// stands still while paused, and the sea it draws at one place on the planet
// moves continuously, across a move of the simulated frame too.
#[test]
fn the_water_runs_on_the_simulation_clock() {
    use crate::ecs::TickAccumulator;
    use crate::render::environment::water::frame::water_time;
    let mut clock = TickAccumulator::default();
    let mut q = sea_query();
    let mut frame = LocalFrame::AUTHORED;
    let place = frame.to_world([1500.0, 0.0, 40.0]);
    let mut frames = alloc::vec![(1.0 / 144.0, false); 90];
    frames.extend([(1.0 / 30.0, false); 20]);
    frames.extend([(1.0 / 60.0, true); 30]);
    frames.push((0.5, false));
    frames.extend([(1.0 / 60.0, false); 60]);
    let mut tick_ends = alloc::vec![0.0f64];
    let mut last: Option<(f64, f32)> = None;
    for (i, &(dt, paused)) in frames.iter().enumerate() {
        let (timing, _) = clock.advance(dt, paused);
        for k in 0..timing.ticks {
            tick_ends.push(timing.tick_start(k) + f64::from(timing.tick_dt));
        }
        if i == 150 {
            let at = frame.to_local(place);
            let up = dir_of(&frame, at);
            frame = frame.recentered([at[0], 30.0, at[2]], up).0;
        }
        let drawn = water_time(&timing);
        q.prepare(drawn);
        let height = read(&q, &frame, frame.to_local(place), drawn).depth;
        if tick_ends.len() < 3 {
            continue;
        }
        let [before, after] = [
            tick_ends[tick_ends.len() - 2],
            tick_ends[tick_ends.len() - 1],
        ];
        let blended = before + (after - before) * f64::from(timing.alpha);
        assert!((drawn - blended).abs() < 1e-9, "{i}: {drawn} vs {blended}");
        if let Some((t, h)) = last {
            assert!(drawn >= t, "{i}: ran back");
            if paused {
                assert_eq!((drawn, height), (t, h), "{i}: moved while paused");
            }
            // The sea rises no faster than a few meters a second.
            let bound = 3.0 * narrow_f32(drawn - t) + 2e-3;
            assert!((height - h).abs() <= bound, "{i}: {h} -> {height}");
        }
        last = Some((drawn, height));
    }
}
