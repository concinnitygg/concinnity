// What floating costs: a hundred eight-piece boxes on the planet example's
// sea, from the surf out over the shelf, pushed by the water for a second of
// ticks, and the snapshots of the sea's longest waves those ticks build.

use alloc::format;
use alloc::vec::Vec;

use crate::components::{Buoyancy, Planet, WaterShape, WaterSurface, Wind};
use crate::physics::float::Floater;
use crate::physics::{BodyHandle, ColliderShape, DynamicParams, LayerMask, SimConfig, Simulation};
use crate::planet::LocalFrame;
use crate::test_support::{Pace, bench};
use crate::water::query::WaterQuery;
use crate::wind::WindField;

const TICK: f32 = 1.0 / 60.0;
const SIDE: usize = 10;

fn sea() -> WaterQuery {
    let planet = Planet {
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
    .shape();
    let surface = WaterSurface {
        shape: WaterShape::PlanetSea,
        fetch_meters: 5.0e4,
        wind_response: 1.75,
        ..Default::default()
    };
    let wind = WindField::new(Some(&Wind {
        direction: [1.0, 0.35],
        strength: 4.0,
        ..Wind::default()
    }));
    WaterQuery::new([&surface], &wind, Some(&planet))
}

// The boxes on a grid from the surf east of the start out to sea.
fn fleet() -> (Simulation, Vec<(BodyHandle, Floater)>) {
    let mut sim = Simulation::new(
        SimConfig {
            gravity: 9.81,
            gravity_center: Some([0.0, -50_000.0, 0.0]),
            ..SimConfig::default()
        },
        SIDE * SIDE,
    );
    let shape = ColliderShape::Cuboid {
        half_extents: [0.5, 0.3, 0.5],
    };
    let mut floaters = Vec::with_capacity(SIDE * SIDE);
    for i in 0..SIDE * SIDE {
        let at = [
            240.0 + 18.0 * (i % SIDE) as f32,
            0.0,
            9.0 * (i / SIDE) as f32,
        ];
        let params = DynamicParams {
            mass: 150.0,
            ..DynamicParams::default()
        };
        let handle = sim
            .add_dynamic(&shape, at, [0.0; 3], params, LayerMask::ALL)
            .expect("room for the fleet");
        let stagger = concinnity_core::math::fract(i as f32 * 0.618_034);
        floaters.push((handle, Floater::new(&shape, Buoyancy::default(), stagger)));
    }
    (sim, floaters)
}

fn run(pace: Pace) {
    let ticks: u32 = if pace == Pace::Timed { 60 } else { 6 };
    let (mut sim, mut floaters) = fleet();
    let mut water = sea();
    let frame = LocalFrame::AUTHORED;
    let mut time = 100.0;
    // Settle the snapshot ring and every spot before timing.
    for _ in 0..12 {
        water.prepare(time);
        for (h, f) in &mut floaters {
            f.push(*h, &mut sim, &water, &frame, time, TICK);
        }
        time += f64::from(TICK);
    }
    let pieces = (SIDE * SIDE * 8) as u64;
    bench(
        pace,
        &format!("physics/float_push/{}x8", SIDE * SIDE),
        pieces * u64::from(ticks),
        || {
            for _ in 0..ticks {
                water.prepare(time);
                for (h, f) in &mut floaters {
                    f.push(*h, &mut sim, &water, &frame, time, TICK);
                }
                time += f64::from(TICK);
            }
            floaters.iter().filter(|(_, f)| f.afloat()).count()
        },
    );
    // The snapshots alone: a jump in time builds the two around it whole.
    let mut at = 0.0;
    bench(pace, "physics/float_band_snapshot", 2, || {
        at += 10.0;
        water.prepare(at);
    });
}

#[test]
#[ignore = "benchmark; run with --ignored --test-threads=1"]
fn bench_floating() {
    run(Pace::Timed);
}

#[test]
fn floating_fixtures_build_and_run() {
    run(Pace::Once);
}
