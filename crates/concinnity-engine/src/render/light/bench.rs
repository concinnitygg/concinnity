// Measures light packing for the clustered forward pass: a thousand mixed
// point, spot and rect-area lights drained into the GPU light block.
//
//     cargo test -p concinnity-engine --release -- --ignored --nocapture \
//         --test-threads=1 bench_light_pack

use std::vec::Vec;

use crate::components::{PointLight, RectAreaLight, SpotLight};
use crate::render::light::lights::build_light_data;
use crate::test_support::{Pace, bench};

fn grid(i: usize) -> [f32; 3] {
    [
        (i % 32) as f32 * 4.0 - 64.0,
        2.5,
        (i / 32) as f32 * 4.0 - 64.0,
    ]
}

fn run(pace: Pace) {
    let points: Vec<PointLight> = (0..700)
        .map(|i| PointLight {
            position: grid(i),
            ..Default::default()
        })
        .collect();
    let spots: Vec<SpotLight> = (0..200)
        .map(|i| SpotLight {
            position: grid(i + 700),
            cast_shadows: i % 4 == 0,
            ..Default::default()
        })
        .collect();
    let rects: Vec<RectAreaLight> = (0..100)
        .map(|i| RectAreaLight {
            center: grid(i + 900),
            ..Default::default()
        })
        .collect();
    bench(pace, "render/light_pack/1k", 1_000, || {
        build_light_data(&points, &spots, &rects).lights.len()
    });
}

#[test]
#[ignore = "benchmark; run with --ignored --test-threads=1"]
fn bench_light_pack() {
    run(Pace::Timed);
}

#[test]
fn light_pack_fixtures_build_and_run() {
    run(Pace::Once);
}
