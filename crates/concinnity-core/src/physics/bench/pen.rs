// Benchmarks over a pen of balls under hanging chains with regions over it,
// the shape of the benchmark world's physics station.
//
// `physics/step_pen/*` is the pile held awake: every ball touches its
// neighbors, so the whole pile is one island and the solve cannot be split.
// What its `_serial` twin shows is what a pool costs a world like that, which
// the fan-out gate is meant to keep at nothing.
//
// `physics/step_pen_settled/*` is the same pen asleep under its regions: the
// idle path, where nothing moves and so nothing needs measuring or solving.

use alloc::format;
use alloc::vec::Vec;

use super::{Pool, WORKERS};
use crate::physics::{
    BodyHandle, ColliderShape, DynamicParams, Inline, JointSpec, LayerMask, SimConfig, Simulation,
};
use crate::test_support::{Pace, bench};

const TICK: f32 = 1.0 / 60.0;

const PEN_HALF_WIDTH: f32 = 7.0;
const BALL_RADIUS: f32 = 0.42;
const BALL_SPACING: f32 = 1.15;
const LINK_HALF_EXTENT: f32 = 0.35;
const LINK_DROP: f32 = 1.1;
const CHAIN_ANCHOR_HEIGHT: f32 = 9.0;
const REGIONS: usize = 4;

/// How many of each thing a pen holds.
struct Size {
    balls: [usize; 3],
    chains: usize,
    links: usize,
}

impl Size {
    fn bodies(&self) -> usize {
        self.balls.iter().product::<usize>() + self.chains * self.links
    }
}

/// How the bodies in a pen behave.
#[derive(Clone, Copy)]
struct Temper {
    restitution: f32,
    damping: f32,
}

// The station's own: bouncy and barely damped.
const LIVELY: Temper = Temper {
    restitution: 0.66,
    damping: 0.05,
};

// Dead and heavily damped, so the pile is asleep within seconds.
const DEAD: Temper = Temper {
    restitution: 0.0,
    damping: 1.0,
};

/// Counts and step budgets per pace.
struct Fixture {
    size: Size,
    label: &'static str,
    warmup_steps: usize,
    settle_steps: usize,
    settled: bool,
}

fn fixture(pace: Pace) -> Fixture {
    match pace {
        Pace::Timed => Fixture {
            size: Size {
                balls: [6, 5, 6],
                chains: 6,
                links: 5,
            },
            label: "210",
            warmup_steps: 120,
            settle_steps: 900,
            settled: true,
        },
        Pace::Once => Fixture {
            size: Size {
                balls: [2, 1, 2],
                chains: 1,
                links: 2,
            },
            label: "6",
            warmup_steps: 2,
            settle_steps: 4,
            settled: false,
        },
    }
}

fn spread(i: usize, count: usize, spacing: f32) -> f32 {
    (i as f32 - (count as f32 - 1.0) * 0.5) * spacing
}

fn params(mass: f32, friction: f32, temper: Temper) -> DynamicParams {
    DynamicParams {
        mass,
        friction,
        restitution: temper.restitution,
        gravity_scale: 1.0,
        linear_damping: temper.damping,
    }
}

// The pen: a floor, four walls, the balls stacked a hand's width over the floor
// so the pile forms at once, chains hung from world anchors, and the regions.
fn pen_world(size: &Size, temper: Temper) -> (Simulation, Vec<BodyHandle>) {
    // Each chain's anchor is a body of its own.
    let capacity = 5 + size.bodies() + size.chains + REGIONS;
    let mut sim = Simulation::new(SimConfig::default(), capacity);
    let all = LayerMask::ALL;
    sim.add_fixed(
        &ColliderShape::Cuboid {
            half_extents: [200.0, 1.0, 200.0],
        },
        [0.0, -1.0, 0.0],
        [0.0; 3],
        0.8,
        all,
    )
    .expect("room for the floor");
    let wall = ColliderShape::Cuboid {
        half_extents: [PEN_HALF_WIDTH, 2.0, 0.4],
    };
    for (offset, turn) in [
        ([0.0, PEN_HALF_WIDTH], 0.0),
        ([0.0, -PEN_HALF_WIDTH], 0.0),
        ([PEN_HALF_WIDTH, 0.0], 90.0),
        ([-PEN_HALF_WIDTH, 0.0], 90.0),
    ] {
        sim.add_fixed(
            &wall,
            [offset[0], 2.0, offset[1]],
            [0.0, turn, 0.0],
            0.8,
            all,
        )
        .expect("room for a wall");
    }

    let mut handles = Vec::with_capacity(size.bodies());
    let [nx, ny, nz] = size.balls;
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let position = [
                    spread(x, nx, BALL_SPACING),
                    0.5 + y as f32 * BALL_SPACING,
                    spread(z, nz, BALL_SPACING),
                ];
                let ball = sim
                    .add_dynamic(
                        &ColliderShape::Ball {
                            radius: BALL_RADIUS,
                        },
                        position,
                        [0.0; 3],
                        params(1.0, 0.4, temper),
                        all,
                    )
                    .expect("room for a ball");
                handles.push(ball);
            }
        }
    }

    let link = ColliderShape::Cuboid {
        half_extents: [LINK_HALF_EXTENT; 3],
    };
    for chain in 0..size.chains {
        let along = spread(chain, size.chains, PEN_HALF_WIDTH * 0.5);
        let mut above = sim
            .add_fixed(
                &ColliderShape::Ball { radius: 0.001 },
                [along, CHAIN_ANCHOR_HEIGHT + LINK_DROP, along * 0.4],
                [0.0; 3],
                0.0,
                all,
            )
            .expect("room for an anchor");
        let mut anchor_b = [0.0; 3];
        for index in 0..size.links {
            let position = [
                along,
                CHAIN_ANCHOR_HEIGHT - index as f32 * LINK_DROP,
                along * 0.4,
            ];
            let body = sim
                .add_dynamic(&link, position, [0.0; 3], params(1.5, 0.5, temper), all)
                .expect("room for a link");
            assert!(
                sim.add_joint(
                    body,
                    above,
                    [0.0, LINK_HALF_EXTENT, 0.0],
                    anchor_b,
                    JointSpec::Spherical,
                ),
                "the joint has to be made"
            );
            handles.push(body);
            above = body;
            anchor_b = [0.0, -LINK_HALF_EXTENT, 0.0];
        }
    }

    for index in 0..REGIONS {
        sim.add_sensor(
            &ColliderShape::Cuboid {
                half_extents: [PEN_HALF_WIDTH * 0.3, 2.5, PEN_HALF_WIDTH],
            },
            [spread(index, REGIONS, PEN_HALF_WIDTH * 0.6), 3.0, 0.0],
            [0.0; 3],
            index as u64,
            all,
        )
        .expect("room for a region");
    }
    sim.reserve_workers(WORKERS);
    (sim, handles)
}

fn asleep(sim: &Simulation, handles: &[BodyHandle]) -> usize {
    handles
        .iter()
        .filter(|&&h| sim.is_sleeping(h) == Some(true))
        .count()
}

pub(super) fn run(pace: Pace) {
    let Fixture {
        size,
        label,
        warmup_steps,
        settle_steps,
        settled,
    } = fixture(pace);
    let bodies = size.bodies() as u64;

    // Held awake: a lively pile still settles eventually, and one that fell
    // asleep partway through the window would report how fast it settled.
    for (name, serial) in [("step_pen", false), ("step_pen_serial", true)] {
        let (mut sim, _handles) = pen_world(&size, LIVELY);
        sim.set_config(SimConfig {
            allow_sleep: false,
            ..*sim.config()
        });
        for _ in 0..warmup_steps {
            sim.step_with(TICK, &Pool);
        }
        bench(pace, &format!("physics/{name}/{label}"), bodies, || {
            if serial {
                sim.step_with(TICK, &Inline);
            } else {
                sim.step_with(TICK, &Pool);
            }
            sim.contact_count()
        });
    }

    let (mut resting, handles) = pen_world(&size, DEAD);
    for _ in 0..settle_steps {
        resting.step_with(TICK, &Pool);
    }
    if settled {
        assert_eq!(
            asleep(&resting, &handles),
            handles.len(),
            "the settled pen was still moving when measured"
        );
    }
    bench(
        pace,
        &format!("physics/step_pen_settled/{label}"),
        bodies,
        || {
            resting.step_with(TICK, &Pool);
            resting.sensor_overlap_count()
        },
    );
    bench(
        pace,
        &format!("physics/step_pen_settled_serial/{label}"),
        bodies,
        || {
            resting.step_with(TICK, &Inline);
            resting.sensor_overlap_count()
        },
    );
    if settled {
        assert_eq!(
            resting.sensor_pairs_measured(),
            0,
            "a still pair was measured"
        );
    }
}

#[test]
#[ignore = "benchmark; run with --ignored --test-threads=1"]
fn bench_pen() {
    run(Pace::Timed);
}

#[test]
fn pen_fixtures_build_and_run() {
    run(Pace::Once);
}

#[cfg(test)]
mod tests {
    use super::*;

    // The measured pen is the station's: the same counts, so a number here
    // speaks for the benchmark world's physics row.
    #[test]
    fn the_measured_pen_holds_what_the_station_does() {
        assert_eq!(fixture(Pace::Timed).size.bodies(), 210);
        assert!(fixture(Pace::Once).size.bodies() > 0);
    }

    // A pile that does not fit inside the walls lands on them instead.
    #[test]
    fn the_pile_fits_inside_the_pen() {
        let size = fixture(Pace::Timed).size;
        let half_span = BALL_SPACING * (size.balls[0] - 1) as f32 * 0.5 + BALL_RADIUS;
        assert!(half_span < PEN_HALF_WIDTH);
    }
}
