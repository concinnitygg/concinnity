//! What "at rest" has to mean.
//!
//! The unit tests inside the crate check that each stage answers correctly.
//! These check the thing all of them add up to, and they are the assertions
//! the milestone is actually judged by: a settled body does not creep, does
//! not sink, does not gain energy, and a stack that was standing an
//! instant ago is still standing ten seconds later.
//!
//! Sleeping is off almost everywhere below. A sleeping body cannot jitter, so
//! leaving it on would let the sleep timer pass a test the solver failed. The
//! one place it is on is the test that sleeping engages at all.
//!
//! The query and platform tests at the end ask the same kind of question of
//! the parts that read the world rather than step it: a ray must find the
//! surface a body is actually resting on, and a stack riding a platform must
//! still be riding it a few seconds later.

use concinnity_physics::{
    BodyHandle, ColliderShape, DynamicParams, LayerMask, ShapeCast, SimConfig, Simulation,
};

const TICK: f32 = 1.0 / 60.0;
const CUBE_HALF: f32 = 0.5;
const CUBE: ColliderShape = ColliderShape::Cuboid {
    half_extents: [CUBE_HALF, CUBE_HALF, CUBE_HALF],
};

fn params(friction: f32, restitution: f32) -> DynamicParams {
    DynamicParams {
        mass: 1.0,
        friction,
        restitution,
        gravity_scale: 1.0,
        linear_damping: 0.0,
    }
}

fn awake_config() -> SimConfig {
    SimConfig {
        allow_sleep: false,
        ..SimConfig::default()
    }
}

/// A floor whose top surface is exactly `y = 0`.
fn add_floor(sim: &mut Simulation) {
    sim.add_fixed(
        &ColliderShape::Cuboid {
            half_extents: [20.0, 1.0, 20.0],
        },
        [0.0, -1.0, 0.0],
        [0.0; 3],
        0.8,
        LayerMask::ALL,
    )
    .expect("room for the floor");
}

/// A column of `count` cubes resting on a floor, one cube apart.
fn stack(count: usize, config: SimConfig) -> (Simulation, Vec<BodyHandle>) {
    let mut sim = Simulation::new(config, count + 1);
    add_floor(&mut sim);
    let handles = (0..count)
        .map(|level| {
            sim.add_dynamic(
                &CUBE,
                [0.0, CUBE_HALF + level as f32, 0.0],
                [0.0; 3],
                params(0.6, 0.0),
                LayerMask::ALL,
            )
            .expect("room for a cube")
        })
        .collect();
    (sim, handles)
}

fn positions(sim: &Simulation, handles: &[BodyHandle]) -> Vec<[f32; 3]> {
    handles
        .iter()
        .map(|&h| sim.body_pose(h).expect("live").0)
        .collect()
}

fn settle(sim: &mut Simulation, ticks: usize) {
    for _ in 0..ticks {
        sim.step(TICK);
    }
}

/// The largest single-axis move any body made in one tick, over `ticks` ticks.
fn largest_tick_move(sim: &mut Simulation, handles: &[BodyHandle], ticks: usize) -> f32 {
    let mut previous = positions(sim, handles);
    let mut largest = 0.0f32;
    for _ in 0..ticks {
        sim.step(TICK);
        let current = positions(sim, handles);
        for (before, after) in previous.iter().zip(&current) {
            for axis in 0..3 {
                largest = largest.max((after[axis] - before[axis]).abs());
            }
        }
        previous = current;
    }
    largest
}

// A settled body must be still, not nearly still. Two ticks of a body that
// creeps a tenth of a millimetre look identical; two hundred do not.
#[test]
fn a_settled_body_does_not_creep() {
    let (mut sim, handles) = stack(1, awake_config());
    settle(&mut sim, 300);
    let moved = largest_tick_move(&mut sim, &handles, 120);
    assert!(
        moved < 1.0e-5,
        "a resting body moved {moved:e} in one tick, which is creep, not rest"
    );
}

#[test]
fn a_settled_stack_does_not_creep() {
    let (mut sim, handles) = stack(8, awake_config());
    settle(&mut sim, 600);
    let moved = largest_tick_move(&mut sim, &handles, 120);
    assert!(
        moved < 1.0e-4,
        "a resting stack moved {moved:e} in one tick, which is jitter"
    );
}

// A soft contact rests slightly inside the surface, and that is fine. What is
// not fine is the depth growing: a body that sinks a little further every
// step ends up through the floor.
#[test]
fn a_settled_body_does_not_sink_through_the_floor() {
    let (mut sim, handles) = stack(1, awake_config());
    settle(&mut sim, 120);
    let early = positions(&sim, &handles)[0][1];
    settle(&mut sim, 600);
    let late = positions(&sim, &handles)[0][1];

    let slop = sim.config().linear_slop;
    assert!(
        late > CUBE_HALF - 4.0 * slop,
        "resting depth {:.5} is past the tolerance the solver settles at",
        CUBE_HALF - late
    );
    assert!(
        (late - early).abs() < 1.0e-4,
        "the body sank a further {:.6} over ten seconds",
        early - late
    );
}

#[test]
fn a_settled_stack_does_not_sink_through_the_floor() {
    let (mut sim, handles) = stack(8, awake_config());
    settle(&mut sim, 600);
    let early = positions(&sim, &handles);
    settle(&mut sim, 600);
    let late = positions(&sim, &handles);

    for (level, (before, after)) in early.iter().zip(&late).enumerate() {
        assert!(
            (after[1] - before[1]).abs() < 1.0e-4,
            "level {level} sank a further {:.6} over ten seconds",
            before[1] - after[1]
        );
        // Every cube must still be above the one below it by most of a cube.
        let floor_below = if level == 0 {
            0.0
        } else {
            late[level - 1][1] + CUBE_HALF
        };
        assert!(
            after[1] - CUBE_HALF > floor_below - 0.02,
            "level {level} sank into what is under it: {after:?}"
        );
    }
}

// Energy is the tell a position check can miss: a solver can hold a stack in
// place while quietly pumping it, and the pumping shows up as a collapse the
// first time anything disturbs the stack.
#[test]
fn a_settled_stack_does_not_gain_energy() {
    let (mut sim, _handles) = stack(8, awake_config());
    settle(&mut sim, 600);
    let settled = sim.total_energy();
    let mut peak = settled;
    for _ in 0..600 {
        sim.step(TICK);
        peak = peak.max(sim.total_energy());
    }
    assert!(
        peak <= settled + 1.0e-3,
        "the stack gained {:.6} energy at rest (from {settled:.4})",
        peak - settled
    );
}

#[test]
fn an_eight_high_stack_is_still_standing_after_ten_seconds() {
    let (mut sim, handles) = stack(8, awake_config());
    settle(&mut sim, 600);
    for (level, position) in positions(&sim, &handles).iter().enumerate() {
        let expected = CUBE_HALF + level as f32;
        assert!(
            (position[1] - expected).abs() < 0.05,
            "level {level} should be near {expected:.2}, is at {:.4}",
            position[1]
        );
        assert!(
            position[0].abs() < 0.1 && position[2].abs() < 0.1,
            "level {level} slid out from under the stack: {position:?}"
        );
    }
}

// The same run twice, to the bit. Everything above is worth nothing if the
// answer depends on the day.
#[test]
fn the_same_stack_settles_identically_twice() {
    let run = || {
        let (mut sim, handles) = stack(8, awake_config());
        settle(&mut sim, 600);
        positions(&sim, &handles)
            .iter()
            .map(|p| [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()])
            .collect::<Vec<_>>()
    };
    assert_eq!(run(), run());
}

// A stack that is genuinely still should stop being simulated, and stay put
// once it has.
#[test]
fn a_settled_stack_falls_asleep_and_stays_where_it_was() {
    let (mut sim, handles) = stack(8, SimConfig::default());
    settle(&mut sim, 600);
    assert!(
        handles.iter().all(|&h| sim.is_sleeping(h) == Some(true)),
        "the whole stack should have settled"
    );
    let asleep = positions(&sim, &handles);
    settle(&mut sim, 600);
    assert_eq!(positions(&sim, &handles), asleep);
}

// Sleeping must not become a trapdoor: something landing on a settled stack
// has to bring it back.
#[test]
fn a_sleeping_stack_wakes_when_something_lands_on_it() {
    let (mut sim, handles) = stack(4, SimConfig::default());
    settle(&mut sim, 600);
    assert!(handles.iter().all(|&h| sim.is_sleeping(h) == Some(true)));

    let top = *handles.last().expect("a stack");
    sim.apply_impulse(top, [3.0, 0.0, 0.0]);
    sim.step(TICK);
    assert_eq!(sim.is_sleeping(top), Some(false));
    // The push has to reach the bottom of the stack, not just the box it hit.
    settle(&mut sim, 4);
    assert!(
        handles.iter().all(|&h| sim.is_sleeping(h) == Some(false)),
        "the whole island should be awake again"
    );
}

// Friction has to hold below the friction angle and give way above it.
#[test]
fn a_box_holds_on_a_shallow_slope_and_slides_on_a_steep_one() {
    let slide = |slope_deg: f32, friction: f32| -> f32 {
        let mut sim = Simulation::new(awake_config(), 2);
        sim.add_fixed(
            &ColliderShape::Cuboid {
                half_extents: [20.0, 0.5, 20.0],
            },
            [0.0, 0.0, 0.0],
            [0.0, 0.0, slope_deg],
            friction,
            LayerMask::ALL,
        )
        .expect("room");
        let box_handle = sim
            .add_dynamic(
                &ColliderShape::Cuboid {
                    half_extents: [0.4, 0.4, 0.4],
                },
                [0.0, 1.2, 0.0],
                [0.0, 0.0, slope_deg],
                params(friction, 0.0),
                LayerMask::ALL,
            )
            .expect("room");
        settle(&mut sim, 60);
        let start = sim.body_pose(box_handle).expect("live").0;
        settle(&mut sim, 300);
        let end = sim.body_pose(box_handle).expect("live").0;
        ((end[0] - start[0]).powi(2) + (end[1] - start[1]).powi(2)).sqrt()
    };

    // A coefficient of 0.6 holds to 31 degrees.
    assert!(
        slide(20.0, 0.6) < 0.01,
        "a box must hold well below its friction angle"
    );
    assert!(
        slide(45.0, 0.6) > 0.5,
        "a box must slide well above its friction angle"
    );
    // No friction at all slides on any slope worth the name.
    assert!(slide(20.0, 0.0) > 0.5, "a frictionless box must slide");
}

// Restitution has to decay: every bounce lower than the last, and an end to
// them. A solver that returns more than it took never stops bouncing.
#[test]
fn a_bouncing_ball_loses_height_every_bounce_and_comes_to_rest() {
    let mut sim = Simulation::new(awake_config(), 2);
    add_floor(&mut sim);
    let ball = sim
        .add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 5.0, 0.0],
            [0.0; 3],
            params(0.3, 0.8),
            LayerMask::ALL,
        )
        .expect("room");

    let mut peaks = Vec::new();
    let mut rising = false;
    let mut peak = 0.0f32;
    let mut previous = 5.0f32;
    for _ in 0..1200 {
        sim.step(TICK);
        let height = sim.body_pose(ball).expect("live").0[1];
        if height > previous {
            rising = true;
            peak = peak.max(height);
        } else if rising {
            peaks.push(peak);
            rising = false;
            peak = 0.0;
        }
        previous = height;
    }

    assert!(peaks.len() >= 3, "the ball should bounce more than once");
    assert!(peaks[0] < 5.0, "no bounce may exceed the drop: {peaks:?}");
    for pair in peaks.windows(2) {
        assert!(pair[1] < pair[0], "every bounce must be lower: {peaks:?}");
    }
    let resting = sim.body_pose(ball).expect("live").0[1];
    assert!(
        (resting - 0.5).abs() < 0.02,
        "the ball rests at {resting:.4}"
    );
}

// A face contact has to absorb an impact of any size. The four corners of a
// box's face each sit on a lever arm, so a corner handed the whole approach
// speed on its own spins the box hard enough to throw it: at a couple of
// thousand units a second a tenth-scale box was launched hundreds of units
// into the air by the floor it was resting on. The sweep is not involved and
// is switched off here -- the box starts in contact and never leaves it, so
// there is nothing to tunnel through and nothing but the contact solve to
// account for what happens.
#[test]
fn a_resting_box_driven_hard_into_the_floor_stays_on_it() {
    let launched = |speed: f32| {
        let mut sim = Simulation::new(
            SimConfig {
                ccd_enabled: false,
                ..awake_config()
            },
            2,
        );
        add_floor(&mut sim);
        let half = 0.1;
        let box_body = sim
            .add_dynamic(
                &ColliderShape::Cuboid {
                    half_extents: [half; 3],
                },
                [0.0, half, 0.0],
                [0.0; 3],
                params(0.5, 0.0),
                LayerMask::ALL,
            )
            .expect("room");
        settle(&mut sim, 120);
        sim.set_linear_velocity(box_body, [0.0, -speed, 0.0]);

        let mut highest = half;
        let mut fastest_spin = 0.0f32;
        for _ in 0..120 {
            sim.step(TICK);
            highest = highest.max(sim.body_pose(box_body).expect("live").0[1]);
            let spin = sim.angular_velocity(box_body).expect("live");
            fastest_spin = fastest_spin
                .max((spin[0] * spin[0] + spin[1] * spin[1] + spin[2] * spin[2]).sqrt());
        }
        (
            highest,
            fastest_spin,
            sim.body_pose(box_body).expect("live").0[1],
        )
    };

    for speed in [100.0, 400.0, 600.0, 2000.0] {
        let (highest, fastest_spin, resting) = launched(speed);
        assert!(
            highest < 0.2,
            "at {speed} units a second the floor threw the box to y = {highest}"
        );
        assert!(
            fastest_spin < 10.0,
            "at {speed} units a second the contact spun the box to \
             {fastest_spin} radians a second"
        );
        assert!(
            (resting - 0.1).abs() < 0.02,
            "at {speed} units a second the box came to rest at y = {resting}"
        );
    }
}

// Every shape pair the narrow phase implements has to hold a body off
// another, not just the box pair the stack tests use. Resting height is the
// wrong question for the round ones -- a capsule stood on its cap topples
// because that is what a capsule does -- so the assertion is that the two
// centres never come closer than the shapes allow.
#[test]
fn every_shape_pair_holds_the_other_off_rather_than_passing_through() {
    // Each shape with the closest its surface ever comes to its own centre,
    // which is the distance a contact must never let the other side inside.
    let shapes = [
        ("ball", ColliderShape::Ball { radius: 0.4 }, 0.4),
        ("cuboid", CUBE, CUBE_HALF),
        (
            "capsule",
            ColliderShape::Capsule {
                half_height: 0.3,
                radius: 0.3,
            },
            0.3,
        ),
    ];
    for (lower_name, lower, lower_inradius) in shapes {
        for (upper_name, upper, upper_inradius) in shapes {
            let mut sim = Simulation::new(awake_config(), 3);
            add_floor(&mut sim);
            let base = sim
                .add_fixed(&lower, [0.0, 1.0, 0.0], [0.0; 3], 0.8, LayerMask::ALL)
                .expect("room");
            let dropped = sim
                .add_dynamic(
                    &upper,
                    [0.0, 3.0, 0.0],
                    [0.0; 3],
                    params(0.8, 0.0),
                    LayerMask::ALL,
                )
                .expect("room");

            let centre = sim.body_pose(base).expect("live").0;
            let mut closest = f32::INFINITY;
            for _ in 0..600 {
                sim.step(TICK);
                let position = sim.body_pose(dropped).expect("live").0;
                assert!(
                    position.iter().all(|c| c.is_finite()),
                    "{upper_name} on {lower_name} went unstable: {position:?}"
                );
                let gap = ((position[0] - centre[0]).powi(2)
                    + (position[1] - centre[1]).powi(2)
                    + (position[2] - centre[2]).powi(2))
                .sqrt();
                closest = closest.min(gap);
            }
            let allowed = lower_inradius + upper_inradius - 0.05;
            assert!(
                closest > allowed,
                "{upper_name} reached {closest:.4} from the centre of {lower_name}, \
                 inside the {allowed:.4} the two shapes leave"
            );
        }
    }
}

// A capsule on its side is the case that needs two contact points: with one,
// it rocks end over end instead of lying still.
#[test]
fn a_capsule_lying_on_a_floor_comes_to_rest() {
    let mut sim = Simulation::new(awake_config(), 2);
    add_floor(&mut sim);
    let capsule = sim
        .add_dynamic(
            &ColliderShape::Capsule {
                half_height: 0.6,
                radius: 0.25,
            },
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 90.0],
            params(0.6, 0.0),
            LayerMask::ALL,
        )
        .expect("room");
    settle(&mut sim, 300);
    let resting = sim.body_pose(capsule).expect("live").0[1];
    assert!(
        (resting - 0.25).abs() < 0.02,
        "the capsule should lie at its own radius, is at {resting:.4}"
    );
    let moved = largest_tick_move(&mut sim, &[capsule], 120);
    assert!(moved < 1.0e-4, "the capsule rocked {moved:e} in one tick");
}

// Bodies come and go while a world runs; the stack under them must not care.
#[test]
fn removing_a_body_from_under_a_stack_lets_the_rest_fall_and_settle() {
    let (mut sim, handles) = stack(4, SimConfig::default());
    settle(&mut sim, 600);
    assert!(sim.remove_body(handles[0]));
    settle(&mut sim, 600);
    for (level, position) in positions(&sim, &handles[1..]).iter().enumerate() {
        let expected = CUBE_HALF + level as f32;
        assert!(
            (position[1] - expected).abs() < 0.05,
            "level {level} should have dropped to {expected:.2}, is at {:.4}",
            position[1]
        );
    }
}

// A ray has to agree with the solver about where a body ended up. If it does
// not, an animation foot plant or a camera probe lands somewhere the world is
// not.
#[test]
fn a_ray_lands_on_the_surface_the_stack_settled_at() {
    let (mut sim, handles) = stack(4, awake_config());
    settle(&mut sim, 600);
    for (level, &handle) in handles.iter().enumerate() {
        let resting = sim.body_pose(handle).expect("live").0;
        let hit = sim
            .raycast(
                [resting[0], resting[1] + 4.0, resting[2]],
                [0.0, -1.0, 0.0],
                20.0,
                None,
                LayerMask::ALL,
            )
            .expect("something is down there");
        // The ray must reach the top face of the highest box, whichever level
        // it was fired above.
        let top = sim.body_pose(handles[handles.len() - 1]).expect("live").0[1] + CUBE_HALF;
        assert!(
            (hit.point[1] - top).abs() < 1.0e-3,
            "level {level}: the ray landed at {:.4}, the stack tops out at {top:.4}",
            hit.point[1]
        );
        assert!(hit.normal[1] > 0.999, "level {level}: {:?}", hit.normal);
    }

    // And a sweep of the same capsule the character controller will use stops
    // on that surface rather than inside it.
    let capsule = ColliderShape::Capsule {
        half_height: 0.4,
        radius: 0.3,
    };
    let top = sim.body_pose(handles[3]).expect("live").0[1] + CUBE_HALF;
    let hit = sim
        .shape_cast(&ShapeCast::new(
            capsule,
            [0.0, top + 3.0, 0.0],
            [0.0, -6.0, 0.0],
        ))
        .expect("a landing");
    let centre = top + 3.0 - hit.toi * 6.0;
    assert!(
        (centre - (top + 0.7)).abs() < 0.01,
        "the capsule should stand its own 0.7 above {top:.4}, stands at {centre:.4}"
    );
}

// A platform carries what is standing on it. This is the property the whole
// kinematic path exists for, and the one that breaks quietly: a stack that
// sleeps while the platform moves is left behind, and one the solver pushes
// too hard is thrown off.
#[test]
fn a_stack_rides_a_moving_platform_without_being_left_behind() {
    let mut sim = Simulation::new(SimConfig::default(), 6);
    add_floor(&mut sim);
    let platform = sim
        .add_kinematic(
            &ColliderShape::Cuboid {
                half_extents: [3.0, 0.25, 3.0],
            },
            [0.0, 0.25, 0.0],
            [0.0; 3],
            0.9,
            LayerMask::ALL,
        )
        .expect("room for the platform");
    let riders: Vec<BodyHandle> = (0..3)
        .map(|level| {
            sim.add_dynamic(
                &CUBE,
                [0.0, 1.0 + level as f32, 0.0],
                [0.0; 3],
                params(0.9, 0.0),
                LayerMask::ALL,
            )
            .expect("room for a rider")
        })
        .collect();

    // Long enough for the stack to settle and doze off before the ride.
    settle(&mut sim, 300);
    assert_eq!(sim.is_sleeping(riders[2]), Some(true), "settled first");
    let before = positions(&sim, &riders);

    // Three seconds of travel at two units a second.
    let mut travelled = 0.0f32;
    for _ in 0..180 {
        travelled += 2.0 / 60.0;
        sim.set_kinematic_translation(platform, [travelled, 0.25, 0.0]);
        sim.step(TICK);
    }

    assert_eq!(
        sim.body_pose(platform).expect("live").0,
        [travelled, 0.25, 0.0]
    );
    for (level, (start, now)) in before.iter().zip(positions(&sim, &riders)).enumerate() {
        let drift = (now[0] - start[0] - travelled).abs();
        assert!(
            drift < 0.2,
            "rider {level} slipped {drift:.4} behind the platform's {travelled:.4}"
        );
        assert!(
            (now[1] - start[1]).abs() < 0.05,
            "rider {level} changed height by {:.4}",
            now[1] - start[1]
        );
        assert!(
            now[2].abs() < 0.1,
            "rider {level} wandered sideways to {:.4}",
            now[2]
        );
    }
}
