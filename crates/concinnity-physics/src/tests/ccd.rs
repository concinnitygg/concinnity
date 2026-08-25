//! What "fast" has to mean.
//!
//! The step's own contact test only looks at where a body was and where it
//! ended up, so a body that covered more ground in one tick than the thing in
//! its way is thick was never in contact with it at either sample. These are
//! the cases that produces: a body against a thin slab, against terrain,
//! against another body running at it, a region crossed without stopping, and
//! a platform driven through something it should have pushed.
//!
//! Every tunnelling test is run twice, once with the sweep on and once with it
//! off, so each states the speed and the thickness it pins and fails for the
//! reason it was written for rather than because something else moved.
//!
//! Sleeping is left on where an island is what is being checked and turned off
//! elsewhere, on the same reasoning the settling suite uses: a sleeping body
//! cannot tunnel either.

use crate::{
    BodyHandle, CharacterMoveInput, ColliderShape, DynamicParams, JointSpec, LayerMask, SimConfig,
    Simulation,
};
use alloc::vec;
use alloc::vec::Vec;

const TICK: f32 = 1.0 / 60.0;

/// A slab a tenth of a unit thick, its faces at `y = +/- 0.05`. Thinner than
/// anything thrown at it below travels in a tick, which is the whole point.
const SLAB_HALF_THICKNESS: f32 = 0.05;
const SLAB: ColliderShape = ColliderShape::Cuboid {
    half_extents: [20.0, SLAB_HALF_THICKNESS, 20.0],
};
const FLOOR: ColliderShape = ColliderShape::Cuboid {
    half_extents: [20.0, 1.0, 20.0],
};

const BALL: ColliderShape = ColliderShape::Ball { radius: 0.1 };
const BOX: ColliderShape = ColliderShape::Cuboid {
    half_extents: [0.1, 0.1, 0.1],
};
const CAPSULE: ColliderShape = ColliderShape::Capsule {
    half_height: 0.15,
    radius: 0.1,
};

/// How far each shape reaches below its own centre.
const BALL_REACH: f32 = 0.1;
const BOX_REACH: f32 = 0.1;
const CAPSULE_REACH: f32 = 0.25;

/// Where the slab holds each of them: its face plus that reach.
fn resting_height(reach: f32) -> f32 {
    SLAB_HALF_THICKNESS + reach
}

fn config(ccd: bool) -> SimConfig {
    SimConfig {
        allow_sleep: false,
        ccd_enabled: ccd,
        ..SimConfig::default()
    }
}

fn params(gravity_scale: f32) -> DynamicParams {
    DynamicParams {
        mass: 1.0,
        friction: 0.5,
        restitution: 0.0,
        gravity_scale,
        linear_damping: 0.0,
    }
}

fn spawn(sim: &mut Simulation, shape: &ColliderShape, at: [f32; 3], gravity: f32) -> BodyHandle {
    sim.add_dynamic(shape, at, [0.0; 3], params(gravity), LayerMask::ALL)
        .expect("room for a body")
}

fn slab_world(ccd: bool) -> Simulation {
    let mut sim = Simulation::new(config(ccd), 2);
    sim.add_fixed(&SLAB, [0.0; 3], [0.0; 3], 0.8, LayerMask::ALL)
        .expect("room for the slab");
    sim
}

/// Where a shape dropped straight down at `speed` comes to rest.
fn dropped_onto_slab(shape: &ColliderShape, speed: f32, ccd: bool) -> f32 {
    let mut sim = slab_world(ccd);
    let body = spawn(&mut sim, shape, [0.0, 5.0, 0.0], 1.0);
    sim.set_linear_velocity(body, [0.0, -speed, 0.0]);
    for _ in 0..240 {
        sim.step(TICK);
    }
    assert_eq!(sim.ccd_overflows(), 0, "the sweep was declined");
    sim.body_pose(body).expect("a live body").0[1]
}

/// Where the same shape is at the end of the very step that took it past the
/// slab, which is what the sweep alone is answerable for.
fn arrival_height(shape: &ColliderShape, reach: f32, speed: f32) -> f32 {
    let mut sim = slab_world(true);
    let body = spawn(&mut sim, shape, [0.0, 5.0, 0.0], 1.0);
    sim.set_linear_velocity(body, [0.0, -speed, 0.0]);
    let landing = resting_height(reach) + 0.02;
    let mut y = 5.0f32;
    for _ in 0..600 {
        let before = y;
        sim.step(TICK);
        y = sim.body_pose(body).expect("a live body").0[1];
        if before > landing && y <= landing {
            return y;
        }
    }
    panic!("{shape:?} at {speed} never arrived");
}

/// The test the whole change exists for. A ball 0.2 across, dropped at 600
/// units a second, covers ten units in one tick against a slab a tenth of a
/// unit thick: a hundred times what the pair of them measure together, so
/// neither end of the tick has them anywhere near each other.
#[test]
fn a_fast_ball_crosses_a_thin_slab_without_the_sweep_and_rests_on_it_with_it() {
    let through = dropped_onto_slab(&BALL, 600.0, false);
    assert!(
        through < -10.0,
        "without the sweep the ball has to end up under the slab, at y = {through}"
    );

    let stopped = dropped_onto_slab(&BALL, 600.0, true);
    assert!(
        (stopped - resting_height(BALL_REACH)).abs() < 0.02,
        "with it the ball has to be resting on top, at y = {stopped}"
    );
}

// The speed something can be thrown at is not a world's to declare, so the
// sweep is asked from the slowest speed that can tunnel to an absurd one.
// Nothing here is about what the solver does with the impact afterwards: the
// measurement is the step the body arrived on.
#[test]
fn every_shape_arrives_on_the_slab_at_every_speed() {
    for (shape, reach, name) in [
        (BALL, BALL_REACH, "ball"),
        (BOX, BOX_REACH, "box"),
        (CAPSULE, CAPSULE_REACH, "capsule"),
    ] {
        for speed in [20.0, 60.0, 200.0, 600.0, 2000.0, 10_000.0, 100_000.0] {
            let arrived = arrival_height(&shape, reach, speed);
            assert!(
                arrived > 0.0,
                "{name} at {speed} units a second was already past the slab \
                 at y = {arrived}"
            );
            assert!(
                (arrived - resting_height(reach)).abs() < 0.02,
                "{name} at {speed} units a second arrived at y = {arrived} \
                 rather than on the surface"
            );
        }
    }
}

// And what the solver does with the impact afterwards: arriving on the slab
// is worth nothing if the contact then throws the body off it. A rounded
// shape carries its whole impulse through one contact point on its centre
// line, so it was never in any doubt. A box spreads the same impulse over the
// four corners of a face, each with a lever arm under it, and holds only
// because the manifold's points are solved as one system rather than one
// after another.
#[test]
fn every_shape_is_held_on_the_slab_at_every_speed() {
    for (shape, reach, name) in [
        (BALL, BALL_REACH, "ball"),
        (CAPSULE, CAPSULE_REACH, "capsule"),
        (BOX, BOX_REACH, "box"),
    ] {
        for speed in [20.0, 60.0, 200.0, 300.0, 2000.0, 100_000.0] {
            let rested = dropped_onto_slab(&shape, speed, true);
            assert!(
                (rested - resting_height(reach)).abs() < 0.02,
                "{name} at {speed} units a second came to rest at y = {rested}"
            );
        }
    }
}

// Terrain triangles have no thickness at all, so a grid is the surface a fast
// body is most able to miss.
#[test]
fn a_fast_body_stops_on_a_height_grid_it_would_otherwise_cross() {
    let run = |ccd: bool| {
        let mut sim = Simulation::new(config(ccd), 2);
        sim.add_heightfield(
            5,
            5,
            vec![0.0; 25],
            [40.0, 1.0, 40.0],
            [0.0; 3],
            LayerMask::ALL,
        )
        .expect("room for the terrain");
        let body = spawn(&mut sim, &BALL, [1.0, 5.0, -2.0], 1.0);
        sim.set_linear_velocity(body, [0.0, -600.0, 0.0]);
        for _ in 0..240 {
            sim.step(TICK);
        }
        assert_eq!(sim.heightfield_overflows(), 0, "the grid query gave up");
        sim.body_pose(body).expect("a live body").0[1]
    };
    assert!(
        run(false) < -10.0,
        "the grid has to be crossable without it"
    );
    let stopped = run(true);
    assert!((stopped - 0.1).abs() < 0.02, "resting at y = {stopped}");
}

// Both halves moving is the case a sweep against where the other body started
// gets exactly backwards: each would stop at the other's starting place,
// which is to say they would swap sides.
#[test]
fn two_bodies_running_at_each_other_meet_instead_of_swapping_places() {
    let run = |speed: f32, ccd: bool| {
        let mut sim = Simulation::new(config(ccd), 2);
        let left = spawn(&mut sim, &BALL, [-6.0, 0.0, 0.0], 0.0);
        let right = spawn(&mut sim, &BALL, [6.0, 0.0, 0.0], 0.0);
        sim.set_linear_velocity(left, [speed, 0.0, 0.0]);
        sim.set_linear_velocity(right, [-speed, 0.0, 0.0]);
        let mut closest = f32::INFINITY;
        let mut crossed = false;
        for _ in 0..60 {
            sim.step(TICK);
            let gap =
                sim.body_pose(right).expect("live").0[0] - sim.body_pose(left).expect("live").0[0];
            closest = closest.min(gap);
            crossed |= gap < 0.0;
        }
        assert_eq!(sim.ccd_overflows(), 0, "the sweep was declined");
        (crossed, closest)
    };

    assert!(
        run(200.0, false).0,
        "without the sweep the two have to pass through"
    );
    // Two 0.1 balls stop with their centres 0.2 apart, whatever they closed at.
    for speed in [200.0, 600.0, 2000.0, 10_000.0] {
        let (crossed, closest) = run(speed, true);
        assert!(!crossed, "at {speed} one ended up on the other's side");
        assert!(
            (closest - 0.2).abs() < 0.02,
            "at {speed} they closed to {closest}"
        );
    }
}

// The gate is the whole reason the common case does not pay for any of this.
#[test]
fn a_world_at_ordinary_speeds_never_arms_the_sweep() {
    let mut sim = Simulation::new(SimConfig::default(), 10);
    sim.add_fixed(&FLOOR, [0.0, -1.0, 0.0], [0.0; 3], 0.8, LayerMask::ALL)
        .expect("room for the floor");
    let cube = ColliderShape::Cuboid {
        half_extents: [0.5, 0.5, 0.5],
    };
    for level in 0..8 {
        spawn(&mut sim, &cube, [0.0, 0.5 + level as f32 * 1.2, 0.0], 1.0);
    }
    for tick in 0..600 {
        sim.step(TICK);
        assert_eq!(
            sim.swept_body_count(),
            0,
            "a settling stack took the expensive path on tick {tick}"
        );
    }
}

// An island that has settled has to be simulated again on the step the
// contact will be built, or a fast body leans on a sleeping stack and the
// stack never hears about it.
#[test]
fn a_stopped_body_wakes_the_whole_island_it_ran_into() {
    let mut sim = Simulation::new(SimConfig::default(), 6);
    sim.add_fixed(&FLOOR, [0.0, -1.0, 0.0], [0.0; 3], 0.8, LayerMask::ALL)
        .expect("room for the floor");
    let cube = ColliderShape::Cuboid {
        half_extents: [0.5, 0.5, 0.5],
    };
    let stack: Vec<BodyHandle> = (0..3)
        .map(|level| spawn(&mut sim, &cube, [0.0, 0.5 + level as f32, 0.0], 1.0))
        .collect();
    // Parked out of the way and weightless while the stack settles.
    let bullet = spawn(&mut sim, &BALL, [-8.0, 0.5, 0.0], 0.0);

    for _ in 0..600 {
        sim.step(TICK);
    }
    assert!(
        stack.iter().all(|&h| sim.is_sleeping(h) == Some(true)),
        "the stack has to be asleep before anything hits it"
    );

    sim.set_linear_velocity(bullet, [300.0, 0.0, 0.0]);
    let mut woke_on = None;
    for tick in 0..30 {
        sim.step(TICK);
        if stack.iter().any(|&h| sim.is_sleeping(h) == Some(false)) {
            woke_on = Some(tick);
            break;
        }
    }
    let tick = woke_on.expect("the stack has to wake");
    assert!(
        stack.iter().all(|&h| sim.is_sleeping(h) == Some(false)),
        "the whole island wakes together, not just the body that was hit \
         (tick {tick})"
    );
    let stopped = sim.body_pose(bullet).expect("live").0[0];
    assert!(
        stopped < -0.5,
        "the bullet stopped against the stack rather than inside it, at {stopped}"
    );
}

// A stop is a position change and nothing else, so the contact that follows it
// is an ordinary one: built by the narrow phase, warm started off its own
// feature, and settled rather than left ringing.
#[test]
fn a_body_stopped_by_the_sweep_settles_where_it_landed() {
    let mut sim = slab_world(true);
    let body = spawn(&mut sim, &BALL, [0.0, 5.0, 0.0], 1.0);
    sim.set_linear_velocity(body, [0.0, -2000.0, 0.0]);
    for _ in 0..180 {
        sim.step(TICK);
    }
    let (mut lowest, mut highest) = (f32::INFINITY, f32::NEG_INFINITY);
    for _ in 0..120 {
        sim.step(TICK);
        let y = sim.body_pose(body).expect("live").0[1];
        lowest = lowest.min(y);
        highest = highest.max(y);
    }
    assert!(
        highest - lowest < 1.0e-3,
        "the landing has to be still: {lowest} to {highest}"
    );
    assert!(
        (lowest - resting_height(BALL_REACH)).abs() < 0.02,
        "and it must not sink through, at y = {lowest}"
    );
}

// A region measures overlap at step boundaries, so a body that covered the
// whole of one between two of them was never sampled inside it.
#[test]
fn a_body_that_crosses_a_region_in_one_tick_still_reports_both_boundaries() {
    let run = |ccd: bool| {
        let mut sim = Simulation::new(config(ccd), 2);
        sim.add_sensor(
            &ColliderShape::Cuboid {
                half_extents: [1.0, 2.0, 2.0],
            },
            [0.0; 3],
            [0.0; 3],
            7,
            LayerMask::ALL,
        )
        .expect("room for the region");
        let body = spawn(&mut sim, &BALL, [-8.0, 0.0, 0.0], 0.0);
        sim.set_linear_velocity(body, [600.0, 0.0, 0.0]);
        let mut crossings = Vec::new();
        let mut seen = Vec::new();
        for _ in 0..30 {
            sim.step(TICK);
            sim.drain_sensor_crossings_into(&mut crossings);
            seen.extend(crossings.iter().map(|c| (c.tag, c.entered)));
        }
        assert_eq!(sim.sensor_overflows(), 0, "the crossing queue overflowed");
        assert_eq!(sim.ccd_overflows(), 0, "the sweep was declined");
        seen
    };

    assert!(
        run(false).is_empty(),
        "without the sweep the region never sees it"
    );
    assert_eq!(
        run(true),
        [(7, true), (7, false)],
        "with it the entry and the exit are both recorded, in that order"
    );
}

// A region a body stops inside is still the boundary test's, and reporting it
// here as well would hand a caller the entry twice.
#[test]
fn a_body_that_falls_to_rest_inside_a_region_reports_one_entry_and_no_exit() {
    let mut sim = Simulation::new(config(true), 3);
    sim.add_fixed(&FLOOR, [0.0, -1.0, 0.0], [0.0; 3], 0.8, LayerMask::ALL)
        .expect("room for the floor");
    sim.add_sensor(
        &ColliderShape::Cuboid {
            half_extents: [4.0, 4.0, 4.0],
        },
        [0.0, 1.0, 0.0],
        [0.0; 3],
        11,
        LayerMask::ALL,
    )
    .expect("room for the region");
    let body = spawn(&mut sim, &BALL, [0.0, 20.0, 0.0], 1.0);
    sim.set_linear_velocity(body, [0.0, -600.0, 0.0]);
    let mut crossings = Vec::new();
    let mut seen = Vec::new();
    for _ in 0..120 {
        sim.step(TICK);
        sim.drain_sensor_crossings_into(&mut crossings);
        seen.extend(crossings.iter().map(|c| (c.tag, c.entered)));
    }
    assert_eq!(seen, [(11, true)], "it went in once and stayed there");
}

// A mover that is stopped inside a region never reached the far side of it,
// however far the step was going to carry it. Reporting the pass-through it
// would have made hands the caller an exit that never happened and an entry
// the boundary test is about to report itself.
#[test]
fn a_body_stopped_inside_a_region_reports_one_entry_and_no_exit() {
    let mut sim = Simulation::new(config(true), 3);
    // The wall stands inside the region, so the sweep crosses the boundary
    // and is stopped before it can leave.
    sim.add_sensor(
        &ColliderShape::Cuboid {
            half_extents: [2.0, 2.0, 2.0],
        },
        [0.0; 3],
        [0.0; 3],
        13,
        LayerMask::ALL,
    )
    .expect("room for the region");
    sim.add_fixed(
        &ColliderShape::Cuboid {
            half_extents: [SLAB_HALF_THICKNESS, 2.0, 2.0],
        },
        [0.0; 3],
        [0.0; 3],
        0.8,
        LayerMask::ALL,
    )
    .expect("room for the wall");
    let body = spawn(&mut sim, &BALL, [-8.0, 0.0, 0.0], 0.0);
    sim.set_linear_velocity(body, [600.0, 0.0, 0.0]);

    let mut crossings = Vec::new();
    let mut seen = Vec::new();
    for _ in 0..60 {
        sim.step(TICK);
        sim.drain_sensor_crossings_into(&mut crossings);
        seen.extend(crossings.iter().map(|c| (c.tag, c.entered)));
    }
    assert_eq!(seen, [(13, true)], "it went in once and stopped there");
    let stopped = sim.body_pose(body).expect("live").0[0];
    assert!(
        stopped < -SLAB_HALF_THICKNESS,
        "and it stopped at the wall inside the region, at x = {stopped}"
    );
}

// A driven body arrives exactly where it was sent, so the only way it can
// avoid going through something is for that something to give way.
#[test]
fn a_fast_platform_pushes_a_body_rather_than_passing_through_it() {
    let run = |ccd: bool| {
        let mut sim = Simulation::new(config(ccd), 2);
        let platform = sim
            .add_kinematic(
                &ColliderShape::Cuboid {
                    half_extents: [0.05, 2.0, 2.0],
                },
                [-9.0, 0.0, 0.0],
                [0.0; 3],
                0.8,
                LayerMask::ALL,
            )
            .expect("room for the platform");
        let body = spawn(&mut sim, &BALL, [0.0, 0.0, 0.0], 0.0);
        // Three units a tick, against a face a tenth of a unit thick: the
        // platform is never sampled anywhere near the ball.
        let mut x = -9.0f32;
        for _ in 0..6 {
            x += 3.0;
            sim.set_kinematic_translation(platform, [x, 0.0, 0.0]);
            sim.step(TICK);
        }
        assert_eq!(sim.ccd_overflows(), 0, "the sweep was declined");
        (
            sim.body_pose(platform).expect("live").0[0],
            sim.body_pose(body).expect("live").0[0],
        )
    };

    let (platform, body) = run(false);
    assert!(
        body < platform,
        "without the sweep the platform leaves the body behind it: {body} vs {platform}"
    );

    let (platform, body) = run(true);
    assert!(
        body > platform,
        "with it the body is ahead of the face that pushed it: {body} vs {platform}"
    );
    assert!(
        body - platform < 0.5,
        "and pushed rather than thrown: {body} vs {platform}"
    );
}

// A joint is the stiffest thing in the solve, so a position change made
// outside it is the change most able to make one explode. An arm held on a
// one-unit link and launched sideways at fifty units a second crosses eight
// tenths of a unit per tick, against a link that wants it back and a wall a
// tenth of a unit thick that the tick would otherwise step straight over.
#[test]
fn a_jointed_body_stopped_by_the_sweep_does_not_fight_the_joint() {
    // Thicker than the ball above, so the link has something to hold.
    const ARM: ColliderShape = ColliderShape::Cuboid {
        half_extents: [0.15, 0.15, 0.15],
    };
    let run = |ccd: bool| {
        let mut sim = Simulation::new(config(ccd), 3);
        sim.add_fixed(
            &ColliderShape::Cuboid {
                half_extents: [SLAB_HALF_THICKNESS, 2.0, 2.0],
            },
            [0.7, -1.0, 0.0],
            [0.0; 3],
            0.8,
            LayerMask::ALL,
        )
        .expect("room for the wall");
        let post = sim
            .add_fixed(
                &ColliderShape::Ball { radius: 0.05 },
                [0.0; 3],
                [0.0; 3],
                0.5,
                LayerMask::ALL,
            )
            .expect("room for the post");
        let arm = sim
            .add_dynamic(
                &ARM,
                [0.0, -1.0, 0.0],
                [0.0; 3],
                params(1.0),
                LayerMask::ALL,
            )
            .expect("room for the arm");
        assert!(
            sim.add_joint(post, arm, [0.0; 3], [0.0, 1.0, 0.0], JointSpec::Fixed),
            "the joint has to be made"
        );
        sim.set_linear_velocity(arm, [50.0, 0.0, 0.0]);

        let (mut furthest, mut stretched, mut swept) = (f32::NEG_INFINITY, 0.0f32, 0usize);
        for tick in 0..300 {
            sim.step(TICK);
            swept += sim.swept_body_count();
            let at = sim.body_pose(arm).expect("live").0;
            let v = sim.linear_velocity(arm).expect("live");
            let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!(
                speed < 60.0,
                "the joint gained speed rather than bleeding it: {speed} on tick {tick}"
            );
            furthest = furthest.max(at[0]);
            stretched = stretched.max((at[0] * at[0] + at[1] * at[1] + at[2] * at[2]).sqrt());
        }
        assert_eq!(sim.ccd_overflows(), 0, "the sweep was declined");
        (furthest, stretched, swept)
    };

    let (furthest, _, swept) = run(false);
    assert_eq!(swept, 0, "the sweep is meant to be off");
    assert!(
        furthest > 1.0,
        "without the sweep the arm has to cross the wall, reaching {furthest}"
    );

    let (furthest, stretched, swept) = run(true);
    assert!(swept > 0, "the arm has to have been fast enough to sweep");
    // The wall's near face is at 0.65 and the arm reaches 0.15 in front of
    // its centre, so it is held with its centre at 0.5.
    assert!(
        furthest < 0.52,
        "with it the arm is held at the wall, and reached {furthest}"
    );
    assert!(
        stretched < 1.2,
        "the link held rather than being torn to {stretched}"
    );
}

// The character resolves its own move by sweeping every tick, so it must not
// be reaching the stage that sweeps for everything else.
#[test]
fn a_walking_character_never_arms_the_sweep() {
    let mut sim = Simulation::new(SimConfig::default(), 2);
    sim.add_fixed(&FLOOR, [0.0, -1.0, 0.0], [0.0; 3], 0.8, LayerMask::ALL)
        .expect("room for the floor");
    let capsule = sim
        .add_character(0.6, 0.3, [0.0, 0.9, 0.0], LayerMask::ALL)
        .expect("room for the capsule");
    let shape = Simulation::character_shape(0.6, 0.3);

    let mut center = [0.0f32, 0.9, 0.0];
    for tick in 0..300 {
        let moved = sim.move_character(
            &shape,
            &CharacterMoveInput {
                // Eight units a second, which is a sprint.
                center,
                desired: [8.0 * TICK, 0.0, 0.0],
                dt: TICK,
                exclude: capsule,
                mask: LayerMask::ALL,
            },
        );
        for (axis, delta) in moved.translation.iter().enumerate() {
            center[axis] += delta;
        }
        sim.set_kinematic_translation(capsule, center);
        sim.step(TICK);
        assert_eq!(
            sim.swept_body_count(),
            0,
            "the character took the expensive path on tick {tick}"
        );
    }
    assert!(
        center[0] > 30.0,
        "the character has to have walked: {center:?}"
    );
}

// The sweep reads the same sorted order every other stage does, resolves
// impacts by slot, and mints nothing keyed by a hash, so a scene that leans on
// it has to land bit-identically twice.
#[test]
fn a_scene_full_of_fast_bodies_steps_identically_twice() {
    let run = || {
        let mut sim = Simulation::new(config(true), 34);
        sim.add_fixed(&SLAB, [0.0; 3], [0.0; 3], 0.8, LayerMask::ALL)
            .expect("room for the slab");
        sim.add_sensor(
            &ColliderShape::Cuboid {
                half_extents: [8.0, 1.0, 8.0],
            },
            [0.0, 2.0, 0.0],
            [0.0; 3],
            3,
            LayerMask::ALL,
        )
        .expect("room for the region");
        let bodies: Vec<BodyHandle> = (0..32)
            .map(|i| {
                let x = (i % 8) as f32 * 0.7 - 2.45;
                let z = (i / 8) as f32 * 0.7 - 1.05;
                let body = spawn(&mut sim, &BALL, [x, 6.0 + (i % 3) as f32, z], 1.0);
                sim.set_linear_velocity(body, [0.0, -(200.0 + i as f32 * 37.0), 0.0]);
                body
            })
            .collect();
        let mut crossings = Vec::new();
        let mut recorded = Vec::new();
        for _ in 0..120 {
            sim.step(TICK);
            sim.drain_sensor_crossings_into(&mut crossings);
            recorded.extend(crossings.iter().map(|c| (c.tag, c.entered)));
        }
        assert_eq!(sim.ccd_overflows(), 0, "the sweep was declined");
        let poses: Vec<[u32; 3]> = bodies
            .iter()
            .map(|&h| {
                let p = sim.body_pose(h).expect("live").0;
                [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]
            })
            .collect();
        (poses, recorded)
    };
    let first = run();
    assert!(!first.1.is_empty(), "the fixture has to cross the region");
    assert_eq!(first, run(), "the sweep is no longer deterministic");
}

// Turning the stage off has to be the only thing turning it off does: the same
// ordinary world settles the same way either way, down to the bit.
#[test]
fn disabling_the_sweep_leaves_an_ordinary_world_untouched() {
    let run = |ccd: bool| {
        let mut sim = Simulation::new(config(ccd), 5);
        sim.add_fixed(&FLOOR, [0.0, -1.0, 0.0], [0.0; 3], 0.8, LayerMask::ALL)
            .expect("room for the floor");
        let cube = ColliderShape::Cuboid {
            half_extents: [0.5, 0.5, 0.5],
        };
        let handles: Vec<BodyHandle> = (0..4)
            .map(|level| spawn(&mut sim, &cube, [0.0, 0.5 + level as f32 * 1.1, 0.0], 1.0))
            .collect();
        for _ in 0..600 {
            sim.step(TICK);
        }
        handles
            .iter()
            .map(|&h| sim.body_pose(h).expect("live").0)
            .collect::<Vec<_>>()
    };
    assert_eq!(run(true), run(false));
}
