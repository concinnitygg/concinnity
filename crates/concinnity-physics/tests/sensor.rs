//! What a region has to do, and what it must never do.
//!
//! A sensor is judged on two things at once. It has to report the boundary and
//! only the boundary: once on the way in, once on the way out, and nothing at
//! all for the seconds in between, or a caller wiring a trigger to it fires a
//! door open sixty times a second. And it has to be invisible to everything
//! else: nothing rests on it, no ray stops at it, no character is turned by
//! it, and a world with one in it moves exactly as the same world without.
//!
//! The tests below are written against a falling body wherever they can be,
//! because that is the case where both halves are checked at once: the ball
//! reports its crossings, and where it lands says whether the region held it
//! up.

use concinnity_physics::{
    BodyHandle, CharacterMoveInput, ColliderShape, DynamicParams, LayerMask, SensorCrossing,
    ShapeCast, SimConfig, Simulation,
};

const TICK: f32 = 1.0 / 60.0;
const BALL: ColliderShape = ColliderShape::Ball { radius: 0.25 };
const REGION: ColliderShape = ColliderShape::Cuboid {
    half_extents: [1.0, 1.0, 1.0],
};
const CAPSULE: ColliderShape = ColliderShape::Capsule {
    half_height: 0.6,
    radius: 0.3,
};

fn params() -> DynamicParams {
    DynamicParams {
        mass: 1.0,
        friction: 0.5,
        restitution: 0.0,
        gravity_scale: 1.0,
        linear_damping: 0.0,
    }
}

fn sim(capacity: usize) -> Simulation {
    Simulation::new(SimConfig::default(), capacity)
}

/// A floor whose top surface is exactly `y = 0`.
fn add_floor(sim: &mut Simulation) -> BodyHandle {
    sim.add_fixed(
        &ColliderShape::Cuboid {
            half_extents: [20.0, 1.0, 20.0],
        },
        [0.0, -1.0, 0.0],
        [0.0; 3],
        0.8,
        LayerMask::ALL,
    )
    .expect("room for the floor")
}

/// Step `ticks` times, collecting every crossing recorded along the way.
fn run(sim: &mut Simulation, ticks: usize) -> Vec<SensorCrossing> {
    let mut collected = Vec::new();
    let mut drained = Vec::new();
    for _ in 0..ticks {
        sim.step(TICK);
        sim.drain_sensor_crossings_into(&mut drained);
        collected.extend_from_slice(&drained);
    }
    assert_eq!(sim.sensor_overflows(), 0, "a crossing went unreported");
    collected
}

#[test]
fn a_body_falling_through_a_region_reports_going_in_and_coming_out() {
    let mut sim = sim(2);
    sim.add_sensor(&REGION, [0.0, 2.0, 0.0], [0.0; 3], 7, LayerMask::ALL)
        .expect("room for the region");
    let ball = sim
        .add_dynamic(&BALL, [0.0, 6.0, 0.0], [0.0; 3], params(), LayerMask::ALL)
        .expect("room for the ball");

    let crossings = run(&mut sim, 300);
    assert_eq!(crossings.len(), 2, "{crossings:?}");
    assert!(crossings.iter().all(|c| c.tag == 7));
    assert!(crossings.iter().all(|c| c.other == Some(ball)));
    assert!(crossings[0].entered, "in first");
    assert!(!crossings[1].entered, "out after");
}

// The whole point of tracking the boundary: a body that came to rest inside a
// region reports once, not once a tick for as long as it sits there.
#[test]
fn a_body_resting_inside_a_region_reports_once() {
    let mut sim = sim(3);
    add_floor(&mut sim);
    // The region covers where the ball comes to rest.
    sim.add_sensor(&REGION, [0.0, 1.0, 0.0], [0.0; 3], 3, LayerMask::ALL)
        .expect("room for the region");
    sim.add_dynamic(&BALL, [0.0, 3.0, 0.0], [0.0; 3], params(), LayerMask::ALL)
        .expect("room for the ball");

    let crossings = run(&mut sim, 600);
    assert_eq!(crossings.len(), 1, "{crossings:?}");
    assert!(crossings[0].entered);
    assert_eq!(sim.sensor_overlap_count(), 1, "still being tracked");
}

// The character capsule is position-driven, and a controller that could not
// set a trigger off would leave every authored volume in a world inert.
#[test]
fn a_position_driven_capsule_crosses_a_region() {
    let mut sim = sim(2);
    sim.add_sensor(&REGION, [5.0, 1.0, 0.0], [0.0; 3], 9, LayerMask::ALL)
        .expect("room for the region");
    let capsule = sim
        .add_kinematic(&CAPSULE, [0.0, 1.0, 0.0], [0.0; 3], 0.8, LayerMask::ALL)
        .expect("room for the capsule");

    let mut crossings = Vec::new();
    sim.set_kinematic_translation(capsule, [5.0, 1.0, 0.0]);
    sim.step(TICK);
    sim.step(TICK);
    sim.drain_sensor_crossings_into(&mut crossings);
    assert_eq!(crossings.len(), 1, "{crossings:?}");
    assert!(crossings[0].entered && crossings[0].other == Some(capsule));
    assert_eq!(crossings[0].tag, 9);

    sim.set_kinematic_translation(capsule, [0.0, 1.0, 0.0]);
    sim.step(TICK);
    sim.step(TICK);
    sim.drain_sensor_crossings_into(&mut crossings);
    assert_eq!(crossings.len(), 1, "{crossings:?}");
    assert!(!crossings[0].entered);
}

// A wall standing in a region has not crossed anything: it was there when the
// world was built and it will be there when the world ends.
#[test]
fn immovable_geometry_never_crosses_a_region() {
    let mut sim = sim(3);
    sim.add_sensor(&REGION, [0.0, 2.0, 0.0], [0.0; 3], 1, LayerMask::ALL)
        .expect("room for the region");
    sim.add_fixed(&BALL, [0.0, 2.0, 0.0], [0.0; 3], 0.5, LayerMask::ALL)
        .expect("room for the wall");
    sim.add_heightfield(
        5,
        5,
        vec![2.0; 25],
        [20.0, 1.0, 20.0],
        [0.0; 3],
        LayerMask::ALL,
    )
    .expect("room for the terrain");

    assert!(run(&mut sim, 120).is_empty());
    assert_eq!(sim.sensor_overlap_count(), 0);
}

#[test]
fn two_regions_sharing_space_each_record_the_other() {
    let mut sim = sim(2);
    sim.add_sensor(&REGION, [0.0, 2.0, 0.0], [0.0; 3], 11, LayerMask::ALL)
        .expect("room for a region");
    sim.add_sensor(&REGION, [0.5, 2.0, 0.0], [0.0; 3], 22, LayerMask::ALL)
        .expect("room for a region");

    let crossings = run(&mut sim, 10);
    assert_eq!(crossings.len(), 2, "{crossings:?}");
    assert!(crossings.iter().all(|c| c.entered));
    let mut tags: Vec<u64> = crossings.iter().map(|c| c.tag).collect();
    tags.sort_unstable();
    assert_eq!(tags, [11, 22]);
}

// A caller reacting to an exit has to be able to tell "it walked out" from
// "it stopped existing", and the handle it is given must never name whatever
// took the slot next.
#[test]
fn a_body_removed_while_inside_leaves_without_naming_itself() {
    let mut sim = sim(2);
    sim.add_sensor(&REGION, [0.0, 2.0, 0.0], [0.0; 3], 5, LayerMask::ALL)
        .expect("room for the region");
    let ball = sim
        .add_dynamic(
            &BALL,
            [0.0, 2.0, 0.0],
            [0.0; 3],
            DynamicParams {
                gravity_scale: 0.0,
                ..params()
            },
            LayerMask::ALL,
        )
        .expect("room for the ball");

    let mut crossings = Vec::new();
    sim.step(TICK);
    sim.drain_sensor_crossings_into(&mut crossings);
    assert_eq!(crossings.len(), 1);
    assert!(crossings[0].entered && crossings[0].other == Some(ball));

    assert!(sim.remove_body(ball));
    sim.step(TICK);
    sim.drain_sensor_crossings_into(&mut crossings);
    assert_eq!(crossings.len(), 1, "{crossings:?}");
    assert!(!crossings[0].entered);
    assert_eq!(crossings[0].other, None, "it left the simulation");
    assert_eq!(crossings[0].tag, 5);
}

// A region resists nothing, and the way to say so is that the same world with
// and without one runs to the same numbers.
#[test]
fn a_region_changes_nothing_about_how_the_world_moves() {
    let drop = |region: bool| {
        let mut sim = sim(3);
        add_floor(&mut sim);
        if region {
            sim.add_sensor(
                &ColliderShape::Cuboid {
                    half_extents: [4.0, 4.0, 4.0],
                },
                [0.0, 2.0, 0.0],
                [0.0; 3],
                1,
                LayerMask::ALL,
            )
            .expect("room for the region");
        }
        let ball = sim
            .add_dynamic(&BALL, [0.0, 6.0, 0.0], [0.0; 3], params(), LayerMask::ALL)
            .expect("room for the ball");
        let mut path = Vec::new();
        for _ in 0..240 {
            sim.step(TICK);
            path.push(sim.body_pose(ball).expect("live").0.map(f32::to_bits));
        }
        path
    };
    let without = drop(false);
    assert_eq!(drop(true), without, "the region deflected the ball");
    assert!(
        (f32::from_bits(without[239][1]) - 0.25).abs() < 0.02,
        "the ball rests on the floor at {}",
        f32::from_bits(without[239][1])
    );
}

#[test]
fn a_ray_passes_straight_through_a_region() {
    let mut sim = sim(2);
    add_floor(&mut sim);
    sim.add_sensor(&REGION, [0.0, 2.0, 0.0], [0.0; 3], 1, LayerMask::ALL)
        .expect("room for the region");
    sim.step(TICK);

    let hit = sim
        .raycast(
            [0.0, 5.0, 0.0],
            [0.0, -1.0, 0.0],
            10.0,
            None,
            LayerMask::ALL,
        )
        .expect("the ray reaches the floor");
    assert!(hit.point[1].abs() < 1.0e-3, "{:?}", hit.point);
    assert!((hit.distance - 5.0).abs() < 1.0e-3, "{}", hit.distance);
}

#[test]
fn a_swept_shape_passes_straight_through_a_region() {
    let mut sim = sim(2);
    add_floor(&mut sim);
    sim.add_sensor(&REGION, [0.0, 2.0, 0.0], [0.0; 3], 1, LayerMask::ALL)
        .expect("room for the region");
    sim.step(TICK);

    let hit = sim
        .shape_cast(&ShapeCast::new(BALL, [0.0, 5.0, 0.0], [0.0, -8.0, 0.0]))
        .expect("the sweep reaches the floor");
    let landed = 5.0 - hit.toi * 8.0;
    assert!((landed - 0.25).abs() < 0.01, "landed at {landed}");
}

#[test]
fn a_character_walks_straight_through_a_region() {
    let mut sim = sim(3);
    add_floor(&mut sim);
    sim.add_sensor(
        &ColliderShape::Cuboid {
            half_extents: [2.0, 2.0, 0.5],
        },
        [0.0, 1.0, 1.0],
        [0.0; 3],
        4,
        LayerMask::ALL,
    )
    .expect("room for the region");
    let center = [0.0, 0.9, 0.0];
    let capsule = sim
        .add_kinematic(&CAPSULE, center, [0.0; 3], 0.8, LayerMask::ALL)
        .expect("room for the capsule");

    let shape = Simulation::character_shape(0.6, 0.3);
    let moved = sim.move_character(
        &shape,
        &CharacterMoveInput {
            center,
            desired: [0.0, -0.01, 2.0],
            dt: TICK,
            exclude: capsule,
            mask: LayerMask::ALL,
        },
    );
    assert!(
        (moved.translation[2] - 2.0).abs() < 0.01,
        "the region stopped the walk at {}",
        moved.translation[2]
    );
    assert!(moved.grounded);
}

#[test]
fn a_region_only_sees_the_layers_it_interacts_with() {
    let first = LayerMask {
        memberships: 0b01,
        filter: 0b01,
    };
    let second = LayerMask {
        memberships: 0b10,
        filter: 0b10,
    };
    let mut sim = sim(3);
    sim.add_sensor(&REGION, [0.0, 2.0, 0.0], [0.0; 3], 1, first)
        .expect("room for the region");
    sim.add_dynamic(
        &BALL,
        [0.0, 2.0, 0.0],
        [0.0; 3],
        DynamicParams {
            gravity_scale: 0.0,
            ..params()
        },
        second,
    )
    .expect("room for the ball");
    assert!(run(&mut sim, 60).is_empty(), "the layers do not meet");

    let seen = sim
        .add_dynamic(
            &BALL,
            [0.0, 2.0, 0.0],
            [0.0; 3],
            DynamicParams {
                gravity_scale: 0.0,
                ..params()
            },
            first,
        )
        .expect("room for the ball");
    let crossings = run(&mut sim, 60);
    assert_eq!(crossings.len(), 1, "{crossings:?}");
    assert_eq!(crossings[0].other, Some(seen));
}

// A caller running several fixed ticks per frame drains once at the end of
// them, so the queue has to hold every crossing rather than the last step's.
#[test]
fn crossings_wait_in_the_queue_until_they_are_drained() {
    let mut sim = sim(2);
    sim.add_sensor(&REGION, [0.0, 2.0, 0.0], [0.0; 3], 6, LayerMask::ALL)
        .expect("room for the region");
    sim.add_dynamic(&BALL, [0.0, 6.0, 0.0], [0.0; 3], params(), LayerMask::ALL)
        .expect("room for the ball");

    for _ in 0..300 {
        sim.step(TICK);
    }
    let mut out = Vec::new();
    sim.drain_sensor_crossings_into(&mut out);
    assert_eq!(out.len(), 2, "in and out both waited: {out:?}");
    assert!(out[0].entered && !out[1].entered);
    assert_eq!(sim.sensor_overflows(), 0);
    sim.drain_sensor_crossings_into(&mut out);
    assert!(out.is_empty(), "the drain emptied it");
}

// A caller draining every tick is what the queue is sized for, and the drain
// has to hand the buffers back rather than replacing them.
#[test]
fn draining_crossings_reallocates_neither_side() {
    let mut sim = sim(4);
    sim.add_sensor(&REGION, [0.0, 2.0, 0.0], [0.0; 3], 1, LayerMask::ALL)
        .expect("room for the region");
    sim.add_dynamic(&BALL, [0.0, 6.0, 0.0], [0.0; 3], params(), LayerMask::ALL)
        .expect("room for the ball");

    let mut out = Vec::with_capacity(8);
    let capacity = out.capacity();
    let mut total = 0;
    for _ in 0..300 {
        sim.step(TICK);
        sim.drain_sensor_crossings_into(&mut out);
        total += out.len();
        assert_eq!(out.capacity(), capacity, "the caller's buffer was replaced");
    }
    assert_eq!(total, 2);
    assert_eq!(sim.sensor_overflows(), 0);
}

#[test]
fn two_identical_runs_report_the_same_crossings() {
    let once = || {
        let mut sim = sim(6);
        add_floor(&mut sim);
        for (index, x) in [-3.0f32, 0.0, 3.0].into_iter().enumerate() {
            sim.add_sensor(
                &REGION,
                [x, 2.0, 0.0],
                [0.0; 3],
                index as u64,
                LayerMask::ALL,
            )
            .expect("room for the region");
        }
        for x in [-3.0f32, 0.0] {
            sim.add_dynamic(&BALL, [x, 6.0, 0.0], [0.0; 3], params(), LayerMask::ALL)
                .expect("room for the ball");
        }
        run(&mut sim, 240)
            .into_iter()
            .map(|c| {
                (
                    c.tag,
                    c.other.map(|h| (h.index(), h.generation())),
                    c.entered,
                )
            })
            .collect::<Vec<_>>()
    };
    let first = once();
    assert!(!first.is_empty(), "the scene has to record something");
    assert_eq!(first, once());
}

// A world with more regions overlapping at once than the reservation covers
// is declined and counted, never grown inside a step.
#[test]
fn a_world_past_its_reservation_declines_and_counts() {
    // Four bodies: three regions nested inside each other, and a ball in all
    // of them. Six overlapping pairs against a reservation of four.
    let mut sim = sim(4);
    for tag in 0..3 {
        sim.add_sensor(&REGION, [0.0, 2.0, 0.0], [0.0; 3], tag, LayerMask::ALL)
            .expect("room for the region");
    }
    sim.add_dynamic(
        &BALL,
        [0.0, 2.0, 0.0],
        [0.0; 3],
        DynamicParams {
            gravity_scale: 0.0,
            ..params()
        },
        LayerMask::ALL,
    )
    .expect("room for the ball");

    let mut out = Vec::new();
    sim.step(TICK);
    sim.drain_sensor_crossings_into(&mut out);
    assert_eq!(
        sim.sensor_overlap_count(),
        4,
        "the reservation, and no more"
    );
    assert!(
        sim.sensor_overflows() > 0,
        "the shortfall has to be reported"
    );

    // Whatever it had room for stays steady rather than churning: the pairs
    // it dropped are the same ones every step.
    sim.clear_sensor_overflows();
    sim.step(TICK);
    sim.drain_sensor_crossings_into(&mut out);
    assert!(out.is_empty(), "{out:?}");
    assert!(sim.sensor_overflows() > 0);
    sim.clear_sensor_overflows();
    assert_eq!(sim.sensor_overflows(), 0);
}

#[test]
fn a_full_pool_declines_a_region() {
    let mut sim = sim(1);
    assert!(
        sim.add_sensor(&REGION, [0.0; 3], [0.0; 3], 1, LayerMask::ALL)
            .is_some()
    );
    assert!(
        sim.add_sensor(&REGION, [0.0; 3], [0.0; 3], 2, LayerMask::ALL)
            .is_none()
    );
}
