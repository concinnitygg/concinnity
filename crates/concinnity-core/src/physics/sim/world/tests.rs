// A capsule swept at a floor stops where its lower cap meets it, and the
// hit reports the surface it landed on.
#[test]
fn a_shape_cast_stops_at_the_first_body_in_its_path() {
    let mut sim = Simulation::with_capacity(1);
    sim.add_fixed(
        &ColliderShape::Cuboid {
            half_extents: [10.0, 0.5, 10.0],
        },
        [0.0, -0.5, 0.0],
        [0.0; 3],
        0.8,
        LayerMask::ALL,
    );

    let capsule = ColliderShape::Capsule {
        half_height: 0.6,
        radius: 0.3,
    };
    let hit = sim
        .shape_cast(&ShapeCast::new(capsule, [0.0, 4.0, 0.0], [0.0, -8.0, 0.0]))
        .expect("the floor is down there");

    // The capsule reaches 0.9 below its center, so it stops with its
    // center 0.9 above the floor.
    let landed = 4.0 - hit.toi * 8.0;
    assert!((landed - 0.9).abs() < 0.01, "landed at {landed}");
    assert!(hit.normal[1] > 0.99, "standing on it: {:?}", hit.normal);
}
use super::*;
use crate::physics::sim::query::ShapeCast;
use crate::physics::{CharacterMoveInput, ColliderShape, DynamicParams, LayerMask};
use alloc::vec::Vec;

const TICK: f32 = 1.0 / 60.0;

fn params(restitution: f32, damping: f32) -> DynamicParams {
    DynamicParams {
        mass: 1.0,
        friction: 0.5,
        restitution,
        gravity_scale: 1.0,
        linear_damping: damping,
    }
}

fn floor(sim: &mut Simulation) -> BodyHandle {
    sim.add_fixed(
        &ColliderShape::Cuboid {
            half_extents: [50.0, 1.0, 50.0],
        },
        [0.0, -1.0, 0.0],
        [0.0; 3],
        0.8,
        LayerMask::ALL,
    )
    .expect("room")
}

#[test]
fn a_body_falls_under_gravity() {
    let mut sim = Simulation::with_capacity(1);
    let ball = sim
        .add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 10.0, 0.0],
            [0.0; 3],
            params(0.0, 0.0),
            LayerMask::ALL,
        )
        .expect("room");
    for _ in 0..60 {
        sim.step(TICK);
    }
    let (pos, _) = sim.body_pose(ball).expect("live");
    // One second of free fall at g = 20 covers about ten units.
    assert!((pos[1] - (10.0 - 10.0)).abs() < 0.4, "y = {}", pos[1]);
    assert!(sim.linear_velocity(ball).expect("live")[1] < -19.0);
}

#[test]
fn a_zero_or_negative_step_changes_nothing() {
    let mut sim = Simulation::with_capacity(1);
    let ball = sim
        .add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 10.0, 0.0],
            [0.0; 3],
            params(0.0, 0.0),
            LayerMask::ALL,
        )
        .expect("room");
    sim.step(0.0);
    sim.step(-1.0);
    assert_eq!(sim.body_pose(ball).expect("live").0, [0.0, 10.0, 0.0]);
}

#[test]
fn a_full_pool_declines_rather_than_growing() {
    let mut sim = Simulation::with_capacity(1);
    assert!(
        sim.add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 1.0, 0.0],
            [0.0; 3],
            params(0.0, 0.0),
            LayerMask::ALL
        )
        .is_some()
    );
    assert!(
        sim.add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 3.0, 0.0],
            [0.0; 3],
            params(0.0, 0.0),
            LayerMask::ALL
        )
        .is_none()
    );
    assert_eq!(sim.body_count(), 1);
    assert_eq!(sim.capacity(), 1);
    assert!(sim.reserved_bytes() > 0);
}

#[test]
fn a_removed_bodys_handle_stops_naming_anything() {
    let mut sim = Simulation::with_capacity(2);
    let ball = sim
        .add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 1.0, 0.0],
            [0.0; 3],
            params(0.0, 0.0),
            LayerMask::ALL,
        )
        .expect("room");
    assert!(sim.remove_body(ball));
    assert!(!sim.remove_body(ball));
    assert!(sim.body_pose(ball).is_none());
    assert_eq!(sim.body_count(), 0);
}

#[test]
fn a_body_lands_on_the_floor_and_stays_on_it() {
    let mut sim = Simulation::with_capacity(2);
    floor(&mut sim);
    let ball = sim
        .add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 6.0, 0.0],
            [0.0; 3],
            params(0.0, 0.0),
            LayerMask::ALL,
        )
        .expect("room");
    for _ in 0..240 {
        sim.step(TICK);
    }
    let (pos, _) = sim.body_pose(ball).expect("live");
    assert!((pos[1] - 0.5).abs() < 0.02, "y = {}", pos[1]);
    assert!(sim.contact_count() > 0);
}

#[test]
fn layers_that_do_not_interact_pass_through_each_other() {
    let mut sim = Simulation::with_capacity(2);
    sim.add_fixed(
        &ColliderShape::Cuboid {
            half_extents: [50.0, 1.0, 50.0],
        },
        [0.0, -1.0, 0.0],
        [0.0; 3],
        0.8,
        LayerMask {
            memberships: 0b01,
            filter: 0b01,
        },
    );
    let ball = sim
        .add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 4.0, 0.0],
            [0.0; 3],
            params(0.0, 0.0),
            LayerMask {
                memberships: 0b10,
                filter: 0b10,
            },
        )
        .expect("room");
    for _ in 0..120 {
        sim.step(TICK);
    }
    assert!(sim.body_pose(ball).expect("live").0[1] < -2.0);
}

#[test]
fn an_impulse_moves_a_body_and_wakes_it() {
    let mut sim = Simulation::with_capacity(2);
    floor(&mut sim);
    let ball = sim
        .add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 0.5, 0.0],
            [0.0; 3],
            params(0.0, 0.5),
            LayerMask::ALL,
        )
        .expect("room");
    for _ in 0..120 {
        sim.step(TICK);
    }
    assert_eq!(sim.is_sleeping(ball), Some(true));
    sim.apply_impulse(ball, [0.0, 8.0, 0.0]);
    assert_eq!(sim.is_sleeping(ball), Some(false));
    sim.step(TICK);
    assert!(sim.body_pose(ball).expect("live").0[1] > 0.55);
}

// A body put to sleep in mid-air hangs there through any number of steps,
// and falls the moment it is woken.
#[test]
fn a_body_put_to_sleep_hangs_until_it_is_woken() {
    let mut sim = Simulation::with_capacity(1);
    let ball = sim
        .add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 10.0, 0.0],
            [0.0; 3],
            params(0.0, 0.0),
            LayerMask::ALL,
        )
        .expect("room");
    assert!(sim.sleep_body(ball));
    for _ in 0..120 {
        sim.step(TICK);
    }
    assert_eq!(sim.is_sleeping(ball), Some(true));
    assert_eq!(sim.body_pose(ball).expect("live").0, [0.0, 10.0, 0.0]);

    assert!(sim.wake_body(ball));
    for _ in 0..30 {
        sim.step(TICK);
    }
    assert!(sim.body_pose(ball).expect("live").0[1] < 9.0, "it fell");
}

// Only a body the solver moves can sleep: a wall is not put to sleep, and
// a removed body is not found.
#[test]
fn only_a_live_dynamic_body_can_be_put_to_sleep() {
    let mut sim = Simulation::with_capacity(2);
    let ground = floor(&mut sim);
    assert!(!sim.sleep_body(ground));
    let ball = sim
        .add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 4.0, 0.0],
            [0.0; 3],
            params(0.0, 0.0),
            LayerMask::ALL,
        )
        .expect("room");
    assert!(sim.remove_body(ball));
    assert!(!sim.sleep_body(ball));
    assert!(!sim.wake_body(ball));
}

// A sleeper is not out of the world: a body falling onto it wakes it, and
// the two fall on together.
#[test]
fn a_sleeping_body_struck_by_a_falling_one_wakes() {
    let mut sim = Simulation::with_capacity(2);
    let sleeper = sim
        .add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 4.0, 0.0],
            [0.0; 3],
            params(0.0, 0.0),
            LayerMask::ALL,
        )
        .expect("room");
    sim.add_dynamic(
        &ColliderShape::Ball { radius: 0.5 },
        [0.0, 6.0, 0.0],
        [0.0; 3],
        params(0.0, 0.0),
        LayerMask::ALL,
    )
    .expect("room");
    assert!(sim.sleep_body(sleeper));
    for _ in 0..90 {
        sim.step(TICK);
    }
    assert_eq!(sim.is_sleeping(sleeper), Some(false), "it was struck");
    assert!(sim.body_pose(sleeper).expect("live").0[1] < 4.0, "it fell");
}

// A teleport is a placement, not a push: the body arrives exactly where it
// was put, at rest, and falls again from there.
#[test]
fn a_teleport_places_a_body_at_rest() {
    let mut sim = Simulation::with_capacity(1);
    let ball = sim
        .add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 10.0, 0.0],
            [0.0; 3],
            params(0.0, 0.0),
            LayerMask::ALL,
        )
        .expect("room");
    for _ in 0..30 {
        sim.step(TICK);
    }
    assert!(
        sim.linear_velocity(ball).expect("live")[1] < -1.0,
        "falling"
    );

    assert!(sim.teleport_body(ball, [4.0, 20.0, -2.0], [0.0, 90.0, 0.0]));
    assert_eq!(sim.body_pose(ball).expect("live").0, [4.0, 20.0, -2.0]);
    assert_eq!(sim.linear_velocity(ball), Some([0.0; 3]));
    assert_eq!(sim.angular_velocity(ball), Some([0.0; 3]));
    let yaw = sim.body_pose(ball).expect("live").1[1];
    assert!(
        (yaw - 90.0).abs() < 0.01,
        "the rotation went with it: {yaw}"
    );

    sim.step(TICK);
    let dropped = 20.0 - sim.body_pose(ball).expect("live").0[1];
    assert!(
        dropped > 0.0 && dropped < 0.02,
        "one tick of fall: {dropped}"
    );
}

// The body a teleport disturbs is not only the one that moved: whatever
// was resting on it where it stood is woken to fall.
#[test]
fn a_teleport_wakes_the_body_and_what_leaned_on_it() {
    let mut sim = Simulation::with_capacity(2);
    let ground = floor(&mut sim);
    let ball = sim
        .add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 0.5, 0.0],
            [0.0; 3],
            params(0.0, 0.5),
            LayerMask::ALL,
        )
        .expect("room");
    for _ in 0..120 {
        sim.step(TICK);
    }
    assert_eq!(sim.is_sleeping(ball), Some(true), "settled first");

    // The floor is pulled out from under it.
    assert!(sim.teleport_body(ground, [0.0, -20.0, 0.0], [0.0; 3]));
    assert_eq!(sim.is_sleeping(ball), Some(false), "the sleeper was woken");
    for _ in 0..30 {
        sim.step(TICK);
    }
    assert!(
        sim.body_pose(ball).expect("live").0[1] < 0.0,
        "and fell through where the floor used to be"
    );
}

// Terrain is the world rather than something in it, and a handle naming
// nothing names nothing: neither can be teleported.
#[test]
fn a_teleport_declines_terrain_and_a_dead_handle() {
    let mut sim = Simulation::with_capacity(2);
    let ball = sim
        .add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 5.0, 0.0],
            [0.0; 3],
            params(0.0, 0.0),
            LayerMask::ALL,
        )
        .expect("room");
    let ground = sim
        .add_heightfield(
            2,
            2,
            Vec::from([0.0; 4]),
            [1.0; 3],
            [0.0; 3],
            LayerMask::ALL,
        )
        .expect("room");
    assert!(!sim.teleport_body(ground, [0.0, 5.0, 0.0], [0.0; 3]));

    sim.remove_body(ball);
    assert!(!sim.teleport_body(ball, [0.0, 5.0, 0.0], [0.0; 3]));
}

#[test]
fn velocity_can_be_set_and_read_back() {
    let mut sim = Simulation::with_capacity(1);
    let ball = sim
        .add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 5.0, 0.0],
            [0.0; 3],
            params(0.0, 0.0),
            LayerMask::ALL,
        )
        .expect("room");
    sim.set_linear_velocity(ball, [1.0, 0.0, 0.0]);
    sim.set_angular_velocity(ball, [0.0, 2.0, 0.0]);
    assert_eq!(sim.linear_velocity(ball), Some([1.0, 0.0, 0.0]));
    assert_eq!(sim.angular_velocity(ball), Some([0.0, 2.0, 0.0]));
    assert!(sim.mass(ball).expect("live") > 0.0);
}

// The same world stepped twice must land on the same bits, or nothing
// downstream can be reproduced.
#[test]
fn two_identical_runs_agree_bit_for_bit() {
    let run = || {
        let mut sim = Simulation::with_capacity(16);
        floor(&mut sim);
        let mut handles = Vec::new();
        for i in 0..12 {
            handles.push(
                sim.add_dynamic(
                    &ColliderShape::Cuboid {
                        half_extents: [0.4, 0.4, 0.4],
                    },
                    [(i % 3) as f32 * 0.9, 1.0 + (i / 3) as f32 * 0.9, 0.0],
                    [0.0, i as f32 * 7.0, 0.0],
                    params(0.3, 0.0),
                    LayerMask::ALL,
                )
                .expect("room"),
            );
        }
        for _ in 0..90 {
            sim.step(TICK);
        }
        handles
            .iter()
            .map(|&h| {
                let (p, r) = sim.body_pose(h).expect("live");
                (
                    [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()],
                    [r[0].to_bits(), r[1].to_bits(), r[2].to_bits()],
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(run(), run());
}

#[test]
fn a_ray_finds_the_nearest_body_along_it() {
    let mut sim = Simulation::with_capacity(4);
    for (index, z) in [2.0f32, 5.0, 9.0].into_iter().enumerate() {
        sim.add_fixed(
            &[
                ColliderShape::Ball { radius: 0.5 },
                ColliderShape::Cuboid {
                    half_extents: [0.5, 0.5, 0.5],
                },
                ColliderShape::Capsule {
                    half_height: 0.5,
                    radius: 0.25,
                },
            ][index],
            [0.0, 0.0, z],
            [0.0; 3],
            0.5,
            LayerMask::ALL,
        )
        .expect("room");
    }
    let hit = sim
        .raycast(
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 1.0],
            100.0,
            None,
            LayerMask::ALL,
        )
        .expect("a hit");
    assert!((hit.distance - 6.5).abs() < 1.0e-4, "{hit:?}");
    assert!((hit.normal[2] + 1.0).abs() < 1.0e-4, "{hit:?}");
    assert!((hit.point[2] - 1.5).abs() < 1.0e-4, "{hit:?}");
}

#[test]
fn a_ray_answers_the_same_way_from_either_end_of_the_scene() {
    let mut sim = Simulation::with_capacity(4);
    for z in [2.0f32, 5.0, 9.0] {
        sim.add_fixed(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 0.0, z],
            [0.0; 3],
            0.5,
            LayerMask::ALL,
        )
        .expect("room");
    }
    sim.step(TICK);
    let forward = sim
        .raycast(
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 1.0],
            100.0,
            None,
            LayerMask::ALL,
        )
        .expect("a hit");
    let backward = sim
        .raycast(
            [0.0, 0.0, 20.0],
            [0.0, 0.0, -1.0],
            100.0,
            None,
            LayerMask::ALL,
        )
        .expect("a hit");
    assert!((forward.distance - 6.5).abs() < 1.0e-4, "{forward:?}");
    assert!((backward.distance - 10.5).abs() < 1.0e-4, "{backward:?}");
}

#[test]
fn a_ray_skips_layers_it_does_not_interact_with() {
    let mut sim = Simulation::with_capacity(2);
    let near = LayerMask {
        memberships: 0b01,
        filter: 0b11,
    };
    let far = LayerMask {
        memberships: 0b10,
        filter: 0b11,
    };
    sim.add_fixed(
        &ColliderShape::Ball { radius: 0.5 },
        [0.0, 0.0, 0.0],
        [0.0; 3],
        0.5,
        near,
    )
    .expect("room");
    sim.add_fixed(
        &ColliderShape::Ball { radius: 0.5 },
        [0.0, 0.0, 5.0],
        [0.0; 3],
        0.5,
        far,
    )
    .expect("room");

    let only_far = LayerMask {
        memberships: 0b11,
        filter: 0b10,
    };
    let hit = sim
        .raycast([0.0, 0.0, -5.0], [0.0, 0.0, 1.0], 100.0, None, only_far)
        .expect("a hit");
    assert!(
        (hit.distance - 9.5).abs() < 1.0e-4,
        "the near one is hidden"
    );
    // With no layer in common there is nothing to hit at all.
    assert!(
        sim.raycast(
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 1.0],
            100.0,
            None,
            LayerMask {
                memberships: 0b100,
                filter: 0b100,
            },
        )
        .is_none()
    );
}

#[test]
fn a_ray_can_be_told_to_leave_one_body_out() {
    let mut sim = Simulation::with_capacity(2);
    let near = sim
        .add_fixed(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 0.0, 0.0],
            [0.0; 3],
            0.5,
            LayerMask::ALL,
        )
        .expect("room");
    sim.add_fixed(
        &ColliderShape::Ball { radius: 0.5 },
        [0.0, 0.0, 5.0],
        [0.0; 3],
        0.5,
        LayerMask::ALL,
    )
    .expect("room");

    let cast = |exclude| {
        sim.raycast(
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 1.0],
            100.0,
            exclude,
            LayerMask::ALL,
        )
    };
    assert!((cast(None).expect("a hit").distance - 4.5).abs() < 1.0e-4);
    assert!((cast(Some(near)).expect("a hit").distance - 9.5).abs() < 1.0e-4);
}

// A handle to a removed body must not exclude whatever took its slot.
#[test]
fn a_stale_exclusion_does_not_hide_the_slots_new_occupant() {
    let mut sim = Simulation::with_capacity(1);
    let first = sim
        .add_fixed(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 0.0, 0.0],
            [0.0; 3],
            0.5,
            LayerMask::ALL,
        )
        .expect("room");
    assert!(sim.remove_body(first));
    sim.add_fixed(
        &ColliderShape::Ball { radius: 0.5 },
        [0.0, 0.0, 0.0],
        [0.0; 3],
        0.5,
        LayerMask::ALL,
    )
    .expect("the freed slot");
    assert!(
        sim.raycast(
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 1.0],
            100.0,
            Some(first),
            LayerMask::ALL,
        )
        .is_some(),
        "the stale handle names nobody"
    );
}

#[test]
fn a_removed_body_stops_being_hit() {
    let mut sim = Simulation::with_capacity(1);
    let ball = sim
        .add_fixed(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 0.0, 0.0],
            [0.0; 3],
            0.5,
            LayerMask::ALL,
        )
        .expect("room");
    let cast = |sim: &Simulation| {
        sim.raycast(
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 1.0],
            100.0,
            None,
            LayerMask::ALL,
        )
    };
    assert!(cast(&sim).is_some());
    sim.remove_body(ball);
    assert!(cast(&sim).is_none());
}

#[test]
fn a_ray_respects_its_distance_limit_and_needs_a_direction() {
    let mut sim = Simulation::with_capacity(1);
    sim.add_fixed(
        &ColliderShape::Ball { radius: 0.5 },
        [0.0, 0.0, 0.0],
        [0.0; 3],
        0.5,
        LayerMask::ALL,
    )
    .expect("room");
    let cast = |dir: [f32; 3], max| sim.raycast([0.0, 0.0, -5.0], dir, max, None, LayerMask::ALL);
    // The surface is exactly 4.5 away.
    assert!(cast([0.0, 0.0, 1.0], 4.5).is_some());
    assert!(cast([0.0, 0.0, 1.0], 4.4).is_none());
    assert!(cast([0.0, 0.0, 0.0], 100.0).is_none(), "no direction");
    assert!(cast([0.0, 0.0, 1.0], 0.0).is_none(), "no reach");
    assert!(cast([0.0, 0.0, 1.0], -1.0).is_none());
    assert!(cast([f32::NAN, 0.0, 0.0], 100.0).is_none());
    // An unnormalized direction is the same ray.
    assert!((cast([0.0, 0.0, 7.0], 100.0).expect("a hit").distance - 4.5).abs() < 1.0e-4);
}

#[test]
fn a_ray_finds_a_body_added_since_the_last_step() {
    let mut sim = Simulation::with_capacity(2);
    floor(&mut sim);
    sim.step(TICK);
    sim.add_fixed(
        &ColliderShape::Ball { radius: 0.5 },
        [0.0, 5.0, 0.0],
        [0.0; 3],
        0.5,
        LayerMask::ALL,
    )
    .expect("room");
    let hit = sim
        .raycast(
            [0.0, 9.0, 0.0],
            [0.0, -1.0, 0.0],
            100.0,
            None,
            LayerMask::ALL,
        )
        .expect("a hit");
    assert!((hit.distance - 3.5).abs() < 1.0e-4, "{hit:?}");
}

#[test]
fn a_ray_hits_a_position_driven_body() {
    let mut sim = Simulation::with_capacity(1);
    sim.add_kinematic(
        &ColliderShape::Cuboid {
            half_extents: [1.0, 0.25, 1.0],
        },
        [0.0, 0.0, 0.0],
        [0.0; 3],
        0.5,
        LayerMask::ALL,
    )
    .expect("room");
    let hit = sim
        .raycast(
            [0.0, 5.0, 0.0],
            [0.0, -1.0, 0.0],
            100.0,
            None,
            LayerMask::ALL,
        )
        .expect("a hit");
    assert!((hit.distance - 4.75).abs() < 1.0e-4, "{hit:?}");
}

#[test]
fn a_shape_cast_stops_at_the_nearest_body_and_names_it() {
    let mut sim = Simulation::with_capacity(3);
    let ground = floor(&mut sim);
    let ledge = sim
        .add_fixed(
            &ColliderShape::Cuboid {
                half_extents: [2.0, 0.5, 2.0],
            },
            [0.0, 3.0, 0.0],
            [0.0; 3],
            0.5,
            LayerMask::ALL,
        )
        .expect("room");
    let capsule = ColliderShape::Capsule {
        half_height: 0.5,
        radius: 0.25,
    };
    let hit = sim
        .shape_cast(&ShapeCast::new(capsule, [0.0, 9.0, 0.0], [0.0, -12.0, 0.0]))
        .expect("a hit");
    assert_eq!(hit.body, ledge, "the ledge, not the ground under it");
    let landed = 9.0 - hit.toi * 12.0;
    assert!((landed - 4.25).abs() < 1.0e-2, "landed at {landed}");
    assert!(!hit.started_touching);
    assert_ne!(hit.body, ground);
}

#[test]
fn a_shape_cast_that_reaches_nothing_reports_nothing() {
    let mut sim = Simulation::with_capacity(2);
    floor(&mut sim);
    let ball = ColliderShape::Ball { radius: 0.5 };
    assert!(
        sim.shape_cast(&ShapeCast::new(ball, [0.0, 9.0, 0.0], [0.0, -1.0, 0.0]))
            .is_none()
    );
    assert!(
        sim.shape_cast(&ShapeCast::new(ball, [0.0, 9.0, 0.0], [0.0, 5.0, 0.0]))
            .is_none(),
        "away from everything"
    );
}

#[test]
fn a_shape_cast_says_when_it_began_in_contact() {
    let mut sim = Simulation::with_capacity(2);
    let ground = floor(&mut sim);
    let ball = ColliderShape::Ball { radius: 0.5 };
    let hit = sim
        .shape_cast(&ShapeCast::new(ball, [0.0, 0.4, 0.0], [3.0, 0.0, 0.0]))
        .expect("a hit");
    assert_eq!(hit.body, ground);
    assert_eq!(hit.toi, 0.0);
    assert!(hit.started_touching);
    assert!(hit.normal[1] > 0.9, "{hit:?}");
}

#[test]
fn a_shape_cast_honors_its_layer_filter_and_its_exclusion() {
    let mut sim = Simulation::with_capacity(2);
    let near = sim
        .add_fixed(
            &ColliderShape::Cuboid {
                half_extents: [2.0, 0.5, 2.0],
            },
            [0.0, 2.0, 0.0],
            [0.0; 3],
            0.5,
            LayerMask {
                memberships: 0b01,
                filter: 0b11,
            },
        )
        .expect("room");
    let far = sim
        .add_fixed(
            &ColliderShape::Cuboid {
                half_extents: [2.0, 0.5, 2.0],
            },
            [0.0, 0.0, 0.0],
            [0.0; 3],
            0.5,
            LayerMask {
                memberships: 0b10,
                filter: 0b11,
            },
        )
        .expect("room");
    let ball = ColliderShape::Ball { radius: 0.5 };
    let straight = ShapeCast::new(ball, [0.0, 6.0, 0.0], [0.0, -8.0, 0.0]);
    assert_eq!(sim.shape_cast(&straight).expect("a hit").body, near);
    assert_eq!(
        sim.shape_cast(&ShapeCast {
            exclude: Some(near),
            ..straight
        })
        .expect("a hit")
        .body,
        far
    );
    assert_eq!(
        sim.shape_cast(&ShapeCast {
            mask: LayerMask {
                memberships: 0b11,
                filter: 0b10,
            },
            ..straight
        })
        .expect("a hit")
        .body,
        far
    );
}

#[test]
fn a_driven_body_arrives_exactly_where_it_was_sent() {
    let mut sim = Simulation::with_capacity(1);
    let platform = sim
        .add_kinematic(
            &ColliderShape::Cuboid {
                half_extents: [1.0, 0.25, 1.0],
            },
            [0.0, 0.0, 0.0],
            [0.0; 3],
            0.5,
            LayerMask::ALL,
        )
        .expect("room");
    assert!(sim.set_kinematic_translation(platform, [1.5, 2.0, -3.0]));
    sim.step(TICK);
    assert_eq!(sim.body_pose(platform).expect("live").0, [1.5, 2.0, -3.0]);
    // The target is spent, so a body left alone stops where it arrived.
    sim.step(TICK);
    assert_eq!(sim.body_pose(platform).expect("live").0, [1.5, 2.0, -3.0]);
    assert_eq!(sim.linear_velocity(platform), Some([0.0; 3]));
}

#[test]
fn a_driven_body_ignores_gravity_and_impulses() {
    let mut sim = Simulation::with_capacity(1);
    let platform = sim
        .add_kinematic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 5.0, 0.0],
            [0.0; 3],
            0.5,
            LayerMask::ALL,
        )
        .expect("room");
    sim.apply_impulse(platform, [0.0, 100.0, 0.0]);
    for _ in 0..120 {
        sim.step(TICK);
    }
    assert_eq!(sim.body_pose(platform).expect("live").0, [0.0, 5.0, 0.0]);
    assert_eq!(sim.mass(platform), Some(0.0));
    assert_eq!(sim.is_kinematic(platform), Some(true));
}

// A character capsule is a position-driven body: it holds its place under
// gravity and is moved by a target rather than by the solver.
#[test]
fn a_character_capsule_is_a_position_driven_body() {
    let mut sim = Simulation::with_capacity(1);
    let handle = sim
        .add_character(0.6, 0.3, [0.0, 4.0, 0.0], LayerMask::ALL)
        .expect("room in the pool");
    assert_eq!(sim.is_kinematic(handle), Some(true));

    for _ in 0..60 {
        sim.step(TICK);
    }
    let (position, _) = sim.body_pose(handle).expect("a live body");
    assert_eq!(position[1], 4.0, "gravity does not move a driven capsule");

    assert!(sim.set_kinematic_translation(handle, [0.0, 3.0, 0.0]));
    sim.step(TICK);
    let (position, _) = sim.body_pose(handle).expect("a live body");
    assert!((position[1] - 3.0).abs() < 1.0e-5, "{position:?}");
}

// A body set to a position it already holds must not read as motion, and
// a body that is not driven by position must refuse the instruction.
#[test]
fn only_a_position_driven_body_takes_a_translation_target() {
    let mut sim = Simulation::with_capacity(2);
    let fixed = floor(&mut sim);
    let ball = sim
        .add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 5.0, 0.0],
            [0.0; 3],
            params(0.0, 0.0),
            LayerMask::ALL,
        )
        .expect("room");
    assert!(!sim.set_kinematic_translation(fixed, [0.0, 9.0, 0.0]));
    assert!(!sim.set_kinematic_translation(ball, [0.0, 9.0, 0.0]));
    sim.step(TICK);
    assert!(sim.body_pose(ball).expect("live").0[1] < 5.0, "still falls");
}

#[test]
fn a_driven_body_pushes_a_dynamic_one_it_is_moved_into() {
    let mut sim = Simulation::with_capacity(3);
    floor(&mut sim);
    let crate_body = sim
        .add_dynamic(
            &ColliderShape::Cuboid {
                half_extents: [0.5, 0.5, 0.5],
            },
            [0.0, 0.5, 0.0],
            [0.0; 3],
            params(0.0, 0.0),
            LayerMask::ALL,
        )
        .expect("room");
    let blade = sim
        .add_kinematic(
            &ColliderShape::Cuboid {
                half_extents: [0.5, 0.5, 0.5],
            },
            [-3.0, 0.5, 0.0],
            [0.0; 3],
            0.5,
            LayerMask::ALL,
        )
        .expect("room");

    // Long enough for the crate to settle and doze off before the blade
    // arrives, so this also proves the push wakes it.
    let mut x = -3.0f32;
    for step in 0..180 {
        if step == 60 {
            assert_eq!(sim.is_sleeping(crate_body), Some(true), "settled first");
        }
        if step >= 60 {
            x += 0.03;
            assert!(sim.set_kinematic_translation(blade, [x, 0.5, 0.0]));
        }
        sim.step(TICK);
    }
    assert!((sim.body_pose(blade).expect("live").0[0] - x).abs() < 1.0e-5);
    let pushed = sim.body_pose(crate_body).expect("live").0[0];
    assert!(pushed > 1.5, "the crate was shoved along: {pushed}");
    assert!(
        pushed > x,
        "and it stays ahead of the blade: {pushed} vs {x}"
    );
}

#[test]
fn switching_a_bodys_kind_keeps_its_handle_and_leaves_the_world_standing() {
    let mut sim = Simulation::with_capacity(2);
    floor(&mut sim);
    let ball = sim
        .add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 4.0, 0.0],
            [0.0; 3],
            params(0.0, 0.0),
            LayerMask::ALL,
        )
        .expect("room");
    for _ in 0..180 {
        sim.step(TICK);
    }
    let resting = sim.body_pose(ball).expect("live").0;
    assert!((resting[1] - 0.5).abs() < 0.02, "{resting:?}");

    // Picked up: held in the air by position alone.
    assert!(sim.make_kinematic(ball));
    assert_eq!(sim.is_kinematic(ball), Some(true));
    assert!(sim.set_kinematic_translation(ball, [0.0, 6.0, 0.0]));
    sim.step(TICK);
    for _ in 0..60 {
        sim.step(TICK);
    }
    assert_eq!(sim.body_pose(ball).expect("live").0, [0.0, 6.0, 0.0]);

    // Thrown: the same handle, back under gravity, and it lands again.
    assert!(sim.make_dynamic(ball, [0.0, 0.0, 2.0]));
    assert_eq!(sim.is_kinematic(ball), Some(false));
    assert!(sim.mass(ball).expect("live") > 0.0);
    for _ in 0..300 {
        sim.step(TICK);
    }
    let landed = sim.body_pose(ball).expect("live").0;
    assert!(
        (landed[1] - 0.5).abs() < 0.02,
        "back on the floor: {landed:?}"
    );
    assert!(landed[2] > 0.5, "and it traveled: {landed:?}");
    assert_eq!(sim.body_count(), 2);
    assert_eq!(sim.collider_count(), sim.body_count());
}

// The broad phase filters pairs by whether contact can move either body,
// so a switch that did not reach it would leave a stack leaning on
// nothing.
#[test]
fn a_stack_stays_up_when_the_body_under_it_is_switched() {
    let mut sim = Simulation::with_capacity(3);
    floor(&mut sim);
    let cube = ColliderShape::Cuboid {
        half_extents: [0.5, 0.5, 0.5],
    };
    let lower = sim
        .add_dynamic(
            &cube,
            [0.0, 0.5, 0.0],
            [0.0; 3],
            params(0.0, 0.0),
            LayerMask::ALL,
        )
        .expect("room");
    let upper = sim
        .add_dynamic(
            &cube,
            [0.0, 1.5, 0.0],
            [0.0; 3],
            params(0.0, 0.0),
            LayerMask::ALL,
        )
        .expect("room");
    for _ in 0..180 {
        sim.step(TICK);
    }
    let held = sim.body_pose(lower).expect("live").0;
    assert!(sim.make_kinematic(lower));
    for _ in 0..180 {
        sim.step(TICK);
    }
    let top = sim.body_pose(upper).expect("live").0;
    assert!(
        (top[1] - 1.5).abs() < 0.05,
        "the top box still rests: {top:?}"
    );
    assert_eq!(
        sim.body_pose(lower).expect("live").0,
        held,
        "and the one below it has not stirred"
    );

    // And a lift carries the box above it up too.
    assert!(sim.set_kinematic_translation(lower, [held[0], 1.5, held[2]]));
    for _ in 0..120 {
        sim.step(TICK);
    }
    let lifted = sim.body_pose(upper).expect("live").0;
    assert!(lifted[1] > 2.0, "carried up: {lifted:?}");
}

#[test]
fn switching_the_kind_of_a_body_that_is_not_there_reports_so() {
    let mut sim = Simulation::with_capacity(1);
    let ball = sim
        .add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 1.0, 0.0],
            [0.0; 3],
            params(0.0, 0.0),
            LayerMask::ALL,
        )
        .expect("room");
    assert!(sim.remove_body(ball));
    assert!(!sim.make_kinematic(ball));
    assert!(!sim.make_dynamic(ball, [0.0; 3]));
    assert!(!sim.set_kinematic_translation(ball, [0.0; 3]));
    assert_eq!(sim.is_kinematic(ball), None);
}

// Two runs of the same query on the same world must agree bit for bit,
// or nothing built on a query can be reproduced.
#[test]
fn queries_answer_the_same_bits_twice_running() {
    let mut sim = Simulation::with_capacity(17);
    floor(&mut sim);
    // Terrain answers along its own faces rather than through the sweep
    // the convex shapes use, so it needs its own place in this check.
    let side = 9usize;
    let mut heights = Vec::with_capacity(side * side);
    for row in 0..side {
        for col in 0..side {
            heights.push((row as f32 * 0.37).sin() * 0.4 + (col as f32 * 0.21).cos() * 0.3);
        }
    }
    sim.add_heightfield(
        side,
        side,
        heights,
        [24.0, 1.0, 24.0],
        [0.0, -3.0, 0.0],
        LayerMask::ALL,
    )
    .expect("room");
    for i in 0..12 {
        sim.add_dynamic(
            &ColliderShape::Cuboid {
                half_extents: [0.4, 0.4, 0.4],
            },
            [(i % 3) as f32 * 0.9, 1.0 + (i / 3) as f32 * 0.9, 0.0],
            [0.0, i as f32 * 7.0, 0.0],
            params(0.3, 0.0),
            LayerMask::ALL,
        )
        .expect("room");
    }
    for _ in 0..90 {
        sim.step(TICK);
    }
    let ray = |sim: &Simulation| {
        sim.raycast(
            [-6.0, 1.3, 0.1],
            [1.0, -0.1, 0.0],
            40.0,
            None,
            LayerMask::ALL,
        )
        .map(|hit| (hit.distance.to_bits(), hit.point, hit.normal))
    };
    let sweep = |sim: &Simulation| {
        sim.shape_cast(&ShapeCast::new(
            ColliderShape::Capsule {
                half_height: 0.3,
                radius: 0.2,
            },
            [-6.0, 1.3, 0.1],
            [12.0, 0.0, 0.0],
        ))
        .map(|hit| (hit.toi.to_bits(), hit.point, hit.normal, hit.body))
    };
    let shape = Simulation::character_shape(0.3, 0.2);
    let capsule = sim
        .add_kinematic(
            &ColliderShape::Capsule {
                half_height: 0.3,
                radius: 0.2,
            },
            [-6.0, 1.3, 0.1],
            [0.0; 3],
            0.8,
            LayerMask::ALL,
        )
        .expect("room");
    let walk = |sim: &Simulation| {
        let moved = sim.move_character(
            &shape,
            &CharacterMoveInput {
                center: [-6.0, 1.3, 0.1],
                desired: [12.0, -0.05, 0.0],
                dt: TICK,
                exclude: capsule,
                mask: LayerMask::ALL,
            },
        );
        (moved.translation.map(f32::to_bits), moved.grounded)
    };
    // Fired into the terrain rather than at the stack, so the two answers
    // below are about the height grid and not about the boxes on it.
    let terrain_ray = |sim: &Simulation| {
        sim.raycast(
            [-1.7, 6.0, 2.3],
            [0.0, -1.0, 0.0],
            40.0,
            None,
            LayerMask::ALL,
        )
        .map(|hit| (hit.distance.to_bits(), hit.point, hit.normal))
    };
    let terrain_sweep = |sim: &Simulation| {
        sim.shape_cast(&ShapeCast::new(
            ColliderShape::Ball { radius: 0.4 },
            [-9.0, -2.2, 2.3],
            [18.0, 0.0, 0.0],
        ))
        .map(|hit| (hit.toi.to_bits(), hit.point, hit.normal, hit.body))
    };
    assert!(
        ray(&sim).is_some() && sweep(&sim).is_some(),
        "the scene is in the way"
    );
    assert!(
        terrain_ray(&sim).is_some() && terrain_sweep(&sim).is_some(),
        "and so is the terrain"
    );
    assert_eq!(ray(&sim), ray(&sim));
    assert_eq!(sweep(&sim), sweep(&sim));
    assert_eq!(walk(&sim), walk(&sim));
    assert_eq!(terrain_ray(&sim), terrain_ray(&sim));
    assert_eq!(terrain_sweep(&sim), terrain_sweep(&sim));
}

#[test]
fn config_round_trips_and_takes_effect() {
    // Sleeping is off for the weightless stretch: a body that settles
    // there would stop being simulated, and the test would be measuring
    // that rather than the gravity change.
    let mut sim = Simulation::new(
        SimConfig {
            gravity: 0.0,
            allow_sleep: false,
            ..SimConfig::default()
        },
        1,
    );
    assert_eq!(sim.config().gravity, 0.0);
    let ball = sim
        .add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 5.0, 0.0],
            [0.0; 3],
            params(0.0, 0.0),
            LayerMask::ALL,
        )
        .expect("room");
    for _ in 0..60 {
        sim.step(TICK);
    }
    assert_eq!(sim.body_pose(ball).expect("live").0[1], 5.0);
    sim.set_config(SimConfig::default());
    assert_eq!(sim.config().gravity, crate::physics::GRAVITY);
    sim.step(TICK);
    assert!(sim.body_pose(ball).expect("live").0[1] < 5.0);
}

#[test]
fn gravity_scale_and_damping_do_what_they_say() {
    let mut sim = Simulation::with_capacity(2);
    let floating = sim
        .add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 5.0, 0.0],
            [0.0; 3],
            DynamicParams {
                gravity_scale: 0.0,
                ..params(0.0, 0.0)
            },
            LayerMask::ALL,
        )
        .expect("room");
    let damped = sim
        .add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [5.0, 5.0, 0.0],
            [0.0; 3],
            DynamicParams {
                gravity_scale: 0.0,
                ..params(0.0, 4.0)
            },
            LayerMask::ALL,
        )
        .expect("room");
    sim.set_linear_velocity(floating, [3.0, 0.0, 0.0]);
    sim.set_linear_velocity(damped, [3.0, 0.0, 0.0]);
    for _ in 0..60 {
        sim.step(TICK);
    }
    assert_eq!(sim.body_pose(floating).expect("live").0[1], 5.0);
    let free = sim.linear_velocity(floating).expect("live")[0];
    let slowed = sim.linear_velocity(damped).expect("live")[0];
    assert!((free - 3.0).abs() < 1.0e-5, "{free}");
    assert!(slowed < 0.5, "damping must bleed the speed off: {slowed}");
}
