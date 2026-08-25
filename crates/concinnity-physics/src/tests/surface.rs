//! One scenario across the whole simulation surface.
//!
//! The other integration tests each take one part of the simulation and press
//! on it. This one takes everything a driver asks of a running world -- build
//! it from a budget, add each kind of body, step, query, drive a character,
//! switch a body's kind, drain what the step reported, tear it all down -- and
//! runs it once, in the order a driver does. It is deliberately shallow: what
//! each answer should be is checked elsewhere, and what is checked here is
//! that the whole surface answers at all.

use crate::{
    CharacterMoveInput, ColliderShape, DynamicParams, LayerMask, PhysicsBudget, PhysicsCounts,
    SimConfig, Simulation,
};
use alloc::vec::Vec;

const G: f32 = 20.0;
const TICK: f32 = 1.0 / 60.0;

// Room for the floor, the falling ball, and the character capsule.
fn budget() -> PhysicsBudget {
    PhysicsBudget::derive(
        &PhysicsCounts {
            dynamic_colliders: 1,
            player_capsules: 1,
            ..PhysicsCounts::default()
        },
        0,
    )
}

#[test]
fn the_whole_surface_answers() {
    let budget = budget();
    let mut world = Simulation::new(
        SimConfig {
            gravity: G,
            ..SimConfig::default()
        },
        budget.body_cap() as usize,
    );
    world.set_contact_min_impulse(1.0, TICK);
    world.configure_character(50.0, 0.3, true);

    let floor = world
        .add_fixed(
            &ColliderShape::Cuboid {
                half_extents: [50.0, 5.0, 50.0],
            },
            [0.0, -5.0, 0.0],
            [0.0; 3],
            0.8,
            LayerMask::ALL,
        )
        .expect("room for the floor");
    let ball = world
        .add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 8.0, 0.0],
            [0.0; 3],
            DynamicParams {
                mass: 1.0,
                friction: 0.5,
                restitution: 0.0,
                gravity_scale: 1.0,
                linear_damping: 0.0,
            },
            LayerMask::ALL,
        )
        .expect("room for the ball");
    assert_eq!(world.body_count(), 2);
    assert_eq!(world.collider_count(), 2);

    for _ in 0..120 {
        world.step(TICK);
    }
    let (pos, _) = world.body_pose(ball).expect("a live body");
    assert!(pos[1] < 8.0, "the body must fall, y = {}", pos[1]);
    assert!(pos[1] > -1.0, "the floor must stop it, y = {}", pos[1]);

    // The quaternion pose reports the same position as the Euler one.
    let (quat_pos, _) = world.body_pose_quat(ball).expect("a live body");
    assert!((quat_pos[1] - pos[1]).abs() < 1.0e-5);

    let hit = world
        .raycast(
            [0.0, 20.0, 0.0],
            [0.0, -1.0, 0.0],
            100.0,
            None,
            LayerMask::ALL,
        )
        .expect("a downward ray must find the scene");
    assert!(
        hit.normal[1] > 0.5,
        "an up-facing normal, got {:?}",
        hit.normal
    );

    let capsule = Simulation::character_shape(0.6, 0.3);
    let character = world
        .add_character(0.6, 0.3, [4.0, 2.0, 0.0], LayerMask::ALL)
        .expect("room for the capsule");
    let moved = world.move_character(
        &capsule,
        &CharacterMoveInput {
            center: [4.0, 2.0, 0.0],
            desired: [0.0, -1.0, 0.0],
            dt: TICK,
            exclude: character,
            mask: LayerMask::ALL,
        },
    );
    assert!(moved.translation[1] <= 0.0, "a downward move must not rise");
    assert!(world.set_kinematic_translation(character, [4.0, 1.5, 0.0]));

    assert!(world.make_dynamic(ball, [0.0, 1.0, 0.0]));
    assert!(world.make_kinematic(ball));

    let mut crossings = Vec::new();
    let mut contacts = Vec::new();
    world.step(TICK);
    world.drain_sensor_crossings_into(&mut crossings);
    world.drain_contact_hits_into(&mut contacts);

    assert!(world.remove_body(ball));
    assert!(world.remove_body(floor));
    assert!(world.remove_body(character));
    assert_eq!(world.body_count(), 0, "every body was removed");
    assert!(
        world.body_pose(ball).is_none(),
        "a handle to a removed body names nothing"
    );
}
