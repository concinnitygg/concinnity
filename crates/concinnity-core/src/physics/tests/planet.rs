//! What a planet asks of the simulation: gravity toward a point, a whole
//! world carried into another frame without anything noticing, and a ground
//! grid swapped for another under the bodies resting on it.

use super::fixtures::{TICK, add_floor, awake, drop_ball, position, step_for};
use crate::math::{quat_from_axis_angle, quat_rotate};
use crate::physics::{BodyHandle, DynamicParams, GRAVITY, LayerMask, Simulation};
use alloc::vec;

fn ball() -> DynamicParams {
    DynamicParams {
        mass: 1.0,
        friction: 0.6,
        restitution: 0.0,
        gravity_scale: 1.0,
        linear_damping: 0.0,
    }
}

fn close(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() < tol)
}

// Free fall from either side of a point attractor heads at the point.
#[test]
fn gravity_pulls_toward_its_center_from_anywhere() {
    let mut sim = Simulation::new(awake(), 2);
    sim.set_gravity_center(Some([0.0, -100.0, 0.0]));
    let above = drop_ball(&mut sim, [0.0, 0.0, 0.0], ball());
    let beside = drop_ball(&mut sim, [100.0, -100.0, 0.0], ball());
    step_for(&mut sim, 30);
    let fall = 0.5 * GRAVITY * 0.25;
    let a = position(&sim, above);
    assert!(close(a, [0.0, -fall, 0.0], 0.05), "{a:?}");
    let b = position(&sim, beside);
    assert!(close(b, [100.0 - fall, -100.0, 0.0], 0.05), "{b:?}");
}

// Two copies of one scene, one carried into another frame partway through,
// land in the same place once the carried one is mapped back: the move
// changed nothing the solver could see.
#[test]
fn a_rebased_world_carries_on_as_if_nothing_happened() {
    let build = || {
        let mut sim = Simulation::new(awake(), 4);
        sim.set_gravity_center(Some([0.0, -500.0, 0.0]));
        add_floor(&mut sim);
        let rolling = drop_ball(&mut sim, [2.0, 0.5, 0.0], ball());
        let falling = drop_ball(&mut sim, [-3.0, 4.0, 1.0], ball());
        (sim, rolling, falling)
    };
    let (mut reference, r_roll, r_fall) = build();
    let (mut moved, m_roll, m_fall) = build();
    step_for(&mut reference, 20);
    step_for(&mut moved, 20);

    let rotation = quat_from_axis_angle([0.3, 0.0, 1.0], 0.02);
    let translation = [-700.0, 9.0, 300.0];
    moved.rebase(rotation, translation);
    let carry = |p: [f32; 3]| {
        let r = quat_rotate(rotation, p);
        [
            r[0] + translation[0],
            r[1] + translation[1],
            r[2] + translation[2],
        ]
    };
    assert!(close(
        position(&moved, m_fall),
        carry(position(&reference, r_fall)),
        1e-3
    ));

    for _ in 0..40 {
        reference.step(TICK);
        moved.step(TICK);
    }
    for (r, m) in [(r_roll, m_roll), (r_fall, m_fall)] {
        let expected = carry(position(&reference, r));
        let got = position(&moved, m);
        assert!(close(got, expected, 0.01), "{got:?} vs {expected:?}");
    }
}

// A sensor region is a body like any other: a frame move carries it, and a
// ball dropped into the carried region crosses it there.
#[test]
fn a_rebase_carries_sensor_regions() {
    use crate::physics::{ColliderShape, SensorCrossing};
    let mut sim = Simulation::new(awake(), 2);
    let at = [1_000.0, 1.0, -3.0];
    sim.add_sensor(
        &ColliderShape::Cuboid {
            half_extents: [1.0; 3],
        },
        at,
        [0.0; 3],
        7,
        LayerMask::ALL,
    )
    .expect("room for the sensor");
    let rotation = quat_from_axis_angle([1.0, 0.0, 0.0], 0.02);
    let translation = [-1_000.0, 20.0, 0.0];
    sim.rebase(rotation, translation);
    let r = quat_rotate(rotation, at);
    let carried = [
        r[0] + translation[0],
        r[1] + translation[1],
        r[2] + translation[2],
    ];
    drop_ball(&mut sim, [carried[0], carried[1] + 3.0, carried[2]], ball());
    let mut crossings: alloc::vec::Vec<SensorCrossing> = alloc::vec::Vec::new();
    let mut entered = false;
    for _ in 0..60 {
        sim.step(TICK);
        sim.drain_sensor_crossings_into(&mut crossings);
        entered |= crossings.iter().any(|c| c.tag == 7 && c.entered);
    }
    assert!(entered, "the ball fell into the carried region");
}

fn grid(sim: &mut Simulation, y: f32) -> BodyHandle {
    sim.add_heightfield(
        5,
        5,
        vec![0.0; 25],
        [20.0, 1.0, 20.0],
        [0.0, y, 0.0],
        LayerMask::ALL,
    )
    .expect("room for the grid")
}

// A resting ball wakes when its ground is replaced and settles on the new one.
#[test]
fn a_replaced_grid_carries_what_rests_on_it() {
    let mut sim = Simulation::new(Default::default(), 2);
    let ground = grid(&mut sim, 0.0);
    let b = drop_ball(&mut sim, [0.0, 0.6, 0.0], ball());
    step_for(&mut sim, 120);
    assert!((position(&sim, b)[1] - 0.5).abs() < 0.05);

    assert!(sim.replace_heightfield(
        ground,
        5,
        5,
        vec![0.0; 25],
        [20.0, 1.0, 20.0],
        [0.0, -2.0, 0.0]
    ));
    step_for(&mut sim, 120);
    let y = position(&sim, b)[1];
    assert!((y + 1.5).abs() < 0.05, "settled on the lowered grid: {y}");
    assert_eq!(sim.body_count(), 2, "the same body, not a new one");

    // Only a grid can be replaced by a grid.
    assert!(!sim.replace_heightfield(b, 5, 5, vec![0.0; 25], [20.0, 1.0, 20.0], [0.0; 3]));
}
