//! What terrain has to do once a world is standing on it.
//!
//! The unit tests inside the crate check one triangle, one cell, and one
//! query. These check the surface: a body that rests on it without sinking,
//! one that rolls down it, a character that walks across the boundary between
//! two cells without being caught by the edge they share, and the probes an
//! animation or a camera fires at it.

use concinnity_physics::{
    BodyHandle, CharacterMoveInput, ColliderShape, DynamicParams, GRAVITY, LayerMask, ShapeCast,
    SimConfig, Simulation,
};

const TICK: f32 = 1.0 / 60.0;
/// The terrain fixtures are twenty units square.
const EXTENT: f32 = 20.0;
const HALF_HEIGHT: f32 = 0.6;
const RADIUS: f32 = 0.3;
/// Distance from a character capsule's centre to the ground it stands on.
const STAND: f32 = HALF_HEIGHT + RADIUS;

fn params(friction: f32) -> DynamicParams {
    DynamicParams {
        mass: 1.0,
        friction,
        restitution: 0.0,
        gravity_scale: 1.0,
        linear_damping: 0.0,
    }
}

fn awake() -> SimConfig {
    SimConfig {
        allow_sleep: false,
        ..SimConfig::default()
    }
}

/// A world whose only fixed body is a height grid built from `height`, with
/// the handle naming that grid.
fn terrain(
    side: usize,
    capacity: usize,
    height: impl Fn(f32, f32) -> f32,
) -> (Simulation, BodyHandle) {
    let mut sim = Simulation::new(awake(), capacity + 1);
    let mut heights = Vec::with_capacity(side * side);
    for row in 0..side {
        // Rows run along z, columns along x.
        let z = (row as f32 / (side - 1) as f32 - 0.5) * EXTENT;
        for col in 0..side {
            let x = (col as f32 / (side - 1) as f32 - 0.5) * EXTENT;
            heights.push(height(x, z));
        }
    }
    let ground = sim
        .add_heightfield(
            side,
            side,
            heights,
            [EXTENT, 1.0, EXTENT],
            [0.0; 3],
            LayerMask::ALL,
        )
        .expect("room for the terrain");
    (sim, ground)
}

fn flat(capacity: usize) -> (Simulation, BodyHandle) {
    terrain(9, capacity, |_, _| 0.0)
}

/// Terrain sloping down toward `+x` at one in four.
fn slope(capacity: usize) -> (Simulation, BodyHandle) {
    terrain(9, capacity, |x, _| -x * 0.25)
}

fn drop_ball(sim: &mut Simulation, pos: [f32; 3], friction: f32) -> BodyHandle {
    sim.add_dynamic(
        &ColliderShape::Ball { radius: 0.5 },
        pos,
        [0.0; 3],
        params(friction),
        LayerMask::ALL,
    )
    .expect("room for the ball")
}

fn step_for(sim: &mut Simulation, ticks: usize) {
    for _ in 0..ticks {
        sim.step(TICK);
    }
}

fn position(sim: &Simulation, handle: BodyHandle) -> [f32; 3] {
    sim.body_pose(handle).expect("a live body").0
}

// The first thing terrain has to do, and the one that fails quietly: hold a
// body up without letting it settle a millimetre lower every second.
#[test]
fn a_body_rests_on_flat_terrain_without_sinking() {
    let (mut sim, _ground) = flat(2);
    let ball = drop_ball(&mut sim, [1.3, 3.0, -2.1], 0.6);
    step_for(&mut sim, 300);

    let landed = position(&sim, ball);
    assert!(
        (landed[1] - 0.5).abs() < 0.02,
        "the ball should rest a radius above the surface, rests at {}",
        landed[1]
    );
    step_for(&mut sim, 600);
    let later = position(&sim, ball);
    assert!(
        (later[1] - landed[1]).abs() < 5.0e-3,
        "it sank from {} to {} over ten seconds",
        landed[1],
        later[1]
    );
    assert!(
        (later[0] - landed[0]).abs() < 0.02 && (later[2] - landed[2]).abs() < 0.02,
        "and crept sideways: {landed:?} -> {later:?}"
    );
}

// A box rests on the face it stands on rather than tipping onto a corner,
// which is what having a whole contact patch per triangle buys.
#[test]
fn a_box_rests_flat_on_terrain_rather_than_tipping() {
    let (mut sim, _ground) = flat(2);
    let cube = sim
        .add_dynamic(
            &ColliderShape::Cuboid {
                half_extents: [0.4, 0.4, 0.4],
            },
            [-1.0, 2.0, 1.7],
            [0.0; 3],
            params(0.6),
            LayerMask::ALL,
        )
        .expect("room for the box");
    step_for(&mut sim, 300);

    let (at, rotation) = sim.body_pose(cube).expect("a live body");
    assert!((at[1] - 0.4).abs() < 0.02, "it rests at {}", at[1]);
    for angle in rotation {
        assert!(angle.abs() < 3.0, "it tipped over: {rotation:?}");
    }
}

#[test]
fn a_body_rolls_downhill_on_sloping_terrain() {
    let (mut sim, _ground) = slope(2);
    // A frictionless ball on a one-in-four slope has to run downhill.
    let ball = drop_ball(&mut sim, [-4.0, 2.0, 0.0], 0.0);
    step_for(&mut sim, 30);
    let start = position(&sim, ball);
    step_for(&mut sim, 180);
    let end = position(&sim, ball);

    assert!(
        end[0] - start[0] > 1.0,
        "it should have run downhill: {start:?} -> {end:?}"
    );
    assert!(
        end[1] < start[1] - 0.2,
        "and lost height doing it: {start:?} -> {end:?}"
    );
    assert!(end[2].abs() < 0.2, "it wandered off the fall line: {end:?}");
}

// The one the character controller depends on: crossing from one cell to the
// next must not be stopped by the edge the two triangles share.
#[test]
fn a_character_walks_across_the_cell_boundaries_without_catching() {
    let (mut sim, _ground) = flat(2);
    sim.configure_character(45.0, 0.3, true);
    let start = [-8.0, STAND, 0.35];
    let capsule = sim
        .add_kinematic(
            &ColliderShape::Capsule {
                half_height: HALF_HEIGHT,
                radius: RADIUS,
            },
            start,
            [0.0; 3],
            0.8,
            LayerMask::ALL,
        )
        .expect("room for the capsule");
    let shape = Simulation::character_shape(HALF_HEIGHT, RADIUS);

    let pace = 0.06f32;
    let mut center = start;
    let mut fall = 0.0f32;
    for tick in 0..250 {
        fall -= GRAVITY * TICK;
        let moved = sim.move_character(
            &shape,
            &CharacterMoveInput {
                center,
                desired: [pace, fall * TICK, 0.0],
                dt: TICK,
                exclude: capsule,
                mask: LayerMask::ALL,
            },
        );
        // Every tick has to carry the whole step: a snag reads as a tick that
        // moved less than it was asked to.
        assert!(
            moved.translation[0] > pace * 0.98,
            "tick {tick} at x = {} only moved {}",
            center[0],
            moved.translation[0]
        );
        assert!(moved.grounded, "tick {tick}: the surface is underfoot");
        for (axis, at) in center.iter_mut().enumerate() {
            *at += moved.translation[axis];
        }
        if moved.grounded && fall < 0.0 {
            fall = 0.0;
        }
        assert!(
            (center[1] - STAND).abs() < 0.02,
            "tick {tick}: the walk left the surface at {center:?}"
        );
        sim.set_kinematic_translation(capsule, center);
        sim.step(TICK);
    }
    assert!(
        center[0] > start[0] + 14.0,
        "the walk has to cross the grid: {center:?}"
    );
}

#[test]
fn a_ray_lands_on_the_terrain_surface_with_the_surfaces_normal() {
    let (mut sim, _ground) = flat(1);
    sim.step(TICK);
    for (x, z) in [(0.0, 0.0), (-6.2, 3.7), (4.1, -8.3), (9.4, 9.4)] {
        let hit = sim
            .raycast([x, 6.0, z], [0.0, -1.0, 0.0], 20.0, None, LayerMask::ALL)
            .unwrap_or_else(|| panic!("nothing under ({x}, {z})"));
        assert!(hit.point[1].abs() < 1.0e-3, "({x}, {z}): {hit:?}");
        assert!((hit.distance - 6.0).abs() < 1.0e-3, "({x}, {z}): {hit:?}");
        assert!(hit.normal[1] > 0.999, "({x}, {z}): {hit:?}");
    }
    // Off the edge of the grid there is nothing to hit.
    assert!(
        sim.raycast(
            [40.0, 6.0, 0.0],
            [0.0, -1.0, 0.0],
            20.0,
            None,
            LayerMask::ALL
        )
        .is_none()
    );

    // A slope's normal has to lean, or a foot plant lands flat on a hill.
    let (mut sloped, _ground) = slope(1);
    sloped.step(TICK);
    let hit = sloped
        .raycast(
            [-2.0, 6.0, 1.0],
            [0.0, -1.0, 0.0],
            20.0,
            None,
            LayerMask::ALL,
        )
        .expect("the slope is down there");
    assert!((hit.point[1] - 0.5).abs() < 1.0e-3, "{hit:?}");
    assert!(hit.normal[0] > 0.2 && hit.normal[1] > 0.9, "{hit:?}");
}

#[test]
fn a_shape_cast_stops_on_the_terrain_and_names_it() {
    let (mut sim, _ground) = flat(1);
    sim.step(TICK);
    let capsule = ColliderShape::Capsule {
        half_height: HALF_HEIGHT,
        radius: RADIUS,
    };
    let hit = sim
        .shape_cast(&ShapeCast::new(capsule, [2.2, 4.0, -3.3], [0.0, -8.0, 0.0]))
        .expect("the ground is down there");
    let landed = 4.0 - hit.toi * 8.0;
    assert!((landed - STAND).abs() < 0.01, "landed at {landed}");
    assert!(hit.normal[1] > 0.999, "{hit:?}");
    assert!(!hit.started_touching);
    assert!(sim.body_pose(hit.body).is_some(), "it names the terrain");
}

// A query that wants more surface than the cap allows has to say so rather
// than quietly answering about part of it.
#[test]
fn a_query_past_the_candidate_cap_is_counted_rather_than_hidden() {
    let mut sim = Simulation::new(awake(), 2);
    let side = 64;
    sim.add_heightfield(
        side,
        side,
        vec![0.0; side * side],
        [400.0, 1.0, 400.0],
        [0.0; 3],
        LayerMask::ALL,
    )
    .expect("room for the terrain");
    sim.step(TICK);
    assert_eq!(sim.heightfield_overflows(), 0);

    // A sweep whose swept box covers most of the grid names far more
    // triangles than a query is allowed to spend.
    let hit = sim.shape_cast(&ShapeCast::new(
        ColliderShape::Ball { radius: 1.0 },
        [-180.0, 0.5, -180.0],
        [360.0, 0.0, 360.0],
    ));
    assert!(
        sim.heightfield_overflows() > 0,
        "the cap has to be reported: {hit:?}"
    );
    sim.clear_heightfield_overflows();
    assert_eq!(sim.heightfield_overflows(), 0);

    // A query that fits reports nothing, which is what makes the count mean
    // something.
    sim.raycast(
        [0.0, 6.0, 0.0],
        [0.0, -1.0, 0.0],
        20.0,
        None,
        LayerMask::ALL,
    )
    .expect("the ground is down there");
    assert_eq!(sim.heightfield_overflows(), 0);
}

#[test]
fn a_grid_that_names_no_surface_is_refused_rather_than_built() {
    let mut sim = Simulation::new(awake(), 2);
    assert!(
        sim.add_heightfield(
            1,
            4,
            vec![0.0; 4],
            [4.0, 1.0, 4.0],
            [0.0; 3],
            LayerMask::ALL
        )
        .is_none()
    );
    assert!(
        sim.add_heightfield(
            3,
            3,
            vec![0.0; 4],
            [4.0, 1.0, 4.0],
            [0.0; 3],
            LayerMask::ALL
        )
        .is_none()
    );
    assert!(
        sim.add_heightfield(
            3,
            3,
            vec![0.0; 9],
            [0.0, 1.0, 4.0],
            [0.0; 3],
            LayerMask::ALL
        )
        .is_none()
    );
    assert_eq!(sim.body_count(), 0);
    // And a grid that does name one is built.
    assert!(
        sim.add_heightfield(
            3,
            3,
            vec![0.0; 9],
            [4.0, 1.0, 4.0],
            [0.0; 3],
            LayerMask::ALL
        )
        .is_some()
    );
    assert_eq!(sim.body_count(), 1);
}

// Terrain is a body like any other: removing it takes the ground away.
#[test]
fn removing_the_terrain_lets_what_was_standing_on_it_fall() {
    let (mut sim, ground) = flat(2);
    let ball = drop_ball(&mut sim, [0.0, 2.0, 0.0], 0.6);
    step_for(&mut sim, 240);
    let resting = position(&sim, ball);
    assert!((resting[1] - 0.5).abs() < 0.02, "{resting:?}");

    assert!(sim.remove_body(ground));
    assert!(sim.body_pose(ground).is_none());
    step_for(&mut sim, 120);
    assert!(
        position(&sim, ball)[1] < -5.0,
        "with the ground gone it falls: {:?}",
        position(&sim, ball)
    );
}

// Everything above is worth nothing if the answer depends on the day.
#[test]
fn a_terrain_scene_runs_identically_twice() {
    let run = || {
        let (mut sim, _ground) = slope(3);
        let ball = drop_ball(&mut sim, [-3.0, 2.0, 0.7], 0.4);
        let cube = sim
            .add_dynamic(
                &ColliderShape::Cuboid {
                    half_extents: [0.3, 0.3, 0.3],
                },
                [1.0, 2.0, -1.1],
                [0.0, 25.0, 0.0],
                params(0.5),
                LayerMask::ALL,
            )
            .expect("room for the box");
        step_for(&mut sim, 300);
        let bits = |handle| {
            let at = position(&sim, handle);
            [at[0].to_bits(), at[1].to_bits(), at[2].to_bits()]
        };
        (bits(ball), bits(cube))
    };
    assert_eq!(run(), run());
}
