//! What a character capsule has to do when it is driven at the world.
//!
//! The unit tests inside the crate check the arithmetic a move is built out
//! of. These check the move itself, against scenes built to provoke the cases
//! a controller is actually judged by: a wall it must not pass, a slope it may
//! climb and one it may not, a kerb it steps onto, a lip it stays attached
//! over, and a wedge it must give up on rather than circle inside forever.
//!
//! Every mover here is driven the way the engine drives one: the caller keeps
//! the vertical velocity, adds a tick of gravity to it, asks for the move, and
//! zeroes the fall on landing.

use crate::physics::{
    BodyHandle, CharacterCapsule, CharacterMove, CharacterMoveInput, ColliderShape, GRAVITY,
    LayerMask, Simulation,
};

const TICK: f32 = 1.0 / 60.0;
const HALF_HEIGHT: f32 = 0.6;
const RADIUS: f32 = 0.3;
/// Distance from the capsule's centre to the ground it stands on.
const STAND: f32 = HALF_HEIGHT + RADIUS;
/// Walking speed as a per-tick displacement.
const PACE: f32 = 0.05;

fn scene(capacity: usize) -> Simulation {
    let mut sim = Simulation::with_capacity(capacity + 1);
    sim.configure_character(45.0, 0.3, true);
    sim
}

/// A floor whose top surface is exactly `y = 0`.
fn add_floor(sim: &mut Simulation) {
    add_box(sim, [20.0, 0.5, 20.0], [0.0, -0.5, 0.0], [0.0; 3]);
}

fn add_box(sim: &mut Simulation, half_extents: [f32; 3], pos: [f32; 3], euler_deg: [f32; 3]) {
    sim.add_fixed(
        &ColliderShape::Cuboid { half_extents },
        pos,
        euler_deg,
        0.8,
        LayerMask::ALL,
    )
    .expect("room for a body");
}

/// A character capsule and the world it is driven through, kept together
/// because a move is only meaningful applied: the caller owns the position,
/// and the simulation is told where the capsule ended up.
struct Mover {
    sim: Simulation,
    shape: CharacterCapsule,
    handle: BodyHandle,
    center: [f32; 3],
    mask: LayerMask,
    vy: f32,
    grounded: bool,
}

impl Mover {
    fn new(sim: Simulation, center: [f32; 3]) -> Self {
        Self::layered(sim, center, LayerMask::ALL)
    }

    fn layered(mut sim: Simulation, center: [f32; 3], mask: LayerMask) -> Self {
        let handle = sim
            .add_kinematic(
                &ColliderShape::Capsule {
                    half_height: HALF_HEIGHT,
                    radius: RADIUS,
                },
                center,
                [0.0; 3],
                0.8,
                mask,
            )
            .expect("room for the capsule");
        Mover {
            sim,
            shape: Simulation::character_shape(HALF_HEIGHT, RADIUS),
            handle,
            center,
            mask,
            vy: 0.0,
            grounded: false,
        }
    }

    /// One tick of walking: the requested ground speed plus the fall the
    /// caller has accumulated.
    fn walk(&mut self, x: f32, z: f32) -> CharacterMove {
        self.vy -= GRAVITY * TICK;
        let moved = self.drive([x, self.vy * TICK, z]);
        if moved.grounded && self.vy < 0.0 {
            self.vy = 0.0;
        }
        moved
    }

    /// One tick of a move given exactly as asked, with no fall added.
    fn drive(&mut self, desired: [f32; 3]) -> CharacterMove {
        let moved = self.sim.move_character(
            &self.shape,
            &CharacterMoveInput {
                center: self.center,
                desired,
                dt: TICK,
                exclude: self.handle,
                mask: self.mask,
            },
        );
        for axis in 0..3 {
            self.center[axis] += moved.translation[axis];
        }
        self.sim.set_kinematic_translation(self.handle, self.center);
        self.sim.step(TICK);
        self.grounded = moved.grounded;
        moved
    }

    fn walk_for(&mut self, ticks: usize, x: f32, z: f32) {
        for _ in 0..ticks {
            self.walk(x, z);
        }
    }

    fn x(&self) -> f32 {
        self.center[0]
    }

    fn y(&self) -> f32 {
        self.center[1]
    }

    fn z(&self) -> f32 {
        self.center[2]
    }
}

/// A mover standing on a flat floor, with a wall whose near face is at
/// `z = 1`.
fn walled() -> Mover {
    let mut sim = scene(2);
    add_floor(&mut sim);
    add_box(&mut sim, [4.0, 2.0, 0.5], [0.0, 2.0, 1.5], [0.0; 3]);
    Mover::new(sim, [0.0, STAND, 0.0])
}

#[test]
fn a_move_into_a_wall_is_stopped_by_it() {
    let mut mover = walled();
    mover.walk_for(60, 0.0, PACE);
    // The capsule's surface stops against the face, so its centre stops a
    // radius short of it.
    assert!(
        (mover.z() - (1.0 - RADIUS)).abs() < 0.01,
        "stopped at {}",
        mover.z()
    );
    assert!(mover.grounded, "still standing on the floor");
}

#[test]
fn a_move_at_an_angle_to_a_wall_slides_along_it() {
    let mut mover = walled();
    mover.walk_for(40, PACE, PACE);
    assert!(
        (mover.z() - (1.0 - RADIUS)).abs() < 0.01,
        "held off the wall at {}",
        mover.z()
    );
    // Sideways it keeps almost the whole of what it asked for: only the tick
    // it met the wall on loses anything.
    assert!(mover.x() > 40.0 * PACE - 0.05, "slid to {}", mover.x());
}

/// A mover dropped onto a slope of `degrees`, rising toward `+x`.
fn on_a_slope(degrees: f32, limit_deg: f32) -> Mover {
    let mut sim = Simulation::with_capacity(2);
    sim.configure_character(limit_deg, 0.3, true);
    add_box(
        &mut sim,
        [8.0, 0.5, 8.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, degrees],
    );
    let mut mover = Mover::new(sim, [0.0, 4.0, 0.0]);
    for _ in 0..60 {
        mover.walk(0.0, 0.0);
    }
    mover
}

#[test]
fn a_slope_within_the_limit_is_walked_up() {
    let mut mover = on_a_slope(20.0, 45.0);
    assert!(mover.grounded, "a walkable slope is ground");
    let (from_x, from_y) = (mover.x(), mover.y());
    mover.walk_for(30, PACE, 0.0);
    let (dx, dy) = (mover.x() - from_x, mover.y() - from_y);
    assert!(dx > 0.5, "it has to get somewhere: {dx}");
    // Climbing a 20 degree slope raises the mover by the slope's own ratio.
    let climb = dy / dx;
    assert!((climb - 0.364).abs() < 0.03, "climbed at {climb} per unit");
}

#[test]
fn a_slope_past_the_limit_is_slid_down_rather_than_climbed() {
    let mut mover = on_a_slope(60.0, 45.0);
    assert!(!mover.grounded, "too steep to stand on");
    let (from_x, from_y) = (mover.x(), mover.y());
    // Pushed uphill the whole time, it still ends up lower and further down
    // the hill than it started.
    mover.walk_for(30, PACE, 0.0);
    assert!(
        mover.y() < from_y,
        "{} is no lower than {from_y}",
        mover.y()
    );
    assert!(
        mover.x() < from_x,
        "{} is no further down than {from_x}",
        mover.x()
    );
    assert!(!mover.grounded, "still not standing on it");
}

/// A mover on a floor, facing a step of `height` whose near face is at
/// `z = 1`.
fn stepped(height: f32, grounded: bool) -> Mover {
    let mut sim = Simulation::with_capacity(3);
    sim.configure_character(45.0, 0.3, grounded);
    add_floor(&mut sim);
    add_box(
        &mut sim,
        [4.0, height * 0.5, 2.0],
        [0.0, height * 0.5, 3.0],
        [0.0; 3],
    );
    Mover::new(sim, [0.0, STAND, 0.0])
}

#[test]
fn an_obstacle_no_taller_than_the_step_is_climbed() {
    let mut mover = stepped(0.3, true);
    mover.walk_for(60, 0.0, PACE);
    assert!(
        (mover.y() - (0.3 + STAND)).abs() < 0.01,
        "ended at {} rather than on top",
        mover.y()
    );
    assert!(mover.z() > 1.0, "and past the edge: {}", mover.z());
    assert!(mover.grounded);
}

#[test]
fn an_obstacle_taller_than_the_step_is_not_climbed() {
    let mut mover = stepped(0.8, true);
    mover.walk_for(60, 0.0, PACE);
    assert!(
        (mover.y() - STAND).abs() < 0.01,
        "it climbed to {}",
        mover.y()
    );
    assert!(
        (mover.z() - (1.0 - RADIUS)).abs() < 0.01,
        "stopped at {}",
        mover.z()
    );
}

/// A floor that drops by `fall` at `z = 0`.
fn lipped(fall: f32, grounded: bool) -> Mover {
    let mut sim = Simulation::with_capacity(3);
    sim.configure_character(45.0, 0.3, grounded);
    add_box(&mut sim, [5.0, 0.5, 5.0], [0.0, -0.5, -5.0], [0.0; 3]);
    add_box(&mut sim, [5.0, 0.5, 5.0], [0.0, -0.5 - fall, 5.0], [0.0; 3]);
    Mover::new(sim, [0.0, STAND, -2.0])
}

#[test]
fn a_grounded_mover_stays_attached_walking_off_a_lip() {
    let mut mover = lipped(0.2, true);
    for tick in 0..80 {
        let moved = mover.walk(0.0, PACE);
        assert!(moved.grounded, "left the ground on tick {tick}");
    }
    assert!(mover.z() > 1.0, "it has to cross the lip: {}", mover.z());
    assert!(
        (mover.y() - (STAND - 0.2)).abs() < 0.01,
        "ended at {} rather than on the lower floor",
        mover.y()
    );
}

#[test]
fn a_free_flying_mover_neither_climbs_a_step_nor_sticks_to_the_ground() {
    let mut climbing = stepped(0.3, false);
    for _ in 0..60 {
        climbing.drive([0.0, 0.0, PACE]);
    }
    assert!(
        (climbing.y() - STAND).abs() < 0.01,
        "it climbed to {}",
        climbing.y()
    );
    assert!(
        (climbing.z() - (1.0 - RADIUS)).abs() < 0.01,
        "stopped at {}",
        climbing.z()
    );

    let mut flying = lipped(0.2, false);
    for _ in 0..80 {
        flying.drive([0.0, 0.0, PACE]);
    }
    assert!(flying.z() > 1.0, "it has to cross the lip: {}", flying.z());
    assert!(
        (flying.y() - STAND).abs() < 1.0e-3,
        "it was pulled down to {}",
        flying.y()
    );
}

#[test]
fn grounded_says_the_floor_underfoot_and_nothing_else() {
    let mut sim = scene(1);
    add_floor(&mut sim);
    let mut mover = Mover::new(sim, [0.0, STAND, 0.0]);
    assert!(mover.walk(0.0, 0.0).grounded, "standing on the floor");

    let mut sim = scene(1);
    add_floor(&mut sim);
    let mut airborne = Mover::new(sim, [0.0, STAND + 3.0, 0.0]);
    assert!(!airborne.walk(0.0, 0.0).grounded, "three units up");
    // Falling far enough, it finds the floor again and stays on it.
    airborne.walk_for(90, 0.0, 0.0);
    assert!(airborne.grounded, "it has to land");
    assert!(
        (airborne.y() - STAND).abs() < 0.01,
        "landed at {}",
        airborne.y()
    );
}

#[test]
fn the_movers_own_capsule_is_left_out_but_another_ones_is_not() {
    let mut sim = scene(1);
    add_floor(&mut sim);
    let mut alone = Mover::new(sim, [0.0, STAND, 0.0]);
    alone.walk_for(20, 0.0, PACE);
    assert!(
        (alone.z() - 20.0 * PACE).abs() < 0.01,
        "its own capsule blocked it at {}",
        alone.z()
    );

    let mut sim = scene(2);
    add_floor(&mut sim);
    sim.add_kinematic(
        &ColliderShape::Capsule {
            half_height: HALF_HEIGHT,
            radius: RADIUS,
        },
        [0.0, STAND, 2.0],
        [0.0; 3],
        0.8,
        LayerMask::ALL,
    )
    .expect("room for the other capsule");
    let mut blocked = Mover::new(sim, [0.0, STAND, 0.0]);
    blocked.walk_for(60, 0.0, PACE);
    assert!(
        blocked.z() < 2.0 - 2.0 * RADIUS + 0.01,
        "walked through another character to {}",
        blocked.z()
    );
}

#[test]
fn a_wall_on_a_layer_the_move_ignores_does_not_block_it() {
    let passable = LayerMask {
        memberships: 0b10,
        filter: 0b11,
    };
    let mut sim = scene(2);
    add_floor(&mut sim);
    sim.add_fixed(
        &ColliderShape::Cuboid {
            half_extents: [4.0, 2.0, 0.5],
        },
        [0.0, 2.0, 1.5],
        [0.0; 3],
        0.8,
        passable,
    )
    .expect("room for the wall");
    let mut mover = Mover::layered(
        sim,
        [0.0, STAND, 0.0],
        LayerMask {
            memberships: 0b11,
            filter: 0b01,
        },
    );
    mover.walk_for(40, 0.0, PACE);
    assert!(
        mover.z() > 1.5,
        "a layer it does not interact with stopped it at {}",
        mover.z()
    );
    assert!(
        mover.grounded,
        "the floor is on a layer it does interact with"
    );
}

// The papercut this controller exists to close: a capsule spawned exactly in
// contact with the floor must neither sink into it nor be pushed off it.
#[test]
fn a_capsule_spawned_exactly_on_the_floor_neither_sinks_nor_rises() {
    let mut sim = scene(1);
    add_floor(&mut sim);
    let mut mover = Mover::new(sim, [0.0, STAND, 0.0]);
    for tick in 0..240 {
        let moved = mover.walk(0.0, 0.0);
        assert!(moved.grounded, "it left the floor on tick {tick}");
        assert!(
            (mover.y() - STAND).abs() < 2.0e-3,
            "tick {tick}: standing at {} rather than {STAND}",
            mover.y()
        );
    }
    // The same holds while it walks: a mover crossing a flat floor holds its
    // height.
    mover.walk_for(120, PACE, 0.0);
    assert!(
        (mover.y() - STAND).abs() < 2.0e-3,
        "walked to {}",
        mover.y()
    );
}

// A capsule that starts a little inside the floor climbs back out of it
// rather than staying there.
#[test]
fn a_capsule_spawned_inside_the_floor_separates_from_it() {
    let mut sim = scene(1);
    add_floor(&mut sim);
    let mut mover = Mover::new(sim, [0.0, STAND - 0.1, 0.0]);
    let moved = mover.walk(0.0, 0.0);
    assert!(moved.grounded);
    assert!(
        mover.y() >= STAND - 2.0e-3,
        "still {} inside the floor",
        STAND - mover.y()
    );
}

#[test]
fn a_wedge_ends_the_move_instead_of_circling_inside_it() {
    let mut sim = scene(3);
    add_floor(&mut sim);
    // Two walls converging on the z axis, so a mover driven into them runs
    // out of room rather than out of deflections.
    add_box(
        &mut sim,
        [0.25, 2.0, 6.0],
        [0.9, 2.0, 5.0],
        [0.0, -12.0, 0.0],
    );
    add_box(
        &mut sim,
        [0.25, 2.0, 6.0],
        [-0.9, 2.0, 5.0],
        [0.0, 12.0, 0.0],
    );
    let mut mover = Mover::new(sim, [0.0, STAND, 0.0]);
    mover.walk_for(200, 0.0, PACE);
    let jammed = mover.center;
    mover.walk_for(20, 0.0, PACE);
    let moved = [
        mover.center[0] - jammed[0],
        mover.center[1] - jammed[1],
        mover.center[2] - jammed[2],
    ];
    let drift = (moved[0] * moved[0] + moved[1] * moved[1] + moved[2] * moved[2]).sqrt();
    assert!(drift < 0.01, "still creeping {drift} into the wedge");
    assert!(mover.x().abs() < 0.35, "squeezed sideways to {}", mover.x());
    assert!(mover.grounded, "it never left the floor");
}
