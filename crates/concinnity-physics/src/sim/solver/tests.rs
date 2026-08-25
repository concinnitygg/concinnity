use super::contact::effective_mass;
use super::*;
use crate::Inline;
use crate::sim::body::Body;
use crate::sim::contact::ManifoldPoint;
use crate::sim::math::{Vec3, vec3};
use crate::{ColliderShape, DynamicParams, LayerMask};

const TICK: f32 = 1.0 / 60.0;

fn config() -> SimConfig {
    SimConfig::default()
}

/// A fan-out that reports several workers and runs their chunks back to front.
///
/// Nothing about a solve may depend on which chunk went first, so a visit order
/// no scheduler would ever produce has to leave the world exactly where the
/// ordinary one does.
struct Reversed(usize);

impl Fanout for Reversed {
    fn workers(&self) -> usize {
        self.0
    }

    fn for_each<T, F>(&self, items: &mut [T], body: F)
    where
        T: Send,
        F: Fn(&mut T) + Send + Sync,
    {
        for item in items.iter_mut().rev() {
            body(item);
        }
    }
}

/// One step over the given manifolds, on the calling thread.
fn run(solver: &mut Solver, capacity: usize, manifolds: &mut [Manifold]) {
    run_with(solver, capacity, manifolds, &mut [], &Inline, 1);
}

fn run_with(
    solver: &mut Solver,
    capacity: usize,
    manifolds: &mut [Manifold],
    joints: &mut [Joint],
    fanout: &impl Fanout,
    workers: usize,
) {
    let mut islands = Islands::with_capacity(capacity);
    solver.run(
        Work {
            manifolds,
            joints,
            islands: &mut islands,
            config: &config(),
            dt: TICK,
        },
        fanout,
        workers,
    );
}

fn unit_cube(position: Vec3, damping: f32) -> Body {
    Body::dynamic(
        ColliderShape::Cuboid {
            half_extents: [0.5, 0.5, 0.5],
        },
        position,
        crate::sim::math::Quat::IDENTITY,
        DynamicParams {
            mass: 1.0,
            friction: 0.5,
            restitution: 0.0,
            gravity_scale: 1.0,
            linear_damping: damping,
        },
        LayerMask::ALL,
    )
}

fn ground(position: Vec3) -> Body {
    Body::fixed(
        ColliderShape::Cuboid {
            half_extents: [10.0, 0.5, 10.0],
        },
        position,
        crate::sim::math::Quat::IDENTITY,
        0.8,
        LayerMask::ALL,
    )
}

/// A four-point contact patch under a unit cube resting on `y = 0`.
fn resting_manifold(a: u32, b: u32, separation: f32) -> Manifold {
    let mut manifold = Manifold::new(a, b);
    manifold.normal = Vec3::Y;
    manifold.friction = 0.5;
    manifold.restitution = 0.0;
    for (id, (x, z)) in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
        .into_iter()
        .enumerate()
    {
        manifold.push(ManifoldPoint {
            point: vec3(x, 0.0, z),
            separation,
            id: id as u32,
            ..Default::default()
        });
    }
    manifold
}

#[test]
fn a_free_body_gains_a_steps_worth_of_gravity() {
    let mut solver = Solver::with_capacity(1);
    solver.begin();
    solver.set_body(
        0,
        SolverBody::from_body(&unit_cube(vec3(0.0, 5.0, 0.0), 0.0)),
    );
    run(&mut solver, 1, &mut []);

    let body = solver.body(0);
    let expected = -config().gravity * TICK;
    assert!(
        (body.linear_velocity.y - expected).abs() < 1.0e-5,
        "{:?} should be near {expected}",
        body.linear_velocity
    );
    // Symplectic Euler integrates the new velocity, one substep at a time.
    assert!(
        body.position.y < 5.0 && body.position.y > 4.99,
        "{:?}",
        body.position
    );
}

#[test]
fn damping_bleeds_speed_off_without_reversing_it() {
    let mut solver = Solver::with_capacity(1);
    let mut body = unit_cube(vec3(0.0, 5.0, 0.0), 20.0);
    body.gravity_scale = 0.0;
    body.linear_velocity = vec3(10.0, 0.0, 0.0);
    solver.begin();
    solver.set_body(0, SolverBody::from_body(&body));
    // A damping coefficient far past 1/h would reverse an explicit decay.
    run(&mut solver, 1, &mut []);
    let speed = solver.body(0).linear_velocity.x;
    assert!(speed > 0.0 && speed < 10.0, "{speed}");
}

#[test]
fn an_immovable_body_is_not_moved_by_the_step() {
    let mut solver = Solver::with_capacity(1);
    solver.begin();
    solver.set_body(0, SolverBody::from_body(&ground(vec3(0.0, -0.5, 0.0))));
    run(&mut solver, 1, &mut []);
    assert_eq!(solver.body(0).linear_velocity, Vec3::ZERO);
    assert_eq!(solver.body(0).position, vec3(0.0, -0.5, 0.0));
    assert!(
        solver.is_idle(),
        "a world of immovable bodies moves nothing"
    );
    assert_eq!(solver.island_count(), 0);
}

// The whole point of the contact solve: a falling body meeting a floor
// stops rather than continuing through it.
#[test]
fn a_contact_stops_a_body_falling_into_an_immovable_one() {
    let mut solver = Solver::with_capacity(2);
    let mut cube = unit_cube(vec3(0.0, 0.5, 0.0), 0.0);
    cube.linear_velocity = vec3(0.0, -5.0, 0.0);
    solver.begin();
    solver.set_body(0, SolverBody::from_body(&ground(vec3(0.0, -0.5, 0.0))));
    solver.set_body(1, SolverBody::from_body(&cube));

    let mut manifolds = [resting_manifold(0, 1, 0.0)];
    run(&mut solver, 2, &mut manifolds);

    let landed = solver.body(1);
    assert!(
        landed.linear_velocity.y > -0.2,
        "the contact must arrest the fall, left {:?}",
        landed.linear_velocity
    );
    assert!(
        landed.position.y > 0.49,
        "and not let it through: {:?}",
        landed.position
    );
    assert!(
        manifolds[0].points().iter().any(|p| p.normal_impulse > 0.0),
        "the contact must record what it took to stop the body"
    );
}

// Warm starting: the impulses a step ends with are what the next step
// begins from, which is what a resting stack lives on.
#[test]
fn a_resting_contact_stores_the_impulse_that_held_it() {
    let mut solver = Solver::with_capacity(2);
    let mut manifolds = [resting_manifold(0, 1, -0.001)];
    for _ in 0..4 {
        solver.begin();
        solver.set_body(0, SolverBody::from_body(&ground(vec3(0.0, -0.5, 0.0))));
        let mut cube = unit_cube(vec3(0.0, 0.499, 0.0), 0.0);
        cube.linear_velocity = solver.body(1).linear_velocity;
        solver.set_body(1, SolverBody::from_body(&cube));
        run(&mut solver, 2, &mut manifolds);
    }
    let held: f32 = manifolds[0].points().iter().map(|p| p.normal_impulse).sum();
    // Impulses are what one substep took, not one step: the next substep
    // warm starts from them and adds its own share of the weight.
    let substep_weight = config().gravity * TICK / config().substep_count() as f32;
    assert!(
        (held - substep_weight).abs() < substep_weight * 0.5,
        "the patch should carry about {substep_weight:.4}, carries {held:.4}"
    );
}

// The trap this array exists for: a body the gather left out keeps the
// state of whatever step last took it. Reading that as current would put
// a settled world back to work solving contacts that cannot move.
#[test]
fn a_body_left_out_of_the_gather_builds_no_constraint() {
    let mut solver = Solver::with_capacity(2);
    let mut manifolds = [resting_manifold(0, 1, -0.001)];

    // First step: both bodies taken, one of them moving.
    solver.begin();
    solver.set_body(0, SolverBody::from_body(&ground(vec3(0.0, -0.5, 0.0))));
    solver.set_body(
        1,
        SolverBody::from_body(&unit_cube(vec3(0.0, 0.499, 0.0), 0.0)),
    );
    run(&mut solver, 2, &mut manifolds);
    assert_eq!(solver.constraint_count(), 1, "the moving pair is solved");

    // Second step: neither body is taken, standing for a pair that went
    // to sleep. The stale state must not resurrect the constraint.
    solver.begin();
    run(&mut solver, 2, &mut manifolds);
    assert_eq!(
        solver.constraint_count(),
        0,
        "a pair nothing gathered must not be solved"
    );
    assert!(solver.is_idle());
}

// Friction is a disc: sliding diagonally must be no easier than sliding
// along an axis, which a per-tangent clamp would allow.
#[test]
fn friction_limits_a_diagonal_slide_as_hard_as_a_straight_one() {
    let slide = |velocity: Vec3| -> f32 {
        let mut solver = Solver::with_capacity(2);
        let mut cube = unit_cube(vec3(0.0, 0.499, 0.0), 0.0);
        cube.linear_velocity = velocity;
        solver.begin();
        solver.set_body(0, SolverBody::from_body(&ground(vec3(0.0, -0.5, 0.0))));
        solver.set_body(1, SolverBody::from_body(&cube));
        let mut manifolds = [resting_manifold(0, 1, -0.001)];
        run(&mut solver, 2, &mut manifolds);
        let left = solver.body(1).linear_velocity;
        vec3(left.x, 0.0, left.z).length()
    };
    let straight = slide(vec3(2.0, 0.0, 0.0));
    let diagonal = slide(vec3(2.0, 0.0, 2.0).normalize_or_zero() * 2.0);
    assert!(
        (straight - diagonal).abs() < 0.02,
        "straight {straight:.4} against diagonal {diagonal:.4}"
    );
    assert!(
        straight < 2.0,
        "friction must take something off: {straight}"
    );
}

// Restitution bounces off the speed measured before the step, and only
// above the threshold, or a settling body never settles.
#[test]
fn restitution_bounces_a_fast_approach_and_ignores_a_slow_one() {
    let bounce = |approach: f32, restitution: f32| -> f32 {
        let mut solver = Solver::with_capacity(2);
        let mut cube = unit_cube(vec3(0.0, 0.5, 0.0), 0.0);
        cube.gravity_scale = 0.0;
        cube.linear_velocity = vec3(0.0, -approach, 0.0);
        solver.begin();
        solver.set_body(0, SolverBody::from_body(&ground(vec3(0.0, -0.5, 0.0))));
        solver.set_body(1, SolverBody::from_body(&cube));
        let mut manifolds = [resting_manifold(0, 1, 0.0)];
        manifolds[0].restitution = restitution;
        run(&mut solver, 2, &mut manifolds);
        solver.body(1).linear_velocity.y
    };
    let fast = bounce(6.0, 0.8);
    assert!(fast > 3.0, "a fast approach must bounce back: {fast}");
    assert!(fast < 6.0, "and never faster than it arrived: {fast}");
    assert_eq!(bounce(6.0, 0.0), bounce(6.0, 0.0));
    assert!(bounce(6.0, 0.0) < 0.5, "no restitution, no bounce");
    // Below the threshold nothing bounces, whatever the material says.
    let slow = bounce(config().restitution_threshold * 0.5, 0.8);
    assert!(slow < 0.2, "a slow approach must not bounce: {slow}");
}

#[test]
fn effective_mass_falls_between_the_two_bodies_own() {
    let heavy = SolverBody::from_body(&unit_cube(Vec3::ZERO, 0.0));
    let fixed = SolverBody::from_body(&ground(Vec3::ZERO));
    // Against something immovable, only the moving body's mass counts.
    let against_wall = effective_mass(&fixed, &heavy, Vec3::ZERO, Vec3::ZERO, Vec3::Y);
    assert!((against_wall - heavy.inv_mass.recip()).abs() < 1.0e-4);
    // Against another of the same mass, the pair is half as easy to move.
    let against_peer = effective_mass(&heavy, &heavy, Vec3::ZERO, Vec3::ZERO, Vec3::Y);
    assert!((against_peer - against_wall * 0.5).abs() < 1.0e-4);
    // Two immovable bodies have no mass to share.
    assert_eq!(
        effective_mass(&fixed, &fixed, Vec3::ZERO, Vec3::ZERO, Vec3::Y),
        0.0
    );
    // A lever arm makes the contact harder to move than the mass alone.
    let levered = effective_mass(&fixed, &heavy, Vec3::ZERO, Vec3::X, Vec3::Y);
    assert!(levered < against_wall, "{levered} vs {against_wall}");
}

/// Enough separate stacks that a cut into four is a real cut rather than one
/// island per chunk with the rest empty.
const SPLITTABLE_STACKS: usize = 64;

/// A floor with `stacks` separate two-body columns resting on it, which is
/// the shape the island split is built for.
fn columns(stacks: usize) -> (Solver, alloc::vec::Vec<Manifold>, usize) {
    let capacity = stacks * 2 + 1;
    let mut solver = Solver::with_capacity(capacity);
    solver.begin();
    solver.set_body(0, SolverBody::from_body(&ground(vec3(0.0, -0.5, 0.0))));
    let mut manifolds = alloc::vec::Vec::new();
    for stack in 0..stacks {
        let x = stack as f32 * 4.0;
        let (lower, upper) = (1 + stack as u32 * 2, 2 + stack as u32 * 2);
        solver.set_body(
            lower,
            SolverBody::from_body(&unit_cube(vec3(x, 0.499, 0.0), 0.0)),
        );
        solver.set_body(
            upper,
            SolverBody::from_body(&unit_cube(vec3(x, 1.498, 0.0), 0.0)),
        );
        manifolds.push(resting_manifold(0, lower, -0.001));
        manifolds.push(resting_manifold(lower, upper, -0.001));
    }
    (solver, manifolds, capacity)
}

// The claim the whole split rests on: separate stacks are separate islands
// even though every one of them leans on the same floor.
#[test]
fn stacks_sharing_a_floor_solve_as_separate_islands() {
    let (mut solver, mut manifolds, capacity) = columns(SPLITTABLE_STACKS);
    run_with(
        &mut solver,
        capacity,
        &mut manifolds,
        &mut [],
        &Reversed(4),
        4,
    );
    assert_eq!(solver.island_count(), SPLITTABLE_STACKS);
    assert_eq!(solver.chunk_count(), 4, "the islands share four workers");
    assert_eq!(solver.constraint_count(), SPLITTABLE_STACKS * 2);
}

// Chunk order is not part of the answer: the same step run back to front
// has to land on the same bits, or the solve is not splittable.
#[test]
fn chunk_order_does_not_change_the_result() {
    let poses = |fanout: &dyn Fn(&mut Solver, usize, &mut [Manifold])| {
        let (mut solver, mut manifolds, capacity) = columns(SPLITTABLE_STACKS);
        for _ in 0..8 {
            fanout(&mut solver, capacity, &mut manifolds);
            let taken: alloc::vec::Vec<SolverBody> =
                (0..capacity).map(|s| *solver.body(s as u32)).collect();
            solver.begin();
            for (slot, body) in taken.into_iter().enumerate() {
                solver.set_body(slot as u32, body);
            }
        }
        (0..capacity)
            .map(|s| {
                let body = solver.body(s as u32);
                (
                    body.position.x.to_bits(),
                    body.position.y.to_bits(),
                    body.position.z.to_bits(),
                    body.linear_velocity.y.to_bits(),
                )
            })
            .collect::<alloc::vec::Vec<_>>()
    };
    let serial = poses(&|solver, capacity, manifolds| {
        run_with(solver, capacity, manifolds, &mut [], &Inline, 1)
    });
    let split = poses(&|solver, capacity, manifolds| {
        run_with(solver, capacity, manifolds, &mut [], &Reversed(4), 4)
    });
    assert_eq!(serial, split, "the split changed what the solve produced");
}

// A world that is one island cannot be split, and must still solve.
// A world that is one tall stack is one island, so no number of workers
// splits it. The stage the split cannot reach, stated as a test.
#[test]
fn one_island_takes_one_chunk_however_many_workers_are_offered() {
    const LEVELS: u32 = 256;
    let mut solver = Solver::with_capacity(LEVELS as usize + 1);
    solver.begin();
    solver.set_body(0, SolverBody::from_body(&ground(vec3(0.0, -0.5, 0.0))));
    let mut manifolds = alloc::vec::Vec::new();
    for level in 0..LEVELS {
        solver.set_body(
            level + 1,
            SolverBody::from_body(&unit_cube(vec3(0.0, 0.499 + level as f32, 0.0), 0.0)),
        );
        manifolds.push(resting_manifold(level, level + 1, -0.001));
    }
    run_with(
        &mut solver,
        LEVELS as usize + 1,
        &mut manifolds,
        &mut [],
        &Reversed(8),
        8,
    );
    assert_eq!(solver.island_count(), 1);
    assert_eq!(solver.chunk_count(), 1);
}
