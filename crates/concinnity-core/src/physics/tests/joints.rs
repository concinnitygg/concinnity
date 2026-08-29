//! What a joint has to do once it is holding two bodies.
//!
//! The unit tests inside the crate check the arithmetic each row is built out
//! of. These check the joints themselves, against the scenes a joint is
//! actually judged by: a pendulum that keeps swinging, a hinge that stops at
//! its limit and reaches its motor's speed, an assembly that survives being
//! dropped, a slider that stays on its rail, and the housekeeping around all
//! of it -- islands, sleeping, and what a removed body owes the joints it was
//! in.

use crate::physics::{
    BodyHandle, ColliderShape, DynamicParams, GRAVITY, JointMotor, JointSpec, LayerMask, SimConfig,
    Simulation,
};

const TICK: f32 = 1.0 / 60.0;
const ARM: ColliderShape = ColliderShape::Cuboid {
    half_extents: [0.15, 0.15, 0.15],
};
const POST: ColliderShape = ColliderShape::Ball { radius: 0.05 };

fn params(damping: f32, gravity_scale: f32) -> DynamicParams {
    DynamicParams {
        mass: 1.0,
        friction: 0.5,
        restitution: 0.0,
        gravity_scale,
        linear_damping: damping,
    }
}

fn awake() -> SimConfig {
    SimConfig {
        allow_sleep: false,
        ..SimConfig::default()
    }
}

fn add_post(sim: &mut Simulation, pos: [f32; 3]) -> BodyHandle {
    sim.add_fixed(&POST, pos, [0.0; 3], 0.5, LayerMask::ALL)
        .expect("room for the post")
}

fn add_arm(sim: &mut Simulation, pos: [f32; 3], params: DynamicParams) -> BodyHandle {
    sim.add_dynamic(&ARM, pos, [0.0; 3], params, LayerMask::ALL)
        .expect("room for the arm")
}

fn step_for(sim: &mut Simulation, ticks: usize) {
    for _ in 0..ticks {
        sim.step(TICK);
    }
}

fn position(sim: &Simulation, handle: BodyHandle) -> [f32; 3] {
    sim.body_pose(handle).expect("a live body").0
}

/// How far a body is from a point.
fn reach(from: [f32; 3], to: [f32; 3]) -> f32 {
    ((from[0] - to[0]).powi(2) + (from[1] - to[1]).powi(2) + (from[2] - to[2]).powi(2)).sqrt()
}

/// An arm hung a unit under its post and thrown sideways.
///
/// Nothing else is in the scene, so the joint is the only thing acting on the
/// arm: whatever it does from here, the joint did.
fn thrown(spec: JointSpec, speed: f32) -> (Simulation, BodyHandle) {
    let mut sim = Simulation::new(awake(), 2);
    let post = add_post(&mut sim, [0.0, 0.0, 0.0]);
    let arm = add_arm(&mut sim, [0.0, -1.0, 0.0], params(0.0, 1.0));
    assert!(sim.add_joint(post, arm, [0.0; 3], [0.0, 1.0, 0.0], spec));
    sim.set_linear_velocity(arm, [speed, 0.0, 0.0]);
    (sim, arm)
}

/// The speeds a joint is handed in a game: a limb thrown by a hit, a prop
/// launched by one. The pendulum the tests above drop reaches six.
const THROWN: [f32; 4] = [5.0, 10.0, 20.0, 40.0];

/// A pendulum hinged about `+z` at the origin, one unit out along `+x`.
fn pendulum(spec: JointSpec, gravity_scale: f32) -> (Simulation, BodyHandle, BodyHandle) {
    let mut sim = Simulation::new(awake(), 2);
    let post = add_post(&mut sim, [0.0, 0.0, 0.0]);
    let arm = add_arm(&mut sim, [1.0, 0.0, 0.0], params(0.0, gravity_scale));
    assert!(sim.add_joint(post, arm, [0.0; 3], [-1.0, 0.0, 0.0], spec));
    (sim, post, arm)
}

fn hinge(limits: Option<[f32; 2]>, motor: Option<JointMotor>) -> JointSpec {
    JointSpec::Revolute {
        axis: [0.0, 0.0, 1.0],
        limits,
        motor,
    }
}

/// The angle a pendulum hangs at, measured from `+x` about `+z`.
fn swing(sim: &Simulation, arm: BodyHandle) -> f32 {
    let at = position(sim, arm);
    at[1].atan2(at[0])
}

// A pendulum is the joint's own test: it stays on its arc, and a solver that
// leaks energy through the constraint says so as a swing that grows or dies.
#[test]
fn a_hinged_pendulum_swings_on_its_arc_and_keeps_its_energy() {
    let (mut sim, _post, arm) = pendulum(hinge(None, None), 1.0);
    // Released horizontally, so the whole swing is the drop to the bottom.
    let swept = GRAVITY * 1.0;

    let mut bottom = 0.0f32;
    let mut far_side = 1.0f32;
    for _ in 0..600 {
        sim.step(TICK);
        let at = position(&sim, arm);
        let held = reach(at, [0.0, 0.0, 0.0]);
        assert!(
            (held - 1.0).abs() < 0.02,
            "the arm left its arc, hanging {held} out"
        );
        assert!(
            at[2].abs() < 0.02,
            "and wandered off the hinge plane: {at:?}"
        );
        bottom = bottom.min(at[1]);
        far_side = far_side.min(at[0]);
        let energy = sim.total_energy();
        assert!(
            energy.abs() < 0.06 * swept,
            "energy drifted to {energy} against a swing worth {swept}"
        );
    }
    assert!(bottom < -0.97, "it has to reach the bottom: {bottom}");
    assert!(
        far_side < -0.9,
        "and carry almost all the way up the other side: {far_side}"
    );
}

#[test]
fn a_hinge_limit_stops_the_swing_where_it_is_set() {
    let stop = -0.4f32;
    let (mut sim, _post, arm) = pendulum(hinge(Some([stop, 0.5]), None), 1.0);
    step_for(&mut sim, 300);

    let angle = swing(&sim, arm);
    assert!(
        (angle - stop).abs() < 0.05,
        "the arm should hang at the {stop} limit, hangs at {angle}"
    );
    // And it must not have been pushed the other way out of its range.
    for _ in 0..120 {
        sim.step(TICK);
        let angle = swing(&sim, arm);
        assert!(
            (stop - 0.06..=0.56).contains(&angle),
            "the arm left its range at {angle}"
        );
    }
}

// Limits the wrong way round name the same range, so the arm has to come to
// rest in the same place.
#[test]
fn a_hinge_limit_given_backwards_holds_the_same_range() {
    let forwards = {
        let (mut sim, _post, arm) = pendulum(hinge(Some([-0.4, 0.5]), None), 1.0);
        step_for(&mut sim, 300);
        swing(&sim, arm)
    };
    let backwards = {
        let (mut sim, _post, arm) = pendulum(hinge(Some([0.5, -0.4]), None), 1.0);
        step_for(&mut sim, 300);
        swing(&sim, arm)
    };
    assert!(
        (forwards - backwards).abs() < 1.0e-4,
        "{forwards} against {backwards}"
    );
}

#[test]
fn a_hinge_motor_reaches_the_speed_it_was_given() {
    let target = 2.0;
    let (mut sim, _post, arm) = pendulum(
        hinge(
            None,
            Some(JointMotor {
                target_velocity: target,
                max_force: 200.0,
            }),
        ),
        0.0,
    );
    step_for(&mut sim, 60);

    let spin = sim.angular_velocity(arm).expect("a live body")[2];
    assert!(
        (spin - target).abs() < 0.05,
        "the motor should be turning the arm at {target}, turns it at {spin}"
    );
    // A driven arm goes round, and it stays on its arc doing it.
    let quarter = 30;
    let before = swing(&sim, arm);
    step_for(&mut sim, quarter);
    let after = swing(&sim, arm);
    assert!(after > before, "it has to have turned: {before} -> {after}");
    let held = reach(position(&sim, arm), [0.0, 0.0, 0.0]);
    assert!((held - 1.0).abs() < 0.02, "hanging {held} out");
}

// The ceiling is the point of the motor: gravity asks 20 newton-metres of the
// arm held out horizontally, so a motor with one of them can never carry it
// over the top however long it is left running.
#[test]
fn a_hinge_motor_lifts_nothing_past_its_force_ceiling() {
    let highest = |max_force| {
        let (mut sim, _post, arm) = pendulum(
            hinge(
                None,
                Some(JointMotor {
                    target_velocity: 2.0,
                    max_force,
                }),
            ),
            1.0,
        );
        let mut highest = -1.0f32;
        for _ in 0..600 {
            sim.step(TICK);
            highest = highest.max(position(&sim, arm)[1]);
        }
        highest
    };

    let stalled = highest(1.0);
    assert!(
        stalled < 0.05,
        "a one-newton-metre motor cannot lift the arm above its hinge: {stalled}"
    );
    let driving = highest(200.0);
    assert!(
        driving > 0.9,
        "with force to spare it carries the arm over the top: {driving}"
    );
}

// A fixed joint has to survive the one thing that tests it: being landed on
// something. The two boxes must arrive at the floor still bolted together.
#[test]
fn a_fixed_joint_holds_two_bodies_through_a_landing() {
    let mut sim = Simulation::new(awake(), 3);
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
    let left = add_arm(&mut sim, [-0.2, 3.0, 0.0], params(0.0, 1.0));
    let right = add_arm(&mut sim, [0.2, 3.0, 0.0], params(0.0, 1.0));
    assert!(sim.add_joint(
        left,
        right,
        [0.2, 0.0, 0.0],
        [-0.2, 0.0, 0.0],
        JointSpec::Fixed
    ));

    step_for(&mut sim, 300);

    let (a, rotation_a) = sim.body_pose(left).expect("a live body");
    let (b, rotation_b) = sim.body_pose(right).expect("a live body");
    assert!(
        (reach(a, b) - 0.4).abs() < 0.01,
        "the two came apart: {a:?} {b:?}"
    );
    assert!(
        (a[1] - b[1]).abs() < 0.01,
        "the assembly tipped: {a:?} {b:?}"
    );
    for axis in 0..3 {
        assert!(
            (rotation_a[axis] - rotation_b[axis]).abs() < 2.0,
            "the two turned apart: {rotation_a:?} {rotation_b:?}"
        );
    }
    assert!((a[1] - 0.15).abs() < 0.05, "they rest on the floor: {a:?}");
}

// A ball and socket holds the anchors together and nothing else, so a spin put
// into the bob stays in it.
#[test]
fn a_spherical_joint_holds_the_anchor_and_leaves_the_spin_alone() {
    let (mut sim, _post, bob) = pendulum(JointSpec::Spherical, 0.0);
    // Spun about the line through its socket, which is the turn a ball and
    // socket genuinely leaves free: any other axis would carry the anchor off
    // the socket, and holding it is the joint's whole job.
    sim.set_angular_velocity(bob, [4.0, 0.0, 0.0]);
    step_for(&mut sim, 120);

    let spin = sim.angular_velocity(bob).expect("a live body");
    assert!(spin[0] > 3.5, "the bob has to keep spinning: {spin:?}");
    let held = reach(position(&sim, bob), [0.0, 0.0, 0.0]);
    assert!(
        (held - 1.0).abs() < 0.02,
        "and stay on its socket, hanging {held} out"
    );

    // Pushed hard sideways, it swings rather than separating.
    sim.apply_impulse(bob, [0.0, 0.0, 8.0]);
    step_for(&mut sim, 120);
    let held = reach(position(&sim, bob), [0.0, 0.0, 0.0]);
    assert!((held - 1.0).abs() < 0.03, "pulled off its socket to {held}");
}

/// A slider hanging below a fixed anchor, free to move along `+y`.
fn slider(
    limits: Option<[f32; 2]>,
    motor: Option<JointMotor>,
    gravity_scale: f32,
) -> (Simulation, BodyHandle) {
    let mut sim = Simulation::new(awake(), 2);
    let post = add_post(&mut sim, [0.0, 5.0, 0.0]);
    let carriage = add_arm(&mut sim, [0.0, 4.0, 0.0], params(0.0, gravity_scale));
    assert!(sim.add_joint(
        post,
        carriage,
        [0.0, -1.0, 0.0],
        [0.0; 3],
        JointSpec::Prismatic {
            axis: [0.0, 1.0, 0.0],
            limits,
            motor,
        },
    ));
    (sim, carriage)
}

#[test]
fn a_slider_stays_on_its_rail_and_stops_at_its_limit() {
    let (mut sim, carriage) = slider(Some([-1.0, 0.0]), None, 1.0);
    for _ in 0..600 {
        sim.step(TICK);
        let at = position(&sim, carriage);
        assert!(
            at[0].abs() < 0.01 && at[2].abs() < 0.01,
            "the carriage left the rail: {at:?}"
        );
        assert!(at[1] > 2.9, "and slid past the stop: {at:?}");
    }
    let at = position(&sim, carriage);
    assert!(
        (at[1] - 3.0).abs() < 0.03,
        "it should rest on the lower stop at y = 3, rests at {}",
        at[1]
    );
}

#[test]
fn a_slider_motor_drives_it_along_the_rail() {
    // Driven away from the anchor, so the run measures the motor rather than
    // the carriage arriving on top of the post it hangs from.
    let (mut sim, carriage) = slider(
        None,
        Some(JointMotor {
            target_velocity: -1.0,
            max_force: 200.0,
        }),
        0.0,
    );
    step_for(&mut sim, 60);

    let speed = sim.linear_velocity(carriage).expect("a live body")[1];
    assert!(
        (speed + 1.0).abs() < 0.05,
        "the motor should drive the carriage at 1/s, drives it at {speed}"
    );
    let at = position(&sim, carriage);
    assert!(
        (at[1] - 3.0).abs() < 0.1,
        "a second of it covers a unit: {at:?}"
    );
    assert!(
        at[0].abs() < 0.01 && at[2].abs() < 0.01,
        "off the rail: {at:?}"
    );
}

// Every test above starts a joint from rest and lets gravity do the work,
// which never asks the arm for more than about six units per second. A limb
// thrown by a hit is handed several times that at once, and it is the speed,
// not the joint, that decides whether the rows hold: solving through a mass
// block built before the arm swung overshoots rather than cancels, and past a
// few degrees of turn per substep the overshoot grows. It diverged inside a
// single step.
#[test]
fn a_thrown_arm_stays_on_its_arc_at_every_speed_a_hit_imparts() {
    for spec in [JointSpec::Spherical, hinge(None, None)] {
        for speed in THROWN {
            let (mut sim, arm) = thrown(spec, speed);
            let mut furthest = 0.0f32;
            for tick in 0..300 {
                sim.step(TICK);
                let at = position(&sim, arm);
                assert!(
                    at.iter().all(|v| v.is_finite()),
                    "thrown at {speed}/s the arm left the world on tick {tick}: {at:?}"
                );
                furthest = furthest.max(reach(at, [0.0; 3]));
            }
            assert!(
                furthest < 1.08,
                "thrown at {speed}/s the arm stretched a unit-long joint to {furthest}"
            );
            let held = reach(position(&sim, arm), [0.0; 3]);
            assert!(
                (held - 1.0).abs() < 0.05,
                "and settled {held} out at {speed}/s"
            );
        }
    }
}

// The sharpest statement of the same thing, and the one the divergence broke
// by seven orders of magnitude: a joint does no work. An arm pinned to a post
// can only trade the energy it was thrown with, so the total may fall -- the
// pin itself is inelastic, and the correction bleeds a little more -- and can
// never climb.
#[test]
fn a_thrown_joint_never_gains_energy() {
    for spec in [
        JointSpec::Spherical,
        hinge(None, None),
        hinge(Some([-1.2, 1.2]), None),
        JointSpec::Fixed,
    ] {
        for speed in THROWN {
            let (mut sim, _arm) = thrown(spec, speed);
            let launched = sim.total_energy();
            for tick in 0..300 {
                sim.step(TICK);
                let energy = sim.total_energy();
                assert!(
                    energy < launched + 0.01 * launched.abs(),
                    "{spec:?} thrown at {speed}/s held {energy} on tick {tick}, \
                     against the {launched} it was thrown with"
                );
            }
        }
    }
}

// A joint's own axis carries its own rows, solved through their own mass, and
// a thrown hinge is where that mass moves fastest.
//
// What is checked is where the hinge ends up rather than where the throw first
// carries it. The bound is held by a couple about the axis while the arm is
// also pinned a unit away, so the linear rows take back part of what the bound
// just applied, and an arm arriving at the stop at forty units per second
// stretches most of a unit further out before the two agree. It comes back:
// the arm is on its arc to a thousandth by the time it settles, and inside the
// range it was given.
#[test]
fn a_thrown_hinge_comes_back_onto_its_arc_and_rests_inside_its_limits() {
    let stop = 1.2f32;
    // The arm hangs below the post, so its arc starts a quarter turn round.
    let hanging = -core::f32::consts::FRAC_PI_2;
    for speed in THROWN {
        let (mut sim, arm) = thrown(hinge(Some([-stop, stop]), None), speed);
        let (mut furthest, mut settled_out, mut settled_round) = (0.0f32, 0.0f32, 0.0f32);
        for tick in 0..300 {
            sim.step(TICK);
            let at = position(&sim, arm);
            assert!(
                at.iter().all(|v| v.is_finite()),
                "thrown at {speed}/s the arm left the world on tick {tick}: {at:?}"
            );
            let held = reach(at, [0.0; 3]);
            furthest = furthest.max(held);
            if tick > 120 {
                settled_out = settled_out.max((held - 1.0).abs());
                settled_round = settled_round.max((at[1].atan2(at[0]) - hanging).abs());
            }
        }
        assert!(
            furthest < 2.0,
            "thrown at {speed}/s the arm reached {furthest} out on a unit-long joint"
        );
        assert!(
            settled_out < 0.01,
            "and two seconds on it is still {settled_out} off its arc at {speed}/s"
        );
        assert!(
            settled_round < stop + 0.06,
            "and still reaching {settled_round} against a {stop} stop at {speed}/s"
        );
    }
}

// A slider is the one kind a throw does not put a stale mass in front of: its
// carriage rides its own centre, so there is no lever to turn and the block
// stays what it was. It is here because nothing else in this file hands a
// prismatic joint a speed, and a rail is worth no less than an arc.
#[test]
fn a_thrown_slider_is_pulled_back_onto_its_rail() {
    for speed in THROWN {
        let mut sim = Simulation::new(awake(), 2);
        let post = add_post(&mut sim, [0.0, 5.0, 0.0]);
        let carriage = add_arm(&mut sim, [0.0, 4.0, 0.0], params(0.0, 0.0));
        assert!(sim.add_joint(
            post,
            carriage,
            [0.0, -1.0, 0.0],
            [0.0; 3],
            JointSpec::Prismatic {
                axis: [0.0, 1.0, 0.0],
                limits: Some([-1.0, 1.0]),
                motor: None,
            },
        ));
        // Thrown across the rail, which is the direction the joint holds.
        sim.set_linear_velocity(carriage, [speed, 0.0, 0.0]);
        let mut furthest = 0.0f32;
        for _ in 0..300 {
            sim.step(TICK);
            let at = position(&sim, carriage);
            assert!(at.iter().all(|v| v.is_finite()), "{at:?} at {speed}/s");
            furthest = furthest.max((at[0] * at[0] + at[2] * at[2]).sqrt());
        }
        assert!(
            furthest < 0.02,
            "thrown at {speed}/s the carriage reached {furthest} off its rail"
        );
        let at = position(&sim, carriage);
        assert!(
            at[0].abs() < 0.001 && at[2].abs() < 0.001,
            "and ended off it at {speed}/s: {at:?}"
        );
    }
}

// The contract the driver relies on: a body takes its joints with it.
#[test]
fn removing_a_jointed_body_takes_the_joint_with_it() {
    let (mut sim, post, arm) = pendulum(hinge(None, None), 1.0);
    assert_eq!(sim.joint_count(), 1);
    step_for(&mut sim, 30);

    assert!(sim.remove_body(post));
    assert_eq!(sim.joint_count(), 0, "the joint outlived its body");

    // With nothing holding it, the arm falls freely.
    let before = position(&sim, arm)[1];
    step_for(&mut sim, 60);
    let after = position(&sim, arm)[1];
    assert!(
        before - after > 5.0,
        "the arm should be in free fall: {before} -> {after}"
    );
    // And the slot the post left can be filled without inheriting the joint.
    let fresh = add_post(&mut sim, [0.0, 0.0, 0.0]);
    step_for(&mut sim, 30);
    assert_eq!(sim.joint_count(), 0);
    assert!(sim.body_pose(fresh).is_some());
}

#[test]
fn degenerate_joints_are_repaired_or_refused_rather_than_breaking_the_step() {
    let mut sim = Simulation::new(awake(), 3);
    let post = add_post(&mut sim, [0.0, 0.0, 0.0]);
    let arm = add_arm(&mut sim, [1.0, 0.0, 0.0], params(0.0, 1.0));

    assert!(
        !sim.add_joint(arm, arm, [0.0; 3], [0.0; 3], JointSpec::Fixed),
        "a body cannot be joined to itself"
    );
    assert!(
        !sim.add_joint(
            post,
            arm,
            [f32::NAN, 0.0, 0.0],
            [0.0; 3],
            JointSpec::Spherical
        ),
        "an anchor that is nowhere names no joint"
    );
    let gone = add_post(&mut sim, [0.0, 2.0, 0.0]);
    assert!(sim.remove_body(gone));
    assert!(
        !sim.add_joint(post, gone, [0.0; 3], [0.0; 3], JointSpec::Fixed),
        "a joint needs two live bodies"
    );
    assert_eq!(sim.joint_count(), 0);

    // A hinge with no axis and its limits inside out is still a hinge.
    assert!(sim.add_joint(
        post,
        arm,
        [0.0; 3],
        [-1.0, 0.0, 0.0],
        JointSpec::Revolute {
            axis: [0.0; 3],
            limits: Some([1.0, -1.0]),
            motor: Some(JointMotor {
                target_velocity: 1.0,
                max_force: 0.0,
            }),
        },
    ));
    step_for(&mut sim, 120);
    let at = position(&sim, arm);
    assert!(at.iter().all(|v| v.is_finite()), "{at:?}");
    assert!(
        (reach(at, [0.0, 0.0, 0.0]) - 1.0).abs() < 0.05,
        "the repaired hinge still holds: {at:?}"
    );
}

// Two bodies a joint holds are one island: neither settles while the other is
// moving, and waking one wakes the other.
#[test]
fn a_jointed_pair_settles_and_wakes_as_one_island() {
    let mut sim = Simulation::with_capacity(3);
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
    // Far enough apart that nothing but the joint connects them.
    let near = add_arm(&mut sim, [0.0, 0.15, 0.0], params(0.0, 1.0));
    let far = add_arm(&mut sim, [4.0, 0.15, 0.0], params(0.0, 1.0));
    assert!(sim.add_joint(near, far, [4.0, 0.0, 0.0], [0.0; 3], JointSpec::Fixed));

    step_for(&mut sim, 600);
    assert_eq!(sim.is_sleeping(near), Some(true), "the pair has to settle");
    assert_eq!(sim.is_sleeping(far), Some(true));

    sim.apply_impulse(near, [0.0, 4.0, 0.0]);
    step_for(&mut sim, 2);
    assert_eq!(
        sim.is_sleeping(far),
        Some(false),
        "the far end shares the island that was disturbed"
    );
}

// A motor is doing work whether or not it is getting anywhere, so the island
// it drives must never be allowed to stop being simulated.
#[test]
fn a_motor_driven_island_never_falls_asleep() {
    let mut sim = Simulation::with_capacity(2);
    let post = add_post(&mut sim, [0.0, 0.0, 0.0]);
    let arm = add_arm(&mut sim, [1.0, 0.0, 0.0], params(0.0, 1.0));
    // A ceiling well under what the arm's weight asks for: the motor stalls,
    // so nothing is moving and only the motor keeps the island awake.
    assert!(sim.add_joint(
        post,
        arm,
        [0.0; 3],
        [-1.0, 0.0, 0.0],
        hinge(
            None,
            Some(JointMotor {
                target_velocity: 4.0,
                max_force: 1.0,
            }),
        ),
    ));

    step_for(&mut sim, 900);
    assert_eq!(
        sim.is_sleeping(arm),
        Some(false),
        "a driven arm must not be allowed to settle"
    );

    // The same scene with the motor switched off does settle, which is what
    // makes the check above about the motor rather than about the arm.
    let mut idle = Simulation::with_capacity(2);
    let post = add_post(&mut idle, [0.0, 0.0, 0.0]);
    let arm = add_arm(&mut idle, [1.0, 0.0, 0.0], params(1.5, 1.0));
    assert!(idle.add_joint(post, arm, [0.0; 3], [-1.0, 0.0, 0.0], hinge(None, None)));
    step_for(&mut idle, 900);
    assert_eq!(idle.is_sleeping(arm), Some(true));
}

// Everything above is worth nothing if the answer depends on the day.
#[test]
fn a_jointed_scene_runs_identically_twice() {
    let run = || {
        let (mut sim, _post, arm) = pendulum(
            hinge(
                Some([-1.2, 1.2]),
                Some(JointMotor {
                    target_velocity: 3.0,
                    max_force: 4.0,
                }),
            ),
            1.0,
        );
        step_for(&mut sim, 300);
        let at = position(&sim, arm);
        [at[0].to_bits(), at[1].to_bits(), at[2].to_bits()]
    };
    assert_eq!(run(), run());
}
