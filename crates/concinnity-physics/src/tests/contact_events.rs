//! What counts as an impact.
//!
//! Every body at rest is in contact with something, so the useful question is
//! not "did these touch" but "did this land". The simulation answers it with a
//! force gate: the impulse a pair's contact converged on, over the step it was
//! spread across, against a threshold the caller sets. These tests hold that
//! gate to both of its edges -- a drop reports, a rest does not -- and check
//! that what is reported describes the collision it came from.
//!
//! The impulse figures below are checked against momentum rather than against
//! a recorded number. A body falling `h` under `g` arrives at `sqrt(2gh)`, and
//! stopping it costs its mass times that; a solver whose reported impulse is
//! not near that figure is reporting something other than what it did.

use crate::{ColliderShape, ContactHit, DynamicParams, LayerMask, SimConfig, Simulation};
use alloc::vec;
use alloc::vec::Vec;

const TICK: f32 = 1.0 / 60.0;
const GRAVITY: f32 = crate::GRAVITY;
const BALL: ColliderShape = ColliderShape::Ball { radius: 0.5 };

fn params(mass: f32) -> DynamicParams {
    DynamicParams {
        mass,
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

/// Step `ticks` times, collecting every hit recorded along the way.
fn run(sim: &mut Simulation, ticks: usize) -> Vec<ContactHit> {
    let mut collected = Vec::new();
    let mut drained = Vec::new();
    for _ in 0..ticks {
        sim.step(TICK);
        sim.drain_contact_hits_into(&mut drained);
        // One step's hits leave in slot order; across steps they are in the
        // order the collisions happened.
        assert!(
            drained
                .windows(2)
                .all(|w| (w[0].a.index(), w[0].b.index()) < (w[1].a.index(), w[1].b.index())),
            "a step reported out of slot order"
        );
        collected.extend_from_slice(&drained);
    }
    assert_eq!(sim.contact_hit_overflows(), 0, "a hit went unreported");
    collected
}

/// A ball dropped from `height` onto a floor, with the impulse gate set to
/// `min_impulse`.
fn drop_ball(min_impulse: f32, height: f32, mass: f32) -> (Simulation, Vec<ContactHit>) {
    let mut sim = sim(2);
    sim.set_contact_min_impulse(min_impulse, TICK);
    add_floor(&mut sim);
    sim.add_dynamic(
        &BALL,
        [0.0, height + 0.5, 0.0],
        [0.0; 3],
        params(mass),
        LayerMask::ALL,
    )
    .expect("room for the ball");
    let hits = run(&mut sim, 180);
    (sim, hits)
}

#[test]
fn a_landing_reports_a_hit_and_the_rest_that_follows_does_not() {
    let (mut sim, hits) = drop_ball(1.0, 5.0, 1.0);
    assert!(!hits.is_empty(), "the landing has to be reported");
    assert!(hits.iter().all(|h| h.a.index() == 0 && h.b.index() == 1));

    // Settled: a long window afterwards has to stay silent.
    assert!(run(&mut sim, 600).is_empty(), "resting contact reported");
}

// The normal is what a caller turns into a direction to spray sparks along,
// and the point is where to put them.
#[test]
fn the_hit_describes_the_collision_it_came_from() {
    let (_, hits) = drop_ball(1.0, 5.0, 1.0);
    let landing = hits.first().expect("the landing is reported");
    // The floor is the lower slot, so the normal points up out of it.
    assert!(landing.normal[1] > 0.99, "{:?}", landing.normal);
    // The point sits midway between the two surfaces, so it is a little under
    // the floor by however far the ball reached through it that step.
    assert!(landing.point[1].abs() < 0.2, "{:?}", landing.point);
    assert!(
        landing.point[0].abs() < 0.05 && landing.point[2].abs() < 0.05,
        "{:?}",
        landing.point
    );
}

// A solver reporting an impulse that does not match the momentum it removed
// is reporting a number rather than a measurement.
#[test]
fn the_impulse_is_the_momentum_the_landing_took_out() {
    for (mass, height) in [(1.0f32, 5.0f32), (4.0, 5.0), (1.0, 1.25)] {
        let (_, hits) = drop_ball(0.5, height, mass);
        let total: f32 = hits.iter().map(|h| h.impulse).sum();
        let expected = mass * (2.0 * GRAVITY * height).sqrt();
        assert!(
            total > expected * 0.5 && total < expected * 2.0,
            "mass {mass} from {height}: reported {total}, momentum {expected}"
        );
    }
}

// The gate is the whole feature: raising it has to silence a landing that a
// lower one reports.
#[test]
fn raising_the_threshold_silences_a_weaker_landing() {
    let (_, gentle) = drop_ball(0.2, 0.2, 1.0);
    assert!(!gentle.is_empty(), "a short drop is reported at 0.2");

    let (_, guarded) = drop_ball(5.0, 0.2, 1.0);
    assert!(guarded.is_empty(), "{guarded:?}");

    // The same threshold still lets a real fall through.
    let (_, hard) = drop_ball(5.0, 5.0, 1.0);
    assert!(!hard.is_empty(), "a five metre drop passes 5");
}

// Only a freely simulated body is a source. Two walls touching, or a driven
// platform pressed against one, are not collisions anybody wants told about.
#[test]
fn a_pair_with_nothing_dynamic_in_it_never_reports() {
    let mut sim = sim(4);
    sim.set_contact_min_impulse(0.0, TICK);
    add_floor(&mut sim);
    sim.add_fixed(&BALL, [0.0, 0.4, 0.0], [0.0; 3], 0.8, LayerMask::ALL)
        .expect("room for the wall");
    let platform = sim
        .add_kinematic(&BALL, [4.0, 0.4, 0.0], [0.0; 3], 0.8, LayerMask::ALL)
        .expect("room for the platform");
    for _ in 0..30 {
        sim.set_kinematic_translation(platform, [4.0, 0.2, 0.0]);
        sim.step(TICK);
    }
    let mut out = Vec::new();
    sim.drain_contact_hits_into(&mut out);
    assert!(out.is_empty(), "{out:?}");
}

// A driven body pressing on a dynamic one is a real collision: the dynamic
// side makes the pair a source.
#[test]
fn a_driven_body_pressing_on_a_dynamic_one_reports() {
    let mut sim = sim(3);
    sim.set_contact_min_impulse(0.5, TICK);
    add_floor(&mut sim);
    let ram = sim
        .add_kinematic(
            &ColliderShape::Cuboid {
                half_extents: [1.0, 1.0, 1.0],
            },
            [0.0, 4.0, 0.0],
            [0.0; 3],
            0.8,
            LayerMask::ALL,
        )
        .expect("room for the ram");
    sim.add_dynamic(
        &BALL,
        [0.0, 0.5, 0.0],
        [0.0; 3],
        params(1.0),
        LayerMask::ALL,
    )
    .expect("room for the ball");

    let mut hits = Vec::new();
    let mut collected = Vec::new();
    for tick in 0..90 {
        sim.set_kinematic_translation(ram, [0.0, 4.0 - tick as f32 * 0.06, 0.0]);
        sim.step(TICK);
        sim.drain_contact_hits_into(&mut hits);
        collected.extend_from_slice(&hits);
    }
    assert!(
        collected
            .iter()
            .any(|h| h.a.index() == 1 || h.b.index() == 1),
        "the ram crushing the ball has to report: {collected:?}"
    );
}

// Terrain hands one pair a manifold per triangle, and a caller wants to hear
// that the body landed rather than which triangles it landed across.
#[test]
fn a_landing_on_terrain_reports_one_hit_per_step() {
    let mut sim = sim(2);
    sim.set_contact_min_impulse(1.0, TICK);
    sim.add_heightfield(
        5,
        5,
        vec![0.0; 25],
        [20.0, 1.0, 20.0],
        [0.0; 3],
        LayerMask::ALL,
    )
    .expect("room for the terrain");
    sim.add_dynamic(
        &BALL,
        [1.0, 5.5, -2.0],
        [0.0; 3],
        params(1.0),
        LayerMask::ALL,
    )
    .expect("room for the ball");

    let mut hits = Vec::new();
    let mut most = 0;
    for _ in 0..180 {
        sim.step(TICK);
        sim.drain_contact_hits_into(&mut hits);
        most = most.max(hits.len());
    }
    assert_eq!(most, 1, "one pair, one hit per step");
}

// Hits leave in slot order however the bodies are laid out, so two runs of
// the same scene report the same sequence.
#[test]
fn two_identical_runs_report_the_same_hits() {
    let once = || {
        let mut sim = sim(6);
        sim.set_contact_min_impulse(0.5, TICK);
        add_floor(&mut sim);
        for x in [-4.0f32, 0.0, 4.0, 8.0] {
            sim.add_dynamic(
                &BALL,
                [x, 4.0 + x.abs() * 0.25, 0.0],
                [0.0; 3],
                params(1.0),
                LayerMask::ALL,
            )
            .expect("room for the ball");
        }
        run(&mut sim, 240)
            .into_iter()
            .map(|h| {
                (
                    h.a.index(),
                    h.b.index(),
                    h.impulse.to_bits(),
                    h.point.map(f32::to_bits),
                    h.normal.map(f32::to_bits),
                )
            })
            .collect::<Vec<_>>()
    };
    let first = once();
    assert!(!first.is_empty(), "the scene has to record something");
    assert_eq!(first, once());
}

// A caller running several fixed ticks per frame drains once at the end of
// them, so the queue has to hold what happened rather than what the last step
// happened to report.
#[test]
fn hits_wait_in_the_queue_until_they_are_drained() {
    let mut sim = sim(3);
    sim.set_contact_min_impulse(0.5, TICK);
    add_floor(&mut sim);
    for x in [0.0f32, 4.0] {
        sim.add_dynamic(
            &BALL,
            [x, 0.6 + x * 0.05, 0.0],
            [0.0; 3],
            params(1.0),
            LayerMask::ALL,
        )
        .expect("room for the ball");
    }
    // Stepped without draining: both landings have to still be there.
    for _ in 0..60 {
        sim.step(TICK);
    }
    let mut out = Vec::new();
    sim.drain_contact_hits_into(&mut out);
    let mut landed: Vec<u32> = out.iter().map(|h| h.b.index()).collect();
    landed.sort_unstable();
    landed.dedup();
    assert_eq!(landed, [1, 2], "both landings waited to be collected");
    assert_eq!(sim.contact_hit_overflows(), 0);
    sim.drain_contact_hits_into(&mut out);
    assert!(out.is_empty(), "the drain emptied it");
}

// A caller draining every tick is what the queue is sized for, and the drain
// has to hand the buffers back rather than replacing them.
#[test]
fn draining_hits_reallocates_neither_side() {
    let mut sim = sim(2);
    sim.set_contact_min_impulse(1.0, TICK);
    add_floor(&mut sim);
    sim.add_dynamic(
        &BALL,
        [0.0, 5.5, 0.0],
        [0.0; 3],
        params(1.0),
        LayerMask::ALL,
    )
    .expect("room for the ball");

    let mut out = Vec::with_capacity(8);
    let capacity = out.capacity();
    let mut total = 0;
    for _ in 0..300 {
        sim.step(TICK);
        sim.drain_contact_hits_into(&mut out);
        total += out.len();
        assert_eq!(out.capacity(), capacity, "the caller's buffer was replaced");
    }
    assert!(total > 0);
    assert_eq!(sim.contact_hit_overflows(), 0);
}

// A threshold of zero must still not report contact that carried nothing, or
// every speculative pair in the world would be an impact.
#[test]
fn a_zero_threshold_still_needs_load() {
    let mut sim = sim(3);
    sim.set_contact_min_impulse(0.0, TICK);
    add_floor(&mut sim);
    sim.add_dynamic(
        &BALL,
        [0.0, 6.0, 0.0],
        [0.0; 3],
        DynamicParams {
            gravity_scale: 0.0,
            ..params(1.0)
        },
        LayerMask::ALL,
    )
    .expect("room for the ball");
    let mut out = Vec::new();
    for _ in 0..30 {
        sim.step(TICK);
        sim.drain_contact_hits_into(&mut out);
        assert!(out.is_empty(), "a body touching nothing: {out:?}");
    }
}
