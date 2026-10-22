// Stepping the world: cost estimates, the solver pass, and sleep.

use super::{Simulation, proxy_for};
use crate::physics::sim::body::{Body, NO_FLUID};
use crate::physics::sim::ccd;
use crate::physics::sim::config::SimConfig;
use crate::physics::sim::contact::{Manifold, carry_impulses};
use crate::physics::sim::island::Islands;
use crate::physics::sim::joint::Joint;
use crate::physics::sim::narrow;
use crate::physics::sim::scene::Scene;
use crate::physics::sim::solver::{self, Solver, SolverBody};
use concinnity_core::fanout::FanOut;
use concinnity_core::math::SaturatingFrom;
use concinnity_core::memory::Pool;

// What a step is worth handing out, in units of one body's integration -- a
// contact costs about twenty of those and a joint about sixteen.
//
// The threshold is high on purpose. Gathering a pool's workers from a thread
// that is not one of them costs about as much as three hundred of these units,
// and the gathering is serial: a step that only just covers it comes out level
// at best. So a step hands its work out once the part of it a split can reach
// is worth an order of magnitude more than the gathering, and keeps it
// otherwise.
const CONTACT_COST: usize = 20;
const JOINT_COST: usize = 16;
const MIN_FANOUT_COST: usize = 4000;

impl Simulation {
    /// Advance the simulation by `dt` seconds on the calling thread.
    pub fn step(&mut self, dt: f32) {
        self.step_with(dt, &concinnity_core::fanout::Inline);
    }

    /// Advance the simulation by `dt` seconds, offering the step's independent
    /// work to `fanout`.
    ///
    /// A step splits the same way whatever it is handed, so a world stepped
    /// across a thread pool and the same world stepped on one thread land in
    /// exactly the same place. Nothing is reserved for a fan-out that
    /// [`Simulation::reserve_workers`] was not told about, so a wider one is
    /// used only as far as the reservation goes.
    ///
    /// # Examples
    ///
    /// ```
    /// use concinnity_core::fanout::Inline;
    /// use concinnity_engine::physics::{ColliderShape, DynamicParams, LayerMask, Simulation};
    ///
    /// let mut sim = Simulation::with_capacity(1);
    /// let ball = sim
    ///     .add_dynamic(
    ///         &ColliderShape::Ball { radius: 0.5 },
    ///         [0.0, 10.0, 0.0],
    ///         [0.0; 3],
    ///         DynamicParams {
    ///             mass: 1.0,
    ///             friction: 0.4,
    ///             restitution: 0.2,
    ///             gravity_scale: 1.0,
    ///             linear_damping: 0.0,
    ///         },
    ///         LayerMask::ALL,
    ///     )
    ///     .expect("room for one body");
    /// sim.step_with(1.0 / 60.0, &Inline);
    /// assert!(sim.body_pose_quat(ball).expect("a live body").0[1] < 10.0, "it fell");
    /// ```
    pub fn step_with(&mut self, dt: f32, fanout: &impl FanOut) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let asked = fanout.workers().max(1);
        if asked > self.workers {
            self.worker_overflows = self.worker_overflows.saturating_add(1);
        }
        let workers = asked.min(self.workers);
        self.drive_kinematics(dt);
        let awake = self.refresh_bounds();
        // Reaching a fan-out's workers costs the same whatever is handed to
        // them, so a step with little a split can reach is worth more on the
        // thread it is already on.
        if workers > 1 && self.splittable_cost(awake) >= MIN_FANOUT_COST {
            fanout.scope(|| self.advance(dt, fanout, workers));
        } else {
            self.advance(dt, &concinnity_core::fanout::Inline, 1);
        }
    }

    /// Roughly what the step ahead will cost, from what the world was holding
    /// last step.
    ///
    /// Only a moving body's contacts are work: a settled world keeps a full
    /// pair list and solves none of it, so the pairs are counted against what
    /// is still awake rather than taken at face value.
    pub(super) fn step_cost(&self, awake: usize) -> usize {
        if awake == 0 {
            return 0;
        }
        let contacts = self.broadphase.pair_count().min(awake * 4);
        let joints = self.joints.len().min(awake * 2);
        awake + contacts * CONTACT_COST + joints * JOINT_COST
    }

    /// The part of the step ahead a split could take off one worker, or
    /// nothing when one island dominates it.
    ///
    /// The solve is split by island, so whatever the largest island holds
    /// runs on one worker however many there are. Last step's islands are the
    /// estimate. A step finishes no sooner than its largest island does, so
    /// when that island holds much of the solve what the rest saves does not
    /// cover waking a pool for every stage, and the step is kept whole.
    pub(super) fn splittable_cost(&self, awake: usize) -> usize {
        let spread = self.solver.spread();
        if spread.is_dominated() {
            return 0;
        }
        usize::saturating_from(spread.reachable(self.step_cost(awake) as u64))
    }

    /// Everything a step does once the bounds are current and the workers, if
    /// any, have been gathered.
    pub(super) fn advance(&mut self, dt: f32, fanout: &impl FanOut, workers: usize) {
        let Self {
            config,
            bodies,
            broadphase,
            contacts,
            fields,
            narrow,
            joints,
            islands,
            solver,
            sensors,
            impacts,
            ccd,
            fluids,
            ..
        } = self;
        let sweeping = ccd::enabled(config);

        let pairs = broadphase.sweep(fanout, workers);
        sensors.resolve(bodies, pairs.sensors);
        let (current, previous) = contacts.begin();
        narrow.build(
            narrow::Work {
                bodies,
                fields,
                pairs: pairs.contacts,
                previous,
                out: current,
                margin: config.speculative_margin,
            },
            fanout,
            workers,
        );
        carry_impulses(previous, current);
        wake_driven_contacts(bodies, current);

        solver.begin();
        gather(bodies, solver, current, joints.as_slice());
        solver.run(
            solver::Work {
                manifolds: current,
                joints: joints.as_mut_slice(),
                islands,
                config,
                fluids: fluids.as_slice(),
                dt,
            },
            fanout,
            workers,
        );
        // The solver is asked what it delivered, so a pair whose manifold was
        // carried forward untouched is not read as a fresh collision.
        impacts.collect(bodies, current, solver.loads(), dt);
        ccd.begin();
        for (handle, body) in bodies.iter_mut() {
            if !body.is_simulated() {
                continue;
            }
            let slot = u32::saturating_from(handle.index());
            sensors.mark_moved(slot);
            let solved = solver.body(slot);
            let began_at = body.position;
            body.linear_velocity = solved.linear_velocity;
            body.angular_velocity = solved.angular_velocity;
            body.position = solved.position;
            body.orientation = solved.rotation;
            body.fluid = NO_FLUID;
            // A position-driven body lands exactly where it was sent, not
            // wherever the substeps' arithmetic left it.
            if let Some(target) = body.kinematic_target.take() {
                body.position = target;
            }
            if sweeping {
                ccd.observe(slot, body, began_at, config.ccd_motion_ratio);
            }
        }

        // Every load was read; the next step takes only what is set for it.
        fluids.clear();

        // After the write-back, so a mover is swept along the path the step
        // actually took it on, and before sleep, so what it ran into is awake
        // when the islands are decided.
        if sweeping {
            let scene = Scene {
                bodies,
                broadphase,
                fields,
            };
            ccd.resolve(scene, config, dt);
            ccd.report_crossings(bodies, sensors);
            ccd.apply(bodies, sensors);
        }

        if !solver.is_idle() {
            update_sleep(config, bodies, islands, current, joints.as_slice(), dt);
        }
    }

    /// Re-fatten the bounds of the bodies that moved, and tell the broad phase
    /// about the ones whose bounds actually changed.
    pub(super) fn refresh_bounds(&mut self) -> usize {
        let margin = self.config.bounds_margin;
        let Self {
            bodies, broadphase, ..
        } = self;
        let mut awake = 0;
        for (handle, body) in bodies.iter_mut() {
            if !body.is_simulated() {
                continue;
            }
            awake += 1;
            if body.refresh_bounds(margin) {
                broadphase.set_proxy(u32::saturating_from(handle.index()), proxy_for(body));
            }
        }
        awake
    }
}

/// Hand the solver the bodies this step can reach: the ones it moves, and the
/// immovable ones a contact leans against. Everything else keeps whatever
/// state it had, because nothing will read it, which is what makes a settled
/// world cost close to nothing.
fn gather(bodies: &Pool<Body>, solver: &mut Solver, manifolds: &[Manifold], joints: &[Joint]) {
    for (handle, body) in bodies.iter() {
        if body.is_simulated() {
            solver.set_body(
                u32::saturating_from(handle.index()),
                SolverBody::from_body(body),
            );
        }
    }
    for manifold in manifolds {
        gather_partner(bodies, solver, manifold.a, manifold.b);
    }
    for joint in joints {
        gather_partner(bodies, solver, joint.a, joint.b);
    }
}

/// Take the immovable half of a constraint, so the moving half has something
/// to lean on.
fn gather_partner(bodies: &Pool<Body>, solver: &mut Solver, a: u32, b: u32) {
    let simulated = |slot: u32| bodies.get_at(slot as usize).is_some_and(Body::is_simulated);
    let (moves_a, moves_b) = (simulated(a), simulated(b));
    if moves_a == moves_b {
        // Both moving: already taken. Neither moving: the constraint is
        // skipped, so neither is needed.
        return;
    }
    let resting = if moves_a { b } else { a };
    if let Some(body) = bodies.get_at(resting as usize) {
        solver.set_body(resting, SolverBody::from_body(body));
    }
}

/// Wake whatever a driven body is about to move into, so a platform can
/// disturb a stack that has settled on it.
///
/// Without this a sleeping body would be gathered as immovable and the
/// platform would slide straight through it.
fn wake_driven_contacts(bodies: &mut Pool<Body>, manifolds: &[Manifold]) {
    for manifold in manifolds {
        for (slot, other) in [(manifold.a, manifold.b), (manifold.b, manifold.a)] {
            let driving = bodies.get_at(slot as usize).is_some_and(|body| {
                body.kinematic_target
                    .is_some_and(|target| target != body.position)
            });
            if driving && let Some(body) = bodies.get_at_mut(other as usize) {
                body.wake();
            }
        }
    }
}

/// Decide which islands have settled, and stop simulating the ones that have.
fn update_sleep(
    config: &SimConfig,
    bodies: &mut Pool<Body>,
    islands: &mut Islands,
    manifolds: &[Manifold],
    joints: &[Joint],
    dt: f32,
) {
    for (_, body) in bodies.iter_mut() {
        if !body.is_dynamic() {
            continue;
        }
        if !config.allow_sleep {
            body.wake();
            continue;
        }
        if body.sleeping {
            continue;
        }
        if body.is_still(config.sleep_linear_velocity, config.sleep_angular_velocity) {
            body.sleep_timer += dt;
        } else {
            body.sleep_timer = 0.0;
        }
    }
    if !config.allow_sleep {
        return;
    }

    islands.clear();
    let movable = |slot: u32| bodies.get_at(slot as usize).is_some_and(Body::is_dynamic);
    for manifold in manifolds {
        if movable(manifold.a) && movable(manifold.b) {
            islands.union(manifold.a, manifold.b);
        }
    }
    // Two bodies a joint holds settle together or not at all, exactly as two
    // bodies leaning on each other do.
    for joint in joints {
        if movable(joint.a) && movable(joint.b) {
            islands.union(joint.a, joint.b);
        }
    }
    for (handle, body) in bodies.iter() {
        if body.is_dynamic() {
            islands.mark(
                u32::saturating_from(handle.index()),
                body.sleep_timer >= config.time_to_sleep,
            );
        }
    }
    // A motor with somewhere to go is still doing work however still the
    // bodies look, so its island stays awake.
    for joint in joints {
        if !joint.frame.is_driven() {
            continue;
        }
        for slot in [joint.a, joint.b] {
            if movable(slot) {
                islands.mark(slot, false);
            }
        }
    }
    for (handle, body) in bodies.iter_mut() {
        if !body.is_dynamic() {
            continue;
        }
        if islands.island_is_still(u32::saturating_from(handle.index())) {
            body.sleep();
        } else {
            body.sleeping = false;
        }
    }
}
