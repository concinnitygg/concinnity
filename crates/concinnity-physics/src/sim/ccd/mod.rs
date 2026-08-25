// concinnity-physics/src/sim/ccd/mod.rs
//
// Continuous collision: what stops a body that moves further in one step than
// the thing it is moving toward is thick.
//
// The technique is conservative advancement over the sweep the query side
// already owns, applied as a motion clamp after the solve. A body that passed
// through something during the step is put back where it first met it, and
// nothing else about the step is redone. Two alternatives were available and
// are not what this does. Re-solving the step from the time of impact would
// need the whole pipeline to be re-entrant and would spend a second solve on
// the rarest case in the world. Widening the speculative margin by velocity
// would reach the same bodies through the narrow phase, but a manifold built
// between two shapes a metre apart is a guess at which features will meet,
// and it would put that guess into the same warm-start cache a resting stack
// depends on.
//
// Clamping mints no contact, which is the whole reason it is the safe choice
// here. The body is left inside the speculative margin rather than against
// the surface, so the next step's narrow phase builds the manifold from the
// real geometry with the real feature ids, and the warm start a landing needs
// is the one the cache would have had anyway. Velocity is left alone, so the
// impulse is the solver's to deliver rather than something this stage
// invents.
//
// The gap it stops in is load bearing rather than tidy. A contact the solver
// finds already closed is corrected by the soft spring, which leaves a
// fraction of the approach speed for the pass after: nothing at walking pace,
// a lurch through a thin wall at a thousand units a second. One it finds
// still open goes to the speculative branch instead, which allows exactly
// enough approach to touch and no more at any speed. So a stop is placed half
// the speculative margin short of the surface, near enough that the contact
// is built and far enough that it is built open.
//
// What it does not cover. The sweep is a translation, so a body thin enough
// to pass through a surface by spinning rather than by travelling is not
// caught. A body already touching what it went through is left to the solver,
// since it had a manifold and stopping it where it began would freeze
// anything resting on a surface. A stopped body loses the rest of that step's
// travel: it is stopped, not deflected, which reads as one tick of lag on a
// glancing hit. And what happens after the stop is the solver's -- a shape
// whose manifold is one point carries any impulse through it, while a box
// spreads one over four corners, which the sequential solve stops converging
// on somewhere past a few hundred units a second.
//
// Everything here is ordered by body slot and nothing is keyed by a hash, so
// two runs of the same scene clamp the same bodies in the same order. The
// working set is reserved when the simulation is built.

mod gate;
mod toi;

use alloc::vec::Vec;

use concinnity_memory::Pool;

use super::body::Body;
use super::config::SimConfig;
use super::math::Vec3;
use super::scene::Scene;
use super::sensor::Sensors;

use toi::{Blocked, Probe};

/// A body moving fast enough to be swept, and what the sweep decided.
#[derive(Debug, Clone, Copy)]
struct Mover {
    slot: u32,
    /// Where the body was when the step began.
    start: Vec3,
    /// How far the step moved it.
    motion: Vec3,
    /// Whether the body is driven to a position rather than by forces. A
    /// driven body arrives where it was sent, so what it met moves instead.
    driven: bool,
    outcome: Option<Outcome>,
}

/// What a mover's sweep asks to be changed.
#[derive(Debug, Clone, Copy)]
enum Outcome {
    /// Put the mover back where it first met `target`.
    Stop { target: u32, position: Vec3 },
    /// Move `target` clear, for a mover that keeps the position it was sent
    /// to.
    Shove { target: u32, offset: Vec3 },
}

/// The step's continuous-collision pass: the bodies fast enough to need one,
/// and the regions they went through.
pub(crate) struct Ccd {
    movers: Vec<Mover>,
    /// Mover slot and region slot for every region a mover crossed clean
    /// through, sorted before it is reported.
    crossings: Vec<(u32, u32)>,
    /// The furthest anything travelled this step, squared.
    max_motion_sq: f32,
    overflows: u32,
}

impl Ccd {
    /// One entry per body: a world where everything is moving fast at once is
    /// the honest worst case, and a reservation that could not hold it would
    /// decline exactly when the sweep is most needed.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Ccd {
            movers: Vec::with_capacity(capacity),
            crossings: Vec::with_capacity(capacity),
            max_motion_sq: 0.0,
            overflows: 0,
        }
    }

    pub(crate) fn reserved_bytes(&self) -> u64 {
        (self.movers.capacity() * size_of::<Mover>()
            + self.crossings.capacity() * size_of::<(u32, u32)>()) as u64
    }

    #[cfg(test)]
    /// Movers and crossings the reservation had no room for.
    pub(crate) fn overflows(&self) -> u32 {
        self.overflows
    }

    #[cfg(test)]
    pub(crate) fn clear_overflows(&mut self) {
        self.overflows = 0;
    }

    pub(crate) fn begin(&mut self) {
        self.movers.clear();
        self.crossings.clear();
        self.max_motion_sq = 0.0;
    }

    /// Offer one body's step to the gate. Called for everything the step
    /// moved, since the widest travel decides how far a candidate search has
    /// to reach even when the body that travelled it is not a mover.
    pub(crate) fn observe(&mut self, slot: u32, body: &Body, start: Vec3, ratio: f32) {
        let motion = body.position - start;
        self.max_motion_sq = self.max_motion_sq.max(motion.length_squared());
        let Some(shape) = body.convex() else {
            return;
        };
        if !gate::is_fast(gate::min_extent(shape), motion, ratio) {
            return;
        }
        if self.movers.len() == self.movers.capacity() {
            self.overflows = self.overflows.saturating_add(1);
            return;
        }
        self.movers.push(Mover {
            slot,
            start,
            motion,
            driven: body.is_kinematic(),
            outcome: None,
        });
    }

    /// Sweep every mover's path and decide what each one asks for. Reads the
    /// world and changes nothing in it.
    pub(crate) fn resolve(&mut self, scene: Scene<'_>, config: &SimConfig, dt: f32) {
        if self.movers.is_empty() {
            return;
        }
        let Ccd {
            movers,
            crossings,
            max_motion_sq,
            overflows,
        } = self;
        let expand = libm::sqrtf(*max_motion_sq);
        let standoff = 0.5 * config.speculative_margin;

        for index in 0..movers.len() {
            let mover = movers[index];
            let Some(body) = scene.bodies.get_at(mover.slot as usize) else {
                continue;
            };
            let Some(shape) = body.convex() else {
                continue;
            };
            let probe = Probe {
                slot: mover.slot,
                shape,
                rotation: body.orientation,
                start: mover.start,
                motion: mover.motion,
                mask: body.mask,
                expand,
            };
            let recorded = crossings.len();
            let blocked = {
                let known = &movers[..];
                toi::scan(
                    scene,
                    &probe,
                    |slot, target| motion_of(known, dt, slot, target),
                    |region| {
                        if crossings.len() == crossings.capacity() {
                            *overflows = overflows.saturating_add(1);
                            return;
                        }
                        crossings.push((mover.slot, region));
                    },
                )
            };
            let outcome = blocked.and_then(|hit| outcome_for(scene.bodies, &mover, &hit, standoff));
            if let Some(Outcome::Stop { position, .. }) = outcome {
                keep_crossed(scene, &probe, position - mover.start, crossings, recorded);
            }
            movers[index].outcome = outcome;
        }
    }

    /// Report every region a mover crossed clean through, in slot order.
    pub(crate) fn report_crossings(&mut self, bodies: &Pool<Body>, sensors: &mut Sensors) {
        if self.crossings.is_empty() {
            return;
        }
        self.crossings.sort_unstable();
        for &(mover, region) in &self.crossings {
            sensors.record_pass_through(bodies, mover, region);
        }
    }

    /// Apply what the sweeps decided, in slot order.
    ///
    /// Stops go first so that a body which both stopped against something and
    /// was shoved by a driven body ends up clear of the driven one: that body
    /// arrives wherever it was sent whatever this stage does, so its half of
    /// the pair is the one that cannot be given up.
    pub(crate) fn apply(&self, bodies: &mut Pool<Body>) {
        for mover in &self.movers {
            if let Some(Outcome::Stop { target, position }) = mover.outcome {
                if let Some(body) = bodies.get_at_mut(mover.slot as usize) {
                    body.position = position;
                }
                wake(bodies, target);
            }
        }
        for mover in &self.movers {
            if let Some(Outcome::Shove { target, offset }) = mover.outcome {
                if let Some(body) = bodies.get_at_mut(target as usize) {
                    body.position += offset;
                }
                wake(bodies, target);
            }
        }
    }

    #[cfg(test)]
    /// Bodies the last step was fast enough to sweep.
    pub(crate) fn mover_count(&self) -> usize {
        self.movers.len()
    }
}

/// Drop the regions a mover only crossed because its step was measured before
/// it was stopped. `from` is where this mover's entries begin.
fn keep_crossed(
    scene: Scene<'_>,
    probe: &Probe<'_>,
    motion: Vec3,
    crossings: &mut Vec<(u32, u32)>,
    from: usize,
) {
    let mut kept = from;
    for at in from..crossings.len() {
        let entry = crossings[at];
        if toi::still_crossed(scene, probe, motion, entry.1) {
            crossings[kept] = entry;
            kept += 1;
        }
    }
    crossings.truncate(kept);
}

/// What one mover's impact asks to be changed, or `None` when nothing can be.
///
/// `standoff` is how far short of the surface a stop is placed, so the contact
/// the next step builds is one the solver can still see a gap in.
fn outcome_for(
    bodies: &Pool<Body>,
    mover: &Mover,
    hit: &Blocked,
    standoff: f32,
) -> Option<Outcome> {
    if !mover.driven {
        return Some(Outcome::Stop {
            target: hit.target,
            position: mover.start + mover.motion * hit.toi + hit.normal * standoff,
        });
    }
    // A driven body arrives where it was sent, so the only thing left to move
    // is what it was about to go through, and only if contact can move it.
    let target = bodies.get_at(hit.target as usize)?;
    if !target.is_dynamic() {
        return None;
    }
    // How far the pair would still close along the normal after the contact,
    // which is exactly what the target has to give way by.
    let remaining = hit.relative_motion.dot(hit.normal) * (1.0 - hit.toi);
    Some(Outcome::Shove {
        target: hit.target,
        offset: hit.normal * (remaining - standoff),
    })
}

/// How far a body travelled this step: exactly, for one the gate already
/// measured, and from the velocity it ended with for everything else. A body
/// the gate turned down moved less than its own width, so the difference
/// between the two is smaller than the contact it is being measured for.
fn motion_of(movers: &[Mover], dt: f32, slot: u32, body: &Body) -> Vec3 {
    match movers.binary_search_by_key(&slot, |mover| mover.slot) {
        Ok(at) => movers[at].motion,
        Err(_) => body.linear_velocity * dt,
    }
}

/// Wake what a mover ran into, so an island that had settled is simulated
/// again on the step the contact is built.
fn wake(bodies: &mut Pool<Body>, slot: u32) {
    if let Some(body) = bodies.get_at_mut(slot as usize)
        && body.is_dynamic()
    {
        body.wake();
    }
}

/// Whether this step has any continuous collision to do at all.
pub(crate) fn enabled(config: &SimConfig) -> bool {
    config.ccd_enabled && config.ccd_motion_ratio > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::math::vec3;
    use crate::{ColliderShape, DynamicParams, LayerMask};

    const BULLET: ColliderShape = ColliderShape::Ball { radius: 0.1 };

    fn params() -> DynamicParams {
        DynamicParams {
            mass: 1.0,
            friction: 0.5,
            restitution: 0.0,
            gravity_scale: 1.0,
            linear_damping: 0.0,
        }
    }

    fn dynamic_at(position: Vec3) -> Body {
        Body::dynamic(
            BULLET,
            position,
            super::super::math::Quat::IDENTITY,
            params(),
            LayerMask::ALL,
        )
    }

    fn mover(slot: u32, motion: Vec3) -> Mover {
        Mover {
            slot,
            start: Vec3::ZERO,
            motion,
            driven: false,
            outcome: None,
        }
    }

    #[test]
    fn the_gate_keeps_slow_bodies_out_of_the_working_set() {
        let mut ccd = Ccd::with_capacity(4);
        ccd.begin();
        let mut body = dynamic_at(vec3(0.0, 0.0, 0.01));
        ccd.observe(0, &body, Vec3::ZERO, 0.5);
        assert_eq!(ccd.mover_count(), 0, "a hundredth of a unit is not fast");

        body.position = vec3(0.0, 0.0, 4.0);
        ccd.observe(1, &body, Vec3::ZERO, 0.5);
        assert_eq!(ccd.mover_count(), 1);
    }

    #[test]
    fn the_widest_travel_is_taken_from_every_body_and_not_just_the_movers() {
        let mut ccd = Ccd::with_capacity(4);
        ccd.begin();
        let slow = dynamic_at(vec3(0.0, 0.0, 0.02));
        ccd.observe(0, &slow, Vec3::ZERO, 0.5);
        assert_eq!(ccd.mover_count(), 0);
        assert!((libm::sqrtf(ccd.max_motion_sq) - 0.02).abs() < 1.0e-6);
    }

    // The reservation is one entry per body, so this is the guard rather than
    // a case a world reaches. It still has to decline rather than grow.
    #[test]
    fn a_full_working_set_declines_and_counts() {
        let mut ccd = Ccd::with_capacity(1);
        ccd.begin();
        let body = dynamic_at(vec3(0.0, 0.0, 4.0));
        ccd.observe(0, &body, Vec3::ZERO, 0.5);
        ccd.observe(1, &body, Vec3::ZERO, 0.5);
        assert_eq!(ccd.mover_count(), 1);
        assert_eq!(ccd.overflows(), 1);
        ccd.clear_overflows();
        assert_eq!(ccd.overflows(), 0);
    }

    #[test]
    fn a_measured_mover_reports_its_own_travel_and_everything_else_its_velocity() {
        let movers = [mover(1, vec3(0.0, 0.0, 9.0)), mover(4, Vec3::ZERO)];
        let mut body = dynamic_at(Vec3::ZERO);
        body.linear_velocity = vec3(6.0, 0.0, 0.0);
        assert_eq!(motion_of(&movers, 0.5, 1, &body), vec3(0.0, 0.0, 9.0));
        assert_eq!(motion_of(&movers, 0.5, 2, &body), vec3(3.0, 0.0, 0.0));
    }

    /// Half the default speculative margin, which is what a stop is placed
    /// short of the surface by.
    const STANDOFF: f32 = 0.01;

    #[test]
    fn a_free_mover_stops_short_of_what_it_met() {
        let bodies: Pool<Body> = Pool::with_capacity(1);
        let mut moving = mover(0, vec3(0.0, 0.0, 10.0));
        moving.start = vec3(0.0, 0.0, -5.0);
        let hit = Blocked {
            target: 3,
            toi: 0.25,
            normal: vec3(0.0, 0.0, -1.0),
            relative_motion: vec3(0.0, 0.0, 10.0),
        };
        match outcome_for(&bodies, &moving, &hit, STANDOFF) {
            Some(Outcome::Stop { target, position }) => {
                assert_eq!(target, 3);
                // A quarter of the way along, then backed off the surface.
                assert_eq!(position, vec3(0.0, 0.0, -2.5 - STANDOFF));
            }
            other => panic!("{other:?}"),
        }
    }

    // A driven body keeps the position it was sent to, so what it met is
    // pushed clear by exactly the closing that was left.
    #[test]
    fn a_driven_mover_shoves_what_it_would_have_gone_through() {
        let mut bodies: Pool<Body> = Pool::with_capacity(1);
        bodies.insert(dynamic_at(Vec3::ZERO)).expect("room");
        let mut driven = mover(1, vec3(0.0, 0.0, 8.0));
        driven.driven = true;
        let hit = Blocked {
            target: 0,
            toi: 0.5,
            normal: vec3(0.0, 0.0, -1.0),
            relative_motion: vec3(0.0, 0.0, 8.0),
        };
        match outcome_for(&bodies, &driven, &hit, STANDOFF) {
            Some(Outcome::Shove { target, offset }) => {
                assert_eq!(target, 0);
                // Half the motion was left, all of it into the target, and
                // the same standoff on top of it.
                assert_eq!(offset, vec3(0.0, 0.0, 4.0 + STANDOFF));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_driven_mover_meeting_immovable_geometry_asks_for_nothing() {
        let mut bodies: Pool<Body> = Pool::with_capacity(1);
        bodies
            .insert(Body::fixed(
                BULLET,
                Vec3::ZERO,
                super::super::math::Quat::IDENTITY,
                0.5,
                LayerMask::ALL,
            ))
            .expect("room");
        let mut driven = mover(1, vec3(0.0, 0.0, 8.0));
        driven.driven = true;
        let hit = Blocked {
            target: 0,
            toi: 0.5,
            normal: vec3(0.0, 0.0, -1.0),
            relative_motion: vec3(0.0, 0.0, 8.0),
        };
        assert!(outcome_for(&bodies, &driven, &hit, STANDOFF).is_none());
    }

    #[test]
    fn the_stage_is_off_when_either_knob_turns_it_off() {
        assert!(enabled(&SimConfig::default()));
        assert!(!enabled(&SimConfig {
            ccd_enabled: false,
            ..SimConfig::default()
        }));
        assert!(!enabled(&SimConfig {
            ccd_motion_ratio: 0.0,
            ..SimConfig::default()
        }));
    }
}
