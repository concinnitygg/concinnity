// concinnity-physics/src/sim/solver/chunk.rs
//
// One piece of a step's solve: whole islands, and everything needed to carry
// them from the top of the step to the bottom.
//
// A chunk runs the entire substep loop rather than one pass of it. That is what
// the island split buys over splitting each pass: the pieces have nothing to
// say to each other at any point between the first integration and the last
// bounce, so they are handed out once and joined once, instead of eight times
// per step.
//
// Every array a chunk holds is a run rather than a scatter, because the
// partition laid them out that way, so the runs are cut with plain slice
// splits. The exception is the body states, which stay indexed by pool slot: a
// chunk reaches those through a shared handle whose contract the partition is
// what satisfies.

use crate::sim::config::{SimConfig, Softness};
use crate::sim::contact::Manifold;
use crate::sim::joint::{JointSolver, Prepared, Push};
use crate::sim::math::vec3;

use super::bodies::Bodies;
use super::contact::{self, ContactConstraint};

/// One worker's share of a step's solve.
#[derive(Default)]
pub(crate) struct Chunk<'a> {
    pub(crate) bodies: Bodies<'a>,
    pub(crate) active: &'a [u32],
    pub(crate) constraints: &'a mut [ContactConstraint],
    pub(crate) contact_source: &'a [u32],
    pub(crate) joints: &'a mut [Prepared],
}

/// What every chunk of one step reads: the same tuning, and the manifolds the
/// constraints are built from.
pub(crate) struct Tuning<'a> {
    pub(crate) manifolds: &'a [Manifold],
    pub(crate) config: &'a SimConfig,
    pub(crate) substeps: u32,
    pub(crate) h: f32,
    pub(crate) inv_h: f32,
    pub(crate) soft: Softness,
    pub(crate) joint_biased: Push,
    pub(crate) joint_rigid: Push,
}

/// Carry one chunk's islands through the whole step.
///
/// Joints go first in each pass. A joint is the stiffer constraint of the two,
/// and letting the contacts have the last word keeps a body from being driven
/// into a surface by the joint holding it.
pub(crate) fn run(chunk: &mut Chunk<'_>, tuning: &Tuning<'_>) {
    contact::prepare(
        chunk.constraints,
        chunk.contact_source,
        tuning.manifolds,
        &chunk.bodies,
    );
    for _ in 0..tuning.substeps {
        integrate_velocities(chunk, tuning.config.gravity, tuning.h);
        JointSolver::warm_start(chunk.joints, &mut chunk.bodies);
        contact::warm_start(chunk.constraints, &mut chunk.bodies);
        JointSolver::solve(
            chunk.joints,
            &mut chunk.bodies,
            &tuning.joint_biased,
            tuning.h,
        );
        contact::solve(
            chunk.constraints,
            &mut chunk.bodies,
            &tuning.soft,
            tuning.config,
            tuning.inv_h,
            true,
        );
        integrate_positions(chunk, tuning.h);
        // The poses the joint masses were built from just moved.
        JointSolver::refresh(chunk.joints, &chunk.bodies);
        JointSolver::solve(
            chunk.joints,
            &mut chunk.bodies,
            &tuning.joint_rigid,
            tuning.h,
        );
        contact::solve(
            chunk.constraints,
            &mut chunk.bodies,
            &Softness::RIGID,
            tuning.config,
            tuning.inv_h,
            false,
        );
    }
    contact::apply_restitution(chunk.constraints, &mut chunk.bodies, tuning.config);
}

fn integrate_velocities(chunk: &mut Chunk<'_>, gravity: f32, h: f32) {
    let pull = vec3(0.0, -gravity, 0.0);
    let Chunk { bodies, active, .. } = chunk;
    for &slot in active.iter() {
        let body = bodies.get_mut(slot);
        body.linear_velocity += pull * (body.gravity_scale * h);
        // Implicit damping: stable at any timestep, unlike scaling by
        // (1 - h * damping), which turns a body inside out past h = 1/d.
        let decay = 1.0 / (1.0 + h * body.damping);
        body.linear_velocity = body.linear_velocity * decay;
        body.angular_velocity = body.angular_velocity * decay;
    }
}

fn integrate_positions(chunk: &mut Chunk<'_>, h: f32) {
    let Chunk { bodies, active, .. } = chunk;
    for &slot in active.iter() {
        bodies.get_mut(slot).integrate_position(h);
    }
}
