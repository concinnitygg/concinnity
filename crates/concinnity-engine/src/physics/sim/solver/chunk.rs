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

use crate::physics::sim::config::{SimConfig, Softness};
use crate::physics::sim::contact::Manifold;
use crate::physics::sim::fluid::FluidLoad;
use crate::physics::sim::joint::{JointSolver, Prepared, Push};
use crate::physics::sim::math::Vec3;

use super::bodies::{Bodies, SolverBody};
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
    pub(crate) fluids: &'a [FluidLoad],
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
        integrate_velocities(chunk, tuning.config, tuning.fluids, tuning.h);
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

fn integrate_velocities(chunk: &mut Chunk<'_>, config: &SimConfig, fluids: &[FluidLoad], h: f32) {
    let Chunk { bodies, active, .. } = chunk;
    for &slot in *active {
        let body = bodies.get_mut(slot);
        let pull = config.gravity_at(body.position) * body.gravity_scale;
        match fluids.get(body.fluid as usize) {
            Some(load) => float(body, load, pull, h),
            None => body.linear_velocity += pull * h,
        }
        body.linear_velocity += body.force * (body.inv_mass * h);
        body.angular_velocity += body.inv_inertia.mul_vec3(body.torque) * h;
        // Implicit damping: stable at any timestep, unlike scaling by
        // (1 - h * damping), which turns a body inside out past h = 1/d.
        body.linear_velocity = body.linear_velocity * (1.0 / (1.0 + h * body.linear_damping));
        body.angular_velocity = body.angular_velocity * (1.0 / (1.0 + h * body.angular_damping));
    }
}

// One substep of gravity and the water on a floating body, backward Euler:
// the water is read at the pose the substep is heading for, and the rise and
// the turn take its stiffness `k` implicitly, `v' = v + h a / (1 + h^2 k / m)`,
// gravity sharing the rise's so the body rests exactly where the two
// balance. However stiff the float, the substep cannot gain energy: it damps
// what moves faster than it resolves and leaves the rest alone.
fn float(body: &mut SolverBody, load: &FluidLoad, pull: Vec3, h: f32) {
    let ahead = body.position + body.linear_velocity * h;
    let turned = body.rotation.integrate(body.angular_velocity, h);
    let push = load.push_on(ahead, turned, pull);
    let up = Vec3::from_array(load.up);
    let accel = pull + push.force * body.inv_mass;
    let rise = accel.dot(up);
    let settle = 1.0 / (1.0 + h * h * push.stiffness * body.inv_mass);
    body.linear_velocity += (accel - up * rise) * h + up * (rise * h * settle);
    let l = body.inv_inertia_local;
    let inv_turn = l.x.max(l.y).max(l.z);
    let spin = body.inv_inertia.mul_vec3(push.torque) * h;
    body.angular_velocity += spin * (1.0 / (1.0 + h * h * push.turn_stiffness * inv_turn));
}

fn integrate_positions(chunk: &mut Chunk<'_>, h: f32) {
    let Chunk { bodies, active, .. } = chunk;
    for &slot in *active {
        bodies.get_mut(slot).integrate_position(h);
    }
}
