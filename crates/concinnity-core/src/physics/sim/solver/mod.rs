// Sequential impulses over substeps, with soft contacts and warm starting.
//
// Three choices make the difference between a stack that stands and one that
// hums. Substepping: velocity and position are advanced together several times
// per step, so a contact is re-measured against where the bodies now are
// instead of where they were at the top of the step. Soft contacts: the
// penetration correction is an implicit spring given as a frequency and a
// damping ratio, and each impulse is split between correcting now and being
// remembered, which is what stops the correction from feeding energy back in.
// Warm starting: every point begins from the impulse its own feature carried
// last step, so a resting stack starts each step already holding itself up.
//
// A relax pass follows each biased pass with the bias switched off. It removes
// the velocity the correction added without reopening the penetration, and it
// is why restitution can be applied afterwards from the approach speed
// measured before the step rather than from whatever the solver left behind.
//
// Velocities are gathered into a dense array first and written back at the
// end. The pool's slots are the index, so the solver never borrows two bodies
// out of the pool at once, and the arrays it does touch stay contiguous.
//
// The solve itself is split by island: `partition` groups the step's bodies and
// constraints into the sets that share nothing, and the chunks it cuts run the
// whole substep loop side by side. A split changes nothing about the answer --
// within an island the constraints are visited in the order the manifolds gave
// them, and between islands there is no order to change -- so a world solved on
// one worker and on twelve reaches the same bits. What varies with the world is
// only how much of the solve a split can reach: a hundred stacks share out
// evenly, and one tall stack is one island and stays on one thread.
//
// What a contact carried is reported separately from what it warm starts with.
// The impulse a point ends the step holding is one substep's worth, which is
// what the next step's warm start wants; what an impact is measured by is the
// whole step's, so the applied impulse is accumulated as it is applied.

mod bodies;
mod chunk;
mod contact;
mod partition;

use alloc::vec::Vec;

use crate::physics::fanout::Fanout;

use super::config::{SimConfig, Softness};
use super::contact::Manifold;
use super::island::Islands;
use super::joint::{Joint, JointSolver, Prepared, Push};

use chunk::{Chunk, Tuning};
use contact::ContactConstraint;
use partition::{Ends, Partition};

pub(crate) use bodies::{Bodies, SolverBody};

/// Chunks one step's solve may be cut into. A caller lending more workers than
/// this gets this many, which changes how long the step takes and nothing
/// about where it leaves the world.
pub(crate) const MAX_WORKERS: usize = 64;

/// What one solved manifold's contact carried this step.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ContactLoad {
    /// Index into the manifold list the step solved.
    pub(crate) manifold: u32,
    /// Total normal impulse delivered across the manifold's points.
    pub(crate) impulse: f32,
}

pub(crate) struct Solver {
    bodies: Vec<SolverBody>,
    /// Whether a slot's state was taken this step. A slot left out holds
    /// whatever the last step that did take it left behind, so this is what
    /// separates a body the step can reason about from a stale one.
    taken: Vec<bool>,
    /// Slots the step integrates, in slot order.
    active: Vec<u32>,
    /// Grown to the most constraints any step has held and reused, so a step
    /// pays for the slots it uses and never for initialising them again.
    constraints: Vec<ContactConstraint>,
    /// What each constraint carried, in manifold order. Only the pairs the
    /// step actually solved appear, which is what keeps a settled pair's
    /// carried-over manifold from reading as a fresh collision.
    loads: Vec<ContactLoad>,
    joints: JointSolver,
    partition: Partition,
}

impl Solver {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Solver {
            bodies: alloc::vec![SolverBody::IMMOVABLE; capacity],
            taken: alloc::vec![false; capacity],
            active: Vec::with_capacity(capacity),
            constraints: Vec::with_capacity(capacity * 2),
            loads: Vec::with_capacity(capacity * 2),
            joints: JointSolver::with_capacity(capacity),
            partition: Partition::with_capacity(capacity),
        }
    }

    pub(crate) fn reserved_bytes(&self) -> u64 {
        (self.bodies.capacity() * size_of::<SolverBody>()
            + self.taken.capacity()
            + self.active.capacity() * size_of::<u32>()
            + self.constraints.capacity() * size_of::<ContactConstraint>()
            + self.loads.capacity() * size_of::<ContactLoad>()) as u64
            + self.joints.reserved_bytes()
            + self.partition.reserved_bytes()
    }

    pub(crate) fn begin(&mut self) {
        self.active.clear();
        self.loads.clear();
        self.taken.fill(false);
    }

    /// What each contact the step solved carried, in manifold order.
    pub(crate) fn loads(&self) -> &[ContactLoad] {
        &self.loads
    }

    /// Take a body's state for this step. Bodies the step cannot move and no
    /// contact leans on may be left out; `taken` is what keeps the state they
    /// left behind from being read as this step's.
    pub(crate) fn set_body(&mut self, slot: u32, body: SolverBody) {
        if body.simulated && !self.taken[slot as usize] {
            self.active.push(slot);
        }
        self.taken[slot as usize] = true;
        self.bodies[slot as usize] = body;
    }

    pub(crate) fn body(&self, slot: u32) -> &SolverBody {
        &self.bodies[slot as usize]
    }

    /// Whether the step has anything to move at all.
    pub(crate) fn is_idle(&self) -> bool {
        self.active.is_empty()
    }

    /// Contacts the last step solved.
    #[cfg(test)]
    pub(crate) fn constraint_count(&self) -> usize {
        self.loads.len()
    }

    /// Islands the last step's solve broke into.
    #[cfg(test)]
    pub(crate) fn island_count(&self) -> usize {
        self.partition.islands()
    }

    /// Chunks the last step's solve was cut into.
    #[cfg(test)]
    pub(crate) fn chunk_count(&self) -> usize {
        self.partition.ends().len()
    }

    /// Advance every active body and resolve every contact and joint, offering
    /// the islands to `fanout`.
    pub(crate) fn run(&mut self, work: Work<'_>, fanout: &impl Fanout, workers: usize) {
        let Work {
            manifolds,
            joints,
            islands,
            config,
            dt,
        } = work;
        let substeps = config.substep_count();
        let h = dt / substeps as f32;
        let inv_h = if h > 0.0 { 1.0 / h } else { 0.0 };
        // A constraint cannot be stiffer than the rate it is solved at.
        let hertz = config.contact_hertz.min(0.25 * inv_h);
        let soft = Softness::new(hertz, config.contact_damping_ratio, h);
        let joint_soft = Softness::new(
            config.joint_hertz.min(0.25 * inv_h),
            config.joint_damping_ratio,
            h,
        );
        let push = |soft: Softness, use_bias: bool| Push {
            soft,
            inv_h,
            max_push: config.max_push_velocity,
            use_bias,
        };

        self.partition.build(
            partition::Work {
                bodies: &self.bodies,
                taken: &self.taken,
                active: &self.active,
                manifolds,
                joints,
            },
            islands,
            workers.clamp(1, MAX_WORKERS),
        );
        let held = self.partition.ends().last().copied().unwrap_or_default();
        if self.constraints.len() < held.contacts {
            self.constraints
                .resize_with(held.contacts, ContactConstraint::default);
        }
        // Rows are built here rather than inside a chunk because a joint is
        // read out of body storage the chunks are about to start writing.
        self.joints
            .prepare(self.partition.joint_source(), joints, &self.bodies);

        let tuning = Tuning {
            manifolds,
            config,
            substeps,
            h,
            inv_h,
            soft,
            joint_biased: push(joint_soft, true),
            joint_rigid: push(Softness::RIGID, false),
        };
        let mut chunks: [Chunk<'_>; MAX_WORKERS] = core::array::from_fn(|_| Chunk::default());
        let cut = share_out(
            &mut chunks,
            Split {
                bodies: &mut self.bodies,
                active: self.partition.active(),
                constraints: &mut self.constraints[..held.contacts],
                contact_source: self.partition.contact_source(),
                joints: self.joints.rows_mut(),
                ends: self.partition.ends(),
            },
        );
        fanout.for_each(&mut chunks[..cut], |chunk| chunk::run(chunk, &tuning));

        self.record_loads();
        for at in self.partition.contact_order() {
            self.constraints[*at as usize].store(manifolds);
        }
        self.joints.store(joints);
    }

    /// Total up what each constraint delivered, in the order the manifolds
    /// were given, so a caller reading them back sees the pairs in the order
    /// the narrow phase found them however the solve was split.
    fn record_loads(&mut self) {
        self.loads.clear();
        for at in self.partition.contact_order() {
            let constraint = &self.constraints[*at as usize];
            if constraint.is_empty() {
                continue;
            }
            self.loads.push(ContactLoad {
                manifold: constraint.manifold(),
                impulse: constraint.delivered(),
            });
        }
    }
}

/// What one solve reads and writes back.
pub(crate) struct Work<'a> {
    pub(crate) manifolds: &'a mut [Manifold],
    pub(crate) joints: &'a mut [Joint],
    pub(crate) islands: &'a mut Islands,
    pub(crate) config: &'a SimConfig,
    pub(crate) dt: f32,
}

/// The step's arrays, ready to be cut into runs.
struct Split<'a> {
    bodies: &'a mut [SolverBody],
    active: &'a [u32],
    constraints: &'a mut [ContactConstraint],
    contact_source: &'a [u32],
    joints: &'a mut [Prepared],
    ends: &'a [Ends],
}

/// Cut the step's arrays into one chunk per island run, and return how many
/// chunks that came to.
fn share_out<'a>(out: &mut [Chunk<'a>], split: Split<'a>) -> usize {
    let Split {
        bodies,
        mut active,
        mut constraints,
        mut contact_source,
        mut joints,
        ends,
    } = split;
    let whole = Bodies::new(bodies);
    let mut from = Ends::default();
    let cut = ends.len().min(out.len());
    for (chunk, to) in out.iter_mut().zip(&ends[..cut]) {
        let (own_active, rest) = active.split_at(to.bodies - from.bodies);
        active = rest;
        let (own_constraints, rest) =
            core::mem::take(&mut constraints).split_at_mut(to.contacts - from.contacts);
        constraints = rest;
        let (own_source, rest) = contact_source.split_at(to.contacts - from.contacts);
        contact_source = rest;
        let (own_joints, rest) = core::mem::take(&mut joints).split_at_mut(to.joints - from.joints);
        joints = rest;
        *chunk = Chunk {
            // SAFETY: every chunk writes only the body slots in the islands it
            // was given, and the partition puts each slot the step moves in
            // exactly one island, so no two chunks write the same entry. The
            // slots several chunks do reach -- the immovable and sleeping
            // bodies a contact leans on -- are refused by
            // `SolverBody::apply_impulse` and never integrated, so they are
            // read by all and written by none.
            bodies: unsafe { whole.share() },
            active: own_active,
            constraints: own_constraints,
            contact_source: own_source,
            joints: own_joints,
        };
        from = *to;
    }
    cut
}

#[cfg(test)]
mod tests;
