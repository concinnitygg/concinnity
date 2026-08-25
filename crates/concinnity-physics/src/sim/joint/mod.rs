// concinnity-physics/src/sim/joint/mod.rs
//
// Joints: the constraints a caller builds between two bodies, and the rows the
// solver holds them with.
//
// They live beside the contact path rather than after it because that is the
// only place they can work. A joint solved once the contacts are done fights
// whatever the contacts just decided and shows up as a hinge that hums, so a
// joint is prepared with the contacts, warm started with them, and solved in
// the same substep loop.
//
// The split here is by what each part needs. `frame` reduces an authored spec
// to something the solver can read and needs no bodies at all; `rows` is the
// linear algebra a constraint row is solved through and names no joint kind;
// `constraint` is the only part that touches the solver's dense arrays. The
// storage below is the fourth part: a flat list, ordered by the slot a joint
// was given, so the solve visits joints in an order nothing about the scene
// can change.

mod constraint;
mod frame;
mod rows;

use alloc::vec::Vec;

use crate::sim::math::Vec3;

pub(crate) use constraint::{JointSolver, Prepared};
pub(crate) use frame::{JointFrame, JointKind};
pub(crate) use rows::Push;

/// What a joint carried out of the last substep, so the next one starts from
/// the answer rather than from nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct JointImpulses {
    /// The rows that hold the anchors together.
    pub(crate) linear: Vec3,
    /// The rows that hold the orientations together.
    pub(crate) angular: Vec3,
    /// The two bounds, each of which may only push.
    pub(crate) lower: f32,
    pub(crate) upper: f32,
    pub(crate) motor: f32,
}

/// One constraint between two bodies.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Joint {
    pub(crate) a: u32,
    pub(crate) b: u32,
    /// Where the joint attaches, in each body's own frame.
    pub(crate) anchor_a: Vec3,
    pub(crate) anchor_b: Vec3,
    pub(crate) frame: JointFrame,
    pub(crate) impulses: JointImpulses,
}

impl Joint {
    /// The body at the other end, for a caller holding one of them.
    pub(crate) fn other(&self, slot: u32) -> Option<u32> {
        if self.a == slot {
            Some(self.b)
        } else if self.b == slot {
            Some(self.a)
        } else {
            None
        }
    }
}

/// Every joint in the simulation, in the order they were added.
///
/// A flat list rather than a pool: a joint carries no handle a caller can hold
/// (removing a body is what removes its joints), and the order is the solve
/// order, which a pool's free-list reuse would let a scene's history change.
pub(crate) struct JointSet {
    joints: Vec<Joint>,
}

impl JointSet {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        JointSet {
            joints: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.joints.len()
    }

    pub(crate) fn as_slice(&self) -> &[Joint] {
        &self.joints
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [Joint] {
        &mut self.joints
    }

    /// Add a joint. Growing past the reservation happens while a world is
    /// being built, never while it is stepping.
    pub(crate) fn push(&mut self, joint: Joint) {
        self.joints.push(joint);
    }

    /// Drop every joint attached to `slot`, which is what a removed body owes
    /// the ones it was holding. Returns whether anything was dropped.
    pub(crate) fn remove_incident(&mut self, slot: u32) -> bool {
        let before = self.joints.len();
        self.joints
            .retain(|joint| joint.a != slot && joint.b != slot);
        self.joints.len() != before
    }

    pub(crate) fn reserved_bytes(&self) -> u64 {
        (self.joints.capacity() * size_of::<Joint>()) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::JointSpec;
    use crate::sim::math::Quat;

    fn joint(a: u32, b: u32) -> Joint {
        Joint {
            a,
            b,
            anchor_a: Vec3::ZERO,
            anchor_b: Vec3::ZERO,
            frame: JointFrame::new(JointSpec::Fixed, Quat::IDENTITY, Quat::IDENTITY),
            impulses: JointImpulses::default(),
        }
    }

    #[test]
    fn a_joint_names_the_body_at_the_other_end() {
        let j = joint(3, 7);
        assert_eq!(j.other(3), Some(7));
        assert_eq!(j.other(7), Some(3));
        assert_eq!(j.other(4), None);
    }

    // The contract the driver relies on: a removed body takes its joints with
    // it, and leaves everyone else's alone.
    #[test]
    fn removing_a_body_drops_only_the_joints_it_was_in() {
        let mut set = JointSet::with_capacity(4);
        for pair in [(0, 1), (1, 2), (2, 3)] {
            set.push(joint(pair.0, pair.1));
        }
        assert!(set.remove_incident(1));
        let left: Vec<(u32, u32)> = set.as_slice().iter().map(|j| (j.a, j.b)).collect();
        assert_eq!(left, [(2, 3)]);
        assert!(!set.remove_incident(9), "a body in no joint drops nothing");
        assert_eq!(set.len(), 1);
        assert!(set.reserved_bytes() > 0);
    }

    // Solve order is joint order, so a removal must not shuffle the survivors.
    #[test]
    fn removal_keeps_the_surviving_joints_in_the_order_they_were_added() {
        let mut set = JointSet::with_capacity(8);
        for pair in [(0, 1), (2, 3), (4, 5), (2, 6), (7, 8)] {
            set.push(joint(pair.0, pair.1));
        }
        set.remove_incident(2);
        let left: Vec<(u32, u32)> = set.as_slice().iter().map(|j| (j.a, j.b)).collect();
        assert_eq!(left, [(0, 1), (4, 5), (7, 8)]);
        set.as_mut_slice()[0].impulses.lower = 2.0;
        assert_eq!(set.as_slice()[0].impulses.lower, 2.0);
    }
}
