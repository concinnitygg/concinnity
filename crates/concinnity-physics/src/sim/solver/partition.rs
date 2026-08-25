// concinnity-physics/src/sim/solver/partition.rs
//
// The split of a step's solve into pieces that can run at the same time.
//
// Sequential impulses are sequential on purpose: every constraint answers the
// velocities the ones before it left behind, and that is what makes a stack
// converge in a handful of passes rather than a hundred. So the split cannot be
// over constraints. It is over islands -- the groups union-find makes of the
// bodies a contact or a joint connects -- because two islands share no body the
// step moves, and therefore share nothing an impulse could travel through.
//
// That makes the split free of consequence rather than merely safe: within an
// island the constraints are visited in exactly the order the one-threaded
// solve would visit them, and between islands there is nothing to order. The
// same world solved on one worker and on twelve lands on the same bits, which
// is the only reason a fan-out is allowed near this stage at all.
//
// What the split costs is a handful of counting passes: number the islands in
// the order their lowest-numbered body appears, count what each holds, and lay
// the bodies and constraints out grouped by island so a worker's share is a run
// rather than a scatter. Islands go to workers in that order, cut where the
// running cost passes each worker's share, so a world of one big stack and a
// world of a hundred small ones both get the best split their shape allows --
// which for the first one is no split at all.

use alloc::vec::Vec;

use crate::sim::contact::Manifold;
use crate::sim::island::Islands;
use crate::sim::joint::Joint;

use super::bodies::SolverBody;

/// A contact or joint the step cannot move either side of, so no island owns
/// it and no worker solves it.
const UNOWNED: u32 = u32::MAX;

/// What a contact is worth beside a body when the islands are shared out. The
/// solve spends about twenty times as long on a constraint as the integration
/// does on a body, so a chunk balanced by body count alone would be lopsided
/// wherever the contacts are.
const CONTACT_WEIGHT: u64 = 16;

/// Where one worker's run of each grouped array ends.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Ends {
    pub(crate) bodies: usize,
    pub(crate) contacts: usize,
    pub(crate) joints: usize,
}

/// The islands a step's solve breaks into, and how they are shared out.
pub(crate) struct Partition {
    /// Island number given to each island's union-find root. Every entry is
    /// `UNOWNED` between steps, so a step resets only what it used.
    island_of_root: Vec<u32>,
    /// Roots numbered this step, which is what makes that reset cheap.
    roots: Vec<u32>,
    /// Per island, first its contents and then, once scanned, where its run
    /// starts. Used as a cursor while the runs are filled.
    bodies_at: Vec<u32>,
    contacts_at: Vec<u32>,
    joints_at: Vec<u32>,
    /// Island of each candidate manifold and joint, or `UNOWNED`.
    contact_island: Vec<u32>,
    joint_island: Vec<u32>,
    /// The step's active slots, grouped by island.
    active: Vec<u32>,
    /// Manifold each constraint is built from, and joint each row is built
    /// from: both indexed by the grouped position.
    contact_source: Vec<u32>,
    joint_source: Vec<u32>,
    /// Grouped position of each constraint, in manifold order. What a caller
    /// walks to report the step's contacts the way it found them.
    contact_order: Vec<u32>,
    ends: Vec<Ends>,
    islands: usize,
}

impl Partition {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Partition {
            island_of_root: alloc::vec![UNOWNED; capacity],
            roots: Vec::with_capacity(capacity),
            bodies_at: alloc::vec![0; capacity],
            contacts_at: alloc::vec![0; capacity],
            joints_at: alloc::vec![0; capacity],
            contact_island: Vec::with_capacity(capacity * 2),
            joint_island: Vec::with_capacity(capacity),
            active: Vec::with_capacity(capacity),
            contact_source: Vec::with_capacity(capacity * 2),
            joint_source: Vec::with_capacity(capacity),
            contact_order: Vec::with_capacity(capacity * 2),
            ends: Vec::with_capacity(capacity),
            islands: 0,
        }
    }

    pub(crate) fn reserved_bytes(&self) -> u64 {
        let words = self.island_of_root.capacity()
            + self.roots.capacity()
            + self.bodies_at.capacity()
            + self.contacts_at.capacity()
            + self.joints_at.capacity()
            + self.contact_island.capacity()
            + self.joint_island.capacity()
            + self.active.capacity()
            + self.contact_source.capacity()
            + self.joint_source.capacity()
            + self.contact_order.capacity();
        (words * size_of::<u32>() + self.ends.capacity() * size_of::<Ends>()) as u64
    }

    /// The step's active slots, grouped so each island's are a run.
    pub(crate) fn active(&self) -> &[u32] {
        &self.active
    }

    /// The manifold each constraint slot is built from.
    pub(crate) fn contact_source(&self) -> &[u32] {
        &self.contact_source
    }

    /// The joint each prepared row is built from.
    pub(crate) fn joint_source(&self) -> &[u32] {
        &self.joint_source
    }

    /// Where each constraint went, in manifold order.
    pub(crate) fn contact_order(&self) -> &[u32] {
        &self.contact_order
    }

    /// Where each worker's runs end. One entry per chunk of work, never more
    /// than the workers asked for and never more than there are islands.
    pub(crate) fn ends(&self) -> &[Ends] {
        &self.ends
    }

    /// Islands the step's solve broke into.
    #[cfg(test)]
    pub(crate) fn islands(&self) -> usize {
        self.islands
    }

    /// Group this step's bodies, contacts and joints by island and share the
    /// islands out over at most `workers` chunks.
    pub(crate) fn build(&mut self, work: Work<'_>, islands: &mut Islands, workers: usize) {
        let Work {
            bodies,
            taken,
            active,
            manifolds,
            joints,
        } = work;
        self.active.clear();
        self.contact_source.clear();
        self.joint_source.clear();
        self.contact_order.clear();
        self.contact_island.clear();
        self.joint_island.clear();
        self.ends.clear();
        self.islands = 0;
        if active.is_empty() {
            return;
        }

        let moves = |slot: u32| taken[slot as usize] && bodies[slot as usize].simulated;
        islands.clear();
        for manifold in manifolds {
            if moves(manifold.a) && moves(manifold.b) {
                islands.union(manifold.a, manifold.b);
            }
        }
        for joint in joints {
            if moves(joint.a) && moves(joint.b) {
                islands.union(joint.a, joint.b);
            }
        }

        // Number the islands in the order their lowest-numbered body appears,
        // so the grouping depends on the world and not on how union-find
        // happened to hang the trees.
        for &slot in active {
            let root = islands.find(slot);
            let island = self.island_of_root[root as usize];
            if island == UNOWNED {
                self.island_of_root[root as usize] = self.islands as u32;
                self.roots.push(root);
                self.bodies_at[self.islands] = 1;
                self.contacts_at[self.islands] = 0;
                self.joints_at[self.islands] = 0;
                self.islands += 1;
            } else {
                self.bodies_at[island as usize] += 1;
            }
        }

        for manifold in manifolds {
            let island = self.owner(islands, manifold.a, manifold.b, &moves);
            self.contact_island.push(island);
            if island != UNOWNED {
                self.contacts_at[island as usize] += 1;
            }
        }
        for joint in joints {
            let island = self.owner(islands, joint.a, joint.b, &moves);
            self.joint_island.push(island);
            if island != UNOWNED {
                self.joints_at[island as usize] += 1;
            }
        }

        self.share_out(workers);
        self.place(islands, active);

        for &root in &self.roots {
            self.island_of_root[root as usize] = UNOWNED;
        }
        self.roots.clear();
    }

    /// The island a constraint belongs to: whichever of its two bodies the
    /// step can move. Both movable puts them in the same island by
    /// construction, and neither movable means nobody solves it.
    fn owner(&self, islands: &mut Islands, a: u32, b: u32, moves: &impl Fn(u32) -> bool) -> u32 {
        let slot = if moves(a) {
            a
        } else if moves(b) {
            b
        } else {
            return UNOWNED;
        };
        self.island_of_root[islands.find(slot) as usize]
    }

    /// Turn the per-island counts into the offset each island's run starts at,
    /// and cut the islands into chunks of about equal cost along the way.
    fn share_out(&mut self, workers: usize) {
        let cost = |bodies: u32, contacts: u32, joints: u32| {
            bodies as u64 + (contacts as u64 + joints as u64) * CONTACT_WEIGHT
        };
        let total: u64 = (0..self.islands)
            .map(|i| cost(self.bodies_at[i], self.contacts_at[i], self.joints_at[i]))
            .sum();
        let workers = workers.clamp(1, self.islands).max(1) as u64;

        let (mut bodies, mut contacts, mut joints) = (0usize, 0usize, 0usize);
        let mut running = 0u64;
        let mut cut = 1u64;
        for island in 0..self.islands {
            let held = (
                self.bodies_at[island],
                self.contacts_at[island],
                self.joints_at[island],
            );
            self.bodies_at[island] = bodies as u32;
            self.contacts_at[island] = contacts as u32;
            self.joints_at[island] = joints as u32;
            bodies += held.0 as usize;
            contacts += held.1 as usize;
            joints += held.2 as usize;
            running += cost(held.0, held.1, held.2);
            // Close a chunk once it holds its share, leaving at least one
            // island for every chunk still to come.
            let left = (self.islands - island - 1) as u64;
            if cut < workers && left >= workers - cut && running * workers >= total * cut {
                self.ends.push(Ends {
                    bodies,
                    contacts,
                    joints,
                });
                cut += 1;
            }
        }
        self.ends.push(Ends {
            bodies,
            contacts,
            joints,
        });
    }

    /// Lay the active slots and the constraint sources out grouped by island.
    fn place(&mut self, islands: &mut Islands, active: &[u32]) {
        let ends = *self.ends.last().expect("share_out pushes a final cut");
        self.active.resize(ends.bodies, 0);
        self.contact_source.resize(ends.contacts, 0);
        self.joint_source.resize(ends.joints, 0);

        for &slot in active {
            let island = self.island_of_root[islands.find(slot) as usize];
            let at = &mut self.bodies_at[island as usize];
            self.active[*at as usize] = slot;
            *at += 1;
        }
        for (index, &island) in self.contact_island.iter().enumerate() {
            if island == UNOWNED {
                continue;
            }
            let at = &mut self.contacts_at[island as usize];
            self.contact_source[*at as usize] = index as u32;
            self.contact_order.push(*at);
            *at += 1;
        }
        for (index, &island) in self.joint_island.iter().enumerate() {
            if island == UNOWNED {
                continue;
            }
            let at = &mut self.joints_at[island as usize];
            self.joint_source[*at as usize] = index as u32;
            *at += 1;
        }
    }
}

/// What one partition reads.
pub(crate) struct Work<'a> {
    pub(crate) bodies: &'a [SolverBody],
    pub(crate) taken: &'a [bool],
    pub(crate) active: &'a [u32],
    pub(crate) manifolds: &'a [Manifold],
    pub(crate) joints: &'a [Joint],
}

#[cfg(test)]
mod tests {
    use super::*;

    fn movable() -> SolverBody {
        SolverBody {
            simulated: true,
            ..SolverBody::IMMOVABLE
        }
    }

    fn manifold(a: u32, b: u32) -> Manifold {
        Manifold::new(a, b)
    }

    struct Fixture {
        bodies: alloc::vec::Vec<SolverBody>,
        taken: alloc::vec::Vec<bool>,
        islands: Islands,
    }

    /// `simulated` says which slots the step moves; the rest stand for walls.
    fn fixture(simulated: &[bool]) -> Fixture {
        Fixture {
            bodies: simulated
                .iter()
                .map(|&s| if s { movable() } else { SolverBody::IMMOVABLE })
                .collect(),
            taken: alloc::vec![true; simulated.len()],
            islands: Islands::with_capacity(simulated.len()),
        }
    }

    fn active_of(simulated: &[bool]) -> alloc::vec::Vec<u32> {
        simulated
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s)
            .map(|(i, _)| i as u32)
            .collect()
    }

    fn build(
        partition: &mut Partition,
        fixture: &mut Fixture,
        simulated: &[bool],
        manifolds: &[Manifold],
        workers: usize,
    ) {
        let active = active_of(simulated);
        partition.build(
            Work {
                bodies: &fixture.bodies,
                taken: &fixture.taken,
                active: &active,
                manifolds,
                joints: &[],
            },
            &mut fixture.islands,
            workers,
        );
    }

    #[test]
    fn untouched_bodies_are_one_island_each() {
        let simulated = [true, true, true];
        let mut fixture = fixture(&simulated);
        let mut partition = Partition::with_capacity(3);
        build(&mut partition, &mut fixture, &simulated, &[], 4);
        assert_eq!(partition.islands(), 3);
        assert_eq!(partition.active(), [0, 1, 2]);
        // Three islands cannot fill four chunks, so only three are cut.
        assert_eq!(partition.ends().len(), 3);
    }

    // The shape the whole split rests on: a stack is one island however many
    // contacts hold it together.
    #[test]
    fn a_chain_of_contacts_is_one_island() {
        let simulated = [true, true, true, true];
        let mut fixture = fixture(&simulated);
        let mut partition = Partition::with_capacity(4);
        let manifolds = [manifold(0, 1), manifold(1, 2), manifold(2, 3)];
        build(&mut partition, &mut fixture, &simulated, &manifolds, 4);
        assert_eq!(partition.islands(), 1);
        assert_eq!(partition.ends().len(), 1, "one island cannot be split");
        assert_eq!(partition.contact_source(), [0, 1, 2]);
        assert_eq!(partition.contact_order(), [0, 1, 2]);
    }

    // The case island parallelism exists for, and the one the immovable body
    // would ruin if it joined the islands it is leaned on by.
    #[test]
    fn stacks_sharing_a_floor_stay_separate_islands() {
        // Slot 0 is the floor; 1-2 and 3-4 are two stacks resting on it.
        let simulated = [false, true, true, true, true];
        let mut fixture = fixture(&simulated);
        let mut partition = Partition::with_capacity(5);
        let manifolds = [
            manifold(0, 1),
            manifold(0, 3),
            manifold(1, 2),
            manifold(3, 4),
        ];
        build(&mut partition, &mut fixture, &simulated, &manifolds, 2);
        assert_eq!(partition.islands(), 2);
        assert_eq!(partition.ends().len(), 2);
        // Each stack's bodies land together, in slot order within the island.
        assert_eq!(partition.active(), [1, 2, 3, 4]);
        assert_eq!(partition.ends()[0].bodies, 2);
        // The two contacts against the floor go to the stack that owns the
        // moving half, and the order the manifolds were given is recoverable.
        assert_eq!(partition.contact_source(), [0, 2, 1, 3]);
        assert_eq!(partition.contact_order(), [0, 2, 1, 3]);
    }

    #[test]
    fn a_contact_between_two_walls_belongs_to_nobody() {
        let simulated = [false, false, true];
        let mut fixture = fixture(&simulated);
        let mut partition = Partition::with_capacity(3);
        let manifolds = [manifold(0, 1), manifold(1, 2)];
        build(&mut partition, &mut fixture, &simulated, &manifolds, 2);
        assert_eq!(partition.islands(), 1);
        assert_eq!(partition.contact_source(), [1]);
    }

    // Every slot the step moves must land in exactly one chunk, which is the
    // claim the shared body handle is sound on.
    #[test]
    fn chunks_cover_every_active_slot_exactly_once() {
        let simulated = alloc::vec![true; 32];
        let mut fixture = fixture(&simulated);
        let mut partition = Partition::with_capacity(32);
        let manifolds: alloc::vec::Vec<Manifold> =
            (0..31).step_by(2).map(|i| manifold(i, i + 1)).collect();
        build(&mut partition, &mut fixture, &simulated, &manifolds, 4);
        let mut seen = alloc::vec![0u32; 32];
        let mut from = 0usize;
        for ends in partition.ends() {
            for &slot in &partition.active()[from..ends.bodies] {
                seen[slot as usize] += 1;
            }
            from = ends.bodies;
        }
        assert_eq!(from, 32);
        assert!(seen.iter().all(|&count| count == 1), "{seen:?}");
    }

    // A rebuild must not remember the last one: the root numbering is reset
    // in island time, so a stale entry would silently merge two steps.
    #[test]
    fn a_rebuild_forgets_the_previous_grouping() {
        let simulated = [true, true, true, true];
        let mut fixture = fixture(&simulated);
        let mut partition = Partition::with_capacity(4);
        let joined = [manifold(0, 1), manifold(1, 2), manifold(2, 3)];
        build(&mut partition, &mut fixture, &simulated, &joined, 4);
        assert_eq!(partition.islands(), 1);
        build(&mut partition, &mut fixture, &simulated, &[], 4);
        assert_eq!(partition.islands(), 4);
        assert!(partition.reserved_bytes() > 0);
    }

    #[test]
    fn a_step_with_nothing_to_move_makes_no_chunks() {
        let simulated = [false, false];
        let mut fixture = fixture(&simulated);
        let mut partition = Partition::with_capacity(2);
        build(
            &mut partition,
            &mut fixture,
            &simulated,
            &[manifold(0, 1)],
            4,
        );
        assert_eq!(partition.islands(), 0);
        assert!(partition.ends().is_empty());
        assert!(partition.active().is_empty());
    }
}
