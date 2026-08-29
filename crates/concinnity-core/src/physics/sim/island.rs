// Which bodies are resting on which, so a settled group can stop being
// simulated together.
//
// Sleeping has to be an island decision rather than a per-body one. A body
// that stops moving before the one leaning on it would otherwise sleep under a
// load that is still shifting, and wake a step later looking like a glitch.
// Grouping the bodies a contact connects and requiring the whole group to be
// still makes the stack settle as one.
//
// Union-find over body slots, in two fixed arrays: no allocation while
// stepping, and no traversal order that could vary between runs.

use alloc::vec::Vec;

pub(crate) struct Islands {
    parent: Vec<u32>,
    rank: Vec<u8>,
    /// Per root, whether every member has been still long enough. Only
    /// meaningful once every union for the step has been recorded.
    ready: Vec<bool>,
}

impl Islands {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Islands {
            parent: (0..capacity as u32).collect(),
            rank: alloc::vec![0; capacity],
            ready: alloc::vec![true; capacity],
        }
    }

    pub(crate) fn reserved_bytes(&self) -> u64 {
        (self.parent.capacity() * size_of::<u32>() + self.rank.capacity() + self.ready.capacity())
            as u64
    }

    /// Start a step: every body is its own island again.
    pub(crate) fn clear(&mut self) {
        for (slot, parent) in self.parent.iter_mut().enumerate() {
            *parent = slot as u32;
        }
        self.rank.fill(0);
        self.ready.fill(true);
    }

    pub(crate) fn find(&mut self, slot: u32) -> u32 {
        let mut current = slot;
        while self.parent[current as usize] != current {
            // Path halving: point at the grandparent on the way up, which
            // keeps the trees flat without a second pass.
            let grandparent = self.parent[self.parent[current as usize] as usize];
            self.parent[current as usize] = grandparent;
            current = grandparent;
        }
        current
    }

    pub(crate) fn union(&mut self, a: u32, b: u32) {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.rank[ra as usize] < self.rank[rb as usize] {
            core::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb as usize] = ra;
        if self.rank[ra as usize] == self.rank[rb as usize] {
            self.rank[ra as usize] += 1;
        }
    }

    /// Record whether one body is still. Call after every union for the step.
    pub(crate) fn mark(&mut self, slot: u32, still: bool) {
        let root = self.find(slot);
        self.ready[root as usize] &= still;
    }

    /// Whether every body in this one's island was marked still.
    pub(crate) fn island_is_still(&mut self, slot: u32) -> bool {
        let root = self.find(slot);
        self.ready[root as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_untouched_body_is_its_own_island() {
        let mut islands = Islands::with_capacity(4);
        islands.clear();
        assert_eq!(islands.find(2), 2);
        assert_ne!(islands.find(1), islands.find(2));
    }

    #[test]
    fn unions_are_transitive_across_a_chain() {
        let mut islands = Islands::with_capacity(8);
        islands.clear();
        islands.union(0, 1);
        islands.union(1, 2);
        islands.union(5, 6);
        assert_eq!(islands.find(0), islands.find(2));
        assert_ne!(islands.find(0), islands.find(5));
        assert_eq!(islands.find(5), islands.find(6));
        // Unioning an already joined pair changes nothing.
        islands.union(2, 0);
        assert_eq!(islands.find(0), islands.find(2));
    }

    // The point of grouping: one restless body holds its whole island awake.
    #[test]
    fn one_moving_body_keeps_its_whole_island_awake() {
        let mut islands = Islands::with_capacity(8);
        islands.clear();
        for slot in 0..4 {
            islands.union(slot, slot + 1);
        }
        for slot in 0..5 {
            islands.mark(slot, slot != 3);
        }
        for slot in 0..5 {
            assert!(!islands.island_is_still(slot), "slot {slot}");
        }
    }

    #[test]
    fn an_island_whose_members_are_all_still_is_ready() {
        let mut islands = Islands::with_capacity(8);
        islands.clear();
        islands.union(0, 1);
        islands.union(2, 3);
        for slot in 0..4 {
            islands.mark(slot, slot < 2);
        }
        assert!(islands.island_is_still(0));
        assert!(islands.island_is_still(1));
        assert!(!islands.island_is_still(2));
    }

    #[test]
    fn clearing_forgets_last_steps_grouping() {
        let mut islands = Islands::with_capacity(4);
        islands.clear();
        islands.union(0, 1);
        islands.mark(0, false);
        islands.clear();
        assert_ne!(islands.find(0), islands.find(1));
        islands.mark(0, true);
        assert!(islands.island_is_still(0));
        assert!(islands.reserved_bytes() > 0);
    }

    // Long chains must stay cheap to query, which is what path halving buys.
    #[test]
    fn a_long_chain_still_resolves_to_one_root() {
        let mut islands = Islands::with_capacity(256);
        islands.clear();
        for slot in 0..255 {
            islands.union(slot, slot + 1);
        }
        let root = islands.find(0);
        for slot in 0..256 {
            assert_eq!(islands.find(slot), root, "slot {slot}");
        }
    }
}
