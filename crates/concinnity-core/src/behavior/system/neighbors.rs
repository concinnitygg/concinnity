// The neighbor indexes: one per distinct query tag set, shared by every program
// that declares it, since the same tags select the same entities. Assigned when
// the programs change, reset every tick.

use alloc::vec::Vec;

use crate::behavior::Program;
use crate::behavior::spatial::NeighborIndex;

#[derive(Debug, Default)]
pub(super) struct Neighbors {
    // Parallel: the tag set each index serves, and the index.
    tags: Vec<Vec<u8>>,
    indexes: Vec<NeighborIndex>,
    // Per program, per declared query, which index it asks.
    of: Vec<Vec<usize>>,
}

impl Neighbors {
    // Point every declared query at the index for its tag set. An index keeps
    // its storage for as long as some program still declares its tags.
    pub(super) fn assign(&mut self, programs: &[Program]) {
        let mut k = 0;
        while k < self.tags.len() {
            if programs.iter().any(|p| p.queries.contains(&self.tags[k])) {
                k += 1;
            } else {
                self.tags.swap_remove(k);
                self.indexes.swap_remove(k);
            }
        }
        self.of.resize_with(programs.len(), Vec::new);
        for (of, program) in self.of.iter_mut().zip(programs) {
            of.clear();
            for tags in &program.queries {
                let slot = match self.tags.iter().position(|t| t == tags) {
                    Some(slot) => slot,
                    None => {
                        self.tags.push(tags.clone());
                        self.indexes.push(NeighborIndex::default());
                        self.tags.len() - 1
                    }
                };
                of.push(slot);
            }
        }
    }

    // Forget the trees the last tick built, keeping their storage.
    pub(super) fn reset(&mut self) {
        for index in &mut self.indexes {
            index.reset();
        }
    }

    // The index a program's declared query asks.
    pub(super) fn get(&self, program: usize, query: u16) -> Option<&NeighborIndex> {
        let slot = *self.of.get(program)?.get(query as usize)?;
        self.indexes.get(slot)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    fn declaring(queries: Vec<Vec<u8>>) -> Program {
        Program {
            id: None,
            def: Default::default(),
            scope: Vec::new(),
            local_inits: Vec::new(),
            queries,
            body: Vec::new(),
            paths: Vec::new(),
            bindings: 0,
        }
    }

    fn slot(neighbors: &Neighbors, program: usize, query: u16) -> *const NeighborIndex {
        neighbors
            .get(program, query)
            .expect("every declared query has an index")
    }

    #[test]
    fn programs_declaring_the_same_tags_share_an_index() {
        let mut neighbors = Neighbors::default();
        neighbors.assign(&[
            declaring(vec![vec![1], vec![2, 3]]),
            declaring(vec![vec![2, 3]]),
        ]);
        assert_eq!(slot(&neighbors, 0, 1), slot(&neighbors, 1, 0));
        assert_ne!(slot(&neighbors, 0, 0), slot(&neighbors, 0, 1));
        assert!(neighbors.get(1, 1).is_none());
        assert!(neighbors.get(2, 0).is_none());
    }

    // Tags no program declares any more are dropped, and the survivors are
    // still found by their own tags.
    #[test]
    fn an_edit_drops_the_indexes_no_one_declares() {
        let mut neighbors = Neighbors::default();
        neighbors.assign(&[declaring(vec![vec![1], vec![2], vec![3]])]);
        neighbors.assign(&[declaring(vec![vec![3]]), declaring(vec![vec![4]])]);
        assert_eq!(neighbors.tags, vec![vec![3], vec![4]]);
        assert_ne!(slot(&neighbors, 0, 0), slot(&neighbors, 1, 0));
    }
}
