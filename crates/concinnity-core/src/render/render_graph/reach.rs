// src/render_graph/reach.rs
//
// Transitive-closure bitset over a DAG whose edges all point from a lower
// index to a higher one, which is what a topologically sorted pass list gives
// us. One `u64` word per 64 passes per row, so the whole relation for a ~30
// pass frame graph is 30 words: cheap enough to precompute once per compile and
// answer every "is A ordered before B?" query in constant time.
//
// Two relations are built from it (see `super::schedule`): the dependency
// closure, which is what correctness requires, and the schedule closure, which
// is what the per-queue order plus the cross-queue signal / wait pairs deliver.

use alloc::vec;
use alloc::vec::Vec;

const WORD_BITS: usize = 64;

// Precomputed reachability over a forward-edged DAG. `reaches(a, b)` is true
// when some path leads from `a` to `b`; a node does not reach itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Reachability {
    n: usize,
    words: usize,
    // Row-major: row `i` occupies `bits[i * words .. (i + 1) * words]` and holds
    // one bit per node `i` reaches.
    bits: Vec<u64>,
}

impl Reachability {
    // Build the closure of `edges`, where `edges[i]` lists the successors of
    // `i`. Every edge must point forward (`i < j`), which lets one reverse scan
    // finish each row: a successor's row is already complete when we fold it in.
    pub(crate) fn new(n: usize, edges: &[Vec<usize>]) -> Self {
        let words = n.div_ceil(WORD_BITS);
        let mut bits = vec![0u64; n * words];
        for i in (0..n).rev() {
            for &j in &edges[i] {
                debug_assert!(
                    j > i,
                    "Reachability::new needs forward edges, got {i} -> {j}"
                );
                // Rows never alias: `j > i` puts row `j` strictly after row `i`.
                let (head, tail) = bits.split_at_mut(j * words);
                let row_i = &mut head[i * words..(i + 1) * words];
                let row_j = &tail[..words];
                row_i[j / WORD_BITS] |= 1u64 << (j % WORD_BITS);
                for w in 0..words {
                    row_i[w] |= row_j[w];
                }
            }
        }
        Self { n, words, bits }
    }

    // Number of nodes the relation covers.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.n
    }

    // Whether a path leads from `from` to `to`. False for equal indices and for
    // anything out of range.
    pub(crate) fn reaches(&self, from: usize, to: usize) -> bool {
        if from >= self.n || to >= self.n {
            return false;
        }
        self.bits[from * self.words + to / WORD_BITS] & (1u64 << (to % WORD_BITS)) != 0
    }

    // Whether one of the two nodes reaches the other, i.e. the DAG fixes their
    // relative order.
    pub(crate) fn ordered(&self, a: usize, b: usize) -> bool {
        self.reaches(a, b) || self.reaches(b, a)
    }

    // Whether the two distinct nodes may run at the same time: neither reaches
    // the other, so nothing orders them. False for equal indices and for
    // anything out of range, which no node occupies.
    pub(crate) fn concurrent(&self, a: usize, b: usize) -> bool {
        a != b && a < self.n && b < self.n && !self.ordered(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Adjacency for `n` nodes from a list of forward edges.
    fn edges(n: usize, pairs: &[(usize, usize)]) -> Vec<Vec<usize>> {
        let mut e = vec![Vec::new(); n];
        for &(a, b) in pairs {
            e[a].push(b);
        }
        e
    }

    #[test]
    fn a_chain_reaches_transitively() {
        let r = Reachability::new(4, &edges(4, &[(0, 1), (1, 2), (2, 3)]));
        assert_eq!(r.len(), 4);
        for a in 0..4 {
            for b in 0..4 {
                assert_eq!(r.reaches(a, b), a < b, "{a} -> {b}");
            }
        }
    }

    #[test]
    fn a_node_does_not_reach_itself() {
        let r = Reachability::new(2, &edges(2, &[(0, 1)]));
        assert!(!r.reaches(0, 0));
        assert!(!r.reaches(1, 1));
        assert!(
            !r.concurrent(0, 0),
            "one node is not concurrent with itself"
        );
    }

    #[test]
    fn parallel_branches_are_concurrent() {
        // 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3. Nothing orders 1 against 2.
        let r = Reachability::new(4, &edges(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]));
        assert!(r.concurrent(1, 2));
        assert!(!r.ordered(1, 2));
        assert!(r.ordered(0, 3));
        assert!(r.reaches(0, 3));
        assert!(!r.reaches(3, 0));
    }

    #[test]
    fn an_isolated_node_reaches_nothing() {
        let r = Reachability::new(3, &edges(3, &[(0, 1)]));
        assert!(!r.reaches(2, 0));
        assert!(!r.reaches(0, 2));
        assert!(r.concurrent(0, 2));
        assert!(r.concurrent(1, 2));
    }

    #[test]
    fn an_empty_relation_answers_false() {
        let r = Reachability::new(0, &[]);
        assert_eq!(r.len(), 0);
        assert!(!r.reaches(0, 0));
        assert!(!r.ordered(0, 1));
        assert!(!r.concurrent(0, 1));
    }

    #[test]
    fn rows_span_more_than_one_word() {
        // 70 nodes in a chain crosses the 64-bit row boundary, which is where a
        // single-word implementation would silently drop the tail.
        let n = 70;
        let pairs: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let r = Reachability::new(n, &edges(n, &pairs));
        assert!(r.reaches(0, 69));
        assert!(r.reaches(2, 65));
        assert!(!r.reaches(65, 2));
    }
}
