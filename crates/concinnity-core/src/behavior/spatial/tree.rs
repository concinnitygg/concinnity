// A k-d tree over one query's candidate positions, answering `nearest` and
// `count_within` without visiting every candidate.
//
// The tree only decides which candidates get tested. Each test is the same
// `distance_sq` the linear scan makes, and a node is skipped or counted whole
// only on bounds that are exact in f32: subtraction, squaring and addition all
// round monotonically, so a node box's clamped point is never farther than
// anything inside it, and its far corner never nearer. The answers therefore
// equal the linear scan's bit for bit, ties included, whatever shape the tree
// takes.

use alloc::vec::Vec;

use super::distance_sq;

// Most candidates a leaf holds before it is split.
const LEAF: usize = 16;

// A candidate as the build arranges it. `rank` is where it sits in the query's
// stable order, which is what ties are broken on.
#[derive(Debug, Clone, Copy)]
struct Point {
    pos: [f32; 3],
    rank: u32,
}

// A box over the candidates at `start..end` in leaf order. An inner node's left child follows it
// directly and `right` names the other; a leaf has `right == 0`, which the root
// is the only node that could otherwise be.
#[derive(Debug, Clone, Copy)]
struct Node {
    min: [f32; 3],
    max: [f32; 3],
    start: u32,
    end: u32,
    right: u32,
}

/// Candidate positions arranged for neighbor queries. Rebuilt in place, so its
/// storage is kept across rebuilds.
#[derive(Debug, Default)]
pub(crate) struct Tree {
    // The build's working order, which ends as leaf order.
    points: Vec<Point>,
    // The same candidates in leaf order, one array per field, which is the
    // layout the leaf scans read.
    xs: Vec<f32>,
    ys: Vec<f32>,
    zs: Vec<f32>,
    ranks: Vec<u32>,
    nodes: Vec<Node>,
    indexed: bool,
}

impl Tree {
    /// Rebuild over `points`, each a candidate's rank and position.
    ///
    /// A non-finite position leaves the tree unindexed: the linear scan's
    /// answer around a NaN depends on candidate order in ways no bound can
    /// reproduce, so such a set is answered by the scan itself.
    pub(crate) fn rebuild(&mut self, points: impl IntoIterator<Item = (u32, [f32; 3])>) {
        self.clear();
        self.points
            .extend(points.into_iter().map(|(rank, pos)| Point { pos, rank }));
        self.indexed = self
            .points
            .iter()
            .all(|p| p.pos.iter().all(|c| c.is_finite()));
        if !self.indexed {
            return;
        }
        if !self.points.is_empty() {
            self.split(0, self.points.len());
        }
        self.xs.extend(self.points.iter().map(|p| p.pos[0]));
        self.ys.extend(self.points.iter().map(|p| p.pos[1]));
        self.zs.extend(self.points.iter().map(|p| p.pos[2]));
        self.ranks.extend(self.points.iter().map(|p| p.rank));
    }

    /// Empty the tree and leave it unindexed.
    pub(crate) fn clear(&mut self) {
        self.points.clear();
        self.xs.clear();
        self.ys.clear();
        self.zs.clear();
        self.ranks.clear();
        self.nodes.clear();
        self.indexed = false;
    }

    /// Whether the queries below answer for the set it was built over.
    pub(crate) fn is_indexed(&self) -> bool {
        self.indexed
    }

    // Arrange `points[start..end]` under a new node, halving along the box's
    // widest axis until a leaf is small enough. Halving bounds the depth by
    // the bit width of the count.
    fn split(&mut self, start: usize, end: usize) {
        let (min, max) = bounds(&self.points[start..end]);
        let node = self.nodes.len();
        self.nodes.push(Node {
            min,
            max,
            start: start as u32,
            end: end as u32,
            right: 0,
        });
        let len = end - start;
        if len <= LEAF {
            return;
        }
        let axis = widest(min, max);
        let half = len / 2;
        self.points[start..end]
            .select_nth_unstable_by(half, |a, b| a.pos[axis].total_cmp(&b.pos[axis]));
        self.split(start, start + half);
        self.nodes[node].right = self.nodes.len() as u32;
        self.split(start + half, end);
    }

    /// The rank of the point nearest `point`, other than `skip`. Ties go to the
    /// lower rank. `point` must be finite.
    pub(crate) fn nearest(&self, point: [f32; 3], skip: Option<u32>) -> Option<u32> {
        let root = self.nodes.first()?;
        let mut best = None;
        self.visit_nearest(0, lower_bound(root, point), point, skip, &mut best);
        best.map(|(_, rank)| rank)
    }

    fn visit_nearest(
        &self,
        node: usize,
        bound: f32,
        point: [f32; 3],
        skip: Option<u32>,
        best: &mut Option<(f32, u32)>,
    ) {
        // An equal bound is still visited: it may hold a lower rank at the
        // same distance.
        if best.is_some_and(|(d, _)| bound > d) {
            return;
        }
        let n = self.nodes[node];
        if n.right == 0 {
            let leaf = n.start as usize..n.end as usize;
            let (xs, ys, zs) = (
                &self.xs[leaf.clone()],
                &self.ys[leaf.clone()],
                &self.zs[leaf.clone()],
            );
            for (((&x, &y), &z), &rank) in xs.iter().zip(ys).zip(zs).zip(&self.ranks[leaf]) {
                if Some(rank) == skip {
                    continue;
                }
                let d = distance_sq([x, y, z], point);
                if best.is_none_or(|(bd, br)| d < bd || (d == bd && rank < br)) {
                    *best = Some((d, rank));
                }
            }
            return;
        }
        let (left, right) = (node + 1, n.right as usize);
        let left_bound = lower_bound(&self.nodes[left], point);
        let right_bound = lower_bound(&self.nodes[right], point);
        if right_bound < left_bound {
            self.visit_nearest(right, right_bound, point, skip, best);
            self.visit_nearest(left, left_bound, point, skip, best);
        } else {
            self.visit_nearest(left, left_bound, point, skip, best);
            self.visit_nearest(right, right_bound, point, skip, best);
        }
    }

    /// How many points lie within `limit` squared distance of `point`. `point`
    /// must be finite.
    pub(crate) fn count_within(&self, point: [f32; 3], limit: f32) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        self.visit_count(0, point, limit)
    }

    fn visit_count(&self, node: usize, point: [f32; 3], limit: f32) -> usize {
        let n = self.nodes[node];
        if lower_bound(&n, point) > limit {
            return 0;
        }
        if upper_bound(&n, point) <= limit {
            return (n.end - n.start) as usize;
        }
        if n.right == 0 {
            let leaf = n.start as usize..n.end as usize;
            let (xs, ys, zs) = (
                &self.xs[leaf.clone()],
                &self.ys[leaf.clone()],
                &self.zs[leaf],
            );
            return xs
                .iter()
                .zip(ys)
                .zip(zs)
                .map(|((&x, &y), &z)| usize::from(distance_sq([x, y, z], point) <= limit))
                .sum();
        }
        self.visit_count(node + 1, point, limit) + self.visit_count(n.right as usize, point, limit)
    }
}

fn bounds(points: &[Point]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for p in points {
        for axis in 0..3 {
            min[axis] = min[axis].min(p.pos[axis]);
            max[axis] = max[axis].max(p.pos[axis]);
        }
    }
    (min, max)
}

fn widest(min: [f32; 3], max: [f32; 3]) -> usize {
    let extent = |axis: usize| max[axis] - min[axis];
    let mut axis = 0;
    for candidate in 1..3 {
        if extent(candidate) > extent(axis) {
            axis = candidate;
        }
    }
    axis
}

// The box's point nearest `point`, measured the way a candidate is: never more
// than the distance of anything in the box.
fn lower_bound(node: &Node, point: [f32; 3]) -> f32 {
    let nearest = core::array::from_fn(|axis| point[axis].max(node.min[axis]).min(node.max[axis]));
    distance_sq(nearest, point)
}

// The box's corner farthest from `point`, measured the way a candidate is:
// never less than the distance of anything in the box.
fn upper_bound(node: &Node, point: [f32; 3]) -> f32 {
    let farthest = core::array::from_fn(|axis| {
        let (lo, hi) = (node.min[axis], node.max[axis]);
        if (lo - point[axis]).abs() >= (hi - point[axis]).abs() {
            lo
        } else {
            hi
        }
    });
    distance_sq(farthest, point)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A fixed-seed generator, so every run tests the same sets.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as u32
        }

        // A coordinate on a coarse lattice, so exact distance ties are common.
        fn lattice(&mut self, cells: u32) -> f32 {
            (self.next() % cells) as f32 * 0.5
        }

        fn unit(&mut self) -> f32 {
            self.next() as f32 / (1u64 << 31) as f32
        }
    }

    fn scan_nearest(points: &[(u32, [f32; 3])], point: [f32; 3], skip: Option<u32>) -> Option<u32> {
        let mut best: Option<(f32, u32)> = None;
        for &(rank, pos) in points {
            if Some(rank) == skip {
                continue;
            }
            let d = distance_sq(pos, point);
            if best.is_none_or(|(bd, br)| d < bd || (d == bd && rank < br)) {
                best = Some((d, rank));
            }
        }
        best.map(|(_, rank)| rank)
    }

    fn scan_count(points: &[(u32, [f32; 3])], point: [f32; 3], limit: f32) -> usize {
        points
            .iter()
            .filter(|(_, pos)| distance_sq(*pos, point) <= limit)
            .count()
    }

    fn built(points: &[(u32, [f32; 3])]) -> Tree {
        let mut tree = Tree::default();
        tree.rebuild(points.iter().copied());
        tree
    }

    // Ranks are shuffled against position, so a tie is decided by rank and
    // never by where the point landed in the tree.
    fn scattered(rng: &mut Lcg, count: usize, cells: u32) -> Vec<(u32, [f32; 3])> {
        let mut ranks: Vec<u32> = (0..count as u32).collect();
        for i in (1..ranks.len()).rev() {
            ranks.swap(i, rng.next() as usize % (i + 1));
        }
        ranks
            .into_iter()
            .map(|rank| {
                (
                    rank,
                    [rng.lattice(cells), rng.lattice(cells), rng.lattice(cells)],
                )
            })
            .collect()
    }

    #[test]
    fn nearest_matches_the_scan_on_random_sets() {
        let mut rng = Lcg(7);
        for round in 0..200 {
            let count = 1 + round % 97 * 3;
            let points = scattered(&mut rng, count, 6 + round as u32 % 20);
            let tree = built(&points);
            for _ in 0..40 {
                let point = [
                    rng.lattice(40) - 5.0,
                    rng.lattice(40) - 5.0,
                    rng.unit() * 20.0 - 5.0,
                ];
                let skip = rng
                    .next()
                    .is_multiple_of(3)
                    .then(|| rng.next() % count as u32);
                assert_eq!(
                    tree.nearest(point, skip),
                    scan_nearest(&points, point, skip),
                    "round {round} at {point:?} skipping {skip:?}",
                );
            }
        }
    }

    #[test]
    fn count_within_matches_the_scan_on_random_sets() {
        let mut rng = Lcg(11);
        for round in 0..200 {
            let count = 1 + round % 97 * 3;
            let points = scattered(&mut rng, count, 6 + round as u32 % 20);
            let tree = built(&points);
            for _ in 0..40 {
                let point = [rng.lattice(30), rng.lattice(30), rng.unit() * 15.0];
                // Lattice radii land candidates exactly on the edge.
                let radius = if rng.next().is_multiple_of(2) {
                    rng.lattice(12)
                } else {
                    rng.unit() * 8.0
                };
                let limit = radius * radius;
                assert_eq!(
                    tree.count_within(point, limit),
                    scan_count(&points, point, limit),
                    "round {round} at {point:?} within {radius}",
                );
            }
        }
    }

    // Every point in one place: the split cannot separate them, so ties are
    // decided entirely by rank.
    #[test]
    fn coincident_points_answer_the_lowest_rank() {
        let points: Vec<_> = [5, 3, 9, 1, 7, 2, 8, 4, 6, 0, 11, 10]
            .into_iter()
            .map(|rank| (rank, [1.0, 1.0, 1.0]))
            .collect();
        let tree = built(&points);
        assert_eq!(tree.nearest([0.0; 3], None), Some(0));
        assert_eq!(tree.nearest([0.0; 3], Some(0)), Some(1));
        assert_eq!(tree.count_within([0.0; 3], 3.0), 12);
        assert_eq!(tree.count_within([0.0; 3], 2.999), 0);
    }

    // A point far outside every box still finds its nearest and counts nothing
    // it should not.
    #[test]
    fn a_point_outside_the_set_is_answered() {
        let mut rng = Lcg(3);
        let points = scattered(&mut rng, 64, 8);
        let tree = built(&points);
        let far = [1.0e6, -1.0e6, 3.0e5];
        assert_eq!(tree.nearest(far, None), scan_nearest(&points, far, None));
        assert_eq!(tree.count_within(far, 1.0), 0);
        assert_eq!(tree.count_within(far, f32::INFINITY), 64);
    }

    // Distances that overflow to infinity tie at infinity, and the lowest rank
    // still wins, as it does in the scan.
    #[test]
    fn overflowing_distances_tie_by_rank() {
        let points = [
            (2, [3.0e38, 0.0, 0.0]),
            (0, [-3.0e38, 0.0, 0.0]),
            (1, [0.0, 3.0e38, 0.0]),
        ];
        let tree = built(&points);
        let point = [0.0, -3.0e38, 0.0];
        assert_eq!(
            tree.nearest(point, None),
            scan_nearest(&points, point, None)
        );
    }

    #[test]
    fn a_non_finite_position_leaves_the_tree_unindexed() {
        let mut tree = Tree::default();
        tree.rebuild([(0, [0.0; 3]), (1, [f32::NAN, 0.0, 0.0])]);
        assert!(!tree.is_indexed());
        tree.rebuild([(0, [0.0; 3]), (1, [f32::INFINITY, 0.0, 0.0])]);
        assert!(!tree.is_indexed());
        tree.rebuild([(0, [0.0; 3])]);
        assert!(tree.is_indexed());
    }

    #[test]
    fn an_empty_tree_answers_nothing() {
        let tree = built(&[]);
        assert!(tree.is_indexed());
        assert_eq!(tree.nearest([0.0; 3], None), None);
        assert_eq!(tree.count_within([0.0; 3], 100.0), 0);
    }

    // The only point skipped leaves nothing to answer.
    #[test]
    fn skipping_the_only_point_answers_nothing() {
        let tree = built(&[(0, [1.0, 2.0, 3.0])]);
        assert_eq!(tree.nearest([0.0; 3], Some(0)), None);
    }

    // Rebuilding keeps the storage and forgets the previous set.
    #[test]
    fn a_rebuild_replaces_the_previous_set() {
        let mut tree = built(&[(0, [0.0; 3]), (1, [10.0, 0.0, 0.0])]);
        tree.rebuild([(0, [50.0, 0.0, 0.0])]);
        assert_eq!(tree.nearest([9.0, 0.0, 0.0], None), Some(0));
        assert_eq!(tree.count_within([0.0; 3], 400.0), 0);
    }
}
