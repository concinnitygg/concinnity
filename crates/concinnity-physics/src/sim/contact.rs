// concinnity-physics/src/sim/contact.rs
//
// Contact manifolds and the memory that makes stacks stand still.
//
// A sequential-impulse solver converges slowly from a cold start, so each
// contact point begins the step with the impulse its own feature carried last
// step. That requires two things of this module: every point produced by the
// narrow phase carries an id naming the geometric feature it came from, and a
// point's impulse follows that id from one step to the next.
//
// The matching is a merge, not a lookup. Manifolds arrive sorted by body slot
// and are kept that way, so last step's list and this step's list walk in step
// with each other -- no hashing, and so no iteration order that could differ
// between two runs of the same scene.

use alloc::vec::Vec;

use super::broadphase::Pair;
use super::math::Vec3;

/// Contact points one manifold may carry. Four is what a face-on-face contact
/// needs to hold a box still; more points cost solver time without adding
/// constraint.
pub(crate) const MAX_MANIFOLD_POINTS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct ManifoldPoint {
    /// World-space contact point, midway between the two surfaces.
    pub(crate) point: Vec3,
    /// Gap along the normal: negative when the shapes overlap.
    pub(crate) separation: f32,
    /// The geometric feature this point came from. Stable while the same
    /// features stay in contact, which is what lets the impulse follow it.
    pub(crate) id: u32,
    pub(crate) normal_impulse: f32,
    pub(crate) tangent_impulse: [f32; 2],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Manifold {
    pub(crate) a: u32,
    pub(crate) b: u32,
    /// Unit length, pointing from `a` toward `b`.
    pub(crate) normal: Vec3,
    /// The two bodies' materials, already combined, so the solver never reads
    /// body storage while it holds its dense arrays.
    pub(crate) friction: f32,
    pub(crate) restitution: f32,
    pub(crate) points: [ManifoldPoint; MAX_MANIFOLD_POINTS],
    pub(crate) count: u8,
}

impl Manifold {
    pub(crate) fn new(a: u32, b: u32) -> Self {
        Manifold {
            a,
            b,
            normal: Vec3::Y,
            friction: 0.0,
            restitution: 0.0,
            points: [ManifoldPoint::default(); MAX_MANIFOLD_POINTS],
            count: 0,
        }
    }

    pub(crate) fn pair(&self) -> Pair {
        (self.a, self.b)
    }

    pub(crate) fn push(&mut self, point: ManifoldPoint) {
        let at = self.count as usize;
        if at < MAX_MANIFOLD_POINTS {
            self.points[at] = point;
            self.count += 1;
        }
    }

    pub(crate) fn points(&self) -> &[ManifoldPoint] {
        &self.points[..self.count as usize]
    }

    pub(crate) fn points_mut(&mut self) -> &mut [ManifoldPoint] {
        &mut self.points[..self.count as usize]
    }
}

/// Impulses last step's manifolds ended with, carried onto this step's points
/// by feature id.
///
/// Both lists are sorted by pair, so one walk over each is enough.
pub(crate) fn carry_impulses(previous: &[Manifold], current: &mut [Manifold]) {
    let mut cursor = 0usize;
    for manifold in current.iter_mut() {
        let pair = manifold.pair();
        while cursor < previous.len() && previous[cursor].pair() < pair {
            cursor += 1;
        }
        // Terrain gives one pair several manifolds, one per triangle, so the
        // match is against the whole run rather than against its first entry.
        let mut end = cursor;
        while end < previous.len() && previous[end].pair() == pair {
            end += 1;
        }
        for point in manifold.points_mut() {
            let matched = previous[cursor..end]
                .iter()
                .flat_map(Manifold::points)
                .find(|p| p.id == point.id);
            if let Some(matched) = matched {
                point.normal_impulse = matched.normal_impulse;
                point.tangent_impulse = matched.tangent_impulse;
            }
        }
    }
}

/// Every manifold last step recorded for `pair`. Used to carry a sleeping
/// pair's contacts forward untouched rather than recomputing a contact neither
/// body can have changed.
///
/// A run rather than one entry: a body resting on terrain leans on one
/// manifold per triangle, and carrying only the first would drop the rest.
pub(crate) fn find(manifolds: &[Manifold], pair: Pair) -> &[Manifold] {
    let start = manifolds.partition_point(|m| m.pair() < pair);
    let end = manifolds.partition_point(|m| m.pair() <= pair);
    &manifolds[start..end]
}

/// Two manifold buffers swapped each step, so neither is reallocated.
pub(crate) struct ContactCache {
    current: Vec<Manifold>,
    previous: Vec<Manifold>,
}

impl ContactCache {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        ContactCache {
            current: Vec::with_capacity(capacity),
            previous: Vec::with_capacity(capacity),
        }
    }

    /// Retire this step's manifolds and hand back an empty buffer to fill,
    /// alongside the manifolds that were just retired.
    pub(crate) fn begin(&mut self) -> (&mut Vec<Manifold>, &[Manifold]) {
        core::mem::swap(&mut self.current, &mut self.previous);
        self.current.clear();
        (&mut self.current, &self.previous)
    }

    #[cfg(test)]
    pub(crate) fn manifolds(&self) -> &[Manifold] {
        &self.current
    }

    pub(crate) fn manifolds_mut(&mut self) -> &mut [Manifold] {
        &mut self.current
    }

    pub(crate) fn reserved_bytes(&self) -> u64 {
        ((self.current.capacity() + self.previous.capacity()) * size_of::<Manifold>()) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifold(a: u32, b: u32, ids: &[(u32, f32)]) -> Manifold {
        let mut m = Manifold::new(a, b);
        for &(id, normal_impulse) in ids {
            m.push(ManifoldPoint {
                id,
                normal_impulse,
                tangent_impulse: [normal_impulse * 0.1, 0.0],
                ..Default::default()
            });
        }
        m
    }

    #[test]
    fn a_manifold_holds_at_most_its_point_budget() {
        let mut m = Manifold::new(0, 1);
        for id in 0..8 {
            m.push(ManifoldPoint {
                id,
                ..Default::default()
            });
        }
        assert_eq!(m.count as usize, MAX_MANIFOLD_POINTS);
        assert_eq!(m.points().len(), MAX_MANIFOLD_POINTS);
    }

    // The point of the whole module: an impulse follows its feature id.
    #[test]
    fn impulses_follow_their_feature_id_across_a_step() {
        let previous = [manifold(0, 1, &[(10, 5.0), (11, 6.0)])];
        let mut current = [manifold(0, 1, &[(11, 0.0), (10, 0.0)])];
        carry_impulses(&previous, &mut current);
        assert_eq!(current[0].points()[0].normal_impulse, 6.0);
        assert_eq!(current[0].points()[1].normal_impulse, 5.0);
        assert_eq!(current[0].points()[0].tangent_impulse, [0.6, 0.0]);
    }

    // A feature that was not in contact last step starts cold rather than
    // inheriting a neighbour's impulse.
    #[test]
    fn an_unmatched_feature_starts_from_zero() {
        let previous = [manifold(0, 1, &[(10, 5.0)])];
        let mut current = [manifold(0, 1, &[(99, 0.0)])];
        carry_impulses(&previous, &mut current);
        assert_eq!(current[0].points()[0].normal_impulse, 0.0);
    }

    // The merge has to survive pairs appearing and disappearing on either
    // side, which is the case a naive index walk gets wrong.
    #[test]
    fn the_merge_skips_pairs_present_on_only_one_side() {
        let previous = [
            manifold(0, 1, &[(1, 1.0)]),
            manifold(0, 5, &[(1, 2.0)]),
            manifold(3, 4, &[(1, 3.0)]),
        ];
        let mut current = [
            manifold(0, 2, &[(1, 0.0)]),
            manifold(0, 5, &[(1, 0.0)]),
            manifold(3, 4, &[(1, 0.0)]),
            manifold(9, 9, &[(1, 0.0)]),
        ];
        carry_impulses(&previous, &mut current);
        assert_eq!(current[0].points()[0].normal_impulse, 0.0);
        assert_eq!(current[1].points()[0].normal_impulse, 2.0);
        assert_eq!(current[2].points()[0].normal_impulse, 3.0);
        assert_eq!(current[3].points()[0].normal_impulse, 0.0);
    }

    #[test]
    fn lookup_finds_a_recorded_pair_and_misses_an_absent_one() {
        let manifolds = [manifold(0, 1, &[(1, 1.0)]), manifold(2, 9, &[(1, 2.0)])];
        assert_eq!(find(&manifolds, (2, 9)).len(), 1);
        assert_eq!(find(&manifolds, (2, 9))[0].a, 2);
        assert!(find(&manifolds, (0, 9)).is_empty());
    }

    // Terrain gives one pair a manifold per triangle, so both the lookup and
    // the impulse merge have to see all of them.
    #[test]
    fn a_pair_with_several_manifolds_carries_every_one_of_them() {
        let previous = [
            manifold(0, 1, &[(10, 5.0)]),
            manifold(0, 1, &[(20, 6.0)]),
            manifold(0, 1, &[(30, 7.0)]),
            manifold(2, 3, &[(40, 8.0)]),
        ];
        assert_eq!(find(&previous, (0, 1)).len(), 3);
        assert_eq!(find(&previous, (2, 3)).len(), 1);

        let mut current = [
            manifold(0, 1, &[(30, 0.0)]),
            manifold(0, 1, &[(10, 0.0)]),
            manifold(2, 3, &[(40, 0.0)]),
        ];
        carry_impulses(&previous, &mut current);
        assert_eq!(current[0].points()[0].normal_impulse, 7.0);
        assert_eq!(current[1].points()[0].normal_impulse, 5.0);
        assert_eq!(current[2].points()[0].normal_impulse, 8.0);
    }

    // The two buffers swap rather than reallocate, and what was current last
    // step is what the merge reads this step.
    #[test]
    fn the_cache_swaps_buffers_instead_of_reallocating() {
        let mut cache = ContactCache::with_capacity(8);
        {
            let (current, previous) = cache.begin();
            assert!(previous.is_empty());
            current.push(manifold(0, 1, &[(7, 4.0)]));
        }
        let capacities = {
            let (current, previous) = cache.begin();
            assert_eq!(previous.len(), 1);
            assert_eq!(previous[0].points()[0].normal_impulse, 4.0);
            current.push(manifold(0, 1, &[(7, 0.0)]));
            (current.capacity(), previous.len())
        };
        assert_eq!(capacities.0, 8, "the buffer kept its reservation");
        assert_eq!(cache.manifolds().len(), 1);
        assert!(cache.reserved_bytes() > 0);
    }
}
