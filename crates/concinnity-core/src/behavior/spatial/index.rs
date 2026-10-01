// One query's neighbor index for one tick: built on demand, shared read-only by
// every worker that asks, and forgotten when the tick ends.
//
// Positions cannot move while bodies run (effects land after every body), so a
// tree built from the first asks' view of the world is the view every later
// ask would have scanned.

use core::sync::atomic::{AtomicU32, Ordering};

use spin::Once;
use spin::mutex::SpinMutex;

use super::tree::Tree;
use super::{count_within_linear, distance_sq, nearest_linear};
use crate::behavior::position;
use crate::ecs::{ComponentStorage, Entity};

// Below this many candidates a scan is as quick as any tree.
const MIN_CANDIDATES: usize = 32;

// The ask that builds the tree. Earlier asks scan, so a query asked once or
// twice a tick never pays for a build it could not earn back.
const BUILD_ON_ASK: u32 = 3;

/// Answers `nearest` and `count_within` over one query's candidates, exactly as
/// the linear scans in this module would.
///
/// The candidates must be the same slice on every ask between two
/// [`reset`](NeighborIndex::reset)s, sorted by entity bits, and the world must
/// not change in between.
#[derive(Debug, Default)]
pub(crate) struct NeighborIndex {
    asks: AtomicU32,
    built: Once<Tree>,
    // The previous tick's tree, kept for its storage. Only the one ask that
    // builds takes it, so the lock is never contended.
    spare: SpinMutex<Tree>,
}

impl NeighborIndex {
    /// Forget the tree the last tick built, keeping its storage for the next.
    pub(crate) fn reset(&mut self) {
        *self.asks.get_mut() = 0;
        if let Some(tree) = core::mem::take(&mut self.built).try_into_inner() {
            *self.spare.get_mut() = tree;
        }
    }

    /// The candidate nearest `point`, skipping `exclude` and anything with no
    /// position. Ties go to the earlier candidate.
    pub(crate) fn nearest(
        &self,
        components: &ComponentStorage,
        candidates: &[Entity],
        point: [f32; 3],
        exclude: Option<Entity>,
    ) -> Option<Entity> {
        match self.tree(components, candidates) {
            Some(tree) if is_finite(point) => {
                let skip = exclude.and_then(|e| rank_of(candidates, e));
                let rank = tree.nearest(point, skip)?;
                candidates.get(rank as usize).copied()
            }
            _ => nearest_linear(components, candidates, point, exclude),
        }
    }

    /// How many candidates lie within `radius` of `point`, skipping `exclude`.
    /// A negative or non-finite radius counts nothing.
    pub(crate) fn count_within(
        &self,
        components: &ComponentStorage,
        candidates: &[Entity],
        point: [f32; 3],
        radius: f32,
        exclude: Option<Entity>,
    ) -> i32 {
        if !(radius.is_finite() && radius >= 0.0) {
            return 0;
        }
        match self.tree(components, candidates) {
            Some(tree) if is_finite(point) => {
                let limit = radius * radius;
                let counted = tree.count_within(point, limit);
                // The excluded entity is in the tree like any other candidate,
                // so it is taken back out on the same test that counted it.
                let excluded = exclude
                    .filter(|e| rank_of(candidates, *e).is_some())
                    .and_then(|e| position::of(components, e))
                    .is_some_and(|p| distance_sq(p, point) <= limit);
                counted.saturating_sub(usize::from(excluded)) as i32
            }
            _ => count_within_linear(components, candidates, point, radius, exclude),
        }
    }

    // The tree to answer from, or `None` to scan: too few candidates, too few
    // asks yet, another worker still building, or a set the tree cannot index.
    fn tree(&self, components: &ComponentStorage, candidates: &[Entity]) -> Option<&Tree> {
        if let Some(tree) = self.built.get() {
            return tree.is_indexed().then_some(tree);
        }
        if candidates.len() < MIN_CANDIDATES || u32::try_from(candidates.len()).is_err() {
            return None;
        }
        if self.asks.fetch_add(1, Ordering::Relaxed) != BUILD_ON_ASK - 1 {
            return None;
        }
        let mut tree = core::mem::take(&mut *self.spare.lock());
        // Exclusion finds its rank by binary search, which needs the order a
        // query's gather produces.
        if candidates.is_sorted_by_key(|e| e.to_bits()) {
            tree.rebuild(
                candidates
                    .iter()
                    .zip(0u32..)
                    .filter_map(|(e, rank)| position::of(components, *e).map(|p| (rank, p))),
            );
        } else {
            tree.clear();
        }
        let tree = self.built.call_once(|| tree);
        tree.is_indexed().then_some(tree)
    }
}

fn is_finite(point: [f32; 3]) -> bool {
    point.iter().all(|c| c.is_finite())
}

fn rank_of(candidates: &[Entity], entity: Entity) -> Option<u32> {
    let rank = candidates
        .binary_search_by_key(&entity.to_bits(), |e| e.to_bits())
        .ok()?;
    u32::try_from(rank).ok()
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::components::Transform;

    // A fixed-seed generator, so every run tests the same worlds.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as u32
        }

        fn lattice(&mut self) -> f32 {
            (self.next() % 12) as f32 * 0.5
        }
    }

    // `count` entities on a coarse lattice, one in five with no position, in
    // the sorted order a gather hands over.
    fn world(rng: &mut Lcg, count: usize) -> (ComponentStorage, Vec<Entity>) {
        let mut components = ComponentStorage::default();
        let mut entities = Vec::new();
        for _ in 0..count {
            let entity = components.spawn();
            if !rng.next().is_multiple_of(5) {
                let position = [rng.lattice(), rng.lattice(), rng.lattice()];
                components.insert_typed(
                    entity,
                    Transform {
                        position,
                        ..Default::default()
                    },
                );
            }
            entities.push(entity);
        }
        entities.sort_unstable_by_key(|e| e.to_bits());
        (components, entities)
    }

    fn at(components: &mut ComponentStorage, position: [f32; 3]) -> Entity {
        let entity = components.spawn();
        components.insert_typed(
            entity,
            Transform {
                position,
                ..Default::default()
            },
        );
        entity
    }

    // Enough asks that the tree is built, so every one after answers from it.
    fn warmed(components: &ComponentStorage, candidates: &[Entity]) -> NeighborIndex {
        let index = NeighborIndex::default();
        for _ in 0..BUILD_ON_ASK {
            index.nearest(components, candidates, [0.0; 3], None);
        }
        index
    }

    #[test]
    fn answers_match_the_scans_with_exclusion_and_bare_entities() {
        let mut rng = Lcg(5);
        for round in 0..40 {
            let (components, candidates) = world(&mut rng, MIN_CANDIDATES + round * 9);
            let index = warmed(&components, &candidates);
            assert!(index.built.get().is_some_and(Tree::is_indexed));
            for _ in 0..60 {
                let point = [rng.lattice(), rng.lattice(), rng.lattice()];
                let exclude = candidates
                    .get(rng.next() as usize % candidates.len())
                    .copied();
                let radius = (rng.next() % 8) as f32 * 0.5;
                assert_eq!(
                    index.nearest(&components, &candidates, point, exclude),
                    nearest_linear(&components, &candidates, point, exclude),
                );
                assert_eq!(
                    index.count_within(&components, &candidates, point, radius, exclude),
                    count_within_linear(&components, &candidates, point, radius, exclude),
                );
            }
        }
    }

    // The asks before the building one scan, and every answer agrees.
    #[test]
    fn the_tree_is_built_on_the_building_ask() {
        let mut rng = Lcg(9);
        let (components, candidates) = world(&mut rng, 64);
        let index = NeighborIndex::default();
        for ask in 1..=BUILD_ON_ASK {
            let point = [rng.lattice(), rng.lattice(), rng.lattice()];
            assert_eq!(
                index.nearest(&components, &candidates, point, None),
                nearest_linear(&components, &candidates, point, None),
            );
            assert_eq!(index.built.get().is_some(), ask == BUILD_ON_ASK);
        }
    }

    #[test]
    fn a_small_set_is_never_built() {
        let mut rng = Lcg(2);
        let (components, candidates) = world(&mut rng, MIN_CANDIDATES - 1);
        let index = warmed(&components, &candidates);
        index.nearest(&components, &candidates, [0.0; 3], None);
        assert!(index.built.get().is_none());
    }

    // A reset forgets the tree, so the next tick builds from its own world.
    #[test]
    fn a_reset_rebuilds_from_the_next_world() {
        let mut components = ComponentStorage::default();
        let candidates: Vec<Entity> = (0..MIN_CANDIDATES)
            .map(|i| at(&mut components, [i as f32 * 10.0, 0.0, 0.0]))
            .collect();
        let mut index = warmed(&components, &candidates);
        assert_eq!(
            index.nearest(&components, &candidates, [0.0; 3], None),
            candidates.first().copied(),
        );

        let last = *candidates.last().expect("the set is not empty");
        components
            .get_mut::<Transform>(last)
            .expect("placed with a transform")
            .position = [-1.0, 0.0, 0.0];
        index.reset();
        assert!(index.built.get().is_none());
        for _ in 0..BUILD_ON_ASK {
            index.nearest(&components, &candidates, [0.0; 3], None);
        }
        assert_eq!(
            index.nearest(&components, &candidates, [-2.0, 0.0, 0.0], None),
            Some(last),
        );
    }

    // A NaN candidate wins `nearest` when it comes first in the scan, which no
    // bound reproduces, so its set is answered by the scan.
    #[test]
    fn a_non_finite_candidate_is_answered_by_the_scan() {
        let mut components = ComponentStorage::default();
        let mut candidates: Vec<Entity> = (0..MIN_CANDIDATES)
            .map(|i| at(&mut components, [i as f32, 0.0, 0.0]))
            .collect();
        candidates.push(at(&mut components, [f32::NAN, 0.0, 0.0]));
        let index = warmed(&components, &candidates);
        assert!(index.built.get().is_some_and(|tree| !tree.is_indexed()));
        for point in [[0.0; 3], [5.0, 0.0, 0.0]] {
            assert_eq!(
                index.nearest(&components, &candidates, point, None),
                nearest_linear(&components, &candidates, point, None),
            );
            assert_eq!(
                index.count_within(&components, &candidates, point, 2.0, None),
                count_within_linear(&components, &candidates, point, 2.0, None),
            );
        }
    }

    // A non-finite point is answered by the scan, whatever the tree holds.
    #[test]
    fn a_non_finite_point_is_answered_by_the_scan() {
        let mut rng = Lcg(4);
        let (components, candidates) = world(&mut rng, 64);
        let index = warmed(&components, &candidates);
        for point in [[f32::NAN, 0.0, 0.0], [0.0, f32::INFINITY, 0.0]] {
            assert_eq!(
                index.nearest(&components, &candidates, point, None),
                nearest_linear(&components, &candidates, point, None),
            );
            assert_eq!(
                index.count_within(&components, &candidates, point, 3.0, None),
                count_within_linear(&components, &candidates, point, 3.0, None),
            );
        }
    }

    #[test]
    fn a_degenerate_radius_counts_nothing() {
        let mut rng = Lcg(6);
        let (components, candidates) = world(&mut rng, 64);
        let index = warmed(&components, &candidates);
        for radius in [-1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(
                index.count_within(&components, &candidates, [0.0; 3], radius, None),
                0
            );
        }
    }

    // An excluded entity is taken out of the count only when it was counted:
    // inside the radius, outside it, and with no position at all.
    #[test]
    fn exclusion_takes_out_only_what_was_counted() {
        let mut components = ComponentStorage::default();
        let mut candidates: Vec<Entity> = (0..MIN_CANDIDATES)
            .map(|i| at(&mut components, [i as f32, 0.0, 0.0]))
            .collect();
        let bare = components.spawn();
        candidates.push(bare);
        let index = warmed(&components, &candidates);
        let inside = candidates[1];
        let outside = candidates[20];
        for exclude in [None, Some(inside), Some(outside), Some(bare)] {
            assert_eq!(
                index.count_within(&components, &candidates, [0.0; 3], 3.0, exclude),
                count_within_linear(&components, &candidates, [0.0; 3], 3.0, exclude),
            );
        }
        assert_eq!(
            index.count_within(&components, &candidates, [0.0; 3], 3.0, Some(inside)),
            3
        );
    }

    // Candidates out of gather order cannot be ranked by search, so the set is
    // answered by the scan rather than by a tree that would misplace ties.
    #[test]
    fn an_unsorted_set_is_answered_by_the_scan() {
        let mut rng = Lcg(8);
        let (components, mut candidates) = world(&mut rng, 64);
        candidates.reverse();
        let index = warmed(&components, &candidates);
        assert!(index.built.get().is_some_and(|tree| !tree.is_indexed()));
        let point = [1.0, 1.0, 1.0];
        let exclude = candidates.first().copied();
        assert_eq!(
            index.nearest(&components, &candidates, point, exclude),
            nearest_linear(&components, &candidates, point, exclude),
        );
    }
}
