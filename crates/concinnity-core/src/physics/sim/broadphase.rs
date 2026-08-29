// Candidate pairs, by sweep and prune along one axis.
//
// Sweep and prune rather than a bounding-volume hierarchy because of what this
// stage is asked for right now: every body's bounds are revisited each step,
// most of them barely move, and the answer wanted is the whole overlapping set
// rather than a query against it. A sorted array in that regime is nearly free
// to maintain -- last step's order is almost this step's order, so an insertion
// pass costs about a linear scan -- and it needs no tree to rebuild, refit, or
// rebalance. A hierarchy wins where this one does not compete: point and ray
// queries against a large static set, which is a later milestone's problem and
// a separate structure when it arrives.
//
// Two properties are load bearing. Nothing here is keyed by a hash, so pair
// order depends only on the numbers that produced it; and the emitted list is
// sorted by body slot afterwards, so the solver visits pairs in an order that
// does not shift as bodies move past each other in the sweep.
//
// The sorted array is also what a query walks. A ray or a swept shape spans an
// interval on the sweep axis, and the proxies that could meet it are the
// contiguous run whose lower bounds fall inside that interval widened by the
// widest proxy -- so a query looks at a window rather than at every body.
//
// A sensor is swept alongside everything else and leaves by its own door. Its
// pairs answer a different question -- what is inside it, rather than what it
// has to push back on -- so the sweep sorts them into a second list rather
// than making every later stage re-ask which kind of pair it is holding.
//
// The scan for overlaps is the half a caller's threads can take. Each proxy's
// forward scan reads the sorted order and writes nothing, so a range of them
// is independent of every other range; the ranges collect into their own
// buffers and the lists are sorted by slot afterwards, which they were anyway.
// The insertion sort before it stays on one thread: it is a few percent of the
// stage and nearly ordered already, so there is nothing there to win.

use alloc::vec::Vec;

use crate::physics::LayerMask;
use crate::physics::fanout::Fanout;

use super::aabb::Aabb;

/// Proxies below which the scan runs on the calling thread. Fanning out costs
/// a dispatch whatever the world holds, and a handful of proxies is answered
/// before one would have finished.
const MIN_FANOUT_PROXIES: usize = 192;

/// What part a body plays in a pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    /// Immovable geometry: contact never moves it, and it crosses nothing.
    Static,
    /// Driven by position: contact never moves it, but it crosses sensors.
    Driven,
    /// Freely simulated: contact moves it, and it crosses sensors.
    Dynamic,
    /// A region that records overlap and resists nothing.
    Sensor,
}

impl Role {
    /// Whether contact can change this body's motion. A contact pair where
    /// neither side responds is never worth reporting.
    fn responds(self) -> bool {
        self == Role::Dynamic
    }

    /// Whether this body can be somewhere: it either crosses a sensor or is
    /// one. Immovable geometry is the world rather than something in it, so
    /// it crosses nothing.
    fn crosses(self) -> bool {
        self != Role::Static
    }
}

/// Which of the sweep's two lists a pair belongs in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Contact,
    Sensor,
}

/// Whether a pair is worth reporting, and as which kind.
fn reportable(a: Role, b: Role) -> Option<Kind> {
    if a == Role::Sensor || b == Role::Sensor {
        // Two sensors overlapping record a crossing each, and a wall inside a
        // region has not crossed anything.
        return (a.crosses() && b.crosses()).then_some(Kind::Sensor);
    }
    (a.responds() || b.responds()).then_some(Kind::Contact)
}

/// What the broad phase knows about one body.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Proxy {
    pub(crate) bounds: Aabb,
    pub(crate) mask: LayerMask,
    pub(crate) role: Role,
}

/// One overlapping pair, ordered so `a < b`.
pub(crate) type Pair = (u32, u32);

/// What a sweep found, split by the question each pair answers.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Pairs<'a> {
    /// Pairs the narrow phase builds contact manifolds for.
    pub(crate) contacts: &'a [Pair],
    /// Pairs with a sensor region on at least one side.
    pub(crate) sensors: &'a [Pair],
}

/// One worker's share of the overlap scan: the run of the sorted order it
/// walks, and what that run found.
#[derive(Debug, Default)]
struct Scan {
    from: usize,
    to: usize,
    contacts: Vec<Pair>,
    sensors: Vec<Pair>,
}

pub(crate) struct SweepPrune {
    /// Indexed by body slot; only slots named by `order` hold a live proxy.
    proxies: Vec<Proxy>,
    /// Live slots, kept sorted by the sweep axis's lower bound.
    order: Vec<u32>,
    axis: usize,
    pairs: Vec<Pair>,
    sensor_pairs: Vec<Pair>,
    /// The widest live proxy's extent along the sweep axis, as of the last
    /// sort. A query needs it to know how far back in the sorted order a
    /// proxy could still reach forward into its interval.
    max_extent: f32,
    /// Whether any proxy has moved, arrived, or left since the last sweep.
    /// A world where nothing moved has the same pairs it had last step, and
    /// re-deriving them would be work with a known answer.
    dirty: bool,
    /// One collecting buffer per worker the caller lends, so a fanned-out scan
    /// writes nowhere another worker is writing.
    scans: Vec<Scan>,
}

impl SweepPrune {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let empty = Proxy {
            bounds: Aabb::EMPTY,
            mask: LayerMask {
                memberships: 0,
                filter: 0,
            },
            role: Role::Static,
        };
        SweepPrune {
            proxies: alloc::vec![empty; capacity],
            order: Vec::with_capacity(capacity),
            axis: 0,
            // A body in a settled stack touches a handful of neighbours; this
            // is where that guess is spent, once, so stepping never allocates.
            pairs: Vec::with_capacity(capacity * 4),
            // Sensors are a small minority of a world's bodies, so their
            // pairs are reserved against the body count rather than against
            // the neighbour count.
            sensor_pairs: Vec::with_capacity(capacity),
            max_extent: 0.0,
            dirty: true,
            scans: Vec::new(),
        }
    }

    /// Reserve a collecting buffer per worker. Called while the world is
    /// built; a simulation nobody lends threads to keeps none of these.
    pub(crate) fn reserve_workers(&mut self, workers: usize, capacity: usize) {
        self.scans.clear();
        if workers < 2 {
            return;
        }
        // The whole sweep's reservation, shared out: a worker walks its share
        // of the proxies, so it finds about its share of the pairs.
        let share = (capacity * 4).div_ceil(workers);
        self.scans.resize_with(workers, || Scan {
            from: 0,
            to: 0,
            contacts: Vec::with_capacity(share),
            sensors: Vec::with_capacity(capacity.div_ceil(workers)),
        });
    }

    pub(crate) fn insert(&mut self, slot: u32) {
        debug_assert!(!self.order.contains(&slot), "slot {slot} added twice");
        self.order.push(slot);
        self.dirty = true;
    }

    pub(crate) fn remove(&mut self, slot: u32) {
        if let Some(at) = self.order.iter().position(|&s| s == slot) {
            // Removing in place keeps the rest of the array sorted, which a
            // swap-remove would not.
            self.order.remove(at);
            self.dirty = true;
        }
    }

    pub(crate) fn set_proxy(&mut self, slot: u32, proxy: Proxy) {
        self.proxies[slot as usize] = proxy;
        self.dirty = true;
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.order.len()
    }

    /// Pairs the last sweep reported. A caller sizing up the step it is about
    /// to run reads this: what the world was holding last step is the best
    /// guess at what it is holding now.
    pub(crate) fn pair_count(&self) -> usize {
        self.pairs.len() + self.sensor_pairs.len()
    }

    /// The axis the proxies are currently sorted along.
    pub(crate) fn axis(&self) -> usize {
        self.axis
    }

    pub(crate) fn proxy(&self, slot: u32) -> &Proxy {
        &self.proxies[slot as usize]
    }

    /// The slots a query spanning `[low, high]` on the sweep axis has to
    /// examine, in traversal order.
    ///
    /// While the order is current this is a window of it: nothing before the
    /// window reaches forward far enough, and nothing after it starts early
    /// enough. A proxy that has moved since the last sweep leaves the order
    /// unsorted, and the whole of it is the only honest answer.
    pub(crate) fn slab_window(&self, low: f32, high: f32) -> &[u32] {
        if self.dirty {
            return &self.order;
        }
        let axis = self.axis;
        let key = |&slot: &u32| self.proxies[slot as usize].bounds.min.get(axis);
        let start = self
            .order
            .partition_point(|s| key(s) < low - self.max_extent);
        let end = self.order.partition_point(|s| key(s) <= high);
        // A query behind every proxy leaves `end` before `start`.
        if end <= start {
            return &[];
        }
        &self.order[start..end]
    }

    pub(crate) fn reserved_bytes(&self) -> u64 {
        let scans: usize = self
            .scans
            .iter()
            .map(|scan| (scan.contacts.capacity() + scan.sensors.capacity()) * size_of::<Pair>())
            .sum();
        (self.proxies.capacity() * size_of::<Proxy>()
            + self.order.capacity() * size_of::<u32>()
            + (self.pairs.capacity() + self.sensor_pairs.capacity()) * size_of::<Pair>()
            + scans) as u64
    }

    /// Sweep the current proxies and return the overlapping pairs, sorted by
    /// slot.
    pub(crate) fn sweep(&mut self, fanout: &impl Fanout, workers: usize) -> Pairs<'_> {
        if self.dirty {
            self.choose_axis();
            self.sort();
            self.collect_pairs(fanout, workers);
            self.dirty = false;
        }
        Pairs {
            contacts: &self.pairs,
            sensors: &self.sensor_pairs,
        }
    }

    /// Sweep along the axis the bodies are most spread out on: the more the
    /// lower bounds differ, the sooner the inner scan can stop.
    fn choose_axis(&mut self) {
        let count = self.order.len();
        if count < 2 {
            return;
        }
        let inv = 1.0 / count as f32;
        let mut sum = [0.0f32; 3];
        let mut sum_sq = [0.0f32; 3];
        for &slot in &self.order {
            let center = self.proxies[slot as usize].bounds.center();
            for (axis, (s, sq)) in sum.iter_mut().zip(sum_sq.iter_mut()).enumerate() {
                let v = center.get(axis);
                *s += v;
                *sq += v * v;
            }
        }
        let mut best = 0;
        let mut best_variance = f32::NEG_INFINITY;
        for axis in 0..3 {
            let mean = sum[axis] * inv;
            let variance = sum_sq[axis] * inv - mean * mean;
            if variance > best_variance {
                best_variance = variance;
                best = axis;
            }
        }
        if best != self.axis {
            self.axis = best;
            // The stored order is sorted by the axis being left behind, which
            // an insertion pass would have to undo one element at a time.
            let axis = self.axis;
            let proxies = &self.proxies;
            self.order.sort_unstable_by(|&x, &y| {
                let kx = proxies[x as usize].bounds.min.get(axis);
                let ky = proxies[y as usize].bounds.min.get(axis);
                kx.total_cmp(&ky).then(x.cmp(&y))
            });
        }
    }

    /// Insertion sort: last step's order is nearly this step's, so each
    /// element usually stays where it is.
    fn sort(&mut self) {
        let axis = self.axis;
        self.max_extent = self.order.iter().fold(0.0f32, |widest, &slot| {
            let bounds = self.proxies[slot as usize].bounds;
            widest.max(bounds.max.get(axis) - bounds.min.get(axis))
        });
        let key = |proxies: &[Proxy], slot: u32| proxies[slot as usize].bounds.min.get(axis);
        for i in 1..self.order.len() {
            let slot = self.order[i];
            let k = key(&self.proxies, slot);
            let mut j = i;
            while j > 0 {
                let prev = self.order[j - 1];
                let kp = key(&self.proxies, prev);
                if kp < k || (kp == k && prev <= slot) {
                    break;
                }
                self.order[j] = prev;
                j -= 1;
            }
            self.order[j] = slot;
        }
    }

    fn collect_pairs(&mut self, fanout: &impl Fanout, workers: usize) {
        self.pairs.clear();
        self.sensor_pairs.clear();
        let count = self.order.len();
        let workers = workers.min(self.scans.len());
        if workers < 2 || count < MIN_FANOUT_PROXIES {
            let SweepPrune {
                proxies,
                order,
                axis,
                pairs,
                sensor_pairs,
                ..
            } = self;
            scan_range(proxies, order, *axis, 0, count, pairs, sensor_pairs);
        } else {
            let SweepPrune {
                proxies,
                order,
                axis,
                pairs,
                sensor_pairs,
                scans,
                ..
            } = self;
            let axis = *axis;
            let share = count.div_ceil(workers);
            let scans = &mut scans[..workers];
            for (index, scan) in scans.iter_mut().enumerate() {
                scan.from = (index * share).min(count);
                scan.to = ((index + 1) * share).min(count);
            }
            fanout.for_each(scans, |scan| {
                scan.contacts.clear();
                scan.sensors.clear();
                scan_range(
                    proxies,
                    order,
                    axis,
                    scan.from,
                    scan.to,
                    &mut scan.contacts,
                    &mut scan.sensors,
                );
            });
            for scan in scans.iter() {
                pairs.extend_from_slice(&scan.contacts);
                sensor_pairs.extend_from_slice(&scan.sensors);
            }
        }
        // Solve order must not depend on where a body happens to sit in the
        // sweep, so pairs leave here keyed by slot. It is also what makes the
        // scan's split invisible: whichever worker found a pair, the list it
        // ends up in is the same one.
        self.pairs.sort_unstable();
        self.sensor_pairs.sort_unstable();
    }
}

/// Report every overlap the proxies at `from..to` reach forward into.
///
/// A proxy only ever looks past itself in the sorted order, so two ranges of
/// it never write the same pair and neither reads what the other wrote.
fn scan_range(
    proxies: &[Proxy],
    order: &[u32],
    axis: usize,
    from: usize,
    to: usize,
    contacts: &mut Vec<Pair>,
    sensors: &mut Vec<Pair>,
) {
    for i in from..to {
        let a = order[i];
        let pa = proxies[a as usize];
        let reach = pa.bounds.max.get(axis);
        for &b in &order[i + 1..] {
            let pb = proxies[b as usize];
            if pb.bounds.min.get(axis) > reach {
                break;
            }
            let Some(kind) = reportable(pa.role, pb.role) else {
                continue;
            };
            if !pa.mask.interacts_with(pb.mask) || !pa.bounds.overlaps(pb.bounds) {
                continue;
            }
            let pair = if a < b { (a, b) } else { (b, a) };
            match kind {
                Kind::Contact => contacts.push(pair),
                Kind::Sensor => sensors.push(pair),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::physics::sim::math::{Vec3, vec3};

    fn proxy(center: Vec3, half: f32, role: Role) -> Proxy {
        Proxy {
            bounds: Aabb::from_center_half_extents(center, Vec3::splat(half)),
            mask: LayerMask::ALL,
            role,
        }
    }

    /// The two roles almost every test below needs: something contact moves,
    /// and something it does not.
    fn role(responds: bool) -> Role {
        if responds {
            Role::Dynamic
        } else {
            Role::Static
        }
    }

    fn build(centers: &[(Vec3, bool)]) -> SweepPrune {
        let mut sap = SweepPrune::with_capacity(centers.len().max(1));
        for (slot, (center, responds)) in centers.iter().enumerate() {
            sap.insert(slot as u32);
            sap.set_proxy(slot as u32, proxy(*center, 0.5, role(*responds)));
        }
        sap
    }

    /// A world of `roles`, each a half-unit box a half unit along from the
    /// last, so every neighbouring pair overlaps.
    fn touching(roles: &[Role]) -> SweepPrune {
        let mut sap = SweepPrune::with_capacity(roles.len().max(1));
        for (slot, &role) in roles.iter().enumerate() {
            sap.insert(slot as u32);
            sap.set_proxy(
                slot as u32,
                proxy(vec3(slot as f32 * 0.5, 0.0, 0.0), 0.5, role),
            );
        }
        sap
    }

    #[test]
    fn overlapping_boxes_pair_and_distant_ones_do_not() {
        let mut sap = build(&[
            (Vec3::ZERO, true),
            (vec3(0.5, 0.0, 0.0), true),
            (vec3(10.0, 0.0, 0.0), true),
        ]);
        assert_eq!(sap.sweep(&crate::physics::Inline, 1).contacts, [(0, 1)]);
        assert_eq!(sap.len(), 3);
    }

    #[test]
    fn a_pair_of_immovable_bodies_is_never_reported() {
        let mut sap = build(&[(Vec3::ZERO, false), (vec3(0.5, 0.0, 0.0), false)]);
        assert!(sap.sweep(&crate::physics::Inline, 1).contacts.is_empty());

        let mut mixed = build(&[(Vec3::ZERO, false), (vec3(0.5, 0.0, 0.0), true)]);
        assert_eq!(mixed.sweep(&crate::physics::Inline, 1).contacts, [(0, 1)]);
    }

    #[test]
    fn a_one_way_filter_still_blocks_the_pair() {
        let mut sap = build(&[(Vec3::ZERO, true), (vec3(0.5, 0.0, 0.0), true)]);
        sap.set_proxy(
            1,
            Proxy {
                mask: LayerMask {
                    memberships: u32::MAX,
                    filter: 0,
                },
                ..proxy(vec3(0.5, 0.0, 0.0), 0.5, Role::Dynamic)
            },
        );
        assert!(sap.sweep(&crate::physics::Inline, 1).contacts.is_empty());
    }

    #[test]
    fn pairs_come_out_ordered_by_slot_whatever_the_layout() {
        // Spread along z so the sweep picks that axis, with the slots laid out
        // against the sweep order.
        let mut sap = build(&[
            (vec3(0.0, 0.0, 2.0), true),
            (vec3(0.0, 0.0, 1.0), true),
            (vec3(0.0, 0.0, 0.0), true),
            (vec3(0.0, 0.0, 1.5), true),
        ]);
        let pairs = sap.sweep(&crate::physics::Inline, 1).contacts.to_vec();
        let mut sorted = pairs.clone();
        sorted.sort_unstable();
        assert_eq!(pairs, sorted);
        assert!(pairs.iter().all(|(a, b)| a < b));
        assert!(pairs.contains(&(0, 3)), "{pairs:?}");
        assert!(pairs.contains(&(1, 3)), "{pairs:?}");
        assert!(!pairs.contains(&(0, 2)), "{pairs:?}");
    }

    // Whichever axis the sweep chooses, the answer is the same set. The two
    // layouts below differ only in which axis has the spread.
    #[test]
    fn the_pair_set_does_not_depend_on_the_chosen_axis() {
        let along_x: Vec<(Vec3, bool)> = (0..8)
            .map(|i| (vec3(i as f32 * 0.75, 0.0, 0.0), true))
            .collect();
        let along_y: Vec<(Vec3, bool)> = (0..8)
            .map(|i| (vec3(0.0, i as f32 * 0.75, 0.0), true))
            .collect();
        let mut sx = build(&along_x);
        let mut sy = build(&along_y);
        assert_eq!(
            sx.sweep(&crate::physics::Inline, 1).contacts,
            sy.sweep(&crate::physics::Inline, 1).contacts
        );
        assert_eq!(sx.axis, 0);
        assert_eq!(sy.axis, 1);
    }

    // Bodies move between steps, and the sorted order has to follow without
    // losing or inventing pairs.
    #[test]
    fn re_sorting_after_motion_finds_the_same_pairs_as_a_fresh_sweep() {
        let mut sap = build(&[
            (vec3(0.0, 0.0, 0.0), true),
            (vec3(3.0, 0.0, 0.0), true),
            (vec3(6.0, 0.0, 0.0), true),
        ]);
        sap.sweep(&crate::physics::Inline, 1);
        // Slot 2 crosses to the far side of both others.
        sap.set_proxy(2, proxy(vec3(-3.2, 0.0, 0.0), 0.5, Role::Dynamic));
        sap.set_proxy(0, proxy(vec3(-2.8, 0.0, 0.0), 0.5, Role::Dynamic));
        let moved = sap.sweep(&crate::physics::Inline, 1).contacts.to_vec();

        let mut fresh = build(&[
            (vec3(-2.8, 0.0, 0.0), true),
            (vec3(3.0, 0.0, 0.0), true),
            (vec3(-3.2, 0.0, 0.0), true),
        ]);
        assert_eq!(moved, fresh.sweep(&crate::physics::Inline, 1).contacts);
        assert_eq!(moved, [(0, 2)]);
    }

    #[test]
    fn removing_a_body_drops_its_pairs_and_leaves_the_rest() {
        let mut sap = build(&[
            (Vec3::ZERO, true),
            (vec3(0.6, 0.0, 0.0), true),
            (vec3(1.2, 0.0, 0.0), true),
        ]);
        assert_eq!(
            sap.sweep(&crate::physics::Inline, 1).contacts,
            [(0, 1), (1, 2)]
        );
        sap.remove(1);
        assert_eq!(sap.len(), 2);
        assert!(
            sap.sweep(&crate::physics::Inline, 1).contacts.is_empty(),
            "the ends never reached each other"
        );
    }

    #[test]
    fn an_empty_or_single_body_sweep_reports_nothing() {
        let mut empty = SweepPrune::with_capacity(4);
        assert!(empty.sweep(&crate::physics::Inline, 1).contacts.is_empty());
        let mut one = build(&[(Vec3::ZERO, true)]);
        assert!(one.sweep(&crate::physics::Inline, 1).contacts.is_empty());
    }

    // A world where nothing moved has last step's pairs, and the sweep must
    // hand them back rather than deriving them again.
    #[test]
    fn a_sweep_with_nothing_moved_reuses_its_answer() {
        let mut sap = build(&[(Vec3::ZERO, true), (vec3(0.5, 0.0, 0.0), true)]);
        assert_eq!(sap.sweep(&crate::physics::Inline, 1).contacts, [(0, 1)]);
        assert!(!sap.dirty);
        // Reaching past the sweep to break the stored answer proves the second
        // call did not recompute it.
        sap.pairs.clear();
        assert!(sap.sweep(&crate::physics::Inline, 1).contacts.is_empty());
        // A proxy that does move brings the sweep back.
        sap.set_proxy(1, proxy(vec3(0.5, 0.0, 0.0), 0.5, Role::Dynamic));
        assert_eq!(sap.sweep(&crate::physics::Inline, 1).contacts, [(0, 1)]);
    }

    // A query must see every proxy that could meet it, and the window is only
    // worth having if it also leaves out the ones that cannot.
    #[test]
    fn a_query_window_holds_the_reachable_proxies_and_drops_the_rest() {
        let centers: Vec<(Vec3, bool)> = (0..10)
            .map(|i| (vec3(i as f32 * 2.0, 0.0, 0.0), true))
            .collect();
        let mut sap = build(&centers);
        sap.sweep(&crate::physics::Inline, 1);
        assert_eq!(sap.axis(), 0);

        // Slots 2 and 3 sit at x = 4 and x = 6, each half-extent 0.5.
        let window = sap.slab_window(3.6, 6.4);
        assert!(window.contains(&2) && window.contains(&3), "{window:?}");
        assert!(!window.contains(&0) && !window.contains(&9), "{window:?}");
        assert!(window.len() < sap.len(), "the window must prune");

        // Everything, and nothing.
        assert_eq!(sap.slab_window(-100.0, 100.0).len(), sap.len());
        assert!(sap.slab_window(-100.0, -50.0).is_empty());
        assert!(sap.slab_window(50.0, 100.0).is_empty());
    }

    // One wide proxy reaches forward past several narrow ones, so a window
    // that started at its own lower bound would miss it.
    #[test]
    fn a_wide_proxy_stays_in_the_window_of_a_query_far_ahead_of_it() {
        let mut sap = SweepPrune::with_capacity(3);
        for slot in 0..3 {
            sap.insert(slot);
        }
        sap.set_proxy(0, proxy(vec3(0.0, 0.0, 0.0), 20.0, Role::Dynamic));
        sap.set_proxy(1, proxy(vec3(10.0, 0.0, 0.0), 0.5, Role::Dynamic));
        sap.set_proxy(2, proxy(vec3(30.0, 0.0, 0.0), 0.5, Role::Dynamic));
        sap.sweep(&crate::physics::Inline, 1);
        let window = sap.slab_window(9.0, 11.0);
        assert!(
            window.contains(&0),
            "the wide proxy reaches here: {window:?}"
        );
        assert!(window.contains(&1), "{window:?}");
        assert!(!window.contains(&2), "{window:?}");
    }

    // A proxy that moved since the last sweep leaves the order unsorted, and
    // a window over an unsorted array would silently drop bodies.
    #[test]
    fn a_stale_order_widens_the_window_to_everything() {
        let mut sap = build(&[
            (vec3(0.0, 0.0, 0.0), true),
            (vec3(4.0, 0.0, 0.0), true),
            (vec3(8.0, 0.0, 0.0), true),
        ]);
        sap.sweep(&crate::physics::Inline, 1);
        assert!(sap.slab_window(-1.0, 1.0).len() < 3);
        sap.set_proxy(2, proxy(vec3(-9.0, 0.0, 0.0), 0.5, Role::Dynamic));
        assert_eq!(sap.slab_window(-1.0, 1.0).len(), 3);
    }

    // A sensor's whole point: it pairs with what can be inside it, and with
    // nothing else. Static geometry sharing its space is not a crossing.
    #[test]
    fn a_sensor_pairs_with_what_can_cross_it_and_not_with_geometry() {
        let mut sap = touching(&[Role::Sensor, Role::Static]);
        let swept = sap.sweep(&crate::physics::Inline, 1);
        assert!(swept.contacts.is_empty());
        assert!(swept.sensors.is_empty(), "a wall crosses nothing");

        for other in [Role::Dynamic, Role::Driven, Role::Sensor] {
            let mut sap = touching(&[Role::Sensor, other]);
            let swept = sap.sweep(&crate::physics::Inline, 1);
            assert_eq!(swept.sensors, [(0, 1)], "{other:?} must be detected");
            assert!(
                swept.contacts.is_empty(),
                "{other:?} must not collide with a region"
            );
        }
    }

    // A position-driven body still costs the narrow phase nothing where the
    // sensor is not involved: two of them, or one against a wall, are as
    // silent as they always were.
    #[test]
    fn the_sensor_rule_leaves_the_contact_rule_alone() {
        for pair in [
            [Role::Driven, Role::Static],
            [Role::Driven, Role::Driven],
            [Role::Static, Role::Static],
        ] {
            let mut sap = touching(&pair);
            let swept = sap.sweep(&crate::physics::Inline, 1);
            assert!(swept.contacts.is_empty(), "{pair:?}");
            assert!(swept.sensors.is_empty(), "{pair:?}");
        }
        let mut sap = touching(&[Role::Driven, Role::Dynamic]);
        assert_eq!(sap.sweep(&crate::physics::Inline, 1).contacts, [(0, 1)]);
    }

    // The two lists come out of one walk, and each has to be sorted by slot
    // on its own for the stages reading them to be order-independent.
    #[test]
    fn both_lists_leave_the_sweep_sorted_by_slot() {
        let mut sap = touching(&[
            Role::Dynamic,
            Role::Sensor,
            Role::Dynamic,
            Role::Sensor,
            Role::Dynamic,
        ]);
        let swept = sap.sweep(&crate::physics::Inline, 1);
        assert!(swept.contacts.windows(2).all(|w| w[0] < w[1]));
        assert!(swept.sensors.windows(2).all(|w| w[0] < w[1]));
        assert!(swept.sensors.contains(&(1, 2)), "{:?}", swept.sensors);
        assert!(swept.sensors.contains(&(2, 3)), "{:?}", swept.sensors);
        assert!(!swept.contacts.contains(&(1, 2)), "{:?}", swept.contacts);
    }

    // A one-way layer filter is checked once, for both kinds of pair.
    #[test]
    fn a_layer_filter_hides_a_sensor_pair_too() {
        let mut sap = touching(&[Role::Sensor, Role::Dynamic]);
        assert_eq!(sap.sweep(&crate::physics::Inline, 1).sensors, [(0, 1)]);
        sap.set_proxy(
            1,
            Proxy {
                mask: LayerMask {
                    memberships: 0b10,
                    filter: 0b10,
                },
                ..proxy(vec3(0.5, 0.0, 0.0), 0.5, Role::Dynamic)
            },
        );
        sap.set_proxy(
            0,
            Proxy {
                mask: LayerMask {
                    memberships: 0b01,
                    filter: 0b01,
                },
                ..proxy(Vec3::ZERO, 0.5, Role::Sensor)
            },
        );
        assert!(sap.sweep(&crate::physics::Inline, 1).sensors.is_empty());
    }

    // Stepping a settled world must not allocate: the pair buffer is reused.
    #[test]
    fn sweeping_reuses_its_pair_buffer() {
        let centers: Vec<(Vec3, bool)> = (0..16)
            .map(|i| (vec3(i as f32 * 0.4, 0.0, 0.0), true))
            .collect();
        let mut sap = build(&centers);
        sap.sweep(&crate::physics::Inline, 1);
        let capacity = sap.pairs.capacity();
        for slot in 0..8 {
            sap.set_proxy(
                slot,
                proxy(vec3(slot as f32 * 0.4, 0.0, 0.0), 0.5, Role::Dynamic),
            );
            sap.sweep(&crate::physics::Inline, 1);
        }
        assert_eq!(sap.pairs.capacity(), capacity);
    }
}
