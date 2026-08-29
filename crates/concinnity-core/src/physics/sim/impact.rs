// Which of a step's contacts were hard enough to be worth hearing about.
//
// Every resting body is in contact with something, so a report of "these two
// touched" is noise. What a caller wants is the impacts: the pairs whose
// contact carried real load. The gate is a force, from the total impulse the
// solver actually converged on divided by the step it was spread over, which
// is what separates a crate being dropped from the same crate sitting there.
//
// It reads what the solver converged on rather than estimating during it. The
// impulse a contact carries is the answer several substeps and a relax pass
// arrived at, and a figure taken before that is a guess at it. Asking the
// solver rather than the manifold list is also what keeps a settled pair
// quiet: a sleeping stack's manifolds are carried forward with the impulses
// that hold it up, and only the pairs the step actually solved have a load.
//
// One pair, one hit. Terrain gives a pair a manifold per triangle and a face
// contact gives it four points; a caller wants to hear that a body landed, not
// how many triangles it landed across, so the run is summed for the force and
// the deepest point of it stands for where.
//
// The queue holds what has happened since it was last drained rather than what
// the last step reported, so a caller running several ticks between drains
// hears about all of them.

use alloc::vec::Vec;

use crate::memory::Pool;

use crate::physics::ContactHit;

use super::body::Body;
use super::broadphase::Pair;
use super::contact::Manifold;
use super::math::Vec3;
use super::solver::ContactLoad;
use super::world::handle_at;

/// Total contact force a pair passes by default before it is reported, in
/// newtons. High enough that a body's own weight at rest says nothing.
const DEFAULT_FORCE_THRESHOLD: f32 = 60.0;

/// The contact hits a step recorded, and the force a pair has to carry to
/// record one.
pub(crate) struct Impacts {
    hits: Vec<ContactHit>,
    force_threshold: f32,
    overflows: u32,
}

impl Impacts {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Impacts {
            hits: Vec::with_capacity(capacity),
            force_threshold: DEFAULT_FORCE_THRESHOLD,
            overflows: 0,
        }
    }

    /// Set the smallest contact impulse worth reporting, as the force it
    /// stands for at a step of `tick_dt`.
    pub(crate) fn set_min_impulse(&mut self, min_impulse: f32, tick_dt: f32) {
        self.force_threshold = min_impulse.max(0.0) / tick_dt.max(1.0e-6);
    }

    /// Record the pairs whose contact this step carried more than the
    /// threshold, appending to whatever has not been drained yet.
    ///
    /// `loads` is what the solver delivered, one entry per manifold it
    /// solved, in manifold order; `manifolds` is the list those index into,
    /// sorted by slot pair. So the hits come out in slot order, and a pair
    /// spread over several manifolds is summed rather than reported several
    /// times.
    pub(crate) fn collect(
        &mut self,
        bodies: &Pool<Body>,
        manifolds: &[Manifold],
        loads: &[ContactLoad],
        dt: f32,
    ) {
        let inv_dt = if dt > 0.0 { 1.0 / dt } else { 0.0 };
        let pair_of = |load: &ContactLoad| manifolds[load.manifold as usize].pair();
        let mut start = 0usize;
        while start < loads.len() {
            let pair = pair_of(&loads[start]);
            let mut end = start + 1;
            while end < loads.len() && pair_of(&loads[end]) == pair {
                end += 1;
            }
            let run = &loads[start..end];
            start = end;
            if let Some(hit) = self.measure(bodies, manifolds, pair, run, inv_dt) {
                if self.hits.len() == self.hits.capacity() {
                    self.overflows = self.overflows.saturating_add(1);
                    continue;
                }
                self.hits.push(hit);
            }
        }
    }

    /// The hit one pair's solved manifolds add up to, or `None` when the pair
    /// is not a source or did not carry enough load.
    fn measure(
        &self,
        bodies: &Pool<Body>,
        manifolds: &[Manifold],
        pair: Pair,
        run: &[ContactLoad],
        inv_dt: f32,
    ) -> Option<ContactHit> {
        let (body_a, body_b) = (
            bodies.get_at(pair.0 as usize)?,
            bodies.get_at(pair.1 as usize)?,
        );
        // Only a freely simulated body is a source: two walls leaning on each
        // other is the world's shape rather than something that happened.
        if !body_a.is_dynamic() && !body_b.is_dynamic() {
            return None;
        }
        let impulse: f32 = run.iter().map(|load| load.impulse).sum();
        if impulse * inv_dt <= self.force_threshold {
            return None;
        }
        let (point, normal) = deepest(manifolds, run)?;
        Some(ContactHit {
            a: handle_at(bodies, pair.0)?,
            b: handle_at(bodies, pair.1)?,
            point: point.to_array(),
            normal: normal.to_array(),
            impulse,
        })
    }

    /// Move the recorded hits into `out`, oldest first. Both the queue and
    /// `out` keep their capacity.
    pub(crate) fn drain_into(&mut self, out: &mut Vec<ContactHit>) {
        out.clear();
        out.append(&mut self.hits);
    }

    #[cfg(test)]
    /// Hits the reservation had no room for.
    pub(crate) fn overflows(&self) -> u32 {
        self.overflows
    }

    #[cfg(test)]
    pub(crate) fn clear_overflows(&mut self) {
        self.overflows = 0;
    }

    pub(crate) fn reserved_bytes(&self) -> u64 {
        (self.hits.capacity() * size_of::<ContactHit>()) as u64
    }
}

/// The deepest point of one pair's solved manifolds, and the normal of the
/// manifold it came from, which points from the lower slot's body toward the
/// higher one's.
fn deepest(manifolds: &[Manifold], run: &[ContactLoad]) -> Option<(Vec3, Vec3)> {
    let mut found: Option<(f32, Vec3, Vec3)> = None;
    for load in run {
        let manifold = &manifolds[load.manifold as usize];
        for point in manifold.points() {
            if found.is_none_or(|(separation, ..)| point.separation < separation) {
                found = Some((point.separation, point.point, manifold.normal));
            }
        }
    }
    found.map(|(_, point, normal)| (point, normal))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::sim::body::Body;
    use crate::physics::sim::contact::ManifoldPoint;
    use crate::physics::sim::math::{Quat, vec3};
    use crate::physics::{BodyHandle, ColliderShape, DynamicParams, LayerMask};

    const TICK: f32 = 1.0 / 60.0;
    const CUBE: ColliderShape = ColliderShape::Cuboid {
        half_extents: [0.5, 0.5, 0.5],
    };

    fn params() -> DynamicParams {
        DynamicParams {
            mass: 1.0,
            friction: 0.5,
            restitution: 0.0,
            gravity_scale: 1.0,
            linear_damping: 0.0,
        }
    }

    /// A floor in slot 0 and a freely simulated cube in slot 1, which is the
    /// pair every test below reports about.
    fn floor_and_cube() -> Pool<Body> {
        let mut bodies = Pool::with_capacity(8);
        bodies.insert(Body::fixed(
            CUBE,
            vec3(0.0, -0.5, 0.0),
            Quat::IDENTITY,
            0.8,
            LayerMask::ALL,
        ));
        bodies.insert(Body::dynamic(
            CUBE,
            vec3(0.0, 0.5, 0.0),
            Quat::IDENTITY,
            params(),
            LayerMask::ALL,
        ));
        bodies
    }

    /// A two-point contact patch between `a` and `b`, the first point the
    /// deeper of the two.
    fn patch(a: u32, b: u32) -> Manifold {
        let mut manifold = Manifold::new(a, b);
        manifold.normal = -Vec3::Y;
        for (id, (x, separation)) in [(-0.5, -0.01), (0.5, -0.002)].into_iter().enumerate() {
            manifold.push(ManifoldPoint {
                point: vec3(x, 0.0, 0.0),
                separation,
                id: id as u32,
                normal_impulse: 0.0,
                tangent_impulse: [0.0; 2],
            });
        }
        manifold
    }

    fn load(manifold: u32, impulse: f32) -> ContactLoad {
        ContactLoad { manifold, impulse }
    }

    fn drained(impacts: &mut Impacts) -> Vec<ContactHit> {
        let mut out = Vec::new();
        impacts.drain_into(&mut out);
        out
    }

    #[test]
    fn a_pair_carrying_more_than_the_threshold_records_a_hit() {
        let bodies = floor_and_cube();
        let mut impacts = Impacts::with_capacity(4);
        impacts.set_min_impulse(1.0, TICK);
        impacts.collect(&bodies, &[patch(0, 1)], &[load(0, 10.0)], TICK);
        let hits = drained(&mut impacts);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].a, BodyHandle::from_parts(0, 0));
        assert_eq!(hits[0].b, BodyHandle::from_parts(1, 0));
        assert_eq!(hits[0].impulse, 10.0);
        assert_eq!(hits[0].normal, [0.0, -1.0, 0.0], "from a toward b");
        assert_eq!(hits[0].point, [-0.5, 0.0, 0.0], "the deeper of the two");
        assert!(drained(&mut impacts).is_empty(), "the queue was drained");
    }

    // The gate is the whole point: a pair leaning on another below the
    // threshold is not an impact.
    #[test]
    fn a_pair_below_the_threshold_stays_silent() {
        let bodies = floor_and_cube();
        let mut impacts = Impacts::with_capacity(4);
        // 10 impulse over a 60 Hz tick is 600 N.
        impacts.set_min_impulse(20.0, TICK);
        impacts.collect(&bodies, &[patch(0, 1)], &[load(0, 10.0)], TICK);
        assert!(drained(&mut impacts).is_empty());

        impacts.set_min_impulse(5.0, TICK);
        impacts.collect(&bodies, &[patch(0, 1)], &[load(0, 10.0)], TICK);
        assert_eq!(drained(&mut impacts).len(), 1);
    }

    // A threshold and a load that match exactly is not "more than", or a
    // world with a zero threshold would report every contact it has.
    #[test]
    fn the_threshold_is_passed_rather_than_reached() {
        let bodies = floor_and_cube();
        let mut impacts = Impacts::with_capacity(4);
        impacts.set_min_impulse(10.0, TICK);
        impacts.collect(&bodies, &[patch(0, 1)], &[load(0, 10.0)], TICK);
        assert!(drained(&mut impacts).is_empty());
    }

    #[test]
    fn a_pair_of_immovable_bodies_records_nothing() {
        let mut bodies = Pool::with_capacity(4);
        for y in [-0.5, 0.5] {
            bodies.insert(Body::fixed(
                CUBE,
                vec3(0.0, y, 0.0),
                Quat::IDENTITY,
                0.8,
                LayerMask::ALL,
            ));
        }
        let mut impacts = Impacts::with_capacity(4);
        impacts.set_min_impulse(0.1, TICK);
        impacts.collect(&bodies, &[patch(0, 1)], &[load(0, 50.0)], TICK);
        assert!(drained(&mut impacts).is_empty());
    }

    // A settled pair keeps its manifold so the warm start still has it, but
    // the step delivered nothing, so there is no load naming it.
    #[test]
    fn a_manifold_the_step_never_solved_records_nothing() {
        let bodies = floor_and_cube();
        let mut impacts = Impacts::with_capacity(4);
        impacts.set_min_impulse(0.1, TICK);
        impacts.collect(&bodies, &[patch(0, 1)], &[], TICK);
        assert!(drained(&mut impacts).is_empty());
    }

    // Terrain hands one pair a manifold per triangle, and a caller wants to
    // hear that the body landed rather than how many triangles it landed on.
    #[test]
    fn several_manifolds_for_one_pair_add_up_to_one_hit() {
        let bodies = floor_and_cube();
        let mut impacts = Impacts::with_capacity(4);
        impacts.set_min_impulse(1.0, TICK);
        let mut deeper = patch(0, 1);
        deeper.normal = Vec3::Y;
        deeper.points[0].separation = -0.5;
        deeper.points[0].point = vec3(9.0, 9.0, 9.0);
        impacts.collect(
            &bodies,
            &[patch(0, 1), deeper],
            &[load(0, 10.0), load(1, 6.0)],
            TICK,
        );
        let hits = drained(&mut impacts);
        assert_eq!(hits.len(), 1, "one pair, one hit");
        assert_eq!(hits[0].impulse, 16.0);
        assert_eq!(hits[0].point, [9.0, 9.0, 9.0], "the deepest of the run");
        assert_eq!(hits[0].normal, [0.0, 1.0, 0.0], "that manifold's normal");
    }

    // Hits leave in slot order, whatever order the pairs are asked about, so
    // two runs of the same scene report the same sequence.
    #[test]
    fn hits_come_out_in_slot_order() {
        let mut bodies = floor_and_cube();
        for x in [4.0, 8.0] {
            bodies.insert(Body::dynamic(
                CUBE,
                vec3(x, 0.5, 0.0),
                Quat::IDENTITY,
                params(),
                LayerMask::ALL,
            ));
        }
        let mut impacts = Impacts::with_capacity(8);
        impacts.set_min_impulse(1.0, TICK);
        impacts.collect(
            &bodies,
            &[patch(0, 1), patch(0, 2), patch(1, 3)],
            &[load(0, 10.0), load(1, 10.0), load(2, 10.0)],
            TICK,
        );
        let hits = drained(&mut impacts);
        let pairs: Vec<_> = hits.iter().map(|h| (h.a.index(), h.b.index())).collect();
        assert_eq!(pairs, [(0, 1), (0, 2), (1, 3)]);
    }

    // A queue with no room declines rather than growing inside a step, and
    // says how often it had to.
    #[test]
    fn a_full_queue_declines_and_counts() {
        let mut bodies = floor_and_cube();
        bodies.insert(Body::dynamic(
            CUBE,
            vec3(4.0, 0.5, 0.0),
            Quat::IDENTITY,
            params(),
            LayerMask::ALL,
        ));
        let mut impacts = Impacts::with_capacity(1);
        let capacity = impacts.hits.capacity();
        impacts.set_min_impulse(1.0, TICK);
        impacts.collect(
            &bodies,
            &[patch(0, 1), patch(0, 2)],
            &[load(0, 10.0), load(1, 10.0)],
            TICK,
        );
        assert_eq!(impacts.hits.len(), capacity);
        assert_eq!(impacts.overflows(), 1);
        assert_eq!(impacts.hits.capacity(), capacity, "it never grew");
        impacts.clear_overflows();
        assert_eq!(impacts.overflows(), 0);
    }

    #[test]
    fn a_manifold_with_no_points_reports_nothing() {
        let bodies = floor_and_cube();
        let mut impacts = Impacts::with_capacity(4);
        impacts.set_min_impulse(0.0, TICK);
        impacts.collect(&bodies, &[Manifold::new(0, 1)], &[load(0, 10.0)], TICK);
        assert!(drained(&mut impacts).is_empty());
        assert!(impacts.reserved_bytes() > 0);
    }
}
