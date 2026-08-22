// concinnity-physics/src/contacts.rs
//
// Contact-event shaping between the Rapier pipeline and the published queue:
// extraction of one hit per force event, per-frame batching that keeps the
// strongest sample per body pair, and a per-pair refractory gate so a pair in
// sustained contact reports once, not every tick.

use std::collections::HashMap;

use rapier3d::geometry::{ColliderSet, ContactPair};
use rapier3d::math::Real;

use crate::BodyHandle;
use crate::convert::from_vec;

// Ticks a body pair stays silent after reporting a contact (0.25 s at the
// fixed 60 Hz tick).
const REPORT_COOLDOWN_TICKS: u64 = 15;

/// One contact strong enough to pass the pipeline's force threshold: the two
/// bodies, the deepest contact point with its world-space normal (pointing
/// from `a` toward `b`), and the total contact impulse.
#[derive(Debug, Clone, Copy)]
pub struct ContactHit {
    /// One side of the contact.
    pub a: BodyHandle,
    /// The other side of the contact.
    pub b: BodyHandle,
    /// Deepest contact point, in world space.
    pub point: [f32; 3],
    /// Unit-length normal.
    pub normal: [f32; 3],
    /// Total contact impulse magnitude.
    pub impulse: f32,
}

// The `ContactHit` for one Rapier contact-force event, resolved while the
// collider set still contains both sides. None when either collider is gone
// or the pair has no manifold point left.
pub(crate) fn extract_hit(
    dt: Real,
    colliders: &ColliderSet,
    pair: &ContactPair,
    total_force_magnitude: Real,
) -> Option<ContactHit> {
    let (manifold, contact) = pair.find_deepest_contact()?;
    let c1 = colliders.get(pair.collider1)?;
    let c2 = colliders.get(pair.collider2)?;
    let a = BodyHandle(c1.parent()?);
    let b = BodyHandle(c2.parent()?);
    Some(ContactHit {
        a,
        b,
        point: from_vec(c1.position().transform_point(contact.local_p1)),
        normal: from_vec(manifold.data.normal),
        impulse: total_force_magnitude * dt,
    })
}

// The unordered pair key for a hit, so (a, b) and (b, a) batch together.
// Rapier handles have no ordering; the raw arena parts give a stable one.
fn pair_key(hit: &ContactHit) -> (BodyHandle, BodyHandle) {
    if hit.a.0.into_raw_parts() <= hit.b.0.into_raw_parts() {
        (hit.a, hit.b)
    } else {
        (hit.b, hit.a)
    }
}

// Accumulates one frame's ticks worth of hits, keeping the strongest sample
// per body pair. Drained after the frame's tick loop.
#[derive(Debug, Default)]
pub(crate) struct ContactBatch {
    hits: HashMap<(BodyHandle, BodyHandle), ContactHit>,
}

impl ContactBatch {
    pub(crate) fn add(&mut self, hit: ContactHit) {
        let entry = self.hits.entry(pair_key(&hit)).or_insert(hit);
        if hit.impulse > entry.impulse {
            *entry = hit;
        }
    }

    pub(crate) fn drain(&mut self) -> impl Iterator<Item = ContactHit> + '_ {
        self.hits.drain().map(|(_, hit)| hit)
    }
}

// Per-pair refractory: a pair that reported recently is suppressed until the
// cooldown elapses, so a heavy body resting hard enough to pass the force
// threshold every tick still reports only its impacts.
#[derive(Debug, Default)]
pub(crate) struct ContactGate {
    last_report: HashMap<(BodyHandle, BodyHandle), u64>,
    tick: u64,
}

impl ContactGate {
    pub(crate) fn advance_tick(&mut self) {
        self.tick += 1;
        // Long-cooled entries are dead pairs; sweep occasionally so the map
        // tracks live contact, not history.
        if self.last_report.len() > 256 {
            let tick = self.tick;
            self.last_report
                .retain(|_, last| tick - *last < REPORT_COOLDOWN_TICKS);
        }
    }

    pub(crate) fn admit(&mut self, hit: &ContactHit) -> bool {
        let key = pair_key(hit);
        if self
            .last_report
            .get(&key)
            .is_some_and(|last| self.tick - last < REPORT_COOLDOWN_TICKS)
        {
            return false;
        }
        self.last_report.insert(key, self.tick);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapier3d::dynamics::{RigidBodyBuilder, RigidBodySet};

    fn handles(n: usize) -> Vec<BodyHandle> {
        let mut set = RigidBodySet::new();
        (0..n)
            .map(|_| BodyHandle(set.insert(RigidBodyBuilder::fixed().build())))
            .collect()
    }

    fn hit(a: BodyHandle, b: BodyHandle, impulse: f32) -> ContactHit {
        ContactHit {
            a,
            b,
            point: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            impulse,
        }
    }

    #[test]
    fn batch_keeps_the_strongest_sample_per_unordered_pair() {
        let h = handles(2);
        let mut batch = ContactBatch::default();
        batch.add(hit(h[0], h[1], 1.0));
        batch.add(hit(h[1], h[0], 3.0));
        batch.add(hit(h[0], h[1], 2.0));
        let hits: Vec<ContactHit> = batch.drain().collect();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].impulse, 3.0);
        assert_eq!(batch.drain().count(), 0, "drain empties the batch");
    }

    #[test]
    fn gate_suppresses_a_pair_until_the_cooldown_elapses() {
        let h = handles(3);
        let mut gate = ContactGate::default();
        gate.advance_tick();
        assert!(gate.admit(&hit(h[0], h[1], 5.0)));
        assert!(!gate.admit(&hit(h[1], h[0], 5.0)), "reversed pair matches");
        // A different pair is independent.
        assert!(gate.admit(&hit(h[0], h[2], 5.0)));
        for _ in 0..REPORT_COOLDOWN_TICKS - 1 {
            gate.advance_tick();
            assert!(!gate.admit(&hit(h[0], h[1], 5.0)));
        }
        gate.advance_tick();
        assert!(gate.admit(&hit(h[0], h[1], 5.0)), "cooldown elapsed");
    }
}
