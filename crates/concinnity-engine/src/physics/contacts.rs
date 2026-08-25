// src/physics/contacts.rs
//
// Contact-event shaping between the simulation and the published queue:
// per-frame batching that keeps the strongest sample per body pair, and a
// per-pair refractory gate so a pair in sustained contact reports once, not
// every tick. Which contacts are worth reporting at all is the simulation's
// own gate.

use std::collections::HashMap;

use concinnity_physics::{BodyHandle, ContactHit};

// Ticks a body pair stays silent after reporting a contact (0.25 s at the
// fixed 60 Hz tick).
const REPORT_COOLDOWN_TICKS: u64 = 15;

// The unordered pair key for a hit, so (a, b) and (b, a) batch together.
fn pair_key(hit: &ContactHit) -> (BodyHandle, BodyHandle) {
    if hit.a <= hit.b {
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
    // Reserved for the body pairs the world's budget can produce, so a frame's
    // batching never allocates.
    pub(crate) fn with_capacity(pairs: usize) -> Self {
        Self {
            hits: HashMap::with_capacity(pairs),
        }
    }

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
    // Reserved for the body pairs the world's budget can produce; the sweep in
    // `advance_tick` keeps it from growing past live contact.
    pub(crate) fn with_capacity(pairs: usize) -> Self {
        Self {
            last_report: HashMap::with_capacity(pairs),
            tick: 0,
        }
    }

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

    fn handles(n: usize) -> Vec<BodyHandle> {
        (0..n)
            .map(|i| BodyHandle::from_parts(i as u32, 0))
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
