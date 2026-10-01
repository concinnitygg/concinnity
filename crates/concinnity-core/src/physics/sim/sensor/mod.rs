// Regions that record what is inside them and resist nothing.
//
// A sensor rides the same sweep every other body does and leaves it by its own
// door, so this stage is handed the pairs it cares about already filtered and
// already sorted by slot. What is left is to measure each pair exactly
// (`overlap`), remember which pairs were overlapping so a boundary can be told
// from a state (`track`), and turn each boundary into the crossings a caller
// reads.
//
// A boundary test at step boundaries cannot see a body that covered the whole
// region between two of them, so `swept` answers that one off the same sweep
// the continuous-collision stage runs, and both of its crossings are recorded
// on the step it happened.
//
// A pair is measured again only when one of its bodies has moved since the
// last measurement; otherwise last step's answer still stands. A settled world
// is mostly sleeping bodies sitting inside regions, so this is what keeps it
// cheap. Every way a body can come to be somewhere marks it: the step's own
// write-back for whatever it simulated, and the world's edits (adding,
// removing, moving or reclassifying a body) for everything else. The marks
// start set, so the first step measures everything.
//
// The queues are reserved once and capped. A caller that stops draining, or a
// world with more regions in it than the reservation covers, is declined and
// counted rather than quietly growing a buffer inside a step.

mod overlap;
pub(super) mod swept;
mod track;

use alloc::vec::Vec;

use crate::memory::Pool;

use crate::physics::SensorCrossing;

use super::body::Body;
use super::broadphase::Pair;
use super::world::{body_at, handle_at};

use track::Overlap;

/// The sensor regions' side of a step: what is inside each of them, and what
/// crossed a boundary to get there.
pub(crate) struct Sensors {
    overlaps: Vec<Overlap>,
    previous: Vec<Overlap>,
    /// Per body slot, whether it may have moved since the last measurement.
    moved: Vec<bool>,
    /// Pairs the last step measured rather than carried over.
    #[cfg(test)]
    measured: usize,
    crossings: Vec<SensorCrossing>,
    overflows: u32,
}

impl Sensors {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Sensors {
            overlaps: Vec::with_capacity(capacity),
            previous: Vec::with_capacity(capacity),
            moved: alloc::vec![true; capacity],
            #[cfg(test)]
            measured: 0,
            crossings: Vec::with_capacity(capacity),
            overflows: 0,
        }
    }

    /// Measure this step's sensor pairs and record every boundary crossed
    /// since the last one.
    ///
    /// `pairs` is the sweep's sensor list: sorted by slot, with a region on at
    /// least one side of every entry.
    pub(crate) fn resolve(&mut self, bodies: &Pool<Body>, pairs: &[Pair]) {
        core::mem::swap(&mut self.overlaps, &mut self.previous);
        self.overlaps.clear();

        let Sensors {
            overlaps,
            previous,
            moved,
            crossings,
            overflows,
            ..
        } = self;
        #[cfg(test)]
        let mut measured = 0;

        let has_moved = |slot: u32| moved.get(slot as usize).copied().unwrap_or(true);
        let mut was = 0usize;
        let mut declined = false;
        for &pair in pairs {
            if !has_moved(pair.0) && !has_moved(pair.1) {
                // Both lists are sorted by pair, so one cursor walks the old
                // answers beside the new pairs. A carried answer takes a
                // reserved place like a measured one: overlaps measured
                // earlier in the pass may already have filled the list.
                while previous.get(was).is_some_and(|old| old.pair < pair) {
                    was += 1;
                }
                if let Some(&old) = previous.get(was).filter(|old| old.pair == pair) {
                    if overlaps.len() == overlaps.capacity() {
                        *overflows = overflows.saturating_add(1);
                        declined = true;
                    } else {
                        overlaps.push(old);
                    }
                }
                continue;
            }
            let (Some(a), Some(b)) = (
                bodies.get_at(pair.0 as usize),
                bodies.get_at(pair.1 as usize),
            ) else {
                continue;
            };
            #[cfg(test)]
            {
                measured += 1;
            }
            if !overlap::overlapping(a, b) {
                continue;
            }
            let (Some(a), Some(b)) = (handle_at(bodies, pair.0), handle_at(bodies, pair.1)) else {
                continue;
            };
            if overlaps.len() == overlaps.capacity() {
                *overflows = overflows.saturating_add(1);
                declined = true;
                continue;
            }
            overlaps.push(Overlap { pair, a, b });
        }

        // A declined overlap is missing from the answers a still pair would
        // reuse, so after one every pair is measured again.
        moved.fill(declined);
        #[cfg(test)]
        {
            self.measured = measured;
        }
        track::transitions(previous, overlaps, |crossed, entered| {
            // Either side may be a region: two of them overlapping record a
            // crossing each, and a region whose body has gone records none.
            for (sensor, other) in [(crossed.a, crossed.b), (crossed.b, crossed.a)] {
                let Some(tag) = body_at(bodies, sensor).and_then(Body::sensor_tag) else {
                    continue;
                };
                let crossing = SensorCrossing {
                    tag,
                    other: body_at(bodies, other).map(|_| other),
                    entered,
                };
                if crossings.len() == crossings.capacity() {
                    *overflows = overflows.saturating_add(1);
                    continue;
                }
                crossings.push(crossing);
            }
        });
    }

    /// Note that a body may be somewhere new, so the next step measures every
    /// pair it is in rather than reusing what it found before.
    pub(crate) fn mark_moved(&mut self, slot: u32) {
        if let Some(moved) = self.moved.get_mut(slot as usize) {
            *moved = true;
        }
    }

    /// Record a body that crossed clean through a region inside one step:
    /// the entry and the exit both, since neither boundary was ever sampled.
    pub(crate) fn record_pass_through(&mut self, bodies: &Pool<Body>, mover: u32, region: u32) {
        let (Some(tag), Some(other)) = (
            bodies.get_at(region as usize).and_then(Body::sensor_tag),
            handle_at(bodies, mover),
        ) else {
            return;
        };
        for entered in [true, false] {
            if self.crossings.len() == self.crossings.capacity() {
                self.overflows = self.overflows.saturating_add(1);
                continue;
            }
            self.crossings.push(SensorCrossing {
                tag,
                other: Some(other),
                entered,
            });
        }
    }

    /// Move the recorded crossings into `out`, oldest first. Both the queue
    /// and `out` keep their capacity.
    pub(crate) fn drain_into(&mut self, out: &mut Vec<SensorCrossing>) {
        out.clear();
        out.append(&mut self.crossings);
    }

    #[cfg(test)]
    /// Crossings and overlaps the reservation had no room for.
    pub(crate) fn overflows(&self) -> u32 {
        self.overflows
    }

    #[cfg(test)]
    pub(crate) fn clear_overflows(&mut self) {
        self.overflows = 0;
    }

    #[cfg(test)]
    /// Pairs the last step measured rather than carried over.
    pub(crate) fn measured(&self) -> usize {
        self.measured
    }

    #[cfg(test)]
    /// Whether the next step measures `slot`'s pairs afresh.
    pub(crate) fn is_marked_moved(&self, slot: u32) -> bool {
        self.moved.get(slot as usize).copied().unwrap_or(true)
    }

    #[cfg(test)]
    /// Pairs currently overlapping.
    pub(crate) fn overlap_count(&self) -> usize {
        self.overlaps.len()
    }

    pub(crate) fn reserved_bytes(&self) -> u64 {
        ((self.overlaps.capacity() + self.previous.capacity()) * size_of::<Overlap>()
            + self.moved.capacity()
            + self.crossings.capacity() * size_of::<SensorCrossing>()) as u64
    }
}
