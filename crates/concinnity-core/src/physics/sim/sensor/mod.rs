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
// Every pair is measured every step rather than carried forward from the last
// one. A pair only reaches here with a region on one side, so there are few of
// them however many bodies the world holds, and a carried answer would have to
// be invalidated by every way a body can arrive somewhere -- including the one
// where it is placed inside a region and never moves again.
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
    crossings: Vec<SensorCrossing>,
    overflows: u32,
}

impl Sensors {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Sensors {
            overlaps: Vec::with_capacity(capacity),
            previous: Vec::with_capacity(capacity),
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
            crossings,
            overflows,
        } = self;

        for &pair in pairs {
            let (Some(a), Some(b)) = (
                bodies.get_at(pair.0 as usize),
                bodies.get_at(pair.1 as usize),
            ) else {
                continue;
            };
            if !overlap::overlapping(a, b) {
                continue;
            }
            let (Some(a), Some(b)) = (handle_at(bodies, pair.0), handle_at(bodies, pair.1)) else {
                continue;
            };
            if overlaps.len() == overlaps.capacity() {
                *overflows = overflows.saturating_add(1);
                continue;
            }
            overlaps.push(Overlap { pair, a, b });
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
    /// Pairs currently overlapping.
    pub(crate) fn overlap_count(&self) -> usize {
        self.overlaps.len()
    }

    pub(crate) fn reserved_bytes(&self) -> u64 {
        ((self.overlaps.capacity() + self.previous.capacity()) * size_of::<Overlap>()
            + self.crossings.capacity() * size_of::<SensorCrossing>()) as u64
    }
}
