// concinnity-physics/src/sim/narrow.rs
//
// This step's contact manifolds, built from the candidate pairs the sweep
// found.
//
// The stage is a pure function of storage nothing here writes: a pair reads two
// bodies, maybe a height grid, and last step's manifolds, and produces the
// manifolds that pair contributes. Two pairs therefore never depend on each
// other, which is what lets a range of them go to whichever worker a caller
// lent -- and it holds whatever shape the world has, unlike the solve, whose
// split is only as good as the islands the contacts happen to form.
//
// What a split has to preserve is the order. Every later stage walks the
// manifolds expecting them keyed by pair -- impulses are carried across steps
// by a merge, and impacts are grouped by reading runs of the same pair -- so
// each worker fills its own buffer and the buffers are appended in range order.
// Whichever worker did the work, the list that comes out is the one the serial
// pass would have built.

use alloc::vec::Vec;

use concinnity_memory::Pool;

use crate::fanout::Fanout;

use super::body::Body;
use super::broadphase::Pair;
use super::collide::heightfield::{self, FieldPair, Heightfields, Incoming};
use super::collide::{self, Pose};
use super::contact::Manifold;

/// Candidate pairs below which the stage runs on the calling thread. A short
/// list is answered before a dispatch would have finished.
const MIN_FANOUT_PAIRS: usize = 96;

/// One worker's share of the candidate pairs, and the manifolds that share
/// produced.
#[derive(Debug, Default)]
struct Share {
    from: usize,
    to: usize,
    out: Vec<Manifold>,
}

/// The narrow phase's per-worker buffers.
pub(crate) struct Narrow {
    shares: Vec<Share>,
}

impl Narrow {
    pub(crate) fn new() -> Self {
        Narrow { shares: Vec::new() }
    }

    /// Reserve a manifold buffer per worker. Called while the world is built;
    /// a simulation nobody lends threads to keeps none of these.
    pub(crate) fn reserve_workers(&mut self, workers: usize, capacity: usize) {
        self.shares.clear();
        if workers < 2 {
            return;
        }
        // The contact cache's own reservation, shared out: a worker walks its
        // share of the pairs, so it builds about its share of the manifolds.
        let share = (capacity * 2).div_ceil(workers);
        self.shares.resize_with(workers, || Share {
            from: 0,
            to: 0,
            out: Vec::with_capacity(share),
        });
    }

    pub(crate) fn reserved_bytes(&self) -> u64 {
        self.shares
            .iter()
            .map(|share| (share.out.capacity() * size_of::<Manifold>()) as u64)
            .sum()
    }

    /// Build this step's manifolds from the candidate pairs, in pair order.
    pub(crate) fn build(&mut self, work: Work<'_>, fanout: &impl Fanout, workers: usize) {
        let Work {
            bodies,
            fields,
            pairs,
            previous,
            out,
            margin,
        } = work;
        let workers = workers.min(self.shares.len());
        if workers < 2 || pairs.len() < MIN_FANOUT_PAIRS {
            build_range(bodies, fields, pairs, previous, out, margin);
            return;
        }
        let share = pairs.len().div_ceil(workers);
        let shares = &mut self.shares[..workers];
        for (index, slot) in shares.iter_mut().enumerate() {
            slot.from = (index * share).min(pairs.len());
            slot.to = ((index + 1) * share).min(pairs.len());
        }
        fanout.for_each(shares, |slot| {
            slot.out.clear();
            build_range(
                bodies,
                fields,
                &pairs[slot.from..slot.to],
                previous,
                &mut slot.out,
                margin,
            );
        });
        for slot in shares.iter() {
            out.extend_from_slice(&slot.out);
        }
    }
}

/// What one narrow phase reads and where it writes.
pub(crate) struct Work<'a> {
    pub(crate) bodies: &'a Pool<Body>,
    pub(crate) fields: &'a Heightfields,
    pub(crate) pairs: &'a [Pair],
    pub(crate) previous: &'a [Manifold],
    pub(crate) out: &'a mut Vec<Manifold>,
    pub(crate) margin: f32,
}

/// The manifolds one run of candidate pairs produces, in pair order.
///
/// A pair neither of whose bodies is simulated cannot have changed since last
/// step, so its manifold is carried over rather than recomputed. That is what
/// makes a settled world cheap, and it keeps the warm-start impulses a
/// sleeping stack will need when something wakes it.
fn build_range(
    bodies: &Pool<Body>,
    fields: &Heightfields,
    pairs: &[Pair],
    previous: &[Manifold],
    out: &mut Vec<Manifold>,
    margin: f32,
) {
    let mut scratch = Manifold::new(0, 0);
    for &(a, b) in pairs {
        let (Some(body_a), Some(body_b)) = (bodies.get_at(a as usize), bodies.get_at(b as usize))
        else {
            continue;
        };
        if !body_a.is_simulated() && !body_b.is_simulated() {
            out.extend_from_slice(super::contact::find(previous, (a, b)));
            continue;
        }
        // Two rough surfaces are rougher than either alone but not by their
        // sum, and a pair bounces as hard as its bouncier half.
        let pair = FieldPair {
            a,
            b,
            reversed: false,
            friction: libm::sqrtf(body_a.friction * body_b.friction),
            restitution: body_a.restitution.max(body_b.restitution),
        };
        match (body_a.terrain_index(), body_b.terrain_index()) {
            // Terrain never moves, so two grids can never meet.
            (Some(_), Some(_)) => {}
            (Some(index), None) => {
                collide_field(fields, index, body_b, margin, pair, out);
            }
            (None, Some(index)) => {
                collide_field(
                    fields,
                    index,
                    body_a,
                    margin,
                    FieldPair {
                        reversed: true,
                        ..pair
                    },
                    out,
                );
            }
            (None, None) => {
                let (Some(shape_a), Some(shape_b)) = (body_a.convex(), body_b.convex()) else {
                    continue;
                };
                scratch.a = a;
                scratch.b = b;
                if !collide::collide(
                    shape_a,
                    pose_of(body_a),
                    shape_b,
                    pose_of(body_b),
                    margin,
                    &mut scratch,
                ) {
                    continue;
                }
                scratch.friction = pair.friction;
                scratch.restitution = pair.restitution;
                out.push(scratch);
            }
        }
    }
}

/// The manifolds one convex body makes with one height grid.
fn collide_field(
    fields: &Heightfields,
    index: u32,
    body: &Body,
    margin: f32,
    pair: FieldPair,
    out: &mut Vec<Manifold>,
) {
    let Some(shape) = body.convex() else {
        return;
    };
    heightfield::collide_into(
        fields,
        index,
        Incoming {
            shape,
            pose: pose_of(body),
            bounds: body.tight_bounds().expanded(margin),
        },
        margin,
        pair,
        out,
    );
}

fn pose_of(body: &Body) -> Pose {
    Pose {
        position: body.position,
        rotation: body.orientation,
    }
}
