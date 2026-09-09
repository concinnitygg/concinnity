// The read phase of a tick: what the bodies may see, the buffers they run
// against, and how those buffers are worked through.
//
// Bodies never observe another body's same-tick writes (the apply phase runs
// after every body), which is what makes evaluation order unobservable and lets
// a host fan the runs across threads.

use alloc::vec::Vec;

use super::BehaviorSystem;
use super::instance::Instance;
use crate::behavior::{Effect, Program, Spatial, Val, View, exec, position, spatial};
use crate::components::Transform;
use crate::ecs::{ComponentStorage, Entity, EntityByName, PipelineContext};

// Below this many firing instances the fan-out costs more than the work.
pub(super) const PARALLEL_EVAL_MIN_JOBS: usize = 64;

/// One worker's share of a tick's evaluation: a contiguous slice of the tick's
/// job list, its own effect and trace buffers, and its own binding scratch.
/// Opaque to a scheduler, which only hands each bucket to the closure it was
/// given. Everything inside keeps its capacity across ticks.
#[derive(Debug, Default)]
pub struct EvalBucket {
    pub(super) jobs: core::ops::Range<usize>,
    pub(super) effects: Vec<Effect>,
    pub(super) produced: Vec<(usize, Option<Entity>, usize)>,
    pub(super) fired: Vec<(usize, Vec<u32>)>,
    pub(super) bindings: Vec<Option<Val>>,
}

/// Runs a tick's evaluation buckets, in any order and on any thread.
///
/// The buckets share nothing: each reads only the tick's starting state and
/// writes only its own effects, so a host with a thread pool can work them
/// through in parallel. A world whose host installs none evaluates every run on
/// the calling thread.
pub trait EvalScheduler: core::fmt::Debug + Send {
    /// How many buckets to split a tick's firing instances into.
    fn workers(&self) -> usize;

    /// Apply `eval` to every bucket, then return.
    fn run(&self, buckets: &mut [EvalBucket], eval: &(dyn Fn(&mut EvalBucket) + Send + Sync));
}

// The entity sets a body iterates this tick, resolved before anything runs.
//
// Single-entity reads (a name, a position, a liveness test) are not here: no
// body runs while the world is mutable, so they read the world directly and
// cost one lookup each instead of a whole-world copy per tick.
#[derive(Debug, Default)]
pub(super) struct Snapshot {
    // Per program, per declared query, in stable order.
    pub(super) queries: Vec<Vec<Vec<Entity>>>,
    // Per program, the entities its scope matched, in stable order.
    pub(super) scoped: Vec<Vec<Entity>>,
}

// What a body run reads that the tick, rather than the body, determines: the
// per-tick half of the behavior VM's `View`. Everything here is shared and
// immutable, so evaluation can fan across workers; single-entity reads (a name,
// a position, a liveness test) read the storage directly and cost one lookup
// each instead of a whole-world copy per tick.
pub(super) struct EvalCtx<'a> {
    pub(super) components: &'a ComponentStorage,
    // Resolved once per tick rather than per name lookup.
    pub(super) names: Option<&'a EntityByName>,
    pub(super) snapshot: &'a Snapshot,
    pub(super) programs: &'a [Program],
    pub(super) instances: &'a [Vec<Instance>],
    pub(super) vars: &'a [Val],
    pub(super) dt: f32,
    pub(super) elapsed: f32,
    pub(super) tracing: bool,
}

// Run one instance's body against the tick's starting state, appending its
// effects to `out`. Returns how many it appended, plus the nodes it executed
// when tracing.
//
// `bindings` is the caller's reused buffer: resized to the body's compiled
// binding high-water mark and cleared per run, so however many instances fire
// a tick, binding scratch costs zero allocations in steady state. (The
// previous per-run frame-arena grab exhausted the reserve on behavior-heavy
// worlds and degraded to contended heap allocation across eval workers.)
pub(super) fn eval_one(
    ec: &EvalCtx<'_>,
    bindings: &mut Vec<Option<Val>>,
    job: &Job,
    out: &mut Vec<Effect>,
) -> Option<(usize, Vec<u32>)> {
    let (i, entity) = (job.program, job.entity);
    let locals = ec.instances[i]
        .iter()
        .find(|inst| inst.entity == entity)
        .map(|inst| inst.locals.as_slice())?;
    // A deferred block resumes on the frame its `after` node captured; a whole
    // body starts on a cleared one. Either way the frame is the body's compiled
    // width, so every slot a resumed block reads is in range.
    bindings.clear();
    bindings.resize(ec.programs[i].bindings, None);
    let nodes = match &job.resume {
        Some(resume) => {
            for (slot, value) in resume.bindings.iter().enumerate().take(bindings.len()) {
                bindings[slot] = *value;
            }
            ec.programs[i].deferred_block(resume.node)?
        }
        None => &ec.programs[i].body,
    };
    let mut traced: Option<Vec<u32>> = ec.tracing.then(Vec::new);
    let before = out.len();
    let mut view = View {
        dt: ec.dt,
        elapsed: ec.elapsed,
        vars: ec.vars,
        locals,
        bindings: bindings.as_mut_slice(),
        queries: &ec.snapshot.queries[i],
        // A name index entry can outlive its entity, so each is confirmed.
        by_name: &|id| {
            ec.names
                .and_then(|n| n.get(id))
                .filter(|e| ec.components.is_alive(*e))
        },
        positions: &|e| position::of(ec.components, e),
        transforms: &|e| ec.components.get::<Transform>(e).copied(),
        alive: &|e| ec.components.is_alive(e),
        // The behavior's own entity is never an answer: a ray cast from
        // `position(self)` starts inside self's own collider, and an entity is
        // never its own nearest.
        spatial: &|question| answer(ec.components, question, entity),
        self_entity: entity,
        trace: &mut traced,
    };
    exec(nodes, &mut view, out);
    Some((out.len() - before, traced.unwrap_or_default()))
}

// Resolve one spatial question against the world, excluding the running
// instance's own entity.
fn answer(
    components: &ComponentStorage,
    question: &Spatial<'_>,
    self_entity: Option<Entity>,
) -> Option<Val> {
    match *question {
        Spatial::Nearest { candidates, point } => {
            spatial::nearest(components, candidates, point, self_entity).map(Val::Entity)
        }
        Spatial::CountWithin {
            candidates,
            point,
            radius,
        } => Some(Val::Int(spatial::count_within(
            components,
            candidates,
            point,
            radius,
            self_entity,
        ))),
        Spatial::Raycast {
            candidates,
            from,
            dir,
            distance,
        } => spatial::raycast(components, candidates, from, dir, distance, self_entity)
            .map(Val::Entity),
    }
}

/// One run queued for this tick: a whole body, or a block an earlier tick's
/// `after` node deferred.
#[derive(Debug)]
pub(super) struct Job {
    pub(super) program: usize,
    pub(super) entity: Option<Entity>,
    pub(super) resume: Option<Resume>,
}

/// Where a deferred run picks up: the `after` node whose block it runs, and the
/// binding frame that node captured.
#[derive(Debug, Clone)]
pub(super) struct Resume {
    pub(super) node: u32,
    pub(super) bindings: Vec<Option<Val>>,
}

impl BehaviorSystem {
    // Entities carrying every one of these component tags, filled into `out`
    // in stable order. Column order shifts as entities are removed
    // (swap-remove), so the result is sorted: an unstable iteration order
    // would make a body's effects depend on unrelated despawns. `scratch`
    // holds each extra tag's sorted set; both buffers keep their capacity.
    pub(super) fn entities_matching_into(
        ctx: &PipelineContext,
        tags: &[u8],
        scratch: &mut Vec<Entity>,
        out: &mut Vec<Entity>,
    ) {
        out.clear();
        let Some((first, rest)) = tags.split_first() else {
            return;
        };
        out.extend_from_slice(ctx.entities_with_tag(*first));
        for tag in rest {
            scratch.clear();
            scratch.extend_from_slice(ctx.entities_with_tag(*tag));
            scratch.sort_unstable_by_key(|e| e.to_bits());
            out.retain(|e| {
                scratch
                    .binary_search_by_key(&e.to_bits(), |o| o.to_bits())
                    .is_ok()
            });
        }
        out.sort_unstable_by_key(|e| e.to_bits());
    }

    // Refill the reused snapshot with this tick's entity sets. The shape
    // (programs and their query counts) is fixed after `init`, so in steady
    // state every vector here just refills in place.
    pub(super) fn gather(&mut self, ctx: &PipelineContext, snapshot: &mut Snapshot) {
        snapshot.queries.resize_with(self.programs.len(), Vec::new);
        snapshot.scoped.resize_with(self.programs.len(), Vec::new);
        for (i, p) in self.programs.iter().enumerate() {
            snapshot.queries[i].resize_with(p.queries.len(), Vec::new);
            for (q, tags) in p.queries.iter().enumerate() {
                Self::entities_matching_into(
                    ctx,
                    tags,
                    &mut self.tag_scratch,
                    &mut snapshot.queries[i][q],
                );
            }
            Self::entities_matching_into(
                ctx,
                &p.scope,
                &mut self.tag_scratch,
                &mut snapshot.scoped[i],
            );
        }
    }
}
