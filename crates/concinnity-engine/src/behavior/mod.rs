// BehaviorSystem: declarative logic over typed state, world reads, and a
// per-frame tick.
//
//   mod.rs      system, per-tick drive, effect application
//   program.rs  compiled bodies: names resolved to dense slots
//   run.rs      expression evaluation and node execution
//   save.rs     persisted world variables and `once` flags
//
// A behavior with an empty `scope` runs once per firing; a scoped one runs once
// per matching entity, each with its own locals. Each tick is read, run, apply:
// the body sees a snapshot and produces effects, and only then does anything
// change, so no behavior observes another's writes mid-tick.
//
// Scheduled before SpawnSystem, so a spawn
// requested this tick lands this tick, and before SettingsSystem / StorySystem
// / AudioSystem so scene, story, and audio requests land the same tick too.
// Clocks freeze while a menu is open, like the rest of the world clock.

use std::path::PathBuf;

use crate::assets::{
    Behavior, BehaviorSource, DespawnRequest, InteractSignal, PlayCue, ReparentRequest,
    SceneCommand, ScreenCommand, SpawnRequest, StoryCommand, StoryPlayback, Transform, Variables,
    VisibilityRequest, VolumeEvent,
};
use crate::ecs::{Entity, EventCursor, PipelineContext, StepResult, System, asset_id::AssetId};

mod program;
mod run;
mod save;
mod trace;

#[cfg(test)]
mod bench;
#[cfg(test)]
mod tests;

use program::{Program, Val, VarTable};
use run::{Effect, View};

// One behavior's firing state for one entity (or for the world, when the
// behavior is unscoped).
#[derive(Debug)]
struct Instance {
    entity: Option<Entity>,
    locals: Vec<Val>,
    started: bool,
    spawned_pending: bool,
    fired_once: bool,
    cooldown_left: f32,
    timer_accum: f32,
    timer_done: bool,
    last_value: Val,
}

impl Instance {
    fn new(entity: Option<Entity>, locals: Vec<Val>, spawned_pending: bool) -> Self {
        Self {
            entity,
            locals,
            started: false,
            spawned_pending,
            fired_once: false,
            cooldown_left: 0.0,
            timer_accum: 0.0,
            timer_done: false,
            last_value: Val::Int(0),
        }
    }

    // Advance this instance's clocks by dt and decide whether it fires.
    fn due(
        &mut self,
        def: &Behavior,
        vars: &[Val],
        var_slot: Option<u16>,
        dt: f32,
        crossings: &[VolumeEvent],
        presses: &[InteractSignal],
    ) -> bool {
        self.cooldown_left = (self.cooldown_left - dt).max(0.0);
        let sourced = match &def.on {
            BehaviorSource::Start => !std::mem::replace(&mut self.started, true),
            BehaviorSource::Tick => true,
            BehaviorSource::Spawned => std::mem::replace(&mut self.spawned_pending, false),
            BehaviorSource::Timer { interval, repeat } => self.timer_due(*interval, *repeat, dt),
            BehaviorSource::Variable(_) => {
                let current = var_slot
                    .and_then(|s| vars.get(s as usize))
                    .copied()
                    .unwrap_or(Val::Int(0));
                let changed = current != self.last_value;
                self.last_value = current;
                changed
            }
            BehaviorSource::Enter(volume) => crossing_matches(crossings, *volume, true),
            BehaviorSource::Exit(volume) => crossing_matches(crossings, *volume, false),
            BehaviorSource::Interact(target) => {
                target.is_some_and(|target| presses.iter().any(|p| p.target == target))
            }
        };
        if !sourced || (def.once && self.fired_once) || self.cooldown_left > 0.0 {
            return false;
        }
        self.fired_once = true;
        self.cooldown_left = def.cooldown;
        true
    }

    fn timer_due(&mut self, interval: f32, repeat: bool, dt: f32) -> bool {
        if self.timer_done {
            return false;
        }
        self.timer_accum += dt;
        if self.timer_accum < interval {
            return false;
        }
        if repeat {
            // At most one firing per tick. Fixed ticks make the subtraction
            // exact for any interval above the tick length; a sub-tick
            // interval fires every tick without accumulating unbounded debt.
            self.timer_accum = (self.timer_accum - interval).min(interval);
        } else {
            self.timer_done = true;
        }
        true
    }
}

fn crossing_matches(crossings: &[VolumeEvent], volume: Option<AssetId>, entered: bool) -> bool {
    let Some(volume) = volume else { return false };
    crossings
        .iter()
        .any(|e| e.entered == entered && e.volume == volume)
}

#[derive(Debug)]
pub struct BehaviorSystem {
    programs: Vec<Program>,
    // Parallel to `programs`.
    instances: Vec<Vec<Instance>>,
    vars: Vec<Val>,
    var_table: VarTable,
    // Delayed runs: (program, the instance's entity, seconds left).
    pending: Vec<(usize, Option<Entity>, f32)>,
    crossing_cursor: EventCursor,
    press_cursor: EventCursor,
    crossings: Vec<VolumeEvent>,
    presses: Vec<InteractSignal>,
    save_dir: PathBuf,
    // Sampled from the `TransientSaves` resource at init: while true, the
    // state file is neither read nor written, so a preview session starts
    // fresh and leaves the user's saves untouched.
    transient_saves: bool,
    // Execution tracing (see `trace.rs`): the published tick counter, and
    // whether the node-path table has been published this world.
    trace_frame: u64,
    trace_paths_published: bool,
    // Fixed ticks run so far; elapsed simulated time is this times the tick
    // length, so behavior clocks advance with the simulation, not the wall.
    sim_ticks: u64,
    // Instances present before the first tick are the world's initial
    // population, so `spawned` does not fire for them.
    populated: bool,
    // Per-worker evaluation state, grown to the job pool's width on the first
    // parallel tick and reused thereafter so steady-state allocations stay
    // flat.
    eval_buckets: Vec<EvalBucket>,
    // The tick's firing list and the serial path's binding scratch, kept for
    // their capacity across ticks.
    jobs: Vec<(usize, Option<Entity>)>,
    bindings: Vec<Option<Val>>,
    // The tick's resolved entity sets and the tag-intersection scratch, kept
    // for their capacity across ticks like the buffers above.
    snapshot: Snapshot,
    tag_scratch: Vec<Entity>,
}

impl Default for BehaviorSystem {
    fn default() -> Self {
        Self {
            programs: Vec::new(),
            instances: Vec::new(),
            vars: Vec::new(),
            var_table: VarTable::default(),
            pending: Vec::new(),
            crossing_cursor: EventCursor::default(),
            press_cursor: EventCursor::default(),
            crossings: Vec::new(),
            presses: Vec::new(),
            save_dir: concinnity_store::paths::saves_dir(),
            transient_saves: false,
            trace_frame: 0,
            trace_paths_published: false,
            sim_ticks: 0,
            populated: false,
            eval_buckets: Vec::new(),
            jobs: Vec::new(),
            bindings: Vec::new(),
            snapshot: Snapshot::default(),
            tag_scratch: Vec::new(),
        }
    }
}

impl BehaviorSystem {
    pub fn new() -> Self {
        Self::default()
    }
}

impl System for BehaviorSystem {
    fn init(&mut self, ctx: &mut PipelineContext) {
        // The world's declared variables get their slots first, so each carries
        // its authored type and starting value; anything a behavior mentions
        // without a declaration follows as an integer starting at zero.
        let mut var_table = VarTable::default();
        for declared in ctx.query::<Variables>() {
            for decl in &declared.vars {
                var_table.declare(&decl.name, Val::from_literal(&decl.value));
            }
        }
        let defs: Vec<Behavior> = ctx.query::<Behavior>().cloned().collect();
        self.programs = defs
            .into_iter()
            .map(|def| program::compile(def, &mut var_table))
            .collect();
        self.instances = self.programs.iter().map(|_| Vec::new()).collect();
        self.vars = var_table.initial();
        self.var_table = var_table;

        self.transient_saves = ctx
            .resource::<crate::ecs::TransientSaves>()
            .is_some_and(|t| t.0);
        self.trace_frame = 0;
        self.trace_paths_published = false;
        self.sim_ticks = 0;

        // Restore persisted state, but only in a world that saves: any other
        // world starts fresh and never touches the state file.
        let mut restored = 0usize;
        if !self.transient_saves
            && self.programs.iter().any(|p| p.def.saves_state())
            && let Some(state) = save::read_save(&save::state_file(&self.save_dir))
        {
            for (name, value) in &state.vars {
                let Some(slot) = self.var_table.slot_of(name) else {
                    continue;
                };
                // A save written before the world retyped a variable no longer
                // applies to it; the declared starting value stands.
                let value = Val::from_literal(value);
                if self.vars[slot as usize].same_type(value) {
                    self.vars[slot as usize] = value;
                    restored += 1;
                }
            }
            for (id, hash) in state.fired {
                if let Some(i) = self
                    .programs
                    .iter()
                    .position(|p| p.def.asset_id.0 == id && save::def_hash(&p.def) == hash)
                {
                    // World-scoped `once` state restores onto the single
                    // instance; a scoped behavior's per-entity flags are not
                    // persisted, matching its locals.
                    if !self.programs[i].is_scoped() {
                        let mut instance = Instance::new(None, Vec::new(), false);
                        instance.fired_once = true;
                        self.instances[i] = vec![instance];
                    }
                }
            }
        }
        tracing::info!(
            "BehaviorSystem: {} behavior(s), {} variable(s), restored {}",
            self.programs.len(),
            self.vars.len(),
            restored,
        );
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        if self.programs.is_empty() {
            return StepResult::Continue;
        }

        if let Some(events) = ctx.events::<VolumeEvent>() {
            self.crossings
                .extend(events.read(&mut self.crossing_cursor).copied());
        }
        if let Some(events) = ctx.events::<InteractSignal>() {
            self.presses
                .extend(events.read(&mut self.press_cursor).copied());
        }

        let menu_active = ctx
            .resource::<crate::ecs::MenuActive>()
            .map(|m| m.0)
            .unwrap_or(false);
        if menu_active {
            return StepResult::Continue;
        }

        // The frame's fixed-tick budget. Absent (a directly-stepped world with
        // no App), every step runs exactly one tick. Edge events (crossings,
        // presses) are consumed by the frame's first tick; catch-up ticks see
        // none, so an edge never fires twice.
        let timing = ctx
            .resource::<crate::ecs::SimTiming>()
            .copied()
            .unwrap_or_default();
        for _ in 0..timing.ticks {
            self.sim_ticks += 1;
            let elapsed = (self.sim_ticks as f64 * timing.tick_dt as f64) as f32;
            self.tick(ctx, timing.tick_dt, elapsed);
        }
        StepResult::Continue
    }
}

// The entity sets a body iterates this tick, resolved before anything runs.
//
// Single-entity reads (a name, a position, a liveness test) are not here: no
// body runs while the world is mutable, so they read the world directly and
// cost one lookup each instead of a whole-world copy per tick.
#[derive(Debug, Default)]
struct Snapshot {
    // Per program, per declared query, in stable order.
    queries: Vec<Vec<Vec<Entity>>>,
    // Per program, the entities its scope matched, in stable order.
    scoped: Vec<Vec<Entity>>,
}

// What a body run reads that the tick, rather than the body, determines: the
// per-tick half of `run::View`. Everything here is shared and immutable, so
// evaluation can fan across workers; single-entity reads (a name, a position,
// a liveness test) read the storage directly and cost one lookup each instead
// of a whole-world copy per tick.
struct EvalCtx<'a> {
    components: &'a crate::ecs::ComponentStorage,
    // Resolved once per tick rather than per name lookup.
    names: Option<&'a crate::ecs::decompose::EntityByName>,
    snapshot: &'a Snapshot,
    programs: &'a [Program],
    instances: &'a [Vec<Instance>],
    vars: &'a [Val],
    dt: f32,
    elapsed: f32,
    tracing: bool,
}

// Run one instance's body against the tick's starting state, appending its
// effects to `out`. Returns how many it appended, plus the nodes it executed
// when tracing. Bodies never observe another body's same-tick writes (the
// apply phase runs after every body), which is what makes evaluation order
// unobservable and this function safe to run concurrently.
//
// `bindings` is the caller's reused buffer: resized to the body's compiled
// binding high-water mark and cleared per run, so however many instances fire
// a tick, binding scratch costs zero allocations in steady state. (The
// previous per-run frame-arena grab exhausted the reserve on behavior-heavy
// worlds and degraded to contended heap allocation across eval workers.)
fn eval_one(
    ec: &EvalCtx<'_>,
    bindings: &mut Vec<Option<Val>>,
    i: usize,
    entity: Option<Entity>,
    out: &mut Vec<Effect>,
) -> Option<(usize, Vec<u32>)> {
    let locals = ec.instances[i]
        .iter()
        .find(|inst| inst.entity == entity)
        .map(|inst| inst.locals.as_slice())?;
    bindings.clear();
    bindings.resize(ec.programs[i].bindings, None);
    let mut nodes: Option<Vec<u32>> = ec.tracing.then(Vec::new);
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
        transforms: &|e| ec.components.get::<Transform>(e).copied(),
        alive: &|e| ec.components.is_alive(e),
        self_entity: entity,
        trace: &mut nodes,
    };
    run::exec(&ec.programs[i].body, &mut view, out);
    Some((out.len() - before, nodes.unwrap_or_default()))
}

// One worker's share of a parallel evaluation: a contiguous slice of the
// tick's job list, its own effect/trace buffers, and its own binding scratch.
// Everything keeps its capacity across ticks.
#[derive(Debug, Default)]
struct EvalBucket {
    jobs: core::ops::Range<usize>,
    effects: Vec<Effect>,
    produced: Vec<(usize, Option<Entity>, usize)>,
    fired: Vec<(usize, Vec<u32>)>,
    bindings: Vec<Option<Val>>,
}

// Below this many firing instances the fan-out costs more than the work.
const PARALLEL_EVAL_MIN_JOBS: usize = 64;

impl BehaviorSystem {
    // Entities carrying every one of these component tags, filled into `out`
    // in stable order. Column order shifts as entities are removed
    // (swap-remove), so the result is sorted: an unstable iteration order
    // would make a body's effects depend on unrelated despawns. `scratch`
    // holds each extra tag's sorted set; both buffers keep their capacity.
    fn entities_matching_into(
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
    fn gather(&mut self, ctx: &PipelineContext, snapshot: &mut Snapshot) {
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

    // Create instances for newly matching entities and drop those whose entity
    // is gone, preserving the state of everything that persists.
    fn resync_instances(&mut self, snapshot: &Snapshot) {
        // A variable source starts baselined at the variable's current value,
        // so a restored save does not read as a change on the instance's first
        // tick. Read before the loop, which borrows `self.instances` mutably.
        let baselines: Vec<Val> = self
            .programs
            .iter()
            .map(|p| match &p.def.on {
                BehaviorSource::Variable(name) => self
                    .var_table
                    .slot_of(name)
                    .and_then(|s| self.vars.get(s as usize))
                    .copied()
                    .unwrap_or(Val::Int(0)),
                _ => Val::Int(0),
            })
            .collect();
        for (i, program) in self.programs.iter().enumerate() {
            if !program.is_scoped() {
                if self.instances[i].is_empty() {
                    let mut instance = Instance::new(None, Vec::new(), false);
                    instance.last_value = baselines[i];
                    self.instances[i].push(instance);
                }
                continue;
            }
            let matched = &snapshot.scoped[i];
            self.instances[i].retain(|inst| {
                inst.entity.is_some_and(|e| {
                    matched
                        .binary_search_by_key(&e.to_bits(), |o| o.to_bits())
                        .is_ok()
                })
            });
            // The retained instances are a sorted subset of the sorted
            // `matched`, so one merge walk finds the entities without an
            // instance; in the steady state nothing is appended and the order
            // already stands.
            let before = self.instances[i].len();
            let mut j = 0;
            for entity in matched {
                if j < before && self.instances[i][j].entity == Some(*entity) {
                    j += 1;
                    continue;
                }
                let mut instance =
                    Instance::new(Some(*entity), program.local_inits.clone(), self.populated);
                instance.last_value = baselines[i];
                self.instances[i].push(instance);
            }
            if self.instances[i].len() != before {
                self.instances[i].sort_by_key(|inst| inst.entity.map(|e| e.to_bits()));
            }
        }
    }

    fn tick(&mut self, ctx: &mut PipelineContext, dt: f32, elapsed: f32) {
        // Execution tracing is on only while an observer's request stands
        // (the editor's Behavior panel); its absence costs this one lookup.
        let request = ctx.resource::<crate::ecs::TraceRequest>().cloned();
        let tracing = request.is_some();
        let mut fired: Vec<(usize, Vec<u32>)> = Vec::new();

        let mut snapshot = std::mem::take(&mut self.snapshot);
        self.gather(ctx, &mut snapshot);
        self.resync_instances(&snapshot);
        self.populated = true;

        // Fire decisions run against this tick's starting values, so a `set`
        // here is seen by variable-source behaviors next tick and chains
        // advance one link per tick.
        let mut runs: Vec<(usize, Option<Entity>)> = Vec::new();
        for i in 0..self.programs.len() {
            let var_slot = match &self.programs[i].def.on {
                BehaviorSource::Variable(name) => self.var_table.slot_of(name),
                _ => None,
            };
            let def = &self.programs[i].def;
            for instance in &mut self.instances[i] {
                if instance.due(
                    def,
                    &self.vars,
                    var_slot,
                    dt,
                    &self.crossings,
                    &self.presses,
                ) {
                    runs.push((i, instance.entity));
                }
            }
        }
        self.crossings.clear();
        self.presses.clear();

        // Delayed runs from earlier ticks count down first: those now due run
        // before this tick's firings, in discovery order, exactly as before.
        // A fresh delay pushed below starts counting next tick.
        let mut jobs = std::mem::take(&mut self.jobs);
        jobs.clear();
        let mut idx = 0;
        while idx < self.pending.len() {
            self.pending[idx].2 -= dt;
            if self.pending[idx].2 <= 0.0 {
                let (i, entity, _) = self.pending.swap_remove(idx);
                jobs.push((i, entity));
            } else {
                idx += 1;
            }
        }
        for (i, entity) in runs {
            let delay = self.programs[i].def.delay;
            if delay > 0.0 {
                self.pending.push((i, entity, delay));
            } else {
                jobs.push((i, entity));
            }
        }

        // Read phase: every body runs against an unchanged world, so the
        // borrow here is shared and the effects it produces are applied only
        // after it ends. Serially each run appends into one buffer and
        // records how much it added; with enough firing instances under the
        // parallel schedule, contiguous job chunks evaluate on the pool into
        // per-worker buffers instead. Either way a body observes only the
        // tick's starting state, so the results are identical; only the apply
        // order below is observable, and it walks jobs in list order in both
        // modes.
        let parallel = jobs.len() >= PARALLEL_EVAL_MIN_JOBS
            && crate::ecs::ScheduleMode::current(ctx.resources)
                == crate::ecs::ScheduleMode::Parallel;
        let mut effects: Vec<Effect> = Vec::new();
        let mut produced: Vec<(usize, Option<Entity>, usize)> = Vec::new();
        let mut buckets = std::mem::take(&mut self.eval_buckets);
        let mut serial_bindings = std::mem::take(&mut self.bindings);
        {
            let ec = EvalCtx {
                components: ctx.components,
                names: ctx.resource::<crate::ecs::decompose::EntityByName>(),
                snapshot: &snapshot,
                programs: &self.programs,
                instances: &self.instances,
                vars: &self.vars,
                dt,
                elapsed,
                tracing,
            };
            if parallel {
                let workers = crate::jobs::pool().thread_count().max(1);
                while buckets.len() < workers {
                    buckets.push(EvalBucket::default());
                }
                let chunk = jobs.len().div_ceil(buckets.len()).max(1);
                for (b, bucket) in buckets.iter_mut().enumerate() {
                    bucket.jobs = (b * chunk).min(jobs.len())..((b + 1) * chunk).min(jobs.len());
                }
                let jobs = &jobs;
                let ec = &ec;
                crate::jobs::pool().parallel_for(&mut buckets, |bucket| {
                    bucket.effects.clear();
                    bucket.produced.clear();
                    bucket.fired.clear();
                    for &(i, entity) in &jobs[bucket.jobs.clone()] {
                        if let Some((count, nodes)) =
                            eval_one(ec, &mut bucket.bindings, i, entity, &mut bucket.effects)
                        {
                            bucket.produced.push((i, entity, count));
                            if ec.tracing {
                                bucket.fired.push((i, nodes));
                            }
                        }
                    }
                });
            } else {
                for &(i, entity) in &jobs {
                    if let Some((count, nodes)) =
                        eval_one(&ec, &mut serial_bindings, i, entity, &mut effects)
                    {
                        produced.push((i, entity, count));
                        if tracing {
                            fired.push((i, nodes));
                        }
                    }
                }
            }
        }

        // Walking each buffer once hands each run exactly the effects it
        // appended, in record order, without copying or reshuffling them.
        // Bucket order is job order, so the parallel apply is byte-identical
        // to the serial one.
        let mut save_requested = false;
        if parallel {
            for bucket in &mut buckets {
                let mut recorded = bucket.effects.drain(..);
                for k in 0..bucket.produced.len() {
                    let (i, entity, count) = bucket.produced[k];
                    save_requested |= self.apply(ctx, i, entity, recorded.by_ref().take(count));
                }
                if tracing {
                    fired.append(&mut bucket.fired);
                }
            }
        } else {
            let mut recorded = effects.into_iter();
            for (i, entity, count) in produced {
                save_requested |= self.apply(ctx, i, entity, recorded.by_ref().take(count));
            }
        }
        self.eval_buckets = buckets;
        self.jobs = jobs;
        self.bindings = serial_bindings;
        self.snapshot = snapshot;

        // One write per tick, after every effect has landed, so the file holds
        // this tick's final values.
        if save_requested {
            self.write_state();
        }

        if let Some(request) = request {
            self.publish_trace(ctx, &request, &fired);
        }
    }

    // Land one body's effects. Returns whether a `save` was requested.
    fn apply(
        &mut self,
        ctx: &mut PipelineContext,
        i: usize,
        entity: Option<Entity>,
        effects: impl Iterator<Item = Effect>,
    ) -> bool {
        let mut save_requested = false;
        for effect in effects {
            match effect {
                Effect::SetVar { slot, value, add } => {
                    if let Some(current) = self.vars.get_mut(slot as usize) {
                        *current = if add {
                            add_vals(*current, value)
                        } else {
                            value
                        };
                    }
                }
                Effect::SetLocal { slot, value, add } => {
                    let Some(instance) = self.instances[i]
                        .iter_mut()
                        .find(|inst| inst.entity == entity)
                    else {
                        continue;
                    };
                    let Some(current) = instance.locals.get_mut(slot as usize) else {
                        continue;
                    };
                    *current = if add {
                        add_vals(*current, value)
                    } else {
                        value
                    };
                }
                Effect::SetTransform { entity, transform } => {
                    if let Some(current) = ctx.get_mut::<Transform>(entity) {
                        *current = transform;
                    }
                }
                Effect::Spawn(spawn) => {
                    ctx.events_mut::<SpawnRequest>().send(SpawnRequest {
                        template: spawn.template,
                        name: None,
                        transform: spawn.transform,
                        lifetime_secs: spawn.lifetime,
                    });
                }
                Effect::Despawn(entity) => {
                    ctx.events_mut::<DespawnRequest>().send(DespawnRequest {
                        target: entity.into(),
                    });
                }
                Effect::Reparent { child, parent } => {
                    ctx.events_mut::<ReparentRequest>().send(ReparentRequest {
                        child: child.into(),
                        parent: parent.map(Into::into),
                    });
                }
                Effect::Visible(entity, visible) => {
                    ctx.events_mut::<VisibilityRequest>()
                        .send(VisibilityRequest {
                            target: entity.into(),
                            visible,
                        });
                }
                Effect::Sound(cue) => {
                    ctx.events_mut::<PlayCue>().send(cue);
                }
                Effect::Scene { scene, transition } => {
                    ctx.events_mut::<SceneCommand>()
                        .send(SceneCommand { scene, transition });
                }
                Effect::Screen(screen) => {
                    ctx.events_mut::<ScreenCommand>()
                        .send(ScreenCommand::Show(screen));
                }
                Effect::Story(playback) => {
                    let command = match playback {
                        StoryPlayback::Start => StoryCommand::Start,
                        StoryPlayback::Continue => StoryCommand::Continue,
                    };
                    ctx.events_mut::<StoryCommand>().send(command);
                }
                Effect::Save => save_requested = true,
            }
        }
        save_requested
    }

    fn write_state(&self) {
        if self.transient_saves {
            return;
        }
        let state = save::BehaviorSave {
            vars: self
                .var_table
                .names()
                .iter()
                .zip(&self.vars)
                .map(|(name, value)| (name.clone(), value.to_literal()))
                .collect(),
            fired: self
                .programs
                .iter()
                .enumerate()
                .filter(|(i, p)| {
                    p.def.once
                        && !p.is_scoped()
                        && self.instances[*i].iter().any(|inst| inst.fired_once)
                })
                .map(|(_, p)| (p.def.asset_id.0, save::def_hash(&p.def)))
                .collect(),
        };
        if let Err(e) = save::write_save(&save::state_file(&self.save_dir), &state) {
            tracing::warn!("BehaviorSystem: state save failed: {e}");
        }
    }
}

// Adding to a local keeps its declared type.
fn add_vals(current: Val, delta: Val) -> Val {
    match (current, delta) {
        (Val::Int(a), b) => Val::Int(a.saturating_add(b.as_f32().unwrap_or(0.0) as i32)),
        (Val::Float(a), b) => Val::Float(a + b.as_f32().unwrap_or(0.0)),
        (Val::Vec3(a), Val::Vec3(b)) => Val::Vec3([a[0] + b[0], a[1] + b[1], a[2] + b[2]]),
        // Booleans and entities have no addition; assignment wins.
        (_, b) => b,
    }
}
