//! The system that drives compiled behaviors: gathers what a body may read,
//! runs it on the VM, and applies the effects it produced.
//!
//! ```text
//! mod.rs       the system, its per-tick drive, and instance bookkeeping
//! instance.rs  one behavior's firing state and its clocks
//! eval.rs      the read phase: the tick's view, its buffers, its schedule
//! apply.rs     the write phase: effects landing on the world
//! state.rs     persisted variables and `once` flags, behind a host's store
//! trace.rs     the execution trace an observer requests
//! ```
//!
//! A behavior with an empty `scope` runs once per firing; a scoped one runs once
//! per matching entity, each with its own locals. Each tick is read, run, apply:
//! the body sees a snapshot and produces effects, and only then does anything
//! change, so no behavior observes another's writes mid-tick.
//!
//! Clocks advance with the simulation rather than the wall, and freeze while a
//! menu is open, like the rest of the world clock.

mod apply;
mod eval;
mod instance;
mod state;
mod trace;

#[cfg(test)]
mod bench;
#[cfg(test)]
mod test_world;
#[cfg(test)]
mod tests;

use alloc::boxed::Box;
use alloc::vec::Vec;

use eval::{EvalCtx, PARALLEL_EVAL_MIN_JOBS, Snapshot, eval_one};
use instance::Instance;

pub use eval::{EvalBucket, EvalScheduler};
pub use state::{BehaviorState, BehaviorStore, def_hash};

use crate::behavior::{Effect, Program, Val, VarTable, compile};
use crate::components::{Behavior, BehaviorSource, InteractEvent, Variables, VolumeEvent};
use crate::ecs::{
    Entity, EntityByName, EventCursor, FrameContext, MenuActive, PipelineContext, ScheduleMode,
    SimTiming, StepResult, System, TraceRequest, TransientSaves,
};

/// Runs a world's [`Behavior`] components: their firing rules, their bodies,
/// and the effects those produce.
///
/// [`new`](BehaviorSystem::new) evaluates every run on the calling thread and
/// persists nothing; a host lends it a thread pool through
/// [`with_scheduler`](BehaviorSystem::with_scheduler) and somewhere to keep
/// state through [`with_store`](BehaviorSystem::with_store).
#[derive(Debug, Default)]
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
    presses: Vec<InteractEvent>,
    // `None` when the host keeps no behavior state: behaviors run, saving does
    // not.
    store: Option<Box<dyn BehaviorStore>>,
    // `None` when the host has no thread pool to lend: every run evaluates on
    // the calling thread.
    scheduler: Option<Box<dyn EvalScheduler>>,
    // Sampled from the `TransientSaves` resource at init: while true, stored
    // state is neither read nor written, so a preview session starts fresh and
    // leaves the user's saves untouched.
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
    // Per-worker evaluation state, grown to the scheduler's width on the first
    // parallel tick and reused thereafter so steady-state allocations stay
    // flat.
    eval_buckets: Vec<EvalBucket>,
    // The tick's firing list and the serial path's binding scratch, kept for
    // their capacity across ticks.
    jobs: Vec<(usize, Option<Entity>)>,
    bindings: Vec<Option<Val>>,
    // The serial path's effect/record buffers, kept for their capacity like
    // the parallel path's per-worker buckets.
    serial_effects: Vec<Effect>,
    serial_produced: Vec<(usize, Option<Entity>, usize)>,
    // The tick's resolved entity sets and the tag-intersection scratch, kept
    // for their capacity across ticks like the buffers above.
    snapshot: Snapshot,
    tag_scratch: Vec<Entity>,
}

impl BehaviorSystem {
    /// A system that evaluates serially and persists nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Keep this world's variables and `once` flags in `store`.
    pub fn with_store(mut self, store: Box<dyn BehaviorStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Fan a tick's evaluation out through `scheduler` once enough instances
    /// fire at once to pay for it.
    pub fn with_scheduler(mut self, scheduler: Box<dyn EvalScheduler>) -> Self {
        self.scheduler = Some(scheduler);
        self
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
            .map(|def| compile(def, &mut var_table))
            .collect();
        self.instances = self.programs.iter().map(|_| Vec::new()).collect();
        self.vars = var_table.initial();
        self.var_table = var_table;

        self.transient_saves = ctx.resource::<TransientSaves>().is_some_and(|t| t.0);
        self.trace_frame = 0;
        self.trace_paths_published = false;
        self.sim_ticks = 0;

        let restored = self.restore_state();
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
        if let Some(events) = ctx.events::<InteractEvent>() {
            self.presses
                .extend(events.read(&mut self.press_cursor).copied());
        }

        let menu_active = ctx.resource::<MenuActive>().map(|m| m.0).unwrap_or(false);
        if menu_active {
            return StepResult::Continue;
        }

        // The frame's fixed-tick budget. Absent (a directly-stepped world with
        // no App), every step runs exactly one tick. Edge events (crossings,
        // presses) are consumed by the frame's first tick; catch-up ticks see
        // none, so an edge never fires twice.
        let timing = ctx.resource::<SimTiming>().copied().unwrap_or_default();
        for _ in 0..timing.ticks {
            self.sim_ticks += 1;
            let elapsed = (self.sim_ticks as f64 * timing.tick_dt as f64) as f32;
            self.tick(ctx, timing.tick_dt, elapsed);
        }
        StepResult::Continue
    }
}

impl BehaviorSystem {
    // Restore persisted state, but only in a world that saves: any other world
    // starts fresh and never reads the store. Returns how many variables the
    // restore applied.
    fn restore_state(&mut self) -> usize {
        if self.transient_saves || !self.programs.iter().any(|p| p.def.saves_state()) {
            return 0;
        }
        let Some(state) = self.store.as_ref().and_then(|store| store.read()) else {
            return 0;
        };

        let mut restored = 0usize;
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
                .position(|p| p.def.asset_id.0 == id && def_hash(&p.def) == hash)
            {
                // World-scoped `once` state restores onto the single instance;
                // a scoped behavior's per-entity flags are not persisted,
                // matching its locals.
                if !self.programs[i].is_scoped() {
                    let mut instance = Instance::new(None, Vec::new(), false);
                    instance.fired_once = true;
                    self.instances[i].clear();
                    self.instances[i].push(instance);
                }
            }
        }
        restored
    }

    // Create instances for newly matching entities and drop those whose entity
    // is gone, preserving the state of everything that persists.
    fn resync_instances(&mut self, snapshot: &Snapshot, frame: FrameContext) {
        // A variable source starts baselined at the variable's current value,
        // so a restored save does not read as a change on the instance's first
        // tick. Read before the loop, which borrows `self.instances` mutably.
        let baselines = frame.collect(self.programs.iter().map(|p| {
            match &p.def.on {
                BehaviorSource::Variable(name) => self
                    .var_table
                    .slot_of(name)
                    .and_then(|s| self.vars.get(s as usize))
                    .copied()
                    .unwrap_or(Val::Int(0)),
                _ => Val::Int(0),
            }
        }));
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
        let request = ctx.resource::<TraceRequest>().cloned();
        let tracing = request.is_some();
        let mut fired: Vec<(usize, Vec<u32>)> = Vec::new();

        // Copied out of the context so the frame temporaries below can hold
        // scratch while `ctx` stays usable. Main-thread only: the parallel
        // eval workers keep their per-worker persistent buffers (see
        // `eval_one`), and nothing arena-backed crosses into them.
        let frame = ctx.frame;

        let mut snapshot = core::mem::take(&mut self.snapshot);
        self.gather(ctx, &mut snapshot);
        self.resync_instances(&snapshot, frame);
        self.populated = true;

        // Fire decisions run against this tick's starting values, so a `set`
        // here is seen by variable-source behaviors next tick and chains
        // advance one link per tick. Every instance could fire, so the
        // reservation is exact.
        let bound: usize = self.instances.iter().map(Vec::len).sum();
        let mut runs = frame.vec::<(usize, Option<Entity>)>(bound);
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
        let mut jobs = core::mem::take(&mut self.jobs);
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
        for &(i, entity) in runs.iter() {
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
        // parallel schedule, contiguous job chunks evaluate through the host's
        // scheduler into per-worker buffers instead. Either way a body observes
        // only the tick's starting state, so the results are identical; only
        // the apply order below is observable, and it walks jobs in list order
        // in both modes.
        let parallel = self.scheduler.is_some()
            && jobs.len() >= PARALLEL_EVAL_MIN_JOBS
            && ScheduleMode::current(ctx.resources) == ScheduleMode::Parallel;
        let scheduler = self.scheduler.as_deref().filter(|_| parallel);
        let mut effects = core::mem::take(&mut self.serial_effects);
        let mut produced = core::mem::take(&mut self.serial_produced);
        effects.clear();
        produced.clear();
        let mut buckets = core::mem::take(&mut self.eval_buckets);
        let mut serial_bindings = core::mem::take(&mut self.bindings);
        {
            let ec = EvalCtx {
                components: ctx.components,
                names: ctx.resource::<EntityByName>(),
                snapshot: &snapshot,
                programs: &self.programs,
                instances: &self.instances,
                vars: &self.vars,
                dt,
                elapsed,
                tracing,
            };
            if let Some(scheduler) = scheduler {
                let workers = scheduler.workers().max(1);
                while buckets.len() < workers {
                    buckets.push(EvalBucket::default());
                }
                let chunk = jobs.len().div_ceil(buckets.len()).max(1);
                for (b, bucket) in buckets.iter_mut().enumerate() {
                    bucket.jobs = (b * chunk).min(jobs.len())..((b + 1) * chunk).min(jobs.len());
                }
                let jobs = &jobs;
                let ec = &ec;
                scheduler.run(&mut buckets, &|bucket| {
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
            let mut recorded = effects.drain(..);
            for &(i, entity, count) in &produced {
                save_requested |= self.apply(ctx, i, entity, recorded.by_ref().take(count));
            }
        }
        self.eval_buckets = buckets;
        self.jobs = jobs;
        self.bindings = serial_bindings;
        self.serial_effects = effects;
        self.serial_produced = produced;
        self.snapshot = snapshot;

        // One write per tick, after every effect has landed, so the store holds
        // this tick's final values.
        if save_requested {
            self.write_state();
        }

        if let Some(request) = request {
            self.publish_trace(ctx, &request, &fired);
        }
    }

    fn write_state(&self) {
        if self.transient_saves {
            return;
        }
        let Some(store) = self.store.as_ref() else {
            return;
        };
        store.write(&BehaviorState {
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
                .map(|(_, p)| (p.def.asset_id.0, def_hash(&p.def)))
                .collect(),
        });
    }
}
