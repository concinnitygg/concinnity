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
use std::time::Instant;

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
            // At most one firing per tick; dropping whole elapsed intervals
            // keeps a long frame from queueing a burst.
            self.timer_accum %= interval.max(f32::EPSILON);
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
    start_time: Option<Instant>,
    prev_elapsed: f32,
    // Instances present before the first tick are the world's initial
    // population, so `spawned` does not fire for them.
    populated: bool,
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
            save_dir: concinnity_core::paths::saves_dir(),
            transient_saves: false,
            trace_frame: 0,
            trace_paths_published: false,
            start_time: None,
            prev_elapsed: 0.0,
            populated: false,
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
        let elapsed = self
            .start_time
            .get_or_insert_with(Instant::now)
            .elapsed()
            .as_secs_f32();
        let dt = (elapsed - self.prev_elapsed).max(0.0);
        self.prev_elapsed = elapsed;

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
        if !menu_active {
            self.tick(ctx, dt, elapsed);
        }
        StepResult::Continue
    }
}

// The entity sets a body iterates this tick, resolved before anything runs.
//
// Single-entity reads (a name, a position, a liveness test) are not here: no
// body runs while the world is mutable, so they read the world directly and
// cost one lookup each instead of a whole-world copy per tick.
struct Snapshot {
    // Per program, per declared query, in stable order.
    queries: Vec<Vec<Vec<Entity>>>,
    // Per program, the entities its scope matched, in stable order.
    scoped: Vec<Vec<Entity>>,
}

// What a body run reads that the tick, rather than the body, determines. The
// per-tick half of `run::View`, which adds the per-instance half.
struct TickView<'a, 'w> {
    ctx: &'a PipelineContext<'w>,
    snapshot: &'a Snapshot,
    // Resolved once per tick rather than per name lookup.
    names: Option<&'a crate::ecs::decompose::EntityByName>,
    dt: f32,
    elapsed: f32,
    tracing: bool,
}

impl TickView<'_, '_> {
    // A name index entry can outlive its entity, so each is confirmed.
    fn named(&self, id: AssetId) -> Option<Entity> {
        self.names
            .and_then(|n| n.get(id))
            .filter(|e| self.ctx.is_alive(*e))
    }
}

impl BehaviorSystem {
    // Entities carrying every one of these component tags, in stable order.
    // Column order shifts as entities are removed (swap-remove), so the result
    // is sorted: an unstable iteration order would make a body's effects depend
    // on unrelated despawns.
    fn entities_matching(ctx: &PipelineContext, tags: &[u8]) -> Vec<Entity> {
        let Some((first, rest)) = tags.split_first() else {
            return Vec::new();
        };
        let mut out: Vec<Entity> = ctx.entities_with_tag(*first).to_vec();
        for tag in rest {
            let mut also: Vec<Entity> = ctx.entities_with_tag(*tag).to_vec();
            also.sort_unstable_by_key(|e| e.to_bits());
            out.retain(|e| {
                also.binary_search_by_key(&e.to_bits(), |o| o.to_bits())
                    .is_ok()
            });
        }
        out.sort_unstable_by_key(|e| e.to_bits());
        out
    }

    fn gather(&self, ctx: &PipelineContext) -> Snapshot {
        let queries: Vec<Vec<Vec<Entity>>> = self
            .programs
            .iter()
            .map(|p| {
                p.queries
                    .iter()
                    .map(|tags| Self::entities_matching(ctx, tags))
                    .collect()
            })
            .collect();
        let scoped: Vec<Vec<Entity>> = self
            .programs
            .iter()
            .map(|p| Self::entities_matching(ctx, &p.scope))
            .collect();
        Snapshot { queries, scoped }
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
            for entity in matched {
                if self.instances[i]
                    .iter()
                    .any(|inst| inst.entity == Some(*entity))
                {
                    continue;
                }
                let mut instance =
                    Instance::new(Some(*entity), program.local_inits.clone(), self.populated);
                instance.last_value = baselines[i];
                self.instances[i].push(instance);
            }
            self.instances[i].sort_by_key(|inst| inst.entity.map(|e| e.to_bits()));
        }
    }

    fn tick(&mut self, ctx: &mut PipelineContext, dt: f32, elapsed: f32) {
        // Execution tracing is on only while an observer's request stands
        // (the editor's Behavior panel); its absence costs this one lookup.
        let request = ctx.resource::<crate::ecs::TraceRequest>().cloned();
        let tracing = request.is_some();
        let mut fired: Vec<(usize, Vec<u32>)> = Vec::new();

        let snapshot = self.gather(ctx);
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

        // Read phase: every body runs against an unchanged world, so the
        // borrow here is shared and the effects it produces are applied only
        // after it ends.
        let mut effects: Vec<(usize, Option<Entity>, Vec<Effect>)> = Vec::new();
        {
            let reads = TickView {
                ctx,
                snapshot: &snapshot,
                names: ctx.resource::<crate::ecs::decompose::EntityByName>(),
                dt,
                elapsed,
                tracing,
            };

            // Delayed runs from earlier ticks: count down and execute those now
            // due. Before the append below, so a fresh delay starts counting
            // next tick.
            let mut idx = 0;
            while idx < self.pending.len() {
                self.pending[idx].2 -= dt;
                if self.pending[idx].2 <= 0.0 {
                    let (i, entity, _) = self.pending.swap_remove(idx);
                    if let Some((produced, nodes)) = self.run_body(&reads, i, entity) {
                        effects.push((i, entity, produced));
                        if tracing {
                            fired.push((i, nodes));
                        }
                    }
                } else {
                    idx += 1;
                }
            }

            for (i, entity) in runs {
                let delay = self.programs[i].def.delay;
                if delay > 0.0 {
                    self.pending.push((i, entity, delay));
                } else if let Some((produced, nodes)) = self.run_body(&reads, i, entity) {
                    effects.push((i, entity, produced));
                    if tracing {
                        fired.push((i, nodes));
                    }
                }
            }
        }

        let mut save_requested = false;
        for (i, entity, produced) in effects {
            save_requested |= self.apply(ctx, i, entity, produced);
        }

        // One write per tick, after every effect has landed, so the file holds
        // this tick's final values.
        if save_requested {
            self.write_state();
        }

        if let Some(request) = request {
            self.publish_trace(ctx, &request, &fired);
        }
    }

    fn run_body(
        &self,
        reads: &TickView<'_, '_>,
        i: usize,
        entity: Option<Entity>,
    ) -> Option<(Vec<Effect>, Vec<u32>)> {
        let locals = self.instances[i]
            .iter()
            .find(|inst| inst.entity == entity)
            .map(|inst| inst.locals.clone())?;
        let mut bindings: Vec<Option<Val>> = vec![None; self.programs[i].bindings];
        let mut out = Vec::new();
        let mut nodes: Option<Vec<u32>> = reads.tracing.then(Vec::new);
        let mut view = View {
            dt: reads.dt,
            elapsed: reads.elapsed,
            vars: &self.vars,
            locals: &locals,
            bindings: &mut bindings,
            queries: &reads.snapshot.queries[i],
            by_name: &|id| reads.named(id),
            transforms: &|e| reads.ctx.get::<Transform>(e).copied(),
            alive: &|e| reads.ctx.is_alive(e),
            self_entity: entity,
            trace: &mut nodes,
        };
        run::exec(&self.programs[i].body, &mut view, &mut out);
        Some((out, nodes.unwrap_or_default()))
    }

    // Land one body's effects. Returns whether a `save` was requested.
    fn apply(
        &mut self,
        ctx: &mut PipelineContext,
        i: usize,
        entity: Option<Entity>,
        effects: Vec<Effect>,
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
