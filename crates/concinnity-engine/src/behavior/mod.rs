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

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Instant;

use crate::assets::{
    Behavior, BehaviorSource, DespawnRequest, InteractSignal, PlayCue, ReparentRequest,
    SceneCommand, ScreenCommand, SpawnRequest, StoryCommand, StoryPlayback, Transform,
    VisibilityRequest, VolumeEvent,
};
use crate::ecs::{Entity, EventCursor, PipelineContext, StepResult, System, asset_id::AssetId};

mod program;
mod run;
mod save;

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
    last_value: i32,
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
            last_value: 0,
        }
    }

    // Advance this instance's clocks by dt and decide whether it fires.
    fn due(
        &mut self,
        def: &Behavior,
        vars: &[i32],
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
                    .unwrap_or(0);
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
    vars: Vec<i32>,
    var_table: VarTable,
    // Delayed runs: (program, the instance's entity, seconds left).
    pending: Vec<(usize, Option<Entity>, f32)>,
    crossing_cursor: EventCursor,
    press_cursor: EventCursor,
    crossings: Vec<VolumeEvent>,
    presses: Vec<InteractSignal>,
    save_dir: PathBuf,
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
        let defs: Vec<Behavior> = ctx.query::<Behavior>().cloned().collect();
        let mut var_table = VarTable::default();
        self.programs = defs
            .into_iter()
            .map(|def| program::compile(def, &mut var_table))
            .collect();
        self.instances = self.programs.iter().map(|_| Vec::new()).collect();
        self.vars = vec![0; var_table.len()];
        self.var_table = var_table;

        // Restore persisted state, but only in a world that saves: any other
        // world starts fresh and never touches the state file.
        let mut restored = 0usize;
        if self.programs.iter().any(|p| p.def.saves_state())
            && let Some(state) = save::read_save(&save::state_file(&self.save_dir))
        {
            for (name, value) in &state.vars {
                if let Some(slot) = self.var_table.slot_of(name) {
                    self.vars[slot as usize] = *value;
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
                .extend(events.read(&mut self.crossing_cursor).into_iter().copied());
        }
        if let Some(events) = ctx.events::<InteractSignal>() {
            self.presses
                .extend(events.read(&mut self.press_cursor).into_iter().copied());
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

// What the bodies read this tick, gathered before anything runs.
struct Snapshot {
    by_name: BTreeMap<AssetId, Entity>,
    transforms: BTreeMap<Entity, Transform>,
    alive: BTreeSet<Entity>,
    // Per program, per declared query, in stable order.
    queries: Vec<Vec<Vec<Entity>>>,
    // Per program, the entities its scope matched, in stable order.
    scoped: Vec<Vec<Entity>>,
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
        let transforms: BTreeMap<Entity, Transform> = ctx
            .query_with_entity::<Transform>()
            .map(|(e, t)| (e, *t))
            .collect();
        // A name index entry can outlive its entity, so each is confirmed.
        let by_name: BTreeMap<AssetId, Entity> = ctx
            .resource::<crate::ecs::decompose::EntityByName>()
            .map(|n| {
                n.0.iter()
                    .filter(|(_, e)| ctx.is_alive(**e))
                    .map(|(k, v)| (*k, *v))
                    .collect()
            })
            .unwrap_or_default();

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

        // Everything a body can name this tick came from one of these sets, so
        // they define liveness for `alive`.
        let mut alive: BTreeSet<Entity> = transforms.keys().copied().collect();
        alive.extend(by_name.values().copied());
        for per_query in &queries {
            for entities in per_query {
                alive.extend(entities.iter().copied());
            }
        }
        for entities in &scoped {
            alive.extend(entities.iter().copied());
        }

        Snapshot {
            by_name,
            transforms,
            alive,
            queries,
            scoped,
        }
    }

    // Create instances for newly matching entities and drop those whose entity
    // is gone, preserving the state of everything that persists.
    fn resync_instances(&mut self, snapshot: &Snapshot) {
        // A variable source starts baselined at the variable's current value,
        // so a restored save does not read as a change on the instance's first
        // tick. Read before the loop, which borrows `self.instances` mutably.
        let baselines: Vec<i32> = self
            .programs
            .iter()
            .map(|p| match &p.def.on {
                BehaviorSource::Variable(name) => self
                    .var_table
                    .slot_of(name)
                    .and_then(|s| self.vars.get(s as usize))
                    .copied()
                    .unwrap_or(0),
                _ => 0,
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
            let def = self.programs[i].def.clone();
            for instance in &mut self.instances[i] {
                if instance.due(
                    &def,
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

        // Delayed runs from earlier ticks: count down and execute those now
        // due. Before the append below, so a fresh delay starts counting next
        // tick.
        let mut effects: Vec<(usize, Option<Entity>, Vec<Effect>)> = Vec::new();
        let mut idx = 0;
        while idx < self.pending.len() {
            self.pending[idx].2 -= dt;
            if self.pending[idx].2 <= 0.0 {
                let (i, entity, _) = self.pending.swap_remove(idx);
                if let Some(produced) = self.run_body(i, entity, &snapshot, dt, elapsed) {
                    effects.push((i, entity, produced));
                }
            } else {
                idx += 1;
            }
        }

        for (i, entity) in runs {
            let delay = self.programs[i].def.delay;
            if delay > 0.0 {
                self.pending.push((i, entity, delay));
            } else if let Some(produced) = self.run_body(i, entity, &snapshot, dt, elapsed) {
                effects.push((i, entity, produced));
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
    }

    fn run_body(
        &mut self,
        i: usize,
        entity: Option<Entity>,
        snapshot: &Snapshot,
        dt: f32,
        elapsed: f32,
    ) -> Option<Vec<Effect>> {
        let locals = self.instances[i]
            .iter()
            .find(|inst| inst.entity == entity)
            .map(|inst| inst.locals.clone())?;
        let mut bindings: Vec<Option<Val>> = vec![None; self.programs[i].bindings];
        let mut out = Vec::new();
        let mut view = View {
            dt,
            elapsed,
            vars: &self.vars,
            locals: &locals,
            bindings: &mut bindings,
            queries: &snapshot.queries[i],
            by_name: &snapshot.by_name,
            transforms: &|e| snapshot.transforms.get(&e).copied(),
            alive: &|e| snapshot.alive.contains(&e),
            self_entity: entity,
        };
        run::exec(&self.programs[i].body, &mut view, &mut out);
        Some(out)
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
                    if let Some(v) = self.vars.get_mut(slot as usize) {
                        *v = if add { v.saturating_add(value) } else { value };
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
        let state = save::BehaviorSave {
            vars: self
                .var_table
                .names()
                .iter()
                .zip(&self.vars)
                .map(|(name, value)| (name.clone(), *value))
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
