// src/logic/mod.rs
//
// ReactionSystem: declarative when/if/then logic. Each `Reaction` component
// names an event source (world start, a timer, a variable change, a volume
// crossing, an interact press), conditions over the shared `Variables` store,
// and a list of actions dispatched onto the runtime's existing request queues:
//   mod.rs     system + the per-tick drive
//   rules.rs   per-rule firing state and the fire decision
//   actions.rs action -> request-queue dispatch
//   vars.rs    the shared integer variable store
//   save.rs    persisted state (the `save` action; restored at init)
//
// Scheduled before SpawnSystem so spawn/despawn requests fired this tick are
// applied this same tick, and before SettingsSystem / StorySystem /
// AudioSystem so scene, story, and audio requests land the same tick too. A
// `set` this tick is seen by variable-source rules next tick, so rule chains
// advance one link per tick. Clocks (timers, delays, cooldowns) freeze while
// a menu is open (`MenuActive`, published by OverlaySystem earlier this tick),
// like the rest of the world clock.

use std::path::PathBuf;
use std::time::Instant;

use crate::assets::{InteractSignal, Reaction, VolumeEvent};
use crate::ecs::{EventCursor, PipelineContext, StepResult, System};

mod actions;
mod rules;
mod save;
mod vars;

#[cfg(test)]
mod tests;

pub use vars::Variables;

#[derive(Debug)]
pub struct ReactionSystem {
    rules: Vec<rules::Rule>,
    // Delayed action runs: (rule index, seconds left).
    pending: Vec<(usize, f32)>,
    // Cursors into the Events<VolumeEvent> / Events<InteractSignal> queues
    // (physics-published crossings, controller-published presses), drained
    // every step -- paused ticks included, so neither ages out of the event
    // store's retention behind a menu.
    crossing_cursor: EventCursor,
    press_cursor: EventCursor,
    // Events drained but not yet consumed by an unpaused tick.
    crossings: Vec<VolumeEvent>,
    presses: Vec<InteractSignal>,
    // Where the persisted logic state lives (the project data directory).
    save_dir: PathBuf,
    start_time: Option<Instant>,
    prev_elapsed: f32,
}

impl Default for ReactionSystem {
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            pending: Vec::new(),
            crossing_cursor: EventCursor::default(),
            press_cursor: EventCursor::default(),
            crossings: Vec::new(),
            presses: Vec::new(),
            save_dir: concinnity_core::paths::saves_dir(),
            start_time: None,
            prev_elapsed: 0.0,
        }
    }
}

impl ReactionSystem {
    pub fn new() -> Self {
        Self::default()
    }
}

impl System for ReactionSystem {
    fn init(&mut self, ctx: &mut PipelineContext) {
        self.rules = ctx
            .query::<Reaction>()
            .cloned()
            .map(rules::Rule::new)
            .collect();

        // Restore persisted state, but only in a world that saves: any other
        // world starts fresh and never touches the state file.
        let mut vars = Variables::default();
        let mut restored = false;
        if self.rules.iter().any(|r| r.def.saves_state())
            && let Some(state) = save::read_save(&save::state_file(&self.save_dir))
        {
            vars = Variables::from_map(state.vars);
            for (id, hash) in state.fired {
                if let Some(rule) = self
                    .rules
                    .iter_mut()
                    .find(|r| r.def.asset_id.0 == id && r.def_hash() == hash)
                {
                    rule.restore_fired();
                }
            }
            restored = true;
        }
        // Baseline variable sources against the (possibly restored) values,
        // so restoring a variable does not read as a change on tick one.
        for rule in &mut self.rules {
            rule.sync_variable_baseline(&vars);
        }
        let var_count = vars.as_map().len();
        ctx.insert_resource(vars);

        if restored {
            tracing::info!(
                "ReactionSystem: {} rule(s), restored {} variable(s)",
                self.rules.len(),
                var_count,
            );
        } else {
            tracing::info!("ReactionSystem: {} rule(s)", self.rules.len());
        }
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        let elapsed = self
            .start_time
            .get_or_insert_with(Instant::now)
            .elapsed()
            .as_secs_f32();
        let dt = (elapsed - self.prev_elapsed).max(0.0);
        self.prev_elapsed = elapsed;

        // Volume crossings and interact presses published later in last
        // tick's schedule (physics, the camera controller). Drained even
        // while paused; consumed by the next unpaused tick.
        if let Some(events) = ctx.events::<VolumeEvent>() {
            self.crossings
                .extend(events.read(&mut self.crossing_cursor).into_iter().copied());
        }
        if let Some(events) = ctx.events::<InteractSignal>() {
            self.presses
                .extend(events.read(&mut self.press_cursor).into_iter().copied());
        }

        // The menu state OverlaySystem published earlier this tick; a paused
        // world fires nothing and its clocks stand still.
        let menu_active = ctx
            .resource::<crate::ecs::MenuActive>()
            .map(|m| m.0)
            .unwrap_or(false);
        if !menu_active {
            self.tick(ctx, dt);
        }
        StepResult::Continue
    }
}

impl ReactionSystem {
    fn tick(&mut self, ctx: &mut PipelineContext, dt: f32) {
        // Fire decisions run against the variable values as of this tick's
        // start: a `set` fired here is seen by variable-source rules next
        // tick, so rule chains advance one link per tick.
        let mut fired: Vec<usize> = Vec::new();
        {
            let Some(vars) = ctx.resource::<Variables>() else {
                return;
            };
            for (i, rule) in self.rules.iter_mut().enumerate() {
                if rule.due(vars, dt, &self.crossings, &self.presses) {
                    fired.push(i);
                }
            }
        }
        self.crossings.clear();
        self.presses.clear();

        let mut save_requested = false;

        // Delayed runs from earlier ticks: count down and execute the ones
        // now due. Conditions were checked at fire time, not re-checked here.
        // Before the append below, so a fresh delay starts counting next tick.
        let mut idx = 0;
        while idx < self.pending.len() {
            self.pending[idx].1 -= dt;
            if self.pending[idx].1 <= 0.0 {
                let (rule, _) = self.pending.swap_remove(idx);
                save_requested |= actions::execute(ctx, &self.rules[rule].def.actions);
            } else {
                idx += 1;
            }
        }

        for &i in &fired {
            let delay = self.rules[i].def.delay;
            if delay > 0.0 {
                self.pending.push((i, delay));
            } else {
                save_requested |= actions::execute(ctx, &self.rules[i].def.actions);
            }
        }

        // One write per tick, after every action has landed, so the file
        // holds this tick's final variable values.
        if save_requested {
            self.write_state(ctx);
        }
    }

    fn write_state(&self, ctx: &PipelineContext) {
        let Some(vars) = ctx.resource::<Variables>() else {
            return;
        };
        let state = save::LogicSave {
            vars: vars.as_map().clone(),
            fired: self
                .rules
                .iter()
                .filter(|r| r.def.once && r.fired())
                .map(|r| (r.def.asset_id.0, r.def_hash()))
                .collect(),
        };
        if let Err(e) = save::write_save(&save::state_file(&self.save_dir), &state) {
            tracing::warn!("ReactionSystem: state save failed: {e}");
        }
    }
}
