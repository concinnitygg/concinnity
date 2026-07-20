// src/logic/mod.rs
//
// ReactionSystem: declarative when/if/then logic. Each `Reaction` component
// names an event source (world start, a timer, a variable change), conditions
// over the shared `Variables` store, and a list of actions dispatched onto the
// runtime's existing request queues:
//   mod.rs     system + the per-tick drive
//   rules.rs   per-rule firing state and the fire decision
//   actions.rs action -> request-queue dispatch
//   vars.rs    the shared integer variable store
//
// Scheduled before SpawnSystem so spawn/despawn requests fired this tick are
// applied this same tick, and before SettingsSystem / StorySystem /
// AudioSystem so scene, story, and audio requests land the same tick too. A
// `set` this tick is seen by variable-source rules next tick, so rule chains
// advance one link per tick. Clocks (timers, delays, cooldowns) freeze while
// a menu is open (`MenuActive`, published by OverlaySystem earlier this tick),
// like the rest of the world clock.

use std::time::Instant;

use crate::assets::Reaction;
use crate::ecs::{PipelineContext, StepResult, System};

mod actions;
mod rules;
mod vars;

#[cfg(test)]
mod tests;

pub use vars::Variables;

#[derive(Debug, Default)]
pub struct ReactionSystem {
    rules: Vec<rules::Rule>,
    // Delayed action runs: (rule index, seconds left).
    pending: Vec<(usize, f32)>,
    start_time: Option<Instant>,
    prev_elapsed: f32,
}

impl ReactionSystem {
    pub fn new() -> Self {
        Self::default()
    }
}

impl System for ReactionSystem {
    fn init(&mut self, ctx: &mut PipelineContext) {
        ctx.insert_resource(Variables::default());
        self.rules = ctx
            .query::<Reaction>()
            .cloned()
            .map(rules::Rule::new)
            .collect();
        tracing::info!("ReactionSystem: {} rule(s)", self.rules.len());
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        let elapsed = self
            .start_time
            .get_or_insert_with(Instant::now)
            .elapsed()
            .as_secs_f32();
        let dt = (elapsed - self.prev_elapsed).max(0.0);
        self.prev_elapsed = elapsed;

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
                if rule.due(vars, dt) {
                    fired.push(i);
                }
            }
        }

        // Delayed runs from earlier ticks: count down and execute the ones
        // now due. Conditions were checked at fire time, not re-checked here.
        // Before the append below, so a fresh delay starts counting next tick.
        let mut idx = 0;
        while idx < self.pending.len() {
            self.pending[idx].1 -= dt;
            if self.pending[idx].1 <= 0.0 {
                let (rule, _) = self.pending.swap_remove(idx);
                actions::execute(ctx, &self.rules[rule].def.actions);
            } else {
                idx += 1;
            }
        }

        for &i in &fired {
            let delay = self.rules[i].def.delay;
            if delay > 0.0 {
                self.pending.push((i, delay));
            } else {
                actions::execute(ctx, &self.rules[i].def.actions);
            }
        }
    }
}
